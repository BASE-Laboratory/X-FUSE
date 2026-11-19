"""Pixel- and voxel-level uncertainty quantification, calibration, physical
forward-model consistency diagnostics, and finite-sample Split Conformal
Prediction sets for correlative materials fusion.

Provides four complementary uncertainty & statistical guarantee mechanisms:
1. **Aleatoric / Chemical Ambiguity**: Calibrated class/element posterior entropy
   and top-1 vs. top-2 probability margin.
2. **Epistemic / Spatial & Volumetric Variance**: Online Welford variance across
   TTA shift/flip transforms and 2.5D orthogonal tri-plane views (zero extra
   ViT storage overhead), plus BALD mutual information.
3. **Physical Forward-Model Residual**: Discrepancy ``|D(P_HR) - G_LR|`` between
   PSF-downsampled high-resolution predictions and the measured coarse physical
   guide—flagging regions where the structural modality lacks contrast for a
   chemical gradient.
4. **Split Conformal Prediction (``ConformalCalibrator``)**: Distribution-free
   ``(1 - alpha)`` phase prediction sets ``C_{1-alpha}(x)`` and locally adaptive
   coarse physical error bounds calibrated on held-out spatial blocks or labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from .registration import DifferentiableGuideAligner
from .upsamplers import ForwardPSFDownsampler


class WelfordAccumulator:
    """Numerically stable online mean and variance accumulator in ``O(1)`` memory."""

    def __init__(self) -> None:
        self.count: int = 0
        self.mean: torch.Tensor | None = None
        self.m2: torch.Tensor | None = None

    def update(self, sample: torch.Tensor) -> None:
        x = sample.detach().float()
        self.count += 1
        if self.mean is None or self.m2 is None:
            self.mean = x.clone()
            self.m2 = torch.zeros_like(x)
            return
        delta = x - self.mean
        self.mean = self.mean + delta / self.count
        delta2 = x - self.mean
        self.m2 = self.m2 + delta * delta2

    def finalize(self, unbiased: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean, variance)`` tensors."""
        if self.mean is None or self.m2 is None or self.count == 0:
            raise RuntimeError("WelfordAccumulator received zero samples.")
        if self.count == 1:
            return self.mean, torch.zeros_like(self.mean)
        denom = (self.count - 1) if unbiased else self.count
        return self.mean, (self.m2 / denom).clamp(min=0.0)


@dataclass
class UncertaintyReport:
    """Container for pixel/voxel-level certainty maps and conformal prediction sets."""

    probabilities: torch.Tensor  # (B, K, H, W) calibrated probabilities
    hard_mask: torch.Tensor  # (B, H, W) integer class or binary mask (-1 if abstained)
    aleatoric_entropy: torch.Tensor  # (B, H, W) normalized entropy in [0, 1]
    confidence_margin: torch.Tensor  # (B, H, W) p_(1) - p_(2) in [0, 1]
    epistemic_variance: torch.Tensor  # (B, H, W) TTA / tri-plane variance
    physical_residual: torch.Tensor  # (B, H, W) upsampled |D(P_HR) - G_LR|
    combined_uncertainty: (
        torch.Tensor
    )  # (B, H, W) composite uncertainty score in [0, 1]
    unresolved_mask: (
        torch.Tensor
    )  # (B, H, W) bool flag where HR lacks physical contrast
    diagnostics: dict[str, Any] = field(default_factory=dict)
    prediction_sets: torch.Tensor | None = (
        None  # Optional (B, K, H, W) bool (1-alpha) conformal phase set
    )
    conformal_interval_halfwidth: torch.Tensor | None = (
        None  # Optional (B, M, H, W) calibrated coarse physical error halfwidth
    )

    def crop_spatial(self, height: int, width: int) -> UncertaintyReport:
        """Crop all spatial tensors back to unpadded ``(height, width)``."""
        return UncertaintyReport(
            probabilities=self.probabilities[:, :, :height, :width],
            hard_mask=self.hard_mask[:, :height, :width],
            aleatoric_entropy=self.aleatoric_entropy[:, :height, :width],
            confidence_margin=self.confidence_margin[:, :height, :width],
            epistemic_variance=self.epistemic_variance[:, :height, :width],
            physical_residual=self.physical_residual[:, :height, :width],
            combined_uncertainty=self.combined_uncertainty[:, :height, :width],
            unresolved_mask=self.unresolved_mask[:, :height, :width],
            diagnostics=dict(self.diagnostics),
            prediction_sets=(
                self.prediction_sets[:, :, :height, :width]
                if self.prediction_sets is not None
                else None
            ),
            conformal_interval_halfwidth=(
                self.conformal_interval_halfwidth[:, :, :height, :width]
                if self.conformal_interval_halfwidth is not None
                else None
            ),
        )


def _safe_quantile(tensor: torch.Tensor, q: float) -> torch.Tensor:
    """Compute quantile safely on arbitrarily large tensors, striding when
    ``numel() > 4_000_000`` to avoid PyTorch's ``2^24`` ``torch.quantile`` limit.
    """
    flat = torch.nan_to_num(
        tensor.detach().float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0
    )
    n = int(flat.numel())
    if n == 0:
        return torch.zeros((), device=tensor.device, dtype=torch.float32)
    if n > 4_000_000:
        step = math.ceil(n / 4_000_000)
        flat = flat[::step]
    return torch.quantile(flat, float(q))


ConformalMethod = Literal["lac", "aps", "raps"]


def _conformal_quantile(
    scores: torch.Tensor, alpha: float, n_effective: int | None = None
) -> float:
    """Exact finite-sample split-conformal quantile at rank
    ``ceil((n_eff + 1) * (1 - alpha)) / n_eff`` on nonconformity ``scores``.
    """
    flat = torch.nan_to_num(
        scores.detach().float().reshape(-1), nan=1.0, posinf=1.0, neginf=0.0
    )
    n = int(flat.numel())
    if n == 0:
        return 1.0
    if n > 4_000_000:
        step = math.ceil(n / 4_000_000)
        flat = flat[::step]
        n = int(flat.numel())
    a_clamped = min(max(float(alpha), 1e-4), 0.999)
    n_eff = min(n, max(1, int(n_effective))) if n_effective is not None else n
    q_level = min(1.0, math.ceil((n_eff + 1) * (1.0 - a_clamped)) / float(n_eff))
    rank = min(n, max(1, int(math.ceil(q_level * n))))
    sorted_vals = torch.sort(flat).values
    return float(sorted_vals[rank - 1].cpu().item())


def _fit_temperature_golden_section(
    probs_flat: torch.Tensor,
    targets_flat: torch.Tensor,
    t_min: float = 0.15,
    t_max: float = 8.0,
    n_iters: int = 36,
) -> float:
    """Fit post-hoc scalar temperature ``T > 0`` minimizing multiclass NLL
    ``-mean log softmax(log(p) / T)_y`` via exact 1D golden-section search.
    """
    if int(targets_flat.numel()) < 4:
        return 1.0
    log_p = torch.log(probs_flat.clamp(min=1e-7))

    def _nll(temp: float) -> float:
        scaled = log_p / max(temp, 1e-3)
        return float(F.cross_entropy(scaled, targets_flat).cpu().item())

    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = math.log(t_min), math.log(t_max)
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = _nll(math.exp(c)), _nll(math.exp(d))
    for _ in range(n_iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = _nll(math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = _nll(math.exp(d))
    best_t = math.exp(0.5 * (a + b))
    return best_t if _nll(best_t) + 1e-5 < _nll(1.0) else 1.0


class ConformalCalibrator:
    """Distribution-free Split Conformal Prediction calibrator supporting:
    1. **Supervised / Sparse-Label Calibration (``calibrate_labels``)**: Exact
       LAC, APS, and Regularized Adaptive Prediction Sets (``"raps"``) with
       optional spatial-block NLL temperature scaling (``fit_temperature=True``).
    2. **Zero-Label Coarse-Physics Calibration (``calibrate_coarse``)**: Calibrates
       locally adaptive coarse physical forward-PSF error intervals and pure-cell
       phase set thresholds on held-out spatial checkerboard blocks of ``G_LR``.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        method: ConformalMethod = "lac",
        allow_empty_sets: bool = False,
        k_reg: int = 1,
        lambda_reg: float = 0.05,
        randomized: bool = False,
        fit_temperature: bool = False,
        spatial_block_size: int = 1,
        seed: int = 42,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(
                f"Conformal miscoverage alpha must be in (0, 1), got {alpha}."
            )
        self.alpha = float(alpha)
        self.method: ConformalMethod = method
        self.allow_empty_sets = bool(allow_empty_sets)
        self.k_reg = max(1, int(k_reg))
        self.lambda_reg = max(0.0, float(lambda_reg))
        self.randomized = bool(randomized)
        self.fit_temperature = bool(fit_temperature)
        self.spatial_block_size = max(1, int(spatial_block_size))
        self.seed = int(seed)
        self.temperature: float = 1.0
        self.q_label: float | None = None
        self.q_coarse_per_channel: torch.Tensor | None = None
        self._coarse_scale: torch.Tensor | None = None
        self.n_calibration_points: int = 0

    def _apply_temperature(
        self, probs: torch.Tensor, multiclass: bool = True
    ) -> torch.Tensor:
        if abs(self.temperature - 1.0) <= 1e-4:
            return probs
        t = max(self.temperature, 1e-3)
        if multiclass:
            log_p = torch.log(probs.clamp(min=1e-7))
            return F.softmax(log_p / t, dim=1 if probs.ndim >= 3 else -1)
        logits = torch.logit(probs.clamp(1e-6, 1.0 - 1e-6))
        return torch.sigmoid(logits / t)

    def _nonconformity_scores(
        self, p_cal: torch.Tensor, y_cal: torch.Tensor
    ) -> torch.Tensor:
        if self.method in ("aps", "raps"):
            sorted_p, sorted_idx = torch.sort(p_cal, dim=-1, descending=True)
            cumsum_p = torch.cumsum(sorted_p, dim=-1)
            prev_cumsum = cumsum_p - sorted_p
            match = (sorted_idx == y_cal.unsqueeze(-1)).float()
            ranks = (
                torch.arange(1, p_cal.shape[-1] + 1, device=p_cal.device)
                .float()
                .unsqueeze(0)
            )
            reg = (
                self.lambda_reg * (ranks - float(self.k_reg)).clamp(min=0.0)
                if self.method == "raps"
                else torch.zeros_like(ranks)
            )
            if self.randomized:
                gen = torch.Generator(device="cpu").manual_seed(self.seed)
                u = torch.rand(
                    (p_cal.shape[0], 1), generator=gen, dtype=torch.float32
                ).to(p_cal.device)
            else:
                u = torch.ones((p_cal.shape[0], 1), device=p_cal.device)
            return ((prev_cumsum + u * sorted_p + reg) * match).sum(dim=-1)

        true_p = p_cal.gather(1, y_cal.unsqueeze(-1)).squeeze(-1)
        return 1.0 - true_p

    def _predict_multiclass_sets(
        self, p_cls: torch.Tensor, q_lbl: float
    ) -> torch.Tensor:
        if self.method in ("aps", "raps"):
            sorted_p, sorted_idx = torch.sort(p_cls, dim=1, descending=True)
            cumsum_p = torch.cumsum(sorted_p, dim=1)
            prev_cumsum = cumsum_p - sorted_p
            n_cls = p_cls.shape[1]
            ranks = (
                torch.arange(1, n_cls + 1, device=p_cls.device)
                .float()
                .view(1, n_cls, *([1] * (p_cls.ndim - 2)))
            )
            reg = (
                self.lambda_reg * (ranks - float(self.k_reg)).clamp(min=0.0)
                if self.method == "raps"
                else torch.zeros_like(ranks)
            )
            if self.randomized:
                gen = torch.Generator(device="cpu").manual_seed(self.seed + 997)
                u = torch.rand(
                    (p_cls.shape[0], 1, *p_cls.shape[2:]),
                    generator=gen,
                    dtype=torch.float32,
                ).to(p_cls.device)
                in_sorted = (prev_cumsum + u * sorted_p + reg) <= q_lbl
            else:
                # Romano et al. (2020) deterministic APS/RAPS: include up to the
                # first rank whose cumulative mass + penalty reaches >= q_lbl.
                in_sorted = (prev_cumsum + reg) < q_lbl
            if not self.allow_empty_sets:
                in_sorted[:, 0] = True
            pred_sets = torch.zeros_like(p_cls, dtype=torch.bool)
            pred_sets.scatter_(1, sorted_idx, in_sorted)
            return pred_sets

        pred_sets = (1.0 - p_cls) <= q_lbl
        if not self.allow_empty_sets:
            top1 = p_cls.argmax(dim=1, keepdim=True)
            pred_sets.scatter_(1, top1, True)
        return pred_sets

    @staticmethod
    def make_spatial_holdout_mask(
        lr_hw: tuple[int, int],
        holdout_fraction: float = 0.20,
        block_size: int = 2,
        seed: int = 42,
    ) -> torch.Tensor:
        """Create a deterministic ``(H_lr, W_lr)`` boolean calibration mask holding
        out spatial blocks of coarse pixels during probe fitting.
        """
        h_lr, w_lr = max(1, int(lr_hw[0])), max(1, int(lr_hw[1]))
        b_sz = max(1, min(int(block_size), max(1, min(h_lr, w_lr) // 2)))
        gh = max(1, math.ceil(h_lr / b_sz))
        gw = max(1, math.ceil(w_lr / b_sz))
        rng = np.random.default_rng(seed)
        n_blocks = gh * gw
        n_cal = max(1, min(n_blocks - 1, int(round(holdout_fraction * n_blocks))))
        perm = rng.permutation(n_blocks)
        cal_grid = np.zeros(n_blocks, dtype=bool)
        cal_grid[perm[:n_cal]] = True
        cal_2d = np.repeat(
            np.repeat(cal_grid.reshape(gh, gw), b_sz, axis=0), b_sz, axis=1
        )[:h_lr, :w_lr]
        return torch.from_numpy(cal_2d)

    @torch.no_grad()
    def calibrate_labels(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        cal_mask: torch.Tensor | None = None,
    ) -> float:
        """Calibrate phase prediction sets on ground-truth or sparse-scribble
        integer class labels ``targets`` (``(B, H, W)`` or ``(H, W)``).
        """
        self.q_label = None
        self.temperature = 1.0
        p = probs.detach().float()
        if p.ndim == 3:
            p = p.unsqueeze(0)
        if p.shape[1] == 1:
            p = torch.cat([1.0 - p, p], dim=1)
        p = p / p.sum(dim=1, keepdim=True).clamp(min=1e-8)

        y = targets.detach().long().to(p.device)
        if y.ndim == 2:
            y = y.unsqueeze(0)

        bsz, n_classes, height, width = p.shape
        p_flat = p.permute(0, 2, 3, 1).reshape(-1, n_classes)
        y_flat = y.reshape(-1)
        valid = (y_flat >= 0) & (y_flat < n_classes)
        if cal_mask is not None:
            valid = valid & cal_mask.to(p.device).reshape(-1).bool()

        if int(valid.sum().item()) == 0:
            self.n_calibration_points = 0
            self.q_label = 1.0 - self.alpha
            return self.q_label

        if self.fit_temperature and int(valid.sum().item()) >= 16:
            blk = max(self.spatial_block_size, 4)
            yy = torch.arange(height, device=p.device).view(1, height, 1) // blk
            xx = torch.arange(width, device=p.device).view(1, 1, width) // blk
            bb = torch.arange(bsz, device=p.device).view(bsz, 1, 1)
            checker = (((bb + yy + xx) % 2) == 0).reshape(-1)
            temp_sel = valid & checker
            quant_sel = valid & (~checker)
            if int(temp_sel.sum().item()) < 8 or int(quant_sel.sum().item()) < 8:
                idx_valid = torch.nonzero(valid, as_tuple=False).squeeze(-1)
                temp_sel = torch.zeros_like(valid)
                quant_sel = torch.zeros_like(valid)
                temp_sel[idx_valid[0::2]] = True
                quant_sel[idx_valid[1::2]] = True
            self.temperature = _fit_temperature_golden_section(
                p_flat[temp_sel], y_flat[temp_sel]
            )
            cal_sel = quant_sel
        else:
            cal_sel = valid

        if self.spatial_block_size > 1:
            b_step = self.spatial_block_size
            off = b_step // 2
            y_grid = (
                torch.arange(height, device=p.device).view(1, height, 1) % b_step
            ) == off
            x_grid = (
                torch.arange(width, device=p.device).view(1, 1, width) % b_step
            ) == off
            grid_mask = (y_grid & x_grid).expand(bsz, height, width).reshape(-1)
            strided_sel = cal_sel & grid_mask
            if int(strided_sel.sum().item()) >= 16:
                cal_sel = strided_sel

        p_cal = (
            self._apply_temperature(p_flat[cal_sel], multiclass=True)
            if self.fit_temperature
            else p_flat[cal_sel]
        )
        y_cal = y_flat[cal_sel]
        self.n_calibration_points = int(y_cal.numel())
        scores = self._nonconformity_scores(p_cal, y_cal)
        self.q_label = _conformal_quantile(scores, self.alpha)
        return self.q_label

    @torch.no_grad()
    def calibrate_coarse(
        self,
        probs_hr: torch.Tensor,
        guide_lr: torch.Tensor,
        epistemic_var_hr: torch.Tensor | None = None,
        cal_mask_lr: torch.Tensor | None = None,
        downsampler: ForwardPSFDownsampler | None = None,
        aligner: DifferentiableGuideAligner | None = None,
        mode: Literal["multiclass", "multilabel"] = "multiclass",
    ) -> dict[str, Any]:
        """Zero-label coarse physical split-conformal calibration using held-out
        coarse pixels ``cal_mask_lr`` (or all coarse pixels if ``None``).
        """
        self.q_label = None
        self.q_coarse_per_channel = None
        self._coarse_scale = None
        self.temperature = 1.0

        p = probs_hr.detach().float()
        if p.ndim == 3:
            p = p.unsqueeze(0)
        g = torch.nan_to_num(
            guide_lr.detach().float().to(p.device), nan=0.0, posinf=0.0, neginf=0.0
        )
        if g.ndim == 2:
            g = g.unsqueeze(0).unsqueeze(0)
        elif g.ndim == 3:
            g = g.unsqueeze(0)

        n_guide = g.shape[1]
        fg_probs = (
            p[:, 1:]
            if (mode == "multiclass" and p.shape[1] == n_guide + 1)
            else p[:, :n_guide]
        )
        ds = downsampler if downsampler is not None else ForwardPSFDownsampler()
        h_lr, w_lr = int(g.shape[-2]), int(g.shape[-1])
        src = (
            aligner(fg_probs, (h_lr, w_lr))
            if (aligner is not None and aligner.mode != "none")
            else fg_probs
        )
        pred_lr = ds(src, (h_lr, w_lr))

        # Normalize g into [0, 1] scale if raw counts were supplied
        g_eval = g
        if float(g.max().item()) > 1.5:
            g_max = (
                g.flatten(start_dim=2)
                .max(dim=-1)
                .values.unsqueeze(-1)
                .unsqueeze(-1)
                .clamp(min=1e-6)
            )
            self._coarse_scale = g_max.detach().clone()
            g_eval = (g / g_max).clamp(0.0, 1.0)
        else:
            g_eval = g.clamp(0.0, 1.0)

        entropy_hr, _ = compute_posterior_uncertainty(p, mode=mode)
        epi_hr = (
            torch.zeros_like(entropy_hr)
            if epistemic_var_hr is None
            else (
                epistemic_var_hr.mean(dim=1)
                if epistemic_var_hr.ndim == 4
                else epistemic_var_hr
            )
            .float()
            .to(p.device)
        )
        difficulty_hr = (0.5 * entropy_hr + 0.5 * epi_hr.sqrt()).clamp(min=0.08)
        difficulty_lr = F.adaptive_avg_pool2d(
            difficulty_hr.unsqueeze(1), output_size=(h_lr, w_lr)
        )  # (B, 1, H_lr, W_lr)

        norm_res = (pred_lr - g_eval).abs() / difficulty_lr
        mask_flat = (
            cal_mask_lr.to(p.device).reshape(-1).bool()
            if cal_mask_lr is not None
            else torch.ones(h_lr * w_lr, device=p.device, dtype=torch.bool)
        )
        self.n_calibration_points = int(mask_flat.sum().item())

        q_ch = []
        for m in range(n_guide):
            s_m = norm_res[:, m].reshape(norm_res.shape[0], -1)[:, mask_flat]
            q_ch.append(_conformal_quantile(s_m, self.alpha))
        self.q_coarse_per_channel = torch.tensor(
            q_ch, device=p.device, dtype=torch.float32
        )

        # Calibrate phase/element set threshold on high-purity coarse superpixels
        purity_fallback = False
        if mode == "multilabel" and p.shape[1] >= 2:
            p_pool = F.adaptive_avg_pool2d(p.clamp(0.0, 1.0), output_size=(h_lr, w_lr))
            p_m = (
                p_pool[:, :n_guide].permute(0, 2, 3, 1).reshape(-1, n_guide)[mask_flat]
            )
            g_m = (
                g_eval[:, :n_guide].permute(0, 2, 3, 1).reshape(-1, n_guide)[mask_flat]
            )
            hi = g_m >= 0.75
            lo = g_m <= 0.25
            pure = hi | lo
            if int(pure.sum().item()) >= 4:
                scores_ml = torch.where(hi, 1.0 - p_m, p_m)[pure]
                self.q_label = _conformal_quantile(scores_ml, self.alpha)
            else:
                purity_fallback = True
                self.q_label = 1.0 - self.alpha
        else:
            p_full = (
                torch.cat([1.0 - p, p], dim=1)
                if p.shape[1] == 1
                else p / p.sum(dim=1, keepdim=True).clamp(min=1e-8)
            )
            p_pool = F.adaptive_avg_pool2d(p_full, output_size=(h_lr, w_lr))
            if p_full.shape[1] == n_guide + 1:
                bg_g = (1.0 - g_eval.sum(dim=1, keepdim=True)).clamp(0.0, 1.0)
                g_simplex = torch.cat([bg_g, g_eval], dim=1)
            elif p_full.shape[1] == 2 and n_guide == 1:
                g_simplex = torch.cat([1.0 - g_eval, g_eval], dim=1)
            else:
                g_simplex = g_eval

            max_frac, pseudo_cls = g_simplex.max(dim=1)  # (B, H_lr, W_lr)
            pure_mask = (max_frac.reshape(-1) >= 0.75) & mask_flat
            if int(pure_mask.sum().item()) < 4:
                purity_fallback = True
                pure_mask = (max_frac.reshape(-1) >= 0.55) & mask_flat
            if int(pure_mask.sum().item()) >= 4:
                p_pure = p_pool.permute(0, 2, 3, 1).reshape(-1, p_full.shape[1])[
                    pure_mask
                ]
                y_pure = pseudo_cls.reshape(-1)[pure_mask]
                scores_cls = self._nonconformity_scores(p_pure, y_pure)
                self.q_label = _conformal_quantile(scores_cls, self.alpha)
            else:
                self.q_label = 1.0 - self.alpha

        return {
            "conformal_alpha": self.alpha,
            "conformal_method": self.method,
            "conformal_temperature": round(float(self.temperature), 4),
            "conformal_q_label": round(float(self.q_label), 4),
            "conformal_q_coarse": [round(float(v), 4) for v in q_ch],
            "conformal_n_cal": self.n_calibration_points,
            "conformal_purity_fallback": purity_fallback,
        }

    @torch.no_grad()
    def predict_sets_and_intervals(
        self,
        probs_hr: torch.Tensor,
        guide_lr: torch.Tensor | None = None,
        epistemic_var_hr: torch.Tensor | None = None,
        mode: Literal["multiclass", "multilabel"] = "multiclass",
        downsampler: ForwardPSFDownsampler | None = None,
        aligner: DifferentiableGuideAligner | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Construct ``(1 - alpha)`` boolean ``prediction_sets`` ``(B, K, H, W)``
        and optional ``conformal_interval_halfwidth`` ``(B, M, H, W)``.
        """
        p = probs_hr.detach().float()
        if p.ndim == 3:
            p = p.unsqueeze(0)
        q_lbl = self.q_label if self.q_label is not None else (1.0 - self.alpha)

        if mode == "multilabel" and p.shape[1] >= 2:
            p_cls = self._apply_temperature(p.clamp(0.0, 1.0), multiclass=False)
            pred_sets = (1.0 - p_cls) <= q_lbl
        else:
            p_cls = (
                torch.cat([1.0 - p, p], dim=1)
                if p.shape[1] == 1
                else p / p.sum(dim=1, keepdim=True).clamp(min=1e-8)
            )
            p_cls = self._apply_temperature(p_cls, multiclass=True)
            pred_sets = self._predict_multiclass_sets(p_cls, q_lbl)

        halfwidth: torch.Tensor | None = None
        if guide_lr is not None and self.q_coarse_per_channel is not None:
            entropy_hr, _ = compute_posterior_uncertainty(p, mode=mode)
            epi_hr = (
                torch.zeros_like(entropy_hr)
                if epistemic_var_hr is None
                else (
                    epistemic_var_hr.mean(dim=1)
                    if epistemic_var_hr.ndim == 4
                    else epistemic_var_hr
                )
                .float()
                .to(p.device)
            )
            difficulty_hr = (0.5 * entropy_hr + 0.5 * epi_hr.sqrt()).clamp(min=0.08)
            q_vec = self.q_coarse_per_channel.to(p.device).view(1, -1, 1, 1)
            raw_halfwidth = (q_vec * difficulty_hr.unsqueeze(1)).clamp(min=0.0)
            halfwidth = raw_halfwidth.clamp(0.0, 1.0)

            # Physical contrast safeguard: where coarse guide shows excess phase m
            # unexplained by D(P_HR) beyond raw_halfwidth, include phase m in the set!
            g = torch.nan_to_num(
                guide_lr.detach().float().to(p.device), nan=0.0, posinf=0.0, neginf=0.0
            )
            if g.ndim == 2:
                g = g.unsqueeze(0).unsqueeze(0)
            elif g.ndim == 3:
                g = g.unsqueeze(0)
            if (
                self._coarse_scale is not None
                and self._coarse_scale.shape[1] == g.shape[1]
            ):
                g_unit = (g / self._coarse_scale.to(g.device)).clamp(0.0, 1.0)
            elif float(g.max().item()) > 1.5:
                g_max = (
                    g.flatten(start_dim=2)
                    .max(dim=-1)
                    .values.unsqueeze(-1)
                    .unsqueeze(-1)
                    .clamp(min=1e-6)
                )
                g_unit = (g / g_max).clamp(0.0, 1.0)
            else:
                g_unit = g.clamp(0.0, 1.0)

            n_guide = g_unit.shape[1]
            fg_p = (
                p_cls[:, 1:]
                if (mode == "multiclass" and p_cls.shape[1] == n_guide + 1)
                else p_cls[:, :n_guide]
            )
            ds = downsampler if downsampler is not None else ForwardPSFDownsampler()
            h_lr, w_lr = int(g_unit.shape[-2]), int(g_unit.shape[-1])
            src = (
                aligner(fg_p, (h_lr, w_lr))
                if (aligner is not None and aligner.mode != "none")
                else fg_p
            )
            deficit_lr = (g_unit - ds(src, (h_lr, w_lr))).clamp(min=0.0)
            deficit_hr = F.interpolate(
                deficit_lr,
                size=p.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            missing_contrast = deficit_hr > raw_halfwidth
            offset = (
                1 if (mode == "multiclass" and p_cls.shape[1] == n_guide + 1) else 0
            )
            for m in range(min(n_guide, pred_sets.shape[1] - offset)):
                pred_sets[:, m + offset] = (
                    pred_sets[:, m + offset] | missing_contrast[:, m]
                )

        return pred_sets, halfwidth


def _detect_label_boundary_mask(y: np.ndarray) -> np.ndarray | None:
    """Detect ground-truth phase boundaries via ``3 x 3`` neighborhood disagreement
    when ``y`` is a 2D ``(H, W)`` or batched ``(B, H, W)`` / 3D ``(Z, Y, X)`` grid.
    """
    if y.ndim == 2 and min(y.shape) >= 3:
        pad = np.pad(y, 1, mode="edge")
        b = np.zeros_like(y, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                b |= (
                    pad[1 + dy : 1 + dy + y.shape[0], 1 + dx : 1 + dx + y.shape[1]] != y
                )
        return b
    if y.ndim == 3 and min(y.shape[-2], y.shape[-1]) >= 3:
        sub = [_detect_label_boundary_mask(y[i]) for i in range(y.shape[0])]
        return np.stack([m for m in sub if m is not None])
    return None


def empirical_conformal_coverage(
    prediction_sets: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
) -> dict[str, float]:
    """Evaluate empirical marginal coverage ``P(Y in C(X))``, average set
    cardinality ``E[|C(X)|]``, and grain-boundary vs. grain-interior set metrics.
    """
    s = (
        prediction_sets.detach().cpu().numpy().astype(bool)
        if isinstance(prediction_sets, torch.Tensor)
        else np.asarray(prediction_sets, dtype=bool)
    )
    y = (
        targets.detach().cpu().numpy().astype(np.int64)
        if isinstance(targets, torch.Tensor)
        else np.asarray(targets, dtype=np.int64)
    )
    is_spatial_grid = False
    if s.ndim == 2:
        n_classes = s.shape[0]
        s_flat = s.T.reshape(-1, n_classes)
    elif s.ndim >= 3 and s.shape[1:] == y.shape:
        n_classes = s.shape[0]
        s_flat = np.moveaxis(s, 0, -1).reshape(-1, n_classes)
        is_spatial_grid = True
    elif (
        s.ndim == 3
        and y.ndim == 2
        and s.shape[0] == y.shape[0]
        and s.shape[2] == y.shape[1]
    ):
        n_classes = s.shape[1]
        s_flat = np.moveaxis(s, 1, -1).reshape(-1, n_classes)
    else:
        if s.ndim == 3:
            s = s[None, ...]
        n_classes = s.shape[1]
        s_flat = np.moveaxis(s, 1, -1).reshape(-1, n_classes)
        is_spatial_grid = s.shape[2:] == (y.shape[-2:] if y.ndim >= 2 else ())

    y_flat = y.reshape(-1)
    valid = (y_flat >= 0) & (y_flat < n_classes)
    if not np.any(valid):
        return {
            "empirical_coverage": 1.0,
            "mean_set_size": 1.0,
            "singleton_fraction": 1.0,
            "boundary_mean_set_size": 1.0,
            "interior_mean_set_size": 1.0,
            "interior_singleton_fraction": 1.0,
        }
    s_v = s_flat[valid]
    y_v = y_flat[valid]
    covered = s_v[np.arange(len(y_v)), y_v]
    sizes = s_v.sum(axis=-1)

    res: dict[str, float] = {
        "empirical_coverage": float(np.mean(covered)),
        "mean_set_size": float(np.mean(sizes)),
        "singleton_fraction": float(np.mean(sizes == 1)),
    }
    b_mask = _detect_label_boundary_mask(y) if is_spatial_grid else None
    if b_mask is not None and b_mask.size == y_flat.size:
        b_v = b_mask.reshape(-1)[valid]
        i_v = ~b_v
        res["boundary_mean_set_size"] = (
            float(np.mean(sizes[b_v])) if np.any(b_v) else res["mean_set_size"]
        )
        res["interior_mean_set_size"] = (
            float(np.mean(sizes[i_v])) if np.any(i_v) else res["mean_set_size"]
        )
        res["interior_singleton_fraction"] = (
            float(np.mean(sizes[i_v] == 1))
            if np.any(i_v)
            else res["singleton_fraction"]
        )
    else:
        res["boundary_mean_set_size"] = res["mean_set_size"]
        res["interior_mean_set_size"] = res["mean_set_size"]
        res["interior_singleton_fraction"] = res["singleton_fraction"]
    return res


def compute_posterior_uncertainty(
    probs: torch.Tensor,
    mode: Literal["multiclass", "multilabel"] = "multiclass",
    eps: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute normalized predictive entropy in ``[0, 1]`` and confidence margin."""
    p = probs.float().clamp(eps, 1.0 - eps)
    if mode == "multiclass" and p.shape[1] >= 2:
        p = p / p.sum(dim=1, keepdim=True).clamp(min=eps)
        p_clamped = p.clamp(eps, 1.0)
        n_classes = p.shape[1]
        entropy = -(p_clamped * torch.log(p_clamped)).sum(dim=1) / float(
            np.log(n_classes)
        )
        top2 = torch.topk(p, k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        return entropy.clamp(0.0, 1.0), margin.clamp(0.0, 1.0)

    bin_ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)) / float(np.log(2.0))
    entropy = bin_ent.mean(dim=1)
    margin = (2.0 * (p - 0.5).abs()).mean(dim=1)
    return entropy.clamp(0.0, 1.0), margin.clamp(0.0, 1.0)


def compute_bald_mutual_information(
    prob_samples: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """Compute Bayesian Active Learning by Disagreement (BALD) epistemic mutual
    information ``H(E[p]) - E[H(p)]`` from ``(N_samples, B, K, H, W)`` samples.
    """
    p_s = prob_samples.float().clamp(eps, 1.0 - eps)
    mean_p = p_s.mean(dim=0)
    if p_s.shape[2] >= 2:
        norm = float(np.log(p_s.shape[2]))
        ent_mean = -(mean_p * torch.log(mean_p)).sum(dim=1) / norm
        mean_ent = (-(p_s * torch.log(p_s)).sum(dim=2) / norm).mean(dim=0)
    else:
        norm = float(np.log(2.0))
        ent_mean = (
            -(
                mean_p * torch.log(mean_p) + (1 - mean_p) * torch.log(1 - mean_p)
            ).squeeze(1)
            / norm
        )
        mean_ent = (
            -(p_s * torch.log(p_s) + (1 - p_s) * torch.log(1 - p_s)).squeeze(2) / norm
        ).mean(dim=0)
    return (ent_mean - mean_ent).clamp(min=0.0)


def compute_physical_residual(
    pred_probs_hr: torch.Tensor,
    guide_lr: torch.Tensor,
    psf_sigma: float | tuple[float, float] | None = None,
    downsampler: ForwardPSFDownsampler | None = None,
    aligner: DifferentiableGuideAligner | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate normalized forward physical consistency ``|D(W(P_HR)) - G_LR|``."""
    if guide_lr.ndim == 2:
        guide_lr = guide_lr.unsqueeze(0).unsqueeze(0)
    elif guide_lr.ndim == 3:
        guide_lr = guide_lr.unsqueeze(0)

    ds = downsampler if downsampler is not None else ForwardPSFDownsampler()
    h_lr, w_lr = int(guide_lr.shape[-2]), int(guide_lr.shape[-1])
    src = pred_probs_hr.float()
    if aligner is not None and aligner.mode != "none":
        src = aligner(src, (h_lr, w_lr))
    pred_lr = ds(src, (h_lr, w_lr), psf_sigma=psf_sigma)

    g_ref = guide_lr.float().to(pred_lr.device)
    if float(g_ref.max().item()) > 1.5:
        g_max = (
            g_ref.flatten(start_dim=2).max(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
        )
        g_ref = g_ref / g_max.clamp(min=1e-6)
    if g_ref.shape[1] != pred_lr.shape[1]:
        g_ref = g_ref[:, : pred_lr.shape[1]]

    residual_lr = (pred_lr - g_ref).abs()
    residual_mean_lr = residual_lr.mean(dim=1, keepdim=True)
    residual_hr = F.interpolate(
        residual_mean_lr,
        size=pred_probs_hr.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    return residual_hr.clamp(0.0, 1.0), residual_lr


def match_coarse_priors(
    logits_hr: torch.Tensor,
    guide_lr: torch.Tensor,
    mode: Literal["multiclass", "multilabel"] = "multilabel",
    max_iters: int = 40,
    lr: float = 0.25,
) -> torch.Tensor:
    """Shift per-channel logit biases so the spatial mean of HR probabilities
    matches the spatial mean of the normalized coarse guide ``mean(G_LR)``.

    Note: This performs marginal prior-mean matching rather than ground-truth
    temperature scaling; for distribution-free finite-sample coverage guarantees,
    use ``ConformalCalibrator``.
    """
    logits = logits_hr.detach().float().clone()
    if guide_lr.ndim == 2:
        guide_lr = guide_lr.unsqueeze(0).unsqueeze(0)
    elif guide_lr.ndim == 3:
        guide_lr = guide_lr.unsqueeze(0)
    g_ref = guide_lr.float().to(logits.device)
    if float(g_ref.max().item()) > 1.5:
        g_max = (
            g_ref.flatten(start_dim=2).max(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
        )
        g_ref = g_ref / g_max.clamp(min=1e-6)
    target_mean = g_ref.mean(dim=(-2, -1))

    with torch.enable_grad():
        bias = torch.zeros(
            (logits.shape[0], logits.shape[1], 1, 1),
            device=logits.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        opt = torch.optim.Adam([bias], lr=lr)

        for _ in range(max_iters):
            opt.zero_grad()
            shifted = logits + bias
            if mode == "multiclass" and logits.shape[1] > target_mean.shape[1]:
                probs = F.softmax(shifted, dim=1)[:, 1:]
            elif mode == "multiclass":
                probs = F.softmax(shifted, dim=1)
            else:
                probs = torch.sigmoid(shifted)
            pred_mean = probs[:, : target_mean.shape[1]].mean(dim=(-2, -1))
            loss = F.mse_loss(pred_mean, target_mean[:, : pred_mean.shape[1]])
            if loss.item() < 1e-7:
                break
            loss.backward()
            opt.step()

    with torch.no_grad():
        shifted = logits + bias.detach()
        if mode == "multiclass":
            return F.softmax(shifted, dim=1)
        return torch.sigmoid(shifted)


def compute_hard_phase_mask(
    probs: torch.Tensor,
    mode: Literal["multiclass", "multilabel"],
    channel_dim: int = 1,
) -> torch.Tensor:
    """Convert probabilities to integer phase IDs along ``channel_dim``.

    - ``mode="multiclass"`` (``K >= 2``): ``argmax`` in ``0 .. K-1``.
    - ``mode="multilabel"`` (``M == 1``): ``0`` (background) or ``1`` (``p >= 0.5``).
    - ``mode="multilabel"`` (``M >= 2``): ``0`` when all channels ``< 0.5``, else
      ``1 .. M`` for the dominant active channel ``argmax_m p_m + 1``.
    """
    n_ch = int(probs.shape[channel_dim])
    if mode == "multiclass" and n_ch >= 2:
        return probs.argmax(dim=channel_dim).long()
    if n_ch == 1:
        return (probs.select(channel_dim, 0) >= 0.5).long()
    max_p, dom_idx = probs.max(dim=channel_dim)
    return torch.where(max_p >= 0.5, dom_idx + 1, torch.zeros_like(dom_idx)).long()


def build_uncertainty_report(
    probs_hr: torch.Tensor,
    guide_lr: torch.Tensor,
    epistemic_var: torch.Tensor | None = None,
    mode: Literal["multiclass", "multilabel"] = "multiclass",
    abstain_threshold: float | None = None,
    unresolved_residual_cutoff: float = 0.30,
    downsampler: ForwardPSFDownsampler | None = None,
    aligner: DifferentiableGuideAligner | None = None,
    calibrator: ConformalCalibrator | None = None,
    diagnostics: dict[str, Any] | None = None,
    uncertainty_weights: tuple[float, float, float] = (0.45, 0.25, 0.30),
) -> UncertaintyReport:
    """Assemble a complete ``UncertaintyReport`` (including optional Split
    Conformal prediction sets & intervals when ``calibrator`` is provided).
    """
    entropy, margin = compute_posterior_uncertainty(probs_hr, mode=mode)

    n_guide = (
        guide_lr.shape[1]
        if guide_lr.ndim == 4
        else (guide_lr.shape[0] if guide_lr.ndim == 3 else 1)
    )
    if mode == "multiclass" and probs_hr.shape[1] == n_guide + 1:
        fg_probs = probs_hr[:, 1:]
    else:
        fg_probs = probs_hr[:, : min(n_guide, probs_hr.shape[1])]

    phys_res, _ = compute_physical_residual(
        fg_probs, guide_lr, downsampler=downsampler, aligner=aligner
    )

    if epistemic_var is None:
        epi = torch.zeros_like(entropy)
        epi_norm = epi
    else:
        epi = epistemic_var.mean(dim=1) if epistemic_var.ndim == 4 else epistemic_var
        epi = epi.float().to(entropy.device)
        p99 = _safe_quantile(epi, 0.99).clamp(min=1e-6)
        epi_norm = (epi / p99).clamp(0.0, 1.0)

    w_ent, w_epi, w_res = uncertainty_weights
    combined = (w_ent * entropy + w_epi * epi_norm + w_res * phys_res).clamp(0.0, 1.0)
    unresolved = phys_res > unresolved_residual_cutoff
    hard = compute_hard_phase_mask(probs_hr, mode=mode)

    if abstain_threshold is not None:
        hard = torch.where(
            combined > abstain_threshold, torch.full_like(hard, -1), hard
        )

    pred_sets: torch.Tensor | None = None
    conf_hw: torch.Tensor | None = None
    if calibrator is not None:
        pred_sets, conf_hw = calibrator.predict_sets_and_intervals(
            probs_hr=probs_hr,
            guide_lr=guide_lr,
            epistemic_var_hr=epi,
            mode=mode,
            downsampler=downsampler,
            aligner=aligner,
        )

    return UncertaintyReport(
        probabilities=probs_hr,
        hard_mask=hard,
        aleatoric_entropy=entropy,
        confidence_margin=margin,
        epistemic_variance=epi,
        physical_residual=phys_res,
        combined_uncertainty=combined,
        unresolved_mask=unresolved,
        diagnostics=dict(diagnostics) if diagnostics is not None else {},
        prediction_sets=pred_sets,
        conformal_interval_halfwidth=conf_hw,
    )


# ── Quantitative Calibration & Uncertainty Evaluation Metrics ─────────────────


def expected_calibration_error(
    probs: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE) over ``n_bins`` confidence bins."""
    p = (
        probs.detach().cpu().float().numpy()
        if isinstance(probs, torch.Tensor)
        else np.asarray(probs, dtype=np.float32)
    )
    y = (
        targets.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor)
        else np.asarray(targets)
    )

    if p.ndim == 4 and p.shape[1] >= 2:
        confidences = p.max(axis=1).ravel()
        predictions = p.argmax(axis=1).ravel()
        accuracies = (predictions == y.ravel()).astype(np.float32)
    else:
        p_flat = np.clip(p.ravel(), 0.0, 1.0)
        y_flat = (y.ravel() > 0.5).astype(np.float32)
        predictions = (p_flat >= 0.5).astype(np.float32)
        confidences = np.where(predictions == 1.0, p_flat, 1.0 - p_flat)
        accuracies = (predictions == y_flat).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = max(len(confidences), 1)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=False):
        in_bin = (confidences > lo) & (confidences <= hi)
        if lo == 0.0:
            in_bin = (confidences >= lo) & (confidences <= hi)
        count = int(in_bin.sum())
        if count > 0:
            avg_conf = float(confidences[in_bin].mean())
            avg_acc = float(accuracies[in_bin].mean())
            ece += (count / n_total) * abs(avg_acc - avg_conf)
    return float(ece)


def brier_score(
    probs: np.ndarray | torch.Tensor, targets: np.ndarray | torch.Tensor
) -> float:
    """Compute mean Brier score between predicted probabilities and ground truth."""
    p = (
        probs.detach().cpu().float().numpy()
        if isinstance(probs, torch.Tensor)
        else np.asarray(probs, dtype=np.float32)
    )
    y = (
        targets.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor)
        else np.asarray(targets)
    )
    if p.ndim == 4 and p.shape[1] >= 2:
        n_classes = p.shape[1]
        p_hwk = np.moveaxis(p, 1, -1).reshape(-1, n_classes)
        y_onehot = np.eye(n_classes, dtype=np.float32)[y.ravel().astype(int)]
        return float(np.mean(np.sum((p_hwk - y_onehot) ** 2, axis=-1)))
    return float(np.mean((p.ravel() - y.ravel().astype(np.float32)) ** 2))


def sparsification_curve(
    preds: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    uncertainty: np.ndarray | torch.Tensor,
    n_fractions: int = 11,
) -> dict[str, np.ndarray | float]:
    """Evaluate selective prediction (risk-coverage / sparsification curve)."""
    pred_arr = (
        preds.detach().cpu().numpy()
        if isinstance(preds, torch.Tensor)
        else np.asarray(preds)
    ).ravel()
    tgt_arr = (
        targets.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor)
        else np.asarray(targets)
    ).ravel()
    unc_arr = (
        uncertainty.detach().cpu().float().numpy()
        if isinstance(uncertainty, torch.Tensor)
        else np.asarray(uncertainty, dtype=np.float32)
    ).ravel()

    n_pix = len(unc_arr)
    order_model = np.argsort(unc_arr)
    errors = (pred_arr != tgt_arr).astype(np.float32)
    order_oracle = np.argsort(errors)

    fractions = np.linspace(0.0, 0.9, n_fractions)
    retained_acc = np.zeros_like(fractions)
    retained_iou = np.zeros_like(fractions)
    model_err = np.zeros_like(fractions)
    oracle_err = np.zeros_like(fractions)

    classes = [c for c in np.unique(tgt_arr) if c > 0] or [1]

    for idx, q in enumerate(fractions):
        keep = max(1, int(round((1.0 - q) * n_pix)))
        keep_m = order_model[:keep]
        keep_o = order_oracle[:keep]

        retained_acc[idx] = float(1.0 - errors[keep_m].mean())
        model_err[idx] = float(errors[keep_m].mean())
        oracle_err[idx] = float(errors[keep_o].mean())

        ious = []
        for cls in classes:
            p_c = pred_arr[keep_m] == cls
            t_c = tgt_arr[keep_m] == cls
            union = int((p_c | t_c).sum())
            inter = int((p_c & t_c).sum())
            ious.append(inter / union if union > 0 else 1.0)
        retained_iou[idx] = float(np.mean(ious))

    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    assert trapz_fn is not None
    ause = float(trapz_fn(np.maximum(model_err - oracle_err, 0.0), fractions))
    return {
        "abstain_fractions": fractions,
        "retained_accuracy": retained_acc,
        "retained_miou": retained_iou,
        "model_error": model_err,
        "oracle_error": oracle_err,
        "ause": ause,
    }
