"""Pixel- and voxel-level uncertainty quantification, calibration, and physical
forward-model consistency diagnostics for correlative materials fusion.

Provides three complementary uncertainty axes:
1. **Aleatoric / Chemical Ambiguity**: Calibrated class/element posterior entropy
   and top-1 vs. top-2 probability margin.
2. **Epistemic / Spatial & Volumetric Variance**: Online Welford variance across
   TTA shift/flip transforms and 2.5D orthogonal tri-plane views (zero extra
   ViT storage overhead), plus BALD mutual information.
3. **Physical Forward-Model Residual**: Discrepancy ``|D(P_HR) - G_LR|`` between
   PSF-downsampled high-resolution predictions and the measured coarse physical
   guide—flagging regions where the structural modality lacks contrast for a
   chemical gradient.
"""

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F

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

    def finalize(self, unbiased: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean, variance)`` tensors."""
        if self.mean is None or self.m2 is None or self.count == 0:
            raise RuntimeError("WelfordAccumulator received zero samples.")
        if self.count == 1:
            return self.mean, torch.zeros_like(self.mean)
        denom = (self.count - 1) if unbiased else self.count
        return self.mean, (self.m2 / denom).clamp(min=0.0)


@dataclass
class UncertaintyReport:
    """Container for pixel/voxel-level certainty maps alongside phase predictions."""

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


def compute_posterior_uncertainty(
    probs: torch.Tensor,
    mode: Literal["multiclass", "multilabel"] = "multiclass",
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute normalized predictive entropy in ``[0, 1]`` and confidence margin.

    Args:
        probs: ``(B, K, H, W)`` probabilities (summing to 1 along dim=1 if
            ``multiclass``, or independent ``[0, 1]`` Bernoulli channels if
            ``multilabel``).
        mode: ``"multiclass"`` (mutually exclusive phases, K >= 2) or
            ``"multilabel"`` (co-located elements / single-phase binary).

    Returns:
        ``(normalized_entropy, confidence_margin)`` each of shape ``(B, H, W)``.
    """
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

    # Binary / multilabel: mean binary entropy across channels
    bin_ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)) / float(np.log(2.0))
    entropy = bin_ent.mean(dim=1)
    margin = (2.0 * (p - 0.5).abs()).mean(dim=1)
    return entropy.clamp(0.0, 1.0), margin.clamp(0.0, 1.0)


def compute_bald_mutual_information(
    prob_samples: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """Compute Bayesian Active Learning by Disagreement (BALD) epistemic mutual
    information ``H(E[p]) - E[H(p)]`` from ``(N_samples, B, K, H, W)`` TTA/plane
    probability samples."""
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
    psf_sigma: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate forward physical consistency ``|D(P_HR) - G_LR|``.

    Args:
        pred_probs_hr: ``(B, M, H, W)`` predicted foreground phase/element maps.
        guide_lr: ``(B, M, H_lr, W_lr)`` measured coarse guide maps in ``[0, 1]``.

    Returns:
        ``(residual_hr, residual_lr)`` where ``residual_hr`` has shape ``(B, H, W)``
        and ``residual_lr`` has shape ``(B, M, H_lr, W_lr)``.
    """
    if guide_lr.ndim == 2:
        guide_lr = guide_lr.unsqueeze(0).unsqueeze(0)
    elif guide_lr.ndim == 3:
        guide_lr = guide_lr.unsqueeze(0)

    downsampler = ForwardPSFDownsampler()
    h_lr, w_lr = guide_lr.shape[-2], guide_lr.shape[-1]
    pred_lr = downsampler(pred_probs_hr.float(), (h_lr, w_lr), psf_sigma=psf_sigma)
    residual_lr = (pred_lr - guide_lr.float().to(pred_lr.device)).abs()

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
    """Calibrate per-channel logit biases (prior shift) and temperature so the
    spatial mean of high-resolution probabilities matches the measured coarse
    physical phase fractions ``mean(G_LR)`` without needing HR ground-truth labels!"""
    logits = logits_hr.detach().float().clone()
    if guide_lr.ndim == 2:
        guide_lr = guide_lr.unsqueeze(0).unsqueeze(0)
    elif guide_lr.ndim == 3:
        guide_lr = guide_lr.unsqueeze(0)
    target_mean = guide_lr.float().to(logits.device).mean(dim=(-2, -1))  # (B, M)

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
                # First channel is background, remaining M channels match guide_lr
                probs = F.softmax(shifted, dim=1)[:, 1:]
            elif mode == "multiclass":
                probs = F.softmax(shifted, dim=1)
            else:
                probs = torch.sigmoid(shifted)
            pred_mean = probs.mean(dim=(-2, -1))
            loss = F.mse_loss(pred_mean, target_mean)
            if loss.item() < 1e-7:
                break
            loss.backward()
            opt.step()

    with torch.no_grad():
        shifted = logits + bias.detach()
        if mode == "multiclass":
            return F.softmax(shifted, dim=1)
        return torch.sigmoid(shifted)


def build_uncertainty_report(
    probs_hr: torch.Tensor,
    guide_lr: torch.Tensor,
    epistemic_var: torch.Tensor | None = None,
    mode: Literal["multiclass", "multilabel"] = "multiclass",
    abstain_threshold: float | None = None,
    unresolved_residual_cutoff: float = 0.30,
) -> UncertaintyReport:
    """Assemble a complete ``UncertaintyReport`` from HR probabilities, coarse
    guide measurements, and optional TTA / 2.5D epistemic variance."""
    entropy, margin = compute_posterior_uncertainty(probs_hr, mode=mode)

    # Align foreground channels with guide_lr for physical residual check
    n_guide = (
        guide_lr.shape[1]
        if guide_lr.ndim == 4
        else (guide_lr.shape[0] if guide_lr.ndim == 3 else 1)
    )
    if mode == "multiclass" and probs_hr.shape[1] == n_guide + 1:
        fg_probs = probs_hr[:, 1:]
    else:
        fg_probs = probs_hr[:, :n_guide]

    phys_res, _ = compute_physical_residual(fg_probs, guide_lr)

    if epistemic_var is None:
        epi = torch.zeros_like(entropy)
        epi_norm = epi
    else:
        epi = epistemic_var.mean(dim=1) if epistemic_var.ndim == 4 else epistemic_var
        epi = epi.float().to(entropy.device)
        p99 = torch.quantile(epi.flatten(), 0.99).clamp(min=1e-6)
        epi_norm = (epi / p99).clamp(0.0, 1.0)

    # Composite uncertainty blends chemical entropy, spatial epistemic variance,
    # and physical forward-model mismatch
    combined = (0.45 * entropy + 0.25 * epi_norm + 0.30 * phys_res).clamp(0.0, 1.0)
    unresolved = phys_res > unresolved_residual_cutoff

    if mode == "multiclass" and probs_hr.shape[1] >= 2:
        hard = probs_hr.argmax(dim=1).long()
    else:
        hard = (probs_hr[:, 0] >= 0.5).long()

    if abstain_threshold is not None:
        hard = torch.where(
            combined > abstain_threshold, torch.full_like(hard, -1), hard
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
    )


# ── Quantitative Calibration & Uncertainty Evaluation Metrics ─────────────────


def expected_calibration_error(
    probs: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE) over ``n_bins`` confidence bins.

    Supports both binary ``(H, W)`` / ``(B, 1, H, W)`` probabilities and
    multiclass ``(B, K, H, W)`` probabilities against integer ``targets``.
    """
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
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
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
) -> Dict[str, np.ndarray | float]:
    """Evaluate selective prediction (risk-coverage / sparsification curve).

    Progressively abstains on the most uncertain fraction ``q`` of pixels
    (``q in [0, 0.9]``) and measures segmentation accuracy & IoU on retained
    pixels, plus Area Under the Sparsification Error curve (AUSE vs. oracle).
    """
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
    order_model = np.argsort(unc_arr)  # ascending uncertainty (most certain first)
    errors = (pred_arr != tgt_arr).astype(np.float32)
    order_oracle = np.argsort(errors)  # true errors removed first

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

    ause = float(np.trapezoid(np.maximum(model_err - oracle_err, 0.0), fractions))
    return {
        "abstain_fractions": fractions,
        "retained_accuracy": retained_acc,
        "retained_miou": retained_iou,
        "model_error": model_err,
        "oracle_error": oracle_err,
        "ause": ause,
    }
