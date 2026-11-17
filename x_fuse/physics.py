"""Quantitative physical forward operators, likelihood losses, and raw 3D
spectral/diffraction cube unmixing for correlative materials imaging.

Supports:
1. ``SpectralUnmixingDictionary``: Non-negative endmember spectral/diffraction
   dictionary ``S in R_+^(M x E)`` initialized via deterministic multiplicative
   NMF and refined end-to-end against raw ``(E, H_lr, W_lr)`` data cubes.
2. ``PhysicsForwardModel``: Unified physical forward operator
   ``Lambda_LR = tau * D_Sigma(W_phi(P_HR)) @ S + b`` with:
   - Exact Poisson shot-noise deviance negative log-likelihood
     (``"forward_psf_poisson"``) for low-count EDS, EELS, and fast XRD-CT.
   - Simplex-constrained compositional phase-fraction conservation
     (``"compositional_kl"``, guaranteeing ``sum_k P_k(x) = 1`` at every pixel).
   - Partial-volume ``"forward_psf_bce"``, ``"forward_psf_mse"``, ``"bce"``,
     and ``"pearson"`` objectives.
"""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registration import DifferentiableGuideAligner
from .upsamplers import ForwardPSFDownsampler

LossOptions: TypeAlias = Literal[
    "pearson",
    "bce",
    "forward_psf_bce",
    "forward_psf_mse",
    "forward_psf_poisson",
    "compositional_kl",
]

GuideNormalization: TypeAlias = Literal[
    "minmax",
    "preserve_counts",
    "simplex",
    "global_max",
]


class SpectralUnmixingDictionary(nn.Module):
    """Differentiable non-negative spectral/diffraction basis ``S`` of shape
    ``(n_components, spectral_channels)`` with unit-normalized rows
    (``sum_e S_{m,e} = 1``) so abundance maps preserve physical count scale.
    """

    def __init__(self, n_components: int, spectral_channels: int) -> None:
        super().__init__()
        if n_components < 1 or spectral_channels < 2:
            raise ValueError(
                "SpectralUnmixingDictionary requires n_components >= 1 and "
                f"spectral_channels >= 2, got ({n_components}, {spectral_channels})."
            )
        self.n_components = int(n_components)
        self.spectral_channels = int(spectral_channels)
        init = torch.ones(
            (self.n_components, self.spectral_channels), dtype=torch.float32
        )
        self.raw_spectra = nn.Parameter(init)

    def spectra(self) -> torch.Tensor:
        """Return row-normalized non-negative basis ``(M, E)`` summing to 1."""
        pos = F.softplus(self.raw_spectra) + 1e-6
        return pos / pos.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    @torch.no_grad()
    def initialize_from_cube(
        self,
        cube_lr: torch.Tensor,
        n_iters: int = 40,
        seed: int = 0,
    ) -> torch.Tensor:
        """Decompose ``cube_lr`` ``(E, H_lr, W_lr)`` or ``(B, E, H_lr, W_lr)`` via
        multiplicative-update NMF, populate ``self.raw_spectra``, and return initial
        coarse abundance maps ``(B, M, H_lr, W_lr)``.
        """
        x = cube_lr.detach().float()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4 or x.shape[1] != self.spectral_channels:
            raise ValueError(
                f"Expected cube shape ([B,] {self.spectral_channels}, H_lr, W_lr), "
                f"got {tuple(cube_lr.shape)}."
            )
        bsz, spec_e, h_lr, w_lr = x.shape
        # Flatten pixels across batch: Y of shape (N_pix, E) >= 0
        y_mat = x.permute(0, 2, 3, 1).reshape(-1, spec_e).clamp(min=0.0)
        n_pix = y_mat.shape[0]
        m_comp = self.n_components

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))

        # Seed endmembers via farthest-point (SPA-like) selection on spectral cosines
        energies = y_mat.sum(dim=-1)
        chosen: list[int] = [int(energies.argmax().item())]
        y_norm = y_mat / y_mat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        for _ in range(1, m_comp):
            basis_so_far = y_norm[chosen]  # (k, E)
            max_sim = (y_norm @ basis_so_far.T).amax(dim=-1)
            # Prefer high-energy pixels that are spectrally distinct
            score = (1.0 - max_sim) * (energies / energies.max().clamp(min=1e-8) + 0.05)
            for idx_prev in chosen:
                score[idx_prev] = -1.0
            chosen.append(int(score.argmax().item()))

        h_basis = y_mat[chosen].clone() + 1e-3  # (M, E)
        if n_pix < m_comp:
            noise = torch.rand((m_comp, spec_e), generator=gen, dtype=torch.float32).to(
                x.device
            )
            h_basis = h_basis + 0.05 * noise
        h_basis = h_basis / h_basis.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        w_abund = torch.full(
            (n_pix, m_comp), 1.0 / m_comp, device=x.device, dtype=torch.float32
        ) * energies.unsqueeze(-1).clamp(min=1e-4)

        eps = 1e-7
        for _ in range(max(1, int(n_iters))):
            # Update abundances W: (N, M)
            num_w = y_mat @ h_basis.T
            den_w = (w_abund @ (h_basis @ h_basis.T)).clamp(min=eps)
            w_abund = (w_abund * (num_w / den_w)).clamp(min=eps)

            # Update endmember spectra H: (M, E) and normalize rows
            num_h = w_abund.T @ y_mat
            den_h = ((w_abund.T @ w_abund) @ h_basis).clamp(min=eps)
            h_basis = (h_basis * (num_h / den_h)).clamp(min=eps)
            row_sums = h_basis.sum(dim=-1, keepdim=True).clamp(min=eps)
            h_basis = h_basis / row_sums
            w_abund = w_abund * row_sums.T

        # Invert softplus: raw = log(exp(target) - 1) for scaled positive h_basis
        scaled = (h_basis * float(spec_e)).clamp(min=1e-4, max=50.0)
        inv_sp = torch.log(torch.expm1(scaled).clamp(min=1e-6))
        self.raw_spectra.copy_(inv_sp.to(self.raw_spectra.device))

        abund_4d = w_abund.reshape(bsz, h_lr, w_lr, m_comp).permute(0, 3, 1, 2)
        return abund_4d.contiguous()

    def forward(self, abundances_lr: torch.Tensor) -> torch.Tensor:
        """Mix coarse abundances ``(B, M, H_lr, W_lr)`` with basis ``(M, E)``
        into reconstructed spectral cube ``(B, E, H_lr, W_lr)``.
        """
        spec = self.spectra().to(device=abundances_lr.device, dtype=abundances_lr.dtype)
        return torch.einsum("bmhw,me->behw", abundances_lr, spec)


def poisson_deviance_loss(
    pred_rate: torch.Tensor, target_counts: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Exact non-negative Poisson Bregman deviance (shifted Poisson NLL):
    ``mean(pred - y * log(pred) - (y - y * log(y))) >= 0``, achieving ``0`` iff
    ``pred == y`` and maintaining well-conditioned ``O(1)`` values across count regimes.
    """
    lam = pred_rate.float().clamp(min=eps)
    y = target_counts.float().clamp(min=0.0)
    nll = lam - y * torch.log(lam)
    # Subtract target entropy offset where y > 0
    y_safe = y.clamp(min=eps)
    offset = torch.where(y > 0.0, y - y * torch.log(y_safe), torch.zeros_like(y))
    return (nll - offset).mean().clamp(min=0.0)


class PhysicsForwardModel(nn.Module):
    """Unified physical forward model mapping HR logits ``(B, M, H, W)`` to
    coarse measurements ``G_LR`` ``(B, M, H_lr, W_lr)`` or 3D spectral cubes
    ``Y_LR`` ``(B, E, H_lr, W_lr)`` via anisotropic PSF integration, spatial drift
    alignment, flux/background scaling, and optional spectral mixing.
    """

    def __init__(
        self,
        n_channels: int = 1,
        loss_fn: LossOptions = "bce",
        learn_psf: bool = False,
        psf_sigma_ratio: float = 0.35,
        spectral_channels: int | None = None,
        simplex_include_background: bool | None = None,
    ) -> None:
        super().__init__()
        self.n_channels = max(1, int(n_channels))
        self.loss_fn: LossOptions = loss_fn
        self.simplex_include_background = simplex_include_background

        self.psf_downsampler = ForwardPSFDownsampler(
            psf_sigma_ratio=psf_sigma_ratio,
            learnable=learn_psf,
            n_channels=self.n_channels,
        )
        self.spectral_dict: SpectralUnmixingDictionary | None = (
            SpectralUnmixingDictionary(self.n_channels, spectral_channels)
            if spectral_channels is not None and spectral_channels >= 2
            else None
        )

        obs_ch = (
            int(spectral_channels)
            if (spectral_channels is not None and spectral_channels >= 2)
            else self.n_channels
        )
        if loss_fn == "forward_psf_poisson":
            self.log_flux_scale = nn.Parameter(
                torch.zeros((1, self.n_channels, 1, 1), dtype=torch.float32)
            )
            self.raw_background = nn.Parameter(
                torch.full((1, obs_ch, 1, 1), -3.0, dtype=torch.float32)
            )
            self._scale_initialized = False
        else:
            self.register_parameter("log_flux_scale", None)
            self.register_parameter("raw_background", None)
            self._scale_initialized = True

    @torch.no_grad()
    def maybe_initialize_poisson_scale(self, target: torch.Tensor) -> None:
        """Initialize ``log_flux_scale`` and ``raw_background`` from target count
        quantiles on the first training call so optimization starts at physical scale.
        """
        if self._scale_initialized or self.log_flux_scale is None:
            return
        t = target.detach().float()
        if t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 3:
            t = t.unsqueeze(0)
        flat = t.flatten(start_dim=2)  # (B, C_obs, N)
        q95 = torch.quantile(flat, 0.95, dim=-1).mean(dim=0)  # (C_obs,)
        q05 = torch.quantile(flat, 0.05, dim=-1).mean(dim=0)  # (C_obs,)

        if self.spectral_dict is not None:
            # Total integrated counts per pixel ~ sum_e Y(e)
            tot_scale = (
                flat.sum(dim=1).quantile(0.90) / max(self.n_channels, 1)
            ).clamp(min=0.5)
            self.log_flux_scale.fill_(float(torch.log(tot_scale).item()))
        else:
            flux = (q95 - q05).clamp(min=0.25)
            if flux.numel() == self.n_channels:
                self.log_flux_scale.copy_(
                    torch.log(flux)
                    .view(1, self.n_channels, 1, 1)
                    .to(self.log_flux_scale.device)
                )
        if (
            self.raw_background is not None
            and q05.numel() == self.raw_background.shape[1]
        ):
            bg = q05.clamp(min=1e-3, max=1e4)
            inv_sp = torch.log(torch.expm1(bg).clamp(min=1e-6))
            self.raw_background.copy_(
                inv_sp.view(1, -1, 1, 1).to(self.raw_background.device)
            )
        self._scale_initialized = True

    def _use_background_simplex(self, target: torch.Tensor | None = None) -> bool:
        if self.simplex_include_background is not None:
            return self.simplex_include_background
        if self.n_channels == 1:
            return True
        if target is not None and self.spectral_dict is None:
            t = target.detach().float()
            if t.ndim == 3:
                t = t.unsqueeze(0)
            if t.ndim == 4 and t.shape[1] == self.n_channels:
                col_sum = t.sum(dim=1)
                if float((col_sum - 1.0).abs().max().item()) < 0.03:
                    return False
        return True

    def logits_to_simplex_probs(
        self, logits: torch.Tensor, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Convert HR logits ``(B, M, H, W)`` to exact simplex probabilities
        summing to ``1.0`` along channel dimension ``1`` (either ``(B, M+1, H, W)``
        with channel 0 = matrix/background, or ``(B, M, H, W)`` when complete).
        """
        if self._use_background_simplex(target):
            bg = torch.zeros(
                (logits.shape[0], 1, logits.shape[2], logits.shape[3]),
                device=logits.device,
                dtype=logits.dtype,
            )
            return F.softmax(torch.cat([bg, logits], dim=1), dim=1)
        return F.softmax(logits, dim=1)

    def forward_coarse_prediction(
        self,
        logits: torch.Tensor,
        lr_hw: tuple[int, int],
        aligner: DifferentiableGuideAligner | None = None,
        target_hint: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map HR ``logits`` ``(B, M, H, W)`` through spatial alignment, physical
        PSF integration, and optional spectral dictionary mixing.
        """
        warped_logits = (
            aligner(logits, lr_hw)
            if (aligner is not None and aligner.mode != "none")
            else logits
        )
        if self.loss_fn == "compositional_kl":
            simplex_hr = self.logits_to_simplex_probs(
                warped_logits.float(), target=target_hint
            )
            fg_hr = (
                simplex_hr[:, 1:]
                if simplex_hr.shape[1] == self.n_channels + 1
                else simplex_hr
            )
            abund_lr = self.psf_downsampler(fg_hr, lr_hw)
            if self.spectral_dict is not None:
                return self.spectral_dict(abund_lr)
            return abund_lr

        probs_hr = torch.sigmoid(warped_logits.float())
        if self.loss_fn == "forward_psf_poisson" and self.log_flux_scale is not None:
            flux = self.log_flux_scale.exp().clamp(1e-3, 1e6)
            scaled_hr = probs_hr * flux
            abund_lr = self.psf_downsampler(scaled_hr, lr_hw)
            obs_lr = (
                self.spectral_dict(abund_lr)
                if self.spectral_dict is not None
                else abund_lr
            )
            bg = (
                F.softplus(self.raw_background)
                if self.raw_background is not None
                else 0.0
            )
            return (obs_lr + bg).clamp(min=1e-6)

        abund_lr = self.psf_downsampler(probs_hr, lr_hw)
        if self.spectral_dict is not None:
            return self.spectral_dict(abund_lr)
        return abund_lr

    def compute_loss_from_logits(
        self,
        logits: torch.Tensor,
        guide_map: torch.Tensor,
        coarse_target: torch.Tensor | None = None,
        aligner: DifferentiableGuideAligner | None = None,
    ) -> torch.Tensor:
        """Evaluate the configured physical objective between HR/patch ``logits``
        and ``coarse_target`` (or ``guide_map``).
        """
        bsz, _, height, width = logits.shape
        ref = coarse_target if coarse_target is not None else guide_map
        if ref.dim() == 2:
            ref = ref.unsqueeze(0).unsqueeze(0)
        elif ref.dim() == 3:
            ref = ref.unsqueeze(0)
        ref = ref.float().to(logits.device)
        if ref.shape[0] == 1 and bsz > 1:
            ref = ref.expand(bsz, -1, -1, -1)

        lr_hw = (int(ref.shape[-2]), int(ref.shape[-1]))

        if self.loss_fn == "forward_psf_poisson":
            self.maybe_initialize_poisson_scale(ref)
            pred_rate = self.forward_coarse_prediction(
                logits, lr_hw=lr_hw, aligner=aligner, target_hint=ref
            )
            return poisson_deviance_loss(pred_rate, ref)

        if self.loss_fn == "compositional_kl":
            warped = (
                aligner(logits, lr_hw)
                if (aligner is not None and aligner.mode != "none")
                else logits
            )
            simplex_hr = self.logits_to_simplex_probs(warped.float(), target=ref)
            if self.spectral_dict is not None:
                fg_hr = (
                    simplex_hr[:, 1:]
                    if simplex_hr.shape[1] == self.n_channels + 1
                    else simplex_hr
                )
                pred_spec = self.spectral_dict(
                    self.psf_downsampler(fg_hr, lr_hw)
                ).clamp(min=1e-6)
                tgt_spec = ref.clamp(min=0.0)
                tgt_norm = tgt_spec / tgt_spec.sum(dim=1, keepdim=True).clamp(min=1e-6)
                pred_norm = pred_spec / pred_spec.sum(dim=1, keepdim=True).clamp(
                    min=1e-6
                )
                kl = tgt_norm * (
                    torch.log(tgt_norm.clamp(min=1e-6))
                    - torch.log(pred_norm.clamp(min=1e-6))
                )
                return kl.sum(dim=1).mean()

            if simplex_hr.shape[1] == self.n_channels + 1:
                fg_sum = ref.clamp(0.0, 1.0).sum(dim=1, keepdim=True)
                # Normalize if target sum slightly exceeds 1 due to noise
                scale = torch.clamp(fg_sum, min=1.0)
                fg_clean = ref.clamp(0.0, 1.0) / scale
                bg_tgt = (1.0 - fg_clean.sum(dim=1, keepdim=True)).clamp(0.0, 1.0)
                tgt_simplex = torch.cat([bg_tgt, fg_clean], dim=1)
                pred_simplex = self.psf_downsampler(simplex_hr, lr_hw)
                pred_simplex = pred_simplex / pred_simplex.sum(
                    dim=1, keepdim=True
                ).clamp(min=1e-6)
            else:
                tgt_simplex = ref.clamp(min=0.0)
                tgt_simplex = tgt_simplex / tgt_simplex.sum(dim=1, keepdim=True).clamp(
                    min=1e-6
                )
                pred_simplex = self.psf_downsampler(simplex_hr, lr_hw)
                pred_simplex = pred_simplex / pred_simplex.sum(
                    dim=1, keepdim=True
                ).clamp(min=1e-6)

            kl_div = tgt_simplex * (
                torch.log(tgt_simplex.clamp(min=1e-6))
                - torch.log(pred_simplex.clamp(min=1e-6))
            )
            return kl_div.sum(dim=1).mean()

        warped_logits = (
            aligner(logits, lr_hw)
            if (aligner is not None and aligner.mode != "none")
            else logits
        )

        if self.spectral_dict is not None:
            pred_cube = self.spectral_dict(
                self.psf_downsampler(torch.sigmoid(warped_logits.float()), lr_hw)
            )
            return F.mse_loss(pred_cube, ref)

        if self.loss_fn in ("forward_psf_bce", "forward_psf_mse"):
            if self.loss_fn == "forward_psf_mse":
                pooled_probs = self.psf_downsampler(torch.sigmoid(warped_logits), lr_hw)
                return F.mse_loss(pooled_probs, ref)
            if (
                aligner is not None
                and aligner.mode != "none"
                or self.psf_downsampler.learnable
            ):
                probs_hr = torch.sigmoid(warped_logits)
                pooled_p = self.psf_downsampler(probs_hr, lr_hw).clamp(1e-6, 1.0 - 1e-6)
                n_pos = ref.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
                n_neg = (1.0 - ref).sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
                pos_weight = (n_neg / n_pos).clamp(0.1, 20.0)
                bce_elem = -(
                    pos_weight * ref * torch.log(pooled_p)
                    + (1.0 - ref) * torch.log(1.0 - pooled_p)
                )
                return bce_elem.mean()
            pooled_logits = self.psf_downsampler(warped_logits, lr_hw)
            n_pos = ref.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
            n_neg = (1.0 - ref).sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
            pos_weight = (n_neg / n_pos).clamp(0.1, 20.0)
            return F.binary_cross_entropy_with_logits(
                pooled_logits, ref, pos_weight=pos_weight
            )

        if lr_hw != (height, width):
            ref = F.interpolate(
                ref, size=(height, width), mode="bilinear", align_corners=False
            )

        pred_flat = warped_logits.reshape(bsz * self.n_channels, -1)
        target_flat = ref.reshape(bsz * self.n_channels, -1)

        if self.loss_fn == "bce":
            n_pos = target_flat.sum(dim=-1, keepdim=True).clamp(min=1.0)
            n_neg = (1.0 - target_flat).sum(dim=-1, keepdim=True).clamp(min=1.0)
            pos_weight = (n_neg / n_pos).to(pred_flat.device)
            return F.binary_cross_entropy_with_logits(
                pred_flat, target_flat, pos_weight=pos_weight
            )

        pred_c = pred_flat - pred_flat.mean(dim=-1, keepdim=True)
        target_c = target_flat - target_flat.mean(dim=-1, keepdim=True)
        corr = (pred_c * target_c).sum(dim=-1) / (
            pred_c.norm(dim=-1) * target_c.norm(dim=-1) + 1e-8
        )
        return (1.0 - corr).mean()

    @torch.no_grad()
    def diagnostics(
        self, hr_hw: tuple[int, int], lr_hw: tuple[int, int]
    ) -> dict[str, Any]:
        """Return physical forward-model diagnostics (PSF widths, flux, basis)."""
        info: dict[str, Any] = dict(self.psf_downsampler.diagnostics(hr_hw, lr_hw))
        info["loss_fn"] = self.loss_fn
        if self.log_flux_scale is not None:
            flux_vals = self.log_flux_scale.detach().exp().cpu().flatten().tolist()
            info["poisson_flux_scale"] = [round(float(v), 4) for v in flux_vals]
        if self.spectral_dict is not None:
            info["spectral_unmixing_components"] = self.spectral_dict.n_components
            info["spectral_channels"] = self.spectral_dict.spectral_channels
        return info
