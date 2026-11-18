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
        y_mat = torch.nan_to_num(
            x.permute(0, 2, 3, 1).reshape(-1, spec_e),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(min=0.0)
        if float(y_mat.max().item()) <= 1e-12:
            return torch.zeros(
                (bsz, self.n_components, h_lr, w_lr),
                device=x.device,
                dtype=torch.float32,
            )
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

    @torch.no_grad()
    def project_cube(self, cube_lr: torch.Tensor, n_iters: int = 25) -> torch.Tensor:
        """Unmix a new ``([B,] E, H_lr, W_lr)`` spectral cube against the **fixed**
        learned endmember basis ``self.spectra()`` via multiplicative NMF updates,
        returning ``(B, M, H_lr, W_lr)`` abundance maps.
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
        y_mat = torch.nan_to_num(
            x.permute(0, 2, 3, 1).reshape(-1, spec_e),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(min=0.0)
        if float(y_mat.max().item()) <= 1e-12:
            return torch.zeros(
                (bsz, self.n_components, h_lr, w_lr),
                device=x.device,
                dtype=torch.float32,
            )
        h_basis = self.spectra().detach().float().to(x.device)  # (M, E)
        energies = y_mat.sum(dim=-1, keepdim=True).clamp(min=1e-4)
        w_abund = (
            torch.full(
                (y_mat.shape[0], self.n_components),
                1.0 / self.n_components,
                device=x.device,
                dtype=torch.float32,
            )
            * energies
        )
        ht_h = h_basis @ h_basis.T
        num_w = y_mat @ h_basis.T
        eps = 1e-7
        for _ in range(max(1, int(n_iters))):
            den_w = (w_abund @ ht_h).clamp(min=eps)
            w_abund = (w_abund * (num_w / den_w)).clamp(min=0.0)
        return (
            w_abund.reshape(bsz, h_lr, w_lr, self.n_components)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def forward(self, abundances_lr: torch.Tensor) -> torch.Tensor:
        """Mix coarse abundances ``(B, M, H_lr, W_lr)`` with basis ``(M, E)``
        into reconstructed spectral cube ``(B, E, H_lr, W_lr)``.
        """
        spec = self.spectra().to(device=abundances_lr.device, dtype=abundances_lr.dtype)
        return torch.einsum("bmhw,me->behw", abundances_lr, spec)


def _masked_spatial_mean(
    elem: torch.Tensor, mask_lr: torch.Tensor | None = None
) -> torch.Tensor:
    """Average ``elem`` ``(B, C, H_lr, W_lr)`` or ``(B, H_lr, W_lr)`` over valid
    coarse pixels ``mask_lr`` ``(H_lr, W_lr)`` when provided.
    """
    if mask_lr is None:
        return elem.mean()
    m = mask_lr.to(device=elem.device, dtype=torch.bool)
    if int(m.sum().item()) == 0:
        return elem.mean()
    if elem.ndim == 4:
        return elem[:, :, m].mean()
    if elem.ndim == 3:
        return elem[:, m].mean()
    return elem.mean()


def poisson_deviance_loss(
    pred_rate: torch.Tensor,
    target_counts: torch.Tensor,
    eps: float = 1e-6,
    mask_lr: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact non-negative Poisson Bregman deviance (shifted Poisson NLL):
    ``mean(pred - y * log(pred) - (y - y * log(y))) >= 0``, achieving ``0`` iff
    ``pred == y`` and maintaining well-conditioned ``O(1)`` values across count regimes.
    """
    # Use smooth additive floor inside log so gradients never vanish at small rates
    lam_pos = pred_rate.float().clamp(min=0.0) + eps
    y = target_counts.float().clamp(min=0.0)
    nll = lam_pos - y * torch.log(lam_pos)
    y_safe = y.clamp(min=eps)
    offset = torch.where(y > 0.0, y - y * torch.log(y_safe), torch.zeros_like(y))
    dev = (nll - offset).clamp(min=0.0)
    return _masked_spatial_mean(dev, mask_lr=mask_lr)


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
        self._resolved_bg_simplex: bool | None = simplex_include_background

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
        if loss_fn == "forward_psf_poisson" or (
            self.spectral_dict is not None and loss_fn != "compositional_kl"
        ):
            self.log_flux_scale = nn.Parameter(
                torch.zeros((1, self.n_channels, 1, 1), dtype=torch.float32)
            )
            self.raw_background = (
                nn.Parameter(torch.full((1, obs_ch, 1, 1), -3.0, dtype=torch.float32))
                if loss_fn == "forward_psf_poisson"
                else None
            )
            if loss_fn != "forward_psf_poisson":
                self.register_parameter("raw_background", None)
            self._scale_initialized = False
        else:
            self.register_parameter("log_flux_scale", None)
            self.register_parameter("raw_background", None)
            self._scale_initialized = True

    def reset_specimen_state(self) -> None:
        """Reset data-dependent scale and simplex flags before fitting a new sample."""
        if self.log_flux_scale is not None:
            self._scale_initialized = False
        self._resolved_bg_simplex = self.simplex_include_background

    @torch.no_grad()
    def maybe_initialize_poisson_scale(
        self, target: torch.Tensor, mask_lr: torch.Tensor | None = None
    ) -> None:
        """Initialize ``log_flux_scale`` and ``raw_background`` from target count
        quantiles on the first training call so optimization starts at physical scale.
        """
        import warnings

        if self._scale_initialized or self.log_flux_scale is None:
            return
        t = torch.nan_to_num(
            target.detach().float(), nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(min=0.0)
        if t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 3:
            t = t.unsqueeze(0)

        if (
            self.loss_fn == "forward_psf_poisson"
            and float(t.max().item()) <= 1.05
            and float(t.max().item()) > 0.0
        ):
            warnings.warn(
                "forward_psf_poisson received guide values <= 1.05; pass raw integer "
                "photon/electron counts (normalization='preserve_counts') rather than "
                "[0, 1]-normalized maps to preserve true Poisson variance.",
                UserWarning,
                stacklevel=2,
            )

        if mask_lr is not None and int(mask_lr.sum().item()) >= 4:
            m = mask_lr.to(device=t.device, dtype=torch.bool)
            flat = t[:, :, m]  # (B, C_obs, N_train)
        else:
            flat = t.flatten(start_dim=2)  # (B, C_obs, N)
        q95 = torch.quantile(flat, 0.95, dim=-1).mean(dim=0)  # (C_obs,)
        q05 = torch.quantile(flat, 0.05, dim=-1).mean(dim=0)  # (C_obs,)

        if self.spectral_dict is not None:
            tot_scale = (
                flat.sum(dim=1).quantile(0.90) / max(self.n_channels, 1)
            ).clamp(min=0.1)
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
        if self.spectral_dict is not None:
            # Spectral normalization cancels overall foreground scale if an unmixed
            # background channel is dropped; partition directly across M endmembers.
            self._resolved_bg_simplex = False
            return False
        if self.n_channels == 1:
            self._resolved_bg_simplex = True
            return True
        if target is not None:
            t = target.detach().float()
            if t.ndim == 3:
                t = t.unsqueeze(0)
            if t.ndim == 4 and t.shape[1] == self.n_channels:
                col_sum = t.sum(dim=1)
                resolved = not (float((col_sum - 1.0).abs().max().item()) < 0.03)
                self._resolved_bg_simplex = resolved
                return resolved
        if self._resolved_bg_simplex is not None:
            return self._resolved_bg_simplex
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
        valid_hr_hw: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Map HR ``logits`` ``(B, M, H, W)`` through spatial alignment, physical
        PSF integration, and optional spectral dictionary mixing.
        """
        src_logits = (
            logits[:, :, : valid_hr_hw[0], : valid_hr_hw[1]]
            if valid_hr_hw is not None
            else logits
        )
        warped_logits = (
            aligner(src_logits, lr_hw)
            if (aligner is not None and aligner.mode != "none")
            else src_logits
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
        if self.log_flux_scale is not None:
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
        valid_hr_hw: tuple[int, int] | None = None,
        loss_mask_lr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate the configured physical objective between HR/patch ``logits``
        and ``coarse_target`` (or ``guide_map``), cropping divisibility padding
        via ``valid_hr_hw`` and excluding conformal holdout pixels via ``loss_mask_lr``.
        """
        src_logits = (
            logits[:, :, : valid_hr_hw[0], : valid_hr_hw[1]]
            if valid_hr_hw is not None
            else logits
        )
        bsz, _, height, width = src_logits.shape
        ref = coarse_target if coarse_target is not None else guide_map
        if ref.dim() == 2:
            ref = ref.unsqueeze(0).unsqueeze(0)
        elif ref.dim() == 3:
            ref = ref.unsqueeze(0)
        ref = torch.nan_to_num(
            ref.float().to(src_logits.device), nan=0.0, posinf=0.0, neginf=0.0
        )
        if ref.shape[0] == 1 and bsz > 1:
            ref = ref.expand(bsz, -1, -1, -1)

        lr_hw = (int(ref.shape[-2]), int(ref.shape[-1]))

        if self.loss_fn == "forward_psf_poisson":
            self.maybe_initialize_poisson_scale(ref, mask_lr=loss_mask_lr)
            pred_rate = self.forward_coarse_prediction(
                src_logits, lr_hw=lr_hw, aligner=aligner, target_hint=ref
            )
            return poisson_deviance_loss(pred_rate, ref, mask_lr=loss_mask_lr)

        if self.loss_fn == "compositional_kl":
            warped = (
                aligner(src_logits, lr_hw)
                if (aligner is not None and aligner.mode != "none")
                else src_logits
            )
            simplex_hr = self.logits_to_simplex_probs(warped.float(), target=ref)
            if self.spectral_dict is not None:
                pred_spec = self.spectral_dict(self.psf_downsampler(simplex_hr, lr_hw))
                tgt_spec = ref.clamp(min=0.0)
                tgt_norm = tgt_spec / tgt_spec.sum(dim=1, keepdim=True).clamp(min=1e-6)
                pred_norm = pred_spec / pred_spec.sum(dim=1, keepdim=True).clamp(
                    min=1e-6
                )
                kl = tgt_norm * (
                    torch.log(tgt_norm + 1e-7) - torch.log(pred_norm + 1e-7)
                )
                return _masked_spatial_mean(kl.sum(dim=1), mask_lr=loss_mask_lr)

            if simplex_hr.shape[1] == self.n_channels + 1:
                fg_sum = ref.clamp(0.0, 1.0).sum(dim=1, keepdim=True)
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
                torch.log(tgt_simplex + 1e-7) - torch.log(pred_simplex + 1e-7)
            )
            return _masked_spatial_mean(kl_div.sum(dim=1), mask_lr=loss_mask_lr)

        warped_logits = (
            aligner(src_logits, lr_hw)
            if (aligner is not None and aligner.mode != "none")
            else src_logits
        )

        if self.spectral_dict is not None:
            self.maybe_initialize_poisson_scale(ref, mask_lr=loss_mask_lr)
            pred_cube = self.forward_coarse_prediction(
                src_logits, lr_hw=lr_hw, aligner=aligner, target_hint=ref
            )
            sq_err = (pred_cube - ref).square()
            return _masked_spatial_mean(sq_err, mask_lr=loss_mask_lr)

        ref_unit = ref.clamp(0.0, 1.0)

        def _masked_pos_weight(
            tgt_map: torch.Tensor, m_bool: torch.Tensor | None
        ) -> torch.Tensor:
            if m_bool is not None and int(m_bool.sum().item()) >= 4:
                sub = tgt_map[:, :, m_bool.to(device=tgt_map.device, dtype=torch.bool)]
                n_pos = sub.sum(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1.0)
                n_neg = (
                    (1.0 - sub).sum(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1.0)
                )
            else:
                n_pos = tgt_map.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
                n_neg = (1.0 - tgt_map).sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
            return (n_neg / n_pos).clamp(0.1, 20.0)

        if self.loss_fn in ("forward_psf_bce", "forward_psf_mse"):
            if self.loss_fn == "forward_psf_mse":
                pooled_probs = self.psf_downsampler(torch.sigmoid(warped_logits), lr_hw)
                return _masked_spatial_mean(
                    (pooled_probs - ref_unit).square(), mask_lr=loss_mask_lr
                )
            pos_weight = _masked_pos_weight(ref_unit, loss_mask_lr)
            if (
                aligner is not None
                and aligner.mode != "none"
                or self.psf_downsampler.learnable
            ):
                probs_hr = torch.sigmoid(warped_logits)
                pooled_p = self.psf_downsampler(probs_hr, lr_hw)
                bce_elem = -(
                    pos_weight * ref_unit * torch.log(pooled_p + 1e-6)
                    + (1.0 - ref_unit)
                    * torch.log((1.0 - pooled_p).clamp(min=0.0) + 1e-6)
                )
                return _masked_spatial_mean(bce_elem, mask_lr=loss_mask_lr)
            pooled_logits = self.psf_downsampler(warped_logits, lr_hw)
            bce_raw = F.binary_cross_entropy_with_logits(
                pooled_logits, ref_unit, pos_weight=pos_weight, reduction="none"
            )
            return _masked_spatial_mean(bce_raw, mask_lr=loss_mask_lr)

        if lr_hw != (height, width):
            ref_unit = F.interpolate(
                ref_unit, size=(height, width), mode="bilinear", align_corners=False
            )
            if loss_mask_lr is not None:
                m_up = (
                    F.interpolate(
                        loss_mask_lr.float().unsqueeze(0).unsqueeze(0),
                        size=(height, width),
                        mode="nearest",
                    )[0, 0]
                    > 0.5
                )
            else:
                m_up = None
        else:
            m_up = loss_mask_lr

        if self.loss_fn == "bce":
            pos_weight = _masked_pos_weight(ref_unit, m_up).to(warped_logits.device)
            bce_hr = F.binary_cross_entropy_with_logits(
                warped_logits, ref_unit, pos_weight=pos_weight, reduction="none"
            )
            return _masked_spatial_mean(bce_hr, mask_lr=m_up)

        if m_up is not None and int(m_up.sum().item()) >= 4:
            pred_flat = warped_logits[:, :, m_up].reshape(bsz * self.n_channels, -1)
            target_flat = ref_unit[:, :, m_up].reshape(bsz * self.n_channels, -1)
        else:
            pred_flat = warped_logits.reshape(bsz * self.n_channels, -1)
            target_flat = ref_unit.reshape(bsz * self.n_channels, -1)

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
