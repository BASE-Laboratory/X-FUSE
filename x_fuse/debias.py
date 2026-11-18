"""Apositional Vision Transformer feature debiasing to eliminate spatial
coordinate-ramp shortcut learning on homogeneous micrographs and macroscopic
chemical diffusion gradients.

Motivated by recent findings on ViT positional leakage in electron/X-ray
microscopy (e.g., ``DINO-SAW``, Docherty et al., arXiv:2603.16840), frozen
``DINOv2``, ``DINOv3``, ``MAE``, and ``SAM`` backbones contain channels that act
as smooth ``(y, x)`` coordinate ramps. When supervising a probe with a coarse
physical map exhibiting a macroscopic spatial gradient (e.g. core-shell catalyst
pellet or battery electrode lithiation gradient), an un-debiased probe can
overfit to coordinate-ramp channels instead of true grain morphology.

``ApositionalFeatureDebiaser`` removes spatial polynomial ramp components via:
1. Closed-form QR orthonormal spatial polynomial projection (``r2_threshold=0.0``
   projects out ``(y, x)`` polynomials across all channels to machine precision;
   ``r2_threshold > 0`` selectively detrends or attenuates ramp-dominated
   channels), or
2. Content-free noise/flip calibration (``calibrate_from_backbone``), identifying
   intrinsic ViT positional channels on isotropic white noise so genuine
   sample-level microstructural density gradients remain completely untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn


def build_orthonormal_spatial_basis(
    height: int,
    width: int,
    degree: int = 2,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a zero-mean orthonormal spatial polynomial basis ``Q`` of shape
    ``(H * W, P)`` spanning non-constant polynomials in normalized coordinates
    ``(y, x) in [-1, 1]^2`` up to total degree ``degree`` (1 or 2).
    """
    if degree < 1:
        raise ValueError(f"Polynomial degree must be >= 1, got {degree}.")
    ys = (
        torch.zeros(1, device=device, dtype=torch.float32)
        if height <= 1
        else torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=torch.float32)
    )
    xs = (
        torch.zeros(1, device=device, dtype=torch.float32)
        if width <= 1
        else torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=torch.float32)
    )
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    y_flat = yy.reshape(-1)
    x_flat = xx.reshape(-1)

    cols: list[torch.Tensor] = [y_flat, x_flat]
    if degree >= 2:
        cols.extend([y_flat.square(), y_flat * x_flat, x_flat.square()])
    if degree >= 3:
        cols.extend(
            [
                y_flat.pow(3),
                y_flat.square() * x_flat,
                y_flat * x_flat.square(),
                x_flat.pow(3),
            ]
        )

    raw = torch.stack(cols, dim=-1)  # (N, P)
    n_pts = int(raw.shape[0])
    if n_pts <= 2:
        return torch.zeros((n_pts, 0), device=device, dtype=dtype)
    # Retain lower-degree polynomials on small patch grids (e.g. 2x2 -> linear [y, x])
    max_cols = min(int(raw.shape[1]), n_pts - 2)
    centered = raw[:, :max_cols] - raw[:, :max_cols].mean(dim=0, keepdim=True)
    q_mat, r_mat = torch.linalg.qr(centered, mode="reduced")
    diag = r_mat.diagonal().abs()
    keep = diag > 1e-6
    return q_mat[:, keep].to(dtype=dtype)


def measure_positional_r2(features: torch.Tensor, degree: int = 2) -> torch.Tensor:
    """Compute per-channel spatial coordinate ``R^2`` in ``[0, 1]`` for a
    ``(B, C, H, W)`` feature map, returning ``0`` for near-constant channels.
    """
    if features.ndim == 3:
        features = features.unsqueeze(0)
    bsz, channels, height, width = features.shape
    q_basis = build_orthonormal_spatial_basis(
        height, width, degree=degree, device=features.device, dtype=torch.float32
    )
    if q_basis.shape[1] == 0:
        return torch.zeros(channels, device=features.device, dtype=torch.float32)

    f_flat = features.detach().float().reshape(bsz, channels, -1)
    f_centered = f_flat - f_flat.mean(dim=-1, keepdim=True)
    chan_var = f_centered.var(dim=-1, unbiased=False)
    tot_ss = f_centered.square().sum(dim=-1).clamp(min=1e-12)  # (B, C)
    proj_coeff = f_centered @ q_basis  # (B, C, P)
    reg_ss = proj_coeff.square().sum(dim=-1)  # (B, C)
    r2_bc = torch.where(chan_var > 1e-7, reg_ss / tot_ss, torch.zeros_like(reg_ss))
    return r2_bc.mean(dim=0).clamp(0.0, 1.0)


class ApositionalFeatureDebiaser(nn.Module):
    """Project out spatial coordinate-ramp components from dense or patch-level
    Vision Transformer features ``(B, C, H, W)``.

    Modes:
    - ``mode="orthogonal"`` (default): Projects out spatial coordinate polynomials
      up to ``degree``. When ``r2_threshold == 0.0``, applies exact orthogonal
      projection across all channels (driving linear/quadratic coordinate ``R^2``
      to ``0``). When ``r2_threshold > 0.0``, only detrends channels whose
      positional ``R^2`` exceeds ``r2_threshold`` (or calibrated noise channels).
    - ``mode="attenuate"``: Scales each channel by ``(1 - pos_r2)`` to damp
      positional channels without altering spatial profiles of content channels.
    """

    def __init__(
        self,
        enabled: bool = False,
        degree: int = 2,
        mode: Literal["orthogonal", "attenuate"] = "orthogonal",
        r2_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.degree = int(degree)
        self.mode: Literal["orthogonal", "attenuate"] = mode
        self.r2_threshold = float(r2_threshold)
        self._basis_cache: dict[tuple[int, int, str], torch.Tensor] = {}
        self.register_buffer("calibrated_pos_mask", None, persistent=False)
        self.register_buffer("calibrated_channel_weights", None, persistent=False)

    def _get_basis(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        key = (height, width, str(device))
        cached = self._basis_cache.get(key)
        if cached is not None and cached.device == device:
            return cached
        q_mat = build_orthonormal_spatial_basis(
            height, width, degree=self.degree, device=device, dtype=torch.float32
        )
        # Only retain modest-sized bases in RAM cache
        if height * width <= 65536:
            if len(self._basis_cache) >= 16:
                self._basis_cache.clear()
            self._basis_cache[key] = q_mat
        return q_mat

    @torch.no_grad()
    def calibrate_on_homogeneous_noise(
        self,
        feature_extractor_fn: Callable[[torch.Tensor], torch.Tensor],
        image_hw: tuple[int, int] = (224, 224),
        n_samples: int = 3,
        r2_cutoff: float = 0.15,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> torch.Tensor:
        """Identify intrinsic content-free ViT positional channels by probing
        ``feature_extractor_fn`` with isotropic white-noise images (which contain
        zero macroscopic spatial gradient). Any channel exhibiting high spatial
        coordinate ``R^2`` on white noise is an intrinsic ViT positional ramp.

        Returns the per-channel intrinsic positional ``R^2`` vector ``(C,)``.
        """
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        r2_accum: torch.Tensor | None = None
        for _ in range(max(1, int(n_samples))):
            noise = torch.randn(
                (3, image_hw[0], image_hw[1]), generator=gen, dtype=torch.float32
            ).to(device)
            feats = feature_extractor_fn(noise)
            if feats.ndim == 3:
                feats = feats.unsqueeze(0)
            r2_i = measure_positional_r2(feats, degree=self.degree)
            r2_accum = r2_i if r2_accum is None else (r2_accum + r2_i)

        assert r2_accum is not None
        mean_r2 = r2_accum / max(1, int(n_samples))
        self.calibrated_pos_mask = mean_r2 >= float(r2_cutoff)
        self.calibrated_channel_weights = (1.0 - mean_r2).clamp(0.0, 1.0)
        return mean_r2

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return debiased feature tensor of identical shape and dtype."""
        if not self.enabled:
            return features
        squeeze_b = features.ndim == 3
        x = features.unsqueeze(0) if squeeze_b else features
        bsz, channels, height, width = x.shape

        if self.mode == "attenuate":
            if (
                isinstance(self.calibrated_channel_weights, torch.Tensor)
                and self.calibrated_channel_weights.numel() == channels
            ):
                w = self.calibrated_channel_weights.to(device=x.device, dtype=x.dtype)
            else:
                r2 = measure_positional_r2(x, degree=self.degree)
                w = (1.0 - r2).clamp(0.0, 1.0).to(dtype=x.dtype)
            out = x * w.view(1, channels, 1, 1)
            return out.squeeze(0) if squeeze_b else out

        q_basis = self._get_basis(height, width, x.device)
        if q_basis.shape[1] == 0:
            return features

        f_flat = x.float().reshape(bsz, channels, -1)  # (B, C, N)
        f_mean = f_flat.mean(dim=-1, keepdim=True)
        f_cent = f_flat - f_mean
        proj = (f_cent @ q_basis) @ q_basis.T  # (B, C, N) zero-mean ramp component

        if (
            isinstance(self.calibrated_pos_mask, torch.Tensor)
            and self.calibrated_pos_mask.numel() == channels
        ):
            mask = self.calibrated_pos_mask.to(device=x.device).view(1, channels, 1)
            debiased = torch.where(mask, f_flat - proj, f_flat)
        elif self.r2_threshold > 0.0:
            chan_var = f_cent.var(dim=-1, keepdim=True, unbiased=False)
            tot_ss = f_cent.square().sum(dim=-1, keepdim=True).clamp(min=1e-12)
            reg_ss = proj.square().sum(dim=-1, keepdim=True)
            mask = (chan_var > 1e-7) & ((reg_ss / tot_ss) >= self.r2_threshold)
            debiased = torch.where(mask, f_flat - proj, f_flat)
        else:
            debiased = f_flat - proj

        out = debiased.reshape(bsz, channels, height, width).to(features.dtype)
        return out.squeeze(0) if squeeze_b else out
