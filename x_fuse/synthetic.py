"""Controlled 2D and 3D synthetic correlative materials phantom generators for
systematic super-resolution, contrast-to-zero, noise, misalignment, and 2.5D
volumetric benchmarking.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, shift as nd_shift


@dataclass
class CorrelativePhantom2D:
    """Paired 2D structural micrograph + coarse multi-channel physical guide + GT."""

    hr_image: np.ndarray  # (H, W) float32 in [0, 1]
    coarse_guide: np.ndarray  # (M, H_lr, W_lr) float32 in [0, 1]
    gt_labels: np.ndarray  # (H, W) int64 in {0, 1, 2} (0=matrix, 1=Phase A, 2=Phase B)
    gt_phase_channels: np.ndarray  # (2, H, W) float32 binary masks for Phase A & B


@dataclass
class CorrelativePhantom3D:
    """Paired 3D structural tomogram + coarse 3D physical guide + 3D GT."""

    hr_volume: np.ndarray  # (Z, Y, X) float32 in [0, 1]
    coarse_guide_3d: np.ndarray  # (2, Z_lr, Y_lr, X_lr) float32 in [0, 1]
    gt_labels_3d: np.ndarray  # (Z, Y, X) int64 in {0, 1, 2}
    gt_phase_channels_3d: np.ndarray  # (2, Z, Y, X) float32


def generate_correlative_phantom_2d(
    size: int = 128,
    downsample_factor: int = 8,
    phase_b_contrast: float = 0.35,
    hr_noise_std: float = 0.04,
    psf_sigma: float | None = None,
    poisson_counts: float | None = 200.0,
    misalignment_yx: tuple[float, float] = (0.0, 0.0),
    seed: int = 42,
) -> CorrelativePhantom2D:
    """Synthesize a realistic 3-constituent (Matrix + Phase A + Phase B) 2D
    correlative dataset with exact pixel ground truth.

    Scientific control knobs:
    - ``downsample_factor``: Spatial resolution gap ``H / H_lr`` (e.g. 4x, 8x, 16x).
    - ``phase_b_contrast``: Structural intensity difference between Phase B and the
      background matrix. Setting ``phase_b_contrast = 0.0`` makes Phase B
      **completely invisible** in ``hr_image`` while still present in
      ``coarse_guide[1]``—testing whether X-FUSE's physical forward-model residual
      correctly flags chemically distinct but structurally indistinguishable regions!
    - ``poisson_counts``: Photon/electron shot-noise budget on the coarse guide.
    - ``misalignment_yx``: Sub-pixel or multi-pixel spatial drift on the guide map.
    """
    if size % downsample_factor != 0:
        size = (size // downsample_factor) * downsample_factor
    rng = np.random.default_rng(seed)

    # Smooth correlated fields for interconnected dendrites / pores + inclusions
    field_a = gaussian_filter(
        rng.standard_normal((size, size)).astype(np.float32),
        sigma=max(3.0, size / 18.0),
        mode="wrap",
    )
    field_b = gaussian_filter(
        rng.standard_normal((size, size)).astype(np.float32),
        sigma=max(2.5, size / 22.0),
        mode="wrap",
    )

    thresh_a = float(np.quantile(field_a, 0.72))
    thresh_b = float(np.quantile(field_b, 0.75))
    mask_a = field_a >= thresh_a
    mask_b = (field_b >= thresh_b) & (~mask_a)

    gt_labels = np.zeros((size, size), dtype=np.int64)
    gt_labels[mask_a] = 1
    gt_labels[mask_b] = 2

    gt_channels = np.stack(
        [mask_a.astype(np.float32), mask_b.astype(np.float32)], axis=0
    )

    # Build structural image I_HR:
    # Matrix = 0.15, Phase A = 0.85 (always strong contrast),
    # Phase B = 0.15 + phase_b_contrast (sweeps down to 0.15 = invisible!)
    hr = np.full((size, size), 0.15, dtype=np.float32)
    hr[mask_a] = 0.85
    hr[mask_b] = 0.15 + float(phase_b_contrast)
    # Add subtle intra-grain texture + Gaussian detector noise
    grain_tex = 0.015 * gaussian_filter(
        rng.standard_normal((size, size)).astype(np.float32), sigma=1.2
    )
    hr = (
        hr
        + grain_tex
        + rng.normal(0.0, hr_noise_std, size=(size, size)).astype(np.float32)
    )
    hr = np.clip(hr, 0.0, 1.0).astype(np.float32)

    # Forward physical model for G_LR: shift -> PSF blur -> box pool -> Poisson
    sigma = (
        0.5 * downsample_factor
        if psf_sigma is None and downsample_factor > 1
        else (psf_sigma or 0.0)
    )
    h_lr = size // downsample_factor
    coarse = np.zeros((2, h_lr, h_lr), dtype=np.float32)
    for m in range(2):
        ch = gt_channels[m]
        if misalignment_yx != (0.0, 0.0):
            ch = nd_shift(ch, shift=misalignment_yx, order=1, mode="nearest")
        if sigma > 0:
            ch = gaussian_filter(ch, sigma=sigma, mode="nearest")
        pooled = ch.reshape(h_lr, downsample_factor, h_lr, downsample_factor).mean(
            axis=(1, 3)
        )
        if poisson_counts is not None and poisson_counts > 0:
            pooled = rng.poisson(np.clip(pooled, 0.0, 1.0) * poisson_counts).astype(
                np.float32
            ) / float(poisson_counts)
        coarse[m] = np.clip(pooled, 0.0, 1.0)

    return CorrelativePhantom2D(
        hr_image=hr,
        coarse_guide=coarse,
        gt_labels=gt_labels,
        gt_phase_channels=gt_channels,
    )


def generate_correlative_phantom_3d(
    shape: tuple[int, int, int] = (32, 64, 64),
    downsample_factor: int = 4,
    phase_b_contrast: float = 0.35,
    hr_noise_std: float = 0.06,
    poisson_counts: float | None = 150.0,
    seed: int = 123,
) -> CorrelativePhantom3D:
    """Synthesize a 3D multi-phase porous/composite volume ``(Z, Y, X)`` with
    anisotropic or isotropic 3D correlation lengths for benchmarking 2D slice-by-slice
    vs. SuRVoS 2.5D slab & orthogonal tri-plane fusion."""
    z_dim, y_dim, x_dim = ((d // downsample_factor) * downsample_factor for d in shape)
    rng = np.random.default_rng(seed)

    field_a = gaussian_filter(
        rng.standard_normal((z_dim, y_dim, x_dim)).astype(np.float32),
        sigma=(3.5, 4.5, 4.5),
        mode="wrap",
    )
    field_b = gaussian_filter(
        rng.standard_normal((z_dim, y_dim, x_dim)).astype(np.float32),
        sigma=(3.0, 4.0, 4.0),
        mode="wrap",
    )

    mask_a = field_a >= float(np.quantile(field_a, 0.74))
    mask_b = (field_b >= float(np.quantile(field_b, 0.76))) & (~mask_a)

    gt_labels = np.zeros((z_dim, y_dim, x_dim), dtype=np.int64)
    gt_labels[mask_a] = 1
    gt_labels[mask_b] = 2
    gt_channels = np.stack(
        [mask_a.astype(np.float32), mask_b.astype(np.float32)], axis=0
    )

    hr_vol = np.full((z_dim, y_dim, x_dim), 0.15, dtype=np.float32)
    hr_vol[mask_a] = 0.85
    hr_vol[mask_b] = 0.15 + float(phase_b_contrast)
    hr_vol = hr_vol + rng.normal(0.0, hr_noise_std, size=hr_vol.shape).astype(
        np.float32
    )
    hr_vol = np.clip(hr_vol, 0.0, 1.0)

    f = downsample_factor
    z_lr, y_lr, x_lr = z_dim // f, y_dim // f, x_dim // f
    coarse_3d = np.zeros((2, z_lr, y_lr, x_lr), dtype=np.float32)
    for m in range(2):
        blurred = gaussian_filter(gt_channels[m], sigma=0.45 * f, mode="nearest")
        pooled = blurred.reshape(z_lr, f, y_lr, f, x_lr, f).mean(axis=(1, 3, 5))
        if poisson_counts is not None and poisson_counts > 0:
            pooled = rng.poisson(np.clip(pooled, 0.0, 1.0) * poisson_counts).astype(
                np.float32
            ) / float(poisson_counts)
        coarse_3d[m] = np.clip(pooled, 0.0, 1.0)

    return CorrelativePhantom3D(
        hr_volume=hr_vol,
        coarse_guide_3d=coarse_3d,
        gt_labels_3d=gt_labels,
        gt_phase_channels_3d=gt_channels,
    )
