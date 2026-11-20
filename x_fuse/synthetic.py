"""Controlled 2D and 3D synthetic correlative materials phantom generators for
systematic super-resolution, contrast-to-zero, noise, misalignment, and 2.5D
volumetric benchmarking.
"""

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from scipy.ndimage import convolve, gaussian_filter, shift as nd_shift

BeamProfile: TypeAlias = Literal["gaussian", "moffat", "boxcar"]


def _blur_and_pool_2d(
    channel: np.ndarray,
    downsample_factor: int,
    sigma: float,
    beam_profile: BeamProfile = "gaussian",
) -> np.ndarray:
    """Apply realistic synchrotron/EDS probe blur (``gaussian``, heavy-tailed
    Lorentzian/``moffat`` diffraction wings, or flat ``boxcar`` microbeam) and
    integrate onto the coarse ``(H // f, W // f)`` detector grid.
    """
    h, w = channel.shape
    h_lr, w_lr = h // downsample_factor, w // downsample_factor
    src = channel.astype(np.float32)
    if sigma > 0.0 and beam_profile != "boxcar":
        if beam_profile == "gaussian":
            src = gaussian_filter(src, sigma=sigma, mode="nearest")
        elif beam_profile == "moffat":
            rad = max(2, int(round(3.5 * sigma)))
            ax = np.arange(-rad, rad + 1, dtype=np.float32)
            yy, xx = np.meshgrid(ax, ax, indexing="ij")
            alpha_m = max(0.85 * sigma, 1e-3)
            kern = (1.0 + (yy**2 + xx**2) / (alpha_m**2)) ** (-1.75)
            kern = (kern / kern.sum()).astype(np.float32)
            src = convolve(src, kern, mode="nearest")
        else:
            raise ValueError(
                f"Unknown beam_profile={beam_profile!r}; expected 'gaussian', "
                "'moffat', or 'boxcar'."
            )
    elif beam_profile not in ("gaussian", "moffat", "boxcar"):
        raise ValueError(
            f"Unknown beam_profile={beam_profile!r}; expected 'gaussian', "
            "'moffat', or 'boxcar'."
        )
    return src.reshape(h_lr, downsample_factor, w_lr, downsample_factor).mean(
        axis=(1, 3)
    )


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
    beam_profile: BeamProfile = "gaussian",
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
    - ``beam_profile``: Coarse probe profile (``"gaussian"``, ``"moffat"``, or
      ``"boxcar"``) to test super-resolution under mismatched forward PSFs.
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
        pooled = _blur_and_pool_2d(
            ch,
            downsample_factor=downsample_factor,
            sigma=float(sigma),
            beam_profile=beam_profile,
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
    shape_zyx: tuple[int, int, int] | None = None,
    depth_psf_sigma: float = 0.0,
) -> CorrelativePhantom3D:
    """Synthesize a 3D multi-phase porous/composite volume ``(Z, Y, X)`` with
    anisotropic or isotropic 3D correlation lengths for benchmarking 2D slice-by-slice
    vs. SuRVoS 2.5D slab & orthogonal tri-plane fusion."""
    eff_shape = shape_zyx if shape_zyx is not None else shape
    z_dim, y_dim, x_dim = (
        (d // downsample_factor) * downsample_factor for d in eff_shape
    )
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
    sig_xy = 0.45 * f
    sig_z = max(sig_xy, float(depth_psf_sigma))
    for m in range(2):
        blurred = gaussian_filter(
            gt_channels[m], sigma=(sig_z, sig_xy, sig_xy), mode="nearest"
        )
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


def generate_localized_reaction_phantom_3d(
    shape: tuple[int, int, int] = (48, 48, 48),
    downsample_factor: int = 4,
    reaction_z_band: tuple[int, int] = (26, 38),
    depth_psf_sigma: float = 2.0,
    hr_noise_std: float = 0.04,
    poisson_counts: float | None = 220.0,
    seed: int = 42,
) -> CorrelativePhantom3D:
    """Synthesize a 4-class 3D tomogram ``(Z, Y, X)`` with a spatially localized
    intermetallic/reaction-front phase (``label = 3``) confined strictly to
    ``z in [reaction_z_band[0], reaction_z_band[1])``.

    ``coarse_guide_3d`` has shape
    ``(3, Z, Y // downsample_factor, X // downsample_factor)``
    so any candidate slice index ``z in [0, Z)`` recommended during active beamtime
    steering can be queried directly with realistic axial PSF blur.
    """
    f = max(1, int(downsample_factor))
    z_dim = max(8, int(shape[0]))
    y_dim = max(f, (int(shape[1]) // f) * f)
    x_dim = max(f, (int(shape[2]) // f) * f)
    z0 = max(0, min(z_dim - 1, int(reaction_z_band[0])))
    z1 = max(z0 + 1, min(z_dim, int(reaction_z_band[1])))
    rng = np.random.default_rng(seed)

    field_a = gaussian_filter(
        rng.standard_normal((z_dim, y_dim, x_dim)).astype(np.float32),
        sigma=(5.0, 4.5, 4.5),
        mode="nearest",
    )
    field_b = gaussian_filter(
        rng.standard_normal((z_dim, y_dim, x_dim)).astype(np.float32),
        sigma=(4.5, 4.0, 4.0),
        mode="nearest",
    )
    field_c = gaussian_filter(
        rng.standard_normal((z_dim, y_dim, x_dim)).astype(np.float32),
        sigma=(2.2, 3.2, 3.2),
        mode="nearest",
    )

    mask_a = field_a >= float(np.quantile(field_a, 0.72))
    mask_b = (field_b >= float(np.quantile(field_b, 0.72))) & (~mask_a)

    in_band = np.zeros((z_dim, 1, 1), dtype=bool)
    in_band[z0:z1] = True
    band_vals = field_c[z0:z1]
    thresh_c = float(np.quantile(band_vals, 0.42))
    mask_c = in_band & (field_c >= thresh_c)

    mask_a = mask_a & (~mask_c)
    mask_b = mask_b & (~mask_c)

    gt_labels = np.zeros((z_dim, y_dim, x_dim), dtype=np.int64)
    gt_labels[mask_a] = 1
    gt_labels[mask_b] = 2
    gt_labels[mask_c] = 3
    gt_channels = np.stack(
        [
            mask_a.astype(np.float32),
            mask_b.astype(np.float32),
            mask_c.astype(np.float32),
        ],
        axis=0,
    )

    # Phase C sits at intermediate attenuation (0.60) with fine cellular contrast
    # so models trained only on end-cap slices exhibit peak entropy + tri-plane
    # variance across the unvisited reaction zone.
    yy, xx = np.meshgrid(
        np.arange(y_dim, dtype=np.float32),
        np.arange(x_dim, dtype=np.float32),
        indexing="ij",
    )
    fine_tex = 0.09 * (
        np.sin(2.0 * np.pi * yy / 4.0) * np.cos(2.0 * np.pi * xx / 4.0)
    ).astype(np.float32)

    hr_vol = np.full((z_dim, y_dim, x_dim), 0.12, dtype=np.float32)
    hr_vol[mask_a] = 0.88
    hr_vol[mask_b] = 0.36
    hr_vol[mask_c] = (0.61 + np.broadcast_to(fine_tex, hr_vol.shape))[mask_c]
    hr_vol = hr_vol + rng.normal(0.0, hr_noise_std, size=hr_vol.shape).astype(
        np.float32
    )
    hr_vol = np.clip(hr_vol, 0.0, 1.0)

    y_lr, x_lr = y_dim // f, x_dim // f
    coarse_3d = np.zeros((3, z_dim, y_lr, x_lr), dtype=np.float32)
    sig_xy = 0.45 * f
    sig_z = max(0.0, float(depth_psf_sigma))
    for m in range(3):
        blurred = gaussian_filter(
            gt_channels[m], sigma=(sig_z, sig_xy, sig_xy), mode="nearest"
        )
        pooled = blurred.reshape(z_dim, y_lr, f, x_lr, f).mean(axis=(2, 4))
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


@dataclass
class SpectralCubePhantom2D:
    """Paired 2D structural micrograph + coarse 3D Poisson spectral/diffraction cube."""

    hr_image: np.ndarray  # (H, W) float32 in [0, 1]
    spectral_cube_lr: np.ndarray  # (E, H_lr, W_lr) float32 raw photon/X-ray counts
    gt_spectra: np.ndarray  # (2, E) float32 unit-sum endmember spectra
    gt_labels: np.ndarray  # (H, W) int64 in {0, 1, 2}
    gt_phase_channels: np.ndarray  # (2, H, W) float32


def generate_textured_isointensity_phantom_2d(
    size: int = 128,
    downsample_factor: int = 8,
    texture_amplitude: float = 0.28,
    hr_noise_std: float = 0.015,
    beam_profile: BeamProfile = "gaussian",
    seed: int = 42,
) -> CorrelativePhantom2D:
    """Synthesize an iso-intensity textured correlative phantom where Phase A (1)
    and Phase B (2) share the **exact same mean grayscale intensity** (``0.55``),
    differing solely by fine high-frequency precipitate/lamellar texture in Phase B.
    """
    if size % downsample_factor != 0:
        size = (size // downsample_factor) * downsample_factor
    rng = np.random.default_rng(seed)

    pore_field = gaussian_filter(
        rng.standard_normal((size, size)).astype(np.float32),
        sigma=max(2.5, size / 22.0),
        mode="wrap",
    )
    phase_field = gaussian_filter(
        rng.standard_normal((size, size)).astype(np.float32),
        sigma=max(2.2, size / 24.0),
        mode="wrap",
    )
    pores = pore_field <= float(np.quantile(pore_field, 0.14))
    mask_a = (~pores) & (phase_field <= 0.0)
    mask_b = (~pores) & (phase_field > 0.0)

    gt_labels = np.zeros((size, size), dtype=np.int64)
    gt_labels[mask_a] = 1
    gt_labels[mask_b] = 2
    gt_channels = np.stack(
        [mask_a.astype(np.float32), mask_b.astype(np.float32)], axis=0
    )

    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    checker = np.sign(np.sin(np.pi * (yy + 0.5)) * np.sin(np.pi * (xx + 0.5))).astype(
        np.float32
    )

    hr = np.full((size, size), 0.10, dtype=np.float32)
    hr[mask_a] = 0.55
    hr[mask_b] = 0.55 + float(texture_amplitude) * checker[mask_b]
    hr = np.clip(
        hr + rng.normal(0.0, hr_noise_std, size=(size, size)).astype(np.float32),
        0.0,
        1.0,
    )

    h_lr = size // downsample_factor
    coarse = np.zeros((2, h_lr, h_lr), dtype=np.float32)
    for m in range(2):
        coarse[m] = _blur_and_pool_2d(
            gt_channels[m],
            downsample_factor=downsample_factor,
            sigma=0.4 * downsample_factor,
            beam_profile=beam_profile,
        )

    return CorrelativePhantom2D(
        hr_image=hr,
        coarse_guide=coarse,
        gt_labels=gt_labels,
        gt_phase_channels=gt_channels,
    )


def generate_oriented_lamellar_phantom_2d(
    size: int = 224,
    downsample_factor: int = 8,
    lamellar_period_px: float = 4.5,
    lamellar_amplitude: float = 0.22,
    theta_a_deg: float = 25.0,
    theta_b_deg: float = 115.0,
    hr_noise_std: float = 0.015,
    poisson_counts: float | None = 300.0,
    beam_profile: BeamProfile = "moffat",
    seed: int = 42,
) -> CorrelativePhantom2D:
    """Synthesize a 3-constituent eutectic/pearlitic colony phantom where Colony A (1)
    and Colony B (2) share **matched mean intensity, matched variance, and matched
    isotropic gradient magnitude**, differing only in crystallographic lamellar
    orientation (``theta_a_deg`` vs. ``theta_b_deg``), plus dark pores (0).

    Tests orientation-selective foundation-model feature representations against
    scalar intensity thresholding, bilateral range kernels, CNMF, SIREN, and
    isotropic intensity-gradient TV priors.
    """
    if size % downsample_factor != 0:
        size = (size // downsample_factor) * downsample_factor
    rng = np.random.default_rng(seed)

    pore_field = gaussian_filter(
        rng.standard_normal((size, size)).astype(np.float32),
        sigma=max(3.0, size / 40.0),
        mode="wrap",
    )
    colony_field = gaussian_filter(
        rng.standard_normal((size, size)).astype(np.float32),
        sigma=max(3.5, size / 34.0),
        mode="wrap",
    )
    pores = pore_field <= float(np.quantile(pore_field, 0.16))
    mask_a = (~pores) & (colony_field <= 0.0)
    mask_b = (~pores) & (colony_field > 0.0)

    gt_labels = np.zeros((size, size), dtype=np.int64)
    gt_labels[mask_a] = 1
    gt_labels[mask_b] = 2
    gt_channels = np.stack(
        [mask_a.astype(np.float32), mask_b.astype(np.float32)], axis=0
    )

    yy, xx = np.meshgrid(
        np.arange(size, dtype=np.float32),
        np.arange(size, dtype=np.float32),
        indexing="ij",
    )
    freq = 2.0 * np.pi / max(float(lamellar_period_px), 2.0)
    rad_a = np.deg2rad(float(theta_a_deg))
    rad_b = np.deg2rad(float(theta_b_deg))
    wave_a = np.sin(freq * (np.cos(rad_a) * xx + np.sin(rad_a) * yy))
    wave_b = np.sin(freq * (np.cos(rad_b) * xx + np.sin(rad_b) * yy))

    hr = np.full((size, size), 0.12, dtype=np.float32)
    hr[mask_a] = 0.56 + float(lamellar_amplitude) * wave_a[mask_a]
    hr[mask_b] = 0.56 + float(lamellar_amplitude) * wave_b[mask_b]
    hr = np.clip(
        hr + rng.normal(0.0, hr_noise_std, size=(size, size)).astype(np.float32),
        0.0,
        1.0,
    ).astype(np.float32)

    h_lr = size // downsample_factor
    coarse = np.zeros((2, h_lr, h_lr), dtype=np.float32)
    for m in range(2):
        pooled = _blur_and_pool_2d(
            gt_channels[m],
            downsample_factor=downsample_factor,
            sigma=0.45 * downsample_factor,
            beam_profile=beam_profile,
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


def generate_artifact_corrupted_phantom_2d(
    size: int = 224,
    downsample_factor: int = 8,
    curtaining_amplitude: float = 0.13,
    wedge_amplitude: float = 0.16,
    hr_noise_std: float = 0.025,
    poisson_counts: float | None = 250.0,
    beam_profile: BeamProfile = "moffat",
    seed: int = 42,
) -> CorrelativePhantom2D:
    """Synthesize a 3-phase correlative specimen whose high-resolution structural
    micrograph ``hr_image`` suffers from realistic FIB-SEM / STEM preparation
    artifacts—vertical ion-milling curtaining streaks and a linear thickness or
    illumination wedge—while ``coarse_guide`` reflects true phase chemistry.
    """
    base = generate_correlative_phantom_2d(
        size=size,
        downsample_factor=downsample_factor,
        phase_b_contrast=0.32,
        hr_noise_std=hr_noise_std,
        poisson_counts=poisson_counts,
        beam_profile=beam_profile,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 500)
    # Multi-scale vertical FIB curtaining bands modulated slowly along depth Y
    prof_x = gaussian_filter(
        rng.standard_normal(size).astype(np.float32), sigma=1.6, mode="wrap"
    )
    prof_x = prof_x / max(float(np.std(prof_x)), 1e-6)
    depth_mod = np.linspace(0.65, 1.25, size, dtype=np.float32)[:, None]
    curtaining = float(curtaining_amplitude) * depth_mod * prof_x[None, :]

    # Linear thickness / detector shading wedge across (Y, X)
    ramp_x = np.linspace(-1.0, 1.0, size, dtype=np.float32)[None, :]
    ramp_y = np.linspace(-0.5, 0.5, size, dtype=np.float32)[:, None]
    wedge = float(wedge_amplitude) * (0.75 * ramp_x + 0.25 * ramp_y)

    hr_corrupted = np.clip(base.hr_image + curtaining + wedge, 0.0, 1.0).astype(
        np.float32
    )
    return CorrelativePhantom2D(
        hr_image=hr_corrupted,
        coarse_guide=base.coarse_guide,
        gt_labels=base.gt_labels,
        gt_phase_channels=base.gt_phase_channels,
    )


def generate_macroscopic_gradient_phantom_2d(
    size: int = 128,
    downsample_factor: int = 8,
    hr_noise_std: float = 0.02,
    seed: int = 77,
) -> CorrelativePhantom2D:
    """Synthesize a diffusion-couple / electrode-gradient phantom where discrete
    Phase A grains follow a strong left-to-right spatial concentration gradient,
    testing resistance to ViT coordinate-ramp shortcut learning.
    """
    if size % downsample_factor != 0:
        size = (size // downsample_factor) * downsample_factor
    rng = np.random.default_rng(seed)

    base = gaussian_filter(
        rng.standard_normal((size, size)).astype(np.float32),
        sigma=max(2.5, size / 24.0),
        mode="nearest",
    )
    base = (base - base.mean()) / (base.std() + 1e-6)
    ramp_x = np.linspace(1.1, -1.1, size, dtype=np.float32)[None, :]
    mask_a = (base + 0.85 * ramp_x) > 0.45
    mask_b = ((base - 0.85 * ramp_x) > 0.65) & (~mask_a)

    gt_labels = np.zeros((size, size), dtype=np.int64)
    gt_labels[mask_a] = 1
    gt_labels[mask_b] = 2
    gt_channels = np.stack(
        [mask_a.astype(np.float32), mask_b.astype(np.float32)], axis=0
    )

    hr = np.full((size, size), 0.18, dtype=np.float32)
    hr[mask_a] = 0.85
    hr[mask_b] = 0.50
    hr = np.clip(
        hr + rng.normal(0.0, hr_noise_std, size=(size, size)).astype(np.float32),
        0.0,
        1.0,
    )

    h_lr = size // downsample_factor
    coarse = np.zeros((2, h_lr, h_lr), dtype=np.float32)
    for m in range(2):
        blurred = gaussian_filter(
            gt_channels[m], sigma=0.4 * downsample_factor, mode="nearest"
        )
        coarse[m] = blurred.reshape(
            h_lr, downsample_factor, h_lr, downsample_factor
        ).mean(axis=(1, 3))

    return CorrelativePhantom2D(
        hr_image=hr,
        coarse_guide=coarse,
        gt_labels=gt_labels,
        gt_phase_channels=gt_channels,
    )


def generate_spectral_cube_phantom_2d(
    size: int = 96,
    downsample_factor: int = 6,
    n_spectral_bins: int = 24,
    mean_counts: float = 40.0,
    background_counts: float = 0.5,
    bspline_warp_px: float = 0.0,
    beam_profile: BeamProfile = "gaussian",
    seed: int = 99,
) -> SpectralCubePhantom2D:
    """Synthesize a low-dose Poisson 3D hyperspectral/diffraction cube
    ``(E, H_lr, W_lr)`` paired with an ``(H, W)`` structural micrograph and
    optional nonlinear scan-charging distortion ``bspline_warp_px``.
    """
    from scipy.ndimage import map_coordinates

    base = generate_correlative_phantom_2d(
        size=size,
        downsample_factor=downsample_factor,
        phase_b_contrast=0.35,
        hr_noise_std=0.02,
        poisson_counts=None,
        beam_profile=beam_profile,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 10)
    e_grid = np.linspace(0.0, 1.0, n_spectral_bins, dtype=np.float32)
    # Two distinct multi-peak characteristic X-ray / Bragg spectra
    spec_a = np.exp(-((e_grid - 0.25) ** 2) / (2 * 0.06**2)) + 0.45 * np.exp(
        -((e_grid - 0.60) ** 2) / (2 * 0.05**2)
    )
    spec_b = np.exp(-((e_grid - 0.75) ** 2) / (2 * 0.06**2)) + 0.35 * np.exp(
        -((e_grid - 0.42) ** 2) / (2 * 0.05**2)
    )
    gt_spectra = np.stack([spec_a, spec_b], axis=0).astype(np.float32)
    gt_spectra /= gt_spectra.sum(axis=-1, keepdims=True)

    h_lr = size // downsample_factor
    abund_lr = np.zeros((2, h_lr, h_lr), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    if abs(bspline_warp_px) > 1e-5:
        dy = bspline_warp_px * np.sin(np.pi * yy / size) * np.sin(np.pi * xx / size)
        dx = bspline_warp_px * np.sin(np.pi * yy / size)
    else:
        dy = np.zeros_like(yy, dtype=np.float32)
        dx = np.zeros_like(xx, dtype=np.float32)

    for m in range(2):
        src = base.gt_phase_channels[m]
        if abs(bspline_warp_px) > 1e-5:
            src = map_coordinates(
                src, [yy - dy, xx - dx], order=1, mode="nearest"
            ).astype(np.float32)
        abund_lr[m] = _blur_and_pool_2d(
            src,
            downsample_factor=downsample_factor,
            sigma=0.4 * downsample_factor,
            beam_profile=beam_profile,
        )

    noiseless_cube = float(mean_counts) * np.einsum(
        "mhw,me->ehw", abund_lr, gt_spectra
    ) + float(background_counts)
    noisy_cube = rng.poisson(np.clip(noiseless_cube, 0.0, None)).astype(np.float32)

    return SpectralCubePhantom2D(
        hr_image=base.hr_image,
        spectral_cube_lr=noisy_cube,
        gt_spectra=gt_spectra,
        gt_labels=base.gt_labels,
        gt_phase_channels=base.gt_phase_channels,
    )
