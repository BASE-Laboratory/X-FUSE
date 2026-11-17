"""Sample data locator, open correlative dataset registry (Zenodo), and
experimental hold-out downsampling utilities for quantitative benchmarking.
"""

import hashlib
import json
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class OpenCorrelativeDataset:
    """Metadata for an open-access correlative materials dataset."""

    key: str
    title: str
    hr_modality: str
    guide_modality: str
    zenodo_record_id: str
    doi: str
    description: str


OPEN_DATASETS: dict[str, OpenCorrelativeDataset] = {
    "edxia_cement_sem_eds": OpenCorrelativeDataset(
        key="edxia_cement_sem_eds",
        title="EPFL edxia Cementitious Microstructure SEM-BSE + EDS Hypermaps",
        hr_modality="SEM-BSE",
        guide_modality="Multi-Element EDS (Ca, Si, Al, Fe, Mg, S)",
        zenodo_record_id="3686067",
        doi="10.5281/zenodo.3686067",
        description=(
            "Aligned backscattered electron (BSE) micrographs and multi-element "
            "EDS maps of cementitious materials; benchmarked against Joint "
            "Bilateral Filtering in Georget et al., Cem. Concr. Res. (2021)."
        ),
    ),
    "refractory_alloy_bse_eds_ebsd": OpenCorrelativeDataset(
        key="refractory_alloy_bse_eds_ebsd",
        title="Al-Ti-Nb-Zr-Ta Multiphase Alloy Correlative BSE + EDS + EBSD",
        hr_modality="SEM-BSE",
        guide_modality="EDS + EBSD Phase Maps",
        zenodo_record_id="19182505",
        doi="10.5281/zenodo.19182505",
        description=(
            "Full-resolution BSE micrographs paired with EDAX HDF5 EDS hypermaps "
            "and EBSD phase clustering for multiphase refractory alloys."
        ),
    ),
    "ti_rcca_stem_eds": OpenCorrelativeDataset(
        key="ti_rcca_stem_eds",
        title="Ti-Rich Refractory Alloy HAADF-STEM + 5-Element STEM-EDS",
        hr_modality="HAADF-STEM",
        guide_modality="STEM-EDS (Ti, Mo, Ta, Cr, Al)",
        zenodo_record_id="21770333",
        doi="10.5281/zenodo.21770333",
        description=(
            "Nanoscale HAADF-STEM structural micrographs paired with Ti, Mo, Ta, "
            "Cr, and Al STEM-EDS elemental TIFF maps (Radi et al., 2026)."
        ),
    ),
    "cuzn_stem_eels": OpenCorrelativeDataset(
        key="cuzn_stem_eels",
        title="Calibrated Cu/Zn Oxide HAADF-STEM + Core-Loss EELS Spectrum Image",
        hr_modality="HAADF-STEM",
        guide_modality="Core-Loss EELS (Cu-L2,3, Zn-L2,3)",
        zenodo_record_id="21632101",
        doi="10.5281/zenodo.21632101",
        description=(
            "Simultaneously acquired HAADF survey image and Cu/Zn L-edge EELS "
            "spectrum image on carbon nanotubes (HyperSpy / eXSpy archive)."
        ),
    ),
    "isis_imat_bragg_edge": OpenCorrelativeDataset(
        key="isis_imat_bragg_edge",
        title="ISIS IMAT Multi-Metal Powder Phantom ToF Neutron Bragg-Edge CT",
        hr_modality="White-Beam Neutron Attenuation CT",
        guide_modality=(
            "Wavelength-Resolved ToF Bragg-Edge Phase Maps (Al, Fe, Cu, Ni, Zn)"
        ),
        zenodo_record_id="4646694",
        doi="10.5281/zenodo.4646694",
        description=(
            "Time-of-flight neutron tomography of high-purity Al, Fe, Cu, Ni, and "
            "Zn cylinders at ISIS IMAT (Ametova et al., J. Phys. D 2021)."
        ),
    ),
}


def get_repo_root() -> Path:
    """Return the root directory of the repository."""
    return Path(__file__).resolve().parent.parent


def get_sample_dataset_path(name: str = "Zn13X") -> Path:
    """Locate bundled sample slices (``Zn13X`` or ``MOR-GIS``) in ``data/samples``."""
    samples_root = get_repo_root() / "data" / "samples"
    candidate = samples_root / str(name)
    if not candidate.exists():
        available = [p.name for p in samples_root.glob("*")]
        raise FileNotFoundError(
            f"Sample dataset '{name}' not found at {candidate}. Available: {available}"
        )
    return candidate


def list_open_datasets() -> dict[str, OpenCorrelativeDataset]:
    """Return the dictionary of registered open correlative benchmark datasets."""
    return dict(OPEN_DATASETS)


def fetch_open_dataset(
    dataset_key: str,
    dest_root: str | Path | None = None,
    filenames: Iterable[str] | None = None,
) -> list[Path]:
    """Download files for a registered Zenodo dataset with MD5 verification."""
    if dataset_key not in OPEN_DATASETS:
        raise KeyError(
            f"Unknown dataset '{dataset_key}'. Choose from {list(OPEN_DATASETS)}"
        )
    meta = OPEN_DATASETS[dataset_key]
    target_dir = (
        Path(dest_root) / dataset_key
        if dest_root is not None
        else get_repo_root() / "data" / "downloads" / dataset_key
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    api_url = f"https://zenodo.org/api/records/{meta.zenodo_record_id}"
    with urllib.request.urlopen(api_url, timeout=30) as resp:
        record = json.loads(resp.read().decode("utf-8"))

    wanted = set(filenames) if filenames is not None else None
    downloaded: list[Path] = []
    for f_entry in record.get("files", []):
        fname = f_entry["key"]
        if wanted is not None and fname not in wanted:
            continue
        out_path = target_dir / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        checksum = f_entry.get("checksum", "")
        dl_url = f_entry["links"]["self"]

        if not out_path.exists():
            urllib.request.urlretrieve(dl_url, out_path)

        if checksum.startswith("md5:"):
            expected_md5 = checksum.split(":", 1)[1]
            actual_md5 = hashlib.md5(out_path.read_bytes()).hexdigest()
            if actual_md5 != expected_md5:
                out_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"MD5 mismatch for {fname}: {expected_md5} != {actual_md5}"
                )
        downloaded.append(out_path)
    return downloaded


def make_experimental_holdout(
    hr_image: np.ndarray,
    full_guide_map: np.ndarray,
    factor: int = 8,
    psf_sigma: float | None = None,
    poisson_counts: float | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a realistic coarse-guide input from a full-resolution experimental
    chemical/crystallographic map (e.g., STEM-EDS, SEM-EDS, EELS, Bragg-edge) so
    super-resolved predictions and uncertainty can be evaluated against real
    experimental ground truth.

    Args:
        hr_image: High-resolution structural image ``(H, W)`` or ``(C, H, W)``.
        full_guide_map: Full-resolution guide map ``(H, W)`` or ``(M, H, W)``.
        factor: Spatial binning / coarse step ratio (>= 1).
        psf_sigma: Gaussian interaction-volume / probe blur sigma in HR pixels
            (defaults to ``0.5 * factor``).
        poisson_counts: Optional mean photon/electron count per coarse pixel to
            simulate low-dose shot noise.
        seed: Random seed for reproducible Poisson noise.

    Returns:
        Tuple ``(hr_cropped, coarse_guide, gt_guide_cropped)`` where spatial
        dimensions are cropped to exact multiples of ``factor``.
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")

    guide = np.asarray(full_guide_map, dtype=np.float32)
    squeeze_guide = False
    if guide.ndim == 2:
        guide = guide[None, ...]
        squeeze_guide = True
    elif guide.ndim != 3:
        raise ValueError(f"Expected 2D or 3D full_guide_map, got shape {guide.shape}")

    hr = np.asarray(hr_image, dtype=np.float32)
    h, w = guide.shape[-2], guide.shape[-1]
    crop_h = (h // factor) * factor
    crop_w = (w // factor) * factor
    if crop_h == 0 or crop_w == 0:
        raise ValueError(
            f"Guide shape {(h, w)} is smaller than downsample factor {factor}."
        )

    guide_crop = guide[:, :crop_h, :crop_w]
    hr_crop = hr[..., :crop_h, :crop_w]

    sigma = 0.5 * factor if psf_sigma is None and factor > 1 else (psf_sigma or 0.0)
    blurred = np.empty_like(guide_crop)
    for m in range(guide_crop.shape[0]):
        blurred[m] = (
            gaussian_filter(guide_crop[m], sigma=sigma, mode="nearest")
            if sigma > 0
            else guide_crop[m]
        )

    # Physical box-integration / average-pooling onto coarse dwell grid
    n_m = blurred.shape[0]
    h_lr, w_lr = crop_h // factor, crop_w // factor
    coarse = blurred.reshape(n_m, h_lr, factor, w_lr, factor).mean(axis=(2, 4))

    if poisson_counts is not None and poisson_counts > 0:
        rng = np.random.default_rng(seed)
        c_min = coarse.min(axis=(-2, -1), keepdims=True)
        c_max = coarse.max(axis=(-2, -1), keepdims=True)
        span = np.where((c_max - c_min) > 1e-8, c_max - c_min, 1.0)
        normed = np.clip((coarse - c_min) / span, 0.0, 1.0)
        noisy = rng.poisson(normed * poisson_counts).astype(np.float32) / float(
            poisson_counts
        )
        coarse = noisy * span + c_min

    coarse = coarse.astype(np.float32)
    gt_out = guide_crop[0] if squeeze_guide else guide_crop
    coarse_out = coarse[0] if squeeze_guide else coarse
    return hr_crop, coarse_out, gt_out
