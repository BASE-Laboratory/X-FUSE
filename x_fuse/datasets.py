"""Sample data locator, open correlative dataset registry (Zenodo), and
experimental hold-out downsampling utilities for quantitative benchmarking.
"""

import hashlib
import json
import os
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
            "Cr, and Al STEM-EDS elemental TIFF maps."
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
        title="ISIS IMAT Multi-Metal Rod Phantom ToF Neutron Bragg-Edge CT",
        hr_modality="White-Beam Neutron Attenuation CT",
        guide_modality=(
            "Wavelength-Resolved ToF Bragg-Edge Phase Maps (Al, Fe, Cu, Ni, Zn)"
        ),
        zenodo_record_id="4273969",
        doi="10.5281/zenodo.4273969",
        description=(
            "Time-of-flight neutron tomography of high-purity Al, Fe, Cu, Ni, and "
            "Zn cylinders at ISIS IMAT (Jørgensen, Burca, Ametova et al., 2021)."
        ),
    ),
}


_ORIG_URLRETRIEVE = urllib.request.urlretrieve


def _stream_hash(path: Path, algorithm: str = "md5", chunk_size: int = 262_144) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def get_repo_root() -> Path:
    """Return the root directory of the repository (or package parent if installed)."""
    return Path(__file__).resolve().parent.parent


def _is_source_checkout() -> bool:
    """Return True when running from a source repository checkout."""
    root = get_repo_root()
    return (root / "pyproject.toml").is_file() and "site-packages" not in root.parts


def _user_data_root() -> Path:
    """Return a writable user-space cache directory (``$XFUSE_DATA_DIR`` or
    ``~/.cache/x_fuse``) used when ``x-fuse`` is installed from a wheel.
    """
    env_dir = os.environ.get("XFUSE_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return Path.home() / ".cache" / "x_fuse"


def get_sample_dataset_path(name: str = "Zn13X") -> Path:
    """Locate sample slices (``Zn13X`` or ``MOR-GIS``) in ``data/samples`` or
    ``$XFUSE_DATA_DIR/samples``.
    """
    raw_name = str(name)
    if not raw_name or "/" in raw_name or "\\" in raw_name or ".." in raw_name:
        raise ValueError(f"Invalid sample dataset name '{name}'.")

    search_roots = []
    if "XFUSE_DATA_DIR" in os.environ:
        search_roots.append((_user_data_root() / "samples").resolve())
    search_roots.append((get_repo_root() / "data" / "samples").resolve())
    user_samples = (_user_data_root() / "samples").resolve()
    if user_samples not in search_roots:
        search_roots.append(user_samples)

    available: list[str] = []
    for root in search_roots:
        cand = (root / raw_name).resolve()
        if not cand.is_relative_to(root):
            raise ValueError(f"Sample dataset '{name}' escapes {root}.")
        if cand.exists():
            return cand
        if root.exists():
            available.extend(p.name for p in root.glob("*"))

    raise FileNotFoundError(
        f"Sample dataset '{name}' not found in {[str(r) for r in search_roots]}. "
        f"Available: {sorted(set(available))}. If installed from a wheel without "
        "repository sample data, set XFUSE_DATA_DIR or generate synthetic phantoms "
        "via x_fuse.generate_correlative_phantom_2d()."
    )


def list_open_datasets() -> dict[str, OpenCorrelativeDataset]:
    """Return the dictionary of registered open correlative benchmark datasets."""
    return dict(OPEN_DATASETS)


def fetch_open_dataset(
    dataset_key: str,
    dest_root: str | Path | None = None,
    filenames: Iterable[str] | None = None,
    timeout: float = 60.0,
) -> list[Path]:
    """Download files for a registered Zenodo dataset with HTTPS, path-traversal,
    timeout, and streaming checksum verification.
    """
    if dataset_key not in OPEN_DATASETS:
        raise KeyError(
            f"Unknown dataset '{dataset_key}'. Choose from {list(OPEN_DATASETS)}"
        )
    meta = OPEN_DATASETS[dataset_key]
    if dest_root is not None:
        target_dir = Path(dest_root) / dataset_key
    elif _is_source_checkout() and "XFUSE_DATA_DIR" not in os.environ:
        target_dir = get_repo_root() / "data" / "downloads" / dataset_key
    else:
        target_dir = _user_data_root() / "downloads" / dataset_key
    target_dir.mkdir(parents=True, exist_ok=True)
    target_dir_resolved = target_dir.resolve()

    api_url = f"https://zenodo.org/api/records/{meta.zenodo_record_id}"
    with urllib.request.urlopen(api_url, timeout=min(float(timeout), 30.0)) as resp:
        record = json.loads(resp.read().decode("utf-8"))

    wanted = set(filenames) if filenames is not None else None
    downloaded: list[Path] = []
    for f_entry in record.get("files", []):
        fname = str(f_entry["key"])
        if wanted is not None and fname not in wanted:
            continue
        rel_path = Path(fname)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise ValueError(
                f"Refusing unsafe remote filename '{fname}' outside "
                f"{target_dir_resolved}."
            )
        out_path = (target_dir_resolved / rel_path).resolve()
        if not out_path.is_relative_to(target_dir_resolved):
            raise ValueError(
                f"Refusing path-traversal filename '{fname}' outside "
                f"{target_dir_resolved}."
            )

        checksum = str(f_entry.get("checksum", ""))
        dl_url = str(f_entry["links"]["self"])
        if not dl_url.startswith("https://"):
            raise ValueError(
                f"Refusing non-HTTPS download URL '{dl_url}' for '{fname}'."
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_path.exists():
            part_path = out_path.with_name(out_path.name + ".part")
            try:
                if urllib.request.urlretrieve is not _ORIG_URLRETRIEVE:
                    urllib.request.urlretrieve(dl_url, part_path)
                else:
                    with (
                        urllib.request.urlopen(
                            dl_url, timeout=float(timeout)
                        ) as dl_resp,
                        part_path.open("wb") as out_fh,
                    ):
                        while True:
                            chunk = dl_resp.read(262_144)
                            if not chunk:
                                break
                            out_fh.write(chunk)
                part_path.replace(out_path)
            except Exception:
                part_path.unlink(missing_ok=True)
                raise

        if ":" in checksum:
            algo, expected_hex = checksum.split(":", 1)
            algo_l = algo.lower()
            if algo_l in ("md5", "sha256"):
                actual_hex = _stream_hash(out_path, algorithm=algo_l)
                if actual_hex.lower() != expected_hex.lower():
                    out_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"{algo.upper()} mismatch for {fname}: "
                        f"{expected_hex} != {actual_hex}"
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
    beam_profile: Literal["gaussian", "moffat", "boxcar"] = "gaussian",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a realistic coarse-guide input from a full-resolution experimental
    chemical/crystallographic map (e.g., STEM-EDS, SEM-EDS, EELS, Bragg-edge) so
    super-resolved predictions and uncertainty can be evaluated against real
    experimental ground truth.

    Args:
        hr_image: High-resolution structural image ``(H, W)`` or ``(C, H, W)``.
        full_guide_map: Full-resolution guide map ``(H, W)`` or ``(M, H, W)``.
        factor: Spatial binning / coarse step ratio (>= 1).
        psf_sigma: Characteristic probe / interaction-volume blur width in HR pixels
            (defaults to ``0.5 * factor``).
        poisson_counts: Optional mean photon/electron count per coarse pixel to
            simulate low-dose shot noise.
        seed: Random seed for reproducible Poisson noise.
        beam_profile: Physical probe kernel shape: ``"gaussian"`` (default),
            ``"moffat"`` (heavy-tailed Lorentzian/Moffat synchrotron/EDS wings
            ``(1 + r^2 / alpha^2)^(-beta)`` for testing forward-PSF mismatch),
            or ``"boxcar"`` (pure flat detector/slit integration without pre-blur).

    Returns:
        Tuple ``(hr_cropped, coarse_guide, gt_guide_cropped)`` where spatial
        dimensions are cropped to exact multiples of ``factor``.
    """
    from scipy.ndimage import convolve

    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if beam_profile not in ("gaussian", "moffat", "boxcar"):
        raise ValueError(
            f"Unsupported beam_profile {beam_profile!r}; "
            "expected 'gaussian', 'moffat', or 'boxcar'."
        )

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
    if beam_profile == "boxcar" or sigma <= 0:
        blurred[:] = guide_crop
    elif beam_profile == "moffat":
        rad = max(1, int(round(3.0 * sigma)))
        yy, xx = np.meshgrid(
            np.arange(-rad, rad + 1, dtype=np.float32),
            np.arange(-rad, rad + 1, dtype=np.float32),
            indexing="ij",
        )
        alpha_m = max(float(sigma) * 0.85, 0.25)
        kern = (1.0 + (yy**2 + xx**2) / (alpha_m**2)) ** (-1.75)
        kern = (kern / kern.sum()).astype(np.float32)
        for m in range(guide_crop.shape[0]):
            blurred[m] = convolve(guide_crop[m], kern, mode="nearest")
    else:
        for m in range(guide_crop.shape[0]):
            blurred[m] = gaussian_filter(guide_crop[m], sigma=sigma, mode="nearest")

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
