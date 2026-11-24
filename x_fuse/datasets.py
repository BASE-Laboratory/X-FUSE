"""Sample data locator, open correlative dataset registry (Zenodo), and
experimental hold-out downsampling utilities for quantitative benchmarking.
"""

import hashlib
import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from basecoat import ChecksumError, fetch_file
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
    default_filenames: tuple[str, ...] | None = None


OPEN_DATASETS: dict[str, OpenCorrelativeDataset] = {
    "edxia_cement_sem_eds": OpenCorrelativeDataset(
        key="edxia_cement_sem_eds",
        title="EPFL edxia Cementitious Microstructure SEM-BSE + EDS Hypermaps",
        hr_modality="SEM-BSE",
        guide_modality="Multi-Element EDS (Ca, Si, Al, Fe, Mg, S)",
        zenodo_record_id="3862958",
        doi="10.5281/zenodo.3862958",
        description=(
            "Aligned backscattered electron (BSE) micrographs and calibrated wt% "
            "multi-element EDS maps (LC3 cement) benchmarked against Joint "
            "Bilateral Filtering in Georget et al., Cem. Concr. Res. (2021)."
        ),
        default_filenames=("edxia_examples.zip",),
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
        default_filenames=("S3_nano.zip",),
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
        default_filenames=("Fig3_TEM_STEM.zip",),
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
        default_filenames=("CuZn_HAADF.hspy", "CuZn_EELS_mapping_tutorial.hspy"),
    ),
    "isis_imat_bragg_edge": OpenCorrelativeDataset(
        key="isis_imat_bragg_edge",
        title="ISIS IMAT Multi-Metal Rod Phantom White-Beam Neutron Sinograms",
        hr_modality="White-Beam Neutron Attenuation Sinograms / CT",
        guide_modality=(
            "Multi-Metal Rod Phantom (Al, Fe, Cu, Ni, Zn) Attenuation Projections"
        ),
        zenodo_record_id="4273969",
        doi="10.5281/zenodo.4273969",
        description=(
            "Golden-ratio white-beam neutron projections of high-purity Al, Fe, Cu, "
            "Ni, and Zn cylinders at ISIS IMAT (Jørgensen, Burca, Ametova et al., "
            "2021)."
        ),
        default_filenames=("imat_rod_phantom_white_beam.zip",),
    ),
    "diad_zn13x_xrdct": OpenCorrelativeDataset(
        key="diad_zn13x_xrdct",
        title="Diamond Light Source K11 DIAD Multi-Length Zn-13X Zeolite XRD-CT",
        hr_modality="Synchrotron XCT (0.54–2.6 µm)",
        guide_modality=(
            "Multi-Length XRD-CT (20 × 50 µm Volume + 4 × 25 µm ROI, Na-13X & Zn-13X)"
        ),
        zenodo_record_id="13329639",
        doi="10.5281/zenodo.13329639",
        description=(
            "Correlative multi-beamsize K11 DIAD XRD-CT reconstructions (20 scans at "
            "50 µm across the full ion-exchange reaction front + 4 paired ROI scans "
            "at 25 µm for real hardware validation) of partially Zn-exchanged Na-13X "
            "zeolite pellets."
        ),
        default_filenames=None,
    ),
    "diad_zn13x_powder_xrd": OpenCorrelativeDataset(
        key="diad_zn13x_powder_xrd",
        title="Zn-13X & Na-13X Zeolite High-Resolution Cu-Kα Powder XRD References",
        hr_modality="High-Resolution Cu-Kα Powder Diffraction",
        guide_modality="Reference Bragg Profiles (Na-13X, Zn-13X, Empty Capillary)",
        zenodo_record_id="13329670",
        doi="10.5281/zenodo.13329670",
        description=(
            "Rigaku high-resolution 2-Theta/Omega powder XRD patterns of pure Na-13X, "
            "Zn-exchanged 13X, and empty capillary background for phase dictionary "
            "and continuum calibration."
        ),
        default_filenames=(
            "13Xhighres_2-Theta_Omega.asc",
            "Zn13X15minhighres_2-Theta_Omega.asc",
            "Zn13Xhighres_2-Theta_Omega.asc",
            "empty_cap_2-Theta_Omega.asc",
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
    repo_root = get_repo_root()
    repo_samples = (repo_root / "data" / "samples").resolve()
    if "site-packages" not in repo_root.parts and (
        _is_source_checkout() or repo_samples.exists()
    ):
        search_roots.append(repo_samples)
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


def list_open_datasets(
    *, _warn_deprecated: bool = True
) -> dict[str, OpenCorrelativeDataset]:
    """Return the dictionary of registered open correlative benchmark datasets."""
    if _warn_deprecated:
        import warnings

        warnings.warn(
            "OPEN_DATASETS and list_open_datasets() are deprecated under ECOSYSTEM.md "
            "and scheduled to migrate to Layer 4 (MultimodalBenchmark) in a future "
            "release; prefer basecoat.fetch_file for Layer 2 workflows.",
            DeprecationWarning,
            stacklevel=2,
        )
    return dict(OPEN_DATASETS)


def fetch_open_dataset(
    dataset_key: str,
    dest_root: str | Path | None = None,
    filenames: Iterable[str] | None = None,
    timeout: float = 60.0,
    use_default_subset: bool = True,
    *,
    _warn_deprecated: bool = True,
) -> list[Path]:
    """Download files for a registered Zenodo dataset with HTTPS, path-traversal,
    timeout, and streaming checksum verification.

    When ``filenames is None`` and ``use_default_subset=True``, automatically
    restricts downloads to ``meta.default_filenames`` if those files are present
    in the Zenodo record (avoiding multi-gigabyte unrelated archives).
    """
    if _warn_deprecated:
        import warnings

        warnings.warn(
            "fetch_open_dataset() is deprecated under ECOSYSTEM.md and scheduled to "
            "migrate to Layer 4 (MultimodalBenchmark) in a future release; prefer "
            "basecoat.fetch_file for Layer 2 workflows.",
            DeprecationWarning,
            stacklevel=2,
        )
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

    expected_local = (
        list(filenames)
        if filenames is not None
        else (
            list(meta.default_filenames)
            if (use_default_subset and meta.default_filenames is not None)
            else []
        )
    )
    api_url = f"https://zenodo.org/api/records/{meta.zenodo_record_id}"
    try:
        with urllib.request.urlopen(api_url, timeout=min(float(timeout), 30.0)) as resp:
            record = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError:
        if expected_local:
            cached: list[Path] = []
            for fn in expected_local:
                rel_p = Path(fn)
                if rel_p.is_absolute() or ".." in rel_p.parts:
                    raise ValueError(f"Unsafe cached filename '{fn}'.") from None
                cand_p = (target_dir_resolved / rel_p).resolve()
                if cand_p.is_relative_to(target_dir_resolved) and cand_p.is_file():
                    cached.append(cand_p)
            if len(cached) == len(expected_local):
                return cached
        existing = sorted(
            p
            for p in target_dir_resolved.iterdir()
            if p.is_file() and not p.name.endswith(".part")
        )
        if existing and not expected_local:
            return existing
        raise

    record_files = record.get("files", [])
    record_keys = {str(fe.get("key", "")) for fe in record_files}
    if filenames is not None:
        wanted: set[str] | None = set(filenames)
    elif (
        use_default_subset
        and meta.default_filenames is not None
        and any(fn in record_keys for fn in meta.default_filenames)
    ):
        wanted = set(meta.default_filenames)
    else:
        wanted = None
    downloaded: list[Path] = []
    for f_entry in record_files:
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
        algo_l = ""
        expected_hex = ""
        if ":" in checksum:
            algo_raw, expected_hex = checksum.split(":", 1)
            algo_l = algo_raw.lower()

        if (
            urllib.request.urlretrieve is _ORIG_URLRETRIEVE
            and algo_l == "sha256"
            and len(expected_hex.strip()) == 64
        ):
            try:
                fetch_file(
                    dl_url,
                    sha256=expected_hex.strip(),
                    filename=out_path.name,
                    cache_dir=out_path.parent,
                    timeout_s=float(timeout),
                )
            except ChecksumError as exc:
                out_path.unlink(missing_ok=True)
                raise RuntimeError(str(exc)) from exc
            downloaded.append(out_path)
            continue

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

        if algo_l in ("md5", "sha256"):
            actual_hex = _stream_hash(out_path, algorithm=algo_l)
            if actual_hex.lower() != expected_hex.lower():
                out_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"{algo_l.upper()} mismatch for {fname}: "
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
    beam_profile: Literal["gaussian", "moffat", "core_halo", "boxcar"] = "gaussian",
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
            ``(1 + r^2 / alpha^2)^(-beta)``), ``"core_halo"`` (two-Gaussian primary
            core + wide backscattered SEM/EDS halo), or ``"boxcar"`` (pure flat
            detector/slit integration without pre-blur).

    Returns:
        Tuple ``(hr_cropped, coarse_guide, gt_guide_cropped)`` where spatial
        dimensions are cropped to exact multiples of ``factor``.
    """
    from scipy.ndimage import convolve

    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if beam_profile not in ("gaussian", "moffat", "core_halo", "boxcar"):
        raise ValueError(
            f"Unsupported beam_profile {beam_profile!r}; "
            "expected 'gaussian', 'moffat', 'core_halo', or 'boxcar'."
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
    elif beam_profile == "core_halo":
        for m in range(guide_crop.shape[0]):
            core = gaussian_filter(guide_crop[m], sigma=sigma, mode="nearest")
            halo = gaussian_filter(guide_crop[m], sigma=3.0 * sigma, mode="nearest")
            blurred[m] = 0.72 * core + 0.28 * halo
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
