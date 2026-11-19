"""Standardized dataset adapters for bundled synchrotron samples (Zn13X, MOR-GIS)
and registered open-access Zenodo correlative microscopy datasets.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from x_fuse.datasets import (
    OPEN_DATASETS,
    fetch_open_dataset,
    get_sample_dataset_path,
)
from x_fuse.io import load_microscopy_file
from x_fuse.loaders import load_correlative_tiff_pair, load_image_array


@dataclass
class CorrelativeSample:
    """Normalized container for a 2D correlative benchmark pair."""

    key: str
    hr_image: np.ndarray  # (H, W) float32
    guide_map: np.ndarray  # (M, H_g, W_g) float32
    channel_names: list[str]
    is_full_res_ground_truth: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def load_bundled_zn13x_pairs() -> list[CorrelativeSample]:
    """Load both bundled Diamond DIAD synchrotron XCT + XRD-CT slices (4 & 9)."""
    zn_dir = get_sample_dataset_path("Zn13X")
    samples: list[CorrelativeSample] = []
    for slice_id in (4, 9):
        hr_file = zn_dir / f"high_{slice_id}.tiff"
        lr_file = zn_dir / f"low_{slice_id}.tiff"
        if hr_file.exists() and lr_file.exists():
            hr_img, guide, _ = load_correlative_tiff_pair(hr_file, [lr_file])
            samples.append(
                CorrelativeSample(
                    key=f"zn13x_slice_{slice_id}",
                    hr_image=hr_img,
                    guide_map=guide,
                    channel_names=["Zn_phase"],
                    is_full_res_ground_truth=False,
                    metadata={
                        "hr_modality": "Synchrotron XCT",
                        "guide_modality": "Synchrotron XRD-CT",
                        "slice_id": slice_id,
                    },
                )
            )
    return samples


def load_bundled_morgis_samples() -> list[CorrelativeSample]:
    """Load bundled MOR-GIS synchrotron XCT zeolite micrographs paired with
    semi-synthetic multi-phase attenuation percentile maps (for tiling and
    attenuation-consistency smoke tests rather than independent chemical holdout).
    """
    mg_dir = get_sample_dataset_path("MOR-GIS")
    samples: list[CorrelativeSample] = []
    for png_path in sorted(mg_dir.glob("*.png")):
        arr = load_image_array(png_path)
        if arr.ndim == 3:
            arr = arr[..., :3].mean(axis=-1).astype(np.float32)
        p35, p75 = np.percentile(arr, (35.0, 75.0))
        phase_mid = ((arr >= p35) & (arr < p75)).astype(np.float32)
        phase_bright = (arr >= p75).astype(np.float32)
        guide_gt = np.stack([phase_mid, phase_bright], axis=0)
        samples.append(
            CorrelativeSample(
                key=f"morgis_{png_path.stem}",
                hr_image=arr,
                guide_map=guide_gt,
                channel_names=["zeolite_matrix", "dense_framework"],
                is_full_res_ground_truth=True,
                metadata={
                    "hr_modality": "Synchrotron XCT",
                    "guide_modality": "Semi-synthetic XCT percentile reference",
                    "semi_synthetic": True,
                    "source_file": png_path.name,
                },
            )
        )
    return samples


def _safe_extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """Extract ``zip_path`` into ``extract_dir`` while rejecting Zip-Slip path
    traversal (``..`` or absolute paths) and symbolic link entries.
    """
    dest_root = extract_dir.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ValueError(
                    f"Refusing to extract symlink archive entry '{info.filename}'."
                )
            member_target = (dest_root / info.filename).resolve()
            if not member_target.is_relative_to(dest_root):
                raise ValueError(
                    f"Unsafe Zip-Slip path '{info.filename}' escapes {dest_root}."
                )
        zf.extractall(dest_root)


def _ensure_extracted(paths: list[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("No files found in dataset directory.")
    root = paths[0].parent
    for p in paths:
        if p.suffix.lower() == ".zip":
            extract_dir = root / p.stem
            if not extract_dir.exists():
                _safe_extract_zip(p, extract_dir)
    return root


def _discover_hr_and_guides_generic(
    folder: Path,
    hr_keywords: tuple[str, ...],
    elem_keywords: tuple[str, ...],
    subpanel: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    search_root = folder
    if subpanel is not None:
        sub_matches = [
            d
            for d in folder.rglob("*")
            if d.is_dir() and d.name.lower() == subpanel.lower()
        ]
        if sub_matches:
            search_root = sub_matches[0]

    all_files = sorted(
        [
            f
            for f in search_root.rglob("*")
            if f.is_file()
            and f.suffix.lower()
            in {
                ".tif",
                ".tiff",
                ".png",
                ".npy",
                ".txt",
                ".csv",
                ".hspy",
                ".emd",
                ".dm4",
            }
        ]
    )
    if not all_files:
        pdfs = list(search_root.rglob("*.pdf"))
        if pdfs:
            raise RuntimeError(
                f"Dataset folder {search_root} contains only PDF document(s) "
                f"({[p.name for p in pdfs[:3]]}) rather than raw microscopy arrays."
            )
    hr_candidates = [
        f
        for f in all_files
        if any(kw in f.stem.lower() for kw in hr_keywords)
        and "diffraction" not in f.stem.lower()
        and "saed" not in f.stem.lower()
    ]
    if not hr_candidates:
        raise FileNotFoundError(
            f"Could not locate structural HR file matching {hr_keywords} "
            f"under {search_root}."
        )
    # Prefer Fig3b (aged 10Al with strong A2/B2 contrast) when multiple panels exist
    if subpanel is None and len(hr_candidates) > 1:
        fig3b = [f for f in hr_candidates if "fig3b" in str(f).lower()]
        hr_file = fig3b[0] if fig3b else hr_candidates[0]
    else:
        hr_file = hr_candidates[0]

    if hr_file.suffix.lower() in {".txt", ".csv"}:
        hr_arr = np.loadtxt(hr_file, dtype=np.float32)
    else:
        hr_arr, _ = load_microscopy_file(hr_file)
    if hr_arr.ndim == 3:
        hr_arr = hr_arr[..., 0] if hr_arr.shape[-1] <= 4 else hr_arr[0]

    sibling_files = [f for f in all_files if f.parent == hr_file.parent]
    pool_files = sibling_files if len(sibling_files) >= 2 else all_files

    guide_arrays: list[np.ndarray] = []
    names: list[str] = []
    for kw in elem_keywords:
        kw_low = kw.lower()
        exact = [
            f
            for f in pool_files
            if f != hr_file
            and (
                f.stem.lower().endswith(f"-{kw_low}")
                or f.stem.lower().endswith(f"_{kw_low}")
                or f.stem.lower() == kw_low
            )
        ]
        matches = exact or [
            f for f in pool_files if f != hr_file and kw_low in f.stem.lower()
        ]
        if matches:
            gf = matches[0]
            if gf.suffix.lower() in {".txt", ".csv"}:
                g_arr = np.loadtxt(gf, dtype=np.float32)
            else:
                g_arr, _ = load_microscopy_file(gf)
            if g_arr.ndim == 3:
                g_arr = (
                    g_arr[..., :3].max(axis=-1)
                    if g_arr.shape[-1] in (3, 4)
                    else g_arr.sum(axis=-1)
                )
            guide_arrays.append(g_arr.astype(np.float32))
            names.append(kw)

    if not guide_arrays:
        raise FileNotFoundError(
            f"Could not locate guide maps matching {elem_keywords} under {search_root}."
        )
    min_h = min([hr_arr.shape[0]] + [g.shape[0] for g in guide_arrays])
    min_w = min([hr_arr.shape[1]] + [g.shape[1] for g in guide_arrays])
    stack = np.stack([g[:min_h, :min_w] for g in guide_arrays], axis=0)
    return hr_arr[:min_h, :min_w], stack, names


def _parse_eels_energy_axis(meta: dict[str, Any], n_channels: int) -> np.ndarray:
    axes = meta.get("axes", [])
    if isinstance(axes, list):
        for ax in axes:
            if isinstance(ax, dict) and int(ax.get("size", -1)) == n_channels:
                off = float(ax.get("offset", 0.0))
                sc = float(ax.get("scale", 1.0))
                if sc > 0.0 and off > 0.0:
                    return (off + sc * np.arange(n_channels, dtype=np.float32)).astype(
                        np.float32
                    )
    return (700.0 + 8.0 * np.arange(n_channels, dtype=np.float32)).astype(np.float32)


def _strip_eels_powerlaw_window(
    si_hwe: np.ndarray,
    energies_ev: np.ndarray,
    pre_range: tuple[float, float],
    sig_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract an Egerton power-law continuum ``A * E^(-r)`` fit in log-log space
    on ``pre_range`` and return ``(integrated_edge_hw, net_cube_hwe_sig)``."""
    pre = (energies_ev >= pre_range[0]) & (energies_ev <= pre_range[1])
    sig = (energies_ev >= sig_range[0]) & (energies_ev <= sig_range[1])
    if int(pre.sum()) < 2 or int(sig.sum()) < 1:
        mid = si_hwe.shape[-1] // 2
        return si_hwe[..., :mid].mean(axis=-1), si_hwe[..., :mid]
    lx = np.log(energies_ev[pre])
    lxm = float(lx.mean())
    ly = np.log(np.clip(si_hwe[..., pre], 1.0, None))
    lym = ly.mean(axis=-1, keepdims=True)
    denom = float(((lx - lxm) ** 2).sum())
    slope = (
        ((lx - lxm) * (ly - lym)).sum(axis=-1, keepdims=True) / max(denom, 1e-8)
    ).clip(-12.0, 0.0)
    intercept = lym - slope * lxm
    bg_sig = np.exp(intercept + slope * np.log(energies_ev[sig])[None, None, :])
    net_sig = np.clip(si_hwe[..., sig] - bg_sig, 0.0, None).astype(np.float32)
    return net_sig.sum(axis=-1).astype(np.float32), net_sig


def load_zenodo_correlative_sample(
    dataset_key: str,
    dest_root: str | Path | None = None,
    auto_download: bool = False,
    *,
    subpanel: str | None = None,
    crop_clean_roi: bool = True,
) -> CorrelativeSample:
    """Parse a downloaded (or on-demand fetched) Zenodo dataset into a
    standardized ``CorrelativeSample`` ready for ``make_experimental_holdout``.
    """
    if dataset_key not in OPEN_DATASETS:
        raise KeyError(
            f"Unknown dataset_key '{dataset_key}'. Available: {list(OPEN_DATASETS)}"
        )
    meta = OPEN_DATASETS[dataset_key]

    if auto_download:
        file_paths = fetch_open_dataset(dataset_key, dest_root=dest_root)
        folder = _ensure_extracted(file_paths)
    else:
        base = (
            Path(dest_root) / dataset_key
            if dest_root is not None
            else Path(__file__).resolve().parent.parent
            / "data"
            / "downloads"
            / dataset_key
        )
        if not base.exists():
            raise FileNotFoundError(
                f"Dataset '{dataset_key}' is not cached at {base}. "
                "Pass auto_download=True or call fetch_open_dataset() first."
            )
        folder = _ensure_extracted(list(base.iterdir()))

    if dataset_key == "edxia_cement_sem_eds":
        hr, guides, names = _discover_hr_and_guides_generic(
            folder,
            hr_keywords=("bse", "sem"),
            elem_keywords=("Ca", "Si", "Al", "Fe", "Mg", "S"),
            subpanel=subpanel,
        )
    elif dataset_key == "ti_rcca_stem_eds":
        hr, guides, names = _discover_hr_and_guides_generic(
            folder,
            hr_keywords=("haadf", "stem", "adf"),
            elem_keywords=("Ti", "Mo", "Ta", "Cr", "Al"),
            subpanel=subpanel,
        )
        if crop_clean_roi and hr.shape[0] >= 1024 and hr.shape[1] >= 1024:
            # Exclude top-left panel glyph and bottom-right burned-in scale bar
            hr = hr[96:864, 96:864]
            guides = guides[:, 96:864, 96:864]
    elif dataset_key == "cuzn_stem_eels":
        hspy_files = sorted(folder.rglob("*.hspy")) + sorted(folder.rglob("*.dm4"))
        if len(hspy_files) >= 2:
            hr_cand = [
                f for f in hspy_files if "haadf" in f.stem.lower()
            ] or hspy_files[:1]
            eels_cand = [f for f in hspy_files if f != hr_cand[0]]
            hr, _ = load_microscopy_file(hr_cand[0])
            si, si_meta = load_microscopy_file(eels_cand[0])
            if si.ndim == 3:
                energies = _parse_eels_energy_axis(si_meta, int(si.shape[-1]))
                cu_map, _ = _strip_eels_powerlaw_window(
                    si, energies, (780.0, 916.0), (932.0, 1004.0)
                )
                zn_map, _ = _strip_eels_powerlaw_window(
                    si, energies, (964.0, 1012.0), (1028.0, 1132.0)
                )
                guides = np.stack([cu_map, zn_map], axis=0)
                names = ["Cu_L23", "Zn_L23"]
            else:
                guides = si[None, ...]
                names = ["EELS_core"]
        else:
            hr, guides, names = _discover_hr_and_guides_generic(
                folder,
                hr_keywords=("haadf", "survey"),
                elem_keywords=("Cu", "Zn", "eels"),
                subpanel=subpanel,
            )
    elif dataset_key == "refractory_alloy_bse_eds_ebsd":
        h5_files = sorted(folder.rglob("*.h5")) + sorted(folder.rglob("*.hdf5"))
        if h5_files and not list(folder.rglob("*.tif*")):
            with h5py.File(h5_files[0], "r") as hf:
                arrays_2d: list[tuple[str, np.ndarray]] = []

                def _visit(name: str, obj: Any) -> None:
                    if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
                        arrays_2d.append((name, np.array(obj, dtype=np.float32)))

                hf.visititems(_visit)
            if len(arrays_2d) < 2:
                raise ValueError(f"Insufficient 2D maps in HDF5 {h5_files[0]}.")
            arrays_2d.sort(key=lambda item: item[1].size, reverse=True)
            hr = arrays_2d[0][1]
            g_list = [a for _, a in arrays_2d[1:5] if a.shape == arrays_2d[1][1].shape]
            names = [
                Path(n).name
                for n, a in arrays_2d[1:5]
                if a.shape == arrays_2d[1][1].shape
            ]
            guides = np.stack(g_list, axis=0)
        else:
            hr, guides, names = _discover_hr_and_guides_generic(
                folder,
                hr_keywords=("bse", "sem"),
                elem_keywords=("Al", "Ti", "Nb", "Zr", "Ta", "phase"),
                subpanel=subpanel,
            )
    else:  # isis_imat_bragg_edge
        hr, guides, names = _discover_hr_and_guides_generic(
            folder,
            hr_keywords=("white", "open", "attenuation", "recon", "sum"),
            elem_keywords=("Al", "Fe", "Cu", "Ni", "Zn", "bragg"),
            subpanel=subpanel,
        )

    sample_key = f"{dataset_key}_{subpanel.lower()}" if subpanel else dataset_key
    return CorrelativeSample(
        key=sample_key,
        hr_image=hr.astype(np.float32),
        guide_map=guides.astype(np.float32),
        channel_names=names,
        is_full_res_ground_truth=(hr.shape[-2:] == guides.shape[-2:]),
        metadata={
            "title": meta.title,
            "doi": meta.doi,
            "hr_modality": meta.hr_modality,
            "guide_modality": meta.guide_modality,
            "subpanel": subpanel,
        },
    )


def load_cuzn_eels_cube_sample(
    dest_root: str | Path | None = None,
    auto_download: bool = False,
    *,
    coarse_factor: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load the simultaneous HAADF-STEM survey image, pre-edge continuum-subtracted
    3D core-loss EELS spectral cube ``(E_core, H_c, W_c)`` across ``[924, 1180] eV``,
    and full-resolution reference ``(2, H, W)`` ``[Cu_L23, Zn_L23]`` maps from
    ``cuzn_stem_eels``.

    When ``coarse_factor > 1`` is specified, raw unstripped EELS counts are spatially
    binned by ``coarse_factor x coarse_factor`` *before* Egerton power-law pre-edge
    continuum subtraction so zero fine-grid continuum fit information leaks into the
    coarse spectral cube.
    """
    if auto_download:
        file_paths = fetch_open_dataset("cuzn_stem_eels", dest_root=dest_root)
        folder = _ensure_extracted(file_paths)
    else:
        base = (
            Path(dest_root) / "cuzn_stem_eels"
            if dest_root is not None
            else Path(__file__).resolve().parent.parent
            / "data"
            / "downloads"
            / "cuzn_stem_eels"
        )
        folder = _ensure_extracted(list(base.iterdir()))

    hspy_files = sorted(folder.rglob("*.hspy")) + sorted(folder.rglob("*.dm4"))
    hr_cand = [f for f in hspy_files if "haadf" in f.stem.lower()] or hspy_files[:1]
    eels_cand = [f for f in hspy_files if f != hr_cand[0]]
    hr, _ = load_microscopy_file(hr_cand[0])
    si, si_meta = load_microscopy_file(eels_cand[0])
    energies = _parse_eels_energy_axis(si_meta, int(si.shape[-1]))

    div = max(4, int(coarse_factor)) if coarse_factor is not None else 4
    h4 = (hr.shape[0] // div) * div
    w4 = (hr.shape[1] // div) * div
    si_hr = si[:h4, :w4, :]

    cu_map, _ = _strip_eels_powerlaw_window(
        si_hr, energies, (780.0, 916.0), (932.0, 1004.0)
    )
    zn_map, _ = _strip_eels_powerlaw_window(
        si_hr, energies, (964.0, 1012.0), (1028.0, 1132.0)
    )
    if coarse_factor is not None and int(coarse_factor) > 1:
        f = int(coarse_factor)
        si_lr = (
            si_hr.astype(np.float64)
            .reshape(h4 // f, f, w4 // f, f, -1)
            .sum(axis=(1, 3))
        )
        _, net_cube_hwe = _strip_eels_powerlaw_window(
            si_lr, energies, (780.0, 916.0), (924.0, 1180.0)
        )
    else:
        _, net_cube_hwe = _strip_eels_powerlaw_window(
            si_hr, energies, (780.0, 916.0), (924.0, 1180.0)
        )
    hr_crop = hr[:h4, :w4].astype(np.float32)
    cube_ehw = np.transpose(net_cube_hwe, (2, 0, 1)).astype(np.float32)
    gt_maps = np.stack([cu_map, zn_map], axis=0).astype(np.float32)
    return hr_crop, cube_ehw, gt_maps, ["Cu_L23", "Zn_L23"]
