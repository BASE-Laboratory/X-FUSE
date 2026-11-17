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
    multi-phase attenuation reference maps for high-resolution tiling tests.
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
                    "guide_modality": "Multi-phase reference",
                    "source_file": png_path.name,
                },
            )
        )
    return samples


def _ensure_extracted(paths: list[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("No files found in dataset directory.")
    root = paths[0].parent
    for p in paths:
        if p.suffix.lower() == ".zip":
            extract_dir = root / p.stem
            if not extract_dir.exists():
                with zipfile.ZipFile(p, "r") as zf:
                    zf.extractall(extract_dir)
    return root


def _discover_hr_and_guides_generic(
    folder: Path, hr_keywords: tuple[str, ...], elem_keywords: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    all_files = sorted(
        [
            f
            for f in folder.rglob("*")
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
    hr_candidates = [
        f for f in all_files if any(kw in f.stem.lower() for kw in hr_keywords)
    ]
    if not hr_candidates:
        raise FileNotFoundError(
            f"Could not locate structural HR file matching {hr_keywords} "
            f"under {folder}."
        )
    hr_file = hr_candidates[0]
    if hr_file.suffix.lower() in {".txt", ".csv"}:
        hr_arr = np.loadtxt(hr_file, dtype=np.float32)
    else:
        hr_arr, _ = load_microscopy_file(hr_file)
    if hr_arr.ndim == 3:
        hr_arr = hr_arr[..., 0] if hr_arr.shape[-1] <= 4 else hr_arr[0]

    guide_arrays: list[np.ndarray] = []
    names: list[str] = []
    for kw in elem_keywords:
        matches = [
            f for f in all_files if f != hr_file and kw.lower() in f.stem.lower()
        ]
        if matches:
            gf = matches[0]
            if gf.suffix.lower() in {".txt", ".csv"}:
                g_arr = np.loadtxt(gf, dtype=np.float32)
            else:
                g_arr, _ = load_microscopy_file(gf)
            if g_arr.ndim == 3:
                g_arr = g_arr.sum(axis=-1)
            guide_arrays.append(g_arr.astype(np.float32))
            names.append(kw)

    if not guide_arrays:
        raise FileNotFoundError(
            f"Could not locate guide maps matching {elem_keywords} under {folder}."
        )
    min_h = min(g.shape[0] for g in guide_arrays)
    min_w = min(g.shape[1] for g in guide_arrays)
    stack = np.stack([g[:min_h, :min_w] for g in guide_arrays], axis=0)
    return hr_arr, stack, names


def load_zenodo_correlative_sample(
    dataset_key: str,
    dest_root: str | Path | None = None,
    auto_download: bool = False,
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
        )
    elif dataset_key == "ti_rcca_stem_eds":
        hr, guides, names = _discover_hr_and_guides_generic(
            folder,
            hr_keywords=("haadf", "stem", "adf"),
            elem_keywords=("Ti", "Mo", "Ta", "Cr", "Al"),
        )
    elif dataset_key == "cuzn_stem_eels":
        hspy_files = sorted(folder.rglob("*.hspy")) + sorted(folder.rglob("*.dm4"))
        if len(hspy_files) >= 2:
            hr_cand = [
                f for f in hspy_files if "haadf" in f.stem.lower()
            ] or hspy_files[:1]
            eels_cand = [f for f in hspy_files if f != hr_cand[0]]
            hr, _ = load_microscopy_file(hr_cand[0])
            si, _ = load_microscopy_file(eels_cand[0])
            if si.ndim == 3:
                mid = si.shape[-1] // 2
                guides = np.stack(
                    [si[..., :mid].mean(axis=-1), si[..., mid:].mean(axis=-1)], axis=0
                )
                names = ["Cu_L", "Zn_L"]
            else:
                guides = si[None, ...]
                names = ["EELS_core"]
        else:
            hr, guides, names = _discover_hr_and_guides_generic(
                folder,
                hr_keywords=("haadf", "survey"),
                elem_keywords=("Cu", "Zn", "eels"),
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
            )
    else:  # isis_imat_bragg_edge
        hr, guides, names = _discover_hr_and_guides_generic(
            folder,
            hr_keywords=("white", "open", "attenuation", "recon", "sum"),
            elem_keywords=("Al", "Fe", "Cu", "Ni", "Zn", "bragg"),
        )

    return CorrelativeSample(
        key=dataset_key,
        hr_image=hr.astype(np.float32),
        guide_map=guides.astype(np.float32),
        channel_names=names,
        is_full_res_ground_truth=(hr.shape[-2:] == guides.shape[-2:]),
        metadata={
            "title": meta.title,
            "doi": meta.doi,
            "hr_modality": meta.hr_modality,
            "guide_modality": meta.guide_modality,
        },
    )
