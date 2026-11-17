"""Data loading utilities for synchrotron tomography (XCT + XRD-CT), electron
microscopy (SEM/STEM + EDS/EELS), and generic correlative TIFF/HDF5 pairs.
"""

import os
from pathlib import Path
import re
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
from PIL import Image


def _natural_sort_key(name: str):
    return [
        int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)
    ]


def load_xct(xct_path: str, recon: str = "Tomopy", **kwargs) -> np.ndarray:
    """Load reconstructed 3D XCT volume from Diamond HDF5 file as ``(Y, X, Z)``."""
    assert str(xct_path).endswith(".h5"), "Expected a .h5 file for the xct data."

    recon_methods = {
        "Tomopy": "4-TomopyRecon-tomo",
        "Astra": "4-AstraReconGpu-tomo",
    }
    if recon not in recon_methods:
        raise KeyError(
            f"Unknown recon method '{recon}'. Choose from {list(recon_methods)}"
        )

    with h5py.File(xct_path, "r") as f:
        return np.transpose(np.array(f[recon_methods[recon]]["data"]), (1, 2, 0))


def load_xrdct_phase(
    phase_folder: str,
    phases: Sequence[str],
    shape: Sequence[int],
    crop: slice = slice(None),
    entry_names: Dict[str, str] | None = None,
    **kwargs,
) -> List[np.ndarray]:
    """Load stacked NeXus (``.nxs``) XRD-CT slices in deterministic natural order."""
    resolved_entries = entry_names if entry_names is not None else {"default": "data"}
    default_entry = resolved_entries.get("default", "data")

    all_entries = sorted(os.listdir(phase_folder), key=_natural_sort_key)
    xrdct_data: List[np.ndarray] = []

    for phase in phases:
        files = [f for f in all_entries if phase.lower() in f.lower()][crop]

        assert (
            len(files) > 0
        ), f"No files found for phase {phase} in folder {phase_folder}."
        assert any(
            f.endswith(".nxs") for f in files
        ), f"Expected .nxs files for phase {phase}, but found none."
        assert (
            len(files) == shape[0]
        ), f"Expected {shape[0]} files for phase {phase}, but found {len(files)}."

        entry_key = resolved_entries.get(phase, default_entry)
        xrdct_i = np.zeros(shape, dtype=np.float64)
        for idx in range(shape[0]):
            with h5py.File(os.path.join(phase_folder, files[idx]), "r") as file:
                data = np.array(file[entry_key])
                xrdct_i[idx] += data.squeeze()
        xrdct_data.append(xrdct_i)

    return xrdct_data


def load_diad_xct_zn13x(xct_path: str, **kwargs) -> np.ndarray:
    """Load and crop DIAD Zn13X Astra-reconstructed XCT volume."""
    return load_xct(xct_path, recon="Astra")[150:2000, 350:2200, 350:2200]


def load_diad_xrdct(
    phase_folder: str,
    phases: Sequence[str] | None = None,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Load DIAD XRD-CT NeXus phase stacks as ``{phase_name: (Z, Y, X)}``."""
    phase_list = list(phases) if phases is not None else ["Na", "Zn"]
    phase_arrays = load_xrdct_phase(
        phase_folder,
        shape=(21, 20, 20),
        phases=phase_list,
        crop=slice(5, -1),
        entry_names={
            "Na": "entry/peak at q~1.651",
            "Zn": "entry/peak at q~1.656",
            "default": "data",
        },
        **kwargs,
    )

    return {
        phase: np.flip(arr, axis=2).copy()
        for phase, arr in zip(phase_list, phase_arrays)
    }


def load_i13_xct(xct_path: str, **kwargs) -> np.ndarray:
    """Load Diamond I13 Tomopy-reconstructed XCT volume."""
    return load_xct(xct_path, recon="Tomopy")


# ── Modality-Agnostic Correlative Loaders ─────────────────────────────────────


def load_image_array(path: str | Path) -> np.ndarray:
    """Load a 2D/3D TIFF, PNG, or NPY micrograph/map preserving bit depth as float32."""
    p = Path(path)
    if p.suffix.lower() == ".npy":
        return np.load(p).astype(np.float32)
    with Image.open(p) as img:
        return np.array(img, dtype=np.float32)


def load_correlative_tiff_pair(
    hr_path: str | Path,
    guide_paths: Sequence[str | Path] | Dict[str, str | Path],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load a high-res structural image (e.g. BSE-SEM, HAADF-STEM, XCT slice)
    and one or more coarse physical guide maps (e.g. multi-element EDS, XRD-CT).

    Returns:
        ``(hr_image, guide_stack, channel_names)`` where ``guide_stack`` has
        shape ``(M, H_lr, W_lr)``.
    """
    hr_img = load_image_array(hr_path)
    if isinstance(guide_paths, dict):
        names = list(guide_paths.keys())
        paths = list(guide_paths.values())
    else:
        paths = list(guide_paths)
        names = [Path(p).stem for p in paths]

    maps = [load_image_array(p) for p in paths]
    shapes = {m.shape for m in maps}
    if len(shapes) > 1:
        raise ValueError(f"All guide maps must share spatial dimensions, got {shapes}.")
    guide_stack = np.stack(maps, axis=0).astype(np.float32)
    return hr_img, guide_stack, names


def load_eds_text_hypermap(
    folder: str | Path,
    prefix: str,
    elements: Sequence[str],
    bse_suffix: str = "BSE",
    delimiter: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load ``edxia``-style SEM-BSE + multi-element EDS ASCII/CSV maps
    (e.g. ``{prefix}_BSE.txt``, ``{prefix}_Ca.txt``, ``{prefix}_Si.txt``)."""
    root = Path(folder)
    bse_path = root / f"{prefix}_{bse_suffix}.txt"
    hr_bse = np.loadtxt(bse_path, delimiter=delimiter, dtype=np.float32)
    elem_maps = [
        np.loadtxt(root / f"{prefix}_{el}.txt", delimiter=delimiter, dtype=np.float32)
        for el in elements
    ]
    return hr_bse, np.stack(elem_maps, axis=0), list(elements)
