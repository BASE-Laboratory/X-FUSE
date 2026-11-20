"""Data loading utilities for synchrotron tomography (XCT + XRD-CT), electron
microscopy (SEM/STEM + EDS/EELS), and generic correlative TIFF/HDF5 pairs.
"""

import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
from basecoat import (
    detrend_radial_cupping as _bc_detrend_radial_cupping,
    open_savu_h5,
    read_ome_zarr,
    read_volume,
    remove_rings_polar_conservative as _bc_remove_rings_polar,
)
from PIL import Image
from scipy.ndimage import rotate

_ORIG_H5PY_FILE = h5py.File

# Retained in v1.0.x for manuscript reproducibility; canonical home under ECOSYSTEM.md
# is BASE-Laboratory/MultimodalBenchmark (Layer 4).
DIAD_ZN13X_50UM_SCAN_IDS: tuple[int, ...] = tuple(
    list(range(43336, 43348)) + list(range(43393, 43401))
)
DIAD_ZN13X_25UM_SCAN_IDS: tuple[int, ...] = (43322, 43323, 43324, 43325)
PeakAggregationMode = Literal[
    "all_sum",
    "all_mean",
    "positive_sum",
    "median_norm_sum",
    "low_q",
    "high_q",
    "nearest_q",
    "single_peak",
    "legacy",
]

_PEAK_Q_RE = re.compile(r"peak\s+at\s+q~\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_OUTPUT_SCAN_RE = re.compile(r"^output_(\d+)_([a-z0-9]+)\.nxs$", re.IGNORECASE)


def _natural_sort_key(name: str) -> list[int | str]:
    return [
        int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)
    ]


def _indices_to_h5_slice(idx: np.ndarray) -> tuple[slice | np.ndarray, bool]:
    """Convert 1D target indices into an HDF5 positive-step hyperslab + reverse flag."""
    if idx.size == 0:
        return slice(0, 0, 1), False
    if idx.size == 1:
        i0 = int(idx[0])
        return slice(i0, i0 + 1, 1), False
    diffs = np.diff(idx)
    step = int(diffs[0])
    if step != 0 and np.all(diffs == step):
        if step > 0:
            return slice(int(idx[0]), int(idx[-1]) + 1, step), False
        pos_step = -step
        return slice(int(idx[-1]), int(idx[0]) + 1, pos_step), True
    order = np.argsort(idx)
    return idx[order], False


def load_xct_subvolume(
    xct_path: str | Path,
    recon: str = "Astra",
    zyx_slices: tuple[slice, slice, slice] | None = None,
    stride: int | tuple[int, int, int] = 1,
    flip_x: bool = False,
) -> np.ndarray:
    """Slice a Savu/Diamond HDF5 XCT reconstruction out-of-core on disk before
    transposing Savu ``(X, Z, Y)`` layout to canonical ``(Z, Y, X)``.

    Avoids loading full multi-tens-of-gigabyte ``float32`` volumes into RAM when
    extracting subvolumes or strided magnification ladders. When ``flip_x=True``,
    ``zyx_slices`` indexes the post-flip canonical ``(Z, Y, X)`` coordinate frame.
    """
    assert str(xct_path).lower().endswith((".h5", ".hdf5", ".nxs")), (
        "Expected an HDF5 (.h5/.hdf5/.nxs) file for the XCT data."
    )
    recon_methods = {
        "Tomopy": "4-TomopyRecon-tomo",
        "Astra": "4-AstraReconGpu-tomo",
    }
    if recon not in recon_methods:
        raise KeyError(
            f"Unknown recon method '{recon}'. Choose from {list(recon_methods)}"
        )

    if isinstance(stride, int):
        sz = sy = sx = max(1, int(stride))
    else:
        sz, sy, sx = (
            max(1, int(stride[0])),
            max(1, int(stride[1])),
            max(1, int(stride[2])),
        )

    z_sl, y_sl, x_sl = (
        zyx_slices
        if zyx_slices is not None
        else (slice(None), slice(None), slice(None))
    )

    if (
        h5py.File is _ORIG_H5PY_FILE
        and (sz, sy, sx) == (1, 1, 1)
        and not flip_x
        and all((sl.step is None or sl.step == 1) for sl in (z_sl, y_sl, x_sl))
    ):
        ds_path = f"{recon_methods[recon]}/data"
        with open_savu_h5(xct_path, dataset_path=ds_path, savu_xzy=True) as lv:
            z0, z1, _ = z_sl.indices(lv.shape[0])
            y0, y1, _ = y_sl.indices(lv.shape[1])
            x0, x1, _ = x_sl.indices(lv.shape[2])
            if z1 <= z0 or y1 <= y0 or x1 <= x0:
                return np.zeros(
                    (max(0, z1 - z0), max(0, y1 - y0), max(0, x1 - x0)),
                    dtype=lv.dtype,
                )
            return lv.read_region(
                (slice(z0, z1), slice(y0, y1), slice(x0, x1))
            ).data.copy()

    with h5py.File(xct_path, "r") as f:
        ds = f[recon_methods[recon]]["data"]
        if zyx_slices is None and (sz, sy, sx) == (1, 1, 1):
            sub_xzy = np.asarray(ds)
            out_zyx = np.transpose(sub_xzy, (1, 2, 0))
            return np.flip(out_zyx, axis=2).copy() if flip_x else out_zyx

        x_full, z_full, y_full = (int(s) for s in ds.shape)
        z_idx = np.arange(z_full)[z_sl][::sz]
        y_idx = np.arange(y_full)[y_sl][::sy]
        x_post = np.arange(x_full)[x_sl][::sx]
        x_disk = (x_full - 1 - x_post) if flip_x else x_post

        if z_idx.size == 0 or y_idx.size == 0 or x_disk.size == 0:
            return np.zeros((z_idx.size, y_idx.size, x_disk.size), dtype=ds.dtype)

        h5_x, rev_x = _indices_to_h5_slice(x_disk)
        h5_z, rev_z = _indices_to_h5_slice(z_idx)
        h5_y, rev_y = _indices_to_h5_slice(y_idx)
        sub_xzy = np.asarray(ds[h5_x, h5_z, h5_y])

    if rev_x:
        sub_xzy = sub_xzy[::-1, :, :]
    if rev_z:
        sub_xzy = sub_xzy[:, ::-1, :]
    if rev_y:
        sub_xzy = sub_xzy[:, :, ::-1]
    return np.transpose(sub_xzy, (1, 2, 0)).copy()


def load_xct(xct_path: str | Path, recon: str = "Tomopy", **kwargs: Any) -> np.ndarray:
    """Load reconstructed 3D XCT volume from Diamond Savu HDF5 file.

    Supports optional out-of-core ``zyx_slices``, ``stride``, and ``flip_x``
    keyword arguments via :func:`load_xct_subvolume`.
    """
    assert str(xct_path).lower().endswith((".h5", ".hdf5")), (
        "Expected a .h5/.hdf5 file for the xct data."
    )
    if any(k in kwargs for k in ("zyx_slices", "stride", "flip_x")):
        return load_xct_subvolume(
            xct_path,
            recon=recon,
            zyx_slices=kwargs.get("zyx_slices"),
            stride=kwargs.get("stride", 1),
            flip_x=bool(kwargs.get("flip_x", False)),
        )

    recon_methods = {
        "Tomopy": "4-TomopyRecon-tomo",
        "Astra": "4-AstraReconGpu-tomo",
    }
    if recon not in recon_methods:
        raise KeyError(
            f"Unknown recon method '{recon}'. Choose from {list(recon_methods)}"
        )

    if h5py.File is _ORIG_H5PY_FILE:
        ds_path = f"{recon_methods[recon]}/data"
        with open_savu_h5(xct_path, dataset_path=ds_path, savu_xzy=True) as lv:
            return lv.read_region((slice(None), slice(None), slice(None))).data.copy()

    with h5py.File(xct_path, "r") as f:
        return np.transpose(np.array(f[recon_methods[recon]]["data"]), (1, 2, 0))


def load_xrdct_peak_maps(
    path: str | Path,
    flip_x: bool = False,
) -> dict[float, np.ndarray]:
    """Read all fitted Bragg reflection maps (``peak at q~<val>``) from a DAWN
    NeXus ``.nxs`` slice file as ``{q_inv_angstrom: map_2d_float32}``.
    """
    peaks: dict[float, np.ndarray] = {}
    with h5py.File(path, "r") as hf:
        containers: list[Any] = []
        if "entry" in hf and hasattr(hf["entry"], "keys"):
            containers.append(hf["entry"])
        containers.append(hf)
        for grp in containers:
            for key in grp.keys():
                m = _PEAK_Q_RE.search(str(key))
                if m is None:
                    continue
                q_val = round(float(m.group(1)), 6)
                if q_val in peaks:
                    continue
                obj = grp[key]
                if isinstance(obj, h5py.Group):
                    sig_key = str(obj.attrs.get("signal", "data"))
                    if sig_key in obj:
                        obj = obj[sig_key]
                    elif "data" in obj:
                        obj = obj["data"]
                    else:
                        continue
                arr = np.asarray(obj, dtype=np.float32).squeeze()
                if arr.ndim != 2:
                    continue
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                if flip_x:
                    arr = np.fliplr(arr).copy()
                peaks[q_val] = arr
    return dict(sorted(peaks.items(), key=lambda kv: kv[0]))


def aggregate_peak_maps(
    peaks: dict[float, np.ndarray],
    mode: PeakAggregationMode = "all_sum",
    target_q: float | None = None,
    q_tol: float = 0.02,
) -> np.ndarray:
    """Aggregate a dictionary of ``{q: map_2d}`` Bragg reflection maps into a
    single 2D phase intensity map.

    - ``"all_sum"`` / ``"all_mean"``: Sums or averages all fitted reflections for
      maximum SNR and reduced overlapping-doublet crosstalk.
    - ``"positive_sum"``: Clamps negative background baseline excursions before summing.
    - ``"median_norm_sum"``: Normalizes each reflection by its P95 positive scale
      and takes the pixelwise median across reflections.
    - ``"low_q"`` / ``"high_q"``: Sums the lower-``q`` or upper-``q`` half of
      reflections (for q-dependent attenuation / strain split audits).
    - ``"nearest_q"`` / ``"single_peak"``: Selects the single reflection closest
      to ``target_q`` within ``q_tol`` (resolving DAWN bin jitter).
    """
    if not peaks:
        raise ValueError("Cannot aggregate an empty peak map dictionary.")

    ordered = sorted(peaks.items(), key=lambda kv: kv[0])
    if mode in ("nearest_q", "single_peak", "legacy"):
        if target_q is None:
            if mode == "single_peak":
                raise ValueError("target_q is required when mode='single_peak'.")
            return ordered[0][1].astype(np.float32, copy=True)
        best_q = min(peaks.keys(), key=lambda q: abs(q - float(target_q)))
        if abs(best_q - float(target_q)) > float(q_tol):
            raise KeyError(
                f"No reflection within q_tol={q_tol} of target_q={target_q}; "
                f"available q={sorted(peaks.keys())}."
            )
        return peaks[best_q].astype(np.float32, copy=True)

    stacked = np.stack([np.asarray(v, dtype=np.float32) for _, v in ordered], axis=0)
    if mode == "all_sum":
        return np.asarray(np.sum(stacked, axis=0, dtype=np.float32))
    if mode == "all_mean":
        return np.asarray(np.mean(stacked, axis=0, dtype=np.float32))
    if mode == "positive_sum":
        return np.asarray(np.sum(np.maximum(stacked, 0.0), axis=0, dtype=np.float32))
    if mode == "low_q":
        half = max(1, len(ordered) // 2)
        return np.asarray(np.sum(stacked[:half], axis=0, dtype=np.float32))
    if mode == "high_q":
        half = max(1, len(ordered) // 2) if len(ordered) >= 2 else 0
        return np.asarray(np.sum(stacked[half:], axis=0, dtype=np.float32))
    if mode == "median_norm_sum":
        normed = []
        for a in stacked:
            pos = a[a > 0]
            if pos.size >= 4:
                scale = float(np.percentile(pos, 95.0))
            else:
                scale = max(float(np.max(np.abs(a))), 1.0)
            normed.append(a / max(scale, 1e-6))
        return np.asarray(np.median(np.stack(normed, axis=0), axis=0), dtype=np.float32)
    raise ValueError(f"Unsupported peak aggregation mode '{mode}'.")


def load_xrdct_phase(
    phase_folder: str,
    phases: Sequence[str],
    shape: Sequence[int],
    crop: slice = slice(None),
    entry_names: dict[str, str] | None = None,
    **kwargs: Any,
) -> list[np.ndarray]:
    """Load stacked NeXus (``.nxs``) XRD-CT slices in deterministic natural order."""
    resolved_entries = entry_names if entry_names is not None else {"default": "data"}
    default_entry = resolved_entries.get("default", "data")

    all_entries = sorted(os.listdir(phase_folder), key=_natural_sort_key)
    xrdct_data: list[np.ndarray] = []

    for phase in phases:
        files = [f for f in all_entries if phase.lower() in f.lower()][crop]

        assert len(files) > 0, (
            f"No files found for phase {phase} in folder {phase_folder}."
        )
        assert any(f.endswith(".nxs") for f in files), (
            f"Expected .nxs files for phase {phase}, but found none."
        )
        assert len(files) == shape[0], (
            f"Expected {shape[0]} files for phase {phase}, but found {len(files)}."
        )

        entry_key = resolved_entries.get(phase, default_entry)
        xrdct_i = np.zeros(shape, dtype=np.float64)
        for idx in range(shape[0]):
            with h5py.File(os.path.join(phase_folder, files[idx]), "r") as file:
                data = np.array(file[entry_key])
                xrdct_i[idx] += data.squeeze()
        xrdct_data.append(xrdct_i)

    return xrdct_data


def load_diad_xct_zn13x(
    xct_path: str | Path,
    recon: str = "Astra",
    zyx_slices: tuple[slice, slice, slice] | None = (
        slice(150, 2000),
        slice(350, 2200),
        slice(350, 2200),
    ),
    stride: int | tuple[int, int, int] = 1,
    flip_x: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    """Load and crop DIAD Zn13X Astra-reconstructed XCT volume out-of-core."""
    return load_xct_subvolume(
        xct_path,
        recon=recon,
        zyx_slices=zyx_slices,
        stride=stride,
        flip_x=flip_x,
    )


def _load_cleaned_xrdct_bundle(
    bundle_path: Path,
    phase_list: list[str],
    include_auxiliary: bool = False,
) -> dict[str, np.ndarray]:
    """Load an artifact-cleaned XRD-CT `.npz` or `.h5` bundle."""
    out: dict[str, np.ndarray] = {}
    if bundle_path.suffix.lower() == ".npz":
        with np.load(bundle_path, allow_pickle=False) as data:
            lower_map = {k.lower(): k for k in data.files}
            for ph in phase_list:
                k_match = lower_map.get(ph.lower())
                if k_match is None:
                    raise KeyError(
                        f"Phase '{ph}' not found in {bundle_path} ({data.files})."
                    )
                out[ph] = np.asarray(data[k_match], dtype=np.float32)
            if include_auxiliary:
                for aux in ("na_sigma2", "zn_sigma2", "valid_mask", "scan_ids"):
                    if aux in lower_map:
                        out[aux] = np.asarray(data[lower_map[aux]])
        return out

    with h5py.File(bundle_path, "r") as hf:
        lower_map = {str(k).lower(): str(k) for k in hf.keys()}
        for ph in phase_list:
            k_match = lower_map.get(ph.lower())
            if k_match is None:
                raise KeyError(
                    f"Phase '{ph}' not found in {bundle_path} ({list(hf.keys())})."
                )
            out[ph] = np.asarray(hf[k_match], dtype=np.float32)
        if include_auxiliary:
            for aux in ("na_sigma2", "zn_sigma2", "valid_mask", "scan_ids"):
                if aux in lower_map:
                    out[aux] = np.asarray(hf[lower_map[aux]])
    return out


def load_diad_xrdct(
    phase_folder: str | Path,
    phases: Sequence[str] | None = None,
    *,
    resolution: Literal["50um", "25um"] | None = None,
    scan_ids: Sequence[int] | None = None,
    aggregation: PeakAggregationMode = "all_sum",
    target_q_by_phase: dict[str, float] | None = None,
    q_tol: float = 0.02,
    flip_x: bool = True,
    correct_motor_offset_deg: float | None = None,
    include_auxiliary: bool = False,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """Load DIAD K11 XRD-CT NeXus phase stacks or pre-cleaned ``.npz``/``.h5``
    bundles as ``{phase_name: (Z, Y, X)}``.

    Resolves explicit ``50um`` (``43336..43347, 43393..43400``, 20 slices) vs
    ``25um`` ROI (``43322..43325``, 4 slices) scan IDs, aggregates all fitted
    Bragg reflections by default (``aggregation="all_sum"``), and corrects the
    ``+47.5°`` ``gts_cs_theta`` soft-motor zero offset on ``50 µm`` scans
    (``-47.5°`` after horizontal ``flip_x``).
    """
    phase_list = list(phases) if phases is not None else ["Na", "Zn"]
    src_path = Path(phase_folder)

    if src_path.is_file() and src_path.suffix.lower() in (".npz", ".h5", ".hdf5"):
        return _load_cleaned_xrdct_bundle(
            src_path, phase_list=phase_list, include_auxiliary=include_auxiliary
        )

    if src_path.is_dir() and aggregation != "legacy":
        entries = os.listdir(src_path)
        scan_map: dict[tuple[int, str], Path] = {}
        for fn in entries:
            m = _OUTPUT_SCAN_RE.match(fn)
            if m is not None:
                scan_map[(int(m.group(1)), m.group(2).lower())] = src_path / fn

        if scan_map:
            avail_scans = sorted({sid for (sid, _) in scan_map})
            if scan_ids is not None:
                chosen_ids = [int(s) for s in scan_ids]
            elif resolution == "25um":
                chosen_ids = [s for s in DIAD_ZN13X_25UM_SCAN_IDS if s in avail_scans]
            elif resolution == "50um":
                chosen_ids = [s for s in DIAD_ZN13X_50UM_SCAN_IDS if s in avail_scans]
            else:
                s50 = [s for s in DIAD_ZN13X_50UM_SCAN_IDS if s in avail_scans]
                chosen_ids = s50 if s50 else avail_scans

            if not chosen_ids:
                raise FileNotFoundError(
                    f"No matching XRD-CT scan IDs found in {src_path} "
                    f"(available={avail_scans}, resolution={resolution})."
                )

            q_defaults = {"na": 1.651, "zn": 1.656}
            if target_q_by_phase:
                for k_p, v_q in target_q_by_phase.items():
                    q_defaults[k_p.lower()] = float(v_q)

            result: dict[str, np.ndarray] = {}
            for ph in phase_list:
                ph_l = ph.lower()
                slices_2d: list[np.ndarray] = []
                for sid in chosen_ids:
                    fp = scan_map.get((sid, ph_l))
                    if fp is None:
                        raise FileNotFoundError(
                            f"Missing phase '{ph}' file for scan {sid} in {src_path}."
                        )
                    peaks = load_xrdct_peak_maps(fp, flip_x=flip_x)
                    if peaks:
                        sl_2d = aggregate_peak_maps(
                            peaks,
                            mode=aggregation,
                            target_q=q_defaults.get(ph_l),
                            q_tol=q_tol,
                        )
                    else:
                        with h5py.File(fp, "r") as hf_fb:
                            ds_key = "entry/data" if "entry/data" in hf_fb else "data"
                            sl_2d = np.asarray(
                                hf_fb[ds_key], dtype=np.float32
                            ).squeeze()
                            if flip_x:
                                sl_2d = np.fliplr(sl_2d).copy()

                    if correct_motor_offset_deg is None:
                        eff_deg = 47.5 if sid in DIAD_ZN13X_50UM_SCAN_IDS else 0.0
                    else:
                        eff_deg = float(correct_motor_offset_deg)

                    if abs(eff_deg) > 1e-6:
                        rot_angle = -eff_deg if flip_x else eff_deg
                        sl_2d = rotate(
                            sl_2d,
                            angle=rot_angle,
                            reshape=False,
                            order=1,
                            mode="nearest",
                        ).astype(np.float32)
                    slices_2d.append(sl_2d.astype(np.float32))
                result[ph] = np.stack(slices_2d, axis=0)
            return result

    # Legacy fallback (e.g. non-output_* .nxs stacks or mocked test_loaders paths)
    phase_arrays = load_xrdct_phase(
        str(phase_folder),
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
        phase: (np.flip(arr, axis=2).copy() if flip_x else arr.copy())
        for phase, arr in zip(phase_list, phase_arrays, strict=False)
    }


def detrend_radial_cupping(
    volume_or_slice: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Remove low-order local-tomography radial cupping and tilt drift per 2D
    slice via :func:`basecoat.detrend_radial_cupping`.

    .. warning::
       On radially zoned or core--shell specimens (e.g. ion-exchanged catalyst
       pellets where radial ``Z``-contrast tracks chemical exchange), polynomial
       cupping subtraction is confounded with genuine radial attenuation gradients.
       Use only as an optional pore-thresholding helper on chemically uniform
       matrices, and prefer :func:`remove_polar_rings_conservative` when
       preserving radial chemical contrast.
    """
    arr = np.asarray(volume_or_slice, dtype=np.float32)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}.")
    fin = np.isfinite(arr)
    if not np.any(fin):
        return arr.copy()
    eff_mask = fin if valid_mask is None else (np.asarray(valid_mask, dtype=bool) & fin)
    if int(np.count_nonzero(eff_mask)) < 16:
        return arr.copy()
    sanitized = np.where(fin, arr, float(np.median(arr[fin]))).astype(np.float32)
    detrended = _bc_detrend_radial_cupping(sanitized, valid_mask_zyx=eff_mask)
    return np.where(eff_mask, detrended, arr).astype(np.float32)


def remove_polar_rings_conservative(
    volume_or_slice: np.ndarray,
    center: tuple[float, float] | None = None,
    radial_sigma: float = 2.5,
    angular_window_deg: float = 45.0,
    grad_protect_quantile: float = 0.85,
    mad_clip: float = 3.0,
) -> np.ndarray:
    """Suppress concentric synchrotron XCT ring/arc artifacts in polar coordinates
    via :func:`basecoat.remove_rings_polar_conservative`.
    """
    _ = grad_protect_quantile
    arr = np.asarray(volume_or_slice, dtype=np.float32)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}.")
    height, width = (
        (arr.shape[0], arr.shape[1]) if arr.ndim == 2 else (arr.shape[1], arr.shape[2])
    )
    if min(height, width) < 4:
        return arr.copy()

    fin = np.isfinite(arr)
    if not np.any(fin):
        return arr.copy()
    if not np.all(fin):
        f_vals = arr[fin]
        work = np.nan_to_num(
            arr,
            nan=float(np.median(f_vals)),
            posinf=float(np.max(f_vals)),
            neginf=float(np.min(f_vals)),
        ).astype(np.float32)
    else:
        work = arr

    center_yx = (
        ((height - 1) * 0.5, (width - 1) * 0.5)
        if center is None
        else (float(center[0]), float(center[1]))
    )
    cleaned, _ = _bc_remove_rings_polar(
        work,
        center_yx=center_yx,
        radial_sigma_px=max(2.5, float(radial_sigma)),
        angular_window_deg=float(angular_window_deg),
        mad_clip_factor=float(mad_clip),
    )
    return np.where(fin, cleaned, arr).astype(np.float32)


def load_i13_xct(xct_path: str, **kwargs: Any) -> np.ndarray:
    """Load Diamond I13 Tomopy-reconstructed XCT volume."""
    return load_xct(xct_path, recon="Tomopy", **kwargs)


# ── Modality-Agnostic Correlative Loaders ─────────────────────────────────────


def load_image_array(path: str | Path) -> np.ndarray:
    """Load a 2D/3D OME-Zarr, TIFF, PNG, or NPY array as float32."""
    p = Path(path)
    lower_name = p.name.lower()
    sfx = p.suffix.lower()
    if p.is_dir() and lower_name.endswith((".ome.zarr", ".zarr")):
        return np.asarray(read_ome_zarr(p, level=0).data, dtype=np.float32)
    if sfx == ".npy":
        return np.asarray(read_volume(p).data, dtype=np.float32)
    if sfx in (".tif", ".tiff"):
        try:
            return np.asarray(read_volume(p).data, dtype=np.float32)
        except Exception:
            pass
    with Image.open(p) as img:
        return np.array(img, dtype=np.float32)


def load_correlative_tiff_pair(
    hr_path: str | Path,
    guide_paths: Sequence[str | Path] | dict[str, str | Path],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
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
) -> tuple[np.ndarray, np.ndarray, list[str]]:
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
