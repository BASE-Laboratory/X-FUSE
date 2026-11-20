"""Standardized scientific I/O for correlative microscopy & tomography:
- Export and reload ``UncertaintyReport`` / ``VolumetricUncertaintyReport`` in
  **OME-TIFF** (``.ome.tif``), **HDF5/NeXus** (``.h5``), and **Zarr** (``.zarr``).
- Vendor electron/X-ray microscopy file loading via optional **RosettaSciIO**
  (``.hspy``, ``.emd``, ``.dm3``/``.dm4``, ``.bcf``, ``.mrc``, ``.ser``).
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from basecoat import (
    AxisSpec,
    Volume,
    VolumeSet,
    read_ome_zarr,
    read_volume,
    write_ome_zarr,
    write_volume,
)
from PIL import Image

from .uncertainty import UncertaintyReport
from .volumetric import VolumetricUncertaintyReport

_VENDOR_EXTENSIONS = {".hspy", ".emd", ".dm3", ".dm4", ".bcf", ".mrc", ".ser", ".msa"}


def _report_to_arrays(
    report: UncertaintyReport | VolumetricUncertaintyReport,
    channel_names: Sequence[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], bool]:
    is_3d = isinstance(report, VolumetricUncertaintyReport)
    if isinstance(report, VolumetricUncertaintyReport):
        probs = report.probabilities.detach().cpu().float().numpy()  # (K, Z, Y, X)
        arrays: dict[str, np.ndarray] = {
            "probabilities": probs,
            "hard_mask": report.hard_mask.detach().cpu().numpy().astype(np.int32),
            "aleatoric_entropy": (
                report.aleatoric_entropy.detach().cpu().float().numpy()
            ),
            "epistemic_variance": (
                report.triplane_variance.detach().cpu().float().numpy()
            ),
            "physical_residual": (
                report.physical_residual.detach().cpu().float().numpy()
            ),
            "combined_uncertainty": (
                report.combined_uncertainty.detach().cpu().float().numpy()
            ),
        }
        if report.prediction_sets is not None:
            arrays["prediction_sets"] = (
                report.prediction_sets.detach().cpu().numpy().astype(np.uint8)
            )
        if report.conformal_interval_halfwidth is not None:
            arrays["conformal_interval_halfwidth"] = (
                report.conformal_interval_halfwidth.detach().cpu().float().numpy()
            )
        meta: dict[str, Any] = {
            "report_type": "VolumetricUncertaintyReport",
            "z_slice_consistency": float(report.z_slice_consistency),
            "diagnostics": dict(report.diagnostics),
        }
    else:
        probs = report.probabilities[0].detach().cpu().float().numpy()  # (K, H, W)
        arrays = {
            "probabilities": probs,
            "hard_mask": report.hard_mask[0].detach().cpu().numpy().astype(np.int32),
            "aleatoric_entropy": (
                report.aleatoric_entropy[0].detach().cpu().float().numpy()
            ),
            "confidence_margin": (
                report.confidence_margin[0].detach().cpu().float().numpy()
            ),
            "epistemic_variance": (
                report.epistemic_variance[0].detach().cpu().float().numpy()
            ),
            "physical_residual": (
                report.physical_residual[0].detach().cpu().float().numpy()
            ),
            "combined_uncertainty": (
                report.combined_uncertainty[0].detach().cpu().float().numpy()
            ),
            "unresolved_mask": (
                report.unresolved_mask[0].detach().cpu().numpy().astype(np.uint8)
            ),
        }
        if report.prediction_sets is not None:
            arrays["prediction_sets"] = (
                report.prediction_sets[0].detach().cpu().numpy().astype(np.uint8)
            )
        if report.conformal_interval_halfwidth is not None:
            arrays["conformal_interval_halfwidth"] = (
                report.conformal_interval_halfwidth[0].detach().cpu().float().numpy()
            )
        meta = {
            "report_type": "UncertaintyReport",
            "diagnostics": dict(report.diagnostics),
        }

    n_k = int(probs.shape[0])
    if channel_names is not None:
        names = [str(c) for c in channel_names]
    else:
        names = [f"phase_{i}" for i in range(n_k)]
    meta["channel_names"] = names
    meta["spatial_shape"] = [int(d) for d in probs.shape[1:]]
    return arrays, meta, is_3d


def save_uncertainty_report(
    report: UncertaintyReport | VolumetricUncertaintyReport,
    path: str | Path,
    channel_names: Sequence[str] | None = None,
    pixel_size_um: float | tuple[float, ...] | None = None,
) -> Path:
    """Export a 2D ``UncertaintyReport`` or 3D ``VolumetricUncertaintyReport``
    to ``.ome.tif`` / ``.tif``, ``.h5`` / ``.hdf5`` / ``.nxs``, or ``.zarr``.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays, meta, is_3d = _report_to_arrays(report, channel_names=channel_names)
    if pixel_size_um is not None:
        meta["pixel_size_um"] = (
            list(pixel_size_um)
            if isinstance(pixel_size_um, (tuple, list))
            else float(pixel_size_um)
        )

    lower_name = out_path.name.lower()
    suffix = out_path.suffix.lower()

    if suffix in (".h5", ".hdf5", ".nxs"):
        with h5py.File(out_path, "w") as hf:
            grp = hf.create_group("xfuse_report")
            for k_name, arr in arrays.items():
                grp.create_dataset(k_name, data=arr, compression="gzip")
            grp.attrs["metadata_json"] = json.dumps(meta)
        return out_path

    if suffix == ".zarr":
        try:
            import zarr  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Zarr export requires optional I/O extras: `pip install x-fuse[io]`."
            ) from exc
        root: Any = zarr.open_group(str(out_path), mode="w")
        for k_name, arr in arrays.items():
            if hasattr(root, "create_array"):
                root.create_array(k_name, data=arr)
            else:
                root.create_dataset(k_name, data=arr)
        root.attrs["metadata_json"] = json.dumps(meta)
        return out_path

    if lower_name.endswith((".ome.tif", ".ome.tiff", ".tif", ".tiff")):
        probs = arrays["probabilities"].astype(np.float32)
        n_probs = int(probs.shape[0])
        p_names = list(meta["channel_names"])
        while len(p_names) < n_probs:
            p_names.append(f"phase_{len(p_names)}")

        extra_keys = [
            "hard_mask",
            "aleatoric_entropy",
            "epistemic_variance",
            "physical_residual",
            "combined_uncertainty",
        ]
        if "confidence_margin" in arrays:
            extra_keys.append("confidence_margin")
        if "unresolved_mask" in arrays:
            extra_keys.append("unresolved_mask")

        stack_blocks: list[np.ndarray] = [probs]
        all_labels: list[str] = list(p_names[:n_probs])
        layout: dict[str, Any] = {"probabilities": [0, n_probs]}
        cursor = n_probs

        for ek in extra_keys:
            stack_blocks.append(arrays[ek].astype(np.float32)[None, ...])
            all_labels.append(ek)
            layout[ek] = cursor
            cursor += 1

        if "prediction_sets" in arrays:
            ps = arrays["prediction_sets"].astype(np.float32)
            n_ps = int(ps.shape[0])
            stack_blocks.append(ps)
            all_labels.extend([f"prediction_sets_{i}" for i in range(n_ps)])
            layout["prediction_sets"] = [cursor, cursor + n_ps]
            cursor += n_ps

        if "conformal_interval_halfwidth" in arrays:
            hw = arrays["conformal_interval_halfwidth"].astype(np.float32)
            n_hw = int(hw.shape[0])
            stack_blocks.append(hw)
            all_labels.extend([f"conformal_halfwidth_{i}" for i in range(n_hw)])
            layout["conformal_interval_halfwidth"] = [cursor, cursor + n_hw]
            cursor += n_hw

        full_stack = np.concatenate(stack_blocks, axis=0)
        meta["ome_channel_labels"] = all_labels
        meta["stack_layout"] = layout

        try:
            import tifffile  # type: ignore[import-not-found]

            ome_meta: dict[str, Any] = {
                "axes": "CZYX" if is_3d else "CYX",
                "Channel": {"Name": all_labels},
                "Description": json.dumps(meta),
            }
            if pixel_size_um is not None:
                if isinstance(pixel_size_um, (tuple, list)):
                    if is_3d and len(pixel_size_um) >= 3:
                        ome_meta["PhysicalSizeZ"] = float(pixel_size_um[0])
                        ome_meta["PhysicalSizeZUnit"] = "µm"
                        ome_meta["PhysicalSizeY"] = float(pixel_size_um[1])
                        ome_meta["PhysicalSizeYUnit"] = "µm"
                        ome_meta["PhysicalSizeX"] = float(pixel_size_um[2])
                        ome_meta["PhysicalSizeXUnit"] = "µm"
                    elif len(pixel_size_um) >= 2:
                        ome_meta["PhysicalSizeY"] = float(pixel_size_um[0])
                        ome_meta["PhysicalSizeYUnit"] = "µm"
                        ome_meta["PhysicalSizeX"] = float(pixel_size_um[1])
                        ome_meta["PhysicalSizeXUnit"] = "µm"
                else:
                    px = float(pixel_size_um)
                    ome_meta["PhysicalSizeX"] = px
                    ome_meta["PhysicalSizeXUnit"] = "µm"
                    ome_meta["PhysicalSizeY"] = px
                    ome_meta["PhysicalSizeYUnit"] = "µm"
            is_ome = lower_name.endswith((".ome.tif", ".ome.tiff"))
            tifffile.imwrite(
                out_path,
                full_stack,
                photometric="minisblack",
                ome=is_ome,
                metadata=ome_meta if is_ome else None,
                description=None if is_ome else json.dumps(meta),
            )
            return out_path
        except ImportError:
            if is_3d:
                raise ImportError(
                    "3D OME-TIFF export requires `tifffile` (`pip install x-fuse[io]`)."
                ) from None
            from PIL.TiffImagePlugin import ImageFileDirectory_v2

            ifd = ImageFileDirectory_v2()
            ifd[270] = json.dumps(meta)
            frames = [Image.fromarray(ch) for ch in full_stack]
            frames[0].save(
                out_path, save_all=True, append_images=frames[1:], tiffinfo=ifd
            )
            return out_path

    if suffix == ".npz":
        npz_payload: dict[str, Any] = {
            **arrays,
            "metadata_json": np.array(json.dumps(meta)),
        }
        np.savez_compressed(str(out_path), **npz_payload)
        return out_path

    raise ValueError(
        f"Unsupported output extension for '{out_path}'. "
        "Supported: .ome.tif, .tif, .h5, .hdf5, .nxs, .zarr, .npz."
    )


def _normalize_loaded_arrays(out: dict[str, Any]) -> dict[str, Any]:
    if "prediction_sets" in out:
        out["prediction_sets"] = np.asarray(out["prediction_sets"]) > 0.5
    if "unresolved_mask" in out:
        out["unresolved_mask"] = np.asarray(out["unresolved_mask"]) > 0.5
    if "hard_mask" in out:
        out["hard_mask"] = np.rint(np.asarray(out["hard_mask"])).astype(np.int32)
    return out


def load_uncertainty_report(path: str | Path) -> dict[str, Any]:
    """Load an exported X-FUSE report (``.h5``, ``.zarr``, ``.npz``, ``.ome.tif``)."""
    in_path = Path(path)
    suffix = in_path.suffix.lower()
    lower_name = in_path.name.lower()

    if suffix == ".npz":
        with np.load(in_path, allow_pickle=False) as npz:
            out_npz: dict[str, Any] = {
                k: np.asarray(npz[k]) for k in npz.files if k != "metadata_json"
            }
            raw_meta_npz = (
                str(npz["metadata_json"].item())
                if "metadata_json" in npz.files
                else "{}"
            )
            out_npz["metadata"] = json.loads(raw_meta_npz)
        return _normalize_loaded_arrays(out_npz)

    if suffix in (".h5", ".hdf5", ".nxs"):
        with h5py.File(in_path, "r") as hf:
            grp = hf["xfuse_report"]
            out: dict[str, Any] = {k: np.array(grp[k]) for k in grp.keys()}
            raw_meta = grp.attrs.get("metadata_json", "{}")
            out["metadata"] = json.loads(
                raw_meta.decode("utf-8")
                if isinstance(raw_meta, bytes)
                else str(raw_meta)
            )
        return _normalize_loaded_arrays(out)

    if suffix == ".zarr":
        import zarr  # type: ignore[import-not-found]

        root_in: Any = zarr.open_group(str(in_path), mode="r")
        out = {k: np.asarray(root_in[k]) for k in root_in.keys()}
        out["metadata"] = json.loads(str(root_in.attrs.get("metadata_json", "{}")))
        return _normalize_loaded_arrays(out)

    if lower_name.endswith((".ome.tif", ".ome.tiff", ".tif", ".tiff")):
        try:
            import tifffile  # type: ignore[import-not-found]

            with tifffile.TiffFile(in_path) as tf:
                stack = tf.asarray()
                desc = str(getattr(tf.pages[0], "description", "") or "")
        except ImportError:
            from PIL import ImageSequence

            with Image.open(in_path) as im:
                desc = str(im.tag_v2.get(270, "") if hasattr(im, "tag_v2") else "")
                frames_in = [
                    np.array(frame, dtype=np.float32)
                    for frame in ImageSequence.Iterator(im)
                ]
                stack = np.stack(frames_in, axis=0)

        meta_dict: dict[str, Any] = {}
        if desc.startswith("{"):
            meta_dict = json.loads(desc)
        elif "Description=" in desc or "<Description>" in desc:
            import html
            import re

            m = re.search(r"<Description>(.*?)</Description>", desc, flags=re.DOTALL)
            if m:
                unesc = html.unescape(m.group(1)).strip()
                if unesc.startswith("{"):
                    meta_dict = json.loads(unesc)

        spatial_shape = meta_dict.get("spatial_shape")
        if isinstance(spatial_shape, list) and len(spatial_shape) == 3:
            z_s, y_s, x_s = (
                int(spatial_shape[0]),
                int(spatial_shape[1]),
                int(spatial_shape[2]),
            )
            if stack.ndim == 3 and z_s == 1:
                stack = stack[:, np.newaxis, :, :]
            elif (
                stack.ndim == 3
                and z_s > 1
                and stack.shape[0] % z_s == 0
                and tuple(stack.shape[1:]) == (y_s, x_s)
            ):
                stack = stack.reshape(stack.shape[0] // z_s, z_s, y_s, x_s)

        result: dict[str, Any] = {"stack": stack, "metadata": meta_dict}
        layout = meta_dict.get("stack_layout")
        if isinstance(layout, dict):
            for key_name, spec in layout.items():
                if isinstance(spec, list) and len(spec) == 2:
                    result[key_name] = stack[int(spec[0]) : int(spec[1])]
                elif isinstance(spec, int):
                    result[key_name] = stack[int(spec)]
        return _normalize_loaded_arrays(result)

    raise ValueError(f"Unsupported report file format: {in_path}")


def load_microscopy_file(
    path: str | Path, signal_index: int = 0
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a microscopy dataset (``.ome.zarr``, ``.tif``, ``.npy``, ``.npz``,
    ``.h5``/``.nxs``, or vendor formats such as ``.emd``, ``.dm3``/``.dm4``,
    ``.bcf``, ``.hspy`` via RosettaSciIO).

    Returns:
        ``(data_float32, metadata_dict)``.
    """
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Microscopy file not found: {in_path}")

    lower_name = in_path.name.lower()
    suffix = in_path.suffix.lower()
    if in_path.is_dir() and lower_name.endswith((".ome.zarr", ".zarr")):
        vol = read_ome_zarr(in_path, level=0)
        arr = np.asarray(vol.data, dtype=np.float32)
        meta_z: dict[str, Any] = {
            "format": "ome.zarr",
            "shape": list(arr.shape),
            "space": vol.space,
            "provenance": vol.provenance,
        }
        return arr, meta_z

    if suffix == ".npy":
        arr = np.asarray(read_volume(in_path).data, dtype=np.float32)
        return arr, {"format": "npy", "shape": list(arr.shape)}

    if suffix == ".npz":
        with np.load(in_path, allow_pickle=False) as npz:
            arr_keys = [k for k in npz.files if k != "metadata_json"]
            if not arr_keys:
                raise ValueError(f"No array entries found in {in_path}.")
            chosen_k = arr_keys[max(0, min(int(signal_index), len(arr_keys) - 1))]
            arr = np.asarray(npz[chosen_k], dtype=np.float32)
        return arr, {"format": "npz", "key": chosen_k, "shape": list(arr.shape)}

    if suffix in (".h5", ".hdf5", ".nxs"):
        with h5py.File(in_path, "r") as hf:
            preferred = ("data", "entry/data", "xfuse_report/probabilities")
            if int(signal_index) == 0:
                for pk in preferred:
                    if (
                        pk in hf
                        and isinstance(hf[pk], h5py.Dataset)
                        and hf[pk].ndim >= 2
                    ):
                        arr = np.asarray(hf[pk], dtype=np.float32)
                        return arr, {
                            "format": suffix.lstrip("."),
                            "dataset": pk,
                            "shape": list(arr.shape),
                        }
            ds_list: list[tuple[str, h5py.Dataset]] = []

            def _collect(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
                    ds_list.append((name, obj))

            hf.visititems(_collect)
            if not ds_list:
                raise ValueError(f"No >=2D dataset found in HDF5 file: {in_path}")
            idx_c = max(0, min(int(signal_index), len(ds_list) - 1))
            ds_name, ds_obj = ds_list[idx_c]
            arr = np.asarray(ds_obj, dtype=np.float32)
        return arr, {
            "format": suffix.lstrip("."),
            "dataset": ds_name,
            "shape": list(arr.shape),
        }

    if suffix in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
        if suffix in (".tif", ".tiff"):
            try:
                arr = np.asarray(read_volume(in_path).data, dtype=np.float32)
                return arr, {"format": "tiff", "shape": list(arr.shape)}
            except Exception:
                pass
        with Image.open(in_path) as im:
            arr = np.array(im, dtype=np.float32)
        return arr, {"format": suffix.lstrip("."), "shape": list(arr.shape)}

    if suffix in _VENDOR_EXTENSIONS:
        rsciio_mod = _import_rsciio_reader(suffix)
        entries = rsciio_mod.file_reader(str(in_path))
        if isinstance(entries, dict):
            entries = [entries]
        entry = entries[min(int(signal_index), len(entries) - 1)]
        data = np.asarray(entry["data"], dtype=np.float32)
        axes = entry.get("axes", [])
        return data, {
            "format": suffix.lstrip("."),
            "shape": list(data.shape),
            "axes": axes,
            "metadata": entry.get("metadata", {}),
        }

    raise ValueError(
        f"Unsupported microscopy extension '{suffix}'. Install `x-fuse[io]` for "
        f"vendor formats ({sorted(_VENDOR_EXTENSIONS)})."
    )


def _import_rsciio_reader(suffix: str) -> Any:
    reader_map = {
        ".hspy": "rsciio.hspy",
        ".emd": "rsciio.emd",
        ".dm3": "rsciio.digitalmicrograph",
        ".dm4": "rsciio.digitalmicrograph",
        ".bcf": "rsciio.bruker",
        ".mrc": "rsciio.mrc",
        ".ser": "rsciio.tia",
        ".msa": "rsciio.msa",
    }
    mod_name = reader_map.get(suffix)
    if mod_name is None:
        raise ValueError(f"No RosettaSciIO reader mapped for '{suffix}'.")
    try:
        import importlib

        return importlib.import_module(mod_name)
    except ImportError as exc:
        raise ImportError(
            f"Reading '{suffix}' files requires RosettaSciIO. "
            "Install optional microscopy I/O dependencies via `pip install x-fuse[io]`."
        ) from exc


def _write_uncompressed_uint8_3d_tiff(path: Path, vol_u8: np.ndarray) -> None:
    write_volume(
        Volume.uncalibrated(vol_u8),
        path,
        compression=None,
        overwrite=True,
    )


def _extract_phase_and_conformal_arrays(
    vol_report_or_mask: Any,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    dict[str, Any],
]:
    import torch

    probs_np: np.ndarray | None = None
    conf_hw_np: np.ndarray | None = None
    comb_unc_np: np.ndarray | None = None
    prov: dict[str, Any] = {}
    if isinstance(vol_report_or_mask, VolumetricUncertaintyReport):
        raw_mask = vol_report_or_mask.hard_mask.detach().cpu().numpy()
        # Map negative abstained voxels (-1) to 255 rather than 0 (pore/void)
        labeled = np.where(raw_mask < 0, 255, np.clip(raw_mask, 0, 255)).astype(
            np.uint8
        )
        probs_np = vol_report_or_mask.probabilities.detach().cpu().float().numpy()
        comb_unc_np = (
            vol_report_or_mask.combined_uncertainty.detach().cpu().float().numpy()
        )
        pred_sets = (
            vol_report_or_mask.prediction_sets.detach().cpu().numpy().astype(bool)
            if vol_report_or_mask.prediction_sets is not None
            else None
        )
        conf_hw_np = (
            vol_report_or_mask.conformal_interval_halfwidth.detach()
            .cpu()
            .float()
            .numpy()
            if vol_report_or_mask.conformal_interval_halfwidth is not None
            else None
        )
        prov = {
            "z_slice_consistency": float(vol_report_or_mask.z_slice_consistency),
            "diagnostics": dict(vol_report_or_mask.diagnostics),
        }
    elif isinstance(vol_report_or_mask, torch.Tensor):
        raw_mask = vol_report_or_mask.detach().cpu().numpy()
        labeled = np.where(raw_mask < 0, 255, np.clip(raw_mask, 0, 255)).astype(
            np.uint8
        )
        pred_sets = None
    else:
        raw_mask = np.asarray(vol_report_or_mask)
        labeled = np.where(raw_mask < 0, 255, np.clip(raw_mask, 0, 255)).astype(
            np.uint8
        )
        pred_sets = None
    return labeled, probs_np, pred_sets, conf_hw_np, comb_unc_np, prov


def _compute_conformal_phase_envelopes(
    labeled: np.ndarray,
    probs_np: np.ndarray | None,
    pred_sets: np.ndarray | None,
    conf_hw_np: np.ndarray | None,
    active_phase: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if pred_sets is None:
        return None
    act = int(active_phase)
    is_multiclass = (
        probs_np is not None
        and probs_np.shape[0] >= 2
        and pred_sets.shape[0] == probs_np.shape[0]
        and float(np.max(np.abs(np.sum(probs_np, axis=0) - 1.0))) < 1e-3
    )
    in_lower: np.ndarray | None = None
    in_upper: np.ndarray | None = None
    if is_multiclass and act < pred_sets.shape[0]:
        set_sizes = pred_sets.sum(axis=0)
        in_lower = pred_sets[act] & (set_sizes == 1)
        in_upper = pred_sets[act]
    elif not is_multiclass and act <= pred_sets.shape[0]:
        # Multilabel: hard_mask uses 0=background/pore and 1..M=channel m+1
        if act == 0:
            in_lower = ~np.any(pred_sets, axis=0)
            if (
                probs_np is not None
                and conf_hw_np is not None
                and conf_hw_np.shape == probs_np.shape
            ):
                in_upper = np.all((probs_np - conf_hw_np) < 0.5, axis=0)
            else:
                in_upper = in_lower | (labeled == 0)
        else:
            ch_idx = act - 1
            in_upper = pred_sets[ch_idx]
            if (
                probs_np is not None
                and conf_hw_np is not None
                and ch_idx < conf_hw_np.shape[0]
            ):
                in_lower = ((probs_np[ch_idx] - conf_hw_np[ch_idx]) >= 0.5) & in_upper
            else:
                in_lower = in_upper & (labeled == act)

    if in_lower is None or in_upper is None:
        return None
    complement_val = np.uint8(1 if act == 0 else 0)
    act_val = np.uint8(act)
    lower_u8 = np.where(in_lower, act_val, complement_val).astype(np.uint8)
    upper_u8 = np.where(in_upper, act_val, complement_val).astype(np.uint8)
    return lower_u8, upper_u8


def export_volume_ome_zarr(
    vol_report_or_mask: Any,
    store_path: str | Path,
    *,
    image_volume: np.ndarray | None = None,
    voxel_size_um: float | Sequence[float] | None = None,
    origin_um: Sequence[float] | None = None,
    space: str | None = None,
    label_names: Mapping[int, str] | None = None,
    active_phase: int = 0,
    export_conformal_bounds: bool = True,
    extra_uncertainties: Mapping[str, np.ndarray] | None = None,
    multiscale_levels: int = 2,
    overwrite: bool = True,
) -> Path:
    """Export a 3D ``VolumetricUncertaintyReport`` (or phase mask) as an OME-Zarr v0.4
    contract store via ``basecoat``.

    Writes a calibrated primary 3D ``Volume`` (``image_volume`` when supplied,
    otherwise ``combined_uncertainty`` or ``float32`` phase field) together with
    mode-downsampled integer ``labels/phase`` (and optional ``labels/phase_lower`` /
    ``labels/phase_upper`` conformal phase envelopes) plus ``uncertainties/*``
    sidecars so Layer 3 solvers (``OpenImpala``, ``SMS``) and Layer 4 evaluators
    consume ``X-FUSE`` outputs without bespoke glue.
    """
    act = int(active_phase)
    if not (0 <= act <= 255):
        raise ValueError(f"active_phase must lie in [0, 255], got {active_phase}.")

    labeled, probs_np, pred_sets, conf_hw_np, comb_unc_np, prov = (
        _extract_phase_and_conformal_arrays(vol_report_or_mask)
    )
    if labeled.ndim != 3:
        raise ValueError(
            f"Expected 3D (D, H, W) volume for OME-Zarr export, got {labeled.shape}."
        )

    if image_volume is not None:
        img_arr = np.asarray(image_volume, dtype=np.float32)
        if img_arr.shape != labeled.shape:
            raise ValueError(
                f"image_volume shape {img_arr.shape} must match "
                f"phase mask shape {labeled.shape}."
            )
    elif comb_unc_np is not None:
        img_arr = np.asarray(comb_unc_np, dtype=np.float32)
    else:
        img_arr = labeled.astype(np.float32)

    img_vol = Volume.from_array(
        img_arr,
        voxel_size_um=voxel_size_um,
        origin_um=origin_um,
        space=space,
        provenance=prov,
    )
    resolved_names: dict[int, str] | None = None
    if label_names is not None:
        resolved_names = {int(k): str(v) for k, v in label_names.items()}
    else:
        uniq = [int(v) for v in np.unique(labeled)]
        resolved_names = {v: ("abstained" if v == 255 else f"phase_{v}") for v in uniq}

    labels_map: dict[str, Volume] = {
        "phase": Volume(
            labeled,
            axes=img_vol.axes,
            space=img_vol.space,
            label_names=resolved_names,
            provenance={"active_phase": act},
        )
    }

    if export_conformal_bounds:
        envelopes = _compute_conformal_phase_envelopes(
            labeled, probs_np, pred_sets, conf_hw_np, active_phase=act
        )
        if envelopes is not None:
            lower_u8, upper_u8 = envelopes
            comp = 1 if act == 0 else 0
            env_names = {act: f"phase_{act}", comp: "complement"}
            labels_map["phase_lower"] = Volume(
                lower_u8,
                axes=img_vol.axes,
                space=img_vol.space,
                label_names=env_names,
                provenance={"envelope": "conformal_lower", "active_phase": act},
            )
            labels_map["phase_upper"] = Volume(
                upper_u8,
                axes=img_vol.axes,
                space=img_vol.space,
                label_names=env_names,
                provenance={"envelope": "conformal_upper", "active_phase": act},
            )

    unc_map: dict[str, Volume] = {}
    chan_axes = (AxisSpec(name="c", kind="channel", unit="none"), *img_vol.axes)
    if isinstance(vol_report_or_mask, VolumetricUncertaintyReport):
        for ukey, uarr in (
            ("combined_uncertainty", comb_unc_np),
            (
                "aleatoric_entropy",
                vol_report_or_mask.aleatoric_entropy.detach().cpu().float().numpy(),
            ),
            (
                "epistemic_variance",
                vol_report_or_mask.triplane_variance.detach().cpu().float().numpy(),
            ),
            (
                "physical_residual",
                vol_report_or_mask.physical_residual.detach().cpu().float().numpy(),
            ),
        ):
            if uarr is not None:
                unc_map[ukey] = Volume(
                    np.asarray(uarr, dtype=np.float32),
                    axes=img_vol.axes,
                    space=img_vol.space,
                )
        if probs_np is not None:
            unc_map["probabilities"] = Volume(
                np.asarray(probs_np, dtype=np.float32),
                axes=chan_axes,
                space=img_vol.space,
            )
        if conf_hw_np is not None:
            unc_map["conformal_interval_halfwidth"] = Volume(
                np.asarray(conf_hw_np, dtype=np.float32),
                axes=chan_axes,
                space=img_vol.space,
            )

    if extra_uncertainties is not None:
        for ukey, uarr in extra_uncertainties.items():
            arr_f32 = np.asarray(uarr, dtype=np.float32)
            unc_map[ukey] = Volume(
                arr_f32,
                axes=chan_axes if arr_f32.ndim == 4 else img_vol.axes,
                space=img_vol.space,
            )

    vset = VolumeSet(img_vol, labels=labels_map, uncertainties=unc_map)
    return write_ome_zarr(
        vset,
        store_path,
        multiscale_levels=multiscale_levels,
        overwrite=overwrite,
    )


def export_for_openimpala(
    vol_report_or_mask: Any,
    output_dir: str | Path,
    name: str = "sample",
    active_phase: int = 0,
    direction: str = "x",
    max_grid_size: int = 64,
    export_conformal_bounds: bool = True,
    *,
    _warn_deprecated: bool = True,
) -> tuple[Path, Path]:
    """Export a 3D multi-phase chemical segmentation volume (plus optional
    ``(1 - alpha)`` conformal lower/upper phase-envelope geometries) as an
    uncompressed ``uint8`` 3D TIFF and AMReX ``ParmParse`` ``.inputs`` deck
    directly compatible with ``OpenImpala`` (matching ``OpenLSR-X`` conventions).

    .. deprecated:: 1.0.1
        Tool-to-solver bespoke exporters are deprecated under ``ECOSYSTEM.md``
        and will be removed in ``v1.1.0`` once ``OpenImpala v5.1`` reads
        ``.ome.zarr`` directly; prefer :func:`export_volume_ome_zarr`.
    """
    import math

    if _warn_deprecated:
        warnings.warn(
            "export_for_openimpala is deprecated under ECOSYSTEM.md and scheduled "
            "for removal once OpenImpala v5.1 accepts OME-Zarr directly; prefer "
            "x_fuse.io.export_volume_ome_zarr (or basecoat.write_ome_zarr).",
            DeprecationWarning,
            stacklevel=2,
        )

    if direction not in ("x", "y", "z"):
        raise ValueError(f"direction must be 'x', 'y', or 'z'; got '{direction}'.")
    act = int(active_phase)
    if not (0 <= act <= 255):
        raise ValueError(f"active_phase must lie in [0, 255], got {active_phase}.")

    labeled, probs_np, pred_sets, conf_hw_np, _, _ = (
        _extract_phase_and_conformal_arrays(vol_report_or_mask)
    )
    if labeled.ndim != 3:
        raise ValueError(
            f"Expected 3D (D, H, W) volume for OpenImpala export, got {labeled.shape}."
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiff_path = out_dir / f"{name}.tif"
    inputs_path = out_dir / f"{name}_openimpala.inputs"

    _write_uncompressed_uint8_3d_tiff(tiff_path, labeled)

    d_sz, h_sz, w_sz = (
        int(labeled.shape[0]),
        int(labeled.shape[1]),
        int(labeled.shape[2]),
    )
    blocking = max(1, math.gcd(math.gcd(w_sz, h_sz), math.gcd(d_sz, 8)))
    eff_max_grid = max(blocking, (max(1, int(max_grid_size)) // blocking) * blocking)

    deck_lines = [
        "# Auto-generated by X-FUSE for OpenImpala v5.0 (AMReX / HYPRE)",
        'eb2.geom_type = "tiff"',
        f'torteff.tiff_filename = "{tiff_path.name}"',
        f"torteff.phase = {act}",
        f'torteff.direction = "{direction}"',
        f"amr.n_cell = {w_sz} {h_sz} {d_sz}",
        f"amr.blocking_factor = {blocking}",
        f"amr.max_grid_size = {eff_max_grid}",
        "torteff.verbose = 1",
    ]

    if export_conformal_bounds:
        envelopes = _compute_conformal_phase_envelopes(
            labeled, probs_np, pred_sets, conf_hw_np, active_phase=act
        )
        if envelopes is not None:
            lower_u8, upper_u8 = envelopes
            lower_tif = out_dir / f"{name}_phase{act}_lower.tif"
            upper_tif = out_dir / f"{name}_phase{act}_upper.tif"
            _write_uncompressed_uint8_3d_tiff(lower_tif, lower_u8)
            _write_uncompressed_uint8_3d_tiff(upper_tif, upper_u8)
            deck_lines.append(f'# Conformal lower envelope: "{lower_tif.name}"')
            deck_lines.append(f'# Conformal upper envelope: "{upper_tif.name}"')

    inputs_path.write_text("\n".join(deck_lines) + "\n", encoding="utf-8")
    return tiff_path, inputs_path
