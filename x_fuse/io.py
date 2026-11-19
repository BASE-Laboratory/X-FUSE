"""Standardized scientific I/O for correlative microscopy & tomography:
- Export and reload ``UncertaintyReport`` / ``VolumetricUncertaintyReport`` in
  **OME-TIFF** (``.ome.tif``), **HDF5/NeXus** (``.h5``), and **Zarr** (``.zarr``).
- Vendor electron/X-ray microscopy file loading via optional **RosettaSciIO**
  (``.hspy``, ``.emd``, ``.dm3``/``.dm4``, ``.bcf``, ``.mrc``, ``.ser``).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
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

    raise ValueError(
        f"Unsupported output extension for '{out_path}'. "
        "Supported: .ome.tif, .tif, .h5, .hdf5, .nxs, .zarr."
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
    """Load an exported X-FUSE report from ``.h5``, ``.zarr``, or ``.ome.tif``."""
    in_path = Path(path)
    suffix = in_path.suffix.lower()
    lower_name = in_path.name.lower()

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
        if (
            isinstance(spatial_shape, list)
            and len(spatial_shape) == 3
            and stack.ndim == 3
            and spatial_shape[0] == 1
        ):
            # Restore Z=1 axis squeezed by OME-TIFF reader on single-slice 3D reports
            stack = stack[:, np.newaxis, :, :]

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
    """Load a microscopy dataset (``.tif``, ``.npy``, ``.h5``, or vendor formats
    such as ``.emd``, ``.dm3``/``.dm4``, ``.bcf``, ``.hspy`` via RosettaSciIO).

    Returns:
        ``(data_float32, metadata_dict)``.
    """
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Microscopy file not found: {in_path}")

    suffix = in_path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(in_path).astype(np.float32)
        return arr, {"format": "npy", "shape": list(arr.shape)}

    if suffix in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
        try:
            import tifffile  # type: ignore[import-not-found]

            if suffix in (".tif", ".tiff"):
                arr = tifffile.imread(in_path).astype(np.float32)
                return arr, {"format": "tiff", "shape": list(arr.shape)}
        except (ImportError, ValueError):
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
