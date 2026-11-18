"""SuRVoS-inspired 2.5D volumetric correlative fusion, orthogonal tri-plane
inference, inter-plane epistemic uncertainty, lazy HDF5/Zarr/TIFF slice
streaming, and sparse-slice 2D -> 3D chemical propagation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from .fusion import CorrelativeFuser
from .loaders import _natural_sort_key, load_image_array
from .transforms import (
    build_default_tta,
    ensure_spatial_divisible,
    normalize_structural,
)
from .uncertainty import (
    WelfordAccumulator,
    _safe_quantile,
    compute_posterior_uncertainty,
    match_coarse_priors,
)
from .utils import guide_to_tensor


@dataclass
class VolumetricUncertaintyReport:
    """3D phase/element probability volume with volumetric & physical certainty maps."""

    probabilities: torch.Tensor  # (K, Z, Y, X) calibrated 3D probabilities
    hard_mask: torch.Tensor  # (Z, Y, X) int64 phase assignment
    aleatoric_entropy: torch.Tensor  # (Z, Y, X) normalized posterior entropy [0, 1]
    triplane_variance: torch.Tensor  # (Z, Y, X) inter-plane / TTA epistemic variance
    physical_residual: torch.Tensor  # (Z, Y, X) 3D coarse forward-model residual
    combined_uncertainty: torch.Tensor  # (Z, Y, X) composite 3D uncertainty [0, 1]
    z_slice_consistency: float  # Mean adjacent-slice IoU along Z in [0, 1]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    prediction_sets: torch.Tensor | None = (
        None  # Optional (K, Z, Y, X) bool (1-alpha) conformal phase set
    )
    conformal_interval_halfwidth: torch.Tensor | None = (
        None  # Optional (M, Z, Y, X) calibrated coarse physical error halfwidth
    )


class VolumeSliceAccessor:
    """Uniform lazy 3D ``(Z, Y, X)`` slice reader supporting in-memory arrays,
    ``torch.Tensor``, ``h5py.Dataset``, ``zarr.Array``, 3D ``.npy``/``.tif``/``.h5``/
    ``.zarr`` file paths, and sorted 2D image slice directories without
    materializing full volumes in RAM.
    """

    def __init__(self, source: Any) -> None:
        self._kind: str
        self._data: Any
        self._slice_paths: list[Path] = []
        self._h5_path: Path | None = None
        self._h5_key: str | None = None

        if isinstance(source, VolumeSliceAccessor):
            self._kind = source._kind
            self._data = source._data
            self._slice_paths = list(source._slice_paths)
            self._h5_path = source._h5_path
            self._h5_key = source._h5_key
            self.shape: tuple[int, int, int] = source.shape
            return

        if isinstance(source, (str, Path)):
            p = Path(source)
            sfx = p.suffix.lower()
            if sfx == ".zarr" and p.exists():
                import zarr  # type: ignore[import-not-found]

                opened: Any = zarr.open(str(p), mode="r")
                if hasattr(opened, "shape") and len(opened.shape) == 3:
                    self._kind = "lazy"
                    self._data = opened
                    self.shape = (
                        int(opened.shape[0]),
                        int(opened.shape[1]),
                        int(opened.shape[2]),
                    )
                    return
                if hasattr(opened, "keys"):
                    for k in opened.keys():
                        cand = opened[k]
                        if hasattr(cand, "shape") and len(cand.shape) == 3:
                            self._kind = "lazy"
                            self._data = cand
                            self.shape = (
                                int(cand.shape[0]),
                                int(cand.shape[1]),
                                int(cand.shape[2]),
                            )
                            return
                raise ValueError(f"No 3D array found in Zarr store {p}.")
            if p.is_file() and sfx in (".h5", ".hdf5", ".nxs"):
                import h5py

                found_key: str | None = None
                found_shape: tuple[int, int, int] | None = None
                with h5py.File(p, "r") as hf:

                    def _visit(name: str, obj: Any) -> None:
                        nonlocal found_key, found_shape
                        if (
                            found_key is None
                            and isinstance(obj, h5py.Dataset)
                            and len(obj.shape) == 3
                        ):
                            found_key = name
                            found_shape = (
                                int(obj.shape[0]),
                                int(obj.shape[1]),
                                int(obj.shape[2]),
                            )

                    hf.visititems(_visit)
                if found_key is None or found_shape is None:
                    raise ValueError(f"No 3D dataset found in HDF5 file {p}.")
                self._kind = "h5_file"
                self._data = None
                self._h5_path = p
                self._h5_key = found_key
                self.shape = found_shape
                return
            if p.is_file() and sfx == ".npy":
                mm = np.load(p, mmap_mode="r")
                if mm.ndim != 3:
                    raise ValueError(f"Expected 3D .npy volume, got {mm.shape}.")
                self._kind = "lazy"
                self._data = mm
                self.shape = (int(mm.shape[0]), int(mm.shape[1]), int(mm.shape[2]))
                return
            if p.is_file() and sfx in (".tif", ".tiff"):
                import tifffile  # type: ignore[import-not-found]

                arr_tif = tifffile.imread(p).astype(np.float32)
                if arr_tif.ndim != 3:
                    raise ValueError(f"Expected 3D TIFF stack, got {arr_tif.shape}.")
                self._kind = "ndarray"
                self._data = arr_tif
                self.shape = (
                    int(arr_tif.shape[0]),
                    int(arr_tif.shape[1]),
                    int(arr_tif.shape[2]),
                )
                return
            if p.is_dir():
                exts = {".tif", ".tiff", ".png", ".npy"}
                files = sorted(
                    [f for f in p.iterdir() if f.suffix.lower() in exts],
                    key=lambda item: _natural_sort_key(item.name),
                )
                if not files:
                    raise ValueError(f"No 2D slice images found in directory {p}.")
                first = load_image_array(files[0])
                if first.ndim != 2:
                    raise ValueError(
                        f"Expected 2D slice images in {p}, got shape {first.shape}."
                    )
                self._kind = "dir"
                self._data = None
                self._slice_paths = files
                self.shape = (len(files), int(first.shape[0]), int(first.shape[1]))
                return
            raise ValueError(
                f"Unsupported path source {p}; pass an open h5py.Dataset, zarr.Array, "
                "3D file (.h5/.zarr/.npy/.tif), 3D array, or directory of 2D slices."
            )

        if isinstance(source, torch.Tensor):
            arr = source.detach().cpu().float().numpy()
            if arr.ndim != 3:
                raise ValueError(f"Expected 3D volume (Z, Y, X), got {arr.shape}.")
            self._kind = "ndarray"
            self._data = arr
            self.shape = (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))
            return

        if isinstance(source, np.ndarray):
            if source.ndim != 3:
                raise ValueError(f"Expected 3D volume (Z, Y, X), got {source.shape}.")
            self._kind = "ndarray"
            self._data = source
            self.shape = (
                int(source.shape[0]),
                int(source.shape[1]),
                int(source.shape[2]),
            )
            return

        # Duck-typed lazy 3D dataset (h5py.Dataset, zarr.Array, memmap, dask, etc.)
        if hasattr(source, "shape") and hasattr(source, "__getitem__"):
            shp = tuple(int(s) for s in source.shape)
            if len(shp) != 3:
                raise ValueError(f"Expected 3D volume (Z, Y, X), got shape {shp}.")
            self._kind = "lazy"
            self._data = source
            self.shape = (shp[0], shp[1], shp[2])
            return

        raise TypeError(f"Unsupported 3D volume source of type {type(source)}.")

    @property
    def is_lazy(self) -> bool:
        return self._kind in ("lazy", "dir", "h5_file")

    def get_plane(self, axis: int, index: int) -> np.ndarray:
        """Fetch a single 2D float32 plane along ``axis`` (0=Z, 1=Y, 2=X)."""
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or 2; got {axis}.")
        depth = self.shape[axis]
        if not (0 <= index < depth):
            raise IndexError(
                f"Slice index {index} out of bounds for axis {axis} of size {depth}."
            )

        if self._kind == "h5_file":
            import h5py

            assert self._h5_path is not None and self._h5_key is not None
            with h5py.File(self._h5_path, "r") as hf:
                ds = hf[self._h5_key]
                if axis == 0:
                    return np.asarray(ds[index, :, :], dtype=np.float32)
                if axis == 1:
                    return np.asarray(ds[:, index, :], dtype=np.float32)
                return np.asarray(ds[:, :, index], dtype=np.float32)

        if self._kind == "dir":
            if axis == 0:
                return load_image_array(self._slice_paths[index])
            if axis == 1:
                rows = [load_image_array(fp)[index, :] for fp in self._slice_paths]
                return np.stack(rows, axis=0).astype(np.float32)
            cols = [load_image_array(fp)[:, index] for fp in self._slice_paths]
            return np.stack(cols, axis=0).astype(np.float32)

        if axis == 0:
            return np.asarray(self._data[index, :, :], dtype=np.float32)
        if axis == 1:
            return np.asarray(self._data[:, index, :], dtype=np.float32)
        return np.asarray(self._data[:, :, index], dtype=np.float32)

    def estimate_robust_range(
        self, percentiles: tuple[float, float] = (0.5, 99.5)
    ) -> tuple[float, float]:
        """Compute global intensity percentiles via strided subsampling so lazy
        multi-gigabyte HDF5/Zarr stacks are never read into memory in full.
        """
        z_dim, y_dim, x_dim = self.shape
        sz = max(1, z_dim // 32)
        sy = max(1, y_dim // 64)
        sx = max(1, x_dim // 64)

        if self._kind == "ndarray":
            sub = np.asarray(self._data[::sz, ::sy, ::sx], dtype=np.float32)
        else:
            samples = [
                self.get_plane(0, z_i)[::sy, ::sx] for z_i in range(0, z_dim, sz)
            ]
            sub = np.stack(samples, axis=0)

        finite = sub[np.isfinite(sub)]
        if finite.size == 0:
            return 0.0, 1.0
        lo, hi = np.percentile(finite, percentiles)
        if hi - lo < 1e-8:
            lo, hi = float(np.min(finite)), float(np.max(finite))
        return float(lo), float(hi)


def pack_25d_slice(
    volume: Any,
    index: int,
    axis: int = 0,
    delta: int = 1,
) -> np.ndarray:
    """Pack a 3-slice 2.5D structural slab ``(index - delta, index, index + delta)``
    along ``axis`` (``0=Z/XY``, ``1=Y/XZ``, ``2=X/YZ``) into a 3-channel
    ``(3, H_plane, W_plane)`` float32 array with reflection clamping at boundaries.
    """
    accessor = VolumeSliceAccessor(volume)
    depth = accessor.shape[axis]
    if not (0 <= index < depth):
        raise IndexError(f"Slice index {index} out of bounds for axis size {depth}.")

    def _clamp_reflect(idx: int) -> int:
        if depth <= 1:
            return 0
        if idx < 0:
            idx = -idx
        if idx >= depth:
            idx = 2 * (depth - 1) - idx
        return int(np.clip(idx, 0, depth - 1))

    indices = [
        _clamp_reflect(index - delta),
        index,
        _clamp_reflect(index + delta),
    ]
    # Cache repeated indices (e.g. boundary clamping) to avoid duplicate I/O
    plane_cache: dict[int, np.ndarray] = {}
    planes: list[np.ndarray] = []
    for idx in indices:
        if idx not in plane_cache:
            plane_cache[idx] = accessor.get_plane(axis, idx)
        planes.append(plane_cache[idx])

    slab = np.stack(planes, axis=0)
    return np.ascontiguousarray(slab, dtype=np.float32)


def z_slice_consistency_score(mask_3d: np.ndarray | torch.Tensor) -> float:
    """Measure 3D inter-slice structural continuity (mean foreground IoU between
    adjacent slices ``z`` and ``z + 1``) to quantify elimination of 2D slice
    'pancaking' artifacts."""
    arr = (
        mask_3d.detach().cpu().numpy()
        if isinstance(mask_3d, torch.Tensor)
        else np.asarray(mask_3d)
    )
    if arr.ndim != 3 or arr.shape[0] < 2:
        return 1.0

    fg_classes = [int(c) for c in np.unique(arr) if c > 0] or [1]
    pair_ious = []
    for z in range(arr.shape[0] - 1):
        s0, s1 = arr[z], arr[z + 1]
        cls_ious = []
        for cls in fg_classes:
            m0 = s0 == cls
            m1 = s1 == cls
            union = int((m0 | m1).sum())
            inter = int((m0 & m1).sum())
            if union > 0:
                cls_ious.append(inter / union)
        if cls_ious:
            pair_ious.append(float(np.mean(cls_ious)))
    return float(np.mean(pair_ious)) if pair_ious else 1.0


class VolumetricFuser:
    """SuRVoS-style 2.5D volumetric correlative fusion engine supporting
    in-memory arrays and lazy HDF5/Zarr/TIFF-stack streaming.
    """

    def __init__(
        self,
        fuser: CorrelativeFuser,
        tta_shifts: Sequence[int] | None = (1,),
        tta_flips: bool = True,
    ) -> None:
        self.fuser = fuser
        self.fwd_tta, self.inv_tta = build_default_tta(
            shift_dists=list(tta_shifts) if tta_shifts is not None else [],
            pattern="Neumann",
            include_flips=tta_flips,
        )
        self._global_prior: torch.Tensor | None = None

    def _can_tile_head(self) -> bool:
        return self.fuser.xrd_fuse_method in ("learned_gating", "spatial_mlp")

    def _prepare_slab_tensor(
        self,
        accessor: VolumeSliceAccessor,
        index: int,
        axis: int,
        delta: int,
        vmin: float,
        vmax: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        slab = pack_25d_slice(accessor, index=index, axis=axis, delta=delta)
        slab_clean = np.nan_to_num(slab, nan=vmin, posinf=vmax, neginf=vmin)
        slab_clipped = np.clip(slab_clean, vmin, vmax)
        if vmax - vmin > 1e-8:
            slab_01 = (slab_clipped - vmin) / (vmax - vmin)
        else:
            slab_01 = np.zeros_like(slab_clipped)
        t = normalize_structural(slab_01, imagenet_norm=True, device=device)
        if (
            self.fuser.tile_size is not None
            and max(t.shape[-2], t.shape[-1]) > self.fuser.tile_size
            and self._can_tile_head()
        ):
            return t, (int(t.shape[-2]), int(t.shape[-1]))
        padded, orig_hw = ensure_spatial_divisible(
            t, self.fuser.original_patch_size, self.fuser.stride[0]
        )
        return padded, orig_hw

    def fit_sparse_slices(
        self,
        hr_volume: Any,
        sparse_guides: dict[int, np.ndarray | torch.Tensor],
        epochs: int = 12,
        delta: int = 1,
        axis: int = 0,
    ) -> list[float]:
        """Train the ``CorrelativeFuser`` head on sparse 2D guided slices
        ``{slice_idx: guide_map}`` using 2.5D ``(z - delta, z, z + delta)`` slabs."""
        if not sparse_guides:
            raise ValueError("sparse_guides must contain at least one guided slice.")

        accessor = VolumeSliceAccessor(hr_volume)
        vmin, vmax = accessor.estimate_robust_range()
        dev = next(self.fuser.dinov2.parameters()).device
        is_spectral = self.fuser.spectral_channels is not None
        norm_spec: Any = getattr(
            self.fuser,
            "_default_guide_norm",
            "preserve_counts" if self.fuser.loss_fn == "forward_psf_poisson" else True,
        )

        prepared_guides: dict[int, torch.Tensor] = {}
        prepared_slabs: dict[int, torch.Tensor] = {}
        orig_hws: dict[int, tuple[int, int]] = {}
        cal_masks: dict[int, torch.Tensor | None] = {}
        train_masks: dict[int, torch.Tensor | None] = {}
        prior_accum = []
        for idx, g_raw in sparse_guides.items():
            s_i = int(idx)
            g_t = guide_to_tensor(g_raw, device=dev, normalize=norm_spec)
            prepared_guides[s_i] = g_t
            slab_t, orig_hw = self._prepare_slab_tensor(
                accessor, s_i, axis=axis, delta=delta, vmin=vmin, vmax=vmax, device=dev
            )
            prepared_slabs[s_i] = slab_t
            orig_hws[s_i] = orig_hw
            lr_hw = (int(g_t.shape[-2]), int(g_t.shape[-1]))
            if self.fuser.calibrator is not None and lr_hw[0] * lr_hw[1] >= 16:
                cm = self.fuser.calibrator.make_spatial_holdout_mask(
                    lr_hw, holdout_fraction=0.25, seed=42 + s_i
                ).to(dev)
                cal_masks[s_i] = cm
                train_masks[s_i] = ~cm
            else:
                cal_masks[s_i] = None
                train_masks[s_i] = None
            if not is_spectral:
                g_unit = guide_to_tensor(g_raw, device=dev, normalize=True)
                prior_accum.append(g_unit.mean(dim=(-2, -1)))

        if prior_accum:
            self._global_prior = torch.stack(prior_accum, dim=0).mean(dim=0)
        # Ensure all sparse slices fit in the TTA feature cache across epochs
        self.fuser._max_cache_entries = max(
            self.fuser._max_cache_entries, len(prepared_slabs) + 2
        )

        self.fuser.train()
        self.fuser.dinov2.eval()
        epoch_losses: list[float] = []
        slice_keys = sorted(prepared_guides.keys())
        n_ep = max(1, epochs)
        warmup_ep = int(round(0.3 * n_ep)) if self.fuser.align_guide != "none" else 0
        for ep in range(n_ep):
            step_losses = []
            for s_idx in slice_keys:
                self.fuser.set_guide_transforms(
                    prepared_guides[s_idx],
                    self.fwd_tta,
                    self.inv_tta,
                    spectral_unmix=is_spectral,
                    n_components=self.fuser.guide_channels,
                )
                loss_val = self.fuser.train_fusion_step(
                    prepared_slabs[s_idx],
                    update_aligner=(ep >= warmup_ep),
                    valid_hr_hw=orig_hws[s_idx],
                    loss_mask_lr=train_masks[s_idx],
                )
                step_losses.append(loss_val)
            epoch_losses.append(float(np.mean(step_losses)))
        self.fuser.eval()

        # Calibrate zero-label conformal sets on held-out blocks of first guided slice
        if self.fuser.calibrator is not None and slice_keys:
            s0 = slice_keys[0]
            self.fuser.set_guide_transforms(
                prepared_guides[s0],
                self.fwd_tta,
                self.inv_tta,
                spectral_unmix=is_spectral,
                n_components=self.fuser.guide_channels,
            )
            rep0 = self.fuser.predict_with_uncertainty(
                prepared_slabs[s0], mode="multiclass"
            ).crop_spatial(*orig_hws[s0])
            ref_g0 = self.fuser._effective_reference_guide(prepared_guides[s0])
            self.fuser.calibrator.calibrate_coarse(
                probs_hr=rep0.probabilities,
                guide_lr=ref_g0,
                epistemic_var_hr=rep0.epistemic_variance,
                cal_mask_lr=cal_masks[s0],
                downsampler=self.fuser._active_downsampler(),
                aligner=self.fuser.guide_aligner,
                mode="multiclass",
            )
        return epoch_losses

    def _predict_single_plane_stack(
        self,
        accessor: VolumeSliceAccessor,
        axis: int,
        delta: int,
        vmin: float,
        vmax: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run slice-by-slice 2.5D logit inference along ``axis``, returning
        ``(mean_logits_zyx, tta_prob_var_zyx)`` transposed back to ``(M, Z, Y, X)``."""
        depth = accessor.shape[axis]
        n_guides = self.fuser.guide_channels
        z_dim, y_dim, x_dim = accessor.shape

        plane_logits = []
        plane_vars = []
        dummy_guide = (
            self._global_prior.to(device).unsqueeze(-1).unsqueeze(-1)
            if self._global_prior is not None
            else torch.full((1, n_guides, 4, 4), 0.5, device=device)
        )
        self.fuser.set_guide_transforms(
            dummy_guide,
            self.fwd_tta,
            self.inv_tta,
            spectral_unmix=(self.fuser.spectral_channels is not None),
            n_components=n_guides,
        )

        for idx in range(depth):
            slab_t, (orig_h, orig_w) = self._prepare_slab_tensor(
                accessor,
                idx,
                axis=axis,
                delta=delta,
                vmin=vmin,
                vmax=vmax,
                device=device,
            )
            if (
                self.fuser.tile_size is not None
                and max(slab_t.shape[-2], slab_t.shape[-1]) > self.fuser.tile_size
                and self._can_tile_head()
            ):
                from .tiling import run_tiled_2d

                def _tile_fn(tile_chw: torch.Tensor) -> torch.Tensor:
                    ml, vp = self.fuser._predict_logits_and_var_tile(
                        tile_chw, dummy_guide
                    )
                    return torch.cat([ml, vp], dim=1)

                packed = run_tiled_2d(
                    _tile_fn,
                    slab_t,
                    tile_size=self.fuser.tile_size,
                    overlap=self.fuser.tile_overlap,
                    patch_size=self.fuser.original_patch_size,
                    stride=self.fuser.stride[0],
                )
                m_l = packed[:, :n_guides, :orig_h, :orig_w]
                v_p = packed[:, n_guides:, :orig_h, :orig_w]
            else:
                m_l_pad, v_p_pad = self.fuser._predict_logits_and_var_tile(
                    slab_t, dummy_guide
                )
                m_l = m_l_pad[:, :, :orig_h, :orig_w]
                v_p = v_p_pad[:, :, :orig_h, :orig_w]

            plane_logits.append(m_l.squeeze(0).cpu())
            plane_vars.append(v_p.squeeze(0).cpu())

        stacked_l = torch.stack(plane_logits, dim=1)  # (M, depth, H_p, W_p)
        stacked_v = torch.stack(plane_vars, dim=1)

        if axis == 0:
            return stacked_l, stacked_v
        if axis == 1:
            assert stacked_l.shape[1:] == (y_dim, z_dim, x_dim)
            return stacked_l.permute(0, 2, 1, 3), stacked_v.permute(0, 2, 1, 3)
        assert stacked_l.shape[1:] == (x_dim, z_dim, y_dim)
        return stacked_l.permute(0, 2, 3, 1), stacked_v.permute(0, 2, 3, 1)

    @staticmethod
    def _normalize_guide_3d(g3_in: np.ndarray | torch.Tensor) -> torch.Tensor:
        g3 = (
            g3_in.detach().cpu().float()
            if isinstance(g3_in, torch.Tensor)
            else torch.from_numpy(np.asarray(g3_in, dtype=np.float32))
        )
        g3 = torch.nan_to_num(g3, nan=0.0, posinf=0.0, neginf=0.0)
        if g3.ndim == 3:
            g3 = g3.unsqueeze(0)
        if float(g3.max().item()) > 1.5:
            g_max = g3.flatten(1).max(dim=1).values.view(-1, 1, 1, 1).clamp(min=1e-6)
            return (g3 / g_max).clamp(0.0, 1.0)
        return g3.clamp(0.0, 1.0)

    @torch.no_grad()
    def predict_volume(
        self,
        hr_volume: Any,
        mode: Literal["2d", "2.5d_slab", "triplane"] = "2.5d_slab",
        delta: int = 1,
        coarse_guide_3d: np.ndarray | torch.Tensor | None = None,
        sparse_guides: dict[int, np.ndarray | torch.Tensor] | None = None,
        pred_mode: Literal["multiclass", "multilabel"] = "multilabel",
        calibrate_to_guide: bool = True,
    ) -> VolumetricUncertaintyReport:
        """Segment a full 3D structural volume ``(Z, Y, X)`` (in-memory or lazy
        HDF5/Zarr/TIFF directory) using ``"2d"``, ``"2.5d_slab"``, or ``"triplane"``.
        """
        self.fuser.eval()
        accessor = VolumeSliceAccessor(hr_volume)
        z_dim, y_dim, x_dim = accessor.shape
        vmin, vmax = accessor.estimate_robust_range()
        dev = next(self.fuser.dinov2.parameters()).device

        eff_delta = 0 if mode == "2d" else max(1, delta)
        axes = (0, 1, 2) if mode == "triplane" else (0,)

        plane_welford = WelfordAccumulator()
        tta_var_accum = torch.zeros(
            (self.fuser.guide_channels, z_dim, y_dim, x_dim), dtype=torch.float32
        )
        mean_logits_sum = torch.zeros_like(tta_var_accum)

        for ax in axes:
            l_zyx, v_zyx = self._predict_single_plane_stack(
                accessor, axis=ax, delta=eff_delta, vmin=vmin, vmax=vmax, device=dev
            )
            mean_logits_sum += l_zyx
            tta_var_accum += v_zyx
            plane_welford.update(torch.sigmoid(l_zyx))

        mean_logits = mean_logits_sum / len(axes)  # (M, Z, Y, X)
        _, interplane_var = plane_welford.finalize()  # (M, Z, Y, X)
        epi_var_m = interplane_var + (tta_var_accum / len(axes))
        epi_var = epi_var_m.mean(dim=0)  # (Z, Y, X)

        n_guides = mean_logits.shape[0]
        flat_logits = mean_logits.reshape(1, n_guides, z_dim, y_dim * x_dim)
        g3_norm: torch.Tensor | None = None
        if coarse_guide_3d is not None:
            g3_norm = self._normalize_guide_3d(coarse_guide_3d)
            prior_ref: torch.Tensor | None = g3_norm.reshape(1, g3_norm.shape[0], -1, 1)
        elif sparse_guides:
            means = [
                guide_to_tensor(g, normalize=True).mean(dim=(-2, -1))
                for g in sparse_guides.values()
            ]
            prior_ref = (
                torch.stack(means, dim=0).mean(dim=0).unsqueeze(-1).unsqueeze(-1)
            )
        elif self._global_prior is not None:
            prior_ref = self._global_prior.cpu().unsqueeze(-1).unsqueeze(-1)
        else:
            prior_ref = None

        phys = self.fuser._active_physics()
        if pred_mode == "multiclass":
            if self.fuser.loss_fn == "compositional_kl" and phys is not None:
                simplex_flat = phys.logits_to_simplex_probs(flat_logits.to(dev)).cpu()
                n_k = int(simplex_flat.shape[1])
                probs_3d = simplex_flat.squeeze(0).reshape(n_k, z_dim, y_dim, x_dim)
                fg_probs_3d = probs_3d[1:] if n_k == n_guides + 1 else probs_3d
            else:
                bg_l = torch.zeros((1, 1, z_dim, y_dim * x_dim), dtype=torch.float32)
                full_l = torch.cat([bg_l, flat_logits], dim=1)
                probs_flat = (
                    match_coarse_priors(full_l, prior_ref, mode="multiclass")
                    if (calibrate_to_guide and prior_ref is not None)
                    else F.softmax(full_l, dim=1)
                )
                probs_3d = probs_flat.squeeze(0).reshape(
                    n_guides + 1, z_dim, y_dim, x_dim
                )
                fg_probs_3d = probs_3d[1:]
        else:
            probs_flat = (
                match_coarse_priors(flat_logits, prior_ref, mode="multilabel")
                if (calibrate_to_guide and prior_ref is not None)
                else torch.sigmoid(flat_logits)
            )
            probs_3d = probs_flat.squeeze(0).reshape(n_guides, z_dim, y_dim, x_dim)
            fg_probs_3d = probs_3d

        entropy_flat, _ = compute_posterior_uncertainty(
            probs_3d.reshape(1, probs_3d.shape[0], z_dim, y_dim * x_dim),
            mode=pred_mode,
        )
        entropy_3d = entropy_flat.squeeze(0).reshape(z_dim, y_dim, x_dim)

        phys_res_3d = torch.zeros((z_dim, y_dim, x_dim), dtype=torch.float32)
        if g3_norm is not None:
            out_3d = (
                int(g3_norm.shape[-3]),
                int(g3_norm.shape[-2]),
                int(g3_norm.shape[-1]),
            )
            pred_lr_3d = F.adaptive_avg_pool3d(
                fg_probs_3d.unsqueeze(0), output_size=out_3d
            ).squeeze(0)
            n_cmp = min(int(pred_lr_3d.shape[0]), int(g3_norm.shape[0]))
            res_lr = (
                (pred_lr_3d[:n_cmp] - g3_norm[:n_cmp])
                .abs()
                .mean(dim=0, keepdim=True)
                .unsqueeze(0)
            )
            phys_res_3d = (
                F.interpolate(
                    res_lr,
                    size=(z_dim, y_dim, x_dim),
                    mode="trilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
            )
        elif sparse_guides:
            for s_idx, g_s in sparse_guides.items():
                if 0 <= int(s_idx) < z_dim:
                    g_t = guide_to_tensor(g_s, normalize=True).cpu()
                    out_2d = (int(g_t.shape[-2]), int(g_t.shape[-1]))
                    p_s = F.adaptive_avg_pool2d(
                        fg_probs_3d[:, int(s_idx)].unsqueeze(0),
                        output_size=out_2d,
                    )
                    n_cmp = min(int(p_s.shape[1]), int(g_t.shape[1]))
                    res_s = (
                        (p_s[:, :n_cmp] - g_t[:, :n_cmp])
                        .abs()
                        .mean(dim=1, keepdim=True)
                    )
                    phys_res_3d[int(s_idx)] = (
                        F.interpolate(
                            res_s,
                            size=(y_dim, x_dim),
                            mode="bilinear",
                            align_corners=False,
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )

        p99 = _safe_quantile(epi_var, 0.99).clamp(min=1e-6)
        epi_norm = (epi_var / p99).clamp(0.0, 1.0)
        combined_3d = (0.45 * entropy_3d + 0.30 * epi_norm + 0.25 * phys_res_3d).clamp(
            0.0, 1.0
        )

        if pred_mode == "multiclass" and probs_3d.shape[0] >= 2:
            hard_3d = probs_3d.argmax(dim=0).long()
        else:
            hard_3d = (probs_3d[0] >= 0.5).long()

        pred_sets_3d: torch.Tensor | None = None
        conf_hw_3d: torch.Tensor | None = None
        if self.fuser.calibrator is not None:
            ps_flat, hw_flat = self.fuser.calibrator.predict_sets_and_intervals(
                probs_hr=probs_3d.reshape(1, probs_3d.shape[0], z_dim, y_dim * x_dim),
                guide_lr=None,
                epistemic_var_hr=epi_var.reshape(1, z_dim, y_dim * x_dim),
                mode=pred_mode,
            )
            pred_sets_3d = ps_flat.squeeze(0).reshape(
                ps_flat.shape[1], z_dim, y_dim, x_dim
            )
            if hw_flat is not None:
                conf_hw_3d = hw_flat.squeeze(0).reshape(
                    hw_flat.shape[1], z_dim, y_dim, x_dim
                )

        z_cons = z_slice_consistency_score(hard_3d)
        return VolumetricUncertaintyReport(
            probabilities=probs_3d,
            hard_mask=hard_3d,
            aleatoric_entropy=entropy_3d,
            triplane_variance=epi_var,
            physical_residual=phys_res_3d,
            combined_uncertainty=combined_3d,
            z_slice_consistency=z_cons,
            diagnostics={
                "volumetric_mode": mode,
                "delta": eff_delta,
                "volume_shape": [z_dim, y_dim, x_dim],
                "lazy_streaming": accessor.is_lazy,
            },
            prediction_sets=pred_sets_3d,
            conformal_interval_halfwidth=conf_hw_3d,
        )
