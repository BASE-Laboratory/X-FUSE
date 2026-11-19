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
    compute_hard_phase_mask,
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
        self._sparse_guides_norm: dict[int, torch.Tensor] = {}
        self._last_fit_diagnostics: dict[str, Any] = {}

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
        t = normalize_structural(
            slab_clean,
            imagenet_norm=True,
            device=device,
            value_range=(vmin, vmax),
        )
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

    def _forward_slab_logits_grad(
        self,
        slab_t: torch.Tensor,
        orig_hw: tuple[int, int],
        guide_ref: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Differentiably predict canonical HR logits ``(1, M, H_pad, W_pad)``
        ensembled across TTA views while reusing cached frozen ViT features."""
        from .heads import SpatialConvMLPHead

        _, img_h, img_w = slab_t.shape
        kind, cached_views = self.fuser._extract_tta_views(slab_t, want_hr=True)
        img_batch: torch.Tensor | None = None
        if kind == "patch" or self.fuser.xrd_fuse_method == "spatial_mlp":
            x_in = (
                slab_t.to(self.fuser.dtype)
                if self.fuser.dtype != torch.float32
                else slab_t
            )
            with torch.no_grad():
                img_batch = self.fuser.get_transformed_input_batch(
                    x_in, self.fuser.transforms
                )

        n_tr = len(self.fuser.transforms)
        valid_hw = orig_hw if orig_hw != (img_h, img_w) else None
        logits_acc: torch.Tensor | None = None
        for i in range(n_tr):
            tr_img = img_batch[i].unsqueeze(0) if img_batch is not None else None
            feat_in = (
                self.fuser._upsample_patch_map(
                    cached_views[i].float(), (img_h, img_w), guidance_img=tr_img
                )
                if (kind == "patch" and tr_img is not None)
                else cached_views[i]
            )
            if self.fuser.xrd_fuse_method in ("learned_gating", "spatial_mlp"):
                head_mod = (
                    self.fuser.learned_gating
                    if self.fuser.xrd_fuse_method == "learned_gating"
                    else self.fuser.spatial_mlp
                )
                assert head_mod is not None
                l_tr = head_mod.predict_logits(feat_in, guidance_hr=tr_img)
                if l_tr.shape[-2:] != (img_h, img_w):
                    l_tr = F.interpolate(
                        l_tr,
                        size=(img_h, img_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                l_can = self.fuser.inverse_transforms[i](l_tr)
                if isinstance(head_mod, SpatialConvMLPHead) and guide_ref is not None:
                    l_can = head_mod.apply_guide_residual(
                        l_can, guide_ref, valid_hr_hw=valid_hw
                    )
            else:
                assert self.fuser.spatial_attention is not None
                if feat_in.shape[-2:] != (img_h, img_w):
                    feat_in = F.interpolate(
                        feat_in.float(),
                        size=(img_h, img_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                f_can = self.fuser.inverse_transforms[i](feat_in)
                g_att = (
                    guide_ref
                    if guide_ref is not None
                    else torch.full(
                        (1, self.fuser.guide_channels, 4, 4),
                        0.5,
                        device=slab_t.device,
                    )
                )
                l_can = self.fuser.spatial_attention.predict_logits(f_can, g_att)
            logits_acc = l_can if logits_acc is None else (logits_acc + l_can)
        assert logits_acc is not None
        return logits_acc / float(max(n_tr, 1))

    def _logits_to_probs_grad(
        self, logits: torch.Tensor, target_hint: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, str]:
        phys = self.fuser._active_physics()
        if phys is not None and phys.loss_fn == "compositional_kl":
            sp = phys.logits_to_simplex_probs(logits.float(), target=target_hint)
            tag = "simplex_bg" if sp.shape[1] == phys.n_channels + 1 else "simplex_full"
            return sp, tag
        return torch.sigmoid(logits.float()), "sigmoid"

    @staticmethod
    def _probs_to_effective_logits(
        probs: torch.Tensor, mode_tag: str, eps: float = 1e-6
    ) -> torch.Tensor:
        p = probs.clamp(eps, 1.0 - eps)
        if mode_tag == "simplex_bg":
            return torch.log(p[:, 1:]) - torch.log(p[:, :1])
        if mode_tag == "simplex_full":
            return torch.log(p)
        return torch.logit(p, eps=eps)

    def _depth_psf_slice_step(
        self,
        s_idx: int,
        canon_guide: torch.Tensor,
        center_slab: torch.Tensor,
        orig_hw: tuple[int, int],
        loss_mask_lr: torch.Tensor | None,
        stencil_slabs: dict[int, tuple[torch.Tensor, tuple[int, int]]],
        sparse_norm_cpu: dict[int, torch.Tensor],
        depth_ax: int,
        axis: int,
        stencil_radius: int,
        z_step: int,
        sigma_z: torch.Tensor,
        update_aligner: bool,
        dev: torch.device,
    ) -> float:
        assert self.fuser.fusion_optimizer is not None
        self.fuser.fusion_optimizer.zero_grad()
        if self.fuser.align_optimizer is not None:
            self.fuser.align_optimizer.zero_grad()

        g_ref = canon_guide.unsqueeze(0) if canon_guide.ndim == 3 else canon_guide
        dz = torch.tensor(
            [float(j * z_step) for j in range(-stencil_radius, stencil_radius + 1)],
            device=dev,
            dtype=torch.float32,
        )
        w_z = torch.exp(-0.5 * (dz / sigma_z) ** 2)
        w_z = w_z / w_z.sum().clamp(min=1e-8)

        p_bar: torch.Tensor | None = None
        l_center: torch.Tensor | None = None
        mode_tag = "sigmoid"
        for j_i, j in enumerate(range(-stencil_radius, stencil_radius + 1)):
            z_nb = int(np.clip(s_idx + j * z_step, 0, depth_ax - 1))
            nb_slab, nb_hw = stencil_slabs[z_nb]
            g_nb = self._resolve_slice_guide_2d(
                idx=z_nb,
                depth=depth_ax,
                axis=axis,
                coarse_guide_3d_norm=None,
                sparse_guides_norm=sparse_norm_cpu or None,
                fallback_guide=g_ref,
                device=dev,
            )
            l_nb = self._forward_slab_logits_grad(nb_slab, nb_hw, guide_ref=g_nb)
            if j == 0:
                l_center = l_nb
            p_nb, mode_tag = self._logits_to_probs_grad(l_nb, target_hint=g_ref)
            w_p = w_z[j_i] * p_nb
            p_bar = w_p if p_bar is None else (p_bar + w_p)

        assert p_bar is not None
        logits_eff = self._probs_to_effective_logits(p_bar, mode_tag)
        aligner_arg = (
            self.fuser.guide_aligner if self.fuser.align_guide != "none" else None
        )
        valid_hw = orig_hw if orig_hw != tuple(center_slab.shape[-2:]) else None
        if self.fuser.xrd_fuse_method == "spatial_mlp":
            from .heads import edge_aware_tv_loss

            assert self.fuser.spatial_mlp is not None
            loss_t = self.fuser.spatial_mlp.physics.compute_loss_from_logits(
                logits_eff,
                guide_map=g_ref,
                coarse_target=g_ref,
                aligner=aligner_arg,
                valid_hr_hw=valid_hw,
                loss_mask_lr=loss_mask_lr,
            )
            if self.fuser.spatial_mlp.tv_weight > 0.0 and l_center is not None:
                tv_logits = (
                    l_center[:, :, : valid_hw[0], : valid_hw[1]]
                    if valid_hw is not None
                    else l_center
                )
                g_hr_c = center_slab.unsqueeze(0)
                tv_g = (
                    g_hr_c[:, :, : valid_hw[0], : valid_hw[1]]
                    if valid_hw is not None
                    else g_hr_c
                )
                simplex = self.fuser.spatial_mlp.loss_fn in (
                    "compositional_kl",
                    "forward_psf_poisson",
                ) or (self.fuser.spatial_mlp.physics.spectral_dict is not None)
                loss_t = loss_t + self.fuser.spatial_mlp.tv_weight * edge_aware_tv_loss(
                    tv_logits,
                    guidance_hr=tv_g,
                    simplex=simplex,
                    orientation_aware=self.fuser.spatial_mlp.orientation_aware,
                )
        elif self.fuser.xrd_fuse_method == "learned_gating":
            assert self.fuser.learned_gating is not None
            loss_t = self.fuser.learned_gating.compute_loss_from_logits(
                logits_eff,
                xrd_map=g_ref,
                coarse_target=g_ref,
                aligner=aligner_arg,
                valid_hr_hw=valid_hw,
                loss_mask_lr=loss_mask_lr,
            )
        else:
            assert self.fuser.spatial_attention is not None
            loss_t = self.fuser.spatial_attention.compute_loss_from_logits(
                logits_eff[:, :, : orig_hw[0], : orig_hw[1]],
                guide_map=g_ref,
                coarse_target=g_ref,
                aligner=aligner_arg,
                loss_mask_lr=loss_mask_lr,
            )
        if aligner_arg is not None:
            loss_t = loss_t + aligner_arg.regularization_loss(
                (int(g_ref.shape[-2]), int(g_ref.shape[-1])), device=dev
            )
        loss_t.backward()
        all_p = [
            p for pg in self.fuser.fusion_optimizer.param_groups for p in pg["params"]
        ]
        if all_p:
            torch.nn.utils.clip_grad_norm_(all_p, max_norm=1.0)
        self.fuser.fusion_optimizer.step()
        if update_aligner and self.fuser.align_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.fuser.guide_aligner.parameters()), max_norm=1.0
            )
            self.fuser.align_optimizer.step()
        return float(loss_t.item())

    def fit_sparse_slices(
        self,
        hr_volume: Any,
        sparse_guides: dict[int, np.ndarray | torch.Tensor],
        epochs: int = 12,
        delta: int = 1,
        axis: int = 0,
        reset_weights: bool = True,
        *,
        depth_psf_sigma: float = 0.0,
        learn_depth_psf: bool = False,
        depth_stencil_radius: int = 1,
        triplane_consistency_weight: float = 0.0,
        n_consistency_slices: int = 2,
    ) -> list[float]:
        """Train the ``CorrelativeFuser`` head on sparse 2D guided slices
        ``{slice_idx: guide_map}`` using 2.5D ``(z - delta, z, z + delta)`` slabs,
        with optional axial depth-PSF deconvolution and tri-plane consistency."""
        if not sparse_guides:
            raise ValueError("sparse_guides must contain at least one guided slice.")
        if reset_weights:
            self.fuser.reset_trainable_weights()
        elif self.fuser.fusion_optimizer is not None:
            self.fuser.fusion_optimizer.param_groups = [
                pg
                for pg in self.fuser.fusion_optimizer.param_groups
                if not pg.get("_depth_psf_group", False)
            ]

        accessor = VolumeSliceAccessor(hr_volume)
        z_dim, y_dim, x_dim = accessor.shape
        depth_ax = accessor.shape[axis]
        vmin, vmax = accessor.estimate_robust_range()
        dev = next(self.fuser.dinov2.parameters()).device
        is_spectral = self.fuser.spectral_channels is not None
        norm_mode = self.fuser._default_guide_norm(None)

        prepared_guides: dict[int, torch.Tensor] = {}
        sparse_norm_cpu: dict[int, torch.Tensor] = {}
        prepared_slabs: dict[int, torch.Tensor] = {}
        orig_hws: dict[int, tuple[int, int]] = {}
        cal_masks: dict[int, torch.Tensor | None] = {}
        train_masks: dict[int, torch.Tensor | None] = {}
        prior_accum = []
        for idx, g_raw in sparse_guides.items():
            s_i = int(idx)
            g_t = guide_to_tensor(g_raw, device=dev, normalization=norm_mode)
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
                sparse_norm_cpu[s_i] = g_unit.detach().cpu()
                prior_accum.append(g_unit.mean(dim=(-2, -1)))

        self._sparse_guides_norm = sparse_norm_cpu
        if prior_accum:
            self._global_prior = torch.stack(prior_accum, dim=0).mean(dim=0)

        slice_keys = sorted(prepared_guides.keys())
        use_depth_psf = (depth_psf_sigma > 0.0 or learn_depth_psf) and depth_ax >= 2
        use_cons = (
            triplane_consistency_weight > 0.0
            and min(z_dim, y_dim, x_dim) >= 4
            and max(1, int(n_consistency_slices)) >= 1
        )

        self.fuser.set_guide_transforms(
            prepared_guides[slice_keys[0]],
            self.fwd_tta,
            self.inv_tta,
            spectral_unmix=is_spectral,
            n_components=self.fuser.guide_channels,
        )

        raw_depth_sigma: torch.nn.Parameter | None = None
        init_sig = float(
            np.clip(depth_psf_sigma if depth_psf_sigma > 0.0 else 1.5, 0.35, 5.8)
        )
        # Contiguous unit-step stencil (z_step=1) avoids skipping intermediate slices
        stencil_r = (
            min(
                4,
                max(
                    int(depth_stencil_radius),
                    int(np.ceil(1.5 * init_sig)),
                    2 if learn_depth_psf else 1,
                ),
            )
            if use_depth_psf
            else max(1, int(depth_stencil_radius))
        )
        z_step = 1
        stencil_slabs: dict[int, tuple[torch.Tensor, tuple[int, int]]] = {
            s_i: (prepared_slabs[s_i], orig_hws[s_i]) for s_i in slice_keys
        }
        if use_depth_psf:
            if learn_depth_psf and self.fuser.fusion_optimizer is not None:
                frac = float(np.clip((init_sig - 0.25) / 6.0, 1e-3, 1.0 - 1e-3))
                raw_depth_sigma = torch.nn.Parameter(
                    torch.tensor(
                        float(np.log(frac / (1.0 - frac))),
                        device=dev,
                        dtype=torch.float32,
                    )
                )
                self.fuser.fusion_optimizer.add_param_group(
                    {
                        "params": [raw_depth_sigma],
                        "lr": max(self.fuser.lr * 2.0, 2e-2),
                        "_depth_psf_group": True,
                    }
                )
            for s_i in slice_keys:
                for j in range(-stencil_r, stencil_r + 1):
                    z_nb = int(np.clip(s_i + j * z_step, 0, depth_ax - 1))
                    if z_nb not in stencil_slabs:
                        stencil_slabs[z_nb] = self._prepare_slab_tensor(
                            accessor,
                            z_nb,
                            axis=axis,
                            delta=delta,
                            vmin=vmin,
                            vmax=vmax,
                            device=dev,
                        )

        cons_coords: list[tuple[int, int, int]] = []
        cons_slabs: dict[tuple[int, int], tuple[torch.Tensor, tuple[int, int]]] = {}
        if use_cons:
            n_cs = max(1, int(n_consistency_slices))
            n_pool = max(2, n_cs * 2)
            unguided = [
                z_i for z_i in range(z_dim) if (axis != 0 or z_i not in prepared_guides)
            ] or list(range(z_dim))
            rng_c = np.random.default_rng(137 + len(slice_keys))
            z_ps = [
                int(unguided[i % len(unguided)])
                for i in np.linspace(0, len(unguided) - 1, num=n_pool, dtype=int)
            ]
            y_ps = rng_c.integers(1, max(2, y_dim - 1), size=n_pool).tolist()
            x_ps = rng_c.integers(1, max(2, x_dim - 1), size=n_pool).tolist()
            for z_u, y_u, x_u in zip(z_ps, y_ps, x_ps, strict=False):
                cons_coords.append((int(z_u), int(y_u), int(x_u)))
                for ax_c, idx_c in ((0, int(z_u)), (1, int(y_u)), (2, int(x_u))):
                    if (ax_c, idx_c) not in cons_slabs:
                        cons_slabs[(ax_c, idx_c)] = (
                            stencil_slabs[idx_c]
                            if (ax_c == axis and idx_c in stencil_slabs)
                            else self._prepare_slab_tensor(
                                accessor,
                                idx_c,
                                axis=ax_c,
                                delta=delta,
                                vmin=vmin,
                                vmax=vmax,
                                device=dev,
                            )
                        )

        self.fuser._max_cache_entries = max(
            self.fuser._max_cache_entries, len(stencil_slabs) + len(cons_slabs) + 4
        )
        self.fuser.train()
        self.fuser.dinov2.eval()
        epoch_losses: list[float] = []
        n_ep = max(1, epochs)
        warmup_ep = int(round(0.3 * n_ep)) if self.fuser.align_guide != "none" else 0
        dummy_guide = (
            self._global_prior.to(dev).unsqueeze(-1).unsqueeze(-1)
            if self._global_prior is not None
            else torch.full((1, self.fuser.guide_channels, 4, 4), 0.5, device=dev)
        )

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
                if not use_depth_psf:
                    loss_val = self.fuser.train_fusion_step(
                        prepared_slabs[s_idx],
                        update_aligner=(ep >= warmup_ep),
                        valid_hr_hw=orig_hws[s_idx],
                        loss_mask_lr=train_masks[s_idx],
                    )
                else:
                    sig_t = (
                        0.25 + 6.0 * torch.sigmoid(raw_depth_sigma)
                        if raw_depth_sigma is not None
                        else torch.tensor(
                            max(float(depth_psf_sigma), 0.25),
                            device=dev,
                            dtype=torch.float32,
                        )
                    )
                    loss_val = self._depth_psf_slice_step(
                        s_idx=s_idx,
                        canon_guide=prepared_guides[s_idx],
                        center_slab=prepared_slabs[s_idx],
                        orig_hw=orig_hws[s_idx],
                        loss_mask_lr=train_masks[s_idx],
                        stencil_slabs=stencil_slabs,
                        sparse_norm_cpu=sparse_norm_cpu,
                        depth_ax=depth_ax,
                        axis=axis,
                        stencil_radius=stencil_r,
                        z_step=z_step,
                        sigma_z=sig_t,
                        update_aligner=(ep >= warmup_ep),
                        dev=dev,
                    )
                step_losses.append(loss_val)

            if (
                use_cons
                and ep >= int(0.25 * n_ep)
                and self.fuser.fusion_optimizer is not None
                and cons_coords
            ):
                self.fuser.fusion_optimizer.zero_grad()
                n_cs = max(1, int(n_consistency_slices))
                c_sum = torch.tensor(0.0, device=dev)
                for k_c in range(n_cs):
                    z_u, y_u, x_u = cons_coords[(ep * n_cs + k_c) % len(cons_coords)]
                    s_xy, hw_xy = cons_slabs[(0, z_u)]
                    s_xz, hw_xz = cons_slabs[(1, y_u)]
                    s_yz, hw_yz = cons_slabs[(2, x_u)]
                    g_xy = self._resolve_slice_guide_2d(
                        z_u, z_dim, 0, None, sparse_norm_cpu or None, dummy_guide, dev
                    )
                    p_xy = self._logits_to_probs_grad(
                        self._forward_slab_logits_grad(s_xy, hw_xy, guide_ref=g_xy)
                    )[0][0, :, : hw_xy[0], : hw_xy[1]]
                    p_xz = self._logits_to_probs_grad(
                        self._forward_slab_logits_grad(
                            s_xz, hw_xz, guide_ref=dummy_guide
                        )
                    )[0][0, :, : hw_xz[0], : hw_xz[1]]
                    p_yz = self._logits_to_probs_grad(
                        self._forward_slab_logits_grad(
                            s_yz, hw_yz, guide_ref=dummy_guide
                        )
                    )[0][0, :, : hw_yz[0], : hw_yz[1]]

                    lx_a, lx_b = p_xy[:, y_u, :], p_xz[:, z_u, :]
                    bx = 0.55 * lx_a.detach() + 0.45 * lx_b.detach()
                    ly_a, ly_b = p_xy[:, :, x_u], p_yz[:, z_u, :]
                    by = 0.55 * ly_a.detach() + 0.45 * ly_b.detach()
                    lz_a, lz_b = p_xz[:, :, x_u], p_yz[:, :, y_u]
                    bz = 0.50 * (lz_a.detach() + lz_b.detach())
                    c_sum = (
                        c_sum
                        + (
                            F.mse_loss(lx_a, bx)
                            + F.mse_loss(lx_b, bx)
                            + F.mse_loss(ly_a, by)
                            + F.mse_loss(ly_b, by)
                            + F.mse_loss(lz_a, bz)
                            + F.mse_loss(lz_b, bz)
                        )
                        / 6.0
                    )
                (float(triplane_consistency_weight) * (c_sum / float(n_cs))).backward()
                all_p = [
                    p
                    for pg in self.fuser.fusion_optimizer.param_groups
                    for p in pg["params"]
                ]
                if all_p:
                    torch.nn.utils.clip_grad_norm_(all_p, max_norm=1.0)
                self.fuser.fusion_optimizer.step()

            epoch_losses.append(float(np.mean(step_losses)))
        self.fuser.eval()

        learned_sig_z = (
            float((0.25 + 6.0 * torch.sigmoid(raw_depth_sigma)).detach().cpu().item())
            if raw_depth_sigma is not None
            else float(depth_psf_sigma)
        )
        self._last_fit_diagnostics = {
            "depth_psf_enabled": bool(use_depth_psf),
            "depth_psf_sigma_learned": learned_sig_z,
            "triplane_consistency_weight": float(triplane_consistency_weight),
        }

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

    @staticmethod
    def _resolve_slice_guide_2d(
        idx: int,
        depth: int,
        axis: int,
        coarse_guide_3d_norm: torch.Tensor | None,
        sparse_guides_norm: dict[int, torch.Tensor] | None,
        fallback_guide: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if coarse_guide_3d_norm is not None:
            if axis == 0:
                d_lr = int(coarse_guide_3d_norm.shape[1])
                i_lr = min(d_lr - 1, round(idx * (d_lr - 1) / max(depth - 1, 1)))
                return coarse_guide_3d_norm[:, i_lr, :, :].unsqueeze(0).to(device)
            if axis == 1:
                d_lr = int(coarse_guide_3d_norm.shape[2])
                i_lr = min(d_lr - 1, round(idx * (d_lr - 1) / max(depth - 1, 1)))
                return coarse_guide_3d_norm[:, :, i_lr, :].unsqueeze(0).to(device)
            d_lr = int(coarse_guide_3d_norm.shape[3])
            i_lr = min(d_lr - 1, round(idx * (d_lr - 1) / max(depth - 1, 1)))
            return coarse_guide_3d_norm[:, :, :, i_lr].unsqueeze(0).to(device)

        if axis == 0 and sparse_guides_norm:
            zs = sorted(sparse_guides_norm.keys())
            if idx <= zs[0]:
                return sparse_guides_norm[zs[0]].to(device)
            if idx >= zs[-1]:
                return sparse_guides_norm[zs[-1]].to(device)
            for k in range(len(zs) - 1):
                z0, z1 = zs[k], zs[k + 1]
                if z0 <= idx <= z1:
                    if idx == z0 or z1 == z0:
                        return sparse_guides_norm[z0].to(device)
                    w = float(idx - z0) / float(z1 - z0)
                    g0, g1 = sparse_guides_norm[z0], sparse_guides_norm[z1]
                    if g0.shape == g1.shape:
                        return ((1.0 - w) * g0 + w * g1).to(device)
                    return (g0 if w < 0.5 else g1).to(device)
        return fallback_guide

    def _predict_single_plane_stack(
        self,
        accessor: VolumeSliceAccessor,
        axis: int,
        delta: int,
        vmin: float,
        vmax: float,
        device: torch.device,
        coarse_guide_3d_norm: torch.Tensor | None = None,
        sparse_guides_norm: dict[int, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run slice-by-slice 2.5D logit inference along ``axis``, returning
        ``(mean_logits_zyx, tta_prob_var_zyx)`` transposed back to ``(M, Z, Y, X)``."""
        depth = accessor.shape[axis]
        n_guides = self.fuser.guide_channels
        z_dim, y_dim, x_dim = accessor.shape
        needs_spatial_guide = self.fuser.xrd_fuse_method in ("attention", "gating")

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
            slice_guide = self._resolve_slice_guide_2d(
                idx=idx,
                depth=depth,
                axis=axis,
                coarse_guide_3d_norm=coarse_guide_3d_norm,
                sparse_guides_norm=sparse_guides_norm,
                fallback_guide=dummy_guide,
                device=device,
            )
            if needs_spatial_guide:
                self.fuser._bind_guide_only(slice_guide)

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

                def _tile_fn(
                    tile_chw: torch.Tensor, _sg: torch.Tensor = slice_guide
                ) -> torch.Tensor:
                    ml, vp = self.fuser._predict_logits_and_var_tile(tile_chw, _sg)
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
                valid_hw = (
                    (orig_h, orig_w)
                    if (orig_h, orig_w)
                    != (int(slab_t.shape[-2]), int(slab_t.shape[-1]))
                    else None
                )
                m_l_pad, v_p_pad = self.fuser._predict_logits_and_var_tile(
                    slab_t, slice_guide, valid_hr_hw=valid_hw
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
        if (
            mode == "triplane"
            and accessor._kind == "dir"
            and (z_dim * y_dim * x_dim) <= 128 * 1024 * 1024
        ):
            # Read 2D slice directory once rather than (1 + Y + X) * Z times
            vol_cached = np.stack(
                [accessor.get_plane(0, z_i) for z_i in range(z_dim)], axis=0
            )
            accessor = VolumeSliceAccessor(vol_cached)

        g3_norm: torch.Tensor | None = (
            self._normalize_guide_3d(coarse_guide_3d)
            if coarse_guide_3d is not None
            else None
        )
        sparse_norm_map: dict[int, torch.Tensor] | None = (
            {
                int(k): guide_to_tensor(v, device="cpu", normalize=True)
                for k, v in sparse_guides.items()
            }
            if sparse_guides
            else (self._sparse_guides_norm or None)
        )

        plane_welford = WelfordAccumulator()
        tta_var_accum = torch.zeros(
            (self.fuser.guide_channels, z_dim, y_dim, x_dim), dtype=torch.float32
        )
        mean_logits_sum = torch.zeros_like(tta_var_accum)

        for ax in axes:
            l_zyx, v_zyx = self._predict_single_plane_stack(
                accessor,
                axis=ax,
                delta=eff_delta,
                vmin=vmin,
                vmax=vmax,
                device=dev,
                coarse_guide_3d_norm=g3_norm,
                sparse_guides_norm=sparse_norm_map,
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
        if g3_norm is not None:
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
        has_guide_3d = (g3_norm is not None) or bool(sparse_guides)
        combined_3d = (
            (0.45 * entropy_3d + 0.25 * epi_norm + 0.30 * phys_res_3d)
            if has_guide_3d
            else (0.65 * entropy_3d + 0.35 * epi_norm)
        ).clamp(0.0, 1.0)

        hard_3d = compute_hard_phase_mask(probs_3d, mode=pred_mode, channel_dim=0)

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
                **self._last_fit_diagnostics,
                "volumetric_mode": mode,
                "delta": eff_delta,
                "volume_shape": [z_dim, y_dim, x_dim],
                "lazy_streaming": accessor.is_lazy,
            },
            prediction_sets=pred_sets_3d,
            conformal_interval_halfwidth=conf_hw_3d,
        )
