"""SuRVoS-inspired 2.5D volumetric correlative fusion, orthogonal tri-plane
inference, inter-plane epistemic uncertainty, and sparse-slice 2D -> 3D chemical
propagation (e.g., sparse XRD-CT or FIB-EDS slices guiding a dense 3D XCT/FIB stack).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from .fusion import CorrelativeFuser
from .transforms import (
    build_default_tta,
    ensure_spatial_divisible,
    normalize_structural,
)
from .uncertainty import (
    WelfordAccumulator,
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


def pack_25d_slice(
    volume: np.ndarray | torch.Tensor,
    index: int,
    axis: int = 0,
    delta: int = 1,
) -> np.ndarray:
    """Pack a 3-slice 2.5D structural slab ``(index - delta, index, index + delta)``
    along ``axis`` (``0=Z/XY``, ``1=Y/XZ``, ``2=X/YZ``) into a 3-channel
    ``(3, H_plane, W_plane)`` float32 array with reflection clamping at boundaries.

    Feeding ``(z - delta, z, z + delta)`` into the 3 RGB input channels of a ViT
    injects 3D z-context at **zero extra ViT FLOPs** (replacing redundant grayscale
    channel triplication ``[I_z, I_z, I_z]``).
    """
    vol = (
        volume.detach().cpu().float().numpy()
        if isinstance(volume, torch.Tensor)
        else np.asarray(volume, dtype=np.float32)
    )
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume (Z, Y, X), got shape {vol.shape}.")
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}.")

    depth = vol.shape[axis]
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
    slab = np.take(vol, indices, axis=axis)
    if axis == 1:
        # (Z, 3, X) -> (3, Z, X)
        slab = np.moveaxis(slab, 1, 0)
    elif axis == 2:
        # (Z, Y, 3) -> (3, Z, Y)
        slab = np.moveaxis(slab, 2, 0)
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
    """SuRVoS-style 2.5D volumetric correlative fusion engine.

    Supports two primary scientific workflows:
    1. **Sparse-Slice 2D -> 3D Propagation**: Train ``CorrelativeFuser`` on a handful
       of 2D chemical/diffraction slices ``{z_idx: guide_2d}`` (e.g., 21 sparse
       XRD-CT slices inside a 500-slice XCT tomogram), then segment the entire
       3D volume using 2.5D slabs or orthogonal ``XY + XZ + YZ`` tri-planes.
    2. **Dense Coarse-3D -> High-Res-3D Fusion**: Train/calibrate against a coarse
       3D guide volume ``(M, Z_lr, Y_lr, X_lr)`` (e.g., 3D ToF Bragg-edge CT or
       coarse FIB-SEM-EDS) and fuse across tri-planes with inter-plane epistemic
       uncertainty.
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

    def _vol_stats(self, hr_volume: np.ndarray) -> tuple[float, float]:
        lo, hi = np.percentile(hr_volume, (0.5, 99.5))
        if hi - lo < 1e-8:
            lo, hi = float(np.min(hr_volume)), float(np.max(hr_volume))
        return float(lo), float(hi)

    def _prepare_slab_tensor(
        self,
        hr_volume: np.ndarray,
        index: int,
        axis: int,
        delta: int,
        vmin: float,
        vmax: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        slab = pack_25d_slice(hr_volume, index=index, axis=axis, delta=delta)
        slab_clipped = np.clip(slab, vmin, vmax)
        if vmax - vmin > 1e-8:
            slab_01 = (slab_clipped - vmin) / (vmax - vmin)
        else:
            slab_01 = np.zeros_like(slab_clipped)
        # Apply ImageNet channel scaling preserving global volume intensity contrast!
        t = normalize_structural(slab_01, imagenet_norm=True, device=device)
        padded, orig_hw = ensure_spatial_divisible(
            t, self.fuser.original_patch_size, self.fuser.stride[0]
        )
        return padded, orig_hw

    def fit_sparse_slices(
        self,
        hr_volume: np.ndarray | torch.Tensor,
        sparse_guides: dict[int, np.ndarray | torch.Tensor],
        epochs: int = 12,
        delta: int = 1,
        axis: int = 0,
    ) -> list[float]:
        """Train the ``CorrelativeFuser`` head on sparse 2D guided slices
        ``{slice_idx: guide_map}`` using 2.5D ``(z - delta, z, z + delta)`` slabs."""
        if not sparse_guides:
            raise ValueError("sparse_guides must contain at least one guided slice.")

        vol = (
            hr_volume.detach().cpu().float().numpy()
            if isinstance(hr_volume, torch.Tensor)
            else np.asarray(hr_volume, dtype=np.float32)
        )
        vmin, vmax = self._vol_stats(vol)
        dev = next(self.fuser.dinov2.parameters()).device

        prepared_guides: dict[int, torch.Tensor] = {}
        prior_accum = []
        for idx, g_raw in sparse_guides.items():
            g_t = guide_to_tensor(g_raw, device=dev, normalize=True)
            prepared_guides[int(idx)] = g_t
            prior_accum.append(g_t.mean(dim=(-2, -1)))  # (1, M)

        self._global_prior = torch.stack(prior_accum, dim=0).mean(dim=0)  # (1, M)

        epoch_losses: list[float] = []
        slice_keys = sorted(prepared_guides.keys())
        for _ in range(max(1, epochs)):
            step_losses = []
            for s_idx in slice_keys:
                slab_t, _ = self._prepare_slab_tensor(
                    vol, s_idx, axis=axis, delta=delta, vmin=vmin, vmax=vmax, device=dev
                )
                self.fuser.set_guide_transforms(
                    prepared_guides[s_idx], self.fwd_tta, self.inv_tta
                )
                loss_val = self.fuser.train_fusion_step(slab_t)
                step_losses.append(loss_val)
            epoch_losses.append(float(np.mean(step_losses)))
        return epoch_losses

    def _predict_single_plane_stack(
        self,
        vol: np.ndarray,
        axis: int,
        delta: int,
        vmin: float,
        vmax: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run slice-by-slice 2.5D logit inference along ``axis``, returning
        ``(mean_logits_zyx, tta_prob_var_zyx)`` transposed back to ``(M, Z, Y, X)``."""
        depth = vol.shape[axis]
        n_guides = self.fuser.guide_channels
        z_dim, y_dim, x_dim = vol.shape

        plane_logits = []
        plane_vars = []
        dummy_guide = (
            self._global_prior.unsqueeze(-1).unsqueeze(-1)
            if self._global_prior is not None
            else torch.full((1, n_guides, 4, 4), 0.5, device=device)
        )
        self.fuser.set_guide_transforms(dummy_guide, self.fwd_tta, self.inv_tta)

        for idx in range(depth):
            slab_t, (orig_h, orig_w) = self._prepare_slab_tensor(
                vol, idx, axis=axis, delta=delta, vmin=vmin, vmax=vmax, device=device
            )
            x_in = (
                slab_t.to(self.fuser.dtype)
                if self.fuser.dtype != torch.float32
                else slab_t
            )
            img_batch = self.fuser.get_transformed_input_batch(
                x_in, self.fuser.transforms
            )
            _, pad_h, pad_w = x_in.shape
            n_ph, n_pw = self.fuser.get_n_patches(pad_h, pad_w)

            lw = WelfordAccumulator()
            pw = WelfordAccumulator()
            for t_i in range(len(self.fuser.transforms)):
                tr_img = img_batch[t_i].unsqueeze(0)
                out_d = self.fuser.dinov2.forward_feats_attn(  # type: ignore[operator]
                    tr_img, None, "none"
                )
                perm = (
                    out_d["x_norm_patchtokens"]
                    .squeeze(0)
                    .reshape(n_ph, n_pw, self.fuser.feat_dim)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                )
                hr_f = self.fuser._upsample_patch_map(
                    perm.float(), (pad_h, pad_w), guidance_img=tr_img
                )
                if (
                    self.fuser.xrd_fuse_method == "learned_gating"
                    and self.fuser.learned_gating is not None
                ):
                    l_view = self.fuser.learned_gating.predict_logits(hr_f)
                elif (
                    self.fuser.xrd_fuse_method == "attention"
                    and self.fuser.spatial_attention is not None
                ):
                    l_view = self.fuser.spatial_attention.predict_logits(
                        hr_f, dummy_guide
                    )
                else:
                    l_view = hr_f[:, :n_guides]

                inv_l = self.fuser.inverse_transforms[t_i](l_view)[
                    :, :, :orig_h, :orig_w
                ]
                lw.update(inv_l)
                pw.update(torch.sigmoid(inv_l))

            m_l, _ = lw.finalize()  # (1, M, orig_h, orig_w)
            _, v_p = pw.finalize()
            plane_logits.append(m_l.squeeze(0).cpu())
            plane_vars.append(v_p.squeeze(0).cpu())

        stacked_l = torch.stack(plane_logits, dim=1)  # (M, depth, H_p, W_p)
        stacked_v = torch.stack(plane_vars, dim=1)

        if axis == 0:
            # Already (M, Z, Y, X)
            return stacked_l, stacked_v
        if axis == 1:
            # Currently (M, Y, Z, X) -> permute to (M, Z, Y, X)
            assert stacked_l.shape[1:] == (y_dim, z_dim, x_dim)
            return stacked_l.permute(0, 2, 1, 3), stacked_v.permute(0, 2, 1, 3)
        # axis == 2: Currently (M, X, Z, Y) -> permute to (M, Z, Y, X)
        assert stacked_l.shape[1:] == (x_dim, z_dim, y_dim)
        return stacked_l.permute(0, 2, 3, 1), stacked_v.permute(0, 2, 3, 1)

    @torch.no_grad()
    def predict_volume(
        self,
        hr_volume: np.ndarray | torch.Tensor,
        mode: Literal["2d", "2.5d_slab", "triplane"] = "2.5d_slab",
        delta: int = 1,
        coarse_guide_3d: np.ndarray | torch.Tensor | None = None,
        sparse_guides: dict[int, np.ndarray | torch.Tensor] | None = None,
        pred_mode: Literal["multiclass", "multilabel"] = "multilabel",
        calibrate_to_guide: bool = True,
    ) -> VolumetricUncertaintyReport:
        """Segment a full 3D structural volume ``(Z, Y, X)`` using:
        - ``mode="2d"``: independent 2D slices ``(z, z, z)`` (baseline),
        - ``mode="2.5d_slab"``: SuRVoS-style ``(z - delta, z, z + delta)`` slabs,
        - ``mode="triplane"``: orthogonal ``XY + XZ + YZ`` 2.5D fusion with
          exact inter-plane epistemic disagreement variance!
        """
        vol = (
            hr_volume.detach().cpu().float().numpy()
            if isinstance(hr_volume, torch.Tensor)
            else np.asarray(hr_volume, dtype=np.float32)
        )
        z_dim, y_dim, x_dim = vol.shape
        vmin, vmax = self._vol_stats(vol)
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
                vol, axis=ax, delta=eff_delta, vmin=vmin, vmax=vmax, device=dev
            )
            mean_logits_sum += l_zyx
            tta_var_accum += v_zyx
            plane_welford.update(torch.sigmoid(l_zyx))

        mean_logits = mean_logits_sum / len(axes)  # (M, Z, Y, X)
        _, interplane_var = plane_welford.finalize()  # (M, Z, Y, X)
        epi_var_m = interplane_var + (tta_var_accum / len(axes))
        epi_var = epi_var_m.mean(dim=0)  # (Z, Y, X)

        # Calibrate 3D prior fractions against coarse 3D guide or sparse 2D slices
        n_guides = mean_logits.shape[0]
        flat_logits = mean_logits.reshape(1, n_guides, z_dim, y_dim * x_dim)
        if coarse_guide_3d is not None:
            g3 = (
                coarse_guide_3d.detach().cpu().float()
                if isinstance(coarse_guide_3d, torch.Tensor)
                else torch.from_numpy(np.asarray(coarse_guide_3d, dtype=np.float32))
            )
            if g3.ndim == 3:
                g3 = g3.unsqueeze(0)  # (1, Z_lr, Y_lr, X_lr) -> treat M=1
            prior_ref = g3.reshape(1, g3.shape[0], -1, 1)
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

        if pred_mode == "multiclass":
            bg_l = torch.zeros((1, 1, z_dim, y_dim * x_dim), dtype=torch.float32)
            full_l = torch.cat([bg_l, flat_logits], dim=1)
            probs_flat = (
                match_coarse_priors(full_l, prior_ref, mode="multiclass")
                if (calibrate_to_guide and prior_ref is not None)
                else F.softmax(full_l, dim=1)
            )
            probs_3d = probs_flat.squeeze(0).reshape(n_guides + 1, z_dim, y_dim, x_dim)
            fg_probs_3d = probs_3d[1:]
        else:
            probs_flat = (
                match_coarse_priors(flat_logits, prior_ref, mode="multilabel")
                if (calibrate_to_guide and prior_ref is not None)
                else torch.sigmoid(flat_logits)
            )
            probs_3d = probs_flat.squeeze(0).reshape(n_guides, z_dim, y_dim, x_dim)
            fg_probs_3d = probs_3d

        # Compute 3D posterior entropy across (Z, Y*X)
        entropy_flat, _ = compute_posterior_uncertainty(
            probs_3d.reshape(1, probs_3d.shape[0], z_dim, y_dim * x_dim),
            mode=pred_mode,
        )
        entropy_3d = entropy_flat.squeeze(0).reshape(z_dim, y_dim, x_dim)

        # Compute 3D physical residual if coarse_guide_3d or sparse_guides provided
        phys_res_3d = torch.zeros((z_dim, y_dim, x_dim), dtype=torch.float32)
        if coarse_guide_3d is not None:
            g3 = (
                coarse_guide_3d.detach().cpu().float()
                if isinstance(coarse_guide_3d, torch.Tensor)
                else torch.from_numpy(np.asarray(coarse_guide_3d, dtype=np.float32))
            )
            if g3.ndim == 3:
                g3 = g3.unsqueeze(0)  # (1, Z_lr, Y_lr, X_lr)
            # Normalize per channel to [0, 1]
            g3_min = g3.flatten(1).min(dim=1).values.view(-1, 1, 1, 1)
            g3_max = g3.flatten(1).max(dim=1).values.view(-1, 1, 1, 1)
            g3_norm = (g3 - g3_min) / (g3_max - g3_min).clamp(min=1e-6)
            out_3d = (
                int(g3_norm.shape[-3]),
                int(g3_norm.shape[-2]),
                int(g3_norm.shape[-1]),
            )
            pred_lr_3d = F.adaptive_avg_pool3d(
                fg_probs_3d.unsqueeze(0), output_size=out_3d
            ).squeeze(0)
            res_lr = (pred_lr_3d - g3_norm).abs().mean(dim=0, keepdim=True).unsqueeze(0)
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
                    res_s = (p_s - g_t).abs().mean(dim=1, keepdim=True)
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

        p99 = torch.quantile(epi_var.flatten(), 0.99).clamp(min=1e-6)
        epi_norm = (epi_var / p99).clamp(0.0, 1.0)
        combined_3d = (0.45 * entropy_3d + 0.30 * epi_norm + 0.25 * phys_res_3d).clamp(
            0.0, 1.0
        )

        if pred_mode == "multiclass" and probs_3d.shape[0] >= 2:
            hard_3d = probs_3d.argmax(dim=0).long()
        else:
            hard_3d = (probs_3d[0] >= 0.5).long()

        z_cons = z_slice_consistency_score(hard_3d)
        return VolumetricUncertaintyReport(
            probabilities=probs_3d,
            hard_mask=hard_3d,
            aleatoric_entropy=entropy_3d,
            triplane_variance=epi_var,
            physical_residual=phys_res_3d,
            combined_uncertainty=combined_3d,
            z_slice_consistency=z_cons,
        )
