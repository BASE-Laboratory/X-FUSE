"""Staged 3-seed architectural & real-EM design ablation runner for X-FUSE.

Evaluates all five next-generation X-FUSE upgrades with incremental checkpointing:
- Stage A: Spin-2 structure-tensor orientation & continuous sub-token cross-attention
- Stage B: Anisotropic Moffat & Core+Halo forward PSF profiles vs. Gaussian
- Stage C: 3D axial interaction-volume PSF deconvolution & tri-plane consistency
- Stage D: Temperature-scaled RAPS conformal calibration with spatial block split
- Stage E: Real Zenodo 5-element STEM-EDS (`ti_rcca_stem_eds`) & 3D EELS (`cuzn`)
- Stage plot: 300-DPI 6-panel ablation summary figure (`design_ablations_summary.png`)
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

import x_fuse.backbones as _xf_backbones
from x_fuse import (
    ConformalCalibrator,
    CorrelativeFuser,
    VolumetricFuser,
    XFuse,
    bicubic_upsample_baseline,
    compute_segmentation_metrics,
    empirical_conformal_coverage,
    generate_correlative_phantom_2d,
    generate_correlative_phantom_3d,
    generate_localized_reaction_phantom_3d,
    generate_oriented_lamellar_phantom_2d,
    jbu_guided_baseline,
    make_experimental_holdout,
    suggest_next_slices,
)

from .baselines import (
    cnmf_pansharpen_baseline,
    coordinate_inr_baseline,
    guided_variational_sr_baseline,
    otsu_threshold_baseline,
)
from .dataset_adapters import (
    load_bundled_zn13x_pairs,
    load_cuzn_eels_cube_sample,
    load_zenodo_correlative_sample,
)


def _install_ram_backbone_cache() -> None:
    import hashlib
    from collections import OrderedDict
    from types import MethodType

    orig_loader = _xf_backbones.load_backbone
    ram_cache: dict[tuple[str, bool], nn.Module] = {}
    feat_memo: OrderedDict[tuple[Any, ...], torch.Tensor] = OrderedDict()

    def _cached_loader(
        dino_name: str,
        stride: int,
        device: str | torch.device | None = None,
        pretrained: bool = True,
        **kwargs: Any,
    ) -> nn.Module:
        key = (dino_name, bool(pretrained))
        if key not in ram_cache:
            ram_cache[key] = orig_loader(
                dino_name, stride=stride, device="cpu", pretrained=pretrained, **kwargs
            )
        cloned = copy.deepcopy(ram_cache[key])
        if hasattr(cloned, "forward_features") and hasattr(cloned, "patch_embed"):
            cls_ff = type(cloned).forward_features

            def _memo_ff(
                self_m: Any, x: torch.Tensor, *a: Any, **kw: Any
            ) -> torch.Tensor:
                if a or kw or torch.is_grad_enabled():
                    return cls_ff(self_m, x, *a, **kw)
                st = tuple(self_m.patch_embed.proj.stride)
                dig = hashlib.blake2b(
                    x.detach().cpu().contiguous().numpy().tobytes(), digest_size=16
                ).digest()
                f_key = (dino_name, bool(pretrained), st, tuple(x.shape), dig)
                if f_key in feat_memo:
                    feat_memo.move_to_end(f_key)
                    return feat_memo[f_key].to(x.device, dtype=x.dtype)
                out_f = cls_ff(self_m, x)
                feat_memo[f_key] = out_f.detach().cpu()
                while len(feat_memo) > 512:
                    feat_memo.popitem(last=False)
                return out_f

            cloned.forward_features = MethodType(_memo_ff, cloned)
        return cloned.to(device) if device is not None else cloned

    orig_extract = _xf_backbones.HighResBackbone.extract_patch_tokens
    tok_memo: OrderedDict[tuple[Any, ...], torch.Tensor] = OrderedDict()

    def _memo_extract(
        self_b: Any,
        transformed_img: torch.Tensor,
        attn_choice: Any = "none",
    ) -> torch.Tensor:
        if getattr(self_b, "track_grad", False) or transformed_img.requires_grad:
            return orig_extract(self_b, transformed_img, attn_choice=attn_choice)
        dig = hashlib.blake2b(
            transformed_img.detach().cpu().contiguous().numpy().tobytes(),
            digest_size=16,
        ).digest()
        t_key = (
            getattr(self_b, "dino_name", ""),
            bool(self_b.pretrained),
            tuple(self_b.stride),
            tuple(self_b.layer_indices or ()),
            str(attn_choice),
            tuple(transformed_img.shape),
            dig,
        )
        if t_key in tok_memo:
            tok_memo.move_to_end(t_key)
            return tok_memo[t_key].to(
                transformed_img.device, dtype=transformed_img.dtype
            )
        with torch.no_grad():
            out_t = orig_extract(self_b, transformed_img, attn_choice=attn_choice)
        tok_memo[t_key] = out_t.detach().cpu()
        while len(tok_memo) > 384:
            tok_memo.popitem(last=False)
        return out_t

    orig_upsample = _xf_backbones.HighResBackbone._upsample_patch_map
    up_memo: OrderedDict[tuple[Any, ...], torch.Tensor] = OrderedDict()

    def _memo_upsample(
        self_b: Any,
        patch_map: torch.Tensor,
        target_hw: tuple[int, int],
        guidance_img: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ca_mod = getattr(self_b, "cross_attn_upsampler", None)
        ca_grad = ca_mod is not None and any(
            p.requires_grad for p in ca_mod.parameters()
        )
        if patch_map.requires_grad or ca_grad or guidance_img is None:
            return orig_upsample(
                self_b, patch_map, target_hw, guidance_img=guidance_img
            )
        g_dig = hashlib.blake2b(
            guidance_img.detach().cpu().contiguous().numpy().tobytes(),
            digest_size=12,
        ).digest()
        p_dig = hashlib.blake2b(
            patch_map.detach().cpu().contiguous().numpy().tobytes(),
            digest_size=12,
        ).digest()
        u_key = (
            str(self_b.interpolation_mode),
            bool(getattr(self_b, "upsampler_orientation", True)),
            tuple(target_hw),
            tuple(patch_map.shape),
            g_dig,
            p_dig,
        )
        if u_key in up_memo:
            up_memo.move_to_end(u_key)
            return up_memo[u_key].to(patch_map.device, dtype=patch_map.dtype)
        with torch.no_grad():
            out_u = orig_upsample(
                self_b, patch_map, target_hw, guidance_img=guidance_img
            )
        up_memo[u_key] = out_u.detach().cpu()
        while len(up_memo) > 160:
            up_memo.popitem(last=False)
        return out_u

    _xf_backbones.load_backbone = _cached_loader  # type: ignore[assignment]
    _xf_backbones.HighResBackbone.extract_patch_tokens = _memo_extract  # type: ignore[method-assign]
    _xf_backbones.HighResBackbone._upsample_patch_map = _memo_upsample  # type: ignore[method-assign]


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _mean_std(vals: list[float]) -> dict[str, Any]:
    arr = np.asarray(vals, dtype=np.float64)
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "mean": round(float(np.mean(arr)), 4),
        "std": round(sd, 4),
        "values": [round(float(v), 4) for v in vals],
    }


def _save_json(out_dir: Path, payload: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "design_ablations_results.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def _continuous_channel_metrics(
    pred_mhw: np.ndarray | torch.Tensor, gt_mhw: np.ndarray
) -> dict[str, float]:
    p = (
        pred_mhw.detach().cpu().float().numpy()
        if isinstance(pred_mhw, torch.Tensor)
        else np.asarray(pred_mhw, dtype=np.float32)
    )
    if p.ndim == 4:
        p = p[0]
    g = np.asarray(gt_mhw, dtype=np.float32)
    m = min(p.shape[0], g.shape[0])
    rs: list[float] = []
    psnrs: list[float] = []
    ious: list[float] = []
    for c in range(m):
        pc = p[c].astype(np.float32)
        gc = g[c].astype(np.float32)
        g_lo, g_hi = float(np.percentile(gc, 1.0)), float(np.percentile(gc, 99.0))
        gc_n = np.clip((gc - g_lo) / max(g_hi - g_lo, 1e-6), 0.0, 1.0)
        p_lo, p_hi = float(np.percentile(pc, 1.0)), float(np.percentile(pc, 99.0))
        pc_n = np.clip((pc - p_lo) / max(p_hi - p_lo, 1e-6), 0.0, 1.0)
        if float(np.std(pc_n)) > 1e-6 and float(np.std(gc_n)) > 1e-6:
            rs.append(float(np.corrcoef(pc_n.ravel(), gc_n.ravel())[0, 1]))
        else:
            rs.append(0.0)
        a_mat = np.column_stack([pc_n.ravel(), np.ones(pc_n.size, dtype=np.float32)])
        coef, _, _, _ = np.linalg.lstsq(a_mat, gc_n.ravel(), rcond=None)
        pc_cal = np.clip(coef[0] * pc_n + coef[1], 0.0, 1.0)
        mse = float(np.mean((pc_cal - gc_n) ** 2))
        psnrs.append(float(-10.0 * np.log10(max(mse, 1e-10))))
        q_thr = float(np.quantile(gc_n, 0.70))
        pb = pc_cal >= q_thr
        gb = gc_n >= q_thr
        u = int((pb | gb).sum())
        ious.append(float((pb & gb).sum() / u) if u > 0 else 1.0)
    return {
        "pearson_r": float(np.mean(rs)),
        "psnr_db": float(np.mean(psnrs)),
        "support_iou": float(np.mean(ious)),
    }


def _make_core_halo_holdout(
    gt_channels: np.ndarray, factor: int, halo_weight: float = 0.28, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m, h, w = gt_channels.shape
    h_lr, w_lr = h // factor, w // factor
    out = np.zeros((m, h_lr, w_lr), dtype=np.float32)
    sig_c = 0.42 * factor
    sig_h = 3.2 * sig_c
    for c in range(m):
        core = gaussian_filter(gt_channels[c], sigma=sig_c, mode="nearest")
        halo = gaussian_filter(gt_channels[c], sigma=sig_h, mode="nearest")
        mix = (1.0 - halo_weight) * core + halo_weight * halo
        pooled = mix.reshape(h_lr, factor, w_lr, factor).mean(axis=(1, 3))
        noisy = (
            rng.poisson(np.clip(pooled, 0.0, 1.0) * 350.0).astype(np.float32) / 350.0
        )
        out[c] = np.clip(noisy, 0.0, 1.0)
    return out


def run_stage_a_upsampler(
    seeds: tuple[int, ...], backbone: str, pretrained: bool, stride: int, device: str
) -> dict[str, Any]:
    """Stage A: Upsampler & Spin-2 orientation ablation on 8x & 16x Moffat lamellae."""
    variants = [
        ("scalar_dilated_jbu", False, False, False),
        ("spin2_structure_tensor", True, False, False),
        ("spin2_plus_subtoken", True, True, False),
        ("learned_spin2_subtoken", True, True, True),
    ]
    out: dict[str, Any] = {}
    for factor in (8, 16):
        f_key = f"lamellar_{factor}x"
        out[f_key] = {}
        acc = {v[0]: {"mIoU": [], "BF1": []} for v in variants}
        for s in seeds:
            ph = generate_oriented_lamellar_phantom_2d(
                size=96,
                downsample_factor=factor,
                beam_profile="moffat",
                poisson_counts=300.0,
                seed=s,
            )
            for name, ori, subtok, learned_up in variants:
                _set_seeds(s)
                xf = XFuse(
                    backbone=backbone,
                    stride=stride,
                    pretrained=pretrained,
                    device=device,
                    loss_fn="compositional_kl",
                    interpolation_mode="cross_attn",
                    fusion_method="spatial_mlp",
                    learn_psf=True,
                    tv_weight=0.01,
                    lr=1e-2,
                    upsampler_orientation=ori,
                    upsampler_subtoken=subtok,
                    learned_upsampler=learned_up,
                )
                rep = xf.fit_predict(
                    ph.hr_image,
                    ph.coarse_guide,
                    epochs=20,
                    mode="multiclass",
                    tta_shifts=(),
                    tta_flips=False,
                )
                m = compute_segmentation_metrics(rep.hard_mask[0], ph.gt_labels)
                acc[name]["mIoU"].append(m["iou"])
                acc[name]["BF1"].append(m["boundary_f1"])
            print(f"  [Stage A] {f_key} seed={s} done", flush=True)
        for name, _, _, _ in variants:
            out[f_key][name] = {
                "mIoU": _mean_std(acc[name]["mIoU"]),
                "BF1": _mean_std(acc[name]["BF1"]),
            }
    return out


def run_stage_b_psf_physics(
    seeds: tuple[int, ...], backbone: str, pretrained: bool, stride: int, device: str
) -> dict[str, Any]:
    """Stage B: Forward PSF profiles across Core+Halo, Moffat, and real Zn13X."""
    profiles = [
        ("fixed_gaussian", False, "gaussian"),
        ("learn_gaussian", True, "gaussian"),
        ("learn_moffat", True, "moffat"),
        ("learn_core_halo", True, "core_halo"),
    ]
    halo_acc = {p[0]: {"mIoU": [], "BF1": []} for p in profiles}
    moff_acc = {p[0]: {"mIoU": [], "BF1": []} for p in profiles}
    zn_acc = {
        p[0]: {"pearson_r": [], "psnr_db": [], "support_iou": []} for p in profiles
    }

    zn_s4 = load_bundled_zn13x_pairs()[0]
    hr_224 = zn_s4.hr_image[640:752, 640:752]
    gt_56 = zn_s4.guide_map[0, 320:376, 320:376]
    g_lo, g_hi = float(np.percentile(gt_56, 5.0)), float(np.percentile(gt_56, 99.0))
    gt_56_n = np.clip((gt_56 - g_lo) / max(g_hi - g_lo, 1e-6), 0.0, 1.0)
    zn_lr = (
        F.avg_pool2d(
            torch.from_numpy(gt_56_n).unsqueeze(0).unsqueeze(0), kernel_size=4, stride=4
        )
        .squeeze(0)
        .numpy()
    )

    for s in seeds:
        ph = generate_correlative_phantom_2d(
            size=112,
            downsample_factor=8,
            phase_b_contrast=0.35,
            hr_noise_std=0.03,
            poisson_counts=350.0,
            beam_profile="moffat",
            seed=s,
        )
        coarse_halo = _make_core_halo_holdout(
            ph.gt_phase_channels, factor=8, halo_weight=0.30, seed=s + 10
        )
        for p_name, l_psf, prof in profiles:
            for guide_in, store in (
                (coarse_halo, halo_acc),
                (ph.coarse_guide, moff_acc),
            ):
                _set_seeds(s)
                xf = XFuse(
                    backbone=backbone,
                    stride=stride,
                    pretrained=pretrained,
                    device=device,
                    loss_fn="compositional_kl",
                    interpolation_mode="cross_attn",
                    fusion_method="spatial_mlp",
                    learn_psf=l_psf,
                    psf_profile=prof,  # type: ignore[arg-type]
                    tv_weight=0.008,
                    lr=1e-2,
                )
                rep = xf.fit_predict(
                    ph.hr_image,
                    guide_in,
                    epochs=24,
                    mode="multiclass",
                    tta_shifts=(),
                    tta_flips=True,
                )
                mt = compute_segmentation_metrics(rep.hard_mask[0], ph.gt_labels)
                store[p_name]["mIoU"].append(mt["iou"])
                store[p_name]["BF1"].append(mt["boundary_f1"])

            _set_seeds(s)
            xf_zn = XFuse(
                backbone=backbone,
                stride=stride,
                pretrained=pretrained,
                device=device,
                loss_fn="forward_psf_mse",
                interpolation_mode="cross_attn",
                fusion_method="spatial_mlp",
                guide_residual=True,
                learn_psf=l_psf,
                psf_profile=prof,  # type: ignore[arg-type]
                tv_weight=0.01,
                lr=5e-3,
            )
            rep_zn = xf_zn.fit_predict(
                hr_224,
                zn_lr,
                epochs=28,
                mode="multilabel",
                calibrate_to_guide=False,
                tta_shifts=(),
                tta_flips=False,
                guide_normalization="preserve_counts",
            )
            p_56 = (
                F.avg_pool2d(rep_zn.probabilities[:, :1], kernel_size=2, stride=2)[0, 0]
                .cpu()
                .numpy()
            )
            cm = _continuous_channel_metrics(p_56[None], gt_56_n[None])
            for mk in ("pearson_r", "psnr_db", "support_iou"):
                zn_acc[p_name][mk].append(cm[mk])

    return {
        "core_halo_probe_8x": {
            k: {m: _mean_std(v[m]) for m in v} for k, v in halo_acc.items()
        },
        "moffat_probe_8x": {
            k: {m: _mean_std(v[m]) for m in v} for k, v in moff_acc.items()
        },
        "zn13x_synchrotron_8x": {
            k: {m: _mean_std(v[m]) for m in v} for k, v in zn_acc.items()
        },
    }


def run_stage_c_volumetric(
    seeds: tuple[int, ...], backbone: str, pretrained: bool, stride: int, device: str
) -> dict[str, Any]:
    """Stage C: 3D depth-PSF deconvolution, tri-plane consistency & active slices."""
    configs = [
        ("sparse_25d_baseline", 0.0, False, 0.0, "2.5d_slab"),
        ("depth_psf_deconv", 2.2, True, 0.0, "2.5d_slab"),
        ("depth_psf_plus_triplane", 2.2, True, 0.15, "triplane"),
    ]
    acc = {
        c[0]: {"mIoU_3d": [], "mIoU_unguided": [], "z_consistency": []} for c in configs
    }
    act_acc = {"equispaced_4_slices": [], "uncertainty_active_4_slices": []}

    for s in seeds:
        ph = generate_correlative_phantom_3d(
            shape=(20, 48, 48),
            downsample_factor=4,
            phase_b_contrast=0.35,
            hr_noise_std=0.05,
            poisson_counts=220.0,
            depth_psf_sigma=2.6,
            seed=s,
        )
        sparse_3 = {
            2: ph.coarse_guide_3d[:, 0],
            10: ph.coarse_guide_3d[:, 2],
            18: ph.coarse_guide_3d[:, 4],
        }
        unguided_mask = np.ones(20, dtype=bool)
        for z_k in sparse_3:
            unguided_mask[z_k] = False

        for c_name, d_sig, learn_d, cons_w, inf_mode in configs:
            _set_seeds(s)
            fuser = CorrelativeFuser(
                backbone=backbone,
                stride=stride,
                pretrained=pretrained,
                min_input_size=None,
                fusion_method="spatial_mlp",
                loss_fn="compositional_kl",
                interpolation_mode="cross_attn",
                learn_psf=True,
                tv_weight=0.008,
                lr=1e-2,
                guide_channels=2,
                device=device,
            )
            vf = VolumetricFuser(fuser, tta_shifts=None, tta_flips=False)
            vf.fit_sparse_slices(
                ph.hr_volume,
                sparse_3,
                epochs=10,
                depth_psf_sigma=d_sig,
                learn_depth_psf=learn_d,
                depth_stencil_radius=1,
                triplane_consistency_weight=cons_w,
                n_consistency_slices=2,
            )
            rep = vf.predict_volume(
                ph.hr_volume,
                mode=inf_mode,  # type: ignore[arg-type]
                sparse_guides=sparse_3,
                pred_mode="multiclass",
            )
            hm = rep.hard_mask.cpu().numpy()
            m_all = compute_segmentation_metrics(
                hm.reshape(20 * 48, 48), ph.gt_labels_3d.reshape(20 * 48, 48)
            )
            m_ung = compute_segmentation_metrics(
                hm[unguided_mask].reshape(-1, 48),
                ph.gt_labels_3d[unguided_mask].reshape(-1, 48),
            )
            acc[c_name]["mIoU_3d"].append(m_all["iou"])
            acc[c_name]["mIoU_unguided"].append(m_ung["iou"])
            acc[c_name]["z_consistency"].append(float(rep.z_slice_consistency))

            if c_name == "depth_psf_plus_triplane":
                # Compare 2 active slices vs 4 equispaced slices from 2-slice init
                init_2 = {2: ph.coarse_guide_3d[:, 0], 18: ph.coarse_guide_3d[:, 4]}
                vf.fit_sparse_slices(
                    ph.hr_volume,
                    init_2,
                    epochs=8,
                    depth_psf_sigma=2.2,
                    learn_depth_psf=True,
                    triplane_consistency_weight=0.15,
                )
                r_init = vf.predict_volume(
                    ph.hr_volume,
                    mode="triplane",
                    sparse_guides=init_2,
                    pred_mode="multiclass",
                )
                plan = suggest_next_slices(
                    r_init,
                    acquired_indices=list(init_2.keys()),
                    k=2,
                    min_spacing=4,
                )
                act_4 = dict(init_2)
                for z_sel in plan.recommended_indices:
                    act_4[int(z_sel)] = ph.coarse_guide_3d[:, min(4, int(z_sel) // 4)]
                vf.fit_sparse_slices(
                    ph.hr_volume,
                    act_4,
                    epochs=8,
                    depth_psf_sigma=2.2,
                    learn_depth_psf=True,
                    triplane_consistency_weight=0.15,
                )
                r_act = vf.predict_volume(
                    ph.hr_volume,
                    mode="triplane",
                    sparse_guides=act_4,
                    pred_mode="multiclass",
                )
                act_acc["uncertainty_active_4_slices"].append(
                    compute_segmentation_metrics(
                        r_act.hard_mask.cpu().numpy().reshape(20 * 48, 48),
                        ph.gt_labels_3d.reshape(20 * 48, 48),
                    )["iou"]
                )
                eq_4 = {
                    2: ph.coarse_guide_3d[:, 0],
                    6: ph.coarse_guide_3d[:, 1],
                    14: ph.coarse_guide_3d[:, 3],
                    18: ph.coarse_guide_3d[:, 4],
                }
                vf.fit_sparse_slices(
                    ph.hr_volume,
                    eq_4,
                    epochs=8,
                    depth_psf_sigma=2.2,
                    learn_depth_psf=True,
                    triplane_consistency_weight=0.15,
                )
                r_eq = vf.predict_volume(
                    ph.hr_volume,
                    mode="triplane",
                    sparse_guides=eq_4,
                    pred_mode="multiclass",
                )
                act_acc["equispaced_4_slices"].append(
                    compute_segmentation_metrics(
                        r_eq.hard_mask.cpu().numpy().reshape(20 * 48, 48),
                        ph.gt_labels_3d.reshape(20 * 48, 48),
                    )["iou"]
                )

    return {
        "volumetric_variants": {
            k: {m: _mean_std(v[m]) for m in v} for k, v in acc.items()
        },
        "active_slice_steering": {k: _mean_std(v) for k, v in act_acc.items()},
    }


def run_stage_c2_active_curve(
    seeds: tuple[int, ...],
    backbone: str,
    pretrained: bool,
    stride: int,
    device: str,
) -> dict[str, Any]:
    """C2: Multi-budget (k = 2, 3, 4, 6, 8) active slice steering on Z=48 volume
    with a localized reaction-front phase confined to z in [26, 38).
    """
    budgets = (2, 3, 4, 6, 8)
    curve_acc: dict[str, dict[str, dict[str, list[float]]]] = {
        strat: {str(k): {"mIoU_3d": [], "phase_c_iou": []} for k in budgets}
        for strat in ("active_steering", "equispaced")
    }
    for s in seeds:
        ph_loc = generate_localized_reaction_phantom_3d(
            shape=(48, 32, 32),
            downsample_factor=4,
            reaction_z_band=(26, 38),
            depth_psf_sigma=1.6,
            seed=s + 200,
        )
        gt_flat = ph_loc.gt_labels_3d.reshape(48 * 32, 32)
        gt_c = ph_loc.gt_labels_3d == 3

        def _eval_3d_set(
            slice_map: dict[int, np.ndarray],
            seed_val: int,
            ph_ref: Any = ph_loc,
            gt_f: np.ndarray = gt_flat,
            gt_c_mask: np.ndarray = gt_c,
        ) -> tuple[Any, float, float]:
            _set_seeds(seed_val)
            xf_c2 = XFuse(
                backbone=backbone,
                stride=stride,
                pretrained=pretrained,
                min_input_size=None,
                device=device,
                loss_fn="compositional_kl",
                interpolation_mode="cross_attn",
                fusion_method="spatial_mlp",
                guide_channels=3,
                learn_psf=True,
                guide_residual=True,
                tv_weight=0.01,
                lr=1e-2,
            )
            vf_c2 = VolumetricFuser(xf_c2, tta_shifts=None, tta_flips=False)
            vf_c2.fit_sparse_slices(
                ph_ref.hr_volume,
                slice_map,
                epochs=10,
                depth_psf_sigma=1.6,
                learn_depth_psf=True,
                triplane_consistency_weight=0.0,
            )
            rp_c2 = vf_c2.predict_volume(
                ph_ref.hr_volume,
                mode="2.5d_slab",
                sparse_guides=slice_map,
                pred_mode="multiclass",
            )
            hm_c2 = rp_c2.hard_mask.cpu().numpy()
            miou_all = compute_segmentation_metrics(hm_c2.reshape(48 * 32, 32), gt_f)[
                "iou"
            ]
            pred_c = hm_c2 == 3
            inter_c = float(np.logical_and(pred_c, gt_c_mask).sum())
            union_c = float(np.logical_or(pred_c, gt_c_mask).sum())
            iou_c = inter_c / max(union_c, 1.0)
            return rp_c2, miou_all, iou_c

        # Active iterative steering starting from 2 boundary scout slices [3, 44]
        act_slices: dict[int, np.ndarray] = {
            3: ph_loc.coarse_guide_3d[:, 3],
            44: ph_loc.coarse_guide_3d[:, 44],
        }
        cur_rep: Any | None = None
        for k in budgets:
            while len(act_slices) < k:
                assert cur_rep is not None
                add_n = k - len(act_slices)
                rec_plan = suggest_next_slices(
                    cur_rep,
                    acquired_indices=sorted(act_slices.keys()),
                    k=add_n,
                    min_spacing=3,
                )
                for z_new in rec_plan.recommended_indices:
                    act_slices[int(z_new)] = ph_loc.coarse_guide_3d[:, int(z_new)]
            cur_rep, m_act, c_act = _eval_3d_set(act_slices, s)
            curve_acc["active_steering"][str(k)]["mIoU_3d"].append(m_act)
            curve_acc["active_steering"][str(k)]["phase_c_iou"].append(c_act)

            if k == 2:
                m_eq, c_eq = m_act, c_act
            else:
                eq_idx = [int(round(v)) for v in np.linspace(3, 44, k)]
                eq_slices = {zi: ph_loc.coarse_guide_3d[:, zi] for zi in eq_idx}
                _, m_eq, c_eq = _eval_3d_set(eq_slices, s)
            curve_acc["equispaced"][str(k)]["mIoU_3d"].append(m_eq)
            curve_acc["equispaced"][str(k)]["phase_c_iou"].append(c_eq)

    return {
        strat: {
            bk: {mk: _mean_std(vals) for mk, vals in md.items()}
            for bk, md in bd.items()
        }
        for strat, bd in curve_acc.items()
    }


def _nmf_unmix_cube(
    cube_lr: np.ndarray, n_components: int = 2, seed: int = 0
) -> np.ndarray:
    e_bins, h_lr, w_lr = cube_lr.shape
    y_mat = np.maximum(cube_lr.reshape(e_bins, -1).astype(np.float32), 1e-5)
    rng = np.random.default_rng(seed)
    w_dict = rng.uniform(0.1, 1.0, size=(e_bins, n_components)).astype(np.float32)
    h_ab = rng.uniform(0.1, 1.0, size=(n_components, h_lr * w_lr)).astype(np.float32)
    eps = 1e-7
    for _ in range(50):
        h_ab *= (w_dict.T @ y_mat) / np.maximum((w_dict.T @ w_dict) @ h_ab, eps)
        w_dict *= (y_mat @ h_ab.T) / np.maximum(w_dict @ (h_ab @ h_ab.T), eps)
        ws = np.maximum(w_dict.sum(axis=0, keepdims=True), eps)
        w_dict /= ws
        h_ab *= ws.T
    scale = np.maximum(np.percentile(h_ab, 98, axis=1, keepdims=True), eps)
    return (
        np.clip(h_ab / scale, 0.0, 1.0)
        .reshape(n_components, h_lr, w_lr)
        .astype(np.float32)
    )


def _perm_continuous_metrics(
    pred_2hw: np.ndarray | torch.Tensor, gt_2hw: np.ndarray
) -> dict[str, float]:
    p = (
        pred_2hw.detach().cpu().float().numpy()
        if isinstance(pred_2hw, torch.Tensor)
        else np.asarray(pred_2hw, dtype=np.float32)
    )
    if p.ndim == 4:
        p = p[0]
    if p.shape[0] == 3:
        p = p[1:3]
    m0 = _continuous_channel_metrics(p, gt_2hw)
    m1 = _continuous_channel_metrics(p[::-1], gt_2hw)
    return m0 if m0["pearson_r"] >= m1["pearson_r"] else m1


def _prep_ti_rcca_simplex(
    sample: Any, y0: int = 300, size: int = 168
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hr = sample.hr_image[y0 : y0 + size, y0 : y0 + size].astype(np.float32)
    g5 = sample.guide_map[:, y0 : y0 + size, y0 : y0 + size].astype(np.float32)
    gn = np.stack(
        [
            np.clip(
                (c - np.percentile(c, 2.0))
                / max(float(np.percentile(c, 98.0) - np.percentile(c, 2.0)), 1e-6),
                0.0,
                1.0,
            )
            for c in g5
        ],
        axis=0,
    )
    simplex = gn / np.maximum(gn.sum(axis=0, keepdims=True), 1e-6)
    dom = np.argmax(simplex, axis=0).astype(np.int64)
    return hr, simplex, dom


def run_stage_d_conformal(
    seeds: tuple[int, ...],
    backbone: str,
    pretrained: bool,
    stride: int,
    device: str,
    zenodo_root: Path,
) -> dict[str, Any]:
    """Stage D: LAC vs APS vs Temp-Scaled Block RAPS on synthetic & STEM-EDS."""
    alphas = (0.05, 0.10, 0.20)
    cal_specs = [
        ("raw_lac", "lac", False, 1, 0.0),
        ("raw_aps", "aps", False, 1, 0.0),
        ("temp_raps_block", "raps", True, 4, 0.04),
    ]
    metric_keys = (
        "empirical_coverage",
        "mean_set_size",
        "boundary_mean_set_size",
        "interior_mean_set_size",
        "interior_singleton_fraction",
        "temperature",
    )

    def _init_acc() -> dict[str, dict[str, dict[str, list[float]]]]:
        return {
            cs[0]: {f"{a:.2f}": {mk: [] for mk in metric_keys} for a in alphas}
            for cs in cal_specs
        }

    syn_acc = _init_acc()
    eds_acc = _init_acc()

    s_a = load_zenodo_correlative_sample(
        "ti_rcca_stem_eds", dest_root=zenodo_root, subpanel="Fig3a"
    )
    s_b = load_zenodo_correlative_sample(
        "ti_rcca_stem_eds", dest_root=zenodo_root, subpanel="Fig3b"
    )
    hr_a, simp_a, dom_a = _prep_ti_rcca_simplex(s_a, y0=300, size=168)
    hr_b, _, dom_b = _prep_ti_rcca_simplex(s_b, y0=300, size=168)

    for s in seeds:
        # 1) Synthetic cross-specimen oriented lamellar (K=3)
        pa = generate_oriented_lamellar_phantom_2d(
            size=112, downsample_factor=8, beam_profile="moffat", seed=s
        )
        pb = generate_oriented_lamellar_phantom_2d(
            size=112, downsample_factor=8, beam_profile="moffat", seed=s + 500
        )
        _set_seeds(s)
        xf_s = XFuse(
            backbone=backbone,
            stride=stride,
            pretrained=pretrained,
            device=device,
            loss_fn="compositional_kl",
            interpolation_mode="cross_attn",
            fusion_method="spatial_mlp",
            learn_psf=True,
            tv_weight=0.01,
            lr=1e-2,
        )
        ra = xf_s.fit_predict(
            pa.hr_image, pa.coarse_guide, epochs=25, mode="multiclass", tta_shifts=()
        )
        rb = xf_s.predict(
            pb.hr_image, coarse_guide=None, mode="multiclass", calibrate_to_guide=False
        )
        for cname, meth, fit_t, blk, lreg in cal_specs:
            for a in alphas:
                cal = ConformalCalibrator(
                    alpha=a,
                    method=meth,  # type: ignore[arg-type]
                    k_reg=1,
                    lambda_reg=lreg,
                    randomized=False,
                    fit_temperature=fit_t,
                    spatial_block_size=blk,
                )
                cal.calibrate_labels(ra.probabilities, torch.from_numpy(pa.gt_labels))
                ps, _ = cal.predict_sets_and_intervals(
                    rb.probabilities, guide_lr=None, mode="multiclass"
                )
                st = empirical_conformal_coverage(ps, pb.gt_labels)
                st["temperature"] = float(cal.temperature)
                for mk in metric_keys:
                    syn_acc[cname][f"{a:.2f}"][mk].append(st[mk])

        # 2) Real cross-specimen 5-element STEM-EDS (Fig3a 8Al -> Fig3b 10Al, K=5)
        _, coarse_a, _ = make_experimental_holdout(
            hr_a, simp_a[1:], factor=8, psf_sigma=2.2, beam_profile="moffat"
        )
        _set_seeds(s)
        xf_e = XFuse(
            backbone=backbone,
            stride=stride,
            pretrained=pretrained,
            device=device,
            loss_fn="compositional_kl",
            interpolation_mode="cross_attn",
            fusion_method="spatial_mlp",
            learn_psf=True,
            tv_weight=0.01,
            lr=1e-2,
        )
        rea = xf_e.fit_predict(
            hr_a,
            coarse_a,
            epochs=28,
            mode="multiclass",
            tta_shifts=(),
            guide_normalization="preserve_counts",
        )
        reb = xf_e.predict(
            hr_b, coarse_guide=None, mode="multiclass", calibrate_to_guide=False
        )
        for cname, meth, fit_t, blk, lreg in cal_specs:
            for a in alphas:
                cal = ConformalCalibrator(
                    alpha=a,
                    method=meth,  # type: ignore[arg-type]
                    k_reg=1,
                    lambda_reg=lreg,
                    randomized=False,
                    fit_temperature=fit_t,
                    spatial_block_size=blk,
                )
                cal.calibrate_labels(rea.probabilities, torch.from_numpy(dom_a))
                ps_e, _ = cal.predict_sets_and_intervals(
                    reb.probabilities, guide_lr=None, mode="multiclass"
                )
                ste = empirical_conformal_coverage(ps_e, dom_b)
                ste["temperature"] = float(cal.temperature)
                for mk in metric_keys:
                    eds_acc[cname][f"{a:.2f}"][mk].append(ste[mk])

    def _pack(acc_d: dict[str, dict[str, dict[str, list[float]]]]) -> dict[str, Any]:
        return {
            cn: {
                ak: {mk: _mean_std(vals) for mk, vals in md.items()}
                for ak, md in ad.items()
            }
            for cn, ad in acc_d.items()
        }

    return {
        "synthetic_lamellar_8x": _pack(syn_acc),
        "real_ti_rcca_fig3a_to_fig3b_5class": _pack(eds_acc),
    }


def run_stage_e_real_em(
    seeds: tuple[int, ...],
    backbone: str,
    pretrained: bool,
    stride: int,
    device: str,
    zenodo_root: Path,
) -> dict[str, Any]:
    """Stage E: Real Zenodo 5-element STEM-EDS (Ti-RCCA) & 3D Core-Loss EELS (CuZn)."""
    # E1: Ti-RCCA Fig3b 5-element STEM-EDS (8x Moffat hold-out)
    s_b = load_zenodo_correlative_sample(
        "ti_rcca_stem_eds", dest_root=zenodo_root, subpanel="Fig3b"
    )
    hr_b, simp_b, dom_b = _prep_ti_rcca_simplex(s_b, y0=300, size=168)
    _, coarse_b, _ = make_experimental_holdout(
        hr_b, simp_b[1:], factor=8, psf_sigma=2.2, beam_profile="moffat"
    )
    e1_methods = ("otsu", "bicubic", "jbu", "cnmf", "guided_tv", "siren_inr", "xfuse")
    e1_acc = {
        m: {"mIoU": [], "BF1": [], "pearson_r": [], "psnr_db": [], "support_iou": []}
        for m in e1_methods
    }
    norm = "preserve_counts"
    hw_b = (int(hr_b.shape[0]), int(hr_b.shape[1]))

    for s in seeds:
        reps: dict[str, Any] = {
            "otsu": otsu_threshold_baseline(
                hr_b, coarse_b, mode="multiclass", guide_normalization=norm
            ),
            "bicubic": bicubic_upsample_baseline(
                coarse_b, hw_b, mode="multiclass", guide_normalization=norm
            ),
            "jbu": jbu_guided_baseline(
                hr_b, coarse_b, mode="multiclass", guide_normalization=norm
            ),
            "cnmf": cnmf_pansharpen_baseline(
                hr_b,
                coarse_b,
                mode="multiclass",
                n_iters=35,
                device=device,
                guide_normalization=norm,
            ),
            "guided_tv": guided_variational_sr_baseline(
                hr_b,
                coarse_b,
                mode="multiclass",
                n_steps=55,
                device=device,
                guide_normalization=norm,
            ),
            "siren_inr": coordinate_inr_baseline(
                hr_b,
                coarse_b,
                mode="multiclass",
                hidden_dim=48,
                epochs=60,
                seed=s,
                device=device,
                guide_normalization=norm,
            ),
        }
        _set_seeds(s)
        xf1 = XFuse(
            backbone=backbone,
            stride=stride,
            pretrained=pretrained,
            device=device,
            loss_fn="compositional_kl",
            interpolation_mode="cross_attn",
            fusion_method="spatial_mlp",
            learn_psf=True,
            guide_residual=True,
            tv_weight=0.008,
            lr=1e-2,
        )
        reps["xfuse"] = xf1.fit_predict(
            hr_b,
            coarse_b,
            epochs=28,
            mode="multiclass",
            tta_shifts=(),
            tta_flips=False,
            guide_normalization=norm,
        )
        for mk_name, rp in reps.items():
            sm = compute_segmentation_metrics(rp.hard_mask[0], dom_b)
            cm = _continuous_channel_metrics(rp.probabilities[0], simp_b)
            e1_acc[mk_name]["mIoU"].append(sm["iou"])
            e1_acc[mk_name]["BF1"].append(sm["boundary_f1"])
            for ck in ("pearson_r", "psnr_db", "support_iou"):
                e1_acc[mk_name][ck].append(cm[ck])

    # E2: CuZn 3D Core-Loss EELS Spectral Cube (33 energy channels, 4x spatial binning
    # performed on raw unstripped EELS counts BEFORE Egerton power-law continuum fit)
    hr_cz, cube_lr, gt_cuzn, cuzn_names = load_cuzn_eels_cube_sample(
        dest_root=zenodo_root, coarse_factor=4
    )
    hw_cz = (int(hr_cz.shape[0]), int(hr_cz.shape[1]))
    e2_methods = (
        "nmf_bicubic",
        "nmf_jbu",
        "nmf_cnmf",
        "nmf_guided_tv",
        "nmf_siren_inr",
        "xfuse_joint_spectral",
    )
    e2_acc = {
        m: {"pearson_r": [], "psnr_db": [], "support_iou": []} for m in e2_methods
    }

    for s in seeds:
        nmf_lr = _nmf_unmix_cube(cube_lr, n_components=2, seed=s)
        reps_e2: dict[str, Any] = {
            "nmf_bicubic": bicubic_upsample_baseline(
                nmf_lr, hw_cz, mode="multilabel", guide_normalization="preserve_counts"
            ),
            "nmf_jbu": jbu_guided_baseline(
                hr_cz, nmf_lr, mode="multilabel", guide_normalization="preserve_counts"
            ),
            "nmf_cnmf": cnmf_pansharpen_baseline(
                hr_cz,
                nmf_lr,
                mode="multilabel",
                n_iters=35,
                device=device,
                guide_normalization="preserve_counts",
            ),
            "nmf_guided_tv": guided_variational_sr_baseline(
                hr_cz,
                nmf_lr,
                mode="multilabel",
                n_steps=55,
                device=device,
                guide_normalization="preserve_counts",
            ),
            "nmf_siren_inr": coordinate_inr_baseline(
                hr_cz,
                nmf_lr,
                mode="multilabel",
                hidden_dim=48,
                epochs=60,
                seed=s,
                device=device,
                guide_normalization="preserve_counts",
            ),
        }
        _set_seeds(s)
        xf_eels = XFuse(
            backbone=backbone,
            stride=4,
            pretrained=pretrained,
            device=device,
            loss_fn="forward_psf_poisson",
            interpolation_mode="cross_attn",
            fusion_method="spatial_mlp",
            spectral_channels=int(cube_lr.shape[0]),
            guide_channels=2,
            learn_psf=True,
            guide_residual=True,
            tv_weight=0.006,
            lr=1e-2,
        )
        reps_e2["xfuse_joint_spectral"] = xf_eels.fit_predict(
            hr_cz,
            cube_lr,
            epochs=35,
            mode="multilabel",
            tta_shifts=(),
            tta_flips=True,
            spectral_unmix=True,
            n_components=2,
        )
        for mk_name, rp in reps_e2.items():
            cm2 = _perm_continuous_metrics(rp.probabilities[0], gt_cuzn)
            for ck in ("pearson_r", "psnr_db", "support_iou"):
                e2_acc[mk_name][ck].append(cm2[ck])

    # E3: EPFL edxia LC3 Cement SEM-BSE + 6-element Calibrated wt% EDS (8x Core+Halo
    # interaction-volume hold-out; Georget et al., Cem. Concr. Res. 2021)
    s_edxia = load_zenodo_correlative_sample(
        "edxia_cement_sem_eds", dest_root=zenodo_root
    )
    hr_edx, simp_edx, dom_edx = _prep_ti_rcca_simplex(s_edxia, y0=120, size=168)
    _, coarse_edx, _ = make_experimental_holdout(
        hr_edx, simp_edx[1:], factor=8, psf_sigma=2.2, beam_profile="core_halo"
    )
    e3_acc = {
        m: {"mIoU": [], "BF1": [], "pearson_r": [], "psnr_db": [], "support_iou": []}
        for m in e1_methods
    }
    hw_edx = (int(hr_edx.shape[0]), int(hr_edx.shape[1]))

    for s in seeds:
        reps_e3: dict[str, Any] = {
            "otsu": otsu_threshold_baseline(
                hr_edx, coarse_edx, mode="multiclass", guide_normalization=norm
            ),
            "bicubic": bicubic_upsample_baseline(
                coarse_edx, hw_edx, mode="multiclass", guide_normalization=norm
            ),
            "jbu": jbu_guided_baseline(
                hr_edx, coarse_edx, mode="multiclass", guide_normalization=norm
            ),
            "cnmf": cnmf_pansharpen_baseline(
                hr_edx,
                coarse_edx,
                mode="multiclass",
                n_iters=35,
                device=device,
                guide_normalization=norm,
            ),
            "guided_tv": guided_variational_sr_baseline(
                hr_edx,
                coarse_edx,
                mode="multiclass",
                n_steps=55,
                device=device,
                guide_normalization=norm,
            ),
            "siren_inr": coordinate_inr_baseline(
                hr_edx,
                coarse_edx,
                mode="multiclass",
                hidden_dim=48,
                epochs=60,
                seed=s,
                device=device,
                guide_normalization=norm,
            ),
        }
        _set_seeds(s)
        xf3 = XFuse(
            backbone=backbone,
            stride=stride,
            pretrained=pretrained,
            device=device,
            loss_fn="compositional_kl",
            interpolation_mode="cross_attn",
            fusion_method="spatial_mlp",
            psf_profile="core_halo",
            learn_psf=True,
            guide_residual=True,
            tv_weight=0.008,
            lr=1e-2,
        )
        reps_e3["xfuse"] = xf3.fit_predict(
            hr_edx,
            coarse_edx,
            epochs=28,
            mode="multiclass",
            tta_shifts=(),
            tta_flips=False,
            guide_normalization=norm,
        )
        for mk_name, rp in reps_e3.items():
            sm3 = compute_segmentation_metrics(rp.hard_mask[0], dom_edx)
            cm3 = _continuous_channel_metrics(rp.probabilities[0], simp_edx)
            e3_acc[mk_name]["mIoU"].append(sm3["iou"])
            e3_acc[mk_name]["BF1"].append(sm3["boundary_f1"])
            for ck in ("pearson_r", "psnr_db", "support_iou"):
                e3_acc[mk_name][ck].append(cm3[ck])

    return {
        "ti_rcca_5elem_stem_eds_8x": {
            "elements": s_b.channel_names,
            "source_type": "published_tiff_panels_simplex_normalized",
            "guide_residual": True,
            "methods": {k: {m: _mean_std(v[m]) for m in v} for k, v in e1_acc.items()},
        },
        "cuzn_3d_eels_cube_4x": {
            "elements": cuzn_names,
            "n_energy_channels": int(cube_lr.shape[0]),
            "native_si_hw": [int(hw_cz[0]), int(hw_cz[1])],
            "coarse_si_hw": [int(cube_lr.shape[1]), int(cube_lr.shape[2])],
            "continuum_stripped_after_coarse_binning": True,
            "guide_residual": True,
            "small_fov_min_input_size": 112,
            "methods": {k: {m: _mean_std(v[m]) for m in v} for k, v in e2_acc.items()},
        },
        "edxia_lc3_6elem_sem_eds_8x": {
            "elements": s_edxia.channel_names,
            "source_type": "calibrated_wt_pct_csv_simplex_normalized",
            "guide_residual": True,
            "psf_profile": "core_halo",
            "methods": {k: {m: _mean_std(v[m]) for m in v} for k, v in e3_acc.items()},
        },
    }


def render_ablation_figure(
    results: dict[str, Any], fig_dir: Path, artifact_dir: Path | None = None
) -> Path:
    """Render a 300-DPI 2x3 multi-panel summary figure across Stages A-E."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "legend.fontsize": 7.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.8), constrained_layout=True)

    # Panel A: Stage A Upsampler & Spin-2 Orientation
    ax = axes[0, 0]
    if "stage_a_upsampler" in results:
        sa = results["stage_a_upsampler"]
        vkeys = [
            "scalar_dilated_jbu",
            "spin2_structure_tensor",
            "spin2_plus_subtoken",
            "learned_spin2_subtoken",
        ]
        vlbls = [
            "Scalar\nJBU",
            "Spin-2\nTensor",
            "Spin-2 +\nSub-Token",
            "Learned\nSpin-2",
        ]
        x = np.arange(len(vkeys))
        w = 0.36
        for idx, (fk, lbl, col) in enumerate(
            [
                ("lamellar_8x", "8x Moffat", "#1f77b4"),
                ("lamellar_16x", "16x Moffat", "#d62728"),
            ]
        ):
            means = [sa[fk][k]["mIoU"]["mean"] for k in vkeys]
            stds = [sa[fk][k]["mIoU"]["std"] for k in vkeys]
            ax.bar(
                x + (idx - 0.5) * w,
                means,
                w,
                yerr=stds,
                capsize=3,
                label=lbl,
                color=col,
                alpha=0.88,
            )
        ax.set_xticks(x, vlbls)
        ax.set_ylabel("mIoU")
        ax.set_ylim(0.0, 1.02)
        ax.legend(frameon=False)
    ax.set_title("(a) Spin-2 & Sub-Token Upsampling")

    # Panel B: Stage B Forward PSF Profiles
    ax = axes[0, 1]
    if "stage_b_psf_physics" in results:
        sb = results["stage_b_psf_physics"]
        pkeys = ["fixed_gaussian", "learn_gaussian", "learn_moffat", "learn_core_halo"]
        plbls = ["Fixed\nGauss", "Learn\nGauss", "Learn\nMoffat", "Learn\nCore+Halo"]
        x = np.arange(len(pkeys))
        w = 0.26
        for idx, (bk, metric, lbl, col) in enumerate(
            [
                ("core_halo_probe_8x", "mIoU", "Core+Halo (mIoU)", "#ff7f0e"),
                ("moffat_probe_8x", "mIoU", "Moffat (mIoU)", "#2ca02c"),
                ("zn13x_synchrotron_8x", "pearson_r", "Zn13X (Pearson r)", "#9467bd"),
            ]
        ):
            means = [sb[bk][k][metric]["mean"] for k in pkeys]
            stds = [sb[bk][k][metric]["std"] for k in pkeys]
            ax.bar(
                x + (idx - 1.0) * w,
                means,
                w,
                yerr=stds,
                capsize=2.5,
                label=lbl,
                color=col,
                alpha=0.88,
            )
        ax.set_xticks(x, plbls)
        ax.set_ylabel("Score (mIoU / Pearson r)")
        ax.set_ylim(0.0, 1.22)
        ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_title("(b) Forward PSF Physics Profiles")

    # Panel C: Stage C Volumetric Depth-PSF & Tri-Plane
    ax = axes[0, 2]
    if "stage_c_volumetric" in results:
        sc = results["stage_c_volumetric"]["volumetric_variants"]
        ckeys = ["sparse_25d_baseline", "depth_psf_deconv", "depth_psf_plus_triplane"]
        clbls = [
            "2.5D Slab\nBaseline",
            "+ Axial\nDepth-PSF",
            "+ Tri-Plane\nConsistency",
        ]
        x = np.arange(len(ckeys))
        w = 0.26
        for idx, (mk, lbl, col) in enumerate(
            [
                ("mIoU_3d", "3D mIoU", "#1f77b4"),
                ("mIoU_unguided", "Unguided-Z mIoU", "#d62728"),
                ("z_consistency", "Z-Consistency", "#2ca02c"),
            ]
        ):
            means = [sc[k][mk]["mean"] for k in ckeys]
            stds = [sc[k][mk]["std"] for k in ckeys]
            ax.bar(
                x + (idx - 1.0) * w,
                means,
                w,
                yerr=stds,
                capsize=2.5,
                label=lbl,
                color=col,
                alpha=0.88,
            )
        ax.set_xticks(x, clbls)
        ax.set_ylabel("Volumetric Score")
        ax.set_ylim(0.0, 1.18)
        ax.legend(frameon=False, loc="upper left")
    ax.set_title("(c) 3D Depth-PSF & Tri-Plane Fusion")

    # Panel D: Stage D Conformal Efficiency & Cross-Specimen Coverage
    ax = axes[1, 0]
    if "stage_d_conformal" in results:
        sd_syn = results["stage_d_conformal"]["synthetic_lamellar_8x"]
        sd_eds = results["stage_d_conformal"]["real_ti_rcca_fig3a_to_fig3b_5class"]
        c_specs = [
            ("raw_lac", "Raw LAC", "#7f7f7f"),
            ("raw_aps", "Raw APS", "#ff7f0e"),
            ("temp_raps_block", "Temp Block RAPS", "#d62728"),
        ]
        x = np.arange(len(c_specs))
        w = 0.25
        lam_cov = [
            sd_syn[ck]["0.10"]["empirical_coverage"]["mean"] for ck, _, _ in c_specs
        ]
        eds_cov = [
            sd_eds[ck]["0.20"]["empirical_coverage"]["mean"] for ck, _, _ in c_specs
        ]
        lam_eff = [
            sd_syn[ck]["0.10"]["interior_singleton_fraction"]["mean"]
            for ck, _, _ in c_specs
        ]
        ax.bar(x - w, lam_cov, w, label="Lamellar Cov (90% nom.)", color="#1f77b4")
        ax.bar(x, eds_cov, w, label="Ti-RCCA Cov (80% nom.)", color="#2ca02c")
        ax.bar(x + w, lam_eff, w, label="Interior Singleton %", color="#9467bd")
        ax.axhline(0.90, color="#1f77b4", linestyle="--", linewidth=0.9, alpha=0.6)
        ax.axhline(0.80, color="#2ca02c", linestyle="--", linewidth=0.9, alpha=0.6)
        ax.set_xticks(x, [lbl for _, lbl, _ in c_specs])
        ax.set_ylabel("Coverage / Singleton Fraction")
        ax.set_ylim(0.0, 1.22)
        ax.legend(frameon=False, loc="upper left", fontsize=7.4)
    ax.set_title("(d) Conformal Validity & Interior Sharpness")

    # Panel E1: Real 5-Element Ti-RCCA STEM-EDS (8x Hold-Out)
    ax = axes[1, 1]
    if "stage_e_real_em" in results:
        se1 = results["stage_e_real_em"]["ti_rcca_5elem_stem_eds_8x"]["methods"]
        se3 = (
            results["stage_e_real_em"]
            .get("edxia_lc3_6elem_sem_eds_8x", {})
            .get("methods")
        )
        mkeys = ["otsu", "bicubic", "jbu", "cnmf", "guided_tv", "siren_inr", "xfuse"]
        mlbls = ["Otsu", "Bicub", "JBU", "CNMF", "TV", "SIREN", "X-FUSE"]
        x = np.arange(len(mkeys))
        r_m = [se1[k]["pearson_r"]["mean"] for k in mkeys]
        r_s = [se1[k]["pearson_r"]["std"] for k in mkeys]
        iou_m = [se1[k]["mIoU"]["mean"] for k in mkeys]
        iou_s = [se1[k]["mIoU"]["std"] for k in mkeys]
        if se3 is not None:
            w = 0.26
            e_m = [se3[k]["pearson_r"]["mean"] for k in mkeys]
            e_s = [se3[k]["pearson_r"]["std"] for k in mkeys]
            ax.bar(
                x - w,
                r_m,
                w,
                yerr=r_s,
                capsize=2.2,
                label="Ti-RCCA STEM-EDS (r)",
                color="#1f77b4",
            )
            ax.bar(
                x,
                e_m,
                w,
                yerr=e_s,
                capsize=2.2,
                label="edxia SEM-EDS (r)",
                color="#ff7f0e",
            )
            ax.bar(
                x + w,
                iou_m,
                w,
                yerr=iou_s,
                capsize=2.2,
                label="Ti-RCCA 5-Phase mIoU",
                color="#d62728",
            )
        else:
            w = 0.36
            ax.bar(
                x - 0.5 * w,
                r_m,
                w,
                yerr=r_s,
                capsize=2.5,
                label="5-Elem Pearson r",
                color="#1f77b4",
            )
            ax.bar(
                x + 0.5 * w,
                iou_m,
                w,
                yerr=iou_s,
                capsize=2.5,
                label="5-Phase mIoU",
                color="#d62728",
            )
        ax.set_xticks(x, mlbls, rotation=20)
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.08)
        ax.legend(frameon=False, loc="upper left", fontsize=7.2)
    ax.set_title("(e) Real STEM-EDS & SEM-EDS (8x Hold-Out)")

    # Panel E2: Real CuZn 3D Core-Loss EELS Spectral Cube (4x)
    ax = axes[1, 2]
    if "stage_e_real_em" in results:
        se2 = results["stage_e_real_em"]["cuzn_3d_eels_cube_4x"]["methods"]
        ekeys = [
            "nmf_bicubic",
            "nmf_jbu",
            "nmf_cnmf",
            "nmf_guided_tv",
            "nmf_siren_inr",
            "xfuse_joint_spectral",
        ]
        elbls = [
            "NMF+\nBicub",
            "NMF+\nJBU",
            "NMF+\nCNMF",
            "NMF+\nTV",
            "NMF+\nSIREN",
            "X-FUSE\nJoint",
        ]
        x = np.arange(len(ekeys))
        w = 0.36
        rm = [se2[k]["pearson_r"]["mean"] for k in ekeys]
        rs = [se2[k]["pearson_r"]["std"] for k in ekeys]
        sm = [se2[k]["support_iou"]["mean"] for k in ekeys]
        ss = [se2[k]["support_iou"]["std"] for k in ekeys]
        ax.bar(
            x - 0.5 * w, rm, w, yerr=rs, capsize=2.5, label="Pearson r", color="#2ca02c"
        )
        ax.bar(
            x + 0.5 * w,
            sm,
            w,
            yerr=ss,
            capsize=2.5,
            label="Support IoU",
            color="#9467bd",
        )
        ax.set_xticks(x, elbls)
        ax.set_ylabel("Cu/Zn L2,3 Recovery")
        ax.set_ylim(0.0, 1.02)
        ax.legend(frameon=False, loc="upper left")
    ax.set_title("(f) Real CuZn 3D Core-Loss EELS Cube (4x)")

    out_png = fig_dir / "design_ablations_summary.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_png, artifact_dir / out_png.name)
    return out_png


def run_design_ablations(
    output_dir: Path,
    artifact_dir: Path | None = None,
    zenodo_root: Path = Path("/home/antigravity/.cache-xfuse/zenodo"),
    backbone: str = "dinov2_vits14",
    pretrained: bool = True,
    stride: int = 7,
    device: str = "cpu",
    seeds: tuple[int, ...] = (42, 43, 44),
    resume: bool = True,
    stages: set[str] | None = None,
) -> dict[str, Any]:
    _install_ram_backbone_cache()
    active = stages if stages is not None else {"A", "B", "C", "D", "E", "plot"}
    json_path = output_dir / "design_ablations_results.json"
    results: dict[str, Any] = {}
    if resume and json_path.exists():
        results = json.loads(json_path.read_text(encoding="utf-8"))
    results.setdefault(
        "config",
        {
            "backbone": backbone,
            "pretrained": pretrained,
            "stride": stride,
            "seeds": list(seeds),
        },
    )

    t0 = time.perf_counter()
    if "A" in active and not (resume and "stage_a_upsampler" in results):
        print("[Stage A] Running Spin-2 & Sub-Token Upsampler Ablation...", flush=True)
        results["stage_a_upsampler"] = run_stage_a_upsampler(
            seeds, backbone, pretrained, stride, device
        )
        _save_json(output_dir, results)

    if "B" in active and not (resume and "stage_b_psf_physics" in results):
        print("[Stage B] Running Forward PSF Profile Ablation...", flush=True)
        results["stage_b_psf_physics"] = run_stage_b_psf_physics(
            seeds, backbone, pretrained, stride, device
        )
        _save_json(output_dir, results)

    if ("C" in active or "C1" in active) and not (
        resume
        and "stage_c_volumetric" in results
        and "volumetric_variants" in results["stage_c_volumetric"]
    ):
        print(
            "[Stage C1] Running 3D Volumetric Depth-PSF & Tri-Plane Ablation...",
            flush=True,
        )
        c1_out = run_stage_c_volumetric(seeds, backbone, pretrained, stride, device)
        results.setdefault("stage_c_volumetric", {}).update(c1_out)
        _save_json(output_dir, results)

    if ("C" in active or "C2" in active) and not (
        resume
        and "stage_c_volumetric" in results
        and "localized_reaction_active_curve_z48" in results["stage_c_volumetric"]
    ):
        print(
            "[Stage C2] Running Z=48 Active Tomography Steering Curve...",
            flush=True,
        )
        c2_out = run_stage_c2_active_curve(seeds, backbone, pretrained, stride, device)
        results.setdefault("stage_c_volumetric", {})[
            "localized_reaction_active_curve_z48"
        ] = c2_out
        _save_json(output_dir, results)

    if "D" in active and not (resume and "stage_d_conformal" in results):
        print("[Stage D] Running Conformal Calibration Ablation...", flush=True)
        results["stage_d_conformal"] = run_stage_d_conformal(
            seeds, backbone, pretrained, stride, device, zenodo_root
        )
        _save_json(output_dir, results)

    if "E" in active and not (resume and "stage_e_real_em" in results):
        print(
            "[Stage E] Running Real Zenodo STEM-EDS & 3D EELS Benchmarks...", flush=True
        )
        results["stage_e_real_em"] = run_stage_e_real_em(
            seeds, backbone, pretrained, stride, device, zenodo_root
        )
        _save_json(output_dir, results)

    results["last_wall_clock_s"] = round(time.perf_counter() - t0, 2)
    if "plot" in active:
        fig_p = render_ablation_figure(results, output_dir / "figures", artifact_dir)
        results["summary_figure"] = str(fig_p)
        _save_json(output_dir, results)

    print(
        f"Completed stages {sorted(active)} in {results['last_wall_clock_s']:.1f}s.",
        flush=True,
    )
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Run X-FUSE next-gen design ablations.")
    p.add_argument(
        "--output-dir", type=Path, default=Path("benchmarks/outputs/design_ablations")
    )
    p.add_argument("--artifact-dir", type=Path, default=None)
    p.add_argument(
        "--zenodo-root",
        type=Path,
        default=Path("/home/antigravity/.cache-xfuse/zenodo"),
    )
    p.add_argument("--backbone", default="dinov2_vits14")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--stride", type=int, default=7)
    p.add_argument("--device", default="cpu")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--stages", default=None, help="Comma-separated subset: A,B,C,D,E,plot"
    )
    args = p.parse_args()
    st = (
        {s.strip() for s in args.stages.split(",") if s.strip()}
        if args.stages
        else None
    )
    run_design_ablations(
        output_dir=args.output_dir,
        artifact_dir=args.artifact_dir,
        zenodo_root=args.zenodo_root,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        stride=args.stride,
        device=args.device,
        resume=bool(args.resume),
        stages=st,
    )


if __name__ == "__main__":
    main()
