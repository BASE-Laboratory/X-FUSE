"""Task T2 4-Fold Leave-One-Slice-Out (LOSO) Correlative XRD-CT Benchmark Runner.

Evaluates classical interpolation baselines (`Nearest`, `Bilinear`, `Bicubic`),
edge-preserving `Guided Filter`, and `X-FUSE` (`2D JBU`, `2D Cross-Attn`, and
`3D Sparse-Slice Volumetric`) on the 48 real 32-bit Diamond K11 DIAD Zn-13X
NeXus reconstructions (`10.5281/zenodo.13329639`), round-tripping `X-FUSE`
outputs through calibrated OME-Zarr v0.4 (`.ome.zarr`) stores via `basecoat`.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from basecoat import read_ome_zarr
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

from x_fuse import (
    VolumetricFuser,
    XFuse,
    export_report_ome_zarr,
    load_diad_xrdct,
)

SCAN_IDS_50UM: tuple[int, ...] = tuple(
    list(range(43336, 43348)) + list(range(43393, 43401))
)
PAIRED_25_TO_50: tuple[tuple[int, int, int], ...] = (
    (43324, 43345, 9),
    (43322, 43395, 14),
    (43323, 43398, 17),
    (43325, 43399, 18),
)


def resample_50um_to_grid(
    img_20x20: np.ndarray, target_size: int = 40, order: int = 3
) -> np.ndarray:
    """Resample a 20x20 50 um DIAD slice onto ``(target_size, target_size)``
    preserving exact physical ``kb_cs_x`` endpoints ``[-487.5, +487.5] um``.
    """
    arr = np.asarray(img_20x20, dtype=np.float64)
    u = np.linspace(0.0, float(arr.shape[0] - 1), int(target_size), dtype=np.float64)
    yy, xx = np.meshgrid(u, u, indexing="ij")
    warped = map_coordinates(arr, [yy, xx], order=order, mode="nearest")
    return np.clip(warped, 0.0, None).astype(np.float32)


def _fourier_bandlimit_2d(
    img_2d: np.ndarray, cutoff_cyc_per_px: float = 0.43, rolloff: float = 0.08
) -> np.ndarray:
    arr = np.asarray(img_2d, dtype=np.float64)
    h, w = arr.shape
    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    ky, kx = np.meshgrid(fy, fx, indexing="ij")
    kr = np.sqrt(ky**2 + kx**2)
    r0 = max(0.0, float(cutoff_cyc_per_px) - 0.5 * float(rolloff))
    r1 = float(cutoff_cyc_per_px) + 0.5 * float(rolloff)
    filt = np.ones_like(kr)
    trans = (kr > r0) & (kr < r1)
    filt[trans] = 0.5 * (1.0 + np.cos(np.pi * (kr[trans] - r0) / max(r1 - r0, 1e-8)))
    filt[kr >= r1] = 0.0
    return np.real(np.fft.ifft2(np.fft.fft2(arr) * filt)).astype(np.float64)


def compute_t2_slice_metrics(
    pred_zn: np.ndarray,
    pred_na: np.ndarray,
    ref_zn: np.ndarray,
    ref_na: np.ndarray,
    *,
    hr_zn: np.ndarray | None = None,
    hr_na: np.ndarray | None = None,
    hr_pore_mask: np.ndarray | None = None,
    cutoff_cyc_per_px: float = 0.43,
    pixel_size_um: float = 25.0,
) -> dict[str, float]:
    """Exact Task T2 25 um band-limited RMSE, reaction-front & pore-leakage scorer."""
    zp = np.clip(np.asarray(pred_zn, dtype=np.float64), 0.0, None)
    np_ = np.clip(np.asarray(pred_na, dtype=np.float64), 0.0, None)
    zr = np.clip(np.asarray(ref_zn, dtype=np.float64), 0.0, None)
    nr = np.clip(np.asarray(ref_na, dtype=np.float64), 0.0, None)

    zp_bl = _fourier_bandlimit_2d(zp, cutoff_cyc_per_px=cutoff_cyc_per_px)
    zr_bl = _fourier_bandlimit_2d(zr, cutoff_cyc_per_px=cutoff_cyc_per_px)
    np_bl = _fourier_bandlimit_2d(np_, cutoff_cyc_per_px=cutoff_cyc_per_px)
    nr_bl = _fourier_bandlimit_2d(nr, cutoff_cyc_per_px=cutoff_cyc_per_px)

    zn_rmse = float(np.sqrt(np.mean((zp - zr) ** 2)))
    na_rmse = float(np.sqrt(np.mean((np_ - nr) ** 2)))
    zn_bl_rmse = float(np.sqrt(np.mean((zp_bl - zr_bl) ** 2)))
    na_bl_rmse = float(np.sqrt(np.mean((np_bl - nr_bl) ** 2)))

    zn_bl_r = (
        float(np.corrcoef(zp_bl.ravel(), zr_bl.ravel())[0, 1])
        if (np.std(zp_bl) > 1e-12 and np.std(zr_bl) > 1e-12)
        else 0.0
    )
    na_bl_r = (
        float(np.corrcoef(np_bl.ravel(), nr_bl.ravel())[0, 1])
        if (np.std(np_bl) > 1e-12 and np.std(nr_bl) > 1e-12)
        else 0.0
    )

    tot_r = zr + nr
    p95 = float(np.percentile(tot_r, 95.0)) if tot_r.size > 0 else 0.0
    pellet = tot_r > max(0.20 * p95, 1e-8)
    if int(np.count_nonzero(pellet)) < 8:
        pellet = np.ones_like(tot_r, dtype=bool)

    corr_ref = (
        float(np.corrcoef(zr[pellet], nr[pellet])[0, 1])
        if (np.std(zr[pellet]) > 1e-10 and np.std(nr[pellet]) > 1e-10)
        else 0.0
    )
    corr_pred = (
        float(np.corrcoef(zp[pellet], np_[pellet])[0, 1])
        if (np.std(zp[pellet]) > 1e-10 and np.std(np_[pellet]) > 1e-10)
        else 0.0
    )

    depth_um = distance_transform_edt(pellet) * float(pixel_size_um)
    f_zn_r = zr / np.maximum(zr + nr, 1e-8)
    f_zn_p = zp / np.maximum(zp + np_, 1e-8)

    has_front_r = bool(np.any(f_zn_r[pellet] >= 0.5) and np.any(f_zn_r[pellet] < 0.5))
    has_front_p = bool(np.any(f_zn_p[pellet] >= 0.5) and np.any(f_zn_p[pellet] < 0.5))
    if has_front_r and has_front_p:
        b_r = pellet & (np.abs(f_zn_r - 0.5) <= 0.18)
        b_p = pellet & (np.abs(f_zn_p - 0.5) <= 0.18)
        if np.any(b_r) and np.any(b_p):
            dt_to_r = distance_transform_edt(~b_r) * float(pixel_size_um)
            dt_to_p = distance_transform_edt(~b_p) * float(pixel_size_um)
            front_err_um = 0.5 * (
                float(np.mean(dt_to_r[b_p])) + float(np.mean(dt_to_p[b_r]))
            )
        else:
            w_r = float(
                np.sum(f_zn_r[pellet] * depth_um[pellet])
                / (np.sum(f_zn_r[pellet]) + 1e-8)
            )
            w_p = float(
                np.sum(f_zn_p[pellet] * depth_um[pellet])
                / (np.sum(f_zn_p[pellet]) + 1e-8)
            )
            front_err_um = abs(w_p - w_r)
    else:
        w_r = float(
            np.sum(f_zn_r[pellet] * depth_um[pellet]) / (np.sum(f_zn_r[pellet]) + 1e-8)
        )
        w_p = float(
            np.sum(f_zn_p[pellet] * depth_um[pellet]) / (np.sum(f_zn_p[pellet]) + 1e-8)
        )
        front_err_um = abs(w_p - w_r)

    out: dict[str, float] = {
        "mean_rmse": 0.5 * (zn_rmse + na_rmse),
        "mean_bandlimited_rmse": 0.5 * (zn_bl_rmse + na_bl_rmse),
        "mean_bandlimited_pearson_r": 0.5 * (zn_bl_r + na_bl_r),
        "in_pellet_na_zn_corr_abs_err": abs(corr_pred - corr_ref),
        "reaction_front_pos_err_um": float(front_err_um),
    }
    if hr_zn is not None and hr_na is not None and hr_pore_mask is not None:
        pm = np.asarray(hr_pore_mask, dtype=bool)
        tot_hr = np.abs(hr_zn) + np.abs(hr_na)
        scale_hr = float(np.percentile(tot_hr, 90.0)) + 1e-8
        out["pore_leakage_rel"] = (
            float(np.mean(tot_hr[pm]) / scale_hr) if np.any(pm) else 0.0
        )
    return out


def guided_filter_2d(
    guide_2d: np.ndarray, src_2d: np.ndarray, radius: int = 2, eps: float = 1e-3
) -> np.ndarray:
    """Classical edge-preserving He et al. (2013) Guided Filter baseline."""
    from scipy.ndimage import uniform_filter

    k = 2 * int(radius) + 1
    g = np.asarray(guide_2d, dtype=np.float64)
    p = np.asarray(src_2d, dtype=np.float64)
    mean_g = uniform_filter(g, size=k, mode="nearest")
    mean_p = uniform_filter(p, size=k, mode="nearest")
    corr_gg = uniform_filter(g * g, size=k, mode="nearest")
    corr_gp = uniform_filter(g * p, size=k, mode="nearest")
    var_g = np.maximum(corr_gg - mean_g * mean_g, 0.0)
    cov_gp = corr_gp - mean_g * mean_p
    a = cov_gp / (var_g + float(eps))
    b = mean_p - a * mean_g
    mean_a = uniform_filter(a, size=k, mode="nearest")
    mean_b = uniform_filter(b, size=k, mode="nearest")
    return np.clip(mean_a * g + mean_b, 0.0, None).astype(np.float32)


def synthesize_porous_pellet_structural_slab(
    tot_50um_zyx: np.ndarray,
    up_factor: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a 160x160 (6.25 um) structural slab co-registered with the DIAD
    cylindrical pellet & exterior capillary air void for pre-Stage-0C testing.
    """
    rng = np.random.default_rng(seed)
    z_dim, h50, w50 = tot_50um_zyx.shape
    h_hr, w_hr = h50 * up_factor, w50 * up_factor
    hr_vol = np.zeros((z_dim, h_hr, w_hr), dtype=np.float32)
    pore_vol = np.zeros((z_dim, h_hr, w_hr), dtype=bool)

    # Estimate consistent cylindrical capillary pellet support from axial median
    med_50 = np.median(tot_50um_zyx, axis=0)
    med_hr = resample_50um_to_grid(med_50, target_size=h_hr, order=3)
    p95_med = float(np.percentile(med_hr, 95.0)) + 1e-8
    exterior_void = (med_hr / p95_med) < 0.08

    for iz in range(z_dim):
        env_hr = resample_50um_to_grid(tot_50um_zyx[iz], target_size=h_hr, order=3)
        p95 = float(np.percentile(env_hr, 95.0)) + 1e-8
        env_norm = np.clip(env_hr / p95, 0.0, 1.2)
        void_mask = exterior_void & (env_norm < 0.08)
        solid_att = (
            0.65
            + 0.30 * env_norm
            + 0.01 * gaussian_filter(rng.standard_normal((h_hr, w_hr)), sigma=1.0)
        )
        img = np.where(void_mask, 0.04, solid_att).astype(np.float32)
        hr_vol[iz] = np.clip(img, 0.0, 1.5)
        pore_vol[iz] = void_mask
    return hr_vol, pore_vol


def _fit_loso_flux_alpha(
    d50: dict[str, np.ndarray],
    d25: dict[str, np.ndarray],
    idx_25: dict[int, int],
    heldout_s25: int,
    order_interp: int,
) -> float:
    num_f, den_f = 0.0, 0.0
    for tr_s25, _, tr_z in PAIRED_25_TO_50:
        if tr_s25 == heldout_s25:
            continue
        tr_i25 = idx_25[tr_s25]
        u_zn = resample_50um_to_grid(d50["zn"][tr_z], 40, order_interp)
        u_na = resample_50um_to_grid(d50["na"][tr_z], 40, order_interp)
        t25 = np.clip(d25["zn"][tr_i25], 0.0, None) + np.clip(
            d25["na"][tr_i25], 0.0, None
        )
        t50 = u_zn + u_na
        num_f += float(np.sum(t25 * t50))
        den_f += float(np.sum(t50 * t50))
    return num_f / max(den_f, 1e-12)


def _aggregate_loso_table(
    per_fold: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    keys = [
        "mean_bandlimited_rmse",
        "mean_rmse",
        "mean_bandlimited_pearson_r",
        "in_pellet_na_zn_corr_abs_err",
        "reaction_front_pos_err_um",
    ]
    if "pore_leakage_rel" in next(iter(per_fold.values()))["metrics"]:
        keys.append("pore_leakage_rel")
    confirmed = ["43322", "43323", "43325"]
    all_four = ["43324", "43322", "43323", "43325"]
    return {
        "per_slice": per_fold,
        "mean_3_confirmed": {
            k: round(float(np.mean([per_fold[s]["metrics"][k] for s in confirmed])), 4)
            for k in keys
        },
        "mean_all_4": {
            k: round(float(np.mean([per_fold[s]["metrics"][k] for s in all_four])), 4)
            for k in keys
        },
    }


def run_t2_loso_benchmark(
    xrdct_dir: Path,
    output_json: Path,
    *,
    epochs_2d: int = 25,
    epochs_3d: int = 12,
    backbone: str = "timm:vit_tiny_patch16_224",
    pretrained: bool = False,
) -> dict[str, Any]:
    torch.manual_seed(42)
    d50 = load_diad_xrdct(
        xrdct_dir,
        phases=("zn", "na"),
        scan_ids=SCAN_IDS_50UM,
        aggregation="positive_sum",
        correct_motor_offset_deg=47.5,
    )
    d25 = load_diad_xrdct(
        xrdct_dir,
        phases=("zn", "na"),
        scan_ids=[s25 for s25, _, _ in PAIRED_25_TO_50],
        aggregation="positive_sum",
        correct_motor_offset_deg=0.0,
    )
    idx_25 = {s25: i for i, (s25, _, _) in enumerate(PAIRED_25_TO_50)}

    # Build 20-slice 80x80 (12.5 um) HR structural slab (4x over 50 um, 2x over 25 um)
    tot_50_zyx = np.clip(d50["zn"], 0.0, None) + np.clip(d50["na"], 0.0, None)
    hr_vol_zyx, hr_pore_zyx = synthesize_porous_pellet_structural_slab(
        tot_50_zyx, up_factor=4, seed=42
    )

    methods_per_fold: dict[str, dict[str, dict[str, Any]]] = {
        "Nearest": {},
        "Bilinear": {},
        "Bicubic": {},
        "GuidedFilter": {},
        "XFuse_2D_JBU_OMEZarr": {},
        "XFuse_2D_CrossAttn_OMEZarr": {},
        "XFuse_3D_LOSO_OMEZarr": {},
    }

    with tempfile.TemporaryDirectory() as tmp_td:
        td_path = Path(tmp_td)
        for fold_idx, (test_s25, test_s50, test_z) in enumerate(PAIRED_25_TO_50):
            i25 = idx_25[test_s25]
            ref_zn = np.clip(d25["zn"][i25], 0.0, None)
            ref_na = np.clip(d25["na"][i25], 0.0, None)

            alpha_bic = _fit_loso_flux_alpha(d50, d25, idx_25, test_s25, 3)
            for m_cls, ord_i in (("Nearest", 0), ("Bilinear", 1), ("Bicubic", 3)):
                a_f = _fit_loso_flux_alpha(d50, d25, idx_25, test_s25, ord_i)
                p_zn = resample_50um_to_grid(d50["zn"][test_z], 40, ord_i) * a_f
                p_na = resample_50um_to_grid(d50["na"][test_z], 40, ord_i) * a_f
                methods_per_fold[m_cls][str(test_s25)] = {
                    "fold_index": fold_idx,
                    "matched_50um_scan": test_s50,
                    "z_index": test_z,
                    "alpha": round(a_f, 5),
                    "metrics": compute_t2_slice_metrics(p_zn, p_na, ref_zn, ref_na),
                }

            # Guided Filter on 40x40 pooled structural guide
            hr_2d = hr_vol_zyx[test_z]
            pm_2d = hr_pore_zyx[test_z]
            struct_40 = hr_2d.reshape(40, 2, 40, 2).mean(axis=(1, 3))
            bic_zn = resample_50um_to_grid(d50["zn"][test_z], 40, 3) * alpha_bic
            bic_na = resample_50um_to_grid(d50["na"][test_z], 40, 3) * alpha_bic
            gf_zn = guided_filter_2d(struct_40, bic_zn, radius=2, eps=1e-3)
            gf_na = guided_filter_2d(struct_40, bic_na, radius=2, eps=1e-3)
            methods_per_fold["GuidedFilter"][str(test_s25)] = {
                "fold_index": fold_idx,
                "matched_50um_scan": test_s50,
                "z_index": test_z,
                "alpha": round(alpha_bic, 5),
                "metrics": compute_t2_slice_metrics(gf_zn, gf_na, ref_zn, ref_na),
            }

            guide_50_cal = np.stack(
                [
                    resample_50um_to_grid(d50["zn"][test_z], 20, 3) * alpha_bic,
                    resample_50um_to_grid(d50["na"][test_z], 20, 3) * alpha_bic,
                ],
                axis=0,
            ).astype(np.float32)

            for v_key, upsamp in (
                ("XFuse_2D_JBU_OMEZarr", "jbu"),
                ("XFuse_2D_CrossAttn_OMEZarr", "cross_attn"),
            ):
                torch.manual_seed(42 + fold_idx)
                fuser_2d = XFuse(
                    backbone=backbone,
                    fusion_method="spatial_mlp",
                    interpolation_mode=upsamp,
                    loss_fn="forward_psf_mse",
                    guide_residual=True,
                    pore_mask=pm_2d,
                    conformal_alpha=0.10,
                    min_input_size=80,
                    stride=4,
                    pretrained=pretrained,
                )
                rep_2d = fuser_2d.fit_predict(
                    hr_2d,
                    guide_50_cal,
                    epochs=epochs_2d,
                    mode="multilabel",
                    tta_shifts=(),
                    tta_flips=False,
                    pore_mask=pm_2d,
                )
                phys_hr_2d = rep_2d.reconstruct_physical_maps()
                zarr_2d = export_report_ome_zarr(
                    rep_2d,
                    td_path / f"{v_key}_f{fold_idx}.ome.zarr",
                    channel_names=["zn", "na"],
                    pixel_size_um=12.5,
                    target_grid_shape=(40, 40),
                )
                vset_2d = read_ome_zarr(zarr_2d, as_set=True)
                p_zn_2d = np.asarray(vset_2d.image.data[0])
                p_na_2d = np.asarray(vset_2d.image.data[1])
                methods_per_fold[v_key][str(test_s25)] = {
                    "fold_index": fold_idx,
                    "matched_50um_scan": test_s50,
                    "z_index": test_z,
                    "alpha": round(alpha_bic, 5),
                    "metrics": compute_t2_slice_metrics(
                        p_zn_2d,
                        p_na_2d,
                        ref_zn,
                        ref_na,
                        hr_zn=phys_hr_2d[0],
                        hr_na=phys_hr_2d[1],
                        hr_pore_mask=pm_2d,
                    ),
                }

            # 3D Sparse-Slice LOSO on the 3 held-in 25 um training slices
            torch.manual_seed(100 + fold_idx)
            sparse_3d: dict[int, np.ndarray] = {}
            for tr_s25, _, tr_z in PAIRED_25_TO_50:
                if tr_s25 == test_s25:
                    continue
                tr_i25 = idx_25[tr_s25]
                sparse_3d[tr_z] = np.stack(
                    [
                        np.clip(d25["zn"][tr_i25], 0.0, None),
                        np.clip(d25["na"][tr_i25], 0.0, None),
                    ],
                    axis=0,
                ).astype(np.float32)

            vf = VolumetricFuser(
                XFuse(
                    backbone=backbone,
                    fusion_method="spatial_mlp",
                    interpolation_mode="cross_attn",
                    loss_fn="forward_psf_mse",
                    guide_residual=True,
                    conformal_alpha=0.10,
                    min_input_size=80,
                    stride=8,
                    pretrained=pretrained,
                ),
                tta_shifts=None,
                tta_flips=False,
            )
            vf.fit_sparse_slices(
                hr_vol_zyx,
                sparse_guides=sparse_3d,
                epochs=epochs_3d,
                pore_mask=hr_pore_zyx,
                depth_psf_sigma=1.2,
                learn_depth_psf=True,
                triplane_consistency_weight=0.0,
            )
            coarse_3d_cal = np.stack(
                [
                    np.stack(
                        [
                            resample_50um_to_grid(d50["zn"][z_i], 20, 3) * alpha_bic
                            for z_i in range(len(SCAN_IDS_50UM))
                        ],
                        axis=0,
                    ),
                    np.stack(
                        [
                            resample_50um_to_grid(d50["na"][z_i], 20, 3) * alpha_bic
                            for z_i in range(len(SCAN_IDS_50UM))
                        ],
                        axis=0,
                    ),
                ],
                axis=0,
            ).astype(np.float32)
            rep_3d = vf.predict_volume(
                hr_vol_zyx,
                coarse_guide_3d=coarse_3d_cal,
                mode="2.5d_slab",
                sparse_guides=sparse_3d,
                pred_mode="multilabel",
                pore_mask=hr_pore_zyx,
            )
            phys_hr_3d = rep_3d.reconstruct_physical_maps()
            zarr_3d = export_report_ome_zarr(
                rep_3d,
                td_path / f"xfuse3d_f{fold_idx}.ome.zarr",
                channel_names=["zn", "na"],
                pixel_size_um=12.5,
                target_grid_shape=(40, 40),
                eval_slice_z=test_z,
            )
            vset_3d = read_ome_zarr(zarr_3d, as_set=True)
            methods_per_fold["XFuse_3D_LOSO_OMEZarr"][str(test_s25)] = {
                "fold_index": fold_idx,
                "matched_50um_scan": test_s50,
                "z_index": test_z,
                "alpha": round(alpha_bic, 5),
                "metrics": compute_t2_slice_metrics(
                    np.asarray(vset_3d.image.data[0]),
                    np.asarray(vset_3d.image.data[1]),
                    ref_zn,
                    ref_na,
                    hr_zn=phys_hr_3d[0, test_z],
                    hr_na=phys_hr_3d[1, test_z],
                    hr_pore_mask=pm_2d,
                ),
            }

    summary: dict[str, Any] = {
        "protocol": (
            "Task T2 4-Fold LOSO (DIAD Zn-13X DAWN positive_sum + "
            "+47.5 deg + OME-Zarr v0.4)"
        ),
        "note": (
            "HR structural slab uses porous pellet phantom pending SCARF "
            "Stage 0C 0.54 um DIAD XCT alignment."
        ),
        "methods": {
            m: _aggregate_loso_table(folds) for m, folds in methods_per_fold.items()
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Task T2 4-fold LOSO XRD-CT + OME-Zarr evaluation."
    )
    parser.add_argument(
        "--xrdct-dir",
        type=Path,
        default=Path("/home/antigravity/.cache-xfuse/zenodo/diad_zn13x_xrdct"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/outputs/t2_loso_xfuse_eval.json"),
    )
    parser.add_argument("--epochs-2d", type=int, default=10)
    parser.add_argument("--epochs-3d", type=int, default=6)
    parser.add_argument("--backbone", default="timm:vit_tiny_patch16_224")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    res = run_t2_loso_benchmark(
        args.xrdct_dir,
        args.output_json,
        epochs_2d=args.epochs_2d,
        epochs_3d=args.epochs_3d,
        backbone=args.backbone,
        pretrained=bool(args.pretrained),
    )
    hdr = (
        f"{'Method':<28} | {'BL-RMSE (3c/4a)':<17} | "
        f"{'Front um (3c/4a)':<18} | {'|dr| (3c/4a)':<15} | {'PoreLeak'}"
    )
    print(hdr)
    print("-" * len(hdr))
    for m, row in res["methods"].items():
        m3 = row["mean_3_confirmed"]
        m4 = row["mean_all_4"]
        pl = f"{m4['pore_leakage_rel']:.4f}" if "pore_leakage_rel" in m4 else "   N/A"
        line = (
            f"{m:<28} | {m3['mean_bandlimited_rmse']:.4f} / "
            f"{m4['mean_bandlimited_rmse']:.4f} | "
            f"{m3['reaction_front_pos_err_um']:6.2f} / "
            f"{m4['reaction_front_pos_err_um']:6.2f}   | "
            f"{m3['in_pellet_na_zn_corr_abs_err']:.4f} / "
            f"{m4['in_pellet_na_zn_corr_abs_err']:.4f} | {pl}"
        )
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
