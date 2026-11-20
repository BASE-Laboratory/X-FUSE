"""Multi-length synchrotron XCT + XRD-CT coupling, artifact-resilience, chemical
VoDaSuRe synthetic-vs-real gap, and sparse-slice 3D active steering benchmark on
Diamond Light Source K11 DIAD Zn-13X zeolite data (Zenodo 13329639).

Keeps benchmark orchestration strictly decoupled from the `x_fuse` library while
testing:
1. Real dual-beamsize `50 um -> 25 um` XRD-CT hardware audit (`+47.5 deg`
   `gts_cs_theta` motor-zero fix + multi-reflection Bragg peak aggregation),
2. Synchrotron XCT artifact cleanup (polar de-ringing + 7-term radial cupping
   detrend + Fourier Tie-homogeneity fringe-guarded `pore_mask`) & `OpenImpala`
   AMReX geometry export,
3. Multi-length XCT magnification ladder (`Bicubic` vs `2.6 um` vs `1.625 um` vs
   `HR / OpenLSR-X SR`) + Chemical `VoDaSuRe` synthetic-vs-real domain gap,
4. 1D Voronoi sparse-slice 3D active steering across the 20-slice `50 um` stack.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, laplace, rotate, shift, zoom

from x_fuse import (
    DIAD_ZN13X_25UM_SCAN_IDS,
    DIAD_ZN13X_50UM_SCAN_IDS,
    PeakAggregationMode,
    VolumetricUncertaintyReport,
    XFuse,
    detrend_radial_cupping,
    export_for_openimpala,
    fetch_open_dataset,
    get_sample_dataset_path,
    load_diad_xrdct,
    load_microscopy_file,
    make_experimental_holdout,
    remove_polar_rings_conservative,
    segment_pores_otsu,
    suggest_next_slices,
)
from x_fuse.segment import jbu_guided_baseline

try:
    from benchmarks.baselines import (
        cnmf_pansharpen_baseline,
        guided_variational_sr_baseline,
    )
except ImportError:
    from baselines import (  # type: ignore[no-redef]
        cnmf_pansharpen_baseline,
        guided_variational_sr_baseline,
    )

PAIRED_ZN13X_SCANS: tuple[tuple[int, int, int], ...] = (
    (43345, 43324, 9),
    (43395, 43322, 14),
    (43398, 43323, 17),
    (43399, 43325, 18),
)


def _pearson_r(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    x = a[mask].ravel() if mask is not None else a.ravel()
    y = b[mask].ravel() if mask is not None else b.ravel()
    if x.size < 4 or float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _zn_fraction(na: np.ndarray, zn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    na_p = np.maximum(na, 0.0)
    zn_p = np.maximum(zn, 0.0)
    tot = na_p + zn_p
    frac = np.where(tot > 1e-6, zn_p / np.maximum(tot, 1e-6), 0.0)
    return frac.astype(np.float32), tot.astype(np.float32)


def _align_hr_xct_to_25um_support(
    hr_raw: np.ndarray, support_40: np.ndarray, out_size: int = 320
) -> np.ndarray:
    """Co-register bundled DIAD XCT slice to the 40x40 25um pellet support
    using ONLY total crystalline support (Na + Zn) — zero chemical ratio leakage.
    """
    hr_s = zoom(hr_raw, float(out_size) / float(hr_raw.shape[0]), order=1).astype(
        np.float32
    )
    factor = out_size // 40
    best_r = -2.0
    best_img = hr_s
    for fl in (False, True):
        base = np.fliplr(hr_s) if fl else hr_s
        for ang in np.linspace(34.0, 52.0, 19):
            rot = rotate(base, float(ang), reshape=False, order=1, mode="nearest")
            pooled = rot.reshape(40, factor, 40, factor).mean(axis=(1, 3))
            r = _pearson_r(pooled, support_40)
            if r > best_r:
                best_r = r
                best_img = rot
    # Sub-coarse shift refinement (+/- 2 coarse pixels)
    best_shifted = best_img
    for dy in (-factor, -factor // 2, 0, factor // 2, factor):
        for dx in (-factor, -factor // 2, 0, factor // 2, factor):
            cand = shift(best_img, (dy, dx), order=1, mode="nearest")
            pooled = cand.reshape(40, factor, 40, factor).mean(axis=(1, 3))
            r = _pearson_r(pooled, support_40)
            if r > best_r:
                best_r = r
                best_shifted = cand
    return best_shifted.astype(np.float32)


def run_study1_xrdct_hardware_audit(
    nxs_dir: Path, cleaned_xrdct_dir: Path | None = None
) -> dict[str, Any]:
    """Quantify +47.5 deg gts_cs_theta fix and multi-reflection peak aggregation
    across all 4 real paired 50um <-> 25um K11 DIAD scans.
    """
    configs: list[tuple[str, PeakAggregationMode, float]] = [
        ("legacy_single_peak_uncorrected", "nearest_q", 0.0),
        ("single_peak_motor_47p5_fixed", "nearest_q", 47.5),
        ("all_sum_uncorrected", "all_sum", 0.0),
        ("all_sum_motor_47p5_fixed", "all_sum", 47.5),
        ("positive_sum_motor_47p5_fixed", "positive_sum", 47.5),
    ]
    out: dict[str, Any] = {}
    for label, agg, deg in configs:
        d50 = load_diad_xrdct(
            nxs_dir, resolution="50um", aggregation=agg, correct_motor_offset_deg=deg
        )
        d25 = load_diad_xrdct(
            nxs_dir, resolution="25um", aggregation=agg, correct_motor_offset_deg=0.0
        )
        per_pair: dict[str, dict[str, float]] = {}
        r_int_list, r_frac_list, mae_list = [], [], []
        for s50, s25, z_idx in PAIRED_ZN13X_SCANS:
            i50 = DIAD_ZN13X_50UM_SCAN_IDS.index(s50)
            i25 = DIAD_ZN13X_25UM_SCAN_IDS.index(s25)
            na50_up = zoom(d50["Na"][i50], 2.0, order=3)
            zn50_up = zoom(d50["Zn"][i50], 2.0, order=3)
            f50, _ = _zn_fraction(na50_up, zn50_up)
            f25, tot25 = _zn_fraction(d25["Na"][i25], d25["Zn"][i25])
            pellet = tot25 > np.percentile(tot25, 45.0)
            r_int = _pearson_r(zn50_up, d25["Zn"][i25], mask=pellet)
            r_f = _pearson_r(f50, f25, mask=pellet)
            mae_f = float(np.mean(np.abs(f50[pellet] - f25[pellet])))
            per_pair[f"{s50}_to_{s25}_z{z_idx}"] = {
                "zn_intensity_r": round(r_int, 4),
                "zn_fraction_r": round(r_f, 4),
                "zn_fraction_mae": round(mae_f, 4),
            }
            r_int_list.append(r_int)
            r_frac_list.append(r_f)
            mae_list.append(mae_f)
        out[label] = {
            "mean_zn_intensity_r": round(float(np.mean(r_int_list)), 4),
            "mean_zn_fraction_r": round(float(np.mean(r_frac_list)), 4),
            "mean_zn_fraction_mae": round(float(np.mean(mae_list)), 4),
            "pairs": per_pair,
        }

    if cleaned_xrdct_dir is not None and cleaned_xrdct_dir.exists():
        cand_files = sorted(cleaned_xrdct_dir.glob("*.npz")) + sorted(
            cleaned_xrdct_dir.glob("*.h5")
        )
        if cand_files:
            out["cleaned_bundle_files"] = [p.name for p in cand_files]
    return out


_MODEL_CACHE: dict[tuple[str, bool, str], XFuse] = {}


def _make_xfuse_2d(
    backbone: str,
    pretrained: bool,
    pore_mask: Any = None,
    conformal_alpha: float | None = None,
    device: str = "cpu",
) -> XFuse:
    from x_fuse.uncertainty import ConformalCalibrator

    key = (backbone, bool(pretrained), str(device))
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = XFuse(
            backbone=backbone,
            pretrained=pretrained,
            stride=7 if "dinov2" in backbone else 8,
            fusion_method="spatial_mlp",
            interpolation_mode="cross_attn",
            loss_fn="compositional_kl",
            psf_profile="moffat",
            learn_psf=True,
            guide_residual=True,
            tv_weight=0.01,
            lr=1e-2,
            device=device,
        )
        _MODEL_CACHE[key] = model
    model.pore_mask_spec = pore_mask
    model._active_pore_mask_hr = None
    model.conformal_alpha = conformal_alpha
    model.calibrator = (
        ConformalCalibrator(
            alpha=conformal_alpha,
            method="raps",
            fit_temperature=True,
            spatial_block_size=2,
        )
        if conformal_alpha is not None
        else None
    )
    model.clear_feature_cache()
    model.reset_trainable_weights()
    return model


def _eval_probs_against_25um(
    probs_hr: np.ndarray,
    na25: np.ndarray,
    zn25: np.ndarray,
    pore_ref_hr: np.ndarray,
) -> dict[str, float]:
    h_hr, _ = probs_hr.shape[-2], probs_hr.shape[-1]
    fac = h_hr // 40
    if probs_hr.shape[0] == 3:
        na_hr, zn_hr = probs_hr[1], probs_hr[2]
        void_leak = float(probs_hr[1:, pore_ref_hr].max()) if pore_ref_hr.any() else 0.0
    else:
        na_hr, zn_hr = probs_hr[0], probs_hr[1]
        void_leak = float(probs_hr[:, pore_ref_hr].max()) if pore_ref_hr.any() else 0.0

    na_40 = na_hr.reshape(40, fac, 40, fac).mean(axis=(1, 3))
    zn_40 = zn_hr.reshape(40, fac, 40, fac).mean(axis=(1, 3))
    na_gt = np.maximum(na25, 0.0)
    zn_gt = np.maximum(zn25, 0.0)
    phase_mae = 0.5 * float(
        np.mean(np.abs(na_40 - na_gt)) + np.mean(np.abs(zn_40 - zn_gt))
    )
    f_pred, _ = _zn_fraction(na_40, zn_40)
    f_gt, tot25 = _zn_fraction(na25, zn25)
    solid_cells = tot25 > max(float(np.percentile(tot25, 25.0)), 0.05)
    return {
        "zn_intensity_r": round(_pearson_r(zn_40, zn_gt), 4),
        "phase_abundance_mae": round(phase_mae, 4),
        "zn_fraction_r": round(_pearson_r(f_pred, f_gt, mask=solid_cells), 4),
        "zn_fraction_mae": round(
            float(np.mean(np.abs(f_pred[solid_cells] - f_gt[solid_cells]))), 4
        ),
        "max_void_chemical_leakage": round(void_leak, 6),
    }


def run_study2_artifact_cleanup_and_openimpala(
    hr_clean_160: np.ndarray,
    guide50_z9: np.ndarray,
    na25_z9: np.ndarray,
    zn25_z9: np.ndarray,
    output_dir: Path,
    backbone: str,
    pretrained: bool,
    device: str,
) -> dict[str, Any]:
    from x_fuse.uncertainty import ConformalCalibrator

    h_sz, w_sz = hr_clean_160.shape
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, h_sz, dtype=np.float32),
        np.linspace(-1.0, 1.0, w_sz, dtype=np.float32),
        indexing="ij",
    )
    rr = np.hypot(yy, xx)
    pore_ref = segment_pores_otsu(hr_clean_160, tie_homogeneity_alpha_vox2=0.6)

    corrupted = (
        hr_clean_160
        + 0.28 * (rr**4)
        + 0.09 * np.sin(rr * 36.0).astype(np.float32)
        - 0.22 * laplace(hr_clean_160).astype(np.float32)
    )
    cleaned = remove_polar_rings_conservative(
        detrend_radial_cupping(corrupted), radial_sigma=2.0
    )

    m_unguarded = segment_pores_otsu(corrupted, tie_homogeneity_alpha_vox2=0.0)
    m_cleaned_guarded = segment_pores_otsu(cleaned, tie_homogeneity_alpha_vox2=0.6)
    iou_unguarded = float((m_unguarded & pore_ref).sum()) / float(
        max(1, (m_unguarded | pore_ref).sum())
    )
    iou_guarded = float((m_cleaned_guarded & pore_ref).sum()) / float(
        max(1, (m_cleaned_guarded | pore_ref).sum())
    )

    torch.manual_seed(42)
    rep_unclean = _make_xfuse_2d(
        backbone, pretrained, pore_mask=None, device=device
    ).fit_predict(corrupted, guide50_z9, epochs=16, mode="multiclass", tta_shifts=())

    torch.manual_seed(42)
    rep_unguarded_pm = _make_xfuse_2d(
        backbone, pretrained, pore_mask=m_unguarded, device=device
    ).fit_predict(corrupted, guide50_z9, epochs=16, mode="multiclass", tta_shifts=())

    torch.manual_seed(42)
    mod_clean = _make_xfuse_2d(
        backbone,
        pretrained,
        pore_mask="auto_otsu_fringe_guarded",
        conformal_alpha=None,
        device=device,
    )
    rep_clean = mod_clean.fit_predict(
        cleaned, guide50_z9, epochs=16, mode="multiclass", tta_shifts=()
    )
    cal = ConformalCalibrator(alpha=0.10, method="raps", fit_temperature=True)
    g_t = torch.from_numpy(guide50_z9[None]).float()
    cal_mask = ConformalCalibrator.make_spatial_holdout_mask((20, 20), 0.25)
    cal.calibrate_coarse(
        probs_hr=rep_clean.probabilities,
        guide_lr=g_t,
        epistemic_var_hr=rep_clean.epistemic_variance,
        cal_mask_lr=cal_mask,
        downsampler=mod_clean._active_downsampler(),
        mode="multiclass",
    )
    ps_2d, _ = cal.predict_sets_and_intervals(
        probs_hr=rep_clean.probabilities,
        guide_lr=g_t,
        epistemic_var_hr=rep_clean.epistemic_variance,
        mode="multiclass",
    )

    probs_3d = rep_clean.probabilities[0].unsqueeze(1).repeat(1, 8, 1, 1)
    hard_3d = rep_clean.hard_mask[0].unsqueeze(0).repeat(8, 1, 1)
    ent_3d = rep_clean.aleatoric_entropy[0].unsqueeze(0).repeat(8, 1, 1)
    epi_3d = rep_clean.epistemic_variance[0].unsqueeze(0).repeat(8, 1, 1)
    res_3d = rep_clean.physical_residual[0].unsqueeze(0).repeat(8, 1, 1)
    comb_3d = rep_clean.combined_uncertainty[0].unsqueeze(0).repeat(8, 1, 1)
    ps_3d = ps_2d[0].unsqueeze(1).repeat(1, 8, 1, 1)
    vrep = VolumetricUncertaintyReport(
        probabilities=probs_3d,
        hard_mask=hard_3d,
        aleatoric_entropy=ent_3d,
        triplane_variance=epi_3d,
        physical_residual=res_3d,
        combined_uncertainty=comb_3d,
        z_slice_consistency=1.0,
        prediction_sets=ps_3d,
    )
    imp_dir = output_dir / "openimpala_zn13x_z9"
    tif_p, inp_p = export_for_openimpala(
        vrep, imp_dir, name="zn13x_z9_multiphase", active_phase=0, direction="x"
    )

    clean_probs_np = rep_clean.probabilities[0].cpu().numpy()
    clean_metrics = _eval_probs_against_25um(clean_probs_np, na25_z9, zn25_z9, pore_ref)
    clean_metrics["detected_pore_chemical_leakage"] = round(
        float(clean_probs_np[1:, m_cleaned_guarded].max())
        if m_cleaned_guarded.any()
        else 0.0,
        6,
    )

    return {
        "pore_mask_iou_unguarded_corrupted": round(iou_unguarded, 4),
        "pore_mask_iou_cleaned_fringe_guarded": round(iou_guarded, 4),
        "uncleaned_no_pore_mask": _eval_probs_against_25um(
            rep_unclean.probabilities[0].cpu().numpy(), na25_z9, zn25_z9, pore_ref
        ),
        "uncleaned_unguarded_otsu": _eval_probs_against_25um(
            rep_unguarded_pm.probabilities[0].cpu().numpy(), na25_z9, zn25_z9, pore_ref
        ),
        "cleaned_fringe_guarded_xfuse": clean_metrics,
        "openimpala_export": {
            "tiff": str(tif_p),
            "inputs": str(inp_p),
            "has_conformal_envelopes": (
                imp_dir / "zn13x_z9_multiphase_phase0_lower.tif"
            ).exists(),
        },
    }


def run_study3_multilength_ladder_and_vodasure(
    hr_clean_160: np.ndarray,
    d50: dict[str, np.ndarray],
    d25: dict[str, np.ndarray],
    backbone: str,
    pretrained: bool,
    device: str,
    openlsrx_sr_tif: Path | None = None,
    run_ladder: bool = True,
) -> dict[str, Any]:
    pore_ref = segment_pores_otsu(hr_clean_160, tie_homogeneity_alpha_vox2=0.6)
    g50_z9 = np.maximum(np.stack([d50["Na"][9], d50["Zn"][9]], axis=0), 0.0).astype(
        np.float32
    )
    na25_z9, zn25_z9 = d25["Na"][2], d25["Zn"][2]

    na_bic_hr = np.maximum(zoom(g50_z9[0], 8.0, order=3), 0.0).astype(np.float32)
    zn_bic_hr = np.maximum(zoom(g50_z9[1], 8.0, order=3), 0.0).astype(np.float32)
    ladder: dict[str, Any] = {
        "bicubic_50um_xrd_only": _eval_probs_against_25um(
            np.stack([na_bic_hr, zn_bic_hr], axis=0), na25_z9, zn25_z9, pore_ref
        )
    }

    if run_ladder:
        xct_scales: list[tuple[str, np.ndarray]] = [
            ("xfuse_xct_2p6um_equiv", gaussian_filter(hr_clean_160, sigma=2.2)),
            ("xfuse_xct_1p625um_equiv", gaussian_filter(hr_clean_160, sigma=1.1)),
            ("xfuse_xct_hr_native", hr_clean_160),
        ]
        if openlsrx_sr_tif is not None and openlsrx_sr_tif.exists():
            sr_arr, _ = load_microscopy_file(openlsrx_sr_tif)
            if sr_arr.ndim == 3:
                sr_arr = sr_arr[sr_arr.shape[0] // 2]
            sr_160 = zoom(
                sr_arr,
                (160.0 / sr_arr.shape[0], 160.0 / sr_arr.shape[1]),
                order=1,
            ).astype(np.float32)
            xct_scales.append(("xfuse_openlsrx_sr_xct", sr_160))

        for name_s, xct_img in xct_scales:
            torch.manual_seed(42)
            rep_s = _make_xfuse_2d(
                backbone,
                pretrained,
                pore_mask="auto_otsu_fringe_guarded",
                device=device,
            ).fit_predict(xct_img, g50_z9, epochs=16, mode="multiclass", tta_shifts=())
            ladder[name_s] = _eval_probs_against_25um(
                rep_s.probabilities[0].cpu().numpy(), na25_z9, zn25_z9, pore_ref
            )

    vodasure_pairs: dict[str, Any] = {}
    syn_rs, real_rs = [], []
    for s50, s25, z_idx in PAIRED_ZN13X_SCANS:
        i50 = DIAD_ZN13X_50UM_SCAN_IDS.index(s50)
        i25 = DIAD_ZN13X_25UM_SCAN_IDS.index(s25)
        g25_gt = np.maximum(
            np.stack([d25["Na"][i25], d25["Zn"][i25]], axis=0), 0.0
        ).astype(np.float32)
        g50_real = np.maximum(
            np.stack([d50["Na"][i50], d50["Zn"][i50]], axis=0), 0.0
        ).astype(np.float32)
        _, g50_syn, _ = make_experimental_holdout(
            np.ones((40, 40), dtype=np.float32),
            g25_gt,
            factor=2,
            beam_profile="moffat",
        )
        f_gt, tot25 = _zn_fraction(g25_gt[0], g25_gt[1])
        pel = tot25 > np.percentile(tot25, 45.0)
        f_syn, _ = _zn_fraction(
            zoom(g50_syn[0], 2.0, order=3), zoom(g50_syn[1], 2.0, order=3)
        )
        f_real, _ = _zn_fraction(
            zoom(g50_real[0], 2.0, order=3), zoom(g50_real[1], 2.0, order=3)
        )
        r_syn = _pearson_r(f_syn, f_gt, mask=pel)
        r_real = _pearson_r(f_real, f_gt, mask=pel)
        syn_rs.append(r_syn)
        real_rs.append(r_real)
        vodasure_pairs[f"{s50}_to_{s25}_z{z_idx}"] = {
            "synthetic_2x_moffat_r": round(r_syn, 4),
            "real_50um_hardware_r": round(r_real, 4),
            "domain_gap_delta_r": round(r_syn - r_real, 4),
        }

    return {
        "xct_magnification_ladder_z9": ladder,
        "chemical_vodasure_domain_gap": {
            "mean_synthetic_2x_r": round(float(np.mean(syn_rs)), 4),
            "mean_real_50um_r": round(float(np.mean(real_rs)), 4),
            "mean_domain_gap_delta_r": round(
                float(np.mean(syn_rs) - np.mean(real_rs)), 4
            ),
            "pairs": vodasure_pairs,
        },
    }


def run_study4_sparse_3d_active_steering(d50: dict[str, np.ndarray]) -> dict[str, Any]:
    na_vol = np.maximum(d50["Na"], 0.0)
    zn_vol = np.maximum(d50["Zn"], 0.0)
    z_dim, h_sz, w_sz = zn_vol.shape
    f_zn_3d, tot_3d = _zn_fraction(na_vol, zn_vol)
    pellet_3d = tot_3d > np.percentile(tot_3d, 45.0)

    init_anchors = [0, z_dim - 1]
    grad_z = np.abs(np.gradient(f_zn_3d, axis=0)).astype(np.float32)
    ent_3d = torch.from_numpy(
        np.clip(grad_z / max(float(np.percentile(grad_z, 98.0)), 1e-6), 0.0, 1.0)
    )
    epi_3d = ent_3d * 0.8 + 0.05
    probs_dummy = torch.full((3, z_dim, h_sz, w_sz), 1.0 / 3.0)
    rep_init = VolumetricUncertaintyReport(
        probabilities=probs_dummy,
        hard_mask=torch.zeros((z_dim, h_sz, w_sz), dtype=torch.int64),
        aleatoric_entropy=ent_3d,
        triplane_variance=epi_3d,
        physical_residual=torch.zeros_like(ent_3d),
        combined_uncertainty=0.5 * (ent_3d + epi_3d),
        z_slice_consistency=0.92,
    )

    def _interp_stack_mae(known_z: list[int]) -> dict[str, float]:
        kz = sorted(set(known_z))
        recon = np.zeros_like(f_zn_3d)
        for z_i in range(z_dim):
            if z_i <= kz[0]:
                recon[z_i] = f_zn_3d[kz[0]]
            elif z_i >= kz[-1]:
                recon[z_i] = f_zn_3d[kz[-1]]
            else:
                right_pos = int(np.searchsorted(kz, z_i))
                z0, z1 = kz[right_pos - 1], kz[right_pos]
                w = float(z_i - z0) / float(max(1, z1 - z0))
                recon[z_i] = (1.0 - w) * f_zn_3d[z0] + w * f_zn_3d[z1]
        mae_all = float(np.mean(np.abs(recon[pellet_3d] - f_zn_3d[pellet_3d])))
        val_zs = [9, 14, 17, 18]
        val_err = [
            float(
                np.mean(
                    np.abs(recon[z_v][pellet_3d[z_v]] - f_zn_3d[z_v][pellet_3d[z_v]])
                )
            )
            for z_v in val_zs
        ]
        return {
            "volume_zn_mae": round(mae_all, 4),
            "paired_val_slices_mae": round(float(np.mean(val_err)), 4),
        }

    budgets: dict[str, Any] = {}
    for total_k in (3, 4, 6):
        add_k = total_k - len(init_anchors)
        plan = suggest_next_slices(
            rep_init, acquired_indices=init_anchors, k=add_k, min_spacing=2
        )
        act_slices = sorted(init_anchors + plan.recommended_indices)
        eq_slices = [int(round(x)) for x in np.linspace(0, z_dim - 1, num=total_k)]
        budgets[f"k_{total_k}"] = {
            "voronoi_active_slices": act_slices,
            "equispaced_slices": eq_slices,
            "voronoi_active": _interp_stack_mae(act_slices),
            "equispaced": _interp_stack_mae(eq_slices),
        }
    return budgets


def _lowpass_2d_raised_cosine(
    img_2d: np.ndarray,
    cutoff_cyc_per_px: float = 0.43,
    rolloff_cyc_per_px: float = 0.05,
) -> np.ndarray:
    """Band-limit a 2D map to the 25 um XRD-CT FRC cutoff (~58 um -> 0.43 cyc/px)."""
    arr = np.asarray(img_2d, dtype=np.float64)
    ny, nx = arr.shape
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.fftfreq(nx)[None, :]
    kr = np.hypot(fy, fx)
    k0 = max(0.05, float(cutoff_cyc_per_px) - 0.5 * float(rolloff_cyc_per_px))
    k1 = float(cutoff_cyc_per_px) + 0.5 * float(rolloff_cyc_per_px)
    filt = np.ones_like(kr, dtype=np.float64)
    trans = (kr > k0) & (kr < k1)
    filt[trans] = 0.5 * (1.0 + np.cos(np.pi * (kr[trans] - k0) / max(k1 - k0, 1e-6)))
    filt[kr >= k1] = 0.0
    return np.real(np.fft.ifft2(np.fft.fft2(arr) * filt)).astype(np.float64)


def _score_mmbench_t3_pair(
    hz: np.ndarray,
    hn: np.ndarray,
    ref_zn25: np.ndarray,
    ref_na25: np.ndarray,
    sigma2_25: float,
    pore_hr: np.ndarray,
    cutoff_cyc_per_px: float = 0.43,
) -> dict[str, float]:
    pz = hz.reshape(40, 4, 40, 4).mean(axis=(1, 3)).astype(np.float64)
    pn = hn.reshape(40, 4, 40, 4).mean(axis=(1, 3)).astype(np.float64)
    rz = np.asarray(ref_zn25, dtype=np.float64)
    rn = np.asarray(ref_na25, dtype=np.float64)

    pz_bl = _lowpass_2d_raised_cosine(pz, cutoff_cyc_per_px)
    pn_bl = _lowpass_2d_raised_cosine(pn, cutoff_cyc_per_px)
    rz_bl = _lowpass_2d_raised_cosine(rz, cutoff_cyc_per_px)
    rn_bl = _lowpass_2d_raised_cosine(rn, cutoff_cyc_per_px)

    bw_frac = min(1.0, float(np.pi * (cutoff_cyc_per_px**2)))
    s2_bl = max(float(sigma2_25) * max(bw_frac, 0.1), 1e-8)

    bl_rmse = 0.5 * (
        float(np.sqrt(np.mean((pz_bl - rz_bl) ** 2)))
        + float(np.sqrt(np.mean((pn_bl - rn_bl) ** 2)))
    )
    bl_r = 0.5 * (_pearson_r(pz_bl, rz_bl) + _pearson_r(pn_bl, rn_bl))
    bl_chi2 = 0.5 * (
        float(np.mean(((pz_bl - rz_bl) ** 2) / s2_bl))
        + float(np.mean(((pn_bl - rn_bl) ** 2) / s2_bl))
    )
    solid_scale = float(np.percentile(np.abs(hz) + np.abs(hn), 90.0)) + 1e-8
    pore_leak = (
        float(np.mean(np.abs(hz[pore_hr]) + np.abs(hn[pore_hr])) / solid_scale)
        if pore_hr.any()
        else 0.0
    )
    return {
        "mean_bandlimited_reduced_chi2": round(bl_chi2, 4),
        "mean_bandlimited_rmse": round(bl_rmse, 4),
        "mean_bandlimited_pearson_r": round(bl_r, 4),
        "t3b_pore_leakage_rel": round(pore_leak, 6),
    }


def run_study5_mmbench_t3_and_two_track_audit(
    hr_base_160: np.ndarray,
    backbone: str,
    pretrained: bool,
    device: str,
) -> dict[str, Any]:
    """Evaluate MultimodalBenchmark (feat/benchmark-v2) band-limited Task T3
    scoring (k <= 0.43 cyc/px ~ 58 um) and the two-track preprocessing /
    radial cupping chemistry-confounding policy.
    """
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, 160, dtype=np.float32),
        np.linspace(-1.0, 1.0, 160, dtype=np.float32),
        indexing="ij",
    )
    rr = np.hypot(yy, xx)
    pore_ref = segment_pores_otsu(hr_base_160, tie_homogeneity_alpha_vox2=0.6)
    solid_f = (~pore_ref).astype(np.float32)
    chem_shell = np.clip(
        0.20 + 0.70 * np.clip(rr / 1.15, 0.0, 1.0) ** 2, 0.05, 0.95
    ).astype(np.float32)

    # Track 1 (raw_float32): porous XCT + radial Zn Z-contrast + rings + fringe
    hr_raw_track = (
        hr_base_160
        + 0.18 * solid_f * chem_shell
        + 0.06 * np.sin(rr * 42.0).astype(np.float32)
        - 0.15 * laplace(hr_base_160).astype(np.float32)
    ).astype(np.float32)
    # Track 2 (cleaned_dering_v1): conservative polar de-ringing ONLY
    hr_dering_track = remove_polar_rings_conservative(
        hr_raw_track, center=(80.0, 80.0), radial_sigma=3.5
    )
    hr_cupped = detrend_radial_cupping(hr_raw_track)

    delta_ring = hr_raw_track - hr_dering_track
    delta_cup = hr_raw_track - hr_cupped
    r_ring_zn = _pearson_r(delta_ring, chem_shell, mask=~pore_ref)
    r_cup_zn = _pearson_r(delta_cup, chem_shell, mask=~pore_ref)

    zn_hr_gt = solid_f * chem_shell * 20.0
    na_hr_gt = solid_f * (1.0 - chem_shell) * 20.0
    _, g50, _ = make_experimental_holdout(
        hr_dering_track,
        np.stack([na_hr_gt, zn_hr_gt], axis=0),
        factor=8,
        beam_profile="moffat",
    )
    na25_gt = na_hr_gt.reshape(40, 4, 40, 4).mean(axis=(1, 3))
    zn25_gt = zn_hr_gt.reshape(40, 4, 40, 4).mean(axis=(1, 3))
    pm_det = segment_pores_otsu(hr_dering_track, tie_homogeneity_alpha_vox2=0.6)
    solid_det = (~pm_det).astype(np.float32)

    def _rescale_flux(
        p_na: np.ndarray, p_zn: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        sn = float(g50[0].mean()) / max(float(p_na.mean()), 1e-6)
        sz = float(g50[1].mean()) / max(float(p_zn.mean()), 1e-6)
        return p_zn * sz, p_na * sn

    bic_na = np.maximum(zoom(g50[0], 8.0, order=3), 0.0)
    bic_zn = np.maximum(zoom(g50[1], 8.0, order=3), 0.0)

    jbu_p = (
        jbu_guided_baseline(hr_dering_track, g50, mode="multiclass")
        .probabilities[0]
        .cpu()
        .numpy()
    )
    gtv_p = (
        guided_variational_sr_baseline(hr_dering_track, g50, mode="multiclass")
        .probabilities[0]
        .cpu()
        .numpy()
    )
    cnmf_p = (
        cnmf_pansharpen_baseline(hr_dering_track, g50, mode="multiclass")
        .probabilities[0]
        .cpu()
        .numpy()
    )

    torch.manual_seed(42)
    xf_p = (
        _make_xfuse_2d(
            backbone,
            pretrained,
            pore_mask="auto_otsu_fringe_guarded",
            device=device,
        )
        .fit_predict(hr_dering_track, g50, epochs=20, mode="multiclass", tta_shifts=())
        .probabilities[0]
        .cpu()
        .numpy()
    )

    candidates = {
        "Bicubic": (bic_zn, bic_na),
        "JBU": _rescale_flux(jbu_p[1] * solid_det, jbu_p[2] * solid_det),
        "Guided-TV": _rescale_flux(gtv_p[1] * solid_det, gtv_p[2] * solid_det),
        "CNMF": _rescale_flux(cnmf_p[1] * solid_det, cnmf_p[2] * solid_det),
        "X-FUSE (v1.0.0)": _rescale_flux(xf_p[1], xf_p[2]),
    }
    t3_scores = {
        name: _score_mmbench_t3_pair(hz, hn, zn25_gt, na25_gt, 0.25, pore_ref)
        for name, (hz, hn) in candidates.items()
    }
    return {
        "frc_cutoff_cyc_per_px": 0.43,
        "frc_effective_resolution_um": 58.1,
        "cupping_chemistry_confounding": {
            "r_ring_removal_vs_zn": round(r_ring_zn, 4),
            "r_cupping_detrend_vs_zn": round(r_cup_zn, 4),
            "dering_chemistry_safe": abs(r_ring_zn) <= 0.05,
        },
        "t3_bandlimited_leaderboard": t3_scores,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run multi-length Zn-13X XCT + XRD-CT coupling & artifact study."
    )
    parser.add_argument(
        "--zenodo-cache",
        type=Path,
        default=Path.home() / ".cache-xfuse" / "zenodo",
    )
    parser.add_argument("--cleaned-xrdct-dir", type=Path, default=None)
    parser.add_argument("--cleaned-xct-path", type=Path, default=None)
    parser.add_argument("--openlsrx-sr-tif", type=Path, default=None)
    parser.add_argument("--backbone", default="dinov2_vits14")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/outputs/zn13x_multiscale"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = fetch_open_dataset("diad_zn13x_xrdct", dest_root=args.zenodo_cache)
    nxs_dir = paths[0].parent

    d50 = load_diad_xrdct(nxs_dir, resolution="50um", aggregation="positive_sum")
    d25 = load_diad_xrdct(nxs_dir, resolution="25um", aggregation="positive_sum")

    if args.cleaned_xct_path is not None and args.cleaned_xct_path.exists():
        hr_raw, _ = load_microscopy_file(args.cleaned_xct_path)
        if hr_raw.ndim == 3:
            hr_raw = hr_raw[hr_raw.shape[0] // 2]
        _, tot25_z9 = _zn_fraction(d25["Na"][2], d25["Zn"][2])
        hr_aligned_160 = _align_hr_xct_to_25um_support(hr_raw, tot25_z9, out_size=160)
        na25_ref, zn25_ref = d25["Na"][2], d25["Zn"][2]
        g50_z9 = np.maximum(np.stack([d50["Na"][9], d50["Zn"][9]], axis=0), 0.0).astype(
            np.float32
        )
        xct_pairing_mode = "external_coregistered_169066_xct"
    else:
        # Bundled high_9.tiff (691 um sub-ROI) is co-registered to bundled low_9.tiff
        zn_sample_dir = get_sample_dataset_path("Zn13X")
        hr_sub = np.array(Image.open(zn_sample_dir / "high_9.tiff"), dtype=np.float32)[
            :640, :640
        ]
        lr_sub = np.array(Image.open(zn_sample_dir / "low_9.tiff"), dtype=np.float32)[
            :320, :320
        ]
        hr_aligned_160 = zoom(hr_sub, 160.0 / 640.0, order=1).astype(np.float32)
        lr_160 = zoom(lr_sub, 160.0 / 320.0, order=1).astype(np.float32)
        lr_norm = np.clip(
            (lr_160 - np.percentile(lr_160, 5.0))
            / max(
                float(np.percentile(lr_160, 98.0) - np.percentile(lr_160, 5.0)), 1e-6
            ),
            0.0,
            1.0,
        )
        pore_hr = segment_pores_otsu(hr_aligned_160, tie_homogeneity_alpha_vox2=0.6)
        solid_hr = (~pore_hr).astype(np.float32)
        zn_hr_gt = solid_hr * lr_norm
        na_hr_gt = solid_hr * (1.0 - lr_norm)
        _, g50_z9, _ = make_experimental_holdout(
            hr_aligned_160,
            np.stack([na_hr_gt, zn_hr_gt], axis=0),
            factor=8,
            beam_profile="moffat",
        )
        na25_ref = na_hr_gt.reshape(40, 4, 40, 4).mean(axis=(1, 3)).astype(np.float32)
        zn25_ref = zn_hr_gt.reshape(40, 4, 40, 4).mean(axis=(1, 3)).astype(np.float32)
        xct_pairing_mode = "bundled_zn13x_coregistered_porous_holdout_8x"

    pretrained = not bool(args.no_pretrained)
    s1 = run_study1_xrdct_hardware_audit(nxs_dir, args.cleaned_xrdct_dir)
    s2 = run_study2_artifact_cleanup_and_openimpala(
        hr_aligned_160,
        g50_z9,
        na25_ref,
        zn25_ref,
        output_dir=args.output_dir,
        backbone=args.backbone,
        pretrained=pretrained,
        device=args.device,
    )
    d50_for_ladder = dict(d50)
    d25_for_ladder = dict(d25)
    if xct_pairing_mode != "external_coregistered_169066_xct":
        d50_na = d50["Na"].copy()
        d50_zn = d50["Zn"].copy()
        d50_na[9], d50_zn[9] = g50_z9[0], g50_z9[1]
        d25_na = d25["Na"].copy()
        d25_zn = d25["Zn"].copy()
        d25_na[2], d25_zn[2] = na25_ref, zn25_ref
        d50_for_ladder = {"Na": d50_na, "Zn": d50_zn}
        d25_for_ladder = {"Na": d25_na, "Zn": d25_zn}

    s3_ladder = run_study3_multilength_ladder_and_vodasure(
        hr_aligned_160,
        d50=d50_for_ladder,
        d25=d25_for_ladder,
        backbone=args.backbone,
        pretrained=pretrained,
        device=args.device,
        openlsrx_sr_tif=args.openlsrx_sr_tif,
    )["xct_magnification_ladder_z9"]
    s3_vodasure = run_study3_multilength_ladder_and_vodasure(
        hr_aligned_160,
        d50=d50,
        d25=d25,
        backbone=args.backbone,
        pretrained=pretrained,
        device=args.device,
        openlsrx_sr_tif=None,
        run_ladder=False,
    )["chemical_vodasure_domain_gap"]
    s4 = run_study4_sparse_3d_active_steering(d50)
    s5 = run_study5_mmbench_t3_and_two_track_audit(
        hr_aligned_160,
        backbone=args.backbone,
        pretrained=pretrained,
        device=args.device,
    )

    results = {
        "dataset": "Diamond K11 DIAD Zn-13X (Zenodo 13329639 + Bundled DIAD XCT)",
        "xct_pairing_mode": xct_pairing_mode,
        "backbone": args.backbone,
        "pretrained": pretrained,
        "study1_xrdct_hardware_audit": s1,
        "study2_xct_artifact_and_openimpala": s2,
        "study3_multilength_ladder_and_vodasure": {
            "xct_magnification_ladder_z9": s3_ladder,
            "chemical_vodasure_domain_gap": s3_vodasure,
        },
        "study4_sparse_3d_voronoi_steering": s4,
        "study5_mmbench_t3_and_two_track_audit": s5,
    }
    out_json = args.output_dir / "zn13x_multiscale_results.json"
    out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
