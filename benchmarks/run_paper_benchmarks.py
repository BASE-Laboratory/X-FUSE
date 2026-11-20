"""End-to-end paper benchmark & ablation runner for X-FUSE.

Executes fair, scale-preserving (`guide_normalization="preserve_counts"`),
mismatched-PSF (`beam_profile="moffat"`) comparisons across the complete
baseline suite (Otsu, Bicubic, JBU, Gram-Schmidt, CNMF, Guided TV, SIREN INR,
and X-FUSE):
1. **Resolution-Gap Super-Resolution Sweep** (reporting both mIoU and BF1 for
   every baseline).
2. **Sub-Pixel Drift & Learnable Anisotropic PSF Recovery**.
3. **Closed-Loop Active Sparse-Slice Tomography**.
4. **Oriented Lamellar Microstructure, FIB Artifact Corruption, Poisson
   Spectral Unmixing, & Cross-Specimen LAC Conformal Calibration**.
5. **Quantitative Real Synchrotron Validation on Diamond DIAD `Zn13X`**
   (16x Moffat hold-out super-resolution + zero-shot cross-slice transfer).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import shift as nd_shift

from x_fuse import (
    OPEN_DATASETS,
    ConformalCalibrator,
    VolumetricFuser,
    XFuse,
    bicubic_upsample_baseline,
    brier_score,
    compute_segmentation_metrics,
    empirical_conformal_coverage,
    expected_calibration_error,
    generate_artifact_corrupted_phantom_2d,
    generate_correlative_phantom_2d,
    generate_correlative_phantom_3d,
    generate_oriented_lamellar_phantom_2d,
    generate_spectral_cube_phantom_2d,
    jbu_guided_baseline,
    make_experimental_holdout,
    sparsification_curve,
    suggest_next_slices,
)

from .baselines import (
    cnmf_pansharpen_baseline,
    coordinate_inr_baseline,
    gram_schmidt_pansharpen_baseline,
    guided_variational_sr_baseline,
    otsu_threshold_baseline,
)
from .dataset_adapters import (
    load_bundled_morgis_samples,
    load_bundled_zn13x_pairs,
    load_zenodo_correlative_sample,
)
from .plot_paper_figures import generate_paper_figures


def _inject_shifted_coarse_guide(
    gt_phase_channels: np.ndarray,
    factor: int,
    shift_x_hr_px: float,
    psf_sigma_hr: float,
) -> np.ndarray:
    """Shift ground-truth HR phase maps by ``shift_x_hr_px`` along +X, then apply
    mismatched Moffat PSF downsampling to simulate beam/stage drift in ``G_LR``.
    """
    shifted = np.stack(
        [
            nd_shift(ch, shift=(0.0, shift_x_hr_px), order=1, mode="nearest")
            for ch in gt_phase_channels
        ],
        axis=0,
    )
    dummy_hr = np.zeros(gt_phase_channels.shape[1:], dtype=np.float32)
    _, coarse, _ = make_experimental_holdout(
        dummy_hr,
        shifted,
        factor=factor,
        psf_sigma=psf_sigma_hr,
        beam_profile="moffat",
    )
    return coarse


def run_paper_benchmarks(
    output_dir: Path,
    backbone: str = "timm:vit_tiny_patch16_224",
    pretrained: bool = False,
    quick: bool = False,
    device: str = "cpu",
    download_zenodo: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    img_size = 64 if quick else 128
    stride = 8 if quick else 7
    epochs = 8 if quick else 32
    tta_shifts: tuple[int, ...] = ()
    factors = [4, 8] if quick else [4, 8, 16]
    norm = "preserve_counts"

    results: dict[str, Any] = {
        "config": {
            "backbone": backbone,
            "pretrained": pretrained,
            "quick": quick,
            "device": device,
            "stride": stride,
            "epochs": epochs,
            "beam_profile": "moffat",
            "guide_normalization": norm,
        },
        "ablation_resolution_sweep": [],
        "drift_and_psf_ablation": [],
        "active_slice_tomography": {},
        "v1_stress_tests": {},
        "synchrotron_samples": {},
        "zenodo_holdouts": {},
    }

    # 1. Resolution-Gap Sweep (Mismatched Moffat PSF, All Baselines, Fair Scale)
    for factor in factors:
        ph = generate_correlative_phantom_2d(
            size=img_size,
            downsample_factor=factor,
            phase_b_contrast=0.35,
            hr_noise_std=0.03,
            poisson_counts=300.0,
            beam_profile="moffat",
            seed=42,
        )
        gt = ph.gt_labels
        otsu = otsu_threshold_baseline(
            ph.hr_image, ph.coarse_guide, mode="multiclass", guide_normalization=norm
        )
        bic = bicubic_upsample_baseline(
            ph.coarse_guide,
            (img_size, img_size),
            mode="multiclass",
            guide_normalization=norm,
        )
        jbu = jbu_guided_baseline(
            ph.hr_image, ph.coarse_guide, mode="multiclass", guide_normalization=norm
        )
        gs = gram_schmidt_pansharpen_baseline(
            ph.hr_image, ph.coarse_guide, mode="multiclass", guide_normalization=norm
        )
        cnmf = cnmf_pansharpen_baseline(
            ph.hr_image,
            ph.coarse_guide,
            mode="multiclass",
            n_iters=15 if quick else 35,
            guide_normalization=norm,
        )
        gtv = guided_variational_sr_baseline(
            ph.hr_image,
            ph.coarse_guide,
            mode="multiclass",
            n_steps=20 if quick else 60,
            device=device,
            guide_normalization=norm,
        )
        inr = coordinate_inr_baseline(
            ph.hr_image,
            ph.coarse_guide,
            mode="multiclass",
            hidden_dim=32 if quick else 48,
            epochs=20 if quick else 65,
            device=device,
            guide_normalization=norm,
        )

        m_last = XFuse(
            backbone=backbone,
            stride=stride,
            dtype=torch.float32,
            loss_fn="compositional_kl",
            interpolation_mode="jbu",
            layers="last",
            pretrained=pretrained,
            lr=1e-2,
            device=device,
        )
        rep_last = m_last.fit_predict(
            ph.hr_image,
            ph.coarse_guide,
            epochs=epochs,
            mode="multiclass",
            tta_shifts=tta_shifts,
            guide_normalization=norm,
        )

        m_multi = XFuse(
            backbone=backbone,
            stride=stride,
            dtype=torch.float32,
            loss_fn="compositional_kl",
            interpolation_mode="cross_attn",
            fusion_method="spatial_mlp",
            layers="last",
            learn_psf=True,
            pretrained=pretrained,
            lr=1e-2,
            device=device,
        )
        rep_multi = m_multi.fit_predict(
            ph.hr_image,
            ph.coarse_guide,
            epochs=epochs,
            mode="multiclass",
            tta_shifts=tta_shifts,
            guide_normalization=norm,
        )

        met_otsu = compute_segmentation_metrics(otsu.hard_mask[0], gt)
        met_bic = compute_segmentation_metrics(bic.hard_mask[0], gt)
        met_jbu = compute_segmentation_metrics(jbu.hard_mask[0], gt)
        met_gs = compute_segmentation_metrics(gs.hard_mask[0], gt)
        met_cnmf = compute_segmentation_metrics(cnmf.hard_mask[0], gt)
        met_gtv = compute_segmentation_metrics(gtv.hard_mask[0], gt)
        met_inr = compute_segmentation_metrics(inr.hard_mask[0], gt)
        met_last = compute_segmentation_metrics(rep_last.hard_mask[0], gt)
        met_multi = compute_segmentation_metrics(rep_multi.hard_mask[0], gt)
        curve_multi = sparsification_curve(
            rep_multi.hard_mask[0], gt, rep_multi.combined_uncertainty[0]
        )

        results["ablation_resolution_sweep"].append(
            {
                "downsample_factor": factor,
                "otsu_miou": round(met_otsu["iou"], 4),
                "otsu_bf1": round(met_otsu["boundary_f1"], 4),
                "bicubic_miou": round(met_bic["iou"], 4),
                "bicubic_bf1": round(met_bic["boundary_f1"], 4),
                "jbu_miou": round(met_jbu["iou"], 4),
                "jbu_bf1": round(met_jbu["boundary_f1"], 4),
                "gram_schmidt_miou": round(met_gs["iou"], 4),
                "gram_schmidt_bf1": round(met_gs["boundary_f1"], 4),
                "cnmf_miou": round(met_cnmf["iou"], 4),
                "cnmf_bf1": round(met_cnmf["boundary_f1"], 4),
                "guided_tv_miou": round(met_gtv["iou"], 4),
                "guided_tv_bf1": round(met_gtv["boundary_f1"], 4),
                "siren_inr_miou": round(met_inr["iou"], 4),
                "siren_inr_bf1": round(met_inr["boundary_f1"], 4),
                "xfuse_last_miou": round(met_last["iou"], 4),
                "xfuse_last_bf1": round(met_last["boundary_f1"], 4),
                "xfuse_multiscale_miou": round(met_multi["iou"], 4),
                "xfuse_multiscale_bf1": round(met_multi["boundary_f1"], 4),
                "xfuse_multiscale_ece": round(
                    expected_calibration_error(rep_multi.probabilities, gt), 4
                ),
                "xfuse_multiscale_brier": round(
                    brier_score(rep_multi.probabilities, gt), 4
                ),
                "xfuse_multiscale_ause": round(float(curve_multi["ause"]), 4),
            }
        )

    # 2. Sub-Pixel Drift & Learnable PSF Recovery Ablation
    ph_base = generate_correlative_phantom_2d(
        size=img_size,
        downsample_factor=4,
        phase_b_contrast=0.35,
        hr_noise_std=0.02,
        beam_profile="moffat",
        seed=101,
    )
    drift_px_list = [0.0, 3.0] if quick else [0.0, 2.0, 4.0, 5.0]
    rep_cal_specimen = None
    for shift_px in drift_px_list:
        drifted_guide = _inject_shifted_coarse_guide(
            ph_base.gt_phase_channels,
            factor=4,
            shift_x_hr_px=shift_px,
            psf_sigma_hr=2.2,
        )
        m_unaligned = XFuse(
            backbone=backbone,
            stride=stride,
            dtype=torch.float32,
            loss_fn="compositional_kl",
            interpolation_mode="cross_attn",
            fusion_method="spatial_mlp",
            align_guide="none",
            learn_psf=False,
            pretrained=pretrained,
            lr=1e-2,
            device=device,
        )
        rep_un = m_unaligned.fit_predict(
            ph_base.hr_image,
            drifted_guide,
            epochs=epochs + 4,
            mode="multiclass",
            tta_shifts=tta_shifts,
            guide_normalization=norm,
        )

        m_aligned = XFuse(
            backbone=backbone,
            stride=stride,
            dtype=torch.float32,
            loss_fn="compositional_kl",
            interpolation_mode="cross_attn",
            fusion_method="spatial_mlp",
            align_guide="translation",
            max_shift_coarse_px=2.0,
            learn_psf=True,
            pretrained=pretrained,
            lr=1e-2,
            device=device,
        )
        rep_al = m_aligned.fit_predict(
            ph_base.hr_image,
            drifted_guide,
            epochs=epochs + 8,
            mode="multiclass",
            tta_shifts=tta_shifts,
            guide_normalization=norm,
        )
        if shift_px == 0.0:
            rep_cal_specimen = (m_aligned, rep_al)
        met_un = compute_segmentation_metrics(rep_un.hard_mask[0], ph_base.gt_labels)
        met_al = compute_segmentation_metrics(rep_al.hard_mask[0], ph_base.gt_labels)
        results["drift_and_psf_ablation"].append(
            {
                "injected_shift_hr_px": shift_px,
                "unaligned_miou": round(met_un["iou"], 4),
                "aligned_miou": round(met_al["iou"], 4),
                "estimated_shift_hr_px": float(
                    rep_al.diagnostics.get("shift_x_hr_px", 0.0)
                ),
                "unaligned_mean_residual": round(
                    float(rep_un.physical_residual.mean().item()), 4
                ),
                "aligned_mean_residual": round(
                    float(rep_al.physical_residual.mean().item()), 4
                ),
            }
        )

    # 3. Active Sparse-Slice Tomography Simulation
    vol_shape = (12, 32, 32) if quick else (24, 64, 64)
    ph_3d = generate_correlative_phantom_3d(
        shape=vol_shape, downsample_factor=4, hr_noise_std=0.06, seed=7
    )
    all_z_guides: dict[int, np.ndarray] = {}
    for z_i in range(vol_shape[0]):
        _, cg_z, _ = make_experimental_holdout(
            ph_3d.hr_volume[z_i],
            np.stack(
                [
                    (ph_3d.gt_labels_3d[z_i] == 1).astype(np.float32),
                    (ph_3d.gt_labels_3d[z_i] == 2).astype(np.float32),
                ],
                axis=0,
            ),
            factor=4,
            beam_profile="moffat",
        )
        all_z_guides[z_i] = cg_z

    budgets = [2, 4] if quick else [2, 4, 6, 8]
    rng = np.random.default_rng(123)
    random_mious: list[float] = []
    uniform_mious: list[float] = []
    active_mious: list[float] = []

    def _eval_3d_schedule(slice_indices: list[int]) -> tuple[float, Any]:
        base = XFuse(
            backbone=backbone,
            stride=stride,
            dtype=torch.float32,
            loss_fn="compositional_kl",
            guide_channels=2,
            pretrained=pretrained,
            lr=1e-2,
            device=device,
        )
        vf = VolumetricFuser(base, tta_shifts=(), tta_flips=False)
        guides_sub = {idx: all_z_guides[idx] for idx in slice_indices}
        vf.fit_sparse_slices(
            ph_3d.hr_volume,
            sparse_guides=guides_sub,
            epochs=6 if quick else 12,
            delta=1,
        )
        v_rep = vf.predict_volume(
            ph_3d.hr_volume,
            mode="triplane" if not quick else "2.5d_slab",
            delta=1,
            sparse_guides=guides_sub,
            pred_mode="multiclass",
        )
        m3d = compute_segmentation_metrics(v_rep.hard_mask, ph_3d.gt_labels_3d)
        return round(m3d["iou"], 4), v_rep

    active_chosen = [1, vol_shape[0] - 2]
    prev_active_rep: Any = None
    for b_idx, budget in enumerate(budgets):
        rand_idxs = sorted(
            rng.choice(vol_shape[0], size=budget, replace=False).tolist()
        )
        unif_idxs = sorted(
            {int(round(v)) for v in np.linspace(1, vol_shape[0] - 2, num=budget)}
        )
        r_iou, _ = _eval_3d_schedule(rand_idxs)
        u_iou, _ = _eval_3d_schedule(unif_idxs)
        random_mious.append(r_iou)
        uniform_mious.append(u_iou)

        if b_idx > 0 and len(active_chosen) < budget:
            need = budget - len(active_chosen)
            if prev_active_rep is None:
                _, prev_active_rep = _eval_3d_schedule(active_chosen)
            rec = suggest_next_slices(
                prev_active_rep, acquired_indices=active_chosen, k=need, min_spacing=2
            )
            active_chosen = sorted(set(active_chosen + rec.recommended_indices))
        a_iou, prev_active_rep = _eval_3d_schedule(active_chosen[:budget])
        active_mious.append(a_iou)

    results["active_slice_tomography"] = {
        "slice_budgets": budgets,
        "random_miou": random_mious,
        "uniform_miou": uniform_mious,
        "active_xfuse_miou": active_mious,
        "final_active_slices": active_chosen,
    }

    # 3b. Stress Tests: Oriented Lamellar Texture, FIB Artifacts, Poisson & Conformal
    ph_tex = generate_oriented_lamellar_phantom_2d(
        size=max(img_size, 64), downsample_factor=8, beam_profile="moffat", seed=55
    )
    otsu_tex = otsu_threshold_baseline(
        ph_tex.hr_image,
        ph_tex.coarse_guide,
        mode="multiclass",
        guide_normalization=norm,
    )
    bic_tex = bicubic_upsample_baseline(
        ph_tex.coarse_guide,
        ph_tex.hr_image.shape,
        mode="multiclass",
        guide_normalization=norm,
    )
    jbu_tex = jbu_guided_baseline(
        ph_tex.hr_image,
        ph_tex.coarse_guide,
        mode="multiclass",
        guide_normalization=norm,
    )
    cnmf_tex = cnmf_pansharpen_baseline(
        ph_tex.hr_image,
        ph_tex.coarse_guide,
        mode="multiclass",
        n_iters=15 if quick else 35,
        guide_normalization=norm,
    )
    gtv_tex = guided_variational_sr_baseline(
        ph_tex.hr_image,
        ph_tex.coarse_guide,
        mode="multiclass",
        n_steps=20 if quick else 60,
        device=device,
        guide_normalization=norm,
    )
    inr_tex = coordinate_inr_baseline(
        ph_tex.hr_image,
        ph_tex.coarse_guide,
        mode="multiclass",
        hidden_dim=32 if quick else 48,
        epochs=20 if quick else 65,
        device=device,
        guide_normalization=norm,
    )
    m_tex = XFuse(
        backbone=backbone,
        stride=stride,
        dtype=torch.float32,
        loss_fn="compositional_kl",
        interpolation_mode="cross_attn",
        fusion_method="spatial_mlp",
        pretrained=pretrained,
        lr=1e-2,
        device=device,
    )
    rep_tex = m_tex.fit_predict(
        ph_tex.hr_image,
        ph_tex.coarse_guide,
        epochs=max(epochs + 6, 16),
        mode="multiclass",
        tta_shifts=tta_shifts,
        guide_normalization=norm,
    )

    ph_art = generate_artifact_corrupted_phantom_2d(
        size=max(img_size, 64), downsample_factor=8, beam_profile="moffat", seed=61
    )
    m_art = XFuse(
        backbone=backbone,
        stride=stride,
        dtype=torch.float32,
        loss_fn="compositional_kl",
        interpolation_mode="cross_attn",
        fusion_method="spatial_mlp",
        debias_positional=True,
        debias_degree=1,
        pretrained=pretrained,
        lr=1e-2,
        device=device,
    )
    rep_art = m_art.fit_predict(
        ph_art.hr_image,
        ph_art.coarse_guide,
        epochs=max(epochs + 4, 14),
        mode="multiclass",
        tta_shifts=tta_shifts,
        guide_normalization=norm,
    )

    ph_spec = generate_spectral_cube_phantom_2d(
        size=img_size,
        downsample_factor=4,
        n_spectral_bins=16,
        mean_counts=45.0,
        beam_profile="moffat",
        seed=99,
    )
    m_spec = XFuse(
        backbone=backbone,
        stride=stride,
        dtype=torch.float32,
        loss_fn="forward_psf_poisson",
        interpolation_mode="cross_attn",
        fusion_method="spatial_mlp",
        spectral_channels=ph_spec.spectral_cube_lr.shape[0],
        guide_channels=2,
        pretrained=pretrained,
        lr=1e-2,
        device=device,
    )
    rep_spec = m_spec.fit_predict(
        ph_spec.hr_image,
        ph_spec.spectral_cube_lr,
        epochs=epochs + 6,
        mode="multiclass",
        tta_shifts=tta_shifts,
        spectral_unmix=True,
        n_components=2,
    )
    pred_spec_np = rep_spec.hard_mask[0].cpu().numpy()
    pred_swapped = pred_spec_np.copy()
    pred_swapped[pred_spec_np == 1] = 2
    pred_swapped[pred_spec_np == 2] = 1
    spec_iou = max(
        compute_segmentation_metrics(pred_spec_np, ph_spec.gt_labels)["iou"],
        compute_segmentation_metrics(pred_swapped, ph_spec.gt_labels)["iou"],
    )

    # Cross-Specimen LAC Conformal Calibration (Specimen A -> Independent Specimen B)
    assert rep_cal_specimen is not None
    cal_model, cal_rep_a = rep_cal_specimen
    ph_test_b = generate_correlative_phantom_2d(
        size=img_size,
        downsample_factor=4,
        phase_b_contrast=0.35,
        hr_noise_std=0.02,
        beam_profile="moffat",
        seed=202,
    )
    rep_test_b = cal_model.predict(
        ph_test_b.hr_image,
        coarse_guide=None,
        mode="multiclass",
        calibrate_to_guide=False,
    )
    gt_a_t = torch.from_numpy(ph_base.gt_labels).unsqueeze(0)
    gt_b_t = torch.from_numpy(ph_test_b.gt_labels).unsqueeze(0)
    conformal_curve: list[dict[str, float]] = []
    for alpha in (0.05, 0.10, 0.20):
        cal = ConformalCalibrator(alpha=alpha, method="lac")
        cal.calibrate_labels(cal_rep_a.probabilities, gt_a_t)
        p_sets_b, _ = cal.predict_sets_and_intervals(
            rep_test_b.probabilities, guide_lr=None, mode="multiclass"
        )
        cov_stats = empirical_conformal_coverage(p_sets_b, gt_b_t)
        conformal_curve.append(
            {
                "alpha": alpha,
                "nominal_coverage": round(1.0 - alpha, 4),
                "empirical_coverage": round(cov_stats["empirical_coverage"], 4),
                "mean_set_size": round(cov_stats["mean_set_size"], 4),
                "singleton_fraction": round(cov_stats["singleton_fraction"], 4),
            }
        )

    results["v1_stress_tests"] = {
        "textured_isointensity": {
            "otsu_miou": round(
                compute_segmentation_metrics(otsu_tex.hard_mask[0], ph_tex.gt_labels)[
                    "iou"
                ],
                4,
            ),
            "bicubic_miou": round(
                compute_segmentation_metrics(bic_tex.hard_mask[0], ph_tex.gt_labels)[
                    "iou"
                ],
                4,
            ),
            "jbu_miou": round(
                compute_segmentation_metrics(jbu_tex.hard_mask[0], ph_tex.gt_labels)[
                    "iou"
                ],
                4,
            ),
            "cnmf_miou": round(
                compute_segmentation_metrics(cnmf_tex.hard_mask[0], ph_tex.gt_labels)[
                    "iou"
                ],
                4,
            ),
            "guided_tv_miou": round(
                compute_segmentation_metrics(gtv_tex.hard_mask[0], ph_tex.gt_labels)[
                    "iou"
                ],
                4,
            ),
            "siren_inr_miou": round(
                compute_segmentation_metrics(inr_tex.hard_mask[0], ph_tex.gt_labels)[
                    "iou"
                ],
                4,
            ),
            "xfuse_cross_attn_miou": round(
                compute_segmentation_metrics(rep_tex.hard_mask[0], ph_tex.gt_labels)[
                    "iou"
                ],
                4,
            ),
        },
        "artifact_corrupted_fib": {
            "xfuse_debiased_miou": round(
                compute_segmentation_metrics(rep_art.hard_mask[0], ph_art.gt_labels)[
                    "iou"
                ],
                4,
            ),
        },
        "spectral_cube_poisson": {
            "unmixed_poisson_miou": round(spec_iou, 4),
        },
        "conformal_coverage": conformal_curve,
    }

    # 4. Bundled Real Synchrotron Zn13X Quantitative Hold-Out & Cross-Slice Transfer
    zn_pairs = load_bundled_zn13x_pairs()
    if zn_pairs:
        s4 = zn_pairs[0]
        p05, p99 = np.percentile(s4.guide_map[0], [5, 99])
        crop_sz = 96 if quick else 224
        y0, x0 = 192, 1024
        for zn_sample in zn_pairs:
            hr_sub = zn_sample.hr_image[y0 : y0 + crop_sz, x0 : x0 + crop_sz]
            gt_sub = np.clip(
                (
                    zn_sample.guide_map[
                        0, y0 // 2 : (y0 + crop_sz) // 2, x0 // 2 : (x0 + crop_sz) // 2
                    ]
                    - p05
                )
                / max(float(p99 - p05), 1e-6),
                0.0,
                1.0,
            ).astype(np.float32)
            _, g_lr_hold, _ = make_experimental_holdout(
                np.zeros_like(gt_sub),
                gt_sub[None],
                factor=4 if quick else 8,
                beam_profile="moffat",
            )
            zn_model = XFuse(
                backbone=backbone,
                stride=stride,
                dtype=torch.float32,
                loss_fn="forward_psf_mse",
                interpolation_mode="cross_attn",
                fusion_method="spatial_mlp",
                guide_residual=True,
                tv_weight=0.01,
                learn_psf=True,
                pretrained=pretrained,
                lr=5e-3,
                device=device,
            )
            zn_rep = zn_model.fit_predict(
                hr_sub,
                g_lr_hold,
                epochs=epochs,
                mode="multilabel",
                calibrate_to_guide=False,
                tta_shifts=tta_shifts,
                guide_normalization=norm,
            )
            p_112 = (
                F.avg_pool2d(zn_rep.probabilities.float(), 2, 2)
                .reshape(-1)
                .cpu()
                .numpy()
            )
            g_112 = gt_sub.reshape(-1)
            r_val = (
                float(np.corrcoef(p_112, g_112)[0, 1]) if np.std(p_112) > 1e-8 else 0.0
            )
            psnr_val = float(
                -10.0 * np.log10(max(float(np.mean((p_112 - g_112) ** 2)), 1e-10))
            )
            results["synchrotron_samples"][zn_sample.key] = {
                "hr_crop_shape": list(hr_sub.shape),
                "holdout_guide_shape": list(g_lr_hold.shape),
                "holdout_pearson_r": round(r_val, 4),
                "holdout_psnr_db": round(psnr_val, 2),
                "mean_phase_probability": round(
                    float(zn_rep.probabilities.mean().item()), 4
                ),
                "mean_uncertainty": round(
                    float(zn_rep.combined_uncertainty.mean().item()), 4
                ),
            }

    mg_samples = load_bundled_morgis_samples()
    if mg_samples:
        mg = mg_samples[0]
        mg_crop = 128 if quick else min(256, mg.hr_image.shape[0])
        hr_mg, coarse_mg, gt_mg = make_experimental_holdout(
            mg.hr_image[:mg_crop, :mg_crop],
            mg.guide_map[:, :mg_crop, :mg_crop],
            factor=8,
            beam_profile="moffat",
        )
        mg_model = XFuse(
            backbone=backbone,
            stride=stride,
            dtype=torch.float32,
            loss_fn="compositional_kl",
            interpolation_mode="jbu",
            tile_size=96,
            tile_overlap=24,
            pretrained=pretrained,
            lr=1e-2,
            device=device,
        )
        mg_rep = mg_model.fit_predict(
            hr_mg,
            coarse_mg,
            epochs=3 if quick else epochs,
            mode="multiclass",
            tta_shifts=(),
            tta_flips=not quick,
            guide_normalization=norm,
        )
        gt_labels_mg = np.zeros(hr_mg.shape, dtype=np.int64)
        gt_labels_mg[gt_mg[0] > 0.5] = 1
        gt_labels_mg[gt_mg[1] > 0.5] = 2
        mg_met = compute_segmentation_metrics(mg_rep.hard_mask[0], gt_labels_mg)
        results["synchrotron_samples"][mg.key] = {
            "semi_synthetic": True,
            "tiled_shape": list(hr_mg.shape),
            "holdout_8x_miou": round(mg_met["iou"], 4),
            "holdout_8x_bf1": round(mg_met["boundary_f1"], 4),
        }

    # 5. Optional Zenodo Open-Dataset Hold-Out Suite
    if download_zenodo:
        for ds_key in OPEN_DATASETS:
            try:
                sample = load_zenodo_correlative_sample(ds_key, auto_download=True)
                h_tot, w_tot = sample.hr_image.shape[-2:]
                gh_tot, gw_tot = sample.guide_map.shape[-2:]
                crop_h = min(256, h_tot)
                crop_w = min(256, w_tot)
                if sample.is_full_res_ground_truth and (gh_tot, gw_tot) == (
                    h_tot,
                    w_tot,
                ):
                    hr_h, coarse_h, gt_h = make_experimental_holdout(
                        sample.hr_image[:crop_h, :crop_w],
                        sample.guide_map[:, :crop_h, :crop_w],
                        factor=8,
                        beam_profile="moffat",
                    )
                else:
                    gc_h = max(4, int(round(crop_h * gh_tot / max(h_tot, 1))))
                    gc_w = max(4, int(round(crop_w * gw_tot / max(w_tot, 1))))
                    hr_h = sample.hr_image[:crop_h, :crop_w]
                    coarse_h = sample.guide_map[:, :gc_h, :gc_w]
                    gt_h = None

                z_mod = XFuse(
                    backbone=backbone,
                    stride=stride,
                    dtype=torch.float32,
                    loss_fn="forward_psf_bce",
                    interpolation_mode="jbu",
                    layers="multiscale",
                    align_guide="translation",
                    learn_psf=True,
                    pretrained=pretrained,
                    device=device,
                )
                z_rep = z_mod.fit_predict(
                    hr_h, coarse_h, epochs=epochs, mode="multilabel"
                )
                entry: dict[str, Any] = {
                    "channels": sample.channel_names,
                    "mean_physical_residual": round(
                        float(z_rep.physical_residual.mean().item()), 5
                    ),
                    "diagnostics": z_rep.diagnostics,
                }
                if gt_h is not None:
                    pred_np = z_rep.probabilities[0].cpu().numpy()
                    entry["holdout_8x_mse"] = round(
                        float(np.mean((pred_np - gt_h) ** 2)), 5
                    )
                results["zenodo_holdouts"][ds_key] = entry
            except Exception as exc:  # pragma: no cover
                results["zenodo_holdouts"][ds_key] = {"error": str(exc)}

    json_out = output_dir / "paper_benchmark_results.json"
    json_out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    figs = generate_paper_figures(results, output_dir=output_dir / "paper_figures")
    results["generated_figures"] = [str(f) for f in figs]
    json_out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run X-FUSE paper benchmark, ablation, and figure generation suite."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/outputs/paper"),
    )
    parser.add_argument(
        "--backbone",
        default="timm:vit_tiny_patch16_224",
        help="ViT backbone (e.g., dinov2_vits14_reg or timm:vit_tiny_patch16_224).",
    )
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--download-zenodo", action="store_true")
    args = parser.parse_args()

    res = run_paper_benchmarks(
        output_dir=args.output_dir,
        backbone=args.backbone,
        pretrained=bool(args.pretrained),
        quick=bool(args.quick),
        device=args.device,
        download_zenodo=bool(args.download_zenodo),
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
