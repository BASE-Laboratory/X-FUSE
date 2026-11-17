"""End-to-end paper benchmark & ablation runner for X-FUSE (v0.5 / v1.0).

Executes:
1. **Multi-Scale & Layer-Pyramid Super-Resolution Ablation** (Bicubic vs.
   Classical JBU vs. ViT+K-Means vs. X-FUSE Last-Layer vs. X-FUSE Multi-Scale).
2. **Sub-Pixel Drift & Learnable Anisotropic PSF Recovery** (evaluating
   ``DifferentiableGuideAligner`` + ``learn_psf=True`` under controlled shifts).
3. **Closed-Loop Active Sparse-Slice Tomography** (Random vs. Uniform vs.
   X-FUSE ``suggest_next_slices`` beamtime efficiency curve).
4. **Bundled Synchrotron Benchmarks** (``Zn13X`` slices 4 & 9, plus tiled
   ``MOR-GIS`` hold-out evaluation) and optional **Zenodo Open-Dataset Suite**.

Usage:
    # Fast offline CPU verification + vector PDF/PNG figure generation:
    uv run python -m benchmarks.run_paper_benchmarks --quick

    # Full GPU pretrained paper sweep:
    uv run python -m benchmarks.run_paper_benchmarks \\
        --backbone dinov2_vitb14_reg --pretrained --device cuda --download-zenodo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import shift as nd_shift

from x_fuse import (
    OPEN_DATASETS,
    VolumetricFuser,
    XFuse,
    bicubic_upsample_baseline,
    brier_score,
    compute_segmentation_metrics,
    dino_kmeans_baseline,
    expected_calibration_error,
    generate_correlative_phantom_2d,
    generate_correlative_phantom_3d,
    jbu_guided_baseline,
    make_experimental_holdout,
    sparsification_curve,
    suggest_next_slices,
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
    physical PSF downsampling to simulate beam/stage drift in ``G_LR``.
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
        dummy_hr, shifted, factor=factor, psf_sigma=psf_sigma_hr
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
    stride = 8 if quick else 4
    epochs = 8 if quick else 30
    tta_shifts: tuple[int, ...] = () if quick else (1, 2)
    factors = [4, 8] if quick else [4, 8, 16]

    results: dict[str, Any] = {
        "config": {
            "backbone": backbone,
            "pretrained": pretrained,
            "quick": quick,
            "device": device,
            "stride": stride,
            "epochs": epochs,
        },
        "ablation_resolution_sweep": [],
        "drift_and_psf_ablation": [],
        "active_slice_tomography": {},
        "synchrotron_samples": {},
        "zenodo_holdouts": {},
    }

    # 1. Resolution-Gap & Multi-Scale Layer Ablation
    for factor in factors:
        ph = generate_correlative_phantom_2d(
            size=img_size,
            downsample_factor=factor,
            phase_b_contrast=0.35,
            hr_noise_std=0.03,
            poisson_counts=300.0,
            seed=42,
        )
        gt = ph.gt_labels
        bic = bicubic_upsample_baseline(
            ph.coarse_guide, (img_size, img_size), mode="multiclass"
        )
        jbu = jbu_guided_baseline(ph.hr_image, ph.coarse_guide, mode="multiclass")

        m_last = XFuse(
            backbone=backbone,
            stride=stride,
            dtype=torch.float32,
            loss_fn="forward_psf_bce",
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
        )

        # Extract raw ViT features for unsupervised K-Means baseline
        hr_prep, (oh, ow) = m_last._prepare_high_level_hr(ph.hr_image)
        raw_feats = super(XFuse, m_last).forward_sequential(hr_prep)[:, :, :oh, :ow]
        km = dino_kmeans_baseline(raw_feats, ph.coarse_guide, n_clusters=3)

        m_multi = XFuse(
            backbone=backbone,
            stride=stride,
            dtype=torch.float32,
            loss_fn="forward_psf_bce",
            interpolation_mode="jbu",
            layers="multiscale",
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
        )

        met_bic = compute_segmentation_metrics(bic.hard_mask[0], gt)
        met_jbu = compute_segmentation_metrics(jbu.hard_mask[0], gt)
        met_km = compute_segmentation_metrics(km.hard_mask[0], gt)
        met_last = compute_segmentation_metrics(rep_last.hard_mask[0], gt)
        met_multi = compute_segmentation_metrics(rep_multi.hard_mask[0], gt)
        curve_multi = sparsification_curve(
            rep_multi.hard_mask[0], gt, rep_multi.combined_uncertainty[0]
        )

        results["ablation_resolution_sweep"].append(
            {
                "downsample_factor": factor,
                "bicubic_miou": round(met_bic["iou"], 4),
                "bicubic_bf1": round(met_bic["boundary_f1"], 4),
                "jbu_miou": round(met_jbu["iou"], 4),
                "jbu_bf1": round(met_jbu["boundary_f1"], 4),
                "vit_kmeans_miou": round(met_km["iou"], 4),
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
        seed=101,
    )
    drift_px_list = [0.0, 3.0] if quick else [0.0, 2.0, 4.0, 5.0]
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
            loss_fn="forward_psf_bce",
            interpolation_mode="jbu",
            align_guide="none",
            learn_psf=False,
            pretrained=pretrained,
            lr=1e-2,
            device=device,
        )
        rep_un = m_unaligned.fit_predict(
            ph_base.hr_image,
            drifted_guide,
            epochs=epochs + 6,
            mode="multiclass",
            tta_shifts=tta_shifts,
        )

        m_aligned = XFuse(
            backbone=backbone,
            stride=stride,
            dtype=torch.float32,
            loss_fn="forward_psf_bce",
            interpolation_mode="jbu",
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
            epochs=epochs + 10,
            mode="multiclass",
            tta_shifts=tta_shifts,
        )
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
    # Build per-slice 2D coarse guides for every Z plane so schedules can query any z
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
            loss_fn="forward_psf_bce",
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

    # Active schedule starts from 2 boundary seed slices and greedily adds slices
    active_chosen = [1, vol_shape[0] - 2]
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
            _, prev_rep = _eval_3d_schedule(active_chosen)
            rec = suggest_next_slices(
                prev_rep, acquired_indices=active_chosen, k=need, min_spacing=2
            )
            active_chosen = sorted(set(active_chosen + rec.recommended_indices))
        a_iou, _ = _eval_3d_schedule(active_chosen[:budget])
        active_mious.append(a_iou)

    results["active_slice_tomography"] = {
        "slice_budgets": budgets,
        "random_miou": random_mious,
        "uniform_miou": uniform_mious,
        "active_xfuse_miou": active_mious,
        "final_active_slices": active_chosen,
    }

    # 4. Bundled Synchrotron Samples (Zn13X Slices 4 & 9 + Tiled MOR-GIS Hold-Out)
    for zn_sample in load_bundled_zn13x_pairs():
        crop_sz = 96 if quick else min(256, zn_sample.hr_image.shape[0])
        h_full, w_full = zn_sample.hr_image.shape
        y0, x0 = (h_full - crop_sz) // 2, (w_full - crop_sz) // 2
        hr_sub = zn_sample.hr_image[y0 : y0 + crop_sz, x0 : x0 + crop_sz]
        gh_full, gw_full = zn_sample.guide_map.shape[1:]
        gy0 = int(round(y0 * gh_full / h_full))
        gx0 = int(round(x0 * gw_full / w_full))
        gh = max(4, int(round(crop_sz * gh_full / h_full)))
        gw = max(4, int(round(crop_sz * gw_full / w_full)))
        g_sub = zn_sample.guide_map[:, gy0 : gy0 + gh, gx0 : gx0 + gw]

        zn_model = XFuse(
            backbone=backbone,
            stride=stride,
            dtype=torch.float32,
            loss_fn="forward_psf_bce",
            interpolation_mode="jbu",
            layers="multiscale",
            pretrained=pretrained,
            lr=1e-2,
            device=device,
        )
        zn_rep = zn_model.fit_predict(
            hr_sub, g_sub, epochs=epochs, mode="multilabel", tta_shifts=tta_shifts
        )
        results["synchrotron_samples"][zn_sample.key] = {
            "hr_crop_shape": list(hr_sub.shape),
            "guide_crop_shape": list(g_sub.shape),
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
        )
        mg_model = XFuse(
            backbone=backbone,
            stride=stride,
            dtype=torch.float32,
            loss_fn="forward_psf_bce",
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
            tta_shifts=() if quick else (1,),
            tta_flips=not quick,
        )
        gt_labels_mg = np.zeros(hr_mg.shape, dtype=np.int64)
        gt_labels_mg[gt_mg[0] > 0.5] = 1
        gt_labels_mg[gt_mg[1] > 0.5] = 2
        mg_met = compute_segmentation_metrics(mg_rep.hard_mask[0], gt_labels_mg)
        results["synchrotron_samples"][mg.key] = {
            "tiled_shape": list(hr_mg.shape),
            "holdout_8x_miou": round(mg_met["iou"], 4),
            "holdout_8x_bf1": round(mg_met["boundary_f1"], 4),
        }

    # 5. Optional Zenodo Open-Dataset Hold-Out Suite
    if download_zenodo:
        for ds_key in OPEN_DATASETS:
            try:
                sample = load_zenodo_correlative_sample(ds_key, auto_download=True)
                hr_h, coarse_h, gt_h = make_experimental_holdout(
                    sample.hr_image[:256, :256],
                    sample.guide_map[:, :256, :256],
                    factor=8,
                )
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
                pred_np = z_rep.probabilities[0].cpu().numpy()
                mse_val = float(np.mean((pred_np - gt_h) ** 2))
                results["zenodo_holdouts"][ds_key] = {
                    "channels": sample.channel_names,
                    "holdout_8x_mse": round(mse_val, 5),
                    "diagnostics": z_rep.diagnostics,
                }
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
        help="ViT backbone (e.g., dinov2_vitb14_reg or timm:vit_tiny_patch16_224).",
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
