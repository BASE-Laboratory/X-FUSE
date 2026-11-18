"""Reproducible multi-regime correlative super-resolution, uncertainty, and
SuRVoS-2.5D benchmark suite for X-FUSE.

Evaluates:
1. **Resolution-Gap Sweep** (4x, 8x, 16x): Bicubic vs. Classical JBU vs.
   Unsupervised ViT+K-Means vs. ViT+Sign-Aligned PCA/Otsu vs. X-FUSE
   (Forward-PSF Learned Probe + JBU + Prior Calibration).
2. **Contrast-to-Zero Physical Diagnostic Sweep**: Verifies X-FUSE's forward
   physical residual ``|D(P_HR) - G_LR|`` spikes specifically on chemically
   distinct phases whose structural contrast ``Delta_mu -> 0``.
3. **Uncertainty Sparsification & Calibration**: ECE, Brier Score, AUSE, and
   retained mIoU vs. abstained fraction.
4. **SuRVoS 2.5D Volumetric Propagation**: Sparse 2D slices -> full 3D stack
   comparing ``2d``, ``2.5d_slab``, and orthogonal ``triplane`` fusion.
5. **Bundled Experimental Smoke Test**: Synchrotron XCT + XRD-CT ``Zn13X`` slice
   (``data/samples/Zn13X``).

Usage:
    uv run python -m benchmarks.run_synthetic_suite --quick
    uv run python -m benchmarks.run_synthetic_suite --dino timm:vit_small_patch16_224
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from x_fuse import (
    CorrelativeFuser,
    VolumetricFuser,
    bicubic_upsample_baseline,
    brier_score,
    build_default_tta,
    compute_segmentation_metrics,
    dino_kmeans_baseline,
    ensure_spatial_divisible,
    expected_calibration_error,
    generate_correlative_phantom_2d,
    generate_correlative_phantom_3d,
    get_sample_dataset_path,
    guide_to_tensor,
    jbu_guided_baseline,
    load_correlative_tiff_pair,
    normalize_structural,
    pca_otsu_segment,
    sparsification_curve,
)


def _train_and_predict_xfuse_2d(
    hr_img: np.ndarray,
    coarse_guide: np.ndarray,
    dino_name: str,
    pretrained: bool,
    stride: int,
    epochs: int,
    tta_shifts: list[int],
    device: str,
):
    hr_t = normalize_structural(hr_img, device=device)
    hr_pad, (orig_h, orig_w) = ensure_spatial_divisible(
        hr_t, patch_size=16, stride=stride
    )
    guide_t = guide_to_tensor(coarse_guide, device=device, normalize=True)
    n_guides = guide_t.shape[1]

    fwd, inv = build_default_tta(
        shift_dists=tta_shifts, pattern="Neumann", include_flips=bool(tta_shifts)
    )
    fuser = CorrelativeFuser(
        dino_name=dino_name,
        xrd_fuse_method="learned_gating",
        stride=stride,
        dtype=torch.float32,
        loss_fn="forward_psf_bce",
        guide_channels=n_guides,
        interpolation_mode="jbu",
        pretrained=pretrained,
        lr=8e-3,
        device=device,
    )
    fuser.set_guide_transforms(guide_t, fwd, inv)
    for _ in range(epochs):
        fuser.train_fusion_step(hr_pad)

    mode = "multiclass" if n_guides >= 2 else "multilabel"
    rep = fuser.predict_with_uncertainty(
        hr_pad, guide_lr=guide_t, mode=mode, calibrate_to_guide=True
    )
    raw_feats = fuser.forward_raw_features(hr_pad)[:, :, :orig_h, :orig_w]
    if fuser.learned_gating is not None:
        fuser.learned_gating.select_top_k(max(8, fuser.feat_dim // 4))
    gated_feats = fuser.forward_sequential(hr_pad)[:, :, :orig_h, :orig_w]
    return rep, raw_feats, gated_feats, (orig_h, orig_w)


def run_benchmark(
    output_dir: Path,
    dino_name: str = "timm:vit_tiny_patch16_224",
    pretrained: bool = False,
    quick: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    matplotlib.use("Agg")
    output_dir.mkdir(parents=True, exist_ok=True)
    img_size = 64 if quick else 128
    stride = 8 if quick else 4
    epochs = 12 if quick else 25
    tta_shifts = [1] if quick else [1, 2]
    factors = [4, 8] if quick else [4, 8, 16]

    results: dict[str, Any] = {
        "config": {
            "dino_name": dino_name,
            "pretrained": pretrained,
            "quick": quick,
            "img_size": img_size,
            "stride": stride,
            "epochs": epochs,
        },
        "resolution_sweep": [],
        "contrast_sweep": [],
        "volumetric_25d": {},
        "zn13x_sample": {},
    }

    # 1. Resolution Gap Sweep + Sparsification Analysis
    last_curve = None
    for factor in factors:
        phantom = generate_correlative_phantom_2d(
            size=img_size,
            downsample_factor=factor,
            phase_b_contrast=0.35,
            hr_noise_std=0.03,
            poisson_counts=250.0,
            seed=42,
        )
        gt = phantom.gt_labels
        bic = bicubic_upsample_baseline(
            phantom.coarse_guide, (img_size, img_size), mode="multiclass"
        )
        jbu = jbu_guided_baseline(
            phantom.hr_image, phantom.coarse_guide, mode="multiclass"
        )
        xf_rep, raw_feats, gated_feats, (oh, ow) = _train_and_predict_xfuse_2d(
            phantom.hr_image,
            phantom.coarse_guide,
            dino_name=dino_name,
            pretrained=pretrained,
            stride=stride,
            epochs=epochs,
            tta_shifts=tta_shifts,
            device=device,
        )
        km = dino_kmeans_baseline(raw_feats, phantom.coarse_guide, n_clusters=3)
        pca_a, _ = pca_otsu_segment(gated_feats, guide_lr=phantom.coarse_guide[0])

        xf_mask = xf_rep.hard_mask[0, :oh, :ow].cpu().numpy()
        xf_probs = xf_rep.probabilities[:, :, :oh, :ow].cpu()
        xf_unc = xf_rep.combined_uncertainty[0, :oh, :ow].cpu().numpy()

        m_bic = compute_segmentation_metrics(bic.hard_mask[0], gt)
        m_jbu = compute_segmentation_metrics(jbu.hard_mask[0], gt)
        m_km = compute_segmentation_metrics(km.hard_mask[0], gt)
        m_pca_a = compute_segmentation_metrics(
            pca_a, phantom.gt_phase_channels[0].astype(np.int64)
        )
        m_xf = compute_segmentation_metrics(xf_mask, gt)

        ece_xf = expected_calibration_error(xf_probs, gt)
        brier_xf = brier_score(xf_probs, gt)
        curve = sparsification_curve(xf_mask, gt, xf_unc)
        last_curve = curve

        results["resolution_sweep"].append(
            {
                "downsample_factor": factor,
                "bicubic_miou": round(m_bic["iou"], 4),
                "bicubic_bf1": round(m_bic["boundary_f1"], 4),
                "jbu_miou": round(m_jbu["iou"], 4),
                "jbu_bf1": round(m_jbu["boundary_f1"], 4),
                "vit_kmeans_miou": round(m_km["iou"], 4),
                "vit_pca_otsu_phaseA_iou": round(m_pca_a["iou"], 4),
                "xfuse_miou": round(m_xf["iou"], 4),
                "xfuse_bf1": round(m_xf["boundary_f1"], 4),
                "xfuse_ece": round(ece_xf, 4),
                "xfuse_brier": round(brier_xf, 4),
                "xfuse_ause": round(float(curve["ause"]), 4),
                "xfuse_miou_at_20pct_abstain": round(
                    float(curve["retained_miou"][2]), 4
                ),
            }
        )

    # 2. Contrast-to-Zero Physical Residual Sweep
    for contrast_b in (0.35, 0.12, 0.0):
        ph_c = generate_correlative_phantom_2d(
            size=img_size,
            downsample_factor=8,
            phase_b_contrast=contrast_b,
            hr_noise_std=0.03,
            seed=77,
        )
        rep_c, _, _, (oh, ow) = _train_and_predict_xfuse_2d(
            ph_c.hr_image,
            ph_c.coarse_guide,
            dino_name=dino_name,
            pretrained=pretrained,
            stride=stride,
            epochs=epochs,
            tta_shifts=tta_shifts,
            device=device,
        )
        res_map = rep_c.physical_residual[0, :oh, :ow].cpu().numpy()
        unc_map = rep_c.combined_uncertainty[0, :oh, :ow].cpu().numpy()
        mask_b = ph_c.gt_phase_channels[1] > 0.5
        mask_a = ph_c.gt_phase_channels[0] > 0.5
        results["contrast_sweep"].append(
            {
                "phase_b_contrast": contrast_b,
                "mean_residual_in_phase_a": round(float(res_map[mask_a].mean()), 4),
                "mean_residual_in_phase_b": round(float(res_map[mask_b].mean()), 4),
                "mean_uncertainty_in_phase_b": round(float(unc_map[mask_b].mean()), 4),
            }
        )

    # 3. SuRVoS 2.5D Sparse-Slice -> 3D Volume Propagation
    vol_shape = (16, 32, 32) if quick else (24, 64, 64)
    ph_3d = generate_correlative_phantom_3d(
        shape=vol_shape, downsample_factor=4, hr_noise_std=0.08, seed=99
    )
    sparse_slices = {
        2: ph_3d.coarse_guide_3d[:, 0],
        vol_shape[0] // 2: ph_3d.coarse_guide_3d[:, vol_shape[0] // 8],
        vol_shape[0] - 3: ph_3d.coarse_guide_3d[:, -1],
    }
    for v_mode in ("2d", "2.5d_slab", "triplane"):
        base_fuser = CorrelativeFuser(
            dino_name=dino_name,
            xrd_fuse_method="learned_gating",
            stride=stride,
            dtype=torch.float32,
            loss_fn="forward_psf_bce",
            guide_channels=2,
            pretrained=pretrained,
            lr=8e-3,
            device=device,
        )
        vf = VolumetricFuser(base_fuser, tta_shifts=[], tta_flips=False)
        vf.fit_sparse_slices(
            ph_3d.hr_volume,
            sparse_guides=sparse_slices,
            epochs=6 if quick else 12,
            delta=0 if v_mode == "2d" else 1,
        )
        v_rep = vf.predict_volume(
            ph_3d.hr_volume,
            mode=v_mode,  # type: ignore[arg-type]
            delta=1,
            coarse_guide_3d=ph_3d.coarse_guide_3d,
            pred_mode="multiclass",
        )
        v_metrics = compute_segmentation_metrics(v_rep.hard_mask, ph_3d.gt_labels_3d)
        results["volumetric_25d"][v_mode] = {
            "miou_3d": round(v_metrics["iou"], 4),
            "dice_3d": round(v_metrics["dice"], 4),
            "z_slice_consistency": round(v_rep.z_slice_consistency, 4),
            "mean_triplane_variance": round(
                float(v_rep.triplane_variance.mean().item()), 6
            ),
        }

    # 4. Bundled Experimental Sample Smoke Run (Zn13X XCT + XRD-CT)
    zn_dir = get_sample_dataset_path("Zn13X")
    zn_hr_full, zn_guide_full, _ = load_correlative_tiff_pair(
        zn_dir / "high_4.tiff", [zn_dir / "low_4.tiff"]
    )
    # Center-crop structural slice for fast deterministic CPU smoke validation
    crop_sz = 96 if quick else min(224, zn_hr_full.shape[0])
    y0 = (zn_hr_full.shape[0] - crop_sz) // 2
    x0 = (zn_hr_full.shape[1] - crop_sz) // 2
    zn_hr = zn_hr_full[y0 : y0 + crop_sz, x0 : x0 + crop_sz]
    gy0 = int(round(y0 * zn_guide_full.shape[1] / zn_hr_full.shape[0]))
    gx0 = int(round(x0 * zn_guide_full.shape[2] / zn_hr_full.shape[1]))
    gh = max(4, int(round(crop_sz * zn_guide_full.shape[1] / zn_hr_full.shape[0])))
    gw = max(4, int(round(crop_sz * zn_guide_full.shape[2] / zn_hr_full.shape[1])))
    zn_guide = zn_guide_full[:, gy0 : gy0 + gh, gx0 : gx0 + gw]

    zn_rep, _, _, (z_oh, z_ow) = _train_and_predict_xfuse_2d(
        zn_hr,
        zn_guide,
        dino_name=dino_name,
        pretrained=pretrained,
        stride=stride,
        epochs=epochs,
        tta_shifts=tta_shifts,
        device=device,
    )
    results["zn13x_sample"] = {
        "cropped_hr_shape": [int(z_oh), int(z_ow)],
        "coarse_guide_shape": list(zn_guide.shape[1:]),
        "predicted_phase_fraction": round(
            float(zn_rep.probabilities[0, 0, :z_oh, :z_ow].mean().item()), 4
        ),
        "coarse_guide_fraction": round(
            float(guide_to_tensor(zn_guide).mean().item()), 4
        ),
        "mean_uncertainty": round(
            float(zn_rep.combined_uncertainty[0, :z_oh, :z_ow].mean().item()), 4
        ),
    }

    # Save JSON + Markdown Summary + Diagnostic Plot
    json_path = output_dir / "benchmark_results.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n")

    hdr1 = (
        "| Gap | Bicubic mIoU | Classical JBU mIoU | ViT+KMeans mIoU "
        "| X-FUSE mIoU | X-FUSE BF1 | X-FUSE ECE | mIoU @ 20% Abstain |"
    )
    md_lines = [
        "# X-FUSE Quantitative Benchmark Summary",
        "",
        f"- Backbone: `{dino_name}` (`pretrained={pretrained}`, `stride={stride}`)",
        "",
        "## 1. Resolution-Gap Super-Resolution Sweep",
        "",
        hdr1,
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results["resolution_sweep"]:
        md_lines.append(
            f"| {row['downsample_factor']}x | {row['bicubic_miou']:.3f} | "
            f"{row['jbu_miou']:.3f} | {row['vit_kmeans_miou']:.3f} | "
            f"**{row['xfuse_miou']:.3f}** | {row['xfuse_bf1']:.3f} | "
            f"{row['xfuse_ece']:.3f} | {row['xfuse_miou_at_20pct_abstain']:.3f} |"
        )
    hdr2 = (
        "| Phase-B Contrast | Residual in Phase A | Residual in Phase B "
        "| Uncertainty in Phase B |"
    )
    md_lines.extend(
        [
            "",
            "## 2. Invisible-Phase Contrast-to-Zero Physical Residual Check",
            "",
            hdr2,
            "|---:|---:|---:|---:|",
        ]
    )
    for row in results["contrast_sweep"]:
        res_a_s = f"{row['mean_residual_in_phase_a']:.3f}"
        res_b_s = f"{row['mean_residual_in_phase_b']:.3f}"
        unc_b_s = f"{row['mean_uncertainty_in_phase_b']:.3f}"
        md_lines.append(
            f"| {row['phase_b_contrast']:.2f} | {res_a_s} | {res_b_s} | {unc_b_s} |"
        )
    hdr3 = (
        "| Mode | 3D mIoU | 3D Dice | Z-Slice Consistency | Mean Tri-Plane Variance |"
    )
    md_lines.extend(
        [
            "",
            "## 3. SuRVoS-Style 2.5D Sparse-Slice -> 3D Volume Propagation",
            "",
            hdr3,
            "|:---|---:|---:|---:|---:|",
        ]
    )
    for mode_k, row in results["volumetric_25d"].items():
        md_lines.append(
            f"| `{mode_k}` | {row['miou_3d']:.3f} | {row['dice_3d']:.3f} | "
            f"{row['z_slice_consistency']:.3f} | {row['mean_triplane_variance']:.5f} |"
        )
    (output_dir / "benchmark_summary.md").write_text("\n".join(md_lines) + "\n")

    if last_curve is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(
            last_curve["abstain_fractions"] * 100.0,
            last_curve["retained_miou"],
            marker="o",
            label="X-FUSE Retained mIoU",
        )
        axes[0].set_xlabel("Abstained Uncertain Pixels (%)")
        axes[0].set_ylabel("Retained mIoU")
        axes[0].set_title("Uncertainty Sparsification Curve")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        contrasts = [r["phase_b_contrast"] for r in results["contrast_sweep"]]
        res_b = [r["mean_residual_in_phase_b"] for r in results["contrast_sweep"]]
        res_a = [r["mean_residual_in_phase_a"] for r in results["contrast_sweep"]]
        axes[1].plot(contrasts, res_b, marker="s", label="Phase B (swept contrast)")
        axes[1].plot(
            contrasts,
            res_a,
            marker="^",
            linestyle="--",
            label="Phase A (fixed contrast)",
        )
        axes[1].invert_xaxis()
        axes[1].set_xlabel("Phase B Structural Contrast in I_HR")
        axes[1].set_ylabel("Physical Forward Residual |D(P_HR) - G_LR|")
        axes[1].set_title("Invisible-Phase Detection via Forward PSF")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(output_dir / "uncertainty_diagnostics.png", dpi=160)
        plt.close(fig)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run X-FUSE correlative benchmark suite."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/outputs"),
        help="Directory for JSON, Markdown, and diagnostic plots.",
    )
    parser.add_argument(
        "--dino",
        type=str,
        default="timm:vit_tiny_patch16_224",
        help="ViT backbone ID (e.g., dinov2_vits14 or timm:vit_small_patch16_224).",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Download pretrained ViT weights (omit for offline CI/quick runs).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fast smoke-scale sweep suitable for CPU verification.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    res = run_benchmark(
        output_dir=args.output_dir,
        dino_name=args.dino,
        pretrained=args.pretrained,
        quick=args.quick,
        device=args.device,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
