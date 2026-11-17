"""Publication-grade vector PDF & 300-DPI PNG figure generator for X-FUSE
benchmark and ablation outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def generate_paper_figures(
    results: dict[str, Any], output_dir: str | Path
) -> list[Path]:
    """Render multi-panel vector PDF and PNG figures from ``results``."""
    matplotlib.use("Agg")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
        }
    )

    # 1. Resolution-Gap & Architecture Ablation Panel
    ablation_rows = results.get("ablation_resolution_sweep", [])
    if ablation_rows:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        factors = [row["downsample_factor"] for row in ablation_rows]
        series = [
            ("bicubic_miou", "Bicubic Interpolation", "o--", "#7f8c8d"),
            ("jbu_miou", "Classical JBU (edxia)", "s--", "#2980b9"),
            ("cnmf_miou", "CNMF Pan-Sharpening", "v--", "#16a085"),
            ("guided_tv_miou", "Guided Variational TV", "p--", "#27ae60"),
            ("siren_inr_miou", "SIREN Coordinate INR", "h:", "#d35400"),
            ("vit_kmeans_miou", "ViT + K-Means", "^:", "#8e44ad"),
            ("xfuse_last_miou", "X-FUSE (Last Layer)", "d-", "#e67e22"),
            ("xfuse_multiscale_miou", "X-FUSE v1.0 (CrossAttn + PSF)", "*-", "#c0392b"),
        ]
        for key, label, style, color in series:
            if key in ablation_rows[0]:
                vals = [row[key] for row in ablation_rows]
                axes[0].plot(
                    factors, vals, style, color=color, linewidth=1.8, label=label
                )
        axes[0].set_xlabel("Resolution Gap (Coarse Pixel Ratio)")
        axes[0].set_ylabel("Macro Foreground mIoU")
        axes[0].set_title("A. Phase Super-Resolution vs. Resolution Gap")
        axes[0].set_xticks(factors)
        axes[0].set_xticklabels([f"{f}×" for f in factors])
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best", fontsize=7.5)

        bf1_keys = [
            ("bicubic_bf1", "Bicubic", "#7f8c8d"),
            ("jbu_bf1", "Classical JBU", "#2980b9"),
            ("guided_tv_bf1", "Guided TV", "#27ae60"),
            ("xfuse_last_bf1", "X-FUSE (Last)", "#e67e22"),
            ("xfuse_multiscale_bf1", "X-FUSE v1.0", "#c0392b"),
        ]
        x_pos = np.arange(len(factors))
        present_bf1 = [item for item in bf1_keys if item[0] in ablation_rows[0]]
        width = 0.8 / max(len(present_bf1), 1)
        offset0 = -0.5 * (len(present_bf1) - 1)
        for idx, (bkey, blabel, bcol) in enumerate(present_bf1):
            vals = [row[bkey] for row in ablation_rows]
            axes[1].bar(
                x_pos + (offset0 + idx) * width,
                vals,
                width=width,
                label=blabel,
                color=bcol,
            )
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels([f"{f}×" for f in factors])
        axes[1].set_xlabel("Resolution Gap")
        axes[1].set_ylabel("Boundary F1 Score (BF1)")
        axes[1].set_title("B. Sub-Pixel Grain/Pore Boundary Fidelity")
        axes[1].grid(True, axis="y", alpha=0.3)
        axes[1].legend(loc="best")

        fig.tight_layout()
        for ext in ("pdf", "png"):
            p = out_dir / f"fig1_resolution_and_ablation.{ext}"
            fig.savefig(p, dpi=300)
            saved_paths.append(p)
        plt.close(fig)

    # 2. Sub-Pixel Drift & Learnable PSF Recovery Panel
    drift_rows = results.get("drift_and_psf_ablation", [])
    if drift_rows:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        shifts = [row["injected_shift_hr_px"] for row in drift_rows]
        unaligned_iou = [row["unaligned_miou"] for row in drift_rows]
        aligned_iou = [row["aligned_miou"] for row in drift_rows]
        axes[0].plot(
            shifts,
            unaligned_iou,
            "o--",
            color="#7f8c8d",
            linewidth=1.8,
            label="Unaligned X-FUSE",
        )
        axes[0].plot(
            shifts,
            aligned_iou,
            "s-",
            color="#16a085",
            linewidth=2.0,
            label="X-FUSE + Differentiable Drift & PSF",
        )
        axes[0].set_xlabel("Injected Guide Drift (HR Pixels)")
        axes[0].set_ylabel("Segmentation mIoU")
        axes[0].set_title("A. Robustness to Correlative Stage / Beam Drift")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        est_shifts = [abs(row["estimated_shift_hr_px"]) for row in drift_rows]
        axes[1].plot(shifts, shifts, "k:", label="Ideal Recovery (y = x)")
        axes[1].plot(
            shifts,
            est_shifts,
            "D-",
            color="#2980b9",
            linewidth=1.8,
            label="Recovered |Shift| (HR px)",
        )
        axes[1].set_xlabel("True Injected Drift (HR Pixels)")
        axes[1].set_ylabel("Estimated Drift Magnitude (HR Pixels)")
        axes[1].set_title("B. Sub-Coarse-Pixel Drift Estimation")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        for ext in ("pdf", "png"):
            p = out_dir / f"fig2_drift_and_psf_recovery.{ext}"
            fig.savefig(p, dpi=300)
            saved_paths.append(p)
        plt.close(fig)

    # 3. Active Sparse-Slice Tomography Beamtime Curve
    active_data = results.get("active_slice_tomography", {})
    if active_data and "slice_budgets" in active_data:
        fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2))
        budgets = active_data["slice_budgets"]
        for sched_key, label, style, col in [
            ("random_miou", "Random Slice Schedule", "o:", "#95a5a6"),
            ("uniform_miou", "Uniform Equidistant Slices", "s--", "#2980b9"),
            (
                "active_xfuse_miou",
                "X-FUSE Active Uncertainty Steering",
                "*-",
                "#c0392b",
            ),
        ]:
            if sched_key in active_data:
                ax.plot(
                    budgets,
                    active_data[sched_key],
                    style,
                    color=col,
                    linewidth=2.0,
                    markersize=7,
                    label=label,
                )
        ax.set_xlabel("Number of Acquired 2D Chemical Slices (out of 3D Stack)")
        ax.set_ylabel("Full-Volume 3D mIoU")
        ax.set_title("Closed-Loop Active Sparse-Slice Acquisition")
        ax.set_xticks(budgets)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            p = out_dir / f"fig3_active_beamtime_curve.{ext}"
            fig.savefig(p, dpi=300)
            saved_paths.append(p)
        plt.close(fig)

    # 4. v1.0 Stress Tests: Iso-Intensity Texture & Split Conformal Coverage
    stress = results.get("v1_stress_tests", {})
    tex_data = stress.get("textured_isointensity", {})
    conf_rows = stress.get("conformal_coverage", [])
    if tex_data and conf_rows:
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
        methods = [
            ("jbu_miou", "Classical JBU", "#2980b9"),
            ("cnmf_miou", "CNMF", "#16a085"),
            ("guided_tv_miou", "Guided TV", "#27ae60"),
            ("xfuse_cross_attn_miou", "X-FUSE CrossAttn", "#c0392b"),
        ]
        labels = [m[1] for m in methods if m[0] in tex_data]
        vals = [tex_data[m[0]] for m in methods if m[0] in tex_data]
        cols = [m[2] for m in methods if m[0] in tex_data]
        axes[0].bar(np.arange(len(vals)), vals, color=cols, width=0.55)
        axes[0].set_xticks(np.arange(len(vals)))
        axes[0].set_xticklabels(labels, rotation=15, ha="right")
        axes[0].set_ylabel("Macro Foreground mIoU")
        axes[0].set_title("A. Iso-Intensity Textured Phase Disentanglement")
        axes[0].grid(True, axis="y", alpha=0.3)

        nom = [r["nominal_coverage"] for r in conf_rows]
        emp = [r["empirical_coverage"] for r in conf_rows]
        axes[1].plot([0.75, 0.98], [0.75, 0.98], "k--", label="Ideal Guarantee (y = x)")
        axes[1].plot(
            nom,
            emp,
            "o-",
            color="#8e44ad",
            linewidth=2.0,
            markersize=7,
            label="X-FUSE Split Conformal Sets",
        )
        axes[1].set_xlabel("Nominal Target Coverage (1 - α)")
        axes[1].set_ylabel("Empirical Test Coverage")
        axes[1].set_title("B. Finite-Sample Conformal Phase Coverage")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="lower right")

        fig.tight_layout()
        for ext in ("pdf", "png"):
            p = out_dir / f"fig4_field_baselines_and_conformal.{ext}"
            fig.savefig(p, dpi=300)
            saved_paths.append(p)
        plt.close(fig)

    return saved_paths


def main(json_path: str | Path, output_dir: str | Path) -> list[Path]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return generate_paper_figures(data, output_dir=output_dir)
