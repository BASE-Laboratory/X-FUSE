"""Tests for competing field baselines (`benchmarks/baselines.py`), v1.0 CLI
options (`xfuse fuse`), and the paper benchmark & figure suite (`--quick`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from benchmarks.baselines import (
    cnmf_pansharpen_baseline,
    coordinate_inr_baseline,
    gram_schmidt_pansharpen_baseline,
    guided_variational_sr_baseline,
)
from benchmarks.run_paper_benchmarks import run_paper_benchmarks
from x_fuse import compute_segmentation_metrics, generate_correlative_phantom_2d
from x_fuse.cli import main as cli_main


def test_all_four_competing_field_baselines_produce_valid_reports() -> None:
    ph = generate_correlative_phantom_2d(
        size=48, downsample_factor=4, phase_b_contrast=0.4, seed=11
    )
    for fn, kwargs in [
        (gram_schmidt_pansharpen_baseline, {}),
        (cnmf_pansharpen_baseline, {"n_iters": 15}),
        (guided_variational_sr_baseline, {"n_steps": 20}),
        (coordinate_inr_baseline, {"hidden_dim": 24, "epochs": 20}),
    ]:
        rep = fn(ph.hr_image, ph.coarse_guide, mode="multiclass", **kwargs)
        assert rep.probabilities.shape == (1, 3, 48, 48)
        assert rep.hard_mask.shape == (1, 48, 48)
        met = compute_segmentation_metrics(rep.hard_mask[0], ph.gt_labels)
        assert 0.15 < met["iou"] <= 1.0


def test_cli_v1_flags_fuse_with_cross_attn_bspline_and_conformal(
    tmp_path: Path,
) -> None:
    ph = generate_correlative_phantom_2d(size=48, downsample_factor=4, seed=9)
    hr_path = tmp_path / "hr.tif"
    g_path = tmp_path / "guide.tif"
    out_h5 = tmp_path / "out_v1.h5"

    tifffile.imwrite(hr_path, ph.hr_image.astype(np.float32))
    tifffile.imwrite(g_path, ph.coarse_guide.astype(np.float32))

    rc = cli_main(
        [
            "fuse",
            "--hr",
            str(hr_path),
            "--guide",
            str(g_path),
            "--output",
            str(out_h5),
            "--backbone",
            "timm:vit_tiny_patch16_224",
            "--no-pretrained",
            "--fusion-method",
            "spatial_mlp",
            "--loss",
            "compositional_kl",
            "--upsampler",
            "cross_attn",
            "--debias-positional",
            "--conformal-alpha",
            "0.10",
            "--align-guide",
            "bspline",
            "--stride",
            "8",
            "--epochs",
            "3",
        ]
    )
    assert rc == 0
    assert out_h5.exists()


def test_paper_benchmark_quick_generates_v1_results_and_fig4(tmp_path: Path) -> None:
    res = run_paper_benchmarks(
        output_dir=tmp_path / "paper_out",
        backbone="timm:vit_tiny_patch16_224",
        pretrained=False,
        quick=True,
        device="cpu",
    )
    assert "v1_stress_tests" in res
    assert "textured_isointensity" in res["v1_stress_tests"]
    assert "conformal_coverage" in res["v1_stress_tests"]
    fig4_pdf = (
        tmp_path
        / "paper_out"
        / "paper_figures"
        / "fig4_field_baselines_and_conformal.pdf"
    )
    assert fig4_pdf.exists()
