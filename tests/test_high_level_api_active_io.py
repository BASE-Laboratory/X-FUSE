"""Unit tests for the 3-line `XFuse.fit_predict` API on arbitrary odd shapes,
active experiment steering (`x_fuse.active`), scientific I/O (`x_fuse.io`),
the `xfuse fuse` CLI subcommand, and paper benchmark adapters & figure plotters.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from benchmarks.dataset_adapters import (
    load_bundled_morgis_samples,
    load_bundled_zn13x_pairs,
    load_zenodo_correlative_sample,
)
from benchmarks.plot_paper_figures import generate_paper_figures
from x_fuse import (
    VolumetricFuser,
    XFuse,
    generate_correlative_phantom_2d,
    generate_correlative_phantom_3d,
    load_microscopy_file,
    load_uncertainty_report,
    save_uncertainty_report,
    suggest_2d_rois,
    suggest_next_slices,
)
from x_fuse.cli import main as cli_main


def test_three_line_fit_predict_crops_non_divisible_odd_shapes():
    ph = generate_correlative_phantom_2d(size=96, downsample_factor=4, seed=12)
    hr_odd = ph.hr_image[:73, :89]
    guide_odd = ph.coarse_guide[:, :18, :22]

    model = XFuse(
        backbone="timm:vit_tiny_patch16_224",
        stride=8,
        pretrained=False,
        layers="multiscale",
    )
    rep = model.fit_predict(hr_odd, guide_odd, epochs=3, tta_shifts=(1,))
    assert rep.probabilities.shape == (1, 3, 73, 89)
    assert rep.hard_mask.shape == (1, 73, 89)
    assert rep.combined_uncertainty.shape == (1, 73, 89)
    assert rep.unresolved_mask.shape == (1, 73, 89)


def test_active_slice_and_2d_roi_recommendations():
    ph3d = generate_correlative_phantom_3d(
        shape=(16, 32, 32), downsample_factor=4, seed=21
    )
    vf = VolumetricFuser(
        XFuse(backbone="timm:vit_tiny_patch16_224", stride=8, pretrained=False),
        tta_shifts=(1,),
        tta_flips=False,
    )
    vf.fit_sparse_slices(
        ph3d.hr_volume, sparse_guides={0: ph3d.coarse_guide_3d[:, 0]}, epochs=2
    )
    vol_rep = vf.predict_volume(ph3d.hr_volume, mode="2.5d_slab")

    # Inject synthetic epistemic spike at z=11 so it must rank #1
    vol_rep.triplane_variance[11] += 5.0
    rec = suggest_next_slices(vol_rep, acquired_indices=[0, 15], k=3, min_spacing=3)
    assert len(rec.recommended_indices) == 3
    assert rec.recommended_indices[0] == 11
    assert 0 not in rec.recommended_indices and 15 not in rec.recommended_indices
    for i, a in enumerate(rec.recommended_indices):
        for b in rec.recommended_indices[i + 1 :]:
            assert abs(a - b) >= 3

    with pytest.raises(ValueError, match="k must be >= 1"):
        suggest_next_slices(vol_rep, k=0)

    # 2D ROI test with injected residual hot-spot in bottom-right corner
    ph2d = generate_correlative_phantom_2d(size=64, downsample_factor=4, seed=9)
    rep2d = XFuse(
        backbone="timm:vit_tiny_patch16_224", stride=8, pretrained=False
    ).fit_predict(ph2d.hr_image, ph2d.coarse_guide, epochs=2)
    rep2d.physical_residual[0, 44:60, 44:60] = 10.0

    rois = suggest_2d_rois(rep2d, roi_size=(16, 16), k=2, residual_weight=0.8)
    assert len(rois) == 2
    top_roi = rois[0]
    assert top_roi.y0 >= 40 and top_roi.x0 >= 40
    # Verify non-overlapping NMS suppression between ROI 0 and ROI 1
    overlap_y = max(0, min(rois[0].y1, rois[1].y1) - max(rois[0].y0, rois[1].y0))
    overlap_x = max(0, min(rois[0].x1, rois[1].x1) - max(rois[0].x0, rois[1].x0))
    assert overlap_y == 0 or overlap_x == 0

    with pytest.raises(ValueError, match="k must be >= 1"):
        suggest_2d_rois(rep2d, k=0)


@pytest.mark.parametrize("ext", [".ome.tif", ".tif", ".h5", ".zarr"])
def test_save_and_load_uncertainty_report_2d_and_3d(tmp_path: Path, ext: str):
    if ext == ".zarr":
        pytest.importorskip("zarr")

    ph2d = generate_correlative_phantom_2d(size=48, downsample_factor=4, seed=4)
    rep2d = XFuse(
        backbone="timm:vit_tiny_patch16_224", stride=8, pretrained=False
    ).fit_predict(ph2d.hr_image, ph2d.coarse_guide, epochs=2)

    out_2d = save_uncertainty_report(
        rep2d,
        tmp_path / f"report_2d{ext}",
        channel_names=["bg", "pA", "pB"],
        pixel_size_um=0.12,
    )
    loaded_2d = load_uncertainty_report(out_2d)
    assert loaded_2d["metadata"]["report_type"] == "UncertaintyReport"
    assert loaded_2d["metadata"]["channel_names"] == ["bg", "pA", "pB"]
    if ext in (".h5", ".zarr"):
        assert np.allclose(
            loaded_2d["probabilities"], rep2d.probabilities[0].cpu().numpy(), atol=1e-5
        )
    else:
        assert loaded_2d["stack"].shape == (10, 48, 48)
        assert "confidence_margin" in loaded_2d

    # 3D VolumetricUncertaintyReport round-trip
    ph3d = generate_correlative_phantom_3d(
        shape=(6, 32, 32), downsample_factor=4, seed=4
    )
    vf = VolumetricFuser(
        XFuse(backbone="timm:vit_tiny_patch16_224", stride=8, pretrained=False),
        tta_shifts=(1,),
        tta_flips=False,
    )
    vf.fit_sparse_slices(
        ph3d.hr_volume, sparse_guides={2: ph3d.coarse_guide_3d[:, 0]}, epochs=2
    )
    rep3d = vf.predict_volume(ph3d.hr_volume, mode="2.5d_slab")
    out_3d = save_uncertainty_report(
        rep3d,
        tmp_path / f"report_3d{ext}",
        channel_names=["bg", "A", "B"],
        pixel_size_um=(0.5, 0.25, 0.25),
    )
    loaded_3d = load_uncertainty_report(out_3d)
    assert loaded_3d["metadata"]["report_type"] == "VolumetricUncertaintyReport"

    with pytest.raises(ValueError, match="Unsupported"):
        save_uncertainty_report(rep2d, tmp_path / "bad.xyz")
    with pytest.raises(ValueError, match="Unsupported"):
        load_uncertainty_report(tmp_path / "bad.xyz")


def test_load_microscopy_file_and_xfuse_fuse_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    with pytest.raises(FileNotFoundError):
        load_microscopy_file(tmp_path / "missing.dm4")

    npy_path = tmp_path / "arr.npy"
    np.save(npy_path, np.ones((16, 16), dtype=np.float32) * 3.0)
    arr_npy, meta_npy = load_microscopy_file(npy_path)
    assert arr_npy.shape == (16, 16) and meta_npy["format"] == "npy"

    # Mock RosettaSciIO vendor file reader (.dm4)
    fake_dm4 = tmp_path / "survey.dm4"
    fake_dm4.write_bytes(b"mock-dm4")
    fake_mod = types.SimpleNamespace(
        file_reader=lambda p: [
            {
                "data": np.arange(24, dtype=np.float32).reshape(4, 6),
                "axes": [{"name": "y"}, {"name": "x"}],
                "metadata": {"General": {"title": "HAADF"}},
            }
        ]
    )
    with patch("importlib.import_module", return_value=fake_mod):
        dm_arr, dm_meta = load_microscopy_file(fake_dm4)
    assert dm_arr.shape == (4, 6)
    assert dm_meta["format"] == "dm4"

    with pytest.raises(ValueError, match="Unsupported microscopy extension"):
        bad_f = tmp_path / "notes.unknown"
        bad_f.write_text("hi", encoding="utf-8")
        load_microscopy_file(bad_f)

    # End-to-end `xfuse fuse` CLI test
    ph = generate_correlative_phantom_2d(size=48, downsample_factor=4, seed=31)
    hr_tif = tmp_path / "hr.tif"
    g0_tif = tmp_path / "Ca.tif"
    g1_tif = tmp_path / "Si.tif"
    Image.fromarray(ph.hr_image).save(hr_tif)
    Image.fromarray(ph.coarse_guide[0]).save(g0_tif)
    Image.fromarray(ph.coarse_guide[1]).save(g1_tif)

    out_h5 = tmp_path / "cli_fused.h5"
    rc = cli_main(
        [
            "fuse",
            "--hr",
            str(hr_tif),
            "--guide",
            str(g0_tif),
            str(g1_tif),
            "--output",
            str(out_h5),
            "--backbone",
            "timm:vit_tiny_patch16_224",
            "--layers",
            "multiscale",
            "--align-guide",
            "translation",
            "--learn-psf",
            "--stride",
            "8",
            "--epochs",
            "3",
            "--no-pretrained",
            "--pixel-size-um",
            "0.1",
        ]
    )
    assert rc == 0
    cli_out = json.loads(capsys.readouterr().out)
    assert Path(cli_out["saved_report"]).exists()
    reloaded = load_uncertainty_report(out_h5)
    assert reloaded["metadata"]["channel_names"] == ["background", "Ca", "Si"]


def test_dataset_adapters_and_paper_figure_generator(tmp_path: Path):
    try:
        zn_samples = load_bundled_zn13x_pairs()
        mg_samples = load_bundled_morgis_samples()
    except FileNotFoundError:
        zn_samples = []
        mg_samples = []

    if zn_samples:
        assert len(zn_samples) == 2
        assert zn_samples[0].hr_image.ndim == 2
        assert zn_samples[0].guide_map.ndim == 3
    if mg_samples:
        assert len(mg_samples) >= 2
        assert mg_samples[0].is_full_res_ground_truth is True

    # Mock cached Zenodo folder for `ti_rcca_stem_eds`
    ds_dir = tmp_path / "ti_rcca_stem_eds"
    ds_dir.mkdir()
    Image.fromarray(np.random.default_rng(0).random((32, 32), dtype=np.float32)).save(
        ds_dir / "HAADF_survey.tif"
    )
    Image.fromarray(np.random.default_rng(1).random((32, 32), dtype=np.float32)).save(
        ds_dir / "EDS_Ti.tif"
    )
    Image.fromarray(np.random.default_rng(2).random((32, 32), dtype=np.float32)).save(
        ds_dir / "EDS_Mo.tif"
    )
    sample = load_zenodo_correlative_sample(
        "ti_rcca_stem_eds", dest_root=tmp_path, auto_download=False
    )
    assert sample.channel_names == ["Ti", "Mo"]
    assert sample.guide_map.shape == (2, 32, 32)

    # Render all 3 multi-panel PDF + PNG paper figures
    mock_results = {
        "ablation_resolution_sweep": [
            {
                "downsample_factor": 4,
                "bicubic_miou": 0.62,
                "jbu_miou": 0.74,
                "vit_kmeans_miou": 0.58,
                "xfuse_last_miou": 0.81,
                "xfuse_multiscale_miou": 0.86,
                "bicubic_bf1": 0.45,
                "jbu_bf1": 0.68,
                "xfuse_last_bf1": 0.77,
                "xfuse_multiscale_bf1": 0.84,
            },
            {
                "downsample_factor": 8,
                "bicubic_miou": 0.49,
                "jbu_miou": 0.66,
                "vit_kmeans_miou": 0.58,
                "xfuse_last_miou": 0.75,
                "xfuse_multiscale_miou": 0.82,
                "bicubic_bf1": 0.31,
                "jbu_bf1": 0.59,
                "xfuse_last_bf1": 0.70,
                "xfuse_multiscale_bf1": 0.79,
            },
        ],
        "drift_and_psf_ablation": [
            {
                "injected_shift_hr_px": 0.0,
                "unaligned_miou": 0.84,
                "aligned_miou": 0.84,
                "estimated_shift_hr_px": 0.05,
            },
            {
                "injected_shift_hr_px": 3.0,
                "unaligned_miou": 0.61,
                "aligned_miou": 0.82,
                "estimated_shift_hr_px": 2.85,
            },
        ],
        "active_slice_tomography": {
            "slice_budgets": [2, 4, 6],
            "random_miou": [0.58, 0.67, 0.74],
            "uniform_miou": [0.61, 0.71, 0.77],
            "active_xfuse_miou": [0.61, 0.78, 0.84],
        },
    }
    figs = generate_paper_figures(mock_results, output_dir=tmp_path / "figs")
    assert len(figs) == 6
    assert all(p.exists() and p.stat().st_size > 0 for p in figs)
