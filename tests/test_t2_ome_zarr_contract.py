"""Tests for post-v1.0.0 Layer-2 dataset hygiene, physical guide amplitude
calibration, and the 2D/3D Task T2 OME-Zarr v0.4 contract.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from basecoat import read_ome_zarr

from x_fuse import (
    OPEN_DATASETS,
    VolumetricFuser,
    XFuse,
    export_report_ome_zarr,
    extract_guide_physical_scales,
    fetch_open_dataset,
    generate_correlative_phantom_2d,
    generate_correlative_phantom_3d,
    list_open_datasets,
    load_uncertainty_report,
    save_uncertainty_report,
)
from x_fuse.cli import main as cli_main


def test_diad_dataset_has_no_hardcoded_scan_ids_and_emits_deprecation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify Layer-2 dataset registry has zero hardcoded DIAD scan IDs and warns."""
    assert OPEN_DATASETS["diad_zn13x_xrdct"].default_filenames is None

    with pytest.warns(DeprecationWarning, match="ECOSYSTEM.md"):
        reg = list_open_datasets()
    assert "diad_zn13x_xrdct" in reg

    cache_dir = tmp_path / "diad_zn13x_xrdct"
    cache_dir.mkdir(parents=True)
    dummy_nxs = cache_dir / "output_99999_zn.nxs"
    dummy_nxs.write_bytes(b"nexus-placeholder")

    def _offline(*args: Any, **kwargs: Any) -> Any:
        raise urllib.error.URLError("simulated offline")

    monkeypatch.setattr(urllib.request, "urlopen", _offline)
    with pytest.warns(DeprecationWarning, match="ECOSYSTEM.md"):
        cached = fetch_open_dataset("diad_zn13x_xrdct", dest_root=tmp_path)
    assert cached == [dummy_nxs.resolve()]


def test_physical_amplitude_calibration_and_2d_ome_zarr_roundtrip(
    tmp_path: Path,
) -> None:
    """Verify physical guide amplitudes and zero void leakage via .ome.zarr."""
    ph = generate_correlative_phantom_2d(
        size=64, downsample_factor=4, phase_b_contrast=0.4, seed=7
    )
    # Scale guides to realistic non-unit Bragg peak intensities (e.g. Zn=3.6, Na=1.4)
    raw_guides = np.stack(
        [ph.coarse_guide[0] * 3.6, ph.coarse_guide[1] * 1.4], axis=0
    ).astype(np.float32)

    scales = extract_guide_physical_scales(raw_guides, normalization="global_max")
    assert len(scales) == 2
    assert scales[0]["raw_mean"] == pytest.approx(float(raw_guides[0].mean()), rel=1e-5)
    assert scales[1]["raw_mean"] == pytest.approx(float(raw_guides[1].mean()), rel=1e-5)

    model = XFuse(
        backbone="timm:vit_tiny_patch16_224",
        stride=8,
        loss_fn="compositional_kl",
        pore_mask="auto_otsu_fringe_guarded",
        conformal_alpha=0.15,
        pretrained=False,
    )
    rep = model.fit_predict(ph.hr_image, raw_guides, epochs=3)
    phys_hr = rep.reconstruct_physical_maps()
    assert phys_hr.shape == (2, 64, 64)
    assert float(phys_hr[0].mean()) == pytest.approx(
        float(raw_guides[0].mean()), rel=1e-4
    )
    assert float(phys_hr[1].mean()) == pytest.approx(
        float(raw_guides[1].mean()), rel=1e-4
    )

    # Void pixels clamped by pore_mask must have strictly 0.0 physical leakage
    void_hr = rep.probabilities[0, 0].detach().cpu().numpy() >= 1.0 - 1e-5
    assert np.any(void_hr)
    assert float(np.max(np.abs(phys_hr[:, void_hr]))) == 0.0

    # Save via save_uncertainty_report(.ome.zarr) and reload via both basecoat & x_fuse
    store_hr = save_uncertainty_report(
        rep,
        tmp_path / "pred_hr.ome.zarr",
        channel_names=["background", "zn", "na"],
        pixel_size_um=0.54,
    )
    vset_hr = read_ome_zarr(store_hr, as_set=True)
    assert vset_hr.image.axis_names == ("c", "y", "x")
    assert vset_hr.image.shape == (2, 64, 64)
    assert vset_hr.image.spacing_in("um") == pytest.approx((0.54, 0.54))
    assert vset_hr.image.provenance["channel_names"] == ["zn", "na"]
    assert vset_hr.image.provenance["hr_diagnostics"]["hr_pore_leakage_rel"] == 0.0
    assert {"phase", "phase_lower", "phase_upper"}.issubset(vset_hr.labels.keys())

    loaded_dict = load_uncertainty_report(store_hr)
    assert np.allclose(loaded_dict["physical_maps"], phys_hr)
    assert "probabilities" in loaded_dict
    assert "hard_mask" in loaded_dict

    # Export coarse evaluation grid (16x16) and verify integrated flux conservation
    store_eval = export_report_ome_zarr(
        rep,
        tmp_path / "pred_eval16.ome.zarr",
        channel_names=["zn", "na"],
        pixel_size_um=0.54,
        target_grid_shape=(16, 16),
    )
    vset_eval = read_ome_zarr(store_eval, as_set=True)
    assert vset_eval.image.shape == (2, 16, 16)
    assert vset_eval.image.spacing_in("um") == pytest.approx((2.16, 2.16))
    assert float(vset_eval.image.data[0].mean()) == pytest.approx(
        float(raw_guides[0].mean()), rel=1e-4
    )


def test_cli_fuse_and_fuse3d_task_t2_ome_zarr_contract(
    tmp_path: Path, capsys: Any
) -> None:
    """Verify CLI `xfuse fuse` and `xfuse fuse-3d` produce Task-T2-ready .ome.zarr."""
    ph2d = generate_correlative_phantom_2d(size=32, downsample_factor=4, seed=11)
    hr_npy = tmp_path / "hr_2d.npy"
    zn_npy = tmp_path / "zn_50.npy"
    na_npy = tmp_path / "na_50.npy"
    np.save(hr_npy, ph2d.hr_image)
    np.save(zn_npy, ph2d.coarse_guide[0] * 2.5)
    np.save(na_npy, ph2d.coarse_guide[1] * 0.9)

    out_2d = tmp_path / "cli_2d.ome.zarr"
    rc_2d = cli_main(
        [
            "fuse",
            "--hr",
            str(hr_npy),
            "--guide",
            str(zn_npy),
            str(na_npy),
            "--output",
            str(out_2d),
            "--backbone",
            "timm:vit_tiny_patch16_224",
            "--no-pretrained",
            "--loss",
            "compositional_kl",
            "--pore-mask",
            "auto_otsu_fringe_guarded",
            "--conformal-alpha",
            "0.2",
            "--pixel-size-um",
            "0.54",
            "--channel-names",
            "zn",
            "na",
            "--eval-grid",
            "8x8",
            "--stride",
            "8",
            "--epochs",
            "2",
        ]
    )
    assert rc_2d == 0
    _ = json.loads(capsys.readouterr().out)
    v2 = read_ome_zarr(out_2d, as_set=True)
    assert v2.image.axis_names == ("c", "y", "x")
    assert v2.image.shape == (2, 8, 8)
    assert v2.image.provenance["channel_names"] == ["zn", "na"]

    # 3D sparse-slice CLI with --eval-slice-z and --eval-grid
    ph3d = generate_correlative_phantom_3d(
        shape=(8, 32, 32), downsample_factor=4, seed=15
    )
    vf = VolumetricFuser(
        XFuse(backbone="timm:vit_tiny_patch16_224", stride=8, pretrained=False),
        tta_shifts=None,
        tta_flips=False,
    )
    vf.fit_sparse_slices(
        ph3d.hr_volume,
        sparse_guides={
            1: ph3d.coarse_guide_3d[:, 0] * 3.0,
            5: ph3d.coarse_guide_3d[:, 1] * 3.0,
        },
        epochs=2,
        pore_mask="auto_otsu_fringe_guarded",
    )
    rep3d = vf.predict_volume(
        ph3d.hr_volume,
        mode="2.5d_slab",
        pore_mask="auto_otsu_fringe_guarded",
    )
    phys3d = rep3d.reconstruct_physical_maps()
    assert phys3d.shape[1:] == (8, 32, 32)
    expected_train_mean = 0.5 * (
        float((ph3d.coarse_guide_3d[0, 0] * 3.0).mean())
        + float((ph3d.coarse_guide_3d[0, 1] * 3.0).mean())
    )
    assert float(phys3d[0, [1, 5]].mean()) == pytest.approx(
        expected_train_mean, rel=1e-4
    )

    vol_npy = tmp_path / "vol_3d.npy"
    g0_npy = tmp_path / "g_z1.npy"
    g4_npy = tmp_path / "g_z4.npy"
    np.save(vol_npy, ph3d.hr_volume)
    np.save(g0_npy, ph3d.coarse_guide_3d[:, 0] * 3.0)
    np.save(g4_npy, ph3d.coarse_guide_3d[:, 1] * 3.0)

    out_3d_slice = tmp_path / "cli_3d_slice.ome.zarr"
    rc_3d = cli_main(
        [
            "fuse-3d",
            "--hr-volume",
            str(vol_npy),
            "--guide-slice",
            f"1:{g0_npy}",
            f"5:{g4_npy}",
            "--output",
            str(out_3d_slice),
            "--backbone",
            "timm:vit_tiny_patch16_224",
            "--no-pretrained",
            "--no-triplane",
            "--loss",
            "compositional_kl",
            "--pore-mask",
            "auto_otsu_fringe_guarded",
            "--conformal-alpha",
            "0.2",
            "--voxel-size-um",
            "0.54",
            "--channel-names",
            "zn",
            "na",
            "--eval-grid",
            "8x8",
            "--eval-slice-z",
            "3",
            "--stride",
            "8",
            "--epochs",
            "2",
        ]
    )
    assert rc_3d == 0
    _ = json.loads(capsys.readouterr().out)
    v3 = read_ome_zarr(out_3d_slice, as_set=True)
    assert v3.image.axis_names == ("c", "y", "x")
    assert v3.image.shape == (2, 8, 8)
    assert v3.image.provenance["eval_slice_z"] == 3
    assert v3.image.provenance["channel_names"] == ["zn", "na"]

    # Validate error handling on invalid --eval-grid and out-of-range --eval-slice-z
    with pytest.raises(ValueError, match="--eval-grid"):
        cli_main(
            [
                "fuse",
                "--hr",
                str(hr_npy),
                "--guide",
                str(zn_npy),
                "--output",
                str(tmp_path / "bad.ome.zarr"),
                "--eval-grid",
                "badgrid",
            ]
        )
    with pytest.raises(ValueError, match="--eval-slice-z"):
        cli_main(
            [
                "fuse-3d",
                "--hr-volume",
                str(vol_npy),
                "--guide-slice",
                f"1:{g0_npy}",
                "--output",
                str(tmp_path / "bad3d.ome.zarr"),
                "--eval-slice-z",
                "99",
            ]
        )
