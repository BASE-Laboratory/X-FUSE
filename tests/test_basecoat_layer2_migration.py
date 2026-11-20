"""Regression tests for the Layer 2 basecoat migration in X-FUSE."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from basecoat import (
    Volume,
    check_layer_imports,
    open_volume,
    read_ome_zarr,
    write_ome_zarr,
)

from x_fuse import (
    VolumeSliceAccessor,
    VolumetricFuser,
    XFuse,
    export_for_openimpala,
    export_volume_ome_zarr,
    generate_correlative_phantom_3d,
    load_microscopy_file,
    load_xct,
    load_xct_subvolume,
)
from x_fuse.cli import main as cli_main

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_layer2_import_boundary_and_mit_license_preserved() -> None:
    """Verify [tool.basecoat.layers] passes and MIT LICENSE stays intact."""
    violations = check_layer_imports(REPO_ROOT, raise_on_violation=True)
    assert violations == []
    lic_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert lic_text.startswith("MIT License")
    assert "Calum Green" in lic_text


def test_export_volume_ome_zarr_with_conformal_envelopes_and_accessor(
    tmp_path: Path,
) -> None:
    """Verify 3D report exports phase + conformal envelopes to OME-Zarr."""
    ph3d = generate_correlative_phantom_3d(
        shape=(8, 32, 32), downsample_factor=4, seed=12
    )
    fuser = XFuse(
        backbone="timm:vit_tiny_patch16_224",
        stride=8,
        conformal_alpha=0.15,
        pretrained=False,
    )
    vf = VolumetricFuser(fuser, tta_shifts=None, tta_flips=False)
    vf.fit_sparse_slices(
        ph3d.hr_volume,
        sparse_guides={2: ph3d.coarse_guide_3d[:, 0], 5: ph3d.coarse_guide_3d[:, 1]},
        epochs=2,
    )
    rep3d = vf.predict_volume(
        ph3d.hr_volume,
        mode="2.5d_slab",
        sparse_guides={2: ph3d.coarse_guide_3d[:, 0], 5: ph3d.coarse_guide_3d[:, 1]},
    )

    store = export_volume_ome_zarr(
        rep3d,
        tmp_path / "zeolite.ome.zarr",
        image_volume=ph3d.hr_volume,
        voxel_size_um=0.8125,
        origin_um=(5.0, 10.0, 15.0),
        space="zn13x_0p8125um",
        active_phase=1,
        export_conformal_bounds=True,
        multiscale_levels=2,
    )
    assert store.is_dir()

    vset = read_ome_zarr(store, level=0, as_set=True)
    assert np.allclose(vset.image.data, ph3d.hr_volume)
    assert vset.image.spacing_in("um") == pytest.approx((0.8125, 0.8125, 0.8125))
    assert vset.image.origin_in("um") == pytest.approx((5.0, 10.0, 15.0))
    assert vset.image.space == "zn13x_0p8125um"
    assert {"phase", "phase_lower", "phase_upper"}.issubset(vset.labels.keys())
    assert {
        "combined_uncertainty",
        "aleatoric_entropy",
        "epistemic_variance",
        "physical_residual",
        "probabilities",
    }.issubset(vset.uncertainties.keys())
    assert vset.uncertainties["combined_uncertainty"].shape == (8, 32, 32)
    assert vset.uncertainties["probabilities"].shape[1:] == (8, 32, 32)
    # Conformal lower envelope for phase 1 is a subset of hard phase 1 / upper envelope
    in_lo = vset.labels["phase_lower"].data == 1
    in_up = vset.labels["phase_upper"].data == 1
    assert np.all(in_lo <= in_up)

    # Coarse level mode-pools discrete labels and mean-pools uncertainties
    vset_l1 = read_ome_zarr(store, level=1, as_set=True)
    assert vset_l1.image.shape == (4, 16, 16)
    assert set(np.unique(vset_l1.labels["phase_upper"].data)).issubset({0, 1})
    assert vset_l1.uncertainties["combined_uncertainty"].shape == (4, 16, 16)
    assert vset_l1.uncertainties["probabilities"].shape[1:] == (4, 16, 16)

    # VolumeSliceAccessor reads .ome.zarr paths, LazyVolume, and Volume natively
    acc_store = VolumeSliceAccessor(store)
    assert acc_store.is_lazy
    assert acc_store.shape == (8, 32, 32)
    assert np.allclose(acc_store.get_plane(0, 3), ph3d.hr_volume[3])
    assert np.allclose(acc_store.get_plane(1, 10), ph3d.hr_volume[:, 10, :])
    assert np.allclose(acc_store.get_plane(2, 14), ph3d.hr_volume[:, :, 14])

    with open_volume(store, level=0) as lv:
        acc_lv = VolumeSliceAccessor(lv)
        assert acc_lv.is_lazy
        assert np.allclose(acc_lv.get_plane(0, 1), ph3d.hr_volume[1])

    acc_vol = VolumeSliceAccessor(vset.image)
    assert not acc_vol.is_lazy
    assert np.allclose(acc_vol.get_plane(0, 2), ph3d.hr_volume[2])

    arr_m, meta_m = load_microscopy_file(store)
    assert meta_m["format"] == "ome.zarr"
    assert np.allclose(arr_m, ph3d.hr_volume)


def test_savu_h5_loader_and_cli_fuse3d_ome_zarr(tmp_path: Path, capsys: Any) -> None:
    """Verify real Savu HDF5 slicing via basecoat and CLI fuse-3d --export-ome-zarr."""
    rng = np.random.default_rng(21)
    # Savu disk layout (X=16, Z=6, Y=16) -> canonical (Z=6, Y=16, X=16)
    raw_xzy = rng.uniform(0.0, 1.0, size=(16, 6, 16)).astype(np.float32)
    h5_p = tmp_path / "savu_scan.h5"
    with h5py.File(h5_p, "w") as hf:
        hf.create_dataset("4-TomopyRecon-tomo/data", data=raw_xzy)

    full_zyx = load_xct(h5_p, recon="Tomopy")
    assert full_zyx.shape == (6, 16, 16)
    assert np.allclose(full_zyx, np.transpose(raw_xzy, (1, 2, 0)))

    sub_zyx = load_xct_subvolume(
        h5_p,
        recon="Tomopy",
        zyx_slices=(slice(1, 5), slice(2, 12), slice(3, 13)),
    )
    assert sub_zyx.shape == (4, 10, 10)
    assert np.allclose(sub_zyx, full_zyx[1:5, 2:12, 3:13])

    # CLI fuse-3d consuming an .ome.zarr HR volume and writing an .ome.zarr store
    ph3d = generate_correlative_phantom_3d(
        shape=(6, 32, 32), downsample_factor=4, seed=9
    )
    hr_store = write_ome_zarr(
        Volume.from_array(ph3d.hr_volume, voxel_size_um=1.625),
        tmp_path / "hr_input.ome.zarr",
    )
    guide_npy = tmp_path / "guide_z2.npy"
    np.save(guide_npy, ph3d.coarse_guide_3d[:, 0])

    out_store = tmp_path / "fused_out.ome.zarr"
    rc = cli_main(
        [
            "fuse-3d",
            "--hr-volume",
            str(hr_store),
            "--guide-slice",
            f"2:{guide_npy}",
            "--output",
            str(out_store),
            "--backbone",
            "timm:vit_tiny_patch16_224",
            "--no-pretrained",
            "--no-triplane",
            "--stride",
            "8",
            "--epochs",
            "2",
            "--conformal-alpha",
            "0.2",
            "--voxel-size-um",
            "1.625",
            "--openimpala-phase",
            "1",
        ]
    )
    assert rc == 0
    captured = json.loads(capsys.readouterr().out)
    assert Path(captured["saved_report"]) == out_store
    loaded_set = read_ome_zarr(out_store, as_set=True)
    assert loaded_set.image.spacing_in("um") == pytest.approx((1.625, 1.625, 1.625))
    assert "phase" in loaded_set.labels

    # Deprecated export_for_openimpala emits DeprecationWarning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        export_for_openimpala(np.zeros((4, 4, 4), dtype=np.uint8), tmp_path / "imp")
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
