"""Regression tests for the v1.0.0 adversarial red-team audit across physics,
uncertainty, fusion, volumetric, loaders, io, active steering, and CLI.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from x_fuse import (
    VolumetricFuser,
    VolumetricUncertaintyReport,
    XFuse,
    export_for_openimpala,
    generate_correlative_phantom_2d,
    generate_correlative_phantom_3d,
    guide_to_tensor,
    load_microscopy_file,
    load_uncertainty_report,
    save_uncertainty_report,
    segment_pores_otsu,
    suggest_2d_rois,
    suggest_next_slices,
)
from x_fuse.cli import main as cli_main
from x_fuse.loaders import (
    aggregate_peak_maps,
    detrend_radial_cupping,
    load_xct_subvolume,
    load_xrdct_peak_maps,
    remove_polar_rings_conservative,
)

FAST_BACKBONE = "timm:vit_tiny_patch16_224"


def test_spectral_unmix_with_forward_psf_bce_no_duplicate_raw_background() -> None:
    rng = np.random.default_rng(0)
    hr = rng.random((48, 48), dtype=np.float32)
    spec_cube = rng.random((5, 12, 12), dtype=np.float32)
    model = XFuse(
        backbone=FAST_BACKBONE,
        pretrained=False,
        stride=16,
        loss_fn="forward_psf_bce",
        spectral_channels=5,
        guide_channels=2,
        lr=1e-2,
    )
    rep = model.fit_predict(
        hr, spec_cube, epochs=2, spectral_unmix=True, n_components=2, tta_shifts=()
    )
    assert rep.probabilities.shape == (1, 3, 48, 48)
    assert torch.isfinite(rep.probabilities).all()


def test_compositional_kl_spectral_dict_plus_pore_mask() -> None:
    rng = np.random.default_rng(1)
    hr = np.ones((48, 48), dtype=np.float32) * 0.8
    hr[:16, :16] = 0.02  # clear dark void corner
    spec_cube = rng.random((4, 12, 12), dtype=np.float32) + 0.1
    model = XFuse(
        backbone=FAST_BACKBONE,
        pretrained=False,
        stride=16,
        loss_fn="compositional_kl",
        spectral_channels=4,
        guide_channels=2,
        pore_mask="auto_otsu_fringe_guarded",
        lr=1e-2,
    )
    rep = model.fit_predict(
        hr, spec_cube, epochs=2, spectral_unmix=True, n_components=2, tta_shifts=()
    )
    assert rep.probabilities.shape == (1, 3, 48, 48)
    # Void corner should have ~1.0 background probability and 0.0 solid phases
    assert float(rep.probabilities[0, 1:, :12, :12].max().item()) == 0.0


def test_volumetric_spectral_unmix_guide_residual_and_xz_interpolation() -> None:
    rng = np.random.default_rng(2)
    vol = rng.random((6, 32, 32), dtype=np.float32)
    spec0 = rng.random((4, 8, 8), dtype=np.float32) + 0.1
    spec5 = rng.random((4, 8, 8), dtype=np.float32) + 0.2

    fuser = XFuse(
        backbone=FAST_BACKBONE,
        pretrained=False,
        stride=16,
        fusion_method="spatial_mlp",
        loss_fn="forward_psf_bce",
        spectral_channels=4,
        guide_channels=2,
        guide_residual=True,
        conformal_alpha=0.15,
        lr=1e-2,
    )
    vf = VolumetricFuser(fuser, tta_shifts=None, tta_flips=False)
    vf.fit_sparse_slices(
        vol, {0: spec0, 5: spec5}, epochs=2, triplane_consistency_weight=0.1
    )
    assert fuser.spatial_mlp is not None
    assert fuser.spatial_mlp.physics.spectral_dict is not None
    trained_basis = (
        next(fuser.spatial_mlp.physics.spectral_dict.parameters()).detach().clone()
    )

    rep = vf.predict_volume(
        vol,
        mode="triplane",
        sparse_guides={0: spec0, 5: spec5},
        pred_mode="multiclass",
    )
    # _predict_single_plane_stack must NOT reset spectral_dict weights!
    after_basis = next(fuser.spatial_mlp.physics.spectral_dict.parameters()).detach()
    assert torch.allclose(trained_basis, after_basis)
    assert rep.probabilities.shape == (3, 6, 32, 32)
    assert rep.prediction_sets is not None

    # Verify slice_2d alias and invalid mode rejection
    rep_2d = vf.predict_volume(vol, mode="slice_2d", pred_mode="multiclass")
    assert rep_2d.diagnostics["volumetric_mode"] == "2d"
    with pytest.raises(ValueError, match="Unsupported volumetric mode"):
        vf.predict_volume(vol, mode="invalid_mode")  # type: ignore[arg-type]


def test_3d_non_ome_tiff_roundtrip_2d_margin_and_hdf5_microscopy_loader(
    tmp_path: Path,
) -> None:
    ph2 = generate_correlative_phantom_2d(size=32, downsample_factor=4, seed=3)
    m2 = XFuse(backbone=FAST_BACKBONE, pretrained=False, stride=16)
    rep2 = m2.fit_predict(ph2.hr_image, ph2.coarse_guide, epochs=1, tta_shifts=())
    tif2 = save_uncertainty_report(rep2, tmp_path / "r2d.ome.tif")
    loaded2 = load_uncertainty_report(tif2)
    assert "confidence_margin" in loaded2
    assert loaded2["confidence_margin"].shape == (32, 32)

    ph3 = generate_correlative_phantom_3d(
        shape=(4, 32, 32), downsample_factor=4, seed=4
    )
    vf = VolumetricFuser(
        XFuse(backbone=FAST_BACKBONE, pretrained=False, stride=16),
        tta_shifts=None,
        tta_flips=False,
    )
    vf.fit_sparse_slices(ph3.hr_volume, {0: ph3.coarse_guide_3d[:, 0]}, epochs=1)
    rep3 = vf.predict_volume(ph3.hr_volume, mode="2d", pred_mode="multiclass")
    tif3 = save_uncertainty_report(rep3, tmp_path / "r3d_plain.tif")
    loaded3 = load_uncertainty_report(tif3)
    assert loaded3["probabilities"].shape == (3, 4, 32, 32)
    assert loaded3["hard_mask"].shape == (4, 32, 32)

    # load_microscopy_file on .h5 and .npz
    h5_p = tmp_path / "micro.h5"
    with h5py.File(h5_p, "w") as hf:
        hf.create_dataset("entry/data", data=np.ones((16, 16), dtype=np.float32) * 3.5)
    arr_h5, meta_h5 = load_microscopy_file(h5_p)
    assert arr_h5.shape == (16, 16) and np.allclose(arr_h5, 3.5)
    assert meta_h5["dataset"] == "entry/data"

    npz_p = tmp_path / "micro.npz"
    np.savez(npz_p, img=np.ones((12, 12), dtype=np.float32) * 2.0)
    arr_npz, _ = load_microscopy_file(npz_p)
    assert arr_npz.shape == (12, 12) and np.allclose(arr_npz, 2.0)


def test_cli_fuse_3d_openimpala_no_stem_collision_and_conformal_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    vol = np.random.default_rng(5).random((4, 32, 32), dtype=np.float32)
    g0 = np.random.default_rng(6).random((1, 8, 8), dtype=np.float32)
    vol_p = tmp_path / "vol.npy"
    g0_p = tmp_path / "g0.npy"
    np.save(vol_p, vol)
    np.save(g0_p, g0)

    out_tif = tmp_path / "out_dir" / "vol_seg.tif"
    rc = cli_main(
        [
            "fuse-3d",
            "--hr-volume",
            str(vol_p),
            "--guide-slice",
            f"0:{g0_p}",
            "--output",
            str(out_tif),
            "--backbone",
            FAST_BACKBONE,
            "--no-pretrained",
            "--stride",
            "16",
            "--epochs",
            "1",
            "--loss",
            "forward_psf_bce",
            "--axial-psf-sigma",
            "0",
            "--no-triplane",
            "--conformal-alpha",
            "0.15",
            "--export-openimpala-dir",
            str(out_tif.parent),
            "--openimpala-phase",
            "1",
            "--openimpala-direction",
            "z",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["diagnostics"]["depth_psf_enabled"] is False
    assert payload["has_conformal_sets"] is True
    # Report .tif and OpenImpala .tif must be distinct files and reloadable!
    assert Path(payload["openimpala_tiff"]).name == "vol_seg_openimpala.tif"
    reloaded = load_uncertainty_report(out_tif)
    assert reloaded["probabilities"].shape == (1, 4, 32, 32)
    # Single-channel multilabel conformal envelopes for phase 1 must exist!
    assert (out_tif.parent / "vol_seg_openimpala_phase1_lower.tif").exists()
    assert (out_tif.parent / "vol_seg_openimpala_phase1_upper.tif").exists()


def test_openimpala_abstention_maps_to_255_not_pore_zero(tmp_path: Path) -> None:
    mask = np.zeros((4, 8, 8), dtype=np.int64)
    mask[:, :4, :] = -1  # abstained voxels must NOT become 0 (pore)!
    mask[:, 4:, :] = 1
    tif_p, _ = export_for_openimpala(mask, tmp_path, name="abstain_check")
    loaded, _ = load_microscopy_file(tif_p)
    assert np.all(loaded[:, :4, :] == 255)
    assert np.all(loaded[:, 4:, :] == 1)


def test_pore_mask_conformal_and_abstain_threshold_invariants() -> None:
    hr = np.ones((48, 48), dtype=np.float32) * 0.85
    hr[:20, :20] = 0.02  # void region
    g = np.ones((2, 12, 12), dtype=np.float32) * 0.45
    g[0, :6, :6] = 0.8  # deliberate coarse beam bleed into void corner!

    model = XFuse(
        backbone=FAST_BACKBONE,
        pretrained=False,
        stride=16,
        loss_fn="compositional_kl",
        conformal_alpha=0.10,
        pore_mask="auto_otsu_fringe_guarded",
        lr=1e-2,
    )
    model.fit(hr, g, epochs=2, tta_shifts=())
    rep = model.predict(hr, g, mode="multiclass", abstain_threshold=0.01)
    # Deterministic pores must stay class 0 (not -1 abstained) and have 0 uncertainty
    assert int(rep.hard_mask[0, :16, :16].max().item()) == 0
    assert int(rep.hard_mask[0, :16, :16].min().item()) == 0
    assert float(rep.combined_uncertainty[0, :16, :16].max().item()) == 0.0
    assert rep.prediction_sets is not None
    # Conformal sets must NOT inject solid phases (1, 2) into structural pores!
    assert not bool(rep.prediction_sets[0, 1:, :16, :16].any().item())

    rep_ug = model.predict(
        hr, coarse_guide=None, mode="multiclass", abstain_threshold=0.01
    )
    assert int(rep_ug.hard_mask[0, :16, :16].min().item()) == 0
    assert rep_ug.prediction_sets is not None
    assert not bool(rep_ug.prediction_sets[0, 1:, :16, :16].any().item())


def test_segment_pores_otsu_z3_nonfinite_and_global_3d_otsu() -> None:
    vol3 = np.ones((3, 32, 32), dtype=np.float32) * 0.9
    vol3[:, :12, :12] = 0.05
    vol3[0, 0, 0] = np.nan
    vol3[1, 0, 1] = np.inf
    vol3[2, 0, 2] = -np.inf
    pm3 = segment_pores_otsu(vol3, tie_homogeneity_alpha_vox2=0.6)
    assert pm3.shape == (3, 32, 32)
    assert bool(pm3[1, 5, 5]) is True
    assert bool(pm3[1, 25, 25]) is False

    # Global 3D Otsu in VolumetricFuser must NOT split a pure-air end slice 50/50
    vol_end = np.ones((6, 32, 32), dtype=np.float32) * 0.9
    vol_end[0] = 0.02  # pure air end slice!
    vol_end[1:, :10, :10] = 0.02
    vf = VolumetricFuser(
        XFuse(
            backbone=FAST_BACKBONE,
            pretrained=False,
            stride=16,
            pore_mask="auto_otsu_fringe_guarded",
        ),
        tta_shifts=None,
        tta_flips=False,
    )
    vf.fit_sparse_slices(
        vol_end,
        {2: np.ones((2, 8, 8), dtype=np.float32) * 0.4},
        epochs=1,
    )
    rep_end = vf.predict_volume(vol_end, mode="2d", pred_mode="multiclass")
    # Pure-air slice z=0 must be ~100% background (not 50% split!)
    assert float((rep_end.hard_mask[0] == 0).float().mean().item()) > 0.98


def test_loaders_flip_x_subvolume_nxdata_peak_modes_cupping_and_rings(
    tmp_path: Path,
) -> None:
    # 1. Savu (X, Z, Y) out-of-core subvolume with flip_x=True and negative step
    raw_xzy = np.arange(10 * 6 * 8, dtype=np.float32).reshape(10, 6, 8)
    h5_p = tmp_path / "savu.HDF5"
    with h5py.File(h5_p, "w") as hf:
        grp = hf.create_group("4-AstraReconGpu-tomo")
        grp.create_dataset("data", data=raw_xzy)

    full_flipped = load_xct_subvolume(h5_p, recon="Astra", flip_x=True)
    sl = (slice(1, 5), slice(2, 7), slice(1, 6))
    sub_flipped = load_xct_subvolume(
        h5_p, recon="Astra", zyx_slices=sl, stride=(1, 2, 2), flip_x=True
    )
    expected = full_flipped[sl][::1, ::2, ::2]
    assert np.array_equal(sub_flipped, expected)

    sub_neg = load_xct_subvolume(
        h5_p, recon="Astra", zyx_slices=(slice(4, 1, -1), slice(None), slice(None))
    )
    full_unflipped = load_xct_subvolume(h5_p, recon="Astra", flip_x=False)
    assert np.array_equal(sub_neg, full_unflipped[4:1:-1])

    # 2. NXdata group peak map loading + MultimodalBenchmark aggregation modes
    nxs_p = tmp_path / "peaks.nxs"
    with h5py.File(nxs_p, "w") as hf:
        ent = hf.create_group("entry")
        nx_g = ent.create_group("peak at q~1.651")
        nx_g.attrs["signal"] = "data"
        nx_g.create_dataset("data", data=np.ones((8, 8), dtype=np.float32) * 2.0)
        ent.create_dataset(
            "peak at q~2.300", data=np.ones((8, 8), dtype=np.float32) * 4.0
        )
    pks = load_xrdct_peak_maps(nxs_p)
    assert set(pks.keys()) == {1.651, 2.3}
    assert np.allclose(aggregate_peak_maps(pks, mode="all_mean"), 3.0)
    assert np.allclose(aggregate_peak_maps(pks, mode="low_q"), 2.0)
    assert np.allclose(aggregate_peak_maps(pks, mode="high_q"), 4.0)
    assert np.allclose(
        aggregate_peak_maps(pks, mode="single_peak", target_q=1.65, q_tol=0.02), 2.0
    )

    # 3. detrend_radial_cupping exact masked mean preservation & rank-deficient safety
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, 32, dtype=np.float32),
        np.linspace(-1, 1, 32, dtype=np.float32),
        indexing="ij",
    )
    img_cup = 5.0 + 1.5 * (yy**2 + xx**2)
    mask_corner = (yy > 0.0) & (xx > 0.0)
    det = detrend_radial_cupping(img_cup, valid_mask=mask_corner)
    assert abs(float(det[mask_corner].mean() - img_cup[mask_corner].mean())) < 1e-4

    # 4. remove_polar_rings_conservative: tiny <4px, NaN, and non-square
    tiny = np.ones((3, 3), dtype=np.float32)
    assert np.array_equal(remove_polar_rings_conservative(tiny), tiny)
    rect = np.ones((24, 40), dtype=np.float32) * 2.0
    rect[5, 5] = np.nan
    rect_clean = remove_polar_rings_conservative(rect)
    assert np.isnan(rect_clean[5, 5])
    assert np.isfinite(rect_clean[np.isfinite(rect)]).all()

    # 5. guide_to_tensor all-negative FBP noise clamps to 0.0
    neg_g = np.array([[-10.0, -2.0], [-5.0, -1.0]], dtype=np.float32)
    assert float(guide_to_tensor(neg_g, normalize=True).max().item()) == 0.0


def test_active_steering_nan_cold_midpoint_and_saturated_spacing() -> None:
    z_dim = 11
    probs = torch.full((2, z_dim, 8, 8), 0.5)
    ent = torch.full((z_dim, 8, 8), 0.02)
    epi = torch.zeros((z_dim, 8, 8))
    epi[0, 0, 0] = float("nan")
    rep = VolumetricUncertaintyReport(
        probabilities=probs,
        hard_mask=torch.zeros((z_dim, 8, 8), dtype=torch.int64),
        aleatoric_entropy=ent,
        triplane_variance=epi,
        physical_residual=torch.zeros_like(ent),
        combined_uncertainty=ent,
        z_slice_consistency=1.0,
    )
    cold = suggest_next_slices(rep, acquired_indices=(), k=1, min_spacing=3)
    assert cold.recommended_indices == [z_dim // 2]
    assert 0.0 <= cold.expected_uncertainty_gain <= 0.05

    # Saturated min_spacing=10 on Z=11 with anchors [0, 10] must bisect midpoint 5!
    sat = suggest_next_slices(rep, acquired_indices=[0, 10], k=1, min_spacing=10)
    assert sat.recommended_indices == [5]

    # suggest_2d_rois with NaN
    ph2 = generate_correlative_phantom_2d(size=32, downsample_factor=4, seed=9)
    m2 = XFuse(backbone=FAST_BACKBONE, pretrained=False, stride=16)
    rep2 = m2.fit_predict(ph2.hr_image, ph2.coarse_guide, epochs=1, tta_shifts=())
    rep2.combined_uncertainty[0, 0, 0] = float("nan")
    rois = suggest_2d_rois(rep2, roi_size=(8, 8), k=2)
    assert len(rois) == 2


def test_fetch_open_dataset_offline_urlerror_cache_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import urllib.error
    import urllib.request

    from x_fuse import fetch_open_dataset

    ds_dir = tmp_path / "edxia_cement_sem_eds"
    ds_dir.mkdir(parents=True)
    cached_zip = ds_dir / "edxia_examples.zip"
    cached_zip.write_bytes(b"cached_archive_bytes")

    def _fail_urlopen(*args: object, **kwargs: object) -> object:
        raise urllib.error.URLError("Simulated Zenodo 502 / offline")

    monkeypatch.setattr(urllib.request, "urlopen", _fail_urlopen)
    out_paths = fetch_open_dataset("edxia_cement_sem_eds", dest_root=tmp_path)
    assert out_paths == [cached_zip.resolve()]
