"""Adversarial Red Team regression test suite for X-FUSE v1.0.0.

Covers mathematical, statistical, stateful, numerical, architectural, I/O,
notebook, and security edge cases uncovered during the v1.0 Red Team audit.
"""

from __future__ import annotations

import io
import json
import re
import warnings
import zipfile
from pathlib import Path
from unittest.mock import patch

import h5py
import matplotlib
import numpy as np
import pytest
import torch

from benchmarks.baselines import (
    cnmf_pansharpen_baseline,
    coordinate_inr_baseline,
    gram_schmidt_pansharpen_baseline,
    guided_variational_sr_baseline,
)
from benchmarks.dataset_adapters import _safe_extract_zip
from x_fuse import (
    ConformalCalibrator,
    CorrelativeFuser,
    DifferentiableGuideAligner,
    PhysicsForwardModel,
    SpatialConvMLPHead,
    VolumeSliceAccessor,
    VolumetricFuser,
    XFuse,
    build_default_tta,
    empirical_conformal_coverage,
    fetch_open_dataset,
    generate_correlative_phantom_2d,
    generate_spectral_cube_phantom_2d,
    get_sample_dataset_path,
    guide_to_tensor,
    load_uncertainty_report,
    normalize_structural,
    pack_25d_slice,
    save_uncertainty_report,
)
from x_fuse.cli import main as cli_main
from x_fuse.fusion import XRDFusionMethod
from x_fuse.uncertainty import _safe_quantile

FAST_VIT = "timm:vit_tiny_patch16_224"


# ── 1. [P0-TTA] Sub-Pixel TTA Roll Alignment ─────────────────────────────────


def test_p0_tta_subpixel_guide_shift_and_canonical_training():
    """Verify XRDFusionMethod shifts coarse guides by exact sub-pixel HR/LR
    fractions rather than rounding 1 HR pixel to 1 full coarse pixel (8-16x error).
    """
    guide = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    guide[:, :, :, 4:] = 1.0
    fwd, inv = build_default_tta(shift_dists=[1, 2], include_flips=False)

    mod = XRDFusionMethod(xrd_img=guide, transform=fwd, require_grad=False)
    tr_sub = mod.get_tr(hr_hw=(64, 64))

    # Index 1 shifts by 1 HR pixel out of 64 -> 1/8 = 0.125 coarse pixels!
    # An integer coarse roll would shift column 4 all the way to column 5 (diff=1.0).
    diff_1px = (tr_sub[1] - guide).abs().max().item()
    assert 0.05 < diff_1px < 0.35, f"Expected sub-pixel shift, got max diff {diff_1px}"

    # Canonical-space training with align_guide='none' and tta_shifts=(1, 2)
    ph = generate_correlative_phantom_2d(size=64, downsample_factor=8, seed=17)
    model = XFuse(
        backbone=FAST_VIT,
        stride=8,
        pretrained=False,
        align_guide="none",
        lr=1.5e-2,
    )
    losses = model.fit(ph.hr_image, ph.coarse_guide, epochs=6, tta_shifts=(1, 2))
    assert len(losses) == 6
    assert losses[-1] < losses[0]


# ── 2. [P0-CONFORMAL] Out-of-Sample Split Conformal & Safe Quantile ──────────


def test_p0_conformal_strict_holdout_zero_training_leakage():
    """Prove that corrupting held-out conformal calibration pixels in coarse_guide
    produces bit-for-bit identical trained head weights (strict exchangeability).
    """
    ph = generate_correlative_phantom_2d(size=64, downsample_factor=4, seed=29)
    lr_hw = (ph.coarse_guide.shape[1], ph.coarse_guide.shape[2])
    cal_mask = ConformalCalibrator.make_spatial_holdout_mask(
        lr_hw, holdout_fraction=0.25
    ).numpy()

    guide_clean = ph.coarse_guide.copy()
    guide_poisoned = ph.coarse_guide.copy()
    # Poison ONLY the held-out calibration coarse pixels with adversarial values
    guide_poisoned[0, cal_mask] = 1.0
    guide_poisoned[1, cal_mask] = 0.0

    torch.manual_seed(1234)
    m_clean = XFuse(
        backbone=FAST_VIT,
        fusion_method="attention",
        stride=8,
        pretrained=False,
        conformal_alpha=0.10,
    )
    m_clean.fit(ph.hr_image, guide_clean, epochs=4, tta_shifts=(), tta_flips=False)

    torch.manual_seed(1234)
    m_poison = XFuse(
        backbone=FAST_VIT,
        fusion_method="attention",
        stride=8,
        pretrained=False,
        conformal_alpha=0.10,
    )
    m_poison.fit(ph.hr_image, guide_poisoned, epochs=4, tta_shifts=(), tta_flips=False)

    assert m_clean.spatial_attention is not None
    assert m_poison.spatial_attention is not None
    for p1, p2 in zip(
        m_clean.spatial_attention.parameters(),
        m_poison.spatial_attention.parameters(),
        strict=True,
    ):
        assert torch.allclose(p1, p2, atol=1e-6), (
            "Held-out conformal calibration pixels leaked into training weights!"
        )


def test_conformal_aps_lac_multilabel_poisson_scale_and_huge_quantile():
    """Verify APS vs LAC in calibrate_coarse, non-empty LAC sets, multilabel sets,
    Poisson _coarse_scale persistence, and >2^24 element _safe_quantile.
    """
    probs = torch.softmax(torch.randn(1, 3, 32, 32), dim=1)
    guide_counts = torch.zeros(1, 2, 8, 8)
    guide_counts[:, 0, :4, :] = 500.0
    guide_counts[:, 1, 4:, :] = 450.0

    cal_aps = ConformalCalibrator(alpha=0.10, method="aps")
    info_aps = cal_aps.calibrate_coarse(probs, guide_counts, mode="multiclass")
    assert cal_aps._coarse_scale is not None
    sets_aps, hw_aps = cal_aps.predict_sets_and_intervals(
        probs, guide_lr=guide_counts, mode="multiclass"
    )
    assert sets_aps.shape == (1, 3, 32, 32)
    assert hw_aps is not None and hw_aps.max().item() <= 1.0
    assert sets_aps.sum(dim=1).min().item() >= 1
    assert info_aps["conformal_method"] == "aps"

    # LAC with tiny q_label guarantees non-empty sets when allow_empty_sets=False
    cal_lac = ConformalCalibrator(alpha=0.50, method="lac", allow_empty_sets=False)
    cal_lac.q_label = 0.01
    uniform_probs = torch.full((1, 3, 16, 16), 1.0 / 3.0)
    sets_lac, _ = cal_lac.predict_sets_and_intervals(uniform_probs, mode="multiclass")
    assert int(sets_lac.sum(dim=1).min().item()) >= 1

    # Multilabel coarse calibration preserves independent channel sets
    probs_ml = torch.sigmoid(torch.randn(1, 2, 32, 32))
    guide_ml = (guide_counts / 500.0).clamp(0.0, 1.0)
    cal_ml = ConformalCalibrator(alpha=0.15, method="lac")
    cal_ml.calibrate_coarse(probs_ml, guide_ml, mode="multilabel")
    sets_ml, _ = cal_ml.predict_sets_and_intervals(
        probs_ml, guide_lr=guide_ml, mode="multilabel"
    )
    assert sets_ml.shape == (1, 2, 32, 32)

    # empirical_conformal_coverage on unbatched (K, H, W) sets + (H, W) labels
    y_hw = torch.randint(0, 3, (32, 32))
    cov = empirical_conformal_coverage(sets_aps[0], y_hw)
    assert 0.0 <= cov["empirical_coverage"] <= 1.0

    # > 2^24 (16,777,216) tensor quantile must not raise RuntimeError
    huge = torch.linspace(0.0, 1.0, 17_000_000, dtype=torch.float32)
    q99 = float(_safe_quantile(huge, 0.99).item())
    assert 0.985 <= q99 <= 0.995


# ── 3. [P0-STATEFUL] Multi-Specimen Isolation & NaN/Dark/FBP Guards ──────────


def test_p0_stateful_spectral_unmix_and_unguided_transfer_isolation():
    """Verify fit(hr1, cube1) -> predict(hr2, cube2) projects cube2 rather than
    reusing cube1's cached abundances, and predict(hr2) runs unguided transfer.
    """
    ph1 = generate_spectral_cube_phantom_2d(
        size=48, downsample_factor=4, n_spectral_bins=8, mean_counts=80.0, seed=10
    )
    # Create spatially flipped second specimen
    hr2 = np.flip(ph1.hr_image, axis=(0, 1)).copy()
    cube2 = np.flip(ph1.spectral_cube_lr, axis=(1, 2)).copy()

    model = XFuse(
        backbone=FAST_VIT,
        stride=8,
        loss_fn="forward_psf_poisson",
        spectral_channels=8,
        guide_channels=2,
        align_guide="translation",
        pretrained=False,
    )
    model.fit(
        ph1.hr_image,
        ph1.spectral_cube_lr,
        epochs=4,
        tta_shifts=(),
        tta_flips=False,
        spectral_unmix=True,
        n_components=2,
    )

    rep1 = model.predict(ph1.hr_image, ph1.spectral_cube_lr)
    rep2 = model.predict(hr2, cube2)
    # Flipped specimen predictions correlate with flipped rep1, not stale rep1!
    p1 = rep1.probabilities[0].cpu().numpy()
    p2 = rep2.probabilities[0].cpu().numpy()
    err_flipped = np.mean(np.abs(p2 - np.flip(p1, axis=(1, 2))))
    err_stale = np.mean(np.abs(p2 - p1))
    assert err_flipped < err_stale

    # Unguided transfer predict(hr2, coarse_guide=None) must NOT leak hr1's guide
    rep_unguided = model.predict(hr2, coarse_guide=None)
    assert rep_unguided.diagnostics.get("unguided_transfer") is True
    assert float(rep_unguided.physical_residual.abs().max().item()) == 0.0


def test_nan_inf_dark_cube_and_negative_fbp_ringing_guards():
    """Verify NaN/Inf warnings, dark-cube zero preservation, negative FBP clamping,
    and Poisson normalized-input warning.
    """
    hr_nan = np.full((32, 32), 0.5, dtype=np.float32)
    hr_nan[0, 0] = np.nan
    with pytest.warns(UserWarning, match="NaN/Inf"):
        hr_t = normalize_structural(hr_nan)
    assert torch.isfinite(hr_t).all()

    g_nan = np.ones((2, 8, 8), dtype=np.float32)
    g_nan[0, 2, 2] = np.nan
    with pytest.warns(UserWarning, match="NaN/Inf"):
        g_t = guide_to_tensor(g_nan, normalization="minmax")
    assert torch.isfinite(g_t).all()
    assert float(g_t[0, 0].sum().item()) > 0.0  # Finite pixels NOT wiped out!

    # Negative FBP ringing clamp in minmax
    fbp = np.array([[-0.25, 0.0], [0.5, 1.0]], dtype=np.float32)
    fbp_t = guide_to_tensor(fbp, normalization="minmax")
    assert float(fbp_t[0, 0, 0, 0].item()) == 0.0
    assert float(fbp_t[0, 0, 0, 1].item()) == 0.0
    assert np.isclose(float(fbp_t[0, 0, 1, 0].item()), 0.5)

    # All-zero dark cube stays strictly 0.0
    dark = np.zeros((4, 6, 6), dtype=np.float32)
    dark_t = guide_to_tensor(dark, normalization="global_max")
    assert float(dark_t.abs().max().item()) == 0.0

    # Normalized [0, 1] input to forward_psf_poisson warns once
    phys = PhysicsForwardModel(n_channels=1, loss_fn="forward_psf_poisson")
    logits = torch.zeros((1, 1, 16, 16), requires_grad=True)
    unit_guide = torch.rand((1, 1, 4, 4)) * 0.8
    with pytest.warns(UserWarning, match="preserve_counts"):
        loss = phys.compute_loss_from_logits(logits, unit_guide)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


# ── 4. Registration Geometry (Non-Square Rigid + Diffeomorphic B-Spline) ─────


def test_nonsquare_rigid_euclidean_orthogonality_and_bspline_diffeomorphism():
    """Verify rigid rotation on non-square (48, 96) preserves physical Euclidean
    orthogonality and B-spline warps stay strictly fold-free (det J > 0).
    """
    aligner = DifferentiableGuideAligner(mode="rigid", max_shift_coarse_px=2.0)
    with torch.no_grad():
        assert aligner.raw_angle is not None
        aligner.raw_angle.fill_(2.0)  # ~10 deg rotation

    grid = aligner.build_sampling_grid(
        batch_size=1,
        channels=1,
        hr_hw=(48, 96),
        lr_hw=(12, 24),
        device=torch.device("cpu"),
    )[0].detach()
    # Convert normalized grid [-1, 1] back to physical pixel coordinates (x_px, y_px)
    x_px = (grid[..., 0] + 1.0) * 0.5 * 96.0
    y_px = (grid[..., 1] + 1.0) * 0.5 * 48.0
    # Finite-difference Jacobian in physical pixel units at image center
    cy, cx = 24, 48
    vx = np.array(
        [float(x_px[cy, cx + 1] - x_px[cy, cx]), float(y_px[cy, cx + 1] - y_px[cy, cx])]
    )
    vy = np.array(
        [float(x_px[cy + 1, cx] - x_px[cy, cx]), float(y_px[cy + 1, cx] - y_px[cy, cx])]
    )
    dot_xy = abs(float(np.dot(vx, vy)))
    norm_ratio = float(np.linalg.norm(vx) / np.linalg.norm(vy))
    assert dot_xy < 1e-4, f"Rigid rotation sheared non-square image: dot={dot_xy}"
    assert np.isclose(norm_ratio, 1.0, atol=1e-3)

    # B-spline diffeomorphic guarantee even under extreme control-grid saturation
    bspline = DifferentiableGuideAligner(mode="bspline", control_grid_size=6)
    with torch.no_grad():
        assert bspline.raw_control_grid is not None
        gen = torch.Generator().manual_seed(99)
        bspline.raw_control_grid.copy_(50.0 * torch.randn((1, 2, 6, 6), generator=gen))
    min_det = bspline.min_jacobian_determinant(
        hr_hw=(64, 64), lr_hw=(16, 16), device=torch.device("cpu")
    )
    assert min_det > 0.0, f"B-spline folded space: min det(J) = {min_det}"


def test_odd_nondivisible_shape_with_tiling_attention_and_conformal():
    """Verify odd (53, 61) HR input works seamlessly with tile_size, attention head,
    and conformal_alpha without shape errors or stretching valid_hr_hw.
    """
    rng = np.random.default_rng(5)
    hr_odd = rng.random((53, 61), dtype=np.float32)
    guide_odd = rng.random((2, 13, 15), dtype=np.float32)

    model = XFuse(
        backbone=FAST_VIT,
        fusion_method="attention",
        stride=8,
        tile_size=48,
        tile_overlap=16,
        conformal_alpha=0.15,
        pretrained=False,
    )
    rep = model.fit_predict(hr_odd, guide_odd, epochs=3, tta_shifts=(), tta_flips=False)
    assert rep.probabilities.shape == (1, 3, 53, 61)
    assert rep.prediction_sets is not None
    assert rep.prediction_sets.shape == (1, 3, 53, 61)


# ── 5. Cache Collision Resistance, Eval Determinism & Tiled Debiasing ────────


def test_tta_cache_crc32_resists_circular_mask_collision_and_eval_determinism():
    """Verify two circularly masked slices with identical shape, border zeros, and
    global sum/mean do NOT collide in _tta_cache, and attention eval is deterministic.
    """
    yy, xx = np.ogrid[:48, :48]
    inside = (yy - 24) ** 2 + (xx - 24) ** 2 <= 14**2

    s1 = np.zeros((3, 48, 48), dtype=np.float32)
    s2 = np.zeros((3, 48, 48), dtype=np.float32)
    # Identical zero corners/borders AND identical mean/sum, opposite interior halves!
    left_in = inside & (xx < 24)
    right_in = inside & (xx >= 24)
    n_pix = min(int(left_in.sum()), int(right_in.sum()))
    ly, lx = np.where(left_in)
    ry, rx = np.where(right_in)
    s1[:, ly[:n_pix], lx[:n_pix]] = 1.0
    s2[:, ry[:n_pix], rx[:n_pix]] = 1.0

    t1 = torch.from_numpy(s1)
    t2 = torch.from_numpy(s2)
    assert np.isclose(float(t1.sum()), float(t2.sum()))

    fuser = CorrelativeFuser(
        dino_name=FAST_VIT,
        xrd_fuse_method="attention",
        stride=8,
        pretrained=False,
        cache_features=True,
    )
    assert fuser._cache_key(t1, want_hr=True) != fuser._cache_key(t2, want_hr=True)

    # Attention head determinism in predict() after fit()
    guide = np.random.default_rng(3).random((2, 12, 12), dtype=np.float32)
    fuser.fit(s1[0], guide, epochs=2, tta_shifts=(), tta_flips=False)
    r_a = fuser.predict(s1[0], guide)
    r_b = fuser.predict(s1[0], guide)
    assert torch.equal(r_a.probabilities, r_b.probabilities)

    # SpatialConvMLPHead 1x1 spatial input in train() mode must not raise ValueError
    mlp = SpatialConvMLPHead(feat_dim=32, hidden_dim=16, out_channels=2)
    mlp.train()
    out_1x1 = mlp.predict_logits(
        torch.randn(1, 32, 1, 1), guidance_hr=torch.randn(1, 3, 1, 1)
    )
    assert out_1x1.shape == (1, 2, 1, 1)


def test_tiled_debias_positional_matches_full_fov_without_seams():
    """Verify tiled extraction with debias_positional=True applies debiasing after
    full-FOV stitching so tile boundaries do not exhibit discontinuous jumps.
    """
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, 96, dtype=np.float32),
        np.linspace(0.0, 1.0, 96, dtype=np.float32),
        indexing="ij",
    )
    hr = 0.5 * yy + 0.5 * xx
    guide = np.stack([yy[::8, ::8], 1.0 - yy[::8, ::8]], axis=0)

    torch.manual_seed(77)
    m_full = XFuse(
        backbone=FAST_VIT,
        stride=8,
        debias_positional=True,
        tile_size=None,
        pretrained=False,
    )
    f_full = m_full.forward_raw_features(normalize_structural(hr))

    torch.manual_seed(77)
    m_tiled = XFuse(
        backbone=FAST_VIT,
        stride=8,
        debias_positional=True,
        tile_size=64,
        tile_overlap=16,
        pretrained=False,
    )
    f_tiled = m_tiled.forward_raw_features(normalize_structural(hr))
    assert f_full.shape == f_tiled.shape
    # Check continuity across tile seam x=64
    seam_jump = (f_tiled[:, :, :, 64] - f_tiled[:, :, :, 63]).abs().mean().item()
    interior_jump = (f_tiled[:, :, :, 32] - f_tiled[:, :, :, 31]).abs().mean().item()
    assert seam_jump <= 2.5 * max(interior_jump, 1e-3)

    rep_tiled = m_tiled.fit_predict(hr, guide, epochs=2, tta_shifts=(), tta_flips=False)
    assert rep_tiled.probabilities.shape == (1, 3, 96, 96)


# ── 6. 2D & 3D Conformal Round-Trip I/O (Including Z=1 Volumetric) ───────────


def test_conformal_io_roundtrip_ome_tiff_h5_zarr_and_z1_volume(tmp_path: Path):
    """Verify prediction_sets and conformal_interval_halfwidth survive OME-TIFF,
    HDF5, and Zarr round-trips, and Z=1 3D volumes keep their Z=1 axis.
    """
    ph = generate_correlative_phantom_2d(size=48, downsample_factor=4, seed=8)
    model = XFuse(
        backbone=FAST_VIT,
        stride=8,
        conformal_alpha=0.10,
        pretrained=False,
    )
    rep = model.fit_predict(
        ph.hr_image, ph.coarse_guide, epochs=3, tta_shifts=(), tta_flips=False
    )
    assert rep.prediction_sets is not None
    assert rep.conformal_interval_halfwidth is not None

    for ext in (".ome.tif", ".h5", ".zarr"):
        out_p = tmp_path / f"conformal_2d{ext}"
        save_uncertainty_report(rep, out_p, pixel_size_um=0.1)
        loaded = load_uncertainty_report(out_p)
        assert "prediction_sets" in loaded
        assert loaded["prediction_sets"].dtype == bool
        assert np.array_equal(
            loaded["prediction_sets"], rep.prediction_sets[0].cpu().numpy()
        )
        assert "conformal_interval_halfwidth" in loaded
        assert np.allclose(
            loaded["conformal_interval_halfwidth"],
            rep.conformal_interval_halfwidth[0].cpu().numpy(),
            atol=1e-5,
        )

    # Z=1 3D VolumetricUncertaintyReport round-trip to OME-TIFF & HDF5
    vol_z1 = ph.hr_image[None, :, :]  # (Z=1, 48, 48)
    vf = VolumetricFuser(model, tta_shifts=(), tta_flips=False)
    v_rep = vf.predict_volume(
        vol_z1,
        mode="2.5d_slab",
        sparse_guides={0: ph.coarse_guide},
        pred_mode="multiclass",
    )
    assert v_rep.probabilities.shape == (3, 1, 48, 48)
    assert v_rep.prediction_sets is not None

    for ext in (".ome.tif", ".h5"):
        vp = tmp_path / f"vol_z1{ext}"
        save_uncertainty_report(v_rep, vp)
        v_loaded = load_uncertainty_report(vp)
        assert v_loaded["hard_mask"].shape == (1, 48, 48)
        assert v_loaded["prediction_sets"].shape == (3, 1, 48, 48)

    # Direct file-path VolumeSliceAccessor (.npy and .h5)
    npy_p = tmp_path / "tomo.npy"
    np.save(npy_p, np.ones((4, 16, 16), dtype=np.float32))
    acc_npy = VolumeSliceAccessor(npy_p)
    assert acc_npy.shape == (4, 16, 16)

    h5_p = tmp_path / "tomo.h5"
    with h5py.File(h5_p, "w") as hf:
        hf.create_dataset("entry/data", data=np.ones((5, 16, 16), dtype=np.float32))
    acc_h5 = VolumeSliceAccessor(h5_p)
    assert acc_h5.shape == (5, 16, 16)
    assert acc_h5.get_plane(0, 2).shape == (16, 16)
    slab = pack_25d_slice(h5_p, 0, axis=0, delta=1)
    assert slab.shape == (3, 16, 16)


# ── 7. Security, CLI Validation & Competing Baseline Isolation ───────────────


def test_security_path_traversal_https_zipslip_and_cli_validation(tmp_path: Path):
    """Verify path traversal, non-HTTPS URLs, Zip-Slip archives, and invalid CLI
    flags are strictly rejected.
    """
    with pytest.raises(ValueError, match="Invalid sample dataset name"):
        get_sample_dataset_path("../outside")

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    # 1. Path traversal filename in Zenodo API response
    evil_traversal = {
        "files": [
            {
                "key": "../../escaped_evil.tif",
                "checksum": "",
                "links": {"self": "https://zenodo.org/api/files/ok.tif"},
            }
        ]
    }
    with patch(
        "urllib.request.urlopen",
        side_effect=lambda *a, **kw: _Resp(json.dumps(evil_traversal).encode()),
    ):
        with pytest.raises(ValueError, match="unsafe remote filename"):
            fetch_open_dataset("edxia_cement_sem_eds", dest_root=tmp_path)

    # 2. Non-HTTPS file:// or http:// URL in Zenodo API response
    evil_scheme = {
        "files": [
            {
                "key": "map.tif",
                "checksum": "",
                "links": {"self": "file:///etc/passwd"},
            }
        ]
    }
    with patch(
        "urllib.request.urlopen",
        side_effect=lambda *a, **kw: _Resp(json.dumps(evil_scheme).encode()),
    ):
        with pytest.raises(ValueError, match="non-HTTPS"):
            fetch_open_dataset("edxia_cement_sem_eds", dest_root=tmp_path)

    # 3. Zip-Slip archive extraction rejection
    bad_zip = tmp_path / "slip.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../../pwned.txt", "evil")
    with pytest.raises(ValueError, match="Zip-Slip"):
        _safe_extract_zip(bad_zip, tmp_path / "extracted")

    # 4. CLI upfront validation
    hr_f = tmp_path / "hr.npy"
    g_f = tmp_path / "g.npy"
    np.save(hr_f, np.zeros((32, 32), dtype=np.float32))
    np.save(g_f, np.zeros((8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="--stride"):
        cli_main(
            [
                "fuse",
                "--hr",
                str(hr_f),
                "--guide",
                str(g_f),
                "--output",
                str(tmp_path / "o.h5"),
                "--stride",
                "0",
            ]
        )
    with pytest.raises(ValueError, match="--conformal-alpha"):
        cli_main(
            [
                "fuse",
                "--hr",
                str(hr_f),
                "--guide",
                str(g_f),
                "--output",
                str(tmp_path / "o.h5"),
                "--conformal-alpha",
                "1.2",
            ]
        )


def test_baselines_channels_last_rgb_and_global_rng_isolation():
    """Verify all 4 competing baselines handle channels-last (H, W, 3) RGB inputs
    and coordinate_inr_baseline does not mutate global torch RNG state.
    """
    rng = np.random.default_rng(19)
    rgb_hwc = rng.random((32, 40, 3), dtype=np.float32)
    guide = rng.random((2, 8, 10), dtype=np.float32)

    r_gs = gram_schmidt_pansharpen_baseline(rgb_hwc, guide)
    r_cnmf = cnmf_pansharpen_baseline(rgb_hwc, guide, n_iters=6)
    r_gtv = guided_variational_sr_baseline(rgb_hwc, guide, n_steps=5)

    torch.manual_seed(2026)
    rng_before = torch.random.get_rng_state().clone()
    r_inr = coordinate_inr_baseline(rgb_hwc, guide, epochs=5, hidden_dim=16, seed=99)
    rng_after = torch.random.get_rng_state()
    assert torch.equal(rng_before, rng_after), (
        "coordinate_inr_baseline mutated global torch RNG state!"
    )

    for rep in (r_gs, r_cnmf, r_gtv, r_inr):
        assert rep.probabilities.shape == (1, 3, 32, 40)


# ── 8. Tutorial Notebook Smoke Execution ─────────────────────────────────────


@pytest.mark.parametrize(
    "notebook_name",
    [
        "01_correlative_2d_and_uncertainty.ipynb",
        "02_volumetric_25d_propagation.ipynb",
        "03_active_acquisition_and_registration.ipynb",
    ],
)
def test_tutorial_notebooks_execute_cleanly(notebook_name: str):
    """Execute every code cell of each shipped tutorial notebook (with fast
    epoch/grid substitutions) so attribute renames can never silently break tutorials.
    """
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nb_path = Path(__file__).resolve().parent.parent / "notebooks" / notebook_name
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    ns: dict[str, object] = {"__name__": "__notebook_smoke__"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with patch.object(plt, "show", lambda *a, **kw: None):
            for cell in nb["cells"]:
                if cell.get("cell_type") != "code":
                    continue
                code = "".join(cell["source"])
                # Speed up training loops for CI smoke testing of exact tutorial APIs
                code = re.sub(r"range\(30\)", "range(2)", code)
                code = re.sub(r"epochs\s*=\s*\d+", "epochs=2", code)
                code = code.replace("size=128", "size=64").replace("size=96", "size=48")
                code = code.replace("(32, 64, 64)", "(8, 32, 32)")
                code = code.replace("[4, 16, 27]", "[1, 4, 6]")
                code = code.replace("unseen_z = 11", "unseen_z = 3")
                code = code.replace("(2, 16, 4, 16, 4)", "(2, 8, 4, 8, 4)")
                code = code.replace("(20, 48, 48)", "(8, 32, 32)")
                code = code.replace(
                    "17: phantom_3d.coarse_guide_3d[:, 4]",
                    "6: phantom_3d.coarse_guide_3d[:, 1]",
                )
                exec(compile(code, f"{notebook_name}", "exec"), ns)
    plt.close("all")
