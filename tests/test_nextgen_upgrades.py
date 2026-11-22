"""Unit tests for the five next-generation X-FUSE architectural & dataset upgrades:
1. Spin-2 structure-tensor orientation & continuous sub-token cross-attention upsampling
2. Anisotropic Moffat & Core+Halo forward PSF profiles
3. 3D axial interaction-volume PSF deconvolution & unlabeled tri-plane consistency
4. Temperature-scaled RAPS conformal calibration with spatial block subsampling
5. Zenodo default subsetting, Egerton power-law EELS stripping, and LZW/RGB adapters
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from benchmarks.dataset_adapters import (
    _strip_eels_powerlaw_window,
    load_zenodo_correlative_sample,
)
from x_fuse import (
    ConformalCalibrator,
    CorrelativeFuser,
    CrossAttentionFeatureUpsampler,
    ForwardPSFDownsampler,
    VolumetricFuser,
    XFuse,
    empirical_conformal_coverage,
    extract_multiscale_texture_descriptors,
    extract_structure_tensor_field,
)
from x_fuse.datasets import fetch_open_dataset
from x_fuse.heads import SpatialConvMLPHead, edge_aware_tv_loss
from x_fuse.synthetic import generate_correlative_phantom_3d

# ── Upgrade 1: Spin-2 Orientation & Continuous Sub-Token Upsampling ──────────


def test_texture_contract_and_spin2_iso_intensity_separation() -> None:
    yy, xx = torch.meshgrid(
        torch.arange(64, dtype=torch.float32),
        torch.arange(64, dtype=torch.float32),
        indexing="ij",
    )
    th_a = float(np.deg2rad(25.0))
    th_b = float(np.deg2rad(115.0))
    lam_a = 0.5 + 0.35 * torch.cos(
        2.0 * np.pi * (xx * np.cos(th_a) + yy * np.sin(th_a)) / 5.0
    )
    lam_b = 0.5 + 0.35 * torch.cos(
        2.0 * np.pi * (xx * np.cos(th_b) + yy * np.sin(th_b)) / 5.0
    )
    img = torch.where(xx < 32.0, lam_a, lam_b).unsqueeze(0).unsqueeze(0)

    tex = extract_multiscale_texture_descriptors(img)
    assert tex.shape == (1, 8, 64, 64)

    st = extract_structure_tensor_field(img)
    assert st.shape == (1, 4, 64, 64)
    left_vec = st[0, :2, 16:48, 8:24].mean(dim=(-2, -1))
    right_vec = st[0, :2, 16:48, 40:56].mean(dim=(-2, -1))
    cos_sim = float(
        torch.dot(left_vec, right_vec)
        / (left_vec.norm() * right_vec.norm()).clamp(min=1e-8)
    )
    # Orthogonal lamellar colonies (90 deg apart) map to antipodal spin-2 vectors
    assert cos_sim < -0.75


@pytest.mark.parametrize("learned", [False, True])
def test_cross_attn_upsampler_exact_d4_equivariance(learned: bool) -> None:
    torch.manual_seed(7)
    up = CrossAttentionFeatureUpsampler(
        feat_dim=16,
        key_dim=8,
        radius=2,
        learned=learned,
        orientation_aware=True,
        subtoken_offsets=True,
    ).eval()

    patch_feats = torch.randn(1, 16, 6, 6)
    guidance = torch.rand(1, 3, 24, 24)
    ref = up(patch_feats, guidance, (24, 24))
    assert ref.shape == (1, 16, 24, 24)

    # Check mirror flips and 90-degree rotations in D_4
    for k in (1, 2, 3):
        pf_rot = torch.rot90(patch_feats, k=k, dims=(-2, -1))
        gd_rot = torch.rot90(guidance, k=k, dims=(-2, -1))
        out_rot = up(pf_rot, gd_rot, (24, 24))
        out_inv = torch.rot90(out_rot, k=-k, dims=(-2, -1))
        assert torch.allclose(ref, out_inv, atol=1e-5), f"D4 rot k={k} mismatch"

    for dim in (-1, -2):
        pf_flp = torch.flip(patch_feats, dims=(dim,))
        gd_flp = torch.flip(guidance, dims=(dim,))
        out_flp = up(pf_flp, gd_flp, (24, 24))
        out_inv = torch.flip(out_flp, dims=(dim,))
        assert torch.allclose(ref, out_inv, atol=1e-5), f"D4 flip dim={dim} mismatch"


def test_spin2_edge_tv_and_spatial_mlp_coherence_grad() -> None:
    probs = torch.rand(1, 2, 32, 32, requires_grad=True)
    guide_hr = torch.rand(1, 3, 32, 32)
    tv = edge_aware_tv_loss(probs, guide_hr, orientation_aware=True)
    tv.backward()
    assert probs.grad is not None and torch.isfinite(probs.grad).all()

    head = SpatialConvMLPHead(
        feat_dim=16,
        out_channels=2,
        loss_fn="compositional_kl",
        psf_profile="moffat",
        learn_psf=True,
        orientation_aware=True,
    )
    feats = torch.randn(1, 16, 32, 32, requires_grad=True)
    target = torch.rand(1, 2, 8, 8) * 0.4
    logits = head.predict_logits(feats, guidance_hr=guide_hr)
    loss = head.compute_loss_from_logits(logits, xrd_map=target, guidance_hr=guide_hr)
    loss.backward()
    assert head.coh_proj.weight.grad is not None


# ── Upgrade 2: Moffat & Core+Halo Forward PSF Profiles ───────────────────────


@pytest.mark.parametrize("profile", ["gaussian", "moffat", "core_halo"])
def test_forward_psf_profiles_flux_conservation_and_grad(profile: str) -> None:
    ds = ForwardPSFDownsampler(
        psf_sigma_ratio=0.5,
        learnable=True,
        n_channels=2,
        psf_profile=profile,  # type: ignore[arg-type]
    )
    hr = torch.rand(1, 2, 32, 32, requires_grad=True)
    lr = ds(hr, (8, 8))
    assert lr.shape == (1, 2, 8, 8)
    # Average intensity (total flux per unit area) must be conserved
    assert torch.allclose(hr.mean(dim=(-2, -1)), lr.mean(dim=(-2, -1)), atol=2e-2)

    # Constant field must map identically to the same constant to machine precision
    const_hr = torch.full((1, 2, 32, 32), 0.73)
    const_lr = ds(const_hr, (8, 8))
    assert torch.allclose(const_lr, torch.full_like(const_lr, 0.73), atol=1e-5)

    loss = (lr - 0.3).square().mean()
    loss.backward()
    diag = ds.diagnostics((32, 32), (8, 8))
    assert diag["psf_profile"] == profile
    if profile == "moffat":
        assert ds.raw_moffat_beta is not None and ds.raw_moffat_beta.grad is not None
        assert "moffat_beta" in diag
    elif profile == "core_halo":
        assert ds.raw_halo_logit is not None and ds.raw_halo_logit.grad is not None
        assert "halo_weight" in diag


def test_xfuse_end_to_end_with_core_halo_and_raps() -> None:
    rng = np.random.default_rng(11)
    hr = rng.random((56, 56), dtype=np.float32)
    guide = rng.random((2, 14, 14), dtype=np.float32) * 0.4
    model = XFuse(
        backbone="dinov2_vits14",
        pretrained=False,
        device="cpu",
        epochs=2,
        tta_shifts=[],
        tta_flips=False,
        psf_profile="core_halo",
        learn_psf=True,
        conformal_alpha=0.10,
        conformal_method="raps",
        conformal_fit_temperature=True,
        conformal_spatial_block_size=2,
    )
    rep = model.fit_predict(hr, guide)
    assert rep.probabilities.shape == (1, 3, 56, 56)
    assert rep.prediction_sets is not None
    assert rep.diagnostics["psf_profile"] == "core_halo"


# ── Upgrade 3: 3D Axial Depth-PSF & Unlabeled Tri-Plane Consistency ──────────


def test_volumetric_depth_psf_and_triplane_consistency() -> None:
    ph_sharp = generate_correlative_phantom_3d(
        shape=(16, 28, 28), downsample_factor=4, seed=5, depth_psf_sigma=0.0
    )
    ph_thick = generate_correlative_phantom_3d(
        shape=(16, 28, 28), downsample_factor=4, seed=5, depth_psf_sigma=3.0
    )
    # Thick axial PSF reduces Z-slice variance in the coarse chemical volume
    var_sharp = float(np.var(np.diff(ph_sharp.coarse_guide_3d, axis=1)))
    var_thick = float(np.var(np.diff(ph_thick.coarse_guide_3d, axis=1)))
    assert var_thick < var_sharp

    fuser = CorrelativeFuser(
        model_type="dinov2_vits14",
        pretrained=False,
        fuse_method="spatial_mlp",
        loss_fn="compositional_kl",
        guide_channels=2,
        device="cpu",
    )
    vf = VolumetricFuser(fuser, tta_shifts=None, tta_flips=False)
    sparse = {4: ph_thick.coarse_guide_3d[:, 1], 12: ph_thick.coarse_guide_3d[:, 3]}
    losses = vf.fit_sparse_slices(
        ph_thick.hr_volume,
        sparse,
        epochs=3,
        depth_psf_sigma=2.0,
        learn_depth_psf=True,
        depth_stencil_radius=1,
        triplane_consistency_weight=0.1,
        n_consistency_slices=1,
    )
    assert len(losses) == 3 and all(np.isfinite(losses))
    rep3d = vf.predict_volume(
        ph_thick.hr_volume,
        mode="2.5d_slab",
        sparse_guides=sparse,
        pred_mode="multiclass",
    )
    assert rep3d.hard_mask.shape == (16, 28, 28)
    assert rep3d.diagnostics["depth_psf_enabled"] is True
    assert rep3d.diagnostics["depth_psf_sigma_learned"] > 0.25


# ── Upgrade 4: Temperature-Scaled RAPS Conformal Calibration ─────────────────


def test_temperature_scaled_raps_conformal_coverage_and_singleton_interior() -> None:
    h, w = 64, 64
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    gt_a = np.zeros((h, w), dtype=np.int64)
    gt_a[(xx >= 16) & (xx < 40)] = 1
    gt_a[xx >= 40] = 2

    gt_b = np.zeros((h, w), dtype=np.int64)
    gt_b[(yy >= 14) & (yy < 42)] = 1
    gt_b[yy >= 42] = 2

    def _make_overconfident_probs(gt: np.ndarray, seed: int) -> torch.Tensor:
        r = np.random.default_rng(seed)
        logits = np.full((3, h, w), -2.5, dtype=np.float32)
        for c in range(3):
            logits[c][gt == c] = 3.2
        # Add realistic noise and boundary ambiguity so raw logits are miscalibrated
        bnd = np.zeros_like(gt, dtype=bool)
        bnd[:, 1:] |= gt[:, 1:] != gt[:, :-1]
        bnd[1:, :] |= gt[1:, :] != gt[:-1, :]
        logits += r.normal(0.0, 0.9, size=logits.shape).astype(np.float32)
        logits[:, bnd] = r.normal(0.0, 2.2, size=(3, int(bnd.sum()))).astype(np.float32)
        t = torch.from_numpy(logits).unsqueeze(0) * 2.0  # artificially overconfident
        return torch.softmax(t, dim=1)

    probs_a = _make_overconfident_probs(gt_a, seed=1)
    probs_b = _make_overconfident_probs(gt_b, seed=2)

    cal = ConformalCalibrator(
        alpha=0.10,
        method="raps",
        k_reg=1,
        lambda_reg=0.05,
        randomized=False,
        fit_temperature=True,
        spatial_block_size=2,
    )
    cal.calibrate_labels(probs_a, torch.from_numpy(gt_a))
    assert cal.temperature > 1.05  # Temperature scaling softens overconfident logits

    sets_b, _ = cal.predict_sets_and_intervals(probs_b, mode="multiclass")
    stats = empirical_conformal_coverage(sets_b, torch.from_numpy(gt_b))
    assert stats["empirical_coverage"] >= 0.85
    assert stats["boundary_mean_set_size"] > stats["interior_mean_set_size"]
    assert stats["interior_singleton_fraction"] > 0.50


# ── Upgrade 5: Zenodo Subset Filtering & Adapter Hardening ───────────────────


def test_zenodo_default_subset_and_adapters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_payload = {
        "files": [
            {
                "key": "CuZn_HAADF.hspy",
                "links": {"self": "https://zenodo.example/haadf"},
            },
            {
                "key": "CuZn_EELS_mapping_tutorial.hspy",
                "links": {"self": "https://zenodo.example/eels"},
            },
            {
                "key": "extra_large_unused.zip",
                "links": {"self": "https://zenodo.example/unused"},
            },
        ]
    }

    class _FakeResp(io.BytesIO):
        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *args: Any) -> None:
            self.close()

    def _fake_urlopen(req: Any, timeout: float = 30.0) -> _FakeResp:
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "/api/records/" in url:
            return _FakeResp(json.dumps(record_payload).encode("utf-8"))
        return _FakeResp(b"mock_binary")

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    downloaded = fetch_open_dataset("cuzn_stem_eels", dest_root=tmp_path)
    assert [p.name for p in downloaded] == [
        "CuZn_HAADF.hspy",
        "CuZn_EELS_mapping_tutorial.hspy",
    ]

    # Verify PDF-only directory raises explicit RuntimeError
    pdf_dir = tmp_path / "edxia_cement_sem_eds"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "thesis_only.pdf").write_bytes(b"%PDF-1.4")
    with pytest.raises(RuntimeError, match="PDF"):
        load_zenodo_correlative_sample("edxia_cement_sem_eds", dest_root=tmp_path)

    # Verify Egerton power-law EELS continuum stripping recovers pure edge step
    energies = 700.0 + 8.0 * np.arange(80, dtype=np.float32)
    bg = 12000.0 * (energies / 700.0) ** (-3.1)
    step = np.where((energies >= 932.0) & (energies <= 1004.0), 450.0, 0.0)
    cube = np.broadcast_to((bg + step)[None, None, :], (8, 8, 80)).copy()
    edge_map, net_cube = _strip_eels_powerlaw_window(
        cube, energies, (780.0, 916.0), (932.0, 1004.0)
    )
    expected_sum = float(step[(energies >= 932.0) & (energies <= 1004.0)].sum())
    assert np.allclose(edge_map, expected_sum, rtol=1e-2)
    assert net_cube.shape[:2] == (8, 8)


def test_poisson_spectral_high_count_no_nan() -> None:
    from x_fuse.physics import PhysicsForwardModel

    phys = PhysicsForwardModel(
        n_channels=2,
        loss_fn="forward_psf_poisson",
        learn_psf=True,
        spectral_channels=8,
    )
    logits = torch.randn(1, 2, 16, 16, requires_grad=True)
    # Background > 1000 counts previously overflowed float32 torch.expm1(bg)
    high_flux_cube = torch.full((1, 8, 4, 4), 2500.0) + 500.0 * torch.rand(1, 8, 4, 4)
    phys.spectral_dict.initialize_from_cube(high_flux_cube)  # type: ignore[union-attr]
    loss = phys.compute_loss_from_logits(logits, high_flux_cube)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_redteam_bf16_d4_noninteger_and_aps_bijection() -> None:
    from x_fuse.heads import (
        SpatialConvMLPHead,
        WindowedGuideCrossAttentionHead,
        edge_aware_tv_loss,
    )
    from x_fuse.upsamplers import CrossAttentionFeatureUpsampler

    # 1. CPU bfloat16 inputs & module parameters across edge_aware_tv_loss and heads
    logits_bf16 = torch.randn(1, 2, 16, 16, dtype=torch.bfloat16)
    guide_bf16 = torch.rand(1, 3, 16, 16, dtype=torch.bfloat16)
    tv = edge_aware_tv_loss(logits_bf16, guidance_hr=guide_bf16, orientation_aware=True)
    assert torch.isfinite(tv)

    head_mlp = SpatialConvMLPHead(
        feat_dim=16, hidden_dim=16, out_channels=2, guide_residual=True
    ).to(torch.bfloat16)
    feat_bf16 = torch.randn(1, 16, 16, 16, dtype=torch.bfloat16)
    out_mlp = head_mlp.predict_logits(feat_bf16, guidance_hr=guide_bf16)
    out_mlp = head_mlp.apply_guide_residual(
        out_mlp, torch.rand(1, 2, 4, 4, dtype=torch.bfloat16)
    )
    assert out_mlp.dtype == torch.float32 and torch.isfinite(out_mlp).all()

    head_attn = WindowedGuideCrossAttentionHead(
        feat_dim=16, num_heads=4, guide_channels=2
    ).to(torch.bfloat16)
    out_attn = head_attn.predict_logits(
        feat_bf16, torch.rand(1, 2, 4, 4, dtype=torch.bfloat16)
    )
    assert out_attn.dtype == torch.float32 and torch.isfinite(out_attn).all()

    # 2. Deterministic APS/RAPS (randomized=False) bijection & singleton sets
    rng = np.random.default_rng(7)
    p_vec = np.array([0.82, 0.12, 0.06], dtype=np.float32)
    h, w = 64, 64
    probs = torch.from_numpy(
        np.broadcast_to(p_vec[None, :, None, None], (1, 3, h, w)).copy()
    )
    y_cal = torch.from_numpy(rng.choice(3, size=(h, w), p=p_vec))
    y_tst = torch.from_numpy(rng.choice(3, size=(h, w), p=p_vec))

    cal = ConformalCalibrator(
        alpha=0.20,
        method="aps",
        randomized=False,
        fit_temperature=False,
        spatial_block_size=2,
    )
    cal.calibrate_labels(probs, y_cal)
    sets, _ = cal.predict_sets_and_intervals(probs, mode="multiclass")
    stats = empirical_conformal_coverage(sets, y_tst)
    assert stats["mean_set_size"] == pytest.approx(1.0, abs=1e-5)
    assert 0.78 <= stats["empirical_coverage"] <= 0.86

    # 3. Exact D4 mirror & rot90 equivariance on non-integer ratio (4x4 -> 6x6)
    up = CrossAttentionFeatureUpsampler(feat_dim=8, radius=2, learned=True).eval()
    feat_lr = torch.randn(1, 8, 4, 4)
    img_hr = torch.rand(1, 3, 6, 6)
    with torch.no_grad():
        base = up(feat_lr, img_hr, output_size=(6, 6))
        hflip = torch.flip(
            up(
                torch.flip(feat_lr, dims=[-1]),
                torch.flip(img_hr, dims=[-1]),
                output_size=(6, 6),
            ),
            dims=[-1],
        )
        rot = torch.rot90(
            up(
                torch.rot90(feat_lr, k=1, dims=(-2, -1)),
                torch.rot90(img_hr, k=1, dims=(-2, -1)),
                output_size=(6, 6),
            ),
            k=-1,
            dims=(-2, -1),
        )
    assert torch.allclose(base, hflip, atol=1e-5)
    assert torch.allclose(base, rot, atol=1e-5)


def test_spectral_guide_residual_count_scale_no_logit_clamping() -> None:
    from x_fuse.heads import SpatialConvMLPHead

    head = SpatialConvMLPHead(
        feat_dim=16,
        hidden_dim=16,
        out_channels=2,
        loss_fn="forward_psf_poisson",
        spectral_channels=6,
        guide_residual=True,
    )
    # Construct raw photon-count spectral cube (~1000 counts) with left/right split
    cube = torch.zeros(1, 6, 4, 8)
    cube[:, :3, :, :4] = 1200.0
    cube[:, 3:, :, 4:] = 900.0
    cube = cube + 25.0
    assert head.physics.spectral_dict is not None
    head.physics.spectral_dict.initialize_from_cube(cube)

    zero_logits = torch.zeros(1, 2, 16, 32)
    out = head.apply_guide_residual(zero_logits, cube)
    assert torch.isfinite(out).all()
    # Left and right halves must NOT both clamp to +9.21 * alpha!
    left_means = out[0, :, :, :12].mean(dim=(-2, -1))
    left_diff = float((left_means.max() - left_means.min()).abs().item())
    assert left_diff > 0.5

    # Non-spectral raw Poisson count maps (>1.0) also normalize cleanly
    head_p = SpatialConvMLPHead(
        feat_dim=16,
        hidden_dim=16,
        out_channels=2,
        loss_fn="forward_psf_poisson",
        guide_residual=True,
    )
    counts_lr = torch.tensor(
        [[[[50.0, 800.0], [100.0, 1200.0]], [[900.0, 40.0], [700.0, 60.0]]]]
    )
    out_p = head_p.apply_guide_residual(torch.zeros(1, 2, 8, 8), counts_lr)
    assert torch.isfinite(out_p).all()
    assert float(out_p[0, 0, :, 0].mean().item()) < float(
        out_p[0, 0, :, -1].mean().item()
    )


def test_small_fov_vit_auto_upscale_preserves_shape_and_boosts_tokens() -> None:
    model = XFuse(
        backbone="timm:vit_tiny_patch16_224",
        pretrained=False,
        stride=8,
        min_input_size=96,
        fusion_method="spatial_mlp",
        interpolation_mode="cross_attn",
        guide_residual=True,
        lr=1e-2,
    )
    assert model.backbone_extractor.small_fov_upscale_factor(32, 28) == 4
    assert model.backbone_extractor.small_fov_upscale_factor(128, 128) == 1

    from functools import partial

    from x_fuse.transforms import _flip_spatial, iden_partial

    small_sq = torch.rand(1, 3, 30, 26)
    h_flip = partial(_flip_spatial, (-1,))
    with torch.no_grad():
        grid = model.extract_patch_grid(small_sq)
        model_no_up = XFuse(
            backbone="timm:vit_tiny_patch16_224",
            pretrained=False,
            stride=8,
            min_input_size=None,
            fusion_method="spatial_mlp",
        )
        grid_raw = model_no_up.extract_patch_grid(small_sq)
        model.set_transforms([iden_partial, h_flip], [iden_partial, h_flip])
        tta_base = model.backbone_extractor.forward_sequential(small_sq[0])
        tta_flip = torch.flip(
            model.backbone_extractor.forward_sequential(
                torch.flip(small_sq[0], dims=[-1])
            ),
            dims=[-1],
        )
    # Without min_input_size=96, 30x26 at patch=16, stride=8 gives only 2x2=4 tokens;
    # 4x upscale to >=120x104 extracts 14x12=168 tokens and preserves H-flip TTA.
    assert grid_raw.shape[-2:] == (2, 2)
    assert grid.shape[2] >= 12 and grid.shape[3] >= 10
    assert torch.allclose(tta_base, tta_flip, atol=1e-4)

    hr_tiny = (
        np.random.default_rng(3).uniform(0.0, 1.0, size=(30, 26)).astype(np.float32)
    )
    guide_tiny = (
        np.random.default_rng(4).uniform(0.1, 0.9, size=(2, 10, 8)).astype(np.float32)
    )
    rep = model.fit_predict(
        hr_tiny, guide_tiny, epochs=2, tta_shifts=(), tta_flips=False
    )
    assert rep.probabilities.shape == (1, 3, 30, 26)
    assert rep.diagnostics.get("small_fov_upscale_factor", 1) >= 3


def test_localized_reaction_phantom_3d_and_active_discovery() -> None:
    from x_fuse import (
        VolumetricFuser,
        generate_localized_reaction_phantom_3d,
        suggest_next_slices,
    )

    ph = generate_localized_reaction_phantom_3d(
        shape=(24, 32, 32),
        downsample_factor=4,
        reaction_z_band=(9, 15),
        depth_psf_sigma=1.2,
        seed=11,
    )
    assert ph.hr_volume.shape == (24, 32, 32)
    assert ph.coarse_guide_3d.shape == (3, 24, 8, 8)
    assert int((ph.gt_labels_3d[:9] == 3).sum()) == 0
    assert int((ph.gt_labels_3d[15:] == 3).sum()) == 0
    assert int((ph.gt_labels_3d[9:15] == 3).sum()) > 50

    fuser = XFuse(
        backbone="timm:vit_tiny_patch16_224",
        pretrained=False,
        stride=8,
        fusion_method="spatial_mlp",
        loss_fn="compositional_kl",
        interpolation_mode="cross_attn",
        guide_channels=3,
        lr=1e-2,
    )
    vf = VolumetricFuser(fuser, tta_shifts=None, tta_flips=False)
    scout = {1: ph.coarse_guide_3d[:, 1], 22: ph.coarse_guide_3d[:, 22]}
    vf.fit_sparse_slices(ph.hr_volume, scout, epochs=3, depth_psf_sigma=1.2)
    rep = vf.predict_volume(
        ph.hr_volume, mode="triplane", sparse_guides=scout, pred_mode="multiclass"
    )
    rec = suggest_next_slices(rep, acquired_indices=[1, 22], k=2, min_spacing=3)
    assert len(rec.recommended_indices) == 2
    assert any(7 <= z <= 17 for z in rec.recommended_indices)


def test_edxia_csv_adapter_and_cli_v11_fuse_3d(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from benchmarks.dataset_adapters import load_zenodo_correlative_sample
    from x_fuse.cli import main as cli_main

    lc3_dir = tmp_path / "edxia_cement_sem_eds" / "edxia_examples" / "LC3"
    lc3_dir.mkdir(parents=True)
    rng = np.random.default_rng(21)
    bse = rng.uniform(20.0, 220.0, size=(32, 32)).astype(np.float32)
    ca = np.full((32, 32), 35.0, dtype=np.float32)
    si = np.full((32, 32), 18.0, dtype=np.float32)
    s_map = np.full((32, 32), 2.5, dtype=np.float32)
    np.savetxt(lc3_dir / "BSE Wt_.csv", bse, delimiter=",", fmt="%.3f")
    # Include Bruker Esprit-style trailing commas on element CSV rows
    ca_lines = "\n".join(",".join(f"{v:.3f}" for v in row) + "," for row in ca) + "\n"
    (lc3_dir / "Ca Wt_.csv").write_text(ca_lines, encoding="utf-8")
    np.savetxt(lc3_dir / "Si Wt_.csv", si, delimiter=",", fmt="%.3f")
    np.savetxt(lc3_dir / "S Wt_.csv", s_map, delimiter=",", fmt="%.3f")
    (lc3_dir / "pointsWt.csv").write_text(
        "id,Ca,Si,S\nx1,35.0,18.0,2.5\n", encoding="utf-8"
    )

    sample = load_zenodo_correlative_sample(
        "edxia_cement_sem_eds", dest_root=tmp_path, crop_clean_roi=False
    )
    assert sample.channel_names == ["Ca", "Si", "S"]
    # Verify 'S' matched 'S Wt_.csv' (2.5) and NOT 'BSE Wt_.csv' or 'Si Wt_.csv'!
    assert np.allclose(sample.guide_map[2], 2.5, atol=1e-2)

    # CLI 2D fuse with v1.0.0 flags
    hr_p = tmp_path / "hr2d.npy"
    g_p = tmp_path / "g2d.npy"
    out_2d = tmp_path / "out2d.h5"
    np.save(hr_p, bse)
    np.save(g_p, np.stack([ca[:8, :8], si[:8, :8]], axis=0))
    rc2 = cli_main(
        [
            "fuse",
            "--hr",
            str(hr_p),
            "--guide",
            str(g_p),
            "--output",
            str(out_2d),
            "--backbone",
            "timm:vit_tiny_patch16_224",
            "--no-pretrained",
            "--stride",
            "8",
            "--epochs",
            "2",
            "--psf-profile",
            "moffat",
            "--conformal-alpha",
            "0.20",
            "--conformal-method",
            "raps",
            "--conformal-block-size",
            "2",
            "--guide-residual",
        ]
    )
    assert rc2 == 0 and out_2d.exists()
    j2 = json.loads(capsys.readouterr().out)
    assert j2["has_conformal_sets"] is True

    # CLI 3D fuse-3d with active slice recommendation
    vol_p = tmp_path / "vol3d.npy"
    s0_p = tmp_path / "slice1.npy"
    s1_p = tmp_path / "slice6.npy"
    out_3d = tmp_path / "out3d.h5"
    np.save(vol_p, rng.uniform(0.0, 1.0, size=(8, 32, 32)).astype(np.float32))
    np.save(s0_p, rng.uniform(0.1, 0.8, size=(2, 8, 8)).astype(np.float32))
    np.save(s1_p, rng.uniform(0.1, 0.8, size=(2, 8, 8)).astype(np.float32))
    rc3 = cli_main(
        [
            "fuse-3d",
            "--hr-volume",
            str(vol_p),
            "--guide-slice",
            f"1:{s0_p}",
            f"6:{s1_p}",
            "--output",
            str(out_3d),
            "--backbone",
            "timm:vit_tiny_patch16_224",
            "--no-pretrained",
            "--stride",
            "8",
            "--epochs",
            "2",
            "--suggest-next",
            "2",
        ]
    )
    assert rc3 == 0 and out_3d.exists()
    j3 = json.loads(capsys.readouterr().out)
    assert len(j3["suggested_next_slices"]) == 2


def test_export_latex_tables_and_unequal_spectral_flux(tmp_path: Path) -> None:
    from benchmarks.export_latex_tables import export_all_tables
    from x_fuse.datasets import make_experimental_holdout

    # 1. Verify core_halo holdout profile
    hr_m = np.ones((32, 32), dtype=np.float32)
    gt_m = np.zeros((2, 32, 32), dtype=np.float32)
    gt_m[0, :, :16] = 1.0
    gt_m[1, :, 16:] = 1.0
    _, c_halo, _ = make_experimental_holdout(
        hr_m, gt_m, factor=4, psf_sigma=1.5, beam_profile="core_halo"
    )
    assert c_halo.shape == (2, 8, 8)

    # 2. Verify unequal-cross-section 3D Poisson spectral unmixing preserves weak edge
    cube_unequal = np.full((6, 8, 8), 5.0, dtype=np.float32)
    cube_unequal[:3, :, :4] += 1600.0  # Strong edge (e.g. Cu-L2,3)
    cube_unequal[3:, :, 4:] += 220.0  # 7x weaker edge (e.g. Zn-L2,3)
    hr_step = np.zeros((32, 32), dtype=np.float32)
    hr_step[:, :16] = 0.2
    hr_step[:, 16:] = 0.85
    xf_sp = XFuse(
        backbone="timm:vit_tiny_patch16_224",
        pretrained=False,
        stride=8,
        loss_fn="forward_psf_poisson",
        fusion_method="spatial_mlp",
        interpolation_mode="cross_attn",
        spectral_channels=6,
        guide_channels=2,
        guide_residual=True,
        lr=1e-2,
    )
    rep_sp = xf_sp.fit_predict(
        hr_step,
        cube_unequal,
        epochs=6,
        mode="multilabel",
        tta_shifts=(),
        tta_flips=False,
        spectral_unmix=True,
        n_components=2,
    )
    # Weak component must NOT get squashed below 0.5 by global_max prior matching!
    ch_maxes = rep_sp.probabilities[0].amax(dim=(-2, -1))
    assert float(ch_maxes.min().item()) > 0.65

    # 3. Verify booktabs LaTeX table exporter on synthetic JSON fixtures
    d_json = tmp_path / "design.json"
    r_json = tmp_path / "referee.json"
    d_json.write_text(
        json.dumps(
            {
                "stage_a_upsampler": {
                    "lamellar_8x": {
                        "scalar_dilated_jbu": {"mIoU": {"mean": 0.70, "std": 0.02}},
                        "learned_spin2_subtoken": {"mIoU": {"mean": 0.75, "std": 0.01}},
                    }
                },
                "stage_c_volumetric": {
                    "volumetric_variants": {
                        "sparse_25d_baseline": {"mIoU_3d": {"mean": 0.71, "std": 0.03}},
                        "depth_psf_deconv": {"mIoU_3d": {"mean": 0.79, "std": 0.02}},
                    },
                    "localized_reaction_active_curve_z48": {
                        "active_steering": {
                            "4": {
                                "mIoU_3d": {"mean": 0.78, "std": 0.02},
                                "phase_c_iou": {"mean": 0.62, "std": 0.04},
                            }
                        },
                        "equispaced": {
                            "4": {
                                "mIoU_3d": {"mean": 0.68, "std": 0.03},
                                "phase_c_iou": {"mean": 0.15, "std": 0.05},
                            }
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    r_json.write_text(json.dumps({}), encoding="utf-8")
    written = export_all_tables(d_json, r_json, tmp_path / "tables")
    assert "manuscript_tables.tex" in written
    tex_body = written["manuscript_tables.tex"].read_text(encoding="utf-8")
    assert "\\toprule" in tex_body and "\\textbf{" in tex_body


def test_savu_xzy_out_of_core_and_multipeak_diad_xrdct_loader(tmp_path: Path) -> None:
    import h5py
    from scipy.ndimage import rotate

    from x_fuse import (
        aggregate_peak_maps,
        detrend_radial_cupping,
        load_diad_xrdct,
        load_xct_subvolume,
        load_xrdct_peak_maps,
        remove_polar_rings_conservative,
    )

    rng = np.random.default_rng(101)
    raw_xzy = rng.normal(size=(20, 16, 24)).astype(np.float32)
    h5_p = tmp_path / "savu_test.h5"
    with h5py.File(h5_p, "w") as hf:
        grp = hf.create_group("4-AstraReconGpu-tomo")
        grp.create_dataset("data", data=raw_xzy)

    zyx_sl = (slice(2, 12), slice(4, 20), slice(3, 17))
    sub = load_xct_subvolume(h5_p, recon="Astra", zyx_slices=zyx_sl, stride=2)
    ref = np.transpose(raw_xzy, (1, 2, 0))[2:12:2, 4:20:2, 3:17:2]
    assert np.allclose(sub, ref)

    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, 24, dtype=np.float32),
        np.linspace(-1.0, 1.0, 24, dtype=np.float32),
        indexing="ij",
    )
    true_zn = np.exp(-((yy - 0.35) ** 2 + (xx + 0.30) ** 2) / 0.12).astype(np.float32)
    true_na = (1.0 - 0.6 * true_zn).astype(np.float32)
    raw_50_zn = np.fliplr(rotate(true_zn, angle=47.5, reshape=False, order=1))
    raw_50_na = np.fliplr(rotate(true_na, angle=47.5, reshape=False, order=1))

    nxs_dir = tmp_path / "nxs_scans"
    nxs_dir.mkdir()
    for sid, zn_m, na_m in (
        (43345, raw_50_zn, raw_50_na),
        (43324, np.fliplr(true_zn), np.fliplr(true_na)),
    ):
        with h5py.File(nxs_dir / f"output_{sid}_zn.nxs", "w") as hf:
            ent = hf.create_group("entry")
            ent.create_dataset("peak at q~1.655", data=zn_m[None, ...] * 0.6)
            ent.create_dataset("peak at q~2.110", data=zn_m[None, ...] * 0.4)
        with h5py.File(nxs_dir / f"output_{sid}_na.nxs", "w") as hf:
            ent = hf.create_group("entry")
            ent.create_dataset("peak at q~1.651", data=na_m[None, ...])

    peaks_zn = load_xrdct_peak_maps(nxs_dir / "output_43345_zn.nxs", flip_x=True)
    near = aggregate_peak_maps(peaks_zn, mode="nearest_q", target_q=1.656, q_tol=0.02)
    assert near.shape == (24, 24)

    loaded_corr = load_diad_xrdct(
        nxs_dir,
        phases=["Zn", "Na"],
        scan_ids=[43345],
        aggregation="all_sum",
        correct_motor_offset_deg=47.5,
    )
    loaded_uncorr = load_diad_xrdct(
        nxs_dir,
        phases=["Zn", "Na"],
        scan_ids=[43345],
        aggregation="all_sum",
        correct_motor_offset_deg=0.0,
    )
    r_corr = float(np.corrcoef(loaded_corr["Zn"][0].ravel(), true_zn.ravel())[0, 1])
    r_uncorr = float(np.corrcoef(loaded_uncorr["Zn"][0].ravel(), true_zn.ravel())[0, 1])
    assert r_corr > 0.90 and r_corr > r_uncorr + 0.25

    bundle_p = tmp_path / "cleaned_xrdct.npz"
    np.savez(
        bundle_p,
        Na=true_na[None, ...],
        Zn=true_zn[None, ...],
        zn_sigma2=np.full((1, 24, 24), 0.01, dtype=np.float32),
    )
    b_out = load_diad_xrdct(bundle_p, include_auxiliary=True)
    assert "Zn" in b_out and "zn_sigma2" in b_out

    yy64, xx64 = np.meshgrid(
        np.linspace(-1.0, 1.0, 64, dtype=np.float32),
        np.linspace(-1.0, 1.0, 64, dtype=np.float32),
        indexing="ij",
    )
    r64 = np.hypot(yy64, xx64)
    clean_xct = np.where(np.hypot(yy64 - 0.1, xx64 + 0.1) < 0.35, 0.2, 0.8).astype(
        np.float32
    )
    cupped = clean_xct + 0.45 * (r64**4) + 0.20 * yy64
    detrended = detrend_radial_cupping(cupped)
    assert np.std(detrended - clean_xct) < 0.35 * np.std(cupped - clean_xct)

    ringed = clean_xct + 0.12 * np.sin(r64 * 28.0).astype(np.float32)
    derung = remove_polar_rings_conservative(ringed, radial_sigma=2.0)
    assert np.mean((derung - clean_xct) ** 2) < np.mean((ringed - clean_xct) ** 2)


def test_fringe_guarded_pore_mask_and_openimpala_amrex_export(tmp_path: Path) -> None:
    from scipy.ndimage import laplace

    from x_fuse import (
        VolumetricFuser,
        XFuse,
        export_for_openimpala,
        load_uncertainty_report,
        save_uncertainty_report,
        segment_pores_otsu,
    )

    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, 48, dtype=np.float32),
        np.linspace(-1.0, 1.0, 48, dtype=np.float32),
        indexing="ij",
    )
    pore_gt = (np.sin(yy * 7.0) * np.cos(xx * 7.0)) > 0.25
    base_i = np.where(pore_gt, 0.15, 0.85).astype(np.float32)
    rng = np.random.default_rng(7)
    fringe_i = (
        base_i
        - 0.32 * laplace(base_i)
        + rng.normal(0.0, 0.22, size=base_i.shape).astype(np.float32)
    )
    m_raw = segment_pores_otsu(fringe_i, tie_homogeneity_alpha_vox2=0.0)
    m_guard = segment_pores_otsu(fringe_i, tie_homogeneity_alpha_vox2=0.8)
    iou_raw = float((m_raw & pore_gt).sum()) / float(max(1, (m_raw | pore_gt).sum()))
    iou_guard = float((m_guard & pore_gt).sum()) / float(
        max(1, (m_guard | pore_gt).sum())
    )
    assert iou_guard > iou_raw

    solid_zn = ((~pore_gt) & (xx > 0.0)).astype(np.float32)
    solid_na = ((~pore_gt) & (xx <= 0.0)).astype(np.float32)
    guide_lr = (
        np.stack([solid_na, solid_zn], axis=0)
        .reshape(2, 12, 4, 12, 4)
        .mean(axis=(2, 4))
    )
    xf = XFuse(
        backbone="timm:vit_tiny_patch16_224",
        pretrained=False,
        stride=8,
        loss_fn="compositional_kl",
        fusion_method="spatial_mlp",
        interpolation_mode="cross_attn",
        pore_mask=pore_gt,
        lr=1e-2,
    )
    rep = xf.fit_predict(
        base_i, guide_lr, epochs=4, mode="multiclass", tta_shifts=(), tta_flips=False
    )
    fg_on_pores = float(rep.probabilities[0, 1:, pore_gt].max().item())
    assert fg_on_pores < 1e-6
    assert int(rep.hard_mask[0, pore_gt].max().item()) == 0

    npz_p = save_uncertainty_report(
        rep, tmp_path / "rep2d.npz", channel_names=["pore", "Na", "Zn"]
    )
    reloaded = load_uncertainty_report(npz_p)
    assert reloaded["metadata"]["channel_names"] == ["pore", "Na", "Zn"]
    assert np.allclose(reloaded["probabilities"], rep.probabilities[0].cpu().numpy())

    vol_hr = np.repeat(base_i[None, :32, :32], 8, axis=0)
    vol_pm = np.repeat(pore_gt[None, :32, :32], 8, axis=0)
    g3_slice = guide_lr[:, :8, :8]
    xf3 = XFuse(
        backbone="timm:vit_tiny_patch16_224",
        pretrained=False,
        stride=8,
        loss_fn="compositional_kl",
        fusion_method="spatial_mlp",
        conformal_alpha=0.20,
        conformal_method="raps",
        dtype=torch.float32,
    )
    vf = VolumetricFuser(xf3, tta_shifts=None, tta_flips=False)
    vf.fit_sparse_slices(vol_hr, {1: g3_slice, 6: g3_slice}, epochs=2, pore_mask=vol_pm)
    vrep = vf.predict_volume(
        vol_hr, mode="2.5d_slab", pred_mode="multiclass", pore_mask=vol_pm
    )
    assert float(vrep.probabilities[1:, vol_pm].max().item()) < 1e-6

    tif_p, inp_p = export_for_openimpala(
        vrep, tmp_path / "impala", name="zn13x_test", active_phase=0, direction="z"
    )
    assert tif_p.exists() and inp_p.exists()
    assert (tmp_path / "impala" / "zn13x_test_phase0_lower.tif").exists()
    assert (tmp_path / "impala" / "zn13x_test_phase0_upper.tif").exists()
    deck = inp_p.read_text(encoding="utf-8")
    assert "amr.n_cell = 32 32 8" in deck
    assert "amr.blocking_factor = 8" in deck
    assert 'torteff.direction = "z"' in deck


def test_voronoi_active_slice_steering_discovery_and_bulk_coverage() -> None:
    from x_fuse import VolumetricUncertaintyReport, suggest_next_slices

    z_dim, y_dim, x_dim = 48, 16, 16
    probs = torch.full((3, z_dim, y_dim, x_dim), 1.0 / 3.0)
    hard = torch.zeros((z_dim, y_dim, x_dim), dtype=torch.int64)
    ent = torch.full((z_dim, y_dim, x_dim), 0.15)
    epi = torch.full((z_dim, y_dim, x_dim), 0.05)
    ent[34:38] = 0.92
    epi[34:38] = 0.85
    rep = VolumetricUncertaintyReport(
        probabilities=probs,
        hard_mask=hard,
        aleatoric_entropy=ent,
        triplane_variance=epi,
        physical_residual=torch.zeros_like(ent),
        combined_uncertainty=0.5 * (ent + epi),
        z_slice_consistency=0.95,
    )

    plan = suggest_next_slices(rep, acquired_indices=[4, 43], k=4, min_spacing=4)
    recs = plan.recommended_indices
    assert len(recs) == 4
    assert 33 <= recs[0] <= 38
    bulk_picks = [z for z in recs[1:] if 10 <= z <= 29]
    assert len(bulk_picks) >= 2
