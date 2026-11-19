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
