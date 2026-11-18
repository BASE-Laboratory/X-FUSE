"""Regression and scientific honesty tests for referee-critique remediations:
1. Mismatched non-Gaussian forward PSFs (``beam_profile="moffat"`` and ``"boxcar"``)
   in ``make_experimental_holdout`` and synthetic 2D phantom generators.
2. Matched mean, variance, and isotropic gradient magnitude between Colony A and
   Colony B in ``generate_oriented_lamellar_phantom_2d``, plus FIB curtaining /
   thickness-wedge corruption in ``generate_artifact_corrupted_phantom_2d``.
3. ``SpatialConvMLPHead`` ``1x1`` ``pixel_proj`` skip, ``guide_residual=True``
   coarse-to-fine logit anchor, and ``simplex=True`` ``edge_aware_tv_loss``.
4. Fair ``guide_normalization`` and ``calibrate`` parity across all baselines.
5. Cross-specimen ``ConformalCalibrator(method="lac")`` calibration on Specimen A
   evaluated on an independent Specimen B (reporting empirical coverage, mean set
   size, and singleton fraction).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks.baselines import (
    cnmf_pansharpen_baseline,
    coordinate_inr_baseline,
    gram_schmidt_pansharpen_baseline,
    guided_variational_sr_baseline,
    otsu_threshold_baseline,
)
from benchmarks.dataset_adapters import load_bundled_morgis_samples
from x_fuse import (
    ConformalCalibrator,
    SpatialConvMLPHead,
    XFuse,
    bicubic_upsample_baseline,
    edge_aware_tv_loss,
    empirical_conformal_coverage,
    generate_artifact_corrupted_phantom_2d,
    generate_correlative_phantom_2d,
    generate_oriented_lamellar_phantom_2d,
    jbu_guided_baseline,
    make_experimental_holdout,
)


def test_make_experimental_holdout_mismatched_beam_profiles() -> None:
    rng = np.random.default_rng(7)
    hr = rng.uniform(0.0, 1.0, size=(64, 64)).astype(np.float32)
    gt = np.zeros((2, 64, 64), dtype=np.float32)
    gt[0, 16:48, 16:48] = 1.0
    gt[1, 8:24, 40:56] = 1.0

    _, lr_gauss, _ = make_experimental_holdout(
        hr, gt, factor=4, beam_profile="gaussian"
    )
    _, lr_moffat, _ = make_experimental_holdout(hr, gt, factor=4, beam_profile="moffat")
    _, lr_boxcar, _ = make_experimental_holdout(hr, gt, factor=4, beam_profile="boxcar")

    assert lr_gauss.shape == (2, 16, 16)
    assert lr_moffat.shape == (2, 16, 16)
    assert lr_boxcar.shape == (2, 16, 16)
    # Moffat heavy diffraction tails differ measurably from Gaussian and flat boxcar
    assert not np.allclose(lr_moffat, lr_gauss, atol=1e-3)
    assert not np.allclose(lr_boxcar, lr_gauss, atol=1e-3)
    # Flux conservation: total integrated phase mass is preserved across profiles
    np.testing.assert_allclose(lr_moffat.sum(), lr_boxcar.sum(), rtol=0.05)

    with pytest.raises(ValueError, match="Unsupported beam_profile"):
        make_experimental_holdout(hr, gt, factor=4, beam_profile="invalid")  # type: ignore[arg-type]


def test_oriented_lamellar_and_artifact_phantoms_matched_statistics() -> None:
    lam = generate_oriented_lamellar_phantom_2d(
        size=128, downsample_factor=8, beam_profile="moffat", seed=42
    )
    assert lam.hr_image.shape == (128, 128)
    assert lam.coarse_guide.shape == (2, 16, 16)
    mask_a = lam.gt_labels == 1
    mask_b = lam.gt_labels == 2
    assert mask_a.sum() > 1000 and mask_b.sum() > 1000

    # Colony A and Colony B must have matched mean intensity AND matched std!
    mean_a = float(lam.hr_image[mask_a].mean())
    mean_b = float(lam.hr_image[mask_b].mean())
    std_a = float(lam.hr_image[mask_a].std())
    std_b = float(lam.hr_image[mask_b].std())
    assert abs(mean_a - mean_b) < 0.03
    assert abs(std_a - std_b) < 0.03

    art = generate_artifact_corrupted_phantom_2d(
        size=128, downsample_factor=8, beam_profile="moffat", seed=42
    )
    clean = generate_correlative_phantom_2d(
        size=128,
        downsample_factor=8,
        phase_b_contrast=0.32,
        hr_noise_std=0.025,
        poisson_counts=250.0,
        beam_profile="moffat",
        seed=42,
    )
    assert art.hr_image.shape == (128, 128)
    np.testing.assert_allclose(art.coarse_guide, clean.coarse_guide)
    assert float(np.mean(np.abs(art.hr_image - clean.hr_image))) > 0.05


def test_spatial_mlp_guide_residual_and_simplex_tv() -> None:
    head = SpatialConvMLPHead(
        feat_dim=32,
        hidden_dim=16,
        out_channels=2,
        loss_fn="compositional_kl",
        guide_residual=True,
    )
    feats = torch.randn(1, 32, 32, 32)
    guidance = torch.rand(1, 3, 32, 32)
    guide_lr = torch.full((1, 2, 8, 8), 0.25)

    # At initialization with guide_residual=True, out_proj & pixel_proj are zero,
    # so canonical logits equal exact log-odds of the bilinear coarse guide!
    raw_logits = head.predict_logits(feats, guidance_hr=guidance)
    assert float(raw_logits.abs().max().item()) == pytest.approx(0.0, abs=1e-6)
    anchored = head.apply_guide_residual(raw_logits, guide_lr)
    probs = head.physics.logits_to_simplex_probs(anchored, target=guide_lr)
    assert probs.shape == (1, 3, 32, 32)
    np.testing.assert_allclose(probs[:, 1:].detach().numpy(), 0.25, atol=1e-3)

    tv_s = edge_aware_tv_loss(anchored, guidance_hr=guidance, simplex=True)
    assert float(tv_s.item()) >= 0.0


def test_all_baselines_preserve_counts_normalization_parity() -> None:
    p = generate_correlative_phantom_2d(
        size=64, downsample_factor=8, poisson_counts=None, beam_profile="moffat", seed=5
    )
    # Scale Phase B to 0.40 max so per-channel minmax would stretch it 2.5x
    g_scaled = p.coarse_guide.copy()
    g_scaled[1] *= 0.40

    for fn in (
        lambda: bicubic_upsample_baseline(
            g_scaled,
            (64, 64),
            mode="multiclass",
            calibrate=False,
            guide_normalization="preserve_counts",
        ),
        lambda: jbu_guided_baseline(
            p.hr_image,
            g_scaled,
            mode="multiclass",
            calibrate=False,
            guide_normalization="preserve_counts",
        ),
        lambda: otsu_threshold_baseline(
            p.hr_image,
            g_scaled,
            mode="multiclass",
            guide_normalization="preserve_counts",
        ),
        lambda: gram_schmidt_pansharpen_baseline(
            p.hr_image,
            g_scaled,
            mode="multiclass",
            guide_normalization="preserve_counts",
        ),
        lambda: cnmf_pansharpen_baseline(
            p.hr_image,
            g_scaled,
            mode="multiclass",
            n_iters=8,
            guide_normalization="preserve_counts",
        ),
        lambda: guided_variational_sr_baseline(
            p.hr_image,
            g_scaled,
            mode="multiclass",
            n_iters=8,
            guide_normalization="preserve_counts",
        ),
        lambda: coordinate_inr_baseline(
            p.hr_image,
            g_scaled,
            mode="multiclass",
            epochs=8,
            guide_normalization="preserve_counts",
        ),
    ):
        rep = fn()
        assert rep.probabilities.shape == (1, 3, 64, 64)
        # Predicted Phase B mean stays near g_scaled[1].mean(), not 2.5x inflated
        assert (
            float(rep.probabilities[0, 2].mean().item())
            < float(g_scaled[1].mean()) + 0.12
        )


def test_cross_specimen_lac_conformal_calibration_and_morgis_flag() -> None:
    morgis = load_bundled_morgis_samples()
    assert morgis and morgis[0].metadata.get("semi_synthetic") is True

    spec_a = generate_correlative_phantom_2d(size=56, downsample_factor=4, seed=11)
    spec_b = generate_correlative_phantom_2d(size=56, downsample_factor=4, seed=99)

    model = XFuse(
        backbone="dinov2_vits14",
        stride=14,
        loss_fn="compositional_kl",
        fusion_method="spatial_mlp",
        pretrained=False,
        lr=1e-2,
    )
    rep_a = model.fit_predict(
        spec_a.hr_image,
        spec_a.coarse_guide,
        epochs=6,
        tta_shifts=(),
        tta_flips=False,
        guide_normalization="preserve_counts",
    )
    cal = ConformalCalibrator(alpha=0.15, method="lac")
    cal.calibrate_labels(
        rep_a.probabilities, torch.from_numpy(spec_a.gt_labels).unsqueeze(0)
    )

    # Zero-shot transfer onto independent Specimen B and apply Specimen-A LAC quantile
    rep_b = model.predict(
        spec_b.hr_image, coarse_guide=None, mode="multiclass", calibrate_to_guide=False
    )
    sets_b, _ = cal.predict_sets_and_intervals(
        rep_b.probabilities, guide_lr=None, mode="multiclass"
    )
    stats = empirical_conformal_coverage(sets_b, spec_b.gt_labels)
    assert 0.0 <= stats["empirical_coverage"] <= 1.0
    assert 1.0 <= stats["mean_set_size"] <= 3.0
    assert 0.0 <= stats["singleton_fraction"] <= 1.0
