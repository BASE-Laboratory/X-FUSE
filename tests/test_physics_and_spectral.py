"""Unit and property tests for quantitative forward physics (`x_fuse.physics`),
count/simplex preserving guide normalization, and end-to-end 3D spectral cube
NMF unmixing (`SpectralUnmixingDictionary` + `XFuse(..., spectral_unmix=True)`).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from x_fuse import (
    PhysicsForwardModel,
    SpectralUnmixingDictionary,
    XFuse,
    compute_segmentation_metrics,
    generate_correlative_phantom_2d,
    generate_spectral_cube_phantom_2d,
    guide_to_tensor,
    poisson_deviance_loss,
)


def test_poisson_deviance_nonnegative_and_zero_at_optimum() -> None:
    rng = torch.Generator().manual_seed(0)
    target = torch.poisson(torch.rand((1, 2, 16, 16), generator=rng) * 30.0)
    exact_loss = poisson_deviance_loss(target.clamp(min=1e-6), target)
    assert float(exact_loss.item()) == pytest.approx(0.0, abs=1e-4)

    perturbed = (target + 5.5).clamp(min=0.1)
    pos_loss = poisson_deviance_loss(perturbed, target)
    assert float(pos_loss.item()) > 0.05


def test_guide_to_tensor_normalization_modes() -> None:
    raw = np.array(
        [
            [[10.0, 30.0], [0.0, 20.0]],
            [[30.0, 10.0], [40.0, 20.0]],
        ],
        dtype=np.float32,
    )
    t_counts = guide_to_tensor(raw, normalization="preserve_counts")
    assert t_counts.shape == (1, 2, 2, 2)
    assert float(t_counts.max().item()) == pytest.approx(40.0)

    t_gmax = guide_to_tensor(raw, normalization="global_max")
    assert float(t_gmax.max().item()) == pytest.approx(1.0)
    assert float(t_gmax[0, 0, 0, 1].item()) == pytest.approx(0.75)

    t_simplex = guide_to_tensor(raw, normalization="simplex")
    col_sums = t_simplex.sum(dim=1)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-6)


def test_spectral_unmixing_dictionary_recovers_endmembers() -> None:
    ph = generate_spectral_cube_phantom_2d(
        size=48,
        downsample_factor=4,
        n_spectral_bins=20,
        mean_counts=200.0,
        background_counts=0.1,
        seed=12,
    )
    spec_dict = SpectralUnmixingDictionary(n_components=2, spectral_channels=20)
    cube_t = torch.from_numpy(ph.spectral_cube_lr)
    abund = spec_dict.initialize_from_cube(cube_t, n_iters=60, seed=0)
    assert abund.shape == (1, 2, 12, 12)
    assert float(abund.min().item()) >= 0.0

    learned = spec_dict.spectra().detach().numpy()
    assert np.allclose(learned.sum(axis=-1), 1.0, atol=1e-5)

    # Cosine similarity against ground-truth spectra (up to 2x2 permutation)
    gt = ph.gt_spectra
    gt_norm = gt / np.linalg.norm(gt, axis=-1, keepdims=True)
    lr_norm = learned / np.linalg.norm(learned, axis=-1, keepdims=True)
    sim = lr_norm @ gt_norm.T
    best_match = max(
        0.5 * (sim[0, 0] + sim[1, 1]),
        0.5 * (sim[0, 1] + sim[1, 0]),
    )
    assert best_match > 0.90

    recon = spec_dict(abund)
    assert recon.shape == (1, 20, 12, 12)


def test_compositional_kl_enforces_exact_simplex_conservation() -> None:
    ph = generate_correlative_phantom_2d(size=48, downsample_factor=4, seed=21)
    fwd = PhysicsForwardModel(n_channels=2, loss_fn="compositional_kl", learn_psf=True)
    logits = torch.randn((1, 2, 48, 48), requires_grad=True)
    guide_t = guide_to_tensor(ph.coarse_guide, normalization="minmax")

    simplex_probs = fwd.logits_to_simplex_probs(logits, target=guide_t)
    assert simplex_probs.shape == (1, 3, 48, 48)
    max_dev = float((simplex_probs.sum(dim=1) - 1.0).abs().max().item())
    assert max_dev < 1e-6

    loss = fwd.compute_loss_from_logits(logits, guide_t)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

    model = XFuse(
        backbone="timm:vit_tiny_patch16_224",
        pretrained=False,
        stride=8,
        dtype=torch.float32,
        loss_fn="compositional_kl",
        fusion_method="spatial_mlp",
        lr=1e-2,
    )
    rep = model.fit_predict(
        ph.hr_image, ph.coarse_guide, epochs=5, tta_shifts=(), tta_flips=False
    )
    prob_sum = rep.probabilities.sum(dim=1)
    assert float((prob_sum - 1.0).abs().max().item()) < 1e-5


def test_end_to_end_xfuse_poisson_and_3d_spectral_cube_unmixing() -> None:
    ph_spec = generate_spectral_cube_phantom_2d(
        size=48,
        downsample_factor=4,
        n_spectral_bins=16,
        mean_counts=60.0,
        seed=42,
    )
    model = XFuse(
        backbone="timm:vit_tiny_patch16_224",
        pretrained=False,
        stride=8,
        dtype=torch.float32,
        loss_fn="forward_psf_poisson",
        interpolation_mode="cross_attn",
        fusion_method="spatial_mlp",
        spectral_channels=16,
        guide_channels=2,
        lr=1e-2,
    )
    rep = model.fit_predict(
        ph_spec.hr_image,
        ph_spec.spectral_cube_lr,
        epochs=10,
        mode="multiclass",
        tta_shifts=(),
        tta_flips=False,
        spectral_unmix=True,
        n_components=2,
    )
    assert rep.probabilities.shape == (1, 3, 48, 48)
    assert rep.diagnostics.get("spectral_unmixing_components") == 2
    assert "poisson_flux_scale" in rep.diagnostics

    pred = rep.hard_mask[0].cpu().numpy()
    swapped = pred.copy()
    swapped[pred == 1] = 2
    swapped[pred == 2] = 1
    iou = max(
        compute_segmentation_metrics(pred, ph_spec.gt_labels)["iou"],
        compute_segmentation_metrics(swapped, ph_spec.gt_labels)["iou"],
    )
    assert iou > 0.35
