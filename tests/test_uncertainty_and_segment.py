"""Unit tests for tri-part uncertainty quantification, calibration metrics,
sign-aligned PCA+Otsu, classical/ViT baselines, SAM2 certainty prompting, and
end-to-end CPU CorrelativeFuser execution.
"""

import numpy as np
import pytest
import torch

from x_fuse.fusion import CorrelativeFuser
from x_fuse.segment import (
    bicubic_upsample_baseline,
    compute_segmentation_metrics,
    dino_kmeans_baseline,
    jbu_guided_baseline,
    pca_otsu_segment,
    refine_with_sam2,
)
from x_fuse.synthetic import generate_correlative_phantom_2d
from x_fuse.transforms import build_default_tta, normalize_structural
from x_fuse.uncertainty import (
    WelfordAccumulator,
    brier_score,
    build_uncertainty_report,
    compute_bald_mutual_information,
    compute_physical_residual,
    compute_posterior_uncertainty,
    expected_calibration_error,
    match_coarse_priors,
    sparsification_curve,
)
from x_fuse.utils import guide_to_tensor


def test_welford_accumulator_matches_torch_var():
    torch.manual_seed(0)
    samples = torch.randn(9, 1, 4, 8, 8)
    acc = WelfordAccumulator()
    for i in range(samples.shape[0]):
        acc.update(samples[i])

    mean_p, var_p = acc.finalize(unbiased=False)
    mean_u, var_u = acc.finalize(unbiased=True)

    assert torch.allclose(mean_p, samples.mean(dim=0), atol=1e-6)
    assert torch.allclose(var_p, samples.var(dim=0, unbiased=False), atol=1e-6)
    assert torch.allclose(var_u, samples.var(dim=0, unbiased=True), atol=1e-6)


def test_posterior_entropy_and_bald_bounds():
    uniform = torch.full((1, 3, 8, 8), 1.0 / 3.0)
    ent_u, margin_u = compute_posterior_uncertainty(uniform, mode="multiclass")
    assert torch.allclose(ent_u, torch.ones_like(ent_u), atol=1e-4)
    assert torch.allclose(margin_u, torch.zeros_like(margin_u), atol=1e-4)

    peaked = torch.zeros((1, 3, 8, 8))
    peaked[:, 0] = 0.999
    peaked[:, 1] = 0.0005
    peaked[:, 2] = 0.0005
    ent_p, margin_p = compute_posterior_uncertainty(peaked, mode="multiclass")
    assert float(ent_p.max()) < 0.02
    assert float(margin_p.min()) > 0.99

    # BALD should be ~0 when all samples agree, and > 0 when samples disagree
    agree = peaked.unsqueeze(0).expand(4, -1, -1, -1, -1)
    assert float(compute_bald_mutual_information(agree).max()) < 1e-5

    s1 = torch.tensor([[[[[0.95]], [[0.05]]]]])
    s2 = torch.tensor([[[[[0.05]], [[0.95]]]]])
    disagree = torch.cat([s1, s2], dim=0)
    assert float(compute_bald_mutual_information(disagree).item()) > 0.5


def test_match_coarse_priors_and_invisible_phase_residual():
    """Verify prior matching recovers global fraction and physical residual flags
    a chemically distinct phase that has zero structural contrast in I_HR."""
    phantom = generate_correlative_phantom_2d(
        size=64, downsample_factor=8, phase_b_contrast=0.0, seed=11
    )
    guide_t = torch.from_numpy(phantom.coarse_guide).unsqueeze(0)  # (1, 2, 8, 8)

    # Suppose a structural predictor sees Phase A perfectly, but predicts 0 everywhere
    # for Phase B because Phase B has zero contrast against background in I_HR!
    pred_hr = torch.zeros(1, 2, 64, 64)
    pred_hr[0, 0] = torch.from_numpy(phantom.gt_phase_channels[0])
    pred_hr[0, 1] = 0.0

    res_hr, res_lr = compute_physical_residual(pred_hr, guide_t)
    # Channel 1 coarse residual must vastly exceed Channel 0 residual (>3x), and HR
    # residual inside Phase B must substantially exceed background!
    assert float(res_lr[:, 1].mean()) > 3.0 * float(res_lr[:, 0].mean())
    mask_b = torch.from_numpy(phantom.gt_phase_channels[1] > 0.5)
    mask_bg = torch.from_numpy(phantom.gt_labels == 0)
    assert float(res_hr[0][mask_b].mean()) > float(res_hr[0][mask_bg].mean()) + 0.08

    # Also verify match_coarse_priors shifts uncalibrated logits to match coarse mean
    raw_logits = torch.randn(1, 2, 32, 32) - 3.0
    calibrated = match_coarse_priors(raw_logits, guide_t, mode="multilabel")
    target_frac = guide_t.mean(dim=(-2, -1))
    pred_frac = calibrated.mean(dim=(-2, -1))
    assert torch.allclose(pred_frac, target_frac, atol=0.03)


def test_ece_brier_and_sparsification_curve():
    rng = np.random.default_rng(3)
    gt = (rng.random((32, 32)) > 0.6).astype(np.int64)
    # Near-oracle probabilities with slight noise at boundary pixels
    probs = np.where(gt == 1, 0.9, 0.1).astype(np.float32)
    flip_idx = rng.choice(32 * 32, size=100, replace=False)
    probs_flat = probs.ravel()
    probs_flat[flip_idx] = 0.52  # uncertain errors!
    probs = probs_flat.reshape(32, 32)

    unc = 1.0 - 2.0 * np.abs(probs - 0.5)
    preds = (probs >= 0.5).astype(np.int64)

    ece = expected_calibration_error(probs, gt)
    bs = brier_score(probs, gt)
    curve = sparsification_curve(preds, gt, unc)

    assert 0.0 <= ece <= 0.25
    assert 0.0 <= bs <= 0.25
    # Abstaining on the most uncertain pixels should strictly improve accuracy!
    assert curve["retained_accuracy"][-1] > curve["retained_accuracy"][0]
    assert curve["ause"] < 0.05


def test_pca_otsu_sign_invariance_and_baselines():
    phantom = generate_correlative_phantom_2d(
        size=96, downsample_factor=8, phase_b_contrast=0.35, seed=9
    )
    guide_a = phantom.coarse_guide[0]  # (12, 12)
    gt_a = phantom.gt_phase_channels[0].astype(np.int64)

    # Construct synthetic feature map correlated with Phase A
    feat = torch.randn(1, 8, 96, 96) * 0.1
    feat[:, :4] += (
        torch.from_numpy(phantom.gt_phase_channels[0]).view(1, 1, 96, 96) * 2.0
    )

    mask_pos, _ = pca_otsu_segment(feat, guide_lr=guide_a)
    mask_neg, _ = pca_otsu_segment(-feat, guide_lr=guide_a)
    assert np.array_equal(mask_pos, mask_neg)
    assert compute_segmentation_metrics(mask_pos, gt_a)["iou"] > 0.85

    bic_rep = bicubic_upsample_baseline(guide_a, (96, 96), mode="multilabel")
    jbu_rep = jbu_guided_baseline(phantom.hr_image, guide_a, mode="multilabel")
    km_rep = dino_kmeans_baseline(feat, guide_a, n_clusters=2)

    m_bic = compute_segmentation_metrics(bic_rep.hard_mask[0], gt_a)
    m_jbu = compute_segmentation_metrics(jbu_rep.hard_mask[0], gt_a)
    m_km = compute_segmentation_metrics(km_rep.hard_mask[0], gt_a)
    assert m_jbu["iou"] >= m_bic["iou"]
    assert m_km["iou"] > 0.75


def test_refine_with_sam2_mock_predictor():
    class _DummySAM2Predictor:
        def __init__(self):
            self.last_points = None

        def set_image(self, image):
            self.shape = image.shape[:2]

        def predict(self, point_coords, point_labels, box, multimask_output=True):
            self.last_points = (point_coords, point_labels, box)
            m = np.zeros((3, self.shape[0], self.shape[1]), dtype=np.float32)
            x0, y0, x1, y1 = box[0].astype(int)
            m[1, y0 : y1 + 1, x0 : x1 + 1] = 1.0
            return m, np.array([0.2, 0.95, 0.4]), None

    probs = torch.zeros(1, 1, 32, 32)
    probs[:, :, 8:24, 8:24] = 0.92
    guide = torch.nn.functional.adaptive_avg_pool2d(probs, (4, 4))
    rep = build_uncertainty_report(probs, guide, mode="multilabel")

    dummy = _DummySAM2Predictor()
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = refine_with_sam2(rgb, rep, dummy, target_class=1, max_points=6)
    assert mask.shape == (32, 32)
    assert mask[12, 12] == 1
    assert dummy.last_points is not None


@pytest.mark.parametrize("fuse_method", ["gating", "learned_gating", "attention"])
def test_correlative_fuser_end_to_end_cpu(fuse_method):
    """Run full CorrelativeFuser pipeline on CPU using offline tiny ViT."""
    phantom = generate_correlative_phantom_2d(size=48, downsample_factor=8, seed=5)
    hr_t = normalize_structural(phantom.hr_image)
    guide_t = guide_to_tensor(phantom.coarse_guide)  # 2 guide channels simultaneously!

    fwd, inv = build_default_tta(
        shift_dists=[1], pattern="Neumann", include_flips=False
    )
    loss_fn = "forward_psf_bce" if fuse_method == "learned_gating" else "pearson"

    model = CorrelativeFuser(
        dino_name="timm:vit_tiny_patch16_224",
        xrd_fuse_method=fuse_method,
        stride=8,
        dtype=torch.float32,
        loss_fn=loss_fn,
        guide_channels=2,
        interpolation_mode="jbu",
        pretrained=False,
        device="cpu",
    )
    model.set_guide_transforms(guide_t, fwd, inv)

    if fuse_method in ("learned_gating", "attention"):
        loss0 = model.train_fusion_step(hr_t)
        loss1 = model.train_fusion_step(hr_t)
        assert np.isfinite(loss0) and np.isfinite(loss1)
        if fuse_method == "learned_gating" and model.learned_gating is not None:
            model.learned_gating.select_top_k(16)

    feats, feat_var = model.forward_with_variance(hr_t)
    assert feats.shape == (1, 192, 48, 48)
    assert feat_var.shape == (1, 192, 48, 48)

    rep = model.predict_with_uncertainty(
        hr_t, guide_lr=guide_t, mode="multiclass", abstain_threshold=0.95
    )
    assert rep.probabilities.shape == (1, 3, 48, 48)
    assert rep.hard_mask.shape == (1, 48, 48)
    assert rep.combined_uncertainty.shape == (1, 48, 48)
