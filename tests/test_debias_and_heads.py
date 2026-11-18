"""Unit tests for `ApositionalFeatureDebiaser`, `CrossAttentionFeatureUpsampler`,
`SpatialConvMLPHead`, memory-safe $O(HW)$ `WindowedGuideCrossAttentionHead`,
and iso-intensity textured phase disentanglement.
"""

from __future__ import annotations

import torch

from x_fuse import (
    ApositionalFeatureDebiaser,
    CrossAttentionFeatureUpsampler,
    SpatialConvMLPHead,
    WindowedGuideCrossAttentionHead,
    XFuse,
    build_orthonormal_spatial_basis,
    compute_segmentation_metrics,
    edge_aware_tv_loss,
    extract_multiscale_texture_descriptors,
    generate_macroscopic_gradient_phantom_2d,
    generate_textured_isointensity_phantom_2d,
    jbu_guided_baseline,
    load_backbone,
    measure_positional_r2,
)


def test_apositional_debiaser_eliminates_coordinate_polynomials() -> None:
    q = build_orthonormal_spatial_basis(24, 32, degree=2)
    assert q.shape == (24 * 32, 5)
    gram = q.T @ q
    assert torch.allclose(gram, torch.eye(5), atol=1e-5)

    ys = torch.linspace(-1.0, 1.0, 24).view(1, 1, 24, 1).expand(1, 1, 24, 32)
    xs = torch.linspace(-1.0, 1.0, 32).view(1, 1, 1, 32).expand(1, 1, 24, 32)
    quad = 0.7 * ys + 0.4 * xs - 0.5 * (xs * ys) + 2.5  # Pure polynomial + mean
    content = torch.sin(ys * 9.0) * torch.cos(xs * 9.0)
    feats = torch.cat([quad, content], dim=1)

    r2_before = measure_positional_r2(feats, degree=2)
    assert float(r2_before[0].item()) > 0.99

    debiaser = ApositionalFeatureDebiaser(enabled=True, degree=2, mode="orthogonal")
    clean = debiaser(feats)
    assert clean.shape == feats.shape
    # Constant channel mean is strictly preserved
    assert float(clean[0, 0].mean().item()) == float(
        torch.approx(feats[0, 0].mean().item(), abs=1e-5)
        if hasattr(torch, "approx")
        else clean[0, 0].mean().item()
    )
    assert abs(float(clean[0, 0].mean().item() - feats[0, 0].mean().item())) < 1e-4

    # High-frequency content channel has near-zero polynomial component after debiasing
    r2_after = measure_positional_r2(clean[:, 1:2], degree=2)
    assert float(r2_after[0].item()) < 1e-5


def test_apositional_debiaser_noise_calibration_and_attenuate_mode() -> None:
    def _synthetic_vit(img_chw: torch.Tensor) -> torch.Tensor:
        _, h, w = img_chw.shape
        gh, gw = h // 8, w // 8
        ramp = torch.linspace(-1.0, 1.0, gw).view(1, 1, 1, gw).expand(1, 1, gh, gw)
        pooled = torch.nn.functional.adaptive_avg_pool2d(img_chw.unsqueeze(0), (gh, gw))
        return torch.cat([ramp, pooled], dim=1)

    deb = ApositionalFeatureDebiaser(enabled=True, degree=2, mode="attenuate")
    mean_r2 = deb.calibrate_on_homogeneous_noise(
        _synthetic_vit, image_hw=(64, 64), n_samples=2, r2_cutoff=0.2
    )
    assert float(mean_r2[0].item()) > 0.95
    assert bool(deb.calibrated_pos_mask[0].item()) is True
    sample = _synthetic_vit(torch.randn(3, 64, 64))
    attenuated = deb(sample)
    assert float(attenuated[:, 0].std().item()) < 1e-4


def test_alibi_backbone_prefix_loads_cleanly() -> None:
    vit = load_backbone("alibi:vit_tiny_patch16_224", stride=8, pretrained=False)
    x = torch.randn(1, 3, 64, 64)
    out = vit.forward_features(x)
    assert out.ndim == 3


def test_windowed_cross_attention_and_cross_attn_upsampler_large_spatial_scale() -> (
    None
):
    # Verify O(HW) scaling on 256x256 HR grid (which previously OOM'd 65536^2 attention)
    head = WindowedGuideCrossAttentionHead(feat_dim=32, num_heads=4, guide_channels=2)
    feats_256 = torch.randn(1, 32, 256, 256)
    guide_32 = torch.rand(1, 2, 32, 32)
    attended = head(feats_256, guide_32)
    assert attended.shape == (1, 32, 256, 256)
    logits = head.predict_logits(feats_256, guide_32)
    assert logits.shape == (1, 2, 256, 256)

    up = CrossAttentionFeatureUpsampler(feat_dim=32, learned=True)
    lr_tokens = torch.randn(1, 32, 32, 32)
    hr_img = torch.rand(1, 3, 256, 256)
    desc = extract_multiscale_texture_descriptors(hr_img)
    assert desc.shape == (1, 8, 256, 256)
    hr_feats = up(lr_tokens, hr_img, output_size=(256, 256))
    assert hr_feats.shape == (1, 32, 256, 256)

    tv = edge_aware_tv_loss(logits[:, :, :64, :64], guidance_hr=hr_img[:, :, :64, :64])
    assert float(tv.item()) >= 0.0


def test_cross_attn_and_spatial_mlp_disentangle_textured_isointensity_phases() -> None:
    ph = generate_textured_isointensity_phantom_2d(
        size=64, downsample_factor=8, texture_amplitude=0.38, seed=55
    )
    jbu_rep = jbu_guided_baseline(ph.hr_image, ph.coarse_guide, mode="multiclass")
    jbu_iou = compute_segmentation_metrics(jbu_rep.hard_mask[0], ph.gt_labels)["iou"]

    model = XFuse(
        backbone="timm:vit_tiny_patch16_224",
        pretrained=False,
        stride=8,
        dtype=torch.float32,
        loss_fn="forward_psf_bce",
        interpolation_mode="cross_attn",
        learned_upsampler=True,
        fusion_method="spatial_mlp",
        debias_positional=True,
        lr=1.5e-2,
    )
    assert isinstance(model.spatial_mlp, SpatialConvMLPHead)
    rep = model.fit_predict(
        ph.hr_image,
        ph.coarse_guide,
        epochs=22,
        mode="multiclass",
        tta_shifts=(),
        tta_flips=False,
    )
    xf_iou = compute_segmentation_metrics(rep.hard_mask[0], ph.gt_labels)["iou"]
    assert xf_iou > jbu_iou + 0.04


def test_macroscopic_gradient_phantom_with_debias_positional() -> None:
    ph = generate_macroscopic_gradient_phantom_2d(size=64, downsample_factor=4, seed=7)
    model = XFuse(
        backbone="timm:vit_tiny_patch16_224",
        pretrained=False,
        stride=8,
        dtype=torch.float32,
        loss_fn="forward_psf_bce",
        interpolation_mode="cross_attn",
        fusion_method="spatial_mlp",
        debias_positional=True,
        lr=1e-2,
    )
    rep = model.fit_predict(
        ph.hr_image,
        ph.coarse_guide,
        epochs=10,
        mode="multiclass",
        tta_shifts=(),
        tta_flips=False,
    )
    assert rep.diagnostics.get("debias_positional") is True
    iou = compute_segmentation_metrics(rep.hard_mask[0], ph.gt_labels)["iou"]
    assert iou > 0.50
