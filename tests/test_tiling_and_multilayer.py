"""Unit tests for Hann-windowed 2D tiling (`x_fuse.tiling`), multi-layer ViT
feature pyramids (`layers="multiscale"`), and frozen TTA feature caching.
"""

from __future__ import annotations

import pytest
import torch

from x_fuse import (
    CorrelativeFuser,
    HighResBackbone,
    build_default_tta,
    ensure_spatial_divisible,
    generate_correlative_phantom_2d,
    generate_tile_boxes,
    guide_to_tensor,
    hann_window_2d,
    normalize_structural,
    run_tiled_2d,
)


def test_generate_tile_boxes_and_validation():
    with pytest.raises(ValueError, match="tile_size must be positive"):
        generate_tile_boxes(128, 128, tile_size=0, overlap=16)

    with pytest.raises(ValueError, match="overlap must satisfy"):
        generate_tile_boxes(128, 128, tile_size=64, overlap=64)

    with pytest.raises(ValueError, match="overlap must satisfy"):
        generate_tile_boxes(128, 128, tile_size=64, overlap=-4)

    # Image smaller than tile_size returns single full-image box
    assert generate_tile_boxes(40, 52, tile_size=64, overlap=16) == [(0, 0, 40, 52)]

    # Non-multiple dimensions shift final tiles inward to preserve tile_size
    boxes = generate_tile_boxes(110, 130, tile_size=64, overlap=16)
    assert len(boxes) == 6
    for y0, x0, y1, x1 in boxes:
        assert (y1 - y0) == 64
        assert (x1 - x0) == 64
        assert 0 <= y0 and y1 <= 110
        assert 0 <= x0 and x1 <= 130
    assert boxes[-1] == (110 - 64, 130 - 64, 110, 130)


def test_hann_window_2d_and_seamless_run_tiled_2d():
    # Outer corner tile keeps weight 1.0 on top & left borders, tapers on bottom & right
    win = hann_window_2d(
        height=32,
        width=32,
        overlap=8,
        at_top=True,
        at_bottom=False,
        at_left=True,
        at_right=False,
        device=torch.device("cpu"),
    )
    assert win.shape == (1, 1, 32, 32)
    assert torch.isclose(win[0, 0, 0, 0], torch.tensor(1.0))
    assert float(win[0, 0, -1, -1].item()) < 0.25

    with pytest.raises(ValueError, match="Expected 3D"):
        run_tiled_2d(lambda x: x.unsqueeze(0), torch.zeros(1, 3, 32, 32))

    # Linear spatial ramp passed through affine tile_fn must reconstruct seamlessly
    yy = torch.linspace(0.0, 2.0, 95).view(1, 95, 1).expand(3, 95, 107)
    xx = torch.linspace(-1.0, 1.0, 107).view(1, 1, 107).expand(3, 95, 107)
    img = (yy + xx).contiguous()

    blended = run_tiled_2d(
        tile_fn=lambda tile_chw: (2.0 * tile_chw + 0.5).unsqueeze(0),
        image_chw=img,
        tile_size=48,
        overlap=16,
        patch_size=16,
        stride=8,
    )
    assert blended.shape == (1, 3, 95, 107)
    assert torch.allclose(blended[0], 2.0 * img + 0.5, atol=1e-5)


def test_multiscale_backbone_and_explicit_layer_sequence():
    bb_multi = HighResBackbone(
        "timm:vit_tiny_patch16_224",
        stride=8,
        dtype=torch.float32,
        interpolation_mode="bilinear",
        pretrained=False,
        layers="multiscale",
        tile_size=48,
        tile_overlap=16,
    )
    assert bb_multi.layer_indices is not None
    assert len(bb_multi.layer_indices) == 4
    assert bb_multi.feat_dim == 4 * bb_multi.base_feat_dim

    x = torch.randn(3, 72, 80)
    feats = bb_multi.forward_sequential(x)
    assert feats.shape == (1, bb_multi.feat_dim, 72, 80)

    # Explicit negative/positive layer list
    bb_pair = HighResBackbone(
        "timm:vit_tiny_patch16_224",
        stride=8,
        dtype=torch.float32,
        pretrained=False,
        layers=[0, -1],
    )
    assert bb_pair.layer_indices is not None
    assert len(bb_pair.layer_indices) == 2
    assert bb_pair.feat_dim == 2 * bb_pair.base_feat_dim
    feats_pair = bb_pair.forward_sequential(torch.randn(3, 48, 48))
    assert feats_pair.shape == (1, bb_pair.feat_dim, 48, 48)

    with pytest.raises(ValueError, match="Unsupported layers"):
        HighResBackbone(
            "timm:vit_tiny_patch16_224",
            stride=8,
            pretrained=False,
            layers="invalid_mode",  # type: ignore[arg-type]
        )


def test_tta_feature_cache_and_tiled_uncertainty_prediction():
    phantom = generate_correlative_phantom_2d(size=72, downsample_factor=4, seed=3)
    hr = normalize_structural(phantom.hr_image)
    hr_pad, _ = ensure_spatial_divisible(hr, patch_size=16, stride=8)
    guide = guide_to_tensor(phantom.coarse_guide)
    fwd, inv = build_default_tta(shift_dists=[1], include_flips=False)

    # 1. Verify frozen TTA feature caching on untiled pass
    fuser = CorrelativeFuser(
        backbone="timm:vit_tiny_patch16_224",
        fusion_method="learned_gating",
        stride=8,
        dtype=torch.float32,
        loss_fn="forward_psf_bce",
        pretrained=False,
        layers="multiscale",
        cache_features=True,
    )
    fuser.set_guide_transforms(guide, fwd, inv)
    assert len(fuser._tta_cache) == 0

    loss_0 = fuser.train_fusion_step(hr_pad)
    assert len(fuser._tta_cache) == 1

    # Second epoch hits frozen TTA cache without recomputing backbone forward passes
    loss_1 = fuser.train_fusion_step(hr_pad)
    assert loss_0 > 0.0 and loss_1 > 0.0
    assert len(fuser._tta_cache) == 1

    fuser.clear_feature_cache()
    assert len(fuser._tta_cache) == 0

    # 2. Verify overlapping tiled uncertainty prediction when tile_size < image size
    fuser.tile_size = 48
    fuser.tile_overlap = 16
    rep = fuser.predict_with_uncertainty(hr_pad, guide_lr=guide, mode="multiclass")
    assert rep.probabilities.shape == (1, 3, hr_pad.shape[1], hr_pad.shape[2])
    assert torch.isfinite(rep.combined_uncertainty).all()
