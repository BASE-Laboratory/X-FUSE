"""CUDA GPU integration tests for X-FUSE 2D multi-channel fusion, uncertainty
quantification, and 2.5D volumetric propagation (`pytest -m gpu`).
"""

from __future__ import annotations

import pytest
import torch

from x_fuse import (
    CorrelativeFuser,
    VolumetricFuser,
    build_default_tta,
    ensure_spatial_divisible,
    generate_correlative_phantom_2d,
    generate_correlative_phantom_3d,
    guide_to_tensor,
    normalize_structural,
)


@pytest.mark.gpu
def test_cuda_is_available():
    assert torch.cuda.is_available(), "CUDA device is required for -m gpu tests"


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["gating", "learned_gating", "attention"])
def test_correlative_fuser_2d_on_cuda(method: str):
    assert torch.cuda.is_available(), "CUDA device is required for -m gpu tests"
    phantom = generate_correlative_phantom_2d(size=64, downsample_factor=4, seed=11)
    hr_t, (oh, ow) = ensure_spatial_divisible(
        normalize_structural(phantom.hr_image, device="cuda"), patch_size=16, stride=8
    )
    guide_t = guide_to_tensor(phantom.coarse_guide, device="cuda", normalize=True)
    fwd, inv = build_default_tta(shift_dists=[1], pattern="Neumann")

    fuser = CorrelativeFuser(
        dino_name="timm:vit_tiny_patch16_224",
        xrd_fuse_method=method,  # type: ignore[arg-type]
        stride=8,
        dtype=torch.float16,
        loss_fn="forward_psf_bce" if method == "learned_gating" else "bce",
        guide_channels=2,
        interpolation_mode="jbu",
        pretrained=False,
        lr=1e-2,
        device="cuda",
    )
    fuser.set_guide_transforms(guide_t, fwd, inv)
    if method in ("learned_gating", "attention"):
        loss = fuser.train_fusion_step(hr_t)
        assert loss >= 0.0

    rep = fuser.predict_with_uncertainty(hr_t, guide_lr=guide_t, mode="multiclass")
    assert rep.probabilities.device.type == "cuda"
    assert rep.hard_mask[:, :oh, :ow].shape == (1, 64, 64)


@pytest.mark.gpu
def test_volumetric_fuser_triplane_on_cuda():
    assert torch.cuda.is_available(), "CUDA device is required for -m gpu tests"
    ph3 = generate_correlative_phantom_3d(
        shape_zyx=(16, 32, 32), downsample_factor=4, seed=21
    )
    sparse = {2: ph3.coarse_guide_3d[:, 0], 12: ph3.coarse_guide_3d[:, -1]}

    base = CorrelativeFuser(
        dino_name="timm:vit_tiny_patch16_224",
        xrd_fuse_method="learned_gating",
        stride=8,
        dtype=torch.float32,
        loss_fn="forward_psf_bce",
        guide_channels=2,
        interpolation_mode="jbu",
        pretrained=False,
        device="cuda",
    )
    vf = VolumetricFuser(base)
    vf.fit_sparse_slices(ph3.hr_volume, sparse, epochs=3, delta=1)
    rep3d = vf.predict_volume(
        ph3.hr_volume, mode="triplane", sparse_guides=sparse, pred_mode="multiclass"
    )
    assert rep3d.hard_mask.shape == (16, 32, 32)
