"""Unit tests for Vision Transformer patching, ALiBi positional wrappers,
TorchPCA, and HighResBackbone dense feature extraction (100% offline CPU).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from x_fuse.alibi import (
    AlibiVitWrapper,
    DistanceMatrixWrapper,
    PretrainedViTWrapper,
    add_flash_attention,
    get_alibi_slope,
    get_distance_matrix,
)
from x_fuse.backbones import HighResBackbone, infer_vit_params, load_backbone
from x_fuse.transforms import build_default_tta, get_rotation_transforms
from x_fuse.vit_patch import (
    Patch,
    TorchPCA,
    drop_add_residual_stochastic_depth,
    get_qkvo_per_head,
    torch_pca,
)


def test_distance_matrix_and_alibi_slopes():
    d_euc = get_distance_matrix(
        4, 4, n_reg_tokens=2, metric="euclidean", normalize=True, wrap=True
    )
    # 1 CLS + 2 REG + 16 patches = 19 tokens
    assert d_euc.shape == (19, 19)
    assert float(d_euc.max().item()) <= 0.0
    assert float(d_euc.min().item()) >= -1.0 - 1e-6

    d_man = get_distance_matrix(
        3, 4, n_reg_tokens=0, metric="manhattan", normalize=False, add_cls=False
    )
    assert d_man.shape == (12, 12)

    with pytest.raises(ValueError, match="metric must be"):
        get_distance_matrix(2, 2, metric="cosine")

    s_fixed = get_alibi_slope(6, slope_type="fixed")
    s_learned = get_alibi_slope(6, slope_type="learned")
    s_const = get_alibi_slope(6, slope_type="constant")
    assert s_fixed.shape == (6, 1, 1) and not s_fixed.requires_grad
    assert isinstance(s_learned, nn.Parameter) and s_learned.requires_grad
    assert torch.allclose(s_const, torch.ones_like(s_const))

    with pytest.raises(ValueError, match="Unknown slope type"):
        get_alibi_slope(4, slope_type="quadratic")  # type: ignore[arg-type]

    wrapper = DistanceMatrixWrapper(4, 4, n_reg_tokens=2)
    orig_ptr = wrapper.matrix.data_ptr()
    wrapper.update(4, 4, n_reg_tokens=2)  # cache hit no-op
    assert wrapper.matrix.data_ptr() == orig_ptr
    wrapper.update(6, 6, n_reg_tokens=2)  # stale rebuild
    assert wrapper.matrix.shape == (39, 39)


def test_alibi_vit_wrapper_offline_cpu():
    model = AlibiVitWrapper(
        model_identifier="vit_small_patch16_224.dino",
        stride=8,
        pretrained=False,
        slope_type="learned",
        n_reg_tokens=0,
        add_cls=True,
    )
    assert model.n_output_dims == 384
    assert model.last_layer_index == model.num_blocks - 1
    add_flash_attention(model.model)

    x = torch.randn(1, 3, 48, 64)
    feats_2d = model.forward_features(x, make_2D=True, add_reg=False)
    # n_patch_h = (48 - 16)//8 + 1 = 5; n_patch_w = (64 - 16)//8 + 1 = 7
    assert feats_2d.shape == (1, 384, 5, 7)

    inter = model.get_intermediate_layers(x, n=1, reshape=True)
    assert len(inter) == 1
    assert inter[0].shape == (1, 384, 5, 7)

    pt_wrap = PretrainedViTWrapper(
        model_identifier="vit_small_patch16_224.dino",
        stride=16,
        pretrained=False,
    )
    out = pt_wrap(torch.randn(1, 3, 32, 32))
    assert out.shape == (1, 384)


def test_vit_patch_qkvo_stochastic_depth_and_pca():
    q = torch.randn(2, 10, 4, 8)
    k = torch.randn(2, 10, 4, 8)
    v = torch.randn(2, 10, 4, 8)
    x_a = torch.softmax(torch.randn(2, 10, 4, 10), dim=-1)

    for opt in ("q", "k", "v", "o"):
        res = get_qkvo_per_head(q, k, v, x_a, opt, lambda t: t, n_heads=4)
        assert res.shape == (2, 10, 4)

    with pytest.raises(ValueError, match="Invalid attention option"):
        get_qkvo_per_head(q, k, v, x_a, "none", lambda t: t, n_heads=4)

    dropped = drop_add_residual_stochastic_depth(
        torch.ones(4, 5, 8), lambda inp: inp * 0.5, sample_drop_ratio=0.5
    )
    assert dropped.shape == (4, 5, 8)

    # TorchPCA and torch_pca (including subsampling path)
    data = torch.randn(64, 16)
    pca = TorchPCA(n_components=3).fit(data)
    proj = pca.transform(data)
    assert proj.shape == (64, 3)

    feat_map = torch.randn(1, 16, 20, 20)
    proj_map = torch_pca(feat_map, dim=3, max_samples=50)
    assert proj_map.shape == (400, 3)


def test_vit_patch_monkey_patching_on_synthetic_vit():
    class _ToyAttn(nn.Module):
        def __init__(self, dim: int = 32, num_heads: int = 4):
            super().__init__()
            self.num_heads = num_heads
            self.qkv = nn.Linear(dim, 3 * dim)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Identity()
            self.attn_drop = nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class _ToyBlock(nn.Module):
        def __init__(self, dim: int = 32, num_heads: int = 4):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = _ToyAttn(dim, num_heads)
            self.ls1 = nn.Identity()
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Linear(dim, dim)
            self.ls2 = nn.Identity()
            self.sample_drop_ratio = 0.25

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class _ToyViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 32
            self.num_heads = 4
            self.num_register_tokens = 0
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 32))
            self.pos_embed = nn.Parameter(torch.randn(1, 1 + 16, 32))
            self.patch_embed = nn.Module()
            self.patch_embed.proj = nn.Conv2d(3, 32, kernel_size=8, stride=8)
            self.patch_embed.patch_size = (8, 8)
            self.blocks = nn.ModuleList([_ToyBlock(32, 4), _ToyBlock(32, 4)])
            self.norm = nn.LayerNorm(32)

        def forward_PatchEmbed(self, x: torch.Tensor) -> torch.Tensor:
            return self.patch_embed.proj(x).flatten(2).transpose(1, 2)

    toy = _ToyViT()
    toy.patch_embed.forward = toy.forward_PatchEmbed  # type: ignore[method-assign]

    # Exercise HighResBackbone with custom non-timm ViT patching
    hr = HighResBackbone(
        "custom_vit_tiny",
        stride=4,
        pca_dim=4,
        dtype=32,
        interpolation_mode="bilinear",
        backbone=toy,
    )
    fwd, inv = get_rotation_transforms()
    hr.set_transforms(fwd[:2], inv[:2])

    img = torch.rand(3, 32, 32)
    dense = hr.forward_sequential(img, attn_choice="q")
    assert dense.shape[0] == 1 and dense.shape[-2:] == (32, 32)

    # Also test stochastic depth branch in training mode
    toy.train()
    blk_fn = Patch._fix_block_forward_dv2()
    out_train = blk_fn(toy.blocks[0], torch.randn(4, 10, 32), "none")
    assert out_train.shape == (4, 10, 32)


def test_infer_vit_params_and_load_backbone_branches():
    assert infer_vit_params("dinov2_vits14_reg") == (384, 14, 6)
    assert infer_vit_params("dinov2_vitb14") == (768, 14, 12)
    assert infer_vit_params("dinov3_vit7b16") == (4096, 16, 32)
    assert infer_vit_params("my_tiny_model") == (192, 16, 3)
    assert infer_vit_params("my_base_model") == (768, 16, 12)
    assert infer_vit_params("my_large_model") == (1024, 16, 16)
    assert infer_vit_params("unknown_arch") == (384, 16, 6)

    dummy = nn.Conv2d(3, 8, 1)
    assert load_backbone("anything", stride=4, backbone=dummy) is dummy

    hr_jbu = HighResBackbone(
        "timm:vit_tiny_patch16_224",
        stride=8,
        pretrained=False,
        interpolation_mode="jbu",
    )
    fwd, inv = build_default_tta(shift_dists=[1], pattern="Neumann")
    hr_jbu.set_transforms(fwd, inv)
    out = hr_jbu.forward_sequential(torch.rand(3, 48, 48))
    assert out.shape == (1, 192, 48, 48)
