"""Utility functions for synthetic multiphase generation, float-safe guide tensor
conversion, and ALiBi / DINOv3 model checkpoint wrappers.
"""

from collections.abc import Sequence
from types import MethodType
from typing import Any

import cv2
import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from .alibi import (
    MODEL_LIST,
    AlibiSlopeType,
    AlibiVitWrapper,
    DistanceMatrixWrapper,
    PretrainedViTWrapper,
    get_alibi_slope,
)


def get_multiphase(
    imA,
    imA_kwargs,
    imB,
    imB_kwargs,
    extract_spheres: bool = False,
    vis: bool = False,
    dataset_size: int = 1,
):
    """Generate 2D multi-phase grayscale images and binary Phase A / Phase B masks."""
    seed_base = 1337
    rng = np.random.default_rng(seed_base)
    gray_imgs = []
    phase_a_masks = []
    phase_b_masks = []

    im_a_kw = dict(imA_kwargs)
    im_b_kw = dict(imB_kwargs)

    for i in range(dataset_size):
        im_a_kw["seed"] = seed_base + i
        im_b_kw["seed"] = seed_base + i

        im_a = imA(**im_a_kw)
        im_b = imB(**im_b_kw) if not extract_spheres else imB(im=im_a, **im_b_kw)

        multi_phase = np.zeros_like(im_a, dtype=np.uint8)
        multi_phase[im_a & ~im_b] = 1
        multi_phase[~im_a & im_b] = 2
        multi_phase[im_a & im_b] = 2 if not extract_spheres else 3

        gray_img = np.zeros_like(multi_phase, dtype=np.float32)
        n_1 = int((multi_phase == 1).sum())
        n_2 = int((multi_phase == 2).sum())
        if n_1 > 0:
            gray_img[multi_phase == 1] = rng.uniform(0.8, 1.2, size=n_1).astype(
                np.float32
            )
        if n_2 > 0:
            gray_img[multi_phase == 2] = rng.uniform(1.6, 2.0, size=n_2).astype(
                np.float32
            )
        if extract_spheres:
            n_3 = int((multi_phase == 3).sum())
            if n_3 > 0:
                gray_img[multi_phase == 3] = rng.uniform(2.5, 3.0, size=n_3).astype(
                    np.float32
                )

        phase_a = im_a.astype(np.float32)
        phase_b = im_b.astype(np.float32)

        if extract_spheres:
            phase_b_mask = np.zeros_like(phase_b)
            phase_b_mask[phase_a == phase_b] = 0
            phase_b_mask[phase_a != phase_b] = 1
            phase_b = phase_b_mask.astype(np.float32)

        g_min, g_max = float(gray_img.min()), float(gray_img.max())
        if g_max - g_min > 1e-8:
            gray_img = ((gray_img - g_min) / (g_max - g_min)).astype(np.float32)

        gray_imgs.append(gray_img)
        phase_a_masks.append(phase_a)
        phase_b_masks.append(phase_b)

        if vis:
            _, ax = plt.subplots(1, 4, figsize=(24, 12))
            ax[0].imshow(multi_phase, cmap="gray")
            ax[0].set_title("Multi-phase image (0=bg, 1=A, 2=B, 3=overlap)")
            ax[0].axis("off")
            im_handle = ax[1].imshow(gray_img, cmap="gray")
            ax[1].set_title("Grayscale multi-phase image")
            ax[1].axis("off")
            ax[2].imshow(phase_a, cmap="gray")
            ax[2].set_title("Phase A")
            ax[2].axis("off")
            ax[3].imshow(phase_b, cmap="gray")
            ax[3].set_title("Phase B")
            ax[3].axis("off")
            plt.colorbar(im_handle, ax=ax[1])
            plt.tight_layout()
            plt.show()

    return gray_imgs, phase_a_masks, phase_b_masks


def downsample_xrdct(
    gray_imgs: Sequence[np.ndarray],
    phase_A_masks: Sequence[np.ndarray],
    phase_B_masks: Sequence[np.ndarray],
    factor: int = 2,
    vis: bool = False,
):
    """Downsample Phase A and Phase B masks by ``factor``."""
    downsampled_a = []
    downsampled_b = []

    shape_a = phase_A_masks[0].shape
    shape_b = phase_B_masks[0].shape

    for i in range(len(gray_imgs)):
        ds_a = cv2.resize(
            phase_A_masks[i],
            (shape_a[1] // factor, shape_a[0] // factor),
            interpolation=cv2.INTER_AREA,
        )
        ds_b = cv2.resize(
            phase_B_masks[i],
            (shape_b[1] // factor, shape_b[0] // factor),
            interpolation=cv2.INTER_AREA,
        )
        downsampled_a.append(ds_a.astype(np.float32))
        downsampled_b.append(ds_b.astype(np.float32))

    if vis:
        _, ax = plt.subplots(len(gray_imgs), 3, figsize=(24, 12))
        ax_grid = np.atleast_2d(ax)
        for i in range(len(gray_imgs)):
            ax_grid[i, 0].imshow(gray_imgs[i], cmap="gray")
            ax_grid[i, 0].set_title(f"Grayscale Multi-phase Image {i + 1}")
            ax_grid[i, 0].axis("off")
            ax_grid[i, 1].imshow(downsampled_a[i], cmap="gray")
            ax_grid[i, 1].set_title(f"Downsampled Phase A Mask {i + 1}")
            ax_grid[i, 1].axis("off")
            ax_grid[i, 2].imshow(downsampled_b[i], cmap="gray")
            ax_grid[i, 2].set_title(f"Downsampled Phase B Mask {i + 1}")
            ax_grid[i, 2].axis("off")
        plt.tight_layout()
        plt.show()

    return gray_imgs, downsampled_a, downsampled_b


def get_ps_images(IMG_SIZE: int = 224):
    """Generate 10 example synthetic 2-phase PoreSpy phantoms."""
    import porespy as ps

    im1 = ps.generators.blobs
    im2 = ps.generators.random_spheres
    im1_kwargs = {"shape": [IMG_SIZE, IMG_SIZE], "blobiness": 0.5, "porosity": 0.1}
    im2_kwargs = {"r": 40, "clearance": 10}

    gray, phase_a, phase_b = get_multiphase(
        im1,
        im1_kwargs,
        im2,
        im2_kwargs,
        extract_spheres=True,
        vis=False,
        dataset_size=10,
    )
    return downsample_xrdct(gray, phase_a, phase_b, factor=10, vis=False)


def load_xrd_h5(xrd_path: str, key: str = "xrd") -> np.ndarray:
    """Load an XRD/guide array from an HDF5 file cleanly closing the handle."""
    with h5.File(xrd_path, "r") as f:
        return np.array(f[key][:])


def guide_to_tensor(
    guide_img: np.ndarray | torch.Tensor,
    device: str | torch.device = "cpu",
    normalize: bool = True,
) -> torch.Tensor:
    """Convert a 2D ``(H, W)``, 3D ``(M, H, W)``, or 4D ``(B, M, H, W)`` coarse
    physical guide map into a ``(B, M, H, W)`` float32 tensor without 8-bit
    quantization loss.

    When ``normalize=True``, each channel ``m`` is independently min-max scaled
    into ``[0, 1]`` (with zero-variance channels safely mapped to ``0.0``).
    """
    if isinstance(guide_img, torch.Tensor):
        t = guide_img.detach().clone().to(dtype=torch.float32)
    else:
        t = torch.from_numpy(np.asarray(guide_img, dtype=np.float32))

    if t.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.ndim == 3:
        t = t.unsqueeze(0)
    elif t.ndim != 4:
        raise ValueError(f"Expected 2D, 3D, or 4D guide map, got shape {t.shape}.")

    if normalize:
        flat = t.flatten(start_dim=2)
        mn = flat.min(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
        mx = flat.max(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
        span = mx - mn
        t = torch.where(
            span > 1e-12, (t - mn) / span.clamp(min=1e-12), torch.zeros_like(t)
        )

    return t.to(device=device, dtype=torch.float32)


def xrd_to_tensor(
    xrd_img: np.ndarray | torch.Tensor, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """Backward-compatible alias for ``guide_to_tensor`` preserving float precision."""
    return guide_to_tensor(xrd_img, device=device, normalize=True)


def get_alibi_model(
    model_type: str,
    model_path: str,
    device: str,
    stride: int = 14,
) -> PretrainedViTWrapper:
    """Load an ALiBi ViT checkpoint into ``AlibiVitWrapper``."""
    weights = torch.load(model_path, weights_only=True, map_location=device)
    slope_type: AlibiSlopeType = "learned" if "_l" in model_type else "constant"
    add_cls = "nr" not in model_type
    n_reg_tokens = 0 if "nr" in model_type else 4
    arch_to_model = {
        "vits": MODEL_LIST[1] if n_reg_tokens > 0 else MODEL_LIST[0],
        "vitb": MODEL_LIST[3],
        "vitl": MODEL_LIST[15],
        "vitg": MODEL_LIST[16],
    }
    base_model = next(
        (v for k, v in arch_to_model.items() if k in model_type), MODEL_LIST[1]
    )

    model = AlibiVitWrapper(
        base_model,
        stride=stride,
        device=device,
        slope_type=slope_type,
        normalize=True,
        wrap=True,
        add_cls=add_cls,
        n_reg_tokens=n_reg_tokens,
    )
    model.load_state_dict(weights)
    return model


def _inject_alibi_dv3(
    dv3_model: Any,
    distance_matrix: DistanceMatrixWrapper,
    slope_type: AlibiSlopeType,
    device: str | torch.device,
) -> None:
    """Patch DINOv3 attention in-place to add ALiBi spatial distance bias."""
    num_heads = dv3_model.blocks[0].attn.num_heads
    m_init = get_alibi_slope(num_heads, slope_type=slope_type, device=device)
    is_learned = isinstance(m_init, nn.Parameter)

    def _compute_attn(self, qkv, attn_bias=None, rope=None):
        bsz, n_tok, _ = qkv.shape
        channels = self.qkv.in_features
        qkv_r = qkv.reshape(bsz, n_tok, 3, self.num_heads, channels // self.num_heads)
        q, k, v = qkv_r.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        bias = (
            self.m.to(device=q.device, dtype=q.dtype)
            * distance_matrix.matrix.to(device=q.device, dtype=q.dtype)
        ).unsqueeze(0)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return x.transpose(1, 2).reshape(bsz, n_tok, channels)

    for block in dv3_model.blocks:
        if is_learned:
            block.attn.register_parameter("m", nn.Parameter(m_init.data.clone()))
        else:
            block.attn.register_buffer("m", m_init.clone(), persistent=False)
        block.attn.compute_attention = MethodType(_compute_attn, block.attn)


def get_dv3_model(
    model_type: str,
    model_path: str,
    lib_path: str,
    device: str,
    stride: int = 16,
) -> nn.Module:
    """Load a DINOv3 checkpoint (NoPE or ALiBi) saved from ``AlibiVitWrapper``."""
    weights = torch.load(model_path, weights_only=True, map_location=device)
    n_reg_tokens = 0 if "nr" in model_type else 4
    add_cls = "nr" not in model_type
    use_alibi = "alibi" in model_type

    model_dict: dict[str, str] = {
        "vits16": "dinov3_vits16",
        "vits16plus": "dinov3_vits16plus",
        "vitl16": "dinov3_vitl16",
        "vith16": "dinov3_vith16",
        "vith16plus": "dinov3_vith16plus",
        "vit7b16": "dinov3_vit7b16",
    }

    hub_fn = model_dict[model_type.split("_")[2]]
    dv3 = torch.hub.load(lib_path, hub_fn, source="local", pretrained=False)

    if stride != dv3.patch_embed.proj.stride[0]:
        dv3.patch_embed.proj.stride = _pair(stride)

    distance_matrix = None
    if use_alibi:
        slope_type: AlibiSlopeType = "learned" if "_l" in model_type else "constant"
        distance_matrix = DistanceMatrixWrapper(
            n_tokens_h=16,
            n_tokens_w=16,
            n_reg_tokens=n_reg_tokens,
            normalize=True,
            wrap=True,
            add_cls=add_cls,
        )
        _inject_alibi_dv3(dv3, distance_matrix, slope_type, device)

    class _DV3Wrapper(nn.Module):
        def __init__(self, model, dist_mat, stride_val, n_reg):
            super().__init__()
            self.model = model
            self.distance_matrix = dist_mat
            self.stride = stride_val
            self.patch_size = 16
            self.n_reg_tokens = n_reg
            self.n_cls_tokens = 1

        def forward_features(self, x, make_2D: bool = False, **kwargs):
            bsz, _, height, width = x.shape
            p_sz, s_val = self.patch_size, self.stride
            n_h = (height - p_sz) // s_val + 1
            n_w = (width - p_sz) // s_val + 1
            if self.distance_matrix is not None:
                self.distance_matrix.update(n_h, n_w)
            feats = self.model.forward_features(x)["x_norm_patchtokens"]
            feats = feats.permute(0, 2, 1)
            if make_2D:
                feats = feats.reshape(bsz, -1, n_h, n_w)
            return feats

        def forward_feats_attn(self, x, masks=None, attn_choice: str = "none"):
            feats = self.forward_features(x)
            feats = feats.permute(0, 2, 1)
            return {"x_norm_patchtokens": feats, "masks": masks}

    wrapper = _DV3Wrapper(dv3, distance_matrix, stride, n_reg=n_reg_tokens)
    result = wrapper.load_state_dict(weights, strict=False)
    unexpected = result.unexpected_keys
    missing = [k for k in result.missing_keys if "rope" not in k]
    if unexpected or missing:
        raise RuntimeError(
            f"Checkpoint mismatch in _DV3Wrapper.\n"
            f"  Unexpected keys: {unexpected}\n"
            f"  Missing keys (non-rope): {missing}"
        )
    wrapper.to(device)
    return wrapper
