"""Utility functions for quantitative guide tensor conversion, ALiBi / DINOv3
checkpoint wrappers, and backward-compatible synthetic multiphase helpers.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import MethodType
from typing import Any

import h5py as h5
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
from .physics import GuideNormalization


def get_multiphase(
    imA,
    imA_kwargs,
    imB,
    imB_kwargs,
    extract_spheres: bool = False,
    vis: bool = False,
    dataset_size: int = 1,
):
    """Generate 2D multi-phase grayscale images and binary Phase A / Phase B masks
    (preserved for backward compatibility; prefer ``generate_correlative_phantom_2d``).
    """
    seed_base = 1337
    rng = np.random.default_rng(seed_base)
    gray_imgs: list[np.ndarray] = []
    phase_a_masks: list[np.ndarray] = []
    phase_b_masks: list[np.ndarray] = []

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
        phase_b = (
            (phase_a != im_b.astype(np.float32)).astype(np.float32)
            if extract_spheres
            else im_b.astype(np.float32)
        )

        g_min, g_max = float(np.min(gray_img)), float(np.max(gray_img))
        if g_max - g_min > 1e-8:
            gray_img = ((gray_img - g_min) / (g_max - g_min)).astype(np.float32)

        gray_imgs.append(gray_img)
        phase_a_masks.append(phase_a)
        phase_b_masks.append(phase_b)

    return gray_imgs, phase_a_masks, phase_b_masks


def _area_downsample_2d(arr: np.ndarray, factor: int) -> np.ndarray:
    """Pure-PyTorch/NumPy exact area-averaged downsampling without OpenCV dependency."""
    h, w = arr.shape
    th, tw = max(1, h // factor), max(1, w // factor)
    t = torch.from_numpy(np.asarray(arr, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(t, output_size=(th, tw))
    return pooled.squeeze(0).squeeze(0).numpy().astype(np.float32)


def downsample_xrdct(
    gray_imgs: Sequence[np.ndarray],
    phase_A_masks: Sequence[np.ndarray],
    phase_B_masks: Sequence[np.ndarray],
    factor: int = 2,
    vis: bool = False,
):
    """Area-downsample Phase A and Phase B masks by ``factor``."""
    downsampled_a = [_area_downsample_2d(m, factor) for m in phase_A_masks]
    downsampled_b = [_area_downsample_2d(m, factor) for m in phase_B_masks]
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
    guide_img: np.ndarray | torch.Tensor | Sequence[np.ndarray | torch.Tensor],
    device: str | torch.device = "cpu",
    normalize: bool = True,
    normalization: GuideNormalization | None = None,
) -> torch.Tensor:
    """Convert a 2D ``(H, W)``, 3D ``(M, H, W)``, 4D ``(B, M, H, W)`` coarse
    physical guide map (or a sequence of 2D maps) into a ``(B, M, H, W)`` float32
    tensor with safe NaN/Inf & negative FBP baseline handling.

    Normalization modes (``normalization`` overrides ``normalize`` when set):
    - ``"minmax"`` (default when ``normalize=True``): Clamps negative FBP/ringing
      baselines when positive signal exists, then scales each channel into ``[0, 1]``
      (zero-variance channels map to ``0.0``).
    - ``"preserve_counts"`` (default when ``normalize=False``): Clamps negative
      detector baseline artifacts to ``0.0`` while preserving raw physical photon/X-ray
      counts for ``loss_fn="forward_psf_poisson"``.
    - ``"global_max"``: Divides all channels jointly by the global maximum across
      channels so relative stoichiometry and zero/trace channels are preserved!
    - ``"simplex"``: Normalizes non-negative channels per pixel so ``sum_m G_m = 1``
      (when total > 0) for ``loss_fn="compositional_kl"``.
    """
    import warnings

    if isinstance(guide_img, (list, tuple)):
        stacked = [
            g.detach().cpu().float().numpy()
            if isinstance(g, torch.Tensor)
            else np.asarray(g, dtype=np.float32)
            for g in guide_img
        ]
        t = torch.from_numpy(np.stack(stacked, axis=0))
    elif isinstance(guide_img, torch.Tensor):
        t = guide_img.detach().clone().to(dtype=torch.float32)
    else:
        t = torch.from_numpy(np.asarray(guide_img, dtype=np.float32))

    if t.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.ndim == 3:
        t = t.unsqueeze(0)
    elif t.ndim != 4:
        raise ValueError(f"Expected 2D, 3D, or 4D guide map, got shape {t.shape}.")

    if not torch.isfinite(t).all():
        warnings.warn(
            "Non-finite (NaN/Inf) pixels detected in coarse guide; replacing with 0.0.",
            UserWarning,
            stacklevel=2,
        )
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    mode: GuideNormalization = (
        normalization
        if normalization is not None
        else ("minmax" if normalize else "preserve_counts")
    )

    if mode == "minmax":
        # Clamp negative FBP/ringing baselines so absent-phase noise <= 0 maps to 0.0
        t = t.clamp(min=0.0)
        flat = t.flatten(start_dim=2)
        mn = flat.min(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
        mx = flat.max(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
        span = mx - mn
        t = torch.where(
            span > 1e-12, (t - mn) / span.clamp(min=1e-12), torch.zeros_like(t)
        )
    elif mode == "preserve_counts":
        t = t.clamp(min=0.0)
    elif mode == "global_max":
        t_pos = t.clamp(min=0.0)
        g_max = t_pos.flatten(start_dim=1).max(dim=-1).values.view(-1, 1, 1, 1)
        t = torch.where(
            g_max > 1e-6, t_pos / g_max.clamp(min=1e-6), torch.zeros_like(t_pos)
        )
    elif mode == "simplex":
        t_pos = t.clamp(min=0.0)
        denom = t_pos.sum(dim=1, keepdim=True)
        t = torch.where(
            denom > 1e-12, t_pos / denom.clamp(min=1e-12), torch.zeros_like(t_pos)
        )
    else:
        raise ValueError(f"Unknown guide normalization mode '{mode}'.")

    return t.to(device=device, dtype=torch.float32)


def extract_guide_physical_scales(
    guide_img: np.ndarray | torch.Tensor | Sequence[np.ndarray | torch.Tensor],
    *,
    normalize: bool = True,
    normalization: GuideNormalization | None = None,
) -> list[dict[str, float]]:
    """Record per-channel non-negative physical guide statistics ``(offset, scale,
    raw_mean, raw_max)`` prior to normalization so high-resolution predicted
    probabilities can reconstruct calibrated physical intensity units.
    """
    if isinstance(guide_img, (list, tuple)):
        stacked = [
            g.detach().cpu().float().numpy()
            if isinstance(g, torch.Tensor)
            else np.asarray(g, dtype=np.float32)
            for g in guide_img
        ]
        arr = np.stack(stacked, axis=0)
    elif isinstance(guide_img, torch.Tensor):
        arr = guide_img.detach().cpu().float().numpy()
    else:
        arr = np.asarray(guide_img, dtype=np.float32)

    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim == 4:
        arr = arr[0]
    elif arr.ndim != 3:
        raise ValueError(f"Expected 2D, 3D, or 4D guide map, got shape {arr.shape}.")

    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    pos = np.maximum(clean, 0.0)
    mode: GuideNormalization = (
        normalization
        if normalization is not None
        else ("minmax" if normalize else "preserve_counts")
    )
    g_max = float(np.max(pos)) if pos.size > 0 else 0.0

    scales: list[dict[str, float]] = []
    for m in range(pos.shape[0]):
        ch = pos[m]
        mn = float(np.min(ch)) if ch.size > 0 else 0.0
        mx = float(np.max(ch)) if ch.size > 0 else 0.0
        rm = float(np.mean(ch)) if ch.size > 0 else 0.0
        if mode == "minmax":
            offset = mn
            scale = mx - mn
        elif mode == "global_max":
            offset = 0.0
            scale = g_max
        else:
            offset = 0.0
            scale = 1.0
        scales.append(
            {
                "offset": offset,
                "scale": scale,
                "raw_mean": rm,
                "raw_max": mx,
            }
        )
    return scales


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
