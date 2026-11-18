"""Invertible test-time augmentation (TTA) transforms and modality-agnostic
structural image preprocessing for dense Vision Transformer feature extraction.

Portions adapted from HR-Dv2 (https://github.com/tldr-group/HR-Dv2, MIT License).
"""

from functools import partial
from itertools import product
from typing import Literal, TypeAlias

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

PartialTrs: TypeAlias = list[partial]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def iden(x: torch.Tensor) -> torch.Tensor:
    """Identity transform ensuring a 4D batched tensor ``(B, C, H, W)``."""
    if x.ndim == 3:
        return x.unsqueeze(0)
    return x


def true_iden(x: torch.Tensor) -> torch.Tensor:
    """Strict identity transform preserving input shape."""
    return x


iden_partial = partial(iden)
true_iden_partial = partial(true_iden)


def _roll_spatial(shift: tuple[int, int], x: torch.Tensor) -> torch.Tensor:
    return torch.roll(x, shift, dims=(-2, -1))


def _flip_spatial(dims: tuple[int, ...], x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return torch.flip(x, dims)


def _rot90_spatial(dims: tuple[int, int], k: int, x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return torch.rot90(x, k, dims)


def _compose_two(tr_1: partial, tr_2: partial, x: torch.Tensor) -> torch.Tensor:
    return tr_2(tr_1(x))


def compute_shift_directions(
    pattern: Literal["Neumann", "Moore"] = "Moore",
) -> list[tuple[int, int]]:
    """Return unit-step ``(dy, dx)`` shift directions for a 2D neighbourhood."""
    if pattern not in ("Neumann", "Moore"):
        raise ValueError(f"Unknown shift pattern '{pattern}'. Expected Neumann/Moore.")
    shifts = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    directions: list[tuple[int, int]] = []
    for idx, s in enumerate(shifts):
        if pattern == "Neumann" and idx % 2 == 1:
            directions.append(s)
        elif pattern == "Moore" and idx != 4:
            directions.append(s)
    return directions


def get_shift_transforms(
    dists: list[int],
    pattern: Literal["Neumann", "Moore"] = "Moore",
) -> tuple[PartialTrs, PartialTrs]:
    """Build forward and exact inverse spatial roll transforms."""
    fwd_transforms: PartialTrs = [true_iden_partial]
    inv_transforms: PartialTrs = [true_iden_partial]
    shifts = compute_shift_directions(pattern)

    for d in dists:
        for dy, dx in shifts:
            shift = (d * dy, d * dx)
            inv_shift = (-d * dy, -d * dx)
            fwd_transforms.append(partial(_roll_spatial, shift))
            inv_transforms.append(partial(_roll_spatial, inv_shift))
    return fwd_transforms, inv_transforms


def get_flip_transforms() -> tuple[PartialTrs, PartialTrs]:
    """Build forward and inverse horizontal/vertical flip transforms."""
    h_flip = partial(_flip_spatial, (-1,))
    v_flip = partial(_flip_spatial, (-2,))
    fwd = [iden_partial, h_flip, v_flip]
    inv = [iden_partial, h_flip, v_flip]
    return fwd, inv


def get_rotation_transforms() -> tuple[PartialTrs, PartialTrs]:
    """Build forward and inverse 90-degree rotation transforms (0, 90, 180, 270)."""
    fwd: PartialTrs = [iden_partial]
    inv: PartialTrs = [iden_partial]
    for k in (1, 2, 3):
        fwd.append(partial(_rot90_spatial, (-2, -1), k))
        inv.append(partial(_rot90_spatial, (-2, -1), -k))
    return fwd, inv


def combine_transforms(
    fwd_1: PartialTrs,
    fwd_2: PartialTrs,
    inv_1: PartialTrs,
    inv_2: PartialTrs,
) -> tuple[PartialTrs, PartialTrs]:
    """Compose invertible transform lists ``f = f2 o f1`` with ``f^-1 = i1 o i2``."""
    if len(fwd_1) != len(inv_1) or len(fwd_2) != len(inv_2):
        raise ValueError("Forward and inverse transform lists must match in length.")

    fwd: PartialTrs = []
    inv: PartialTrs = []
    for f1, f2 in product(fwd_1, fwd_2):
        fwd.append(partial(_compose_two, f1, f2))
    # Note: inverse of (f2 o f1)(x) = f2(f1(x)) is i1(i2(y))!
    for i1, i2 in product(inv_1, inv_2):
        inv.append(partial(_compose_two, i2, i1))
    return fwd, inv


def build_default_tta(
    shift_dists: list[int] | None = None,
    pattern: Literal["Neumann", "Moore"] = "Neumann",
    include_flips: bool = True,
) -> tuple[PartialTrs, PartialTrs]:
    """Convenience builder for standard sub-patch TTA ensembles."""
    if shift_dists is None:
        shift_dists = [1, 2]
    fwd_s, inv_s = get_shift_transforms(shift_dists, pattern=pattern)
    if not include_flips:
        return fwd_s, inv_s
    fwd_f, inv_f = get_flip_transforms()
    return combine_transforms(fwd_s, fwd_f, inv_s, inv_f)


def normalize_structural(
    img: np.ndarray | torch.Tensor,
    clip_percentiles: tuple[float, float] | None = None,
    imagenet_norm: bool = True,
    device: str | torch.device = "cpu",
    value_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Convert any 2D grayscale or 3-channel scientific image into a ViT-ready
    ``(3, H, W)`` float32 tensor without 8-bit quantization loss.

    When ``value_range=(vmin, vmax)`` is supplied (e.g. global 3D tomogram bounds),
    intensities are linearly mapped from ``[vmin, vmax]`` to ``[0, 1]`` without
    per-slice min-max contrast re-stretching.
    """
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().float().numpy()
    else:
        arr = np.asarray(img, dtype=np.float32)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=0)
    elif arr.ndim == 3:
        if arr.shape[0] == 3:
            pass
        elif arr.shape[0] == 1 and arr.shape[-1] != 3:
            arr = np.repeat(arr, 3, axis=0)
        elif arr.shape[-1] in (1, 3):
            arr = np.transpose(arr, (2, 0, 1))
            if arr.shape[0] == 1:
                arr = np.repeat(arr, 3, axis=0)
        else:
            raise ValueError(
                f"Expected 2D or 1/3-channel 3D image, got shape {arr.shape}."
            )
    else:
        raise ValueError(f"Expected 2D or 3D image, got {arr.ndim}D shape {arr.shape}.")

    if not np.isfinite(arr).all():
        import warnings

        warnings.warn(
            "Non-finite (NaN/Inf) pixels detected in structural HR image; "
            "imputing with finite median.",
            UserWarning,
            stacklevel=2,
        )
        finite_vals = arr[np.isfinite(arr)]
        fill_val = (
            float(np.median(finite_vals))
            if finite_vals.size > 0
            else (float(value_range[0]) if value_range is not None else 0.0)
        )
        arr = np.nan_to_num(arr, nan=fill_val, posinf=fill_val, neginf=fill_val)

    if value_range is not None:
        v_lo, v_hi = float(value_range[0]), float(value_range[1])
        if v_hi - v_lo > 1e-12:
            arr = np.clip((arr - v_lo) / (v_hi - v_lo), 0.0, 1.0)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
    else:
        if clip_percentiles is not None:
            lo, hi = np.percentile(arr, clip_percentiles)
            arr = np.clip(arr, lo, hi)

        mn, mx = float(np.min(arr)), float(np.max(arr))
        if mx - mn > 1e-12:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)

    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    if imagenet_norm:
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        t = (t - mean) / std
    return t.to(device)


# ==================== LEGACY / TORCHVISION HELPERS ====================


def get_input_transform(resize_dim: int, crop_dim: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(resize_dim),
            transforms.CenterCrop(crop_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


to_norm_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

unnormalize = transforms.Normalize(
    mean=tuple(-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD, strict=False)),
    std=tuple(1.0 / s for s in IMAGENET_STD),
)


def centre_crop(crop_h: int, crop_w: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.CenterCrop((crop_h, crop_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def closest_crop(
    h: int, w: int, patch_size: int = 14, to_tensor: bool = True
) -> transforms.Compose:
    new_h = h - (h % patch_size)
    new_w = w - (w % patch_size)
    if to_tensor:
        return transforms.Compose(
            [
                transforms.CenterCrop((new_h, new_w)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return transforms.Compose([transforms.CenterCrop((new_h, new_w))])


def closest_pad(h: int, w: int, patch_size: int = 14) -> transforms.Compose:
    add_h = (patch_size - (h % patch_size)) % patch_size
    add_w = (patch_size - (w % patch_size)) % patch_size
    return transforms.Compose([transforms.Pad((add_w, add_h, 0, 0))])


to_img = transforms.ToPILImage()
to_tensor = transforms.ToTensor()


def load_image(
    path: str, transform: transforms.Compose
) -> tuple[torch.Tensor, Image.Image]:
    image = Image.open(path).convert("RGB")
    tensor = transform(image)
    transformed_img = to_img(unnormalize(tensor))
    return tensor, transformed_img


def to_numpy(x: torch.Tensor, batched: bool = True) -> np.ndarray:
    if batched and x.ndim == 4 and x.shape[0] == 1:
        x = x.squeeze(0)
    return x.detach().cpu().numpy()


def flatten(
    x: torch.Tensor | np.ndarray, h: int, w: int, c: int, convert: bool = False
) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = to_numpy(x)
    return x.reshape((c, h * w)).T


def ensure_spatial_divisible(
    x: torch.Tensor, patch_size: int, stride: int
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad spatial tensor ``(..., H, W)`` by reflection if ``(H - patch_size)``
    is not divisible by ``stride``, returning ``(padded, (orig_h, orig_w))``."""
    orig_h, orig_w = x.shape[-2], x.shape[-1]
    rem_h = (
        (orig_h - patch_size) % stride if orig_h >= patch_size else patch_size - orig_h
    )
    rem_w = (
        (orig_w - patch_size) % stride if orig_w >= patch_size else patch_size - orig_w
    )
    pad_h = (stride - rem_h) % stride if orig_h >= patch_size else rem_h
    pad_w = (stride - rem_w) % stride if orig_w >= patch_size else rem_w
    if pad_h == 0 and pad_w == 0:
        return x, (orig_h, orig_w)
    pad_mode = "reflect" if (pad_h < orig_h and pad_w < orig_w) else "replicate"
    padded = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
    return padded, (orig_h, orig_w)
