"""Overlap-tile inference with raised-cosine (Hann) blending for gigapixel /
full-detector 2D micrographs and tomogram cross-sections without GPU OOM.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .transforms import ensure_spatial_divisible


def generate_tile_boxes(
    height: int,
    width: int,
    tile_size: int = 512,
    overlap: int = 64,
) -> list[tuple[int, int, int, int]]:
    """Generate ``(y0, x0, y1, x1)`` bounding boxes covering ``(height, width)``.

    Edge tiles shift inward when ``height > tile_size`` or ``width > tile_size``
    so every tile maintains full spatial context.
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}.")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError(
            f"overlap must satisfy 0 <= overlap < tile_size ({tile_size}), "
            f"got {overlap}."
        )

    step = tile_size - overlap

    def _axis_starts(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        starts = list(range(0, length - tile_size, step))
        last = length - tile_size
        if not starts or starts[-1] != last:
            starts.append(last)
        return starts

    y_starts = _axis_starts(height)
    x_starts = _axis_starts(width)
    th = min(height, tile_size)
    tw = min(width, tile_size)

    boxes: list[tuple[int, int, int, int]] = []
    for y0 in y_starts:
        for x0 in x_starts:
            boxes.append((y0, x0, y0 + th, x0 + tw))
    return boxes


def hann_window_2d(
    height: int,
    width: int,
    overlap: int,
    at_top: bool,
    at_bottom: bool,
    at_left: bool,
    at_right: bool,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    min_weight: float = 1e-3,
) -> torch.Tensor:
    """Construct a 2D raised-cosine blending mask ``(1, 1, height, width)`` that
    tapers smoothly across interior overlap margins while keeping weight ``1.0``
    on outer image boundaries.
    """
    wy = torch.ones(height, device=device, dtype=dtype)
    wx = torch.ones(width, device=device, dtype=dtype)

    ramp_len_y = min(overlap, height // 2)
    ramp_len_x = min(overlap, width // 2)

    if ramp_len_y > 0:
        t_y = torch.linspace(0.0, 1.0, ramp_len_y + 2, device=device, dtype=dtype)[1:-1]
        ramp_y = (0.5 * (1.0 - torch.cos(torch.pi * t_y))).clamp(min=min_weight)
        if not at_top:
            wy[:ramp_len_y] = ramp_y
        if not at_bottom:
            wy[-ramp_len_y:] = ramp_y.flip(0)

    if ramp_len_x > 0:
        t_x = torch.linspace(0.0, 1.0, ramp_len_x + 2, device=device, dtype=dtype)[1:-1]
        ramp_x = (0.5 * (1.0 - torch.cos(torch.pi * t_x))).clamp(min=min_weight)
        if not at_left:
            wx[:ramp_len_x] = ramp_x
        if not at_right:
            wx[-ramp_len_x:] = ramp_x.flip(0)

    w2d = torch.outer(wy, wx).clamp(min=min_weight)
    return w2d.view(1, 1, height, width)


def run_tiled_2d(
    tile_fn: Callable[[torch.Tensor], torch.Tensor],
    image_chw: torch.Tensor,
    tile_size: int = 512,
    overlap: int = 64,
    patch_size: int = 16,
    stride: int = 4,
) -> torch.Tensor:
    """Evaluate ``tile_fn`` over overlapping spatial tiles of a ``(C, H, W)``
    structural tensor and blend outputs ``(B, C_out, H, W)`` seamlessly.
    """
    if image_chw.ndim != 3:
        raise ValueError(
            "Expected 3D (C, H, W) structural tensor, got shape "
            f"{tuple(image_chw.shape)}."
        )
    _, height, width = image_chw.shape
    boxes = generate_tile_boxes(height, width, tile_size=tile_size, overlap=overlap)

    if len(boxes) == 1:
        padded, (oh, ow) = ensure_spatial_divisible(
            image_chw, patch_size=patch_size, stride=stride
        )
        out_single = tile_fn(padded)
        return out_single[:, :, :oh, :ow]

    accum: torch.Tensor | None = None
    weight_sum: torch.Tensor | None = None
    out_dtype: torch.dtype = torch.float32

    for y0, x0, y1, x1 in boxes:
        sub = image_chw[:, y0:y1, x0:x1]
        padded_sub, (th, tw) = ensure_spatial_divisible(
            sub, patch_size=patch_size, stride=stride
        )
        tile_out = tile_fn(padded_sub)[:, :, :th, :tw]
        out_dtype = tile_out.dtype
        tile_f32 = tile_out.float()

        win = hann_window_2d(
            height=th,
            width=tw,
            overlap=overlap,
            at_top=(y0 == 0),
            at_bottom=(y1 == height),
            at_left=(x0 == 0),
            at_right=(x1 == width),
            device=tile_f32.device,
            dtype=torch.float32,
        )

        if accum is None or weight_sum is None:
            bsz, out_ch, _, _ = tile_f32.shape
            accum = torch.zeros(
                (bsz, out_ch, height, width),
                device=tile_f32.device,
                dtype=torch.float32,
            )
            weight_sum = torch.zeros(
                (1, 1, height, width),
                device=tile_f32.device,
                dtype=torch.float32,
            )

        accum[:, :, y0:y1, x0:x1] += tile_f32 * win
        weight_sum[:, :, y0:y1, x0:x1] += win

    assert accum is not None and weight_sum is not None
    blended = accum / weight_sum.clamp(min=1e-6)
    return blended.to(out_dtype)
