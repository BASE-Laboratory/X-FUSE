"""Differentiable sub-pixel translation, rigid, and affine co-registration
aligning high-resolution structural predictions to drifted coarse physical guides.
"""

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F

AlignMode: TypeAlias = Literal["none", "translation", "rigid", "affine"]


class DifferentiableGuideAligner(nn.Module):
    """Bounded differentiable spatial warp ``W_phi(P_HR)`` modelling stage drift,
    beam shift, or slight rotation/magnification mismatch between ``I_HR`` and ``G_LR``.

    Parameters are constrained via ``tanh`` so optimization remains physically
    well-posed even when supervised solely by very coarse (8x-16x) guides.
    """

    def __init__(
        self,
        mode: AlignMode = "none",
        max_shift_coarse_px: float = 1.5,
        max_rotation_deg: float = 5.0,
        max_scale_delta: float = 0.05,
        reg_weight: float = 1e-3,
    ) -> None:
        super().__init__()
        if mode not in ("none", "translation", "rigid", "affine"):
            raise ValueError(
                f"Invalid align mode '{mode}'. Choose 'none', 'translation', "
                "'rigid', or 'affine'."
            )
        self.mode: AlignMode = mode
        self.max_shift_coarse_px = float(max_shift_coarse_px)
        self.max_rotation_rad = math.radians(float(max_rotation_deg))
        self.max_scale_delta = float(max_scale_delta)
        self.reg_weight = float(reg_weight)

        if mode in ("translation", "rigid", "affine"):
            self.raw_shift = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        else:
            self.register_parameter("raw_shift", None)

        if mode in ("rigid", "affine"):
            self.raw_angle = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        else:
            self.register_parameter("raw_angle", None)

        if mode == "affine":
            self.raw_scale_shear = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        else:
            self.register_parameter("raw_scale_shear", None)

    def _build_theta(
        self,
        batch_size: int,
        lr_hw: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Construct ``(B, 2, 3)`` normalized grid_sample affine matrices."""
        h_lr, w_lr = max(lr_hw[0], 1), max(lr_hw[1], 1)
        if self.mode == "none":
            eye = torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype
            )
            return eye.unsqueeze(0).expand(batch_size, -1, -1)

        assert self.raw_shift is not None
        # Normalized [-1, 1] shift corresponding to max_shift_coarse_px
        max_ty = 2.0 * self.max_shift_coarse_px / float(h_lr)
        max_tx = 2.0 * self.max_shift_coarse_px / float(w_lr)
        # Note: grid_sample theta[:, :, 2] is (tx, ty) in normalized x, y coords.
        # To shift an image forward by (+dx, +dy) pixels in output space, the inverse
        # sampling coordinate offset is (-tx, -ty).
        shift_yx = torch.tanh(self.raw_shift.to(device=device, dtype=dtype))
        ty = -max_ty * shift_yx[0]
        tx = -max_tx * shift_yx[1]

        if self.mode == "translation":
            row0 = torch.stack(
                [
                    torch.ones((), device=device, dtype=dtype),
                    torch.zeros((), device=device, dtype=dtype),
                    tx,
                ]
            )
            row1 = torch.stack(
                [
                    torch.zeros((), device=device, dtype=dtype),
                    torch.ones((), device=device, dtype=dtype),
                    ty,
                ]
            )
            theta = torch.stack([row0, row1], dim=0)
            return theta.unsqueeze(0).expand(batch_size, -1, -1)

        assert self.raw_angle is not None
        ang = -self.max_rotation_rad * torch.tanh(
            self.raw_angle[0].to(device=device, dtype=dtype)
        )
        cos_a = torch.cos(ang)
        sin_a = torch.sin(ang)

        if self.mode == "rigid":
            row0 = torch.stack([cos_a, -sin_a, tx])
            row1 = torch.stack([sin_a, cos_a, ty])
            theta = torch.stack([row0, row1], dim=0)
            return theta.unsqueeze(0).expand(batch_size, -1, -1)

        assert self.raw_scale_shear is not None
        ss = self.max_scale_delta * torch.tanh(
            self.raw_scale_shear.to(device=device, dtype=dtype)
        )
        sx = 1.0 + ss[0]
        sy = 1.0 + ss[1]
        shx = ss[2]
        shy = ss[3]
        row0 = torch.stack([sx * cos_a + shx * sin_a, -sx * sin_a + shx * cos_a, tx])
        row1 = torch.stack([sy * sin_a + shy * cos_a, sy * cos_a - shy * sin_a, ty])
        theta = torch.stack([row0, row1], dim=0)
        return theta.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, hr_map: torch.Tensor, lr_hw: tuple[int, int]) -> torch.Tensor:
        """Warp ``hr_map`` ``(B, C, H, W)`` from ``I_HR`` frame into ``G_LR`` frame."""
        if self.mode == "none":
            return hr_map
        bsz = hr_map.shape[0]
        theta = self._build_theta(
            batch_size=bsz,
            lr_hw=lr_hw,
            device=hr_map.device,
            dtype=torch.float32,
        )
        grid = F.affine_grid(theta, list(hr_map.shape), align_corners=False)
        warped = F.grid_sample(
            hr_map.float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return warped.to(hr_map.dtype)

    def regularization_loss(
        self, lr_hw: tuple[int, int], device: torch.device
    ) -> torch.Tensor:
        """L2 prior nudging warp parameters toward the identity transform."""
        if self.mode == "none" or self.reg_weight <= 0.0:
            return torch.zeros((), device=device, dtype=torch.float32)
        theta = self._build_theta(1, lr_hw=lr_hw, device=device, dtype=torch.float32)[0]
        eye = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, dtype=torch.float32
        )
        return self.reg_weight * ((theta - eye).square().sum())

    @torch.no_grad()
    def diagnostics(
        self, hr_hw: tuple[int, int], lr_hw: tuple[int, int]
    ) -> dict[str, float | str]:
        """Return human-readable drift estimates in HR pixels and degrees."""
        info: dict[str, float | str] = {"align_mode": self.mode}
        if self.mode == "none" or self.raw_shift is None:
            info["shift_y_hr_px"] = 0.0
            info["shift_x_hr_px"] = 0.0
            info["rotation_deg"] = 0.0
            return info

        scale_y = hr_hw[0] / max(lr_hw[0], 1)
        scale_x = hr_hw[1] / max(lr_hw[1], 1)
        s_yx = torch.tanh(self.raw_shift.detach().cpu())
        info["shift_y_hr_px"] = round(
            float(s_yx[0].item() * self.max_shift_coarse_px * scale_y), 4
        )
        info["shift_x_hr_px"] = round(
            float(s_yx[1].item() * self.max_shift_coarse_px * scale_x), 4
        )
        if self.raw_angle is not None:
            ang_rad = self.max_rotation_rad * float(
                torch.tanh(self.raw_angle[0].detach().cpu()).item()
            )
            info["rotation_deg"] = round(math.degrees(ang_rad), 4)
        else:
            info["rotation_deg"] = 0.0
        return info
