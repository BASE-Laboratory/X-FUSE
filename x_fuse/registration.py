"""Differentiable sub-pixel translation, rigid, affine, and smooth cubic B-spline
deformable co-registration aligning high-resolution structural predictions to
drifted or scan-distorted coarse physical guides.
"""

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F

AlignMode: TypeAlias = Literal["none", "translation", "rigid", "affine", "bspline"]


class DifferentiableGuideAligner(nn.Module):
    """Bounded differentiable spatial warp ``W_phi(P_HR)`` modelling stage drift,
    beam shift, slight rotation/magnification mismatch, or nonlinear SEM/STEM
    charging and raster flyback distortion (``mode="bspline"``) between
    ``I_HR`` and ``G_LR``.

    Parameters are constrained via ``tanh`` and (for ``bspline``) second-order
    thin-plate bending-energy regularization so optimization stays diffeomorphic
    (``min det(J) > 0``) even when supervised solely by coarse (8x-16x) guides.
    """

    def __init__(
        self,
        mode: AlignMode = "none",
        max_shift_coarse_px: float = 1.5,
        max_rotation_deg: float = 5.0,
        max_scale_delta: float = 0.05,
        control_grid_size: int = 6,
        reg_weight: float = 1e-3,
    ) -> None:
        super().__init__()
        if mode not in ("none", "translation", "rigid", "affine", "bspline"):
            raise ValueError(
                f"Invalid align mode '{mode}'. Choose 'none', 'translation', "
                "'rigid', 'affine', or 'bspline'."
            )
        self.mode: AlignMode = mode
        self.max_shift_coarse_px = float(max_shift_coarse_px)
        self.max_rotation_rad = math.radians(float(max_rotation_deg))
        self.max_scale_delta = float(max_scale_delta)
        self.control_grid_size = max(3, int(control_grid_size))
        self.reg_weight = float(reg_weight)

        if mode in ("translation", "rigid", "affine", "bspline"):
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

        if mode == "bspline":
            self.raw_control_grid = nn.Parameter(
                torch.zeros(
                    (1, 2, self.control_grid_size, self.control_grid_size),
                    dtype=torch.float32,
                )
            )
        else:
            self.register_parameter("raw_control_grid", None)

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
        max_ty = 2.0 * self.max_shift_coarse_px / float(h_lr)
        max_tx = 2.0 * self.max_shift_coarse_px / float(w_lr)
        shift_yx = torch.tanh(self.raw_shift.to(device=device, dtype=dtype))
        ty = -max_ty * shift_yx[0]
        tx = -max_tx * shift_yx[1]

        if self.mode in ("translation", "bspline"):
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

    def _bspline_displacement_grid(
        self,
        hr_hw: tuple[int, int],
        lr_hw: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Evaluate smooth bicubic B-spline inverse sampling displacement field
        ``(1, H, W, 2)`` in normalized ``[-1, 1]`` ``(x, y)`` coordinates.
        """
        assert self.raw_control_grid is not None
        h_lr, w_lr = max(lr_hw[0], 1), max(lr_hw[1], 1)
        max_ty = 2.0 * self.max_shift_coarse_px / float(h_lr)
        max_tx = 2.0 * self.max_shift_coarse_px / float(w_lr)

        ctrl = torch.tanh(self.raw_control_grid.to(device=device, dtype=dtype))
        # Channel 0 = dy, Channel 1 = dx in forward image space -> negate for sampling
        disp_y = -max_ty * ctrl[:, 0:1]
        disp_x = -max_tx * ctrl[:, 1:2]
        ctrl_xy = torch.cat([disp_x, disp_y], dim=1)  # (1, 2, Gy, Gx)
        dense_xy = F.interpolate(
            ctrl_xy,
            size=hr_hw,
            mode="bicubic",
            align_corners=True,
        )
        return dense_xy.permute(0, 2, 3, 1)  # (1, H, W, 2)

    def build_sampling_grid(
        self,
        batch_size: int,
        channels: int,
        hr_hw: tuple[int, int],
        lr_hw: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """Build complete ``(B, H, W, 2)`` sampling grid in normalized coords."""
        theta = self._build_theta(
            batch_size=batch_size, lr_hw=lr_hw, device=device, dtype=torch.float32
        )
        grid = F.affine_grid(
            theta, [batch_size, channels, hr_hw[0], hr_hw[1]], align_corners=False
        )
        if self.mode == "bspline":
            disp = self._bspline_displacement_grid(
                hr_hw=hr_hw, lr_hw=lr_hw, device=device, dtype=torch.float32
            )
            grid = grid + disp
        return grid

    def forward(self, hr_map: torch.Tensor, lr_hw: tuple[int, int]) -> torch.Tensor:
        """Warp ``hr_map`` ``(B, C, H, W)`` from ``I_HR`` frame into ``G_LR`` frame."""
        if self.mode == "none":
            return hr_map
        bsz, channels, h_hr, w_hr = hr_map.shape
        grid = self.build_sampling_grid(
            batch_size=bsz,
            channels=channels,
            hr_hw=(h_hr, w_hr),
            lr_hw=lr_hw,
            device=hr_map.device,
        )
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
        """Prior nudging global warp toward identity and penalizing second-order
        thin-plate bending energy ``||d^2 u||_2^2`` for ``mode="bspline"``.
        """
        if self.mode == "none" or self.reg_weight <= 0.0:
            return torch.zeros((), device=device, dtype=torch.float32)
        theta = self._build_theta(1, lr_hw=lr_hw, device=device, dtype=torch.float32)[0]
        eye = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, dtype=torch.float32
        )
        base_pen = (theta - eye).square().sum()

        if self.mode == "bspline" and self.raw_control_grid is not None:
            u = torch.tanh(self.raw_control_grid.to(device=device, dtype=torch.float32))
            d2y = u[:, :, 2:, :] - 2.0 * u[:, :, 1:-1, :] + u[:, :, :-2, :]
            d2x = u[:, :, :, 2:] - 2.0 * u[:, :, :, 1:-1] + u[:, :, :, :-2]
            dyx = (
                u[:, :, 1:, 1:]
                - u[:, :, 1:, :-1]
                - u[:, :, :-1, 1:]
                + u[:, :, :-1, :-1]
            )
            bend = (
                d2y.square().mean()
                + d2x.square().mean()
                + 2.0 * dyx.square().mean()
                + 0.05 * u.square().mean()
            )
            base_pen = base_pen + 5.0 * bend

        return self.reg_weight * base_pen

    @torch.no_grad()
    def min_jacobian_determinant(
        self, hr_hw: tuple[int, int], lr_hw: tuple[int, int], device: torch.device
    ) -> float:
        """Compute minimum spatial Jacobian determinant of the sampling map over
        ``hr_hw`` (``> 0`` certifies a diffeomorphic fold-free warp).
        """
        if self.mode == "none":
            return 1.0
        grid = self.build_sampling_grid(1, 1, hr_hw, lr_hw, device=device)[0]  # (H,W,2)
        h_hr, w_hr = max(hr_hw[0], 2), max(hr_hw[1], 2)
        # Finite differences normalized by step size (2 / H, 2 / W)
        step_x = 2.0 / float(w_hr)
        step_y = 2.0 / float(h_hr)
        dg_dx = (grid[:-1, 1:, :] - grid[:-1, :-1, :]) / step_x
        dg_dy = (grid[1:, :-1, :] - grid[:-1, :-1, :]) / step_y
        det = dg_dx[..., 0] * dg_dy[..., 1] - dg_dx[..., 1] * dg_dy[..., 0]
        return float(det.min().cpu().item())

    @torch.no_grad()
    def diagnostics(
        self, hr_hw: tuple[int, int], lr_hw: tuple[int, int]
    ) -> dict[str, float | str]:
        """Return human-readable drift, B-spline RMS warp, and Jacobian diagnostics."""
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

        if self.mode == "bspline" and self.raw_control_grid is not None:
            dev = self.raw_control_grid.device
            disp = self._bspline_displacement_grid(hr_hw, lr_hw, device=dev)[0].cpu()
            # Convert normalized [-1, 1] offsets back to HR pixels
            dy_px = -disp[..., 1] * (0.5 * hr_hw[0])
            dx_px = -disp[..., 0] * (0.5 * hr_hw[1])
            rms = float((dy_px.square() + dx_px.square()).mean().sqrt().item())
            info["bspline_rms_disp_hr_px"] = round(rms, 4)
            info["bspline_min_jacobian_det"] = round(
                self.min_jacobian_determinant(hr_hw, lr_hw, device=dev), 4
            )
        return info
