"""Competing multimodal super-resolution, pan-sharpening, variational inverse,
and Implicit Neural Representation (INR) baselines from adjacent communities:

1. ``gram_schmidt_pansharpen_baseline``: Classical remote-sensing Gram-Schmidt
   component-substitution pan-sharpening injecting high-frequency spatial detail
   from ``I_HR`` into upsampled ``G_LR``.
2. ``cnmf_pansharpen_baseline``: Coupled Non-negative Matrix Factorization
   (``CNMF``, Yokoya et al.) alternating spectral/spatial dictionary unmixing
   between structural intensity bands and coarse physical channels.
3. ``guided_variational_sr_baseline``: Edge-weighted Total Variation physical
   inverse solver minimizing ``||D_Sigma(P_HR) - G_LR||_2^2 + lambda * TV_w(P_HR)``
   (representing variational microscopy / ``SCICO`` / ``edxia`` solvers).
4. ``coordinate_inr_baseline``: Continuous ``SIREN`` sinusoidal coordinate MLP
   ``f_theta(y, x, I_HR(y, x))`` trained from scratch through ``ForwardPSFDownsampler``
   (representing Deep Image Prior / coordinate INR correlative fusion).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from x_fuse.uncertainty import (
    UncertaintyReport,
    build_uncertainty_report,
    match_coarse_priors,
)
from x_fuse.upsamplers import ForwardPSFDownsampler
from x_fuse.utils import guide_to_tensor


def _probs_to_report(
    probs_fg: torch.Tensor,
    guide_t: torch.Tensor,
    mode: Literal["multiclass", "multilabel"],
    method_name: str,
) -> UncertaintyReport:
    eps = 1e-5
    p_clamped = probs_fg.float().clamp(eps, 1.0 - eps)
    logits = torch.log(p_clamped / (1.0 - p_clamped))
    if mode == "multiclass":
        bg = torch.zeros_like(logits[:, :1])
        probs = match_coarse_priors(
            torch.cat([bg, logits], dim=1), guide_t, mode="multiclass"
        )
    else:
        probs = match_coarse_priors(logits, guide_t, mode="multilabel")
    return build_uncertainty_report(
        probs_hr=probs,
        guide_lr=guide_t,
        mode=mode,
        diagnostics={"baseline": method_name},
    )


@torch.no_grad()
def gram_schmidt_pansharpen_baseline(
    hr_image: np.ndarray | torch.Tensor,
    coarse_guide: np.ndarray | torch.Tensor,
    mode: Literal["multiclass", "multilabel"] = "multiclass",
) -> UncertaintyReport:
    """Gram-Schmidt orthogonal component-substitution pan-sharpening."""
    hr_np = (
        hr_image.detach().cpu().float().numpy()
        if isinstance(hr_image, torch.Tensor)
        else np.asarray(hr_image, dtype=np.float32)
    )
    if hr_np.ndim == 3:
        hr_np = hr_np.mean(axis=0)
    h_hr, w_hr = hr_np.shape
    pan = (hr_np - hr_np.min()) / max(float(hr_np.max() - hr_np.min()), 1e-6)

    g_t = guide_to_tensor(coarse_guide, normalize=True)  # (1, M, H_lr, W_lr)
    ms_up = F.interpolate(g_t, size=(h_hr, w_hr), mode="bicubic", align_corners=False)[
        0
    ].numpy()  # (M, H, W)
    m_ch = ms_up.shape[0]

    # Simulated low-resolution panchromatic band via regression on ms_up
    x_ms = ms_up.reshape(m_ch, -1).T  # (N, M)
    x_design = np.concatenate(
        [np.ones((x_ms.shape[0], 1), dtype=np.float32), x_ms], axis=1
    )
    y_pan = pan.reshape(-1, 1)
    weights, _, _, _ = np.linalg.lstsq(x_design, y_pan, rcond=None)
    pan_sim = (x_design @ weights).reshape(-1)
    delta_pan = y_pan.reshape(-1) - pan_sim
    var_sim = max(float(np.var(pan_sim)), 1e-6)

    sharpened = np.zeros_like(ms_up)
    for m in range(m_ch):
        band = ms_up[m].reshape(-1)
        cov = float(np.mean((band - band.mean()) * (pan_sim - pan_sim.mean())))
        gain = np.clip(cov / var_sim, -2.0, 2.0)
        sharpened[m] = np.clip(band + gain * delta_pan, 0.0, 1.0).reshape(h_hr, w_hr)

    probs_fg = torch.from_numpy(sharpened).unsqueeze(0)
    return _probs_to_report(probs_fg, g_t, mode=mode, method_name="gram_schmidt")


@torch.no_grad()
def cnmf_pansharpen_baseline(
    hr_image: np.ndarray | torch.Tensor,
    coarse_guide: np.ndarray | torch.Tensor,
    mode: Literal["multiclass", "multilabel"] = "multiclass",
    n_endmembers: int = 4,
    n_iters: int = 35,
) -> UncertaintyReport:
    """Coupled Non-negative Matrix Factorization (CNMF, Yokoya et al.) fusing
    multi-band structural intensity channels ``[I_HR, 1 - I_HR, I_HR^2]`` with
    coarse physical guide ``G_LR``.
    """
    hr_np = (
        hr_image.detach().cpu().float().numpy()
        if isinstance(hr_image, torch.Tensor)
        else np.asarray(hr_image, dtype=np.float32)
    )
    if hr_np.ndim == 3:
        hr_np = hr_np.mean(axis=0)
    h_hr, w_hr = hr_np.shape
    pan = (hr_np - hr_np.min()) / max(float(hr_np.max() - hr_np.min()), 1e-6)
    # Construct non-negative HR radiometric observation matrix Y_M: (C_hr, N_hr)
    hr_bands = np.stack([pan, 1.0 - pan, pan**2, (1.0 - pan) ** 2], axis=0).astype(
        np.float32
    )
    y_hr = torch.from_numpy(hr_bands.reshape(4, -1)).clamp(min=1e-5)

    g_t = guide_to_tensor(coarse_guide, normalize=True)  # (1, M, H_lr, W_lr)
    m_ch = int(g_t.shape[1])
    height_lr, width_lr = int(g_t.shape[2]), int(g_t.shape[3])
    # Augment coarse guide with implicit matrix channel (1 - max_m G_m)
    bg_lr = (1.0 - g_t[0].amax(dim=0, keepdim=True)).clamp(0.0, 1.0)
    y_lr_4d = torch.cat([bg_lr, g_t[0]], dim=0).unsqueeze(0)  # (1, M+1, H_lr, W_lr)
    c_lr = m_ch + 1
    y_lr = y_lr_4d.reshape(c_lr, -1).clamp(min=1e-5)  # (C_lr, N_lr)

    k_end = max(c_lr, int(n_endmembers))
    gen = torch.Generator(device="cpu").manual_seed(0)
    dict_lr = torch.rand((c_lr, k_end), generator=gen) + 0.1
    # Anchor first c_lr endmembers to identity-like pure phases
    dict_lr[:, :c_lr] = torch.eye(c_lr) + 0.05
    h_lr_ab = torch.full((k_end, height_lr * width_lr), 1.0 / k_end)

    eps = 1e-7
    for _ in range(max(5, n_iters)):
        h_lr_ab = h_lr_ab * (
            (dict_lr.T @ y_lr) / ((dict_lr.T @ dict_lr) @ h_lr_ab).clamp(min=eps)
        )
        h_lr_ab = h_lr_ab / h_lr_ab.sum(dim=0, keepdim=True).clamp(min=eps)
        dict_lr = dict_lr * (
            (y_lr @ h_lr_ab.T) / (dict_lr @ (h_lr_ab @ h_lr_ab.T)).clamp(min=eps)
        )

    # Upsample coarse abundances to initialize HR abundances H_HR: (K, N_hr)
    h_hr_init = (
        F.interpolate(
            h_lr_ab.reshape(1, k_end, height_lr, width_lr),
            size=(h_hr, w_hr),
            mode="bilinear",
            align_corners=False,
        )
        .reshape(k_end, -1)
        .clamp(min=1e-5)
    )
    h_hr_ab = h_hr_init.clone()

    # Estimate HR spectral response W_HR: (4, K) via pooled HR bands vs H_LR
    hr_pooled = F.adaptive_avg_pool2d(
        torch.from_numpy(hr_bands).unsqueeze(0), output_size=(height_lr, width_lr)
    ).reshape(4, -1)
    dict_hr = torch.rand((4, k_end), generator=gen) + 0.1
    for _ in range(max(5, n_iters)):
        dict_hr = dict_hr * (
            (hr_pooled @ h_lr_ab.T) / (dict_hr @ (h_lr_ab @ h_lr_ab.T)).clamp(min=eps)
        )

    # Refine H_HR against Y_HR while softly tethering to coarse spatial consistency
    for _ in range(max(5, n_iters)):
        num = dict_hr.T @ y_hr + 0.15 * h_hr_init
        den = ((dict_hr.T @ dict_hr) @ h_hr_ab + 0.15 * h_hr_ab).clamp(min=eps)
        h_hr_ab = (h_hr_ab * (num / den)).clamp(min=eps)
        h_hr_ab = h_hr_ab / h_hr_ab.sum(dim=0, keepdim=True).clamp(min=eps)

    recon_hr = (dict_lr @ h_hr_ab).reshape(1, c_lr, h_hr, w_hr)
    recon_hr = recon_hr / recon_hr.sum(dim=1, keepdim=True).clamp(min=eps)
    probs_fg = recon_hr[:, 1:].clamp(0.0, 1.0)
    return _probs_to_report(probs_fg, g_t, mode=mode, method_name="cnmf")


def guided_variational_sr_baseline(
    hr_image: np.ndarray | torch.Tensor,
    coarse_guide: np.ndarray | torch.Tensor,
    mode: Literal["multiclass", "multilabel"] = "multiclass",
    n_iters: int = 45,
    n_steps: int | None = None,
    tv_weight: float = 0.04,
    lr: float = 0.12,
    device: str | torch.device = "cpu",
) -> UncertaintyReport:
    """Variational inverse solver minimizing forward-PSF data fidelity plus
    intensity-edge-weighted anisotropic Total Variation:
    ``||D_Sigma(P_HR) - G_LR||_2^2 + tv_weight * TV_{w(I_HR)}(P_HR)``.
    """
    steps = int(n_steps) if n_steps is not None else int(n_iters)
    hr_t = (
        hr_image.detach().clone().float().to(device)
        if isinstance(hr_image, torch.Tensor)
        else torch.from_numpy(np.asarray(hr_image, dtype=np.float32)).to(device)
    )
    if hr_t.ndim == 3:
        hr_t = hr_t.mean(dim=0)
    h_hr, w_hr = int(hr_t.shape[0]), int(hr_t.shape[1])
    pan = (hr_t - hr_t.min()) / (hr_t.max() - hr_t.min()).clamp(min=1e-6)

    g_t = guide_to_tensor(coarse_guide, device=device, normalize=True)
    lr_hw = (int(g_t.shape[-2]), int(g_t.shape[-1]))
    init_up = F.interpolate(
        g_t, size=(h_hr, w_hr), mode="bilinear", align_corners=False
    ).clamp(0.02, 0.98)
    logits = nn.Parameter(torch.log(init_up / (1.0 - init_up)))

    wy = torch.exp(-15.0 * (pan[1:, :] - pan[:-1, :]).abs()).view(1, 1, h_hr - 1, w_hr)
    wx = torch.exp(-15.0 * (pan[:, 1:] - pan[:, :-1]).abs()).view(1, 1, h_hr, w_hr - 1)
    ds = ForwardPSFDownsampler().to(device)
    opt = torch.optim.Adam([logits], lr=lr)

    with torch.enable_grad():
        for _ in range(max(1, steps)):
            opt.zero_grad()
            probs = torch.sigmoid(logits)
            fid = F.mse_loss(ds(probs, lr_hw), g_t)
            dy = (probs[:, :, 1:, :] - probs[:, :, :-1, :]).abs()
            dx = (probs[:, :, :, 1:] - probs[:, :, :, :-1]).abs()
            tv = (wy * dy).mean() + (wx * dx).mean()
            loss = fid + float(tv_weight) * tv
            loss.backward()
            opt.step()

    with torch.no_grad():
        probs_fg = torch.sigmoid(logits.detach()).cpu()
    return _probs_to_report(
        probs_fg, g_t.cpu(), mode=mode, method_name="guided_variational_sr"
    )


class _SineLayer(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, omega_0: float = 15.0, is_first: bool = False
    ):
        super().__init__()
        self.omega_0 = float(omega_0)
        self.linear = nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            bound = (
                (1.0 / in_dim) if is_first else (math.sqrt(6.0 / in_dim) / self.omega_0)
            )
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


def coordinate_inr_baseline(
    hr_image: np.ndarray | torch.Tensor,
    coarse_guide: np.ndarray | torch.Tensor,
    mode: Literal["multiclass", "multilabel"] = "multiclass",
    epochs: int = 60,
    hidden_dim: int = 48,
    lr: float = 5e-3,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> UncertaintyReport:
    """``SIREN`` Implicit Neural Representation baseline mapping continuous
    coordinates + structural intensity ``(y, x, I_HR(y, x))`` to HR phase logits
    supervised strictly through ``ForwardPSFDownsampler``.
    """
    torch.manual_seed(int(seed))
    hr_t = (
        hr_image.detach().clone().float().to(device)
        if isinstance(hr_image, torch.Tensor)
        else torch.from_numpy(np.asarray(hr_image, dtype=np.float32)).to(device)
    )
    if hr_t.ndim == 3:
        hr_t = hr_t.mean(dim=0)
    h_hr, w_hr = int(hr_t.shape[0]), int(hr_t.shape[1])
    pan = (hr_t - hr_t.min()) / (hr_t.max() - hr_t.min()).clamp(min=1e-6)

    ys = torch.linspace(-1.0, 1.0, h_hr, device=device)
    xs = torch.linspace(-1.0, 1.0, w_hr, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords_in = torch.stack([yy, xx, 2.0 * pan - 1.0], dim=-1).reshape(-1, 3)

    g_t = guide_to_tensor(coarse_guide, device=device, normalize=True)
    m_ch = int(g_t.shape[1])
    lr_hw = (int(g_t.shape[-2]), int(g_t.shape[-1]))

    net = nn.Sequential(
        _SineLayer(3, hidden_dim, omega_0=12.0, is_first=True),
        _SineLayer(hidden_dim, hidden_dim, omega_0=12.0),
        nn.Linear(hidden_dim, m_ch),
    ).to(device)
    ds = ForwardPSFDownsampler().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    with torch.enable_grad():
        for _ in range(max(1, int(epochs))):
            opt.zero_grad()
            logits_hr = net(coords_in).T.reshape(1, m_ch, h_hr, w_hr)
            probs_hr = torch.sigmoid(logits_hr)
            loss = F.mse_loss(ds(probs_hr, lr_hw), g_t)
            loss.backward()
            opt.step()

    with torch.no_grad():
        probs_fg = torch.sigmoid(net(coords_in).T.reshape(1, m_ch, h_hr, w_hr)).cpu()
    return _probs_to_report(probs_fg, g_t.cpu(), mode=mode, method_name="siren_inr")
