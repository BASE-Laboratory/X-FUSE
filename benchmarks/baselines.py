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
from x_fuse.utils import GuideNormalization, guide_to_tensor


def _hr_to_pan_tensor(
    hr_image: np.ndarray | torch.Tensor, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """Convert 2D ``(H, W)``, channels-first ``(C, H, W)``, or channels-last
    ``(H, W, C)`` structural image into a normalized ``[0, 1]`` 2D tensor.
    """
    hr_t = (
        hr_image.detach().clone().float().to(device)
        if isinstance(hr_image, torch.Tensor)
        else torch.from_numpy(np.asarray(hr_image, dtype=np.float32)).to(device)
    )
    hr_t = torch.nan_to_num(hr_t, nan=0.0, posinf=1.0, neginf=0.0)
    if hr_t.ndim == 4:
        hr_t = hr_t[0]
    if hr_t.ndim == 3:
        if hr_t.shape[-1] in (1, 2, 3, 4) and hr_t.shape[0] > 4 and hr_t.shape[1] > 4:
            hr_t = hr_t.mean(dim=-1)
        else:
            hr_t = hr_t.mean(dim=0)
    if hr_t.ndim != 2:
        raise ValueError(f"Expected 2D or 3D hr_image, got shape {tuple(hr_t.shape)}.")
    span = (hr_t.max() - hr_t.min()).clamp(min=1e-6)
    return (hr_t - hr_t.min()) / span


def _probs_to_report(
    probs_fg: torch.Tensor,
    guide_t: torch.Tensor,
    mode: Literal["multiclass", "multilabel"],
    method_name: str,
    calibrate: bool = True,
) -> UncertaintyReport:
    eps = 1e-5
    p_clamped = probs_fg.float().clamp(eps, 1.0 - eps)
    if calibrate:
        logits = torch.log(p_clamped / (1.0 - p_clamped))
        if mode == "multiclass":
            bg = torch.zeros_like(logits[:, :1])
            probs = match_coarse_priors(
                torch.cat([bg, logits], dim=1), guide_t, mode="multiclass"
            )
        else:
            probs = match_coarse_priors(logits, guide_t, mode="multilabel")
    elif mode == "multiclass":
        bg = (1.0 - p_clamped.sum(dim=1, keepdim=True)).clamp(0.0, 1.0)
        probs = torch.cat([bg, p_clamped], dim=1)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
    else:
        probs = probs_fg.float().clamp(0.0, 1.0)
    return build_uncertainty_report(
        probs_hr=probs,
        guide_lr=guide_t,
        mode=mode,
        diagnostics={"baseline": method_name},
    )


@torch.no_grad()
def otsu_threshold_baseline(
    hr_image: np.ndarray | torch.Tensor,
    coarse_guide: np.ndarray | torch.Tensor,
    mode: Literal["multiclass", "multilabel"] = "multiclass",
    n_bins: int = 128,
    guide_normalization: GuideNormalization = "minmax",
    calibrate: bool = True,
) -> UncertaintyReport:
    """Multi-Otsu 1D structural intensity thresholding baseline matched to coarse
    guide channels by spatial overlap (testing how much of a segmentation task can
    be solved from raw grayscale attenuation/Z-contrast alone).
    """
    pan_t = _hr_to_pan_tensor(hr_image, device="cpu")
    pan = pan_t.numpy()
    h_hr, w_hr = pan.shape

    g_t = guide_to_tensor(
        coarse_guide, normalization=guide_normalization
    )  # (1, M, H_lr, W_lr)
    m_ch = int(g_t.shape[1])
    h_lr, w_lr = int(g_t.shape[2]), int(g_t.shape[3])
    n_classes = m_ch + 1

    hist, bin_edges = np.histogram(pan.ravel(), bins=n_bins, range=(0.0, 1.0))
    prob_b = hist.astype(np.float64) / max(float(pan.size), 1.0)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    omega_cum = np.cumsum(prob_b)
    mu_cum = np.cumsum(prob_b * centers)
    mu_tot = float(mu_cum[-1])

    if n_classes == 2:
        denom = omega_cum * (1.0 - omega_cum)
        numer = (mu_tot * omega_cum - mu_cum) ** 2
        sigma_b2 = np.where(denom > 1e-12, numer / np.maximum(denom, 1e-12), -1.0)
        t1 = int(np.argmax(sigma_b2[:-1]))
        thresholds = [float(bin_edges[t1 + 1])]
    elif n_classes == 3:
        best_val = -1.0
        best_t = (n_bins // 3, (2 * n_bins) // 3)
        for i in range(1, n_bins - 2):
            w0 = omega_cum[i]
            if w0 < 1e-9:
                continue
            m0 = mu_cum[i] / w0
            w1_vec = omega_cum[i + 1 : n_bins - 1] - w0
            w2_vec = 1.0 - omega_cum[i + 1 : n_bins - 1]
            valid = (w1_vec > 1e-9) & (w2_vec > 1e-9)
            if not np.any(valid):
                continue
            m1_vec = (mu_cum[i + 1 : n_bins - 1] - mu_cum[i]) / np.maximum(w1_vec, 1e-9)
            m2_vec = (mu_tot - mu_cum[i + 1 : n_bins - 1]) / np.maximum(w2_vec, 1e-9)
            sb2 = (
                w0 * (m0 - mu_tot) ** 2
                + w1_vec * (m1_vec - mu_tot) ** 2
                + w2_vec * (m2_vec - mu_tot) ** 2
            )
            sb2 = np.where(valid, sb2, -1.0)
            j_rel = int(np.argmax(sb2))
            if float(sb2[j_rel]) > best_val:
                best_val = float(sb2[j_rel])
                best_t = (i, i + 1 + j_rel)
        thresholds = [
            float(bin_edges[best_t[0] + 1]),
            float(bin_edges[best_t[1] + 1]),
        ]
    else:
        qs = np.linspace(0.0, 1.0, n_classes + 1)[1:-1]
        thresholds = [float(q) for q in np.quantile(pan, qs)]

    labels_1d = np.digitize(pan, thresholds)  # 0 .. K-1
    cluster_means = np.array(
        [
            float(pan[labels_1d == k].mean())
            if np.any(labels_1d == k)
            else (k + 0.5) / n_classes
            for k in range(n_classes)
        ],
        dtype=np.float32,
    )
    sigma_i = max(0.08, 0.5 / n_classes)
    soft_k = np.exp(
        -0.5 * ((pan[None, :, :] - cluster_means[:, None, None]) / sigma_i) ** 2
    ).astype(np.float32)
    soft_k /= np.maximum(soft_k.sum(axis=0, keepdims=True), 1e-6)
    soft_t = torch.from_numpy(soft_k).unsqueeze(0)  # (1, K, H, W)
    pooled_k = F.adaptive_avg_pool2d(soft_t, output_size=(h_lr, w_lr))[0]  # (K, h, w)

    # Greedily assign distinct intensity clusters to coarse guide channels by cosine
    g_flat = g_t[0].reshape(m_ch, -1)
    p_flat = pooled_k.reshape(n_classes, -1)
    g_norm = g_flat / g_flat.norm(dim=1, keepdim=True).clamp(min=1e-6)
    p_norm = p_flat / p_flat.norm(dim=1, keepdim=True).clamp(min=1e-6)
    sim = (g_norm @ p_norm.T).numpy()  # (M, K)

    used_k: set[int] = set()
    chosen = []
    # Match strongest guide channels first so distinct clusters are claimed cleanly
    order = np.argsort(-sim.max(axis=1))
    assigned: dict[int, int] = {}
    for m_idx in order:
        m = int(m_idx)
        ranked_k = np.argsort(-sim[m])
        pick = int(ranked_k[0])
        for cand in ranked_k:
            if int(cand) not in used_k:
                pick = int(cand)
                break
        used_k.add(pick)
        assigned[m] = pick

    for m in range(m_ch):
        chosen.append(soft_t[:, assigned[m] : assigned[m] + 1])
    probs_fg = torch.cat(chosen, dim=1)
    return _probs_to_report(
        probs_fg, g_t, mode=mode, method_name="otsu_threshold", calibrate=calibrate
    )


@torch.no_grad()
def gram_schmidt_pansharpen_baseline(
    hr_image: np.ndarray | torch.Tensor,
    coarse_guide: np.ndarray | torch.Tensor,
    mode: Literal["multiclass", "multilabel"] = "multiclass",
    guide_normalization: GuideNormalization = "minmax",
    calibrate: bool = True,
) -> UncertaintyReport:
    """Gram-Schmidt orthogonal component-substitution pan-sharpening."""
    pan_t = _hr_to_pan_tensor(hr_image, device="cpu")
    pan = pan_t.numpy()
    h_hr, w_hr = pan.shape

    g_t = guide_to_tensor(
        coarse_guide, normalization=guide_normalization
    )  # (1, M, H_lr, W_lr)
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
    return _probs_to_report(
        probs_fg, g_t, mode=mode, method_name="gram_schmidt", calibrate=calibrate
    )


@torch.no_grad()
def cnmf_pansharpen_baseline(
    hr_image: np.ndarray | torch.Tensor,
    coarse_guide: np.ndarray | torch.Tensor,
    mode: Literal["multiclass", "multilabel"] = "multiclass",
    n_endmembers: int = 4,
    n_iters: int = 35,
    device: str | torch.device = "cpu",
    guide_normalization: GuideNormalization = "minmax",
    calibrate: bool = True,
) -> UncertaintyReport:
    """Coupled Non-negative Matrix Factorization (CNMF, Yokoya et al.) fusing
    multi-band structural intensity channels ``[I_HR, 1 - I_HR, I_HR^2]`` with
    coarse physical guide ``G_LR``.
    """
    pan = _hr_to_pan_tensor(hr_image, device=device)
    h_hr, w_hr = int(pan.shape[0]), int(pan.shape[1])
    # Construct non-negative HR radiometric observation matrix Y_M: (C_hr, N_hr)
    hr_bands = torch.stack([pan, 1.0 - pan, pan**2, (1.0 - pan) ** 2], dim=0)
    y_hr = hr_bands.reshape(4, -1).clamp(min=1e-5)

    g_t = guide_to_tensor(
        coarse_guide, device=device, normalization=guide_normalization
    )
    m_ch = int(g_t.shape[1])
    height_lr, width_lr = int(g_t.shape[2]), int(g_t.shape[3])
    # Augment coarse guide with simplex matrix residual (1 - sum_m G_m)
    bg_lr = (1.0 - g_t[0].sum(dim=0, keepdim=True)).clamp(0.0, 1.0)
    y_lr_4d = torch.cat([bg_lr, g_t[0]], dim=0).unsqueeze(0)  # (1, M+1, H_lr, W_lr)
    c_lr = m_ch + 1
    y_lr = y_lr_4d.reshape(c_lr, -1).clamp(min=1e-5)  # (C_lr, N_lr)

    k_end = max(c_lr, int(n_endmembers))
    gen = torch.Generator(device="cpu").manual_seed(0)
    dict_lr = (torch.rand((c_lr, k_end), generator=gen) + 0.1).to(device)
    # Anchor first c_lr endmembers to identity-like pure phases
    dict_lr[:, :c_lr] = torch.eye(c_lr, device=device) + 0.05
    h_lr_ab = torch.full(
        (k_end, height_lr * width_lr), 1.0 / k_end, device=device, dtype=torch.float32
    )

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
        hr_bands.unsqueeze(0), output_size=(height_lr, width_lr)
    ).reshape(4, -1)
    dict_hr = (torch.rand((4, k_end), generator=gen) + 0.1).to(device)
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
    probs_fg = recon_hr[:, 1:].clamp(0.0, 1.0).cpu()
    return _probs_to_report(
        probs_fg, g_t.cpu(), mode=mode, method_name="cnmf", calibrate=calibrate
    )


def guided_variational_sr_baseline(
    hr_image: np.ndarray | torch.Tensor,
    coarse_guide: np.ndarray | torch.Tensor,
    mode: Literal["multiclass", "multilabel"] = "multiclass",
    n_iters: int = 45,
    n_steps: int | None = None,
    tv_weight: float = 0.04,
    lr: float = 0.12,
    device: str | torch.device = "cpu",
    guide_normalization: GuideNormalization = "minmax",
    calibrate: bool = True,
) -> UncertaintyReport:
    """Variational inverse solver minimizing forward-PSF data fidelity plus
    intensity-edge-weighted anisotropic Total Variation:
    ``||D_Sigma(P_HR) - G_LR||_2^2 + tv_weight * TV_{w(I_HR)}(P_HR)``.
    """
    steps = int(n_steps) if n_steps is not None else int(n_iters)
    pan = _hr_to_pan_tensor(hr_image, device=device)
    h_hr, w_hr = int(pan.shape[0]), int(pan.shape[1])

    g_t = guide_to_tensor(
        coarse_guide, device=device, normalization=guide_normalization
    )
    lr_hw = (int(g_t.shape[-2]), int(g_t.shape[-1]))
    init_up = F.interpolate(
        g_t, size=(h_hr, w_hr), mode="bilinear", align_corners=False
    ).clamp(0.02, 0.98)
    logits = nn.Parameter(torch.log(init_up / (1.0 - init_up)))

    wy = (
        torch.exp(-15.0 * (pan[1:, :] - pan[:-1, :]).abs()).view(1, 1, h_hr - 1, w_hr)
        if h_hr >= 2
        else None
    )
    wx = (
        torch.exp(-15.0 * (pan[:, 1:] - pan[:, :-1]).abs()).view(1, 1, h_hr, w_hr - 1)
        if w_hr >= 2
        else None
    )
    ds = ForwardPSFDownsampler().to(device)
    opt = torch.optim.Adam([logits], lr=lr)

    with torch.enable_grad():
        for _ in range(max(1, steps)):
            opt.zero_grad()
            probs = torch.sigmoid(logits)
            fid = F.mse_loss(ds(probs, lr_hw), g_t)
            tv = torch.zeros((), device=probs.device, dtype=probs.dtype)
            if wy is not None:
                dy = (probs[:, :, 1:, :] - probs[:, :, :-1, :]).abs()
                tv = tv + (wy * dy).mean()
            if wx is not None:
                dx = (probs[:, :, :, 1:] - probs[:, :, :, :-1]).abs()
                tv = tv + (wx * dx).mean()
            loss = fid + float(tv_weight) * tv
            loss.backward()
            opt.step()

    with torch.no_grad():
        probs_fg = torch.sigmoid(logits.detach()).cpu()
    return _probs_to_report(
        probs_fg,
        g_t.cpu(),
        mode=mode,
        method_name="guided_variational_sr",
        calibrate=calibrate,
    )


class _SineLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        omega_0: float = 30.0,
        is_first: bool = False,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        self.omega_0 = float(omega_0)
        self.linear = nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            bound = (
                (1.0 / in_dim) if is_first else (math.sqrt(6.0 / in_dim) / self.omega_0)
            )
            w = torch.empty_like(self.linear.weight, device="cpu").uniform_(
                -bound, bound, generator=generator
            )
            self.linear.weight.copy_(w)
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
    guide_normalization: GuideNormalization = "minmax",
    calibrate: bool = True,
) -> UncertaintyReport:
    """``SIREN`` Implicit Neural Representation baseline mapping continuous
    coordinates + structural intensity ``(y, x, I_HR(y, x))`` to HR phase logits
    supervised strictly through ``ForwardPSFDownsampler`` without mutating global
    PyTorch RNG state.
    """
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    pan = _hr_to_pan_tensor(hr_image, device=device)
    h_hr, w_hr = int(pan.shape[0]), int(pan.shape[1])

    ys = (
        torch.linspace(-1.0, 1.0, h_hr, device=device)
        if h_hr > 1
        else torch.zeros(1, device=device)
    )
    xs = (
        torch.linspace(-1.0, 1.0, w_hr, device=device)
        if w_hr > 1
        else torch.zeros(1, device=device)
    )
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords_in = torch.stack([yy, xx, 2.0 * pan - 1.0], dim=-1).reshape(-1, 3)

    g_t = guide_to_tensor(
        coarse_guide, device=device, normalization=guide_normalization
    )
    m_ch = int(g_t.shape[1])
    lr_hw = (int(g_t.shape[-2]), int(g_t.shape[-1]))

    with torch.random.fork_rng(devices=[]):
        last_lin = nn.Linear(hidden_dim, m_ch)
        with torch.no_grad():
            b_out = math.sqrt(6.0 / hidden_dim) / 30.0
            w_out = torch.empty_like(last_lin.weight, device="cpu").uniform_(
                -b_out, b_out, generator=gen
            )
            last_lin.weight.copy_(w_out)
            last_lin.bias.zero_()

        net = nn.Sequential(
            _SineLayer(3, hidden_dim, omega_0=30.0, is_first=True, generator=gen),
            _SineLayer(hidden_dim, hidden_dim, omega_0=30.0, generator=gen),
            last_lin,
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
    return _probs_to_report(
        probs_fg, g_t.cpu(), mode=mode, method_name="siren_inr", calibrate=calibrate
    )
