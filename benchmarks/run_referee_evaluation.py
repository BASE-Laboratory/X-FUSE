"""Unified deterministic multi-seed referee evaluation & figure generator for X-FUSE.

Addresses all six peer-review critiques in a single reproducible script:
1. Complete 7-method baseline parity (`Otsu`, `Bicubic`, `JBU`, `CNMF`, `Guided TV`,
   `SIREN INR`, `X-FUSE`) reported on both `mIoU` and `BF1` across Regime A
   (piecewise-constant intensity cartoon) and Regime B (oriented iso-intensity
   lamellar colonies, low-count Poisson spectral cubes, and stage drift).
2. Quantitative real-data synchrotron validation on Diamond DIAD `Zn13X`
   (multi-ROI 16x Moffat hold-out super-resolution + bidirectional zero-shot
   cross-slice transfer `Slice 4 <-> Slice 9` with `coarse_guide=None`).
3. Non-Gaussian heavy-tailed Moffat forward PSF (`beam_profile="moffat"`) across
   all synthetic and hold-out benchmarks to avoid Gaussian inverse crime.
4. Zero-label single-specimen uncertainty ranking (`AUSE`) separated cleanly from
   supervised cross-specimen `LAC` conformal calibration (`Specimen A -> Specimen B`)
   reporting empirical coverage, mean set size `|C(x)|`, and singleton fraction.
5. Deterministic multi-seed (`N = 3`, seeds `42, 43, 44`) `mean +/- s.d.` statistics
   and wall-clock CPU latency measurements from a single JSON source of truth.
6. Four 300-DPI publication figures rendered directly from the unified results.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import shift as nd_shift

import x_fuse.backbones as _xf_backbones
from x_fuse import (
    ConformalCalibrator,
    XFuse,
    bicubic_upsample_baseline,
    compute_segmentation_metrics,
    empirical_conformal_coverage,
    generate_correlative_phantom_2d,
    generate_oriented_lamellar_phantom_2d,
    generate_spectral_cube_phantom_2d,
    jbu_guided_baseline,
    make_experimental_holdout,
    sparsification_curve,
)

from .baselines import (
    cnmf_pansharpen_baseline,
    coordinate_inr_baseline,
    guided_variational_sr_baseline,
    otsu_threshold_baseline,
)
from .dataset_adapters import load_bundled_zn13x_pairs

METHOD_KEYS = (
    "otsu",
    "bicubic",
    "jbu",
    "cnmf",
    "guided_tv",
    "siren_inr",
    "xfuse",
)
METHOD_LABELS: dict[str, str] = {
    "otsu": "Multi-Otsu (HR only)",
    "bicubic": "Bicubic (LR only)",
    "jbu": "Joint Bilateral (JBU)",
    "cnmf": "CNMF Pan-Sharpen",
    "guided_tv": "Guided Variational TV",
    "siren_inr": "SIREN Coordinate INR",
    "xfuse": "X-FUSE (DINOv2 + Physics)",
}
METHOD_COLORS: dict[str, str] = {
    "otsu": "#7f7f7f",
    "bicubic": "#9467bd",
    "jbu": "#1f77b4",
    "cnmf": "#2ca02c",
    "guided_tv": "#ff7f0e",
    "siren_inr": "#8c564b",
    "xfuse": "#d62728",
}
METHOD_MARKERS: dict[str, str] = {
    "otsu": "v",
    "bicubic": "o",
    "jbu": "s",
    "cnmf": "D",
    "guided_tv": "^",
    "siren_inr": "P",
    "xfuse": "*",
}


def _install_ram_backbone_cache() -> None:
    """Cache frozen ViT weights in RAM so multi-seed instantiations avoid disk I/O
    without caching specimen features (preserving true per-image inference timing).
    """
    orig_loader = _xf_backbones.load_backbone
    ram_cache: dict[tuple[str, bool], nn.Module] = {}

    def _cached_loader(
        dino_name: str,
        stride: int,
        device: str | torch.device | None = None,
        pretrained: bool = True,
        **kwargs: Any,
    ) -> nn.Module:
        key = (dino_name, bool(pretrained))
        if key not in ram_cache:
            base = orig_loader(
                dino_name, stride=stride, device="cpu", pretrained=pretrained, **kwargs
            )
            ram_cache[key] = base
        cloned = copy.deepcopy(ram_cache[key])
        if device is not None:
            cloned = cloned.to(device)
        return cloned

    _xf_backbones.load_backbone = _cached_loader  # type: ignore[assignment]


def _set_all_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _mean_std(vals: list[float]) -> dict[str, Any]:
    arr = np.asarray(vals, dtype=np.float64)
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "mean": round(float(np.mean(arr)), 4),
        "std": round(sd, 4),
        "values": [round(float(v), 4) for v in vals],
    }


def _continuous_metrics_zn13x(
    pred_224: torch.Tensor | np.ndarray,
    gt_112: np.ndarray,
    support_thresh: float = 0.45,
) -> dict[str, float]:
    """Evaluate continuous abundance recovery on the native 112x112 measured
    synchrotron XRD-CT grid (2x2 average-pooled from 224x224 XCT grid).
    """
    p_t = (
        pred_224.detach().cpu().float()
        if isinstance(pred_224, torch.Tensor)
        else torch.from_numpy(np.asarray(pred_224, dtype=np.float32))
    )
    if p_t.ndim == 2:
        p_t = p_t.unsqueeze(0).unsqueeze(0)
    elif p_t.ndim == 3:
        p_t = p_t.unsqueeze(0)
    p_112 = F.avg_pool2d(p_t[:, :1], kernel_size=2, stride=2).reshape(-1).numpy()
    g_112 = gt_112.reshape(-1).astype(np.float32)

    p_std = float(np.std(p_112))
    g_std = float(np.std(g_112))
    pearson_r = (
        float(np.corrcoef(p_112, g_112)[0, 1])
        if (p_std > 1e-7 and g_std > 1e-7)
        else 0.0
    )
    mse = float(np.mean((p_112 - g_112) ** 2))
    psnr_db = float(-10.0 * np.log10(max(mse, 1e-10)))
    mae = float(np.mean(np.abs(p_112 - g_112)))

    pred_bin = p_112 > support_thresh
    gt_bin = g_112 > support_thresh
    union = int(np.sum(pred_bin | gt_bin))
    inter = int(np.sum(pred_bin & gt_bin))
    iou = float(inter / union) if union > 0 else 1.0
    return {
        "pearson_r": pearson_r,
        "psnr_db": psnr_db,
        "mae": mae,
        "support_iou": iou,
    }


def _nmf_unmix_spectral_cube(
    cube_lr: np.ndarray, n_components: int = 2, n_iters: int = 50, seed: int = 0
) -> np.ndarray:
    """Standard per-pixel multiplicative NMF unmixing of a noisy coarse spectral cube
    `(E, H_lr, W_lr)` into normalized `[0, 1]` abundance channels `(M, H_lr, W_lr)`.
    """
    e_bins, h_lr, w_lr = cube_lr.shape
    y_mat = np.maximum(cube_lr.reshape(e_bins, -1).astype(np.float32), 0.0)
    bg_floor = np.percentile(y_mat, 10, axis=1, keepdims=True)
    y_clean = np.maximum(y_mat - bg_floor, 1e-5)

    rng = np.random.default_rng(seed)
    w_dict = rng.uniform(0.1, 1.0, size=(e_bins, n_components)).astype(np.float32)
    h_abund = rng.uniform(0.1, 1.0, size=(n_components, h_lr * w_lr)).astype(np.float32)
    eps = 1e-7
    for _ in range(n_iters):
        h_abund *= (w_dict.T @ y_clean) / np.maximum((w_dict.T @ w_dict) @ h_abund, eps)
        w_dict *= (y_clean @ h_abund.T) / np.maximum(
            w_dict @ (h_abund @ h_abund.T), eps
        )
        w_sum = np.maximum(w_dict.sum(axis=0, keepdims=True), eps)
        w_dict /= w_sum
        h_abund *= w_sum.T

    total_scale = np.maximum(np.percentile(h_abund.sum(axis=0), 98), eps)
    abund_norm = np.clip(h_abund / total_scale, 0.0, 1.0)
    return abund_norm.reshape(n_components, h_lr, w_lr).astype(np.float32)


def _perm_invariant_fg_metrics(
    pred_mask: torch.Tensor | np.ndarray, gt_labels: np.ndarray
) -> dict[str, float]:
    """Symmetric 2-component foreground permutation check `(1, 2)` vs `(2, 1)`
    applied identically to all blind spectral unmixing methods.
    """
    p_np = (
        pred_mask.detach().cpu().numpy()
        if isinstance(pred_mask, torch.Tensor)
        else np.asarray(pred_mask)
    )
    m_orig = compute_segmentation_metrics(p_np, gt_labels)
    p_swap = p_np.copy()
    p_swap[p_np == 1] = 2
    p_swap[p_np == 2] = 1
    m_swap = compute_segmentation_metrics(p_swap, gt_labels)
    return m_orig if m_orig["iou"] >= m_swap["iou"] else m_swap


def _inject_shifted_moffat_guide(
    gt_phase_channels: np.ndarray,
    factor: int,
    shift_x_hr_px: float,
    psf_sigma_hr: float = 2.2,
) -> np.ndarray:
    shifted = np.stack(
        [
            nd_shift(ch, shift=(0.0, shift_x_hr_px), order=1, mode="nearest")
            for ch in gt_phase_channels
        ],
        axis=0,
    )
    dummy_hr = np.zeros(gt_phase_channels.shape[1:], dtype=np.float32)
    _, coarse, _ = make_experimental_holdout(
        dummy_hr,
        shifted,
        factor=factor,
        psf_sigma=psf_sigma_hr,
        beam_profile="moffat",
    )
    return coarse


def _run_all_7_multiclass(
    hr_image: np.ndarray,
    coarse_guide: np.ndarray,
    backbone: str,
    pretrained: bool,
    stride: int,
    epochs: int,
    tv_weight: float,
    seed: int,
    device: str = "cpu",
) -> tuple[dict[str, Any], dict[str, float]]:
    """Run all 7 methods under identical `preserve_counts` normalization."""
    norm = "preserve_counts"
    hw = (int(hr_image.shape[0]), int(hr_image.shape[1]))
    reports: dict[str, Any] = {}
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    reports["otsu"] = otsu_threshold_baseline(
        hr_image, coarse_guide, mode="multiclass", guide_normalization=norm
    )
    timings["otsu"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    reports["bicubic"] = bicubic_upsample_baseline(
        coarse_guide, hw, mode="multiclass", guide_normalization=norm
    )
    timings["bicubic"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    reports["jbu"] = jbu_guided_baseline(
        hr_image, coarse_guide, mode="multiclass", guide_normalization=norm
    )
    timings["jbu"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    reports["cnmf"] = cnmf_pansharpen_baseline(
        hr_image,
        coarse_guide,
        mode="multiclass",
        n_iters=35,
        device=device,
        guide_normalization=norm,
    )
    timings["cnmf"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    reports["guided_tv"] = guided_variational_sr_baseline(
        hr_image,
        coarse_guide,
        mode="multiclass",
        n_steps=60,
        device=device,
        guide_normalization=norm,
    )
    timings["guided_tv"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    reports["siren_inr"] = coordinate_inr_baseline(
        hr_image,
        coarse_guide,
        mode="multiclass",
        hidden_dim=48,
        epochs=65,
        seed=seed,
        device=device,
        guide_normalization=norm,
    )
    timings["siren_inr"] = time.perf_counter() - t0

    _set_all_seeds(seed)
    fuser = XFuse(
        backbone=backbone,
        stride=stride,
        dtype=torch.float32,
        loss_fn="compositional_kl",
        interpolation_mode="cross_attn",
        fusion_method="spatial_mlp",
        layers="last",
        learn_psf=True,
        guide_residual=False,
        tv_weight=tv_weight,
        pretrained=pretrained,
        lr=1e-2,
        device=device,
    )
    t0 = time.perf_counter()
    reports["xfuse"] = fuser.fit_predict(
        hr_image,
        coarse_guide,
        epochs=epochs,
        mode="multiclass",
        tta_shifts=(),
        tta_flips=True,
        guide_normalization=norm,
    )
    timings["xfuse"] = time.perf_counter() - t0
    return reports, timings


def _poly_ols_cross_slice(
    src_hr_224: np.ndarray, src_gt_112: np.ndarray, tgt_hr_224: np.ndarray
) -> np.ndarray:
    """Cubic scalar-intensity OLS baseline fit on source `(XCT, XRD-CT)` and applied
    zero-shot to target `XCT`.
    """
    src_pool = (
        F.avg_pool2d(
            torch.from_numpy(src_hr_224).float().unsqueeze(0).unsqueeze(0), 2, 2
        )
        .squeeze()
        .numpy()
    )
    x_s = src_pool.reshape(-1)
    design_s = np.stack([np.ones_like(x_s), x_s, x_s**2, x_s**3], axis=1)
    coef, _, _, _ = np.linalg.lstsq(design_s, src_gt_112.reshape(-1), rcond=None)

    x_t = tgt_hr_224.reshape(-1)
    design_t = np.stack([np.ones_like(x_t), x_t, x_t**2, x_t**3], axis=1)
    pred_t = (design_t @ coef).reshape(tgt_hr_224.shape)
    return np.clip(pred_t, 0.0, 1.0).astype(np.float32)


def _patch_ridge_cross_slice(
    src_hr_224: np.ndarray,
    src_gt_112: np.ndarray,
    tgt_hr_224: np.ndarray,
    patch_k: int = 5,
    l2_reg: float = 1.0,
) -> np.ndarray:
    """Local `5x5` structural patch + quadratic intensity ridge-regression baseline
    for cross-slice transfer without ViT features.
    """
    pad = patch_k // 2

    def _extract_patch_features(img_2d: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(img_2d).float().unsqueeze(0).unsqueeze(0)
        padded = F.pad(t, (pad, pad, pad, pad), mode="reflect")
        unf = F.unfold(padded, kernel_size=patch_k)  # (1, k*k, H*W)
        patches = unf[0].T.numpy()
        p_mean = patches.mean(axis=1, keepdims=True)
        p_std = patches.std(axis=1, keepdims=True)
        return np.concatenate(
            [np.ones_like(p_mean), patches, patches**2, p_mean, p_std], axis=1
        )

    src_pool = (
        F.avg_pool2d(
            torch.from_numpy(src_hr_224).float().unsqueeze(0).unsqueeze(0), 2, 2
        )
        .squeeze()
        .numpy()
    )
    x_src = _extract_patch_features(src_pool)
    y_src = src_gt_112.reshape(-1)
    gram = x_src.T @ x_src + l2_reg * np.eye(x_src.shape[1], dtype=np.float32)
    gram[0, 0] -= l2_reg
    w_ridge = np.linalg.solve(gram, x_src.T @ y_src)

    x_tgt = _extract_patch_features(tgt_hr_224)
    pred_tgt = (x_tgt @ w_ridge).reshape(tgt_hr_224.shape)
    return np.clip(pred_tgt, 0.0, 1.0).astype(np.float32)


def _render_figures(
    results: dict[str, Any],
    vis_cache: dict[str, Any],
    fig_dir: Path,
    artifact_dir: Path | None = None,
) -> list[Path]:
    """Render all 4 publication-grade 300-DPI figures directly from `results`."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    saved: list[Path] = []

    # ── Figure 1: Regime A (Intensity Cartoon) vs. Regime B (Oriented Lamellar) ─
    fig1, axes1 = plt.subplots(2, 2, figsize=(12.2, 8.6), constrained_layout=True)
    reg_a = results["exp3_regime_a_cartoon_sweep"]
    gaps_a = [row["downsample_factor"] for row in reg_a]

    for m_key in METHOD_KEYS:
        miou_m = [row["methods"][m_key]["miou"]["mean"] for row in reg_a]
        miou_s = [row["methods"][m_key]["miou"]["std"] for row in reg_a]
        bf1_m = [row["methods"][m_key]["bf1"]["mean"] for row in reg_a]
        bf1_s = [row["methods"][m_key]["bf1"]["std"] for row in reg_a]
        lw = 2.4 if m_key == "xfuse" else 1.5
        ms = 8.0 if m_key == "xfuse" else 5.5
        axes1[0, 0].errorbar(
            gaps_a,
            miou_m,
            yerr=miou_s,
            label=METHOD_LABELS[m_key],
            color=METHOD_COLORS[m_key],
            marker=METHOD_MARKERS[m_key],
            linewidth=lw,
            markersize=ms,
            capsize=3,
        )
        axes1[0, 1].errorbar(
            gaps_a,
            bf1_m,
            yerr=bf1_s,
            label=METHOD_LABELS[m_key],
            color=METHOD_COLORS[m_key],
            marker=METHOD_MARKERS[m_key],
            linewidth=lw,
            markersize=ms,
            capsize=3,
        )

    axes1[0, 0].set_title("A. Regime A (Scalar Intensity Cartoon, Moffat PSF) — mIoU")
    axes1[0, 0].set_xlabel("Resolution Gap (HR / Coarse Pixel Ratio)")
    axes1[0, 0].set_ylabel("Foreground mIoU (mean ± s.d., N=3)")
    axes1[0, 0].set_xticks(gaps_a)
    axes1[0, 0].set_xticklabels([f"{g}×" for g in gaps_a])
    axes1[0, 0].set_ylim(0.55, 1.03)
    axes1[0, 0].grid(alpha=0.25, linestyle="--")
    axes1[0, 0].legend(loc="lower left", ncol=2, frameon=True)

    axes1[0, 1].set_title("B. Regime A (Scalar Intensity Cartoon, Moffat PSF) — BF1")
    axes1[0, 1].set_xlabel("Resolution Gap (HR / Coarse Pixel Ratio)")
    axes1[0, 1].set_ylabel("Boundary F1 Score (mean ± s.d., N=3)")
    axes1[0, 1].set_xticks(gaps_a)
    axes1[0, 1].set_xticklabels([f"{g}×" for g in gaps_a])
    axes1[0, 1].set_ylim(0.55, 1.03)
    axes1[0, 1].grid(alpha=0.25, linestyle="--")
    axes1[0, 1].legend(loc="lower left", ncol=2, frameon=True)

    reg_b = results["exp4_regime_b_lamellar_sweep"]
    x_pos = np.arange(len(METHOD_KEYS))
    width = 0.36
    for idx_f, row_b in enumerate(reg_b):
        fac = row_b["downsample_factor"]
        offset = -width / 2 if idx_f == 0 else width / 2
        miou_vals = [row_b["methods"][k]["miou"]["mean"] for k in METHOD_KEYS]
        miou_errs = [row_b["methods"][k]["miou"]["std"] for k in METHOD_KEYS]
        bf1_vals = [row_b["methods"][k]["bf1"]["mean"] for k in METHOD_KEYS]
        bf1_errs = [row_b["methods"][k]["bf1"]["std"] for k in METHOD_KEYS]
        bar_colors = [METHOD_COLORS[k] for k in METHOD_KEYS]
        alpha_b = 0.92 if idx_f == 0 else 0.60
        hatch = "" if idx_f == 0 else "//"

        axes1[1, 0].bar(
            x_pos + offset,
            miou_vals,
            width=width,
            yerr=miou_errs,
            capsize=3,
            color=bar_colors,
            alpha=alpha_b,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.6,
            label=f"{fac}× Moffat PSF",
        )
        axes1[1, 1].bar(
            x_pos + offset,
            bf1_vals,
            width=width,
            yerr=bf1_errs,
            capsize=3,
            color=bar_colors,
            alpha=alpha_b,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.6,
            label=f"{fac}× Moffat PSF",
        )

    short_names = ["Otsu", "Bicubic", "JBU", "CNMF", "Guided TV", "SIREN", "X-FUSE"]
    for col_i, metric_lbl, panel_tag in [
        (
            0,
            "Foreground mIoU (mean ± s.d., N=3)",
            "C. Regime B (Oriented Iso-Intensity Lamellae) — mIoU",
        ),
        (
            1,
            "Boundary F1 Score (mean ± s.d., N=3)",
            "D. Regime B (Oriented Iso-Intensity Lamellae) — BF1",
        ),
    ]:
        axes1[1, col_i].set_title(panel_tag)
        axes1[1, col_i].set_xticks(x_pos)
        axes1[1, col_i].set_xticklabels(short_names, rotation=20, ha="right")
        axes1[1, col_i].set_ylabel(metric_lbl)
        axes1[1, col_i].set_ylim(0.0, 1.12)
        axes1[1, col_i].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axes1[1, col_i].grid(axis="y", alpha=0.25, linestyle="--")
        axes1[1, col_i].legend(loc="upper left", frameon=True)

    p1 = fig_dir / "referee_fig1_regimes_and_baselines.png"
    fig1.savefig(p1, dpi=300)
    fig1.savefig(p1.with_suffix(".pdf"))
    plt.close(fig1)
    saved.append(p1)

    # ── Figure 2: Real Synchrotron Diamond DIAD Zn13X Validation ─────────────
    zn_vis = vis_cache["zn13x"]
    fig2, axes2 = plt.subplots(2, 4, figsize=(15.4, 7.8), constrained_layout=True)

    im0 = axes2[0, 0].imshow(zn_vis["s4_hr"], cmap="gray", vmin=0.0, vmax=1.0)
    axes2[0, 0].set_title("A. Slice 4 XCT\n(224×224 HR)")
    axes2[0, 0].axis("off")
    fig2.colorbar(im0, ax=axes2[0, 0], fraction=0.046, pad=0.03)

    im1 = axes2[0, 1].imshow(zn_vis["s4_holdout_lr"], cmap="magma", vmin=0.0, vmax=1.0)
    axes2[0, 1].set_title("B. 16× Moffat Input\n(14×14 Coarse Guide)")
    axes2[0, 1].axis("off")
    fig2.colorbar(im1, ax=axes2[0, 1], fraction=0.046, pad=0.03)

    im2 = axes2[0, 2].imshow(zn_vis["s4_gt_112"], cmap="magma", vmin=0.0, vmax=1.0)
    axes2[0, 2].set_title("C. Measured XRD-CT\n(112×112 Ground Truth)")
    axes2[0, 2].axis("off")
    fig2.colorbar(im2, ax=axes2[0, 2], fraction=0.046, pad=0.03)

    im3 = axes2[0, 3].imshow(zn_vis["s4_xfuse"], cmap="magma", vmin=0.0, vmax=1.0)
    axes2[0, 3].set_title(
        f"D. X-FUSE 16× SR\n(IoU={zn_vis['s4_xfuse_iou']:.3f}, "
        f"{zn_vis['s4_xfuse_psnr']:.1f} dB)"
    )
    axes2[0, 3].axis("off")
    fig2.colorbar(im3, ax=axes2[0, 3], fraction=0.046, pad=0.03)

    im4 = axes2[1, 0].imshow(zn_vis["s9_gt_112"], cmap="magma", vmin=0.0, vmax=1.0)
    axes2[1, 0].set_title("E. Unseen Slice 9 XRD-CT\n(112×112 Ground Truth)")
    axes2[1, 0].axis("off")
    fig2.colorbar(im4, ax=axes2[1, 0], fraction=0.046, pad=0.03)

    im5 = axes2[1, 1].imshow(zn_vis["s9_poly"], cmap="magma", vmin=0.0, vmax=1.0)
    axes2[1, 1].set_title(
        f"F. Slice 4→9 Poly-OLS\n(r={zn_vis['s9_poly_r']:.3f}, "
        f"IoU={zn_vis['s9_poly_iou']:.3f})"
    )
    axes2[1, 1].axis("off")
    fig2.colorbar(im5, ax=axes2[1, 1], fraction=0.046, pad=0.03)

    im6 = axes2[1, 2].imshow(zn_vis["s9_xfuse_zs"], cmap="magma", vmin=0.0, vmax=1.0)
    axes2[1, 2].set_title(
        f"G. Slice 4→9 X-FUSE Zero-Shot\n(r={zn_vis['s9_xfuse_r']:.3f}, "
        f"IoU={zn_vis['s9_xfuse_iou']:.3f})"
    )
    axes2[1, 2].axis("off")
    fig2.colorbar(im6, ax=axes2[1, 2], fraction=0.046, pad=0.03)

    exp1 = results["exp1_zn13x_guided_16x_holdout"]["methods"]
    exp2 = results["exp2_zn13x_cross_slice_zeroshot"]["methods"]
    bar_ax = axes2[1, 3]
    rows_z = [
        ("Otsu (16×)", exp1["otsu"], METHOD_COLORS["otsu"], ""),
        ("Bicubic (16×)", exp1["bicubic"], METHOD_COLORS["bicubic"], ""),
        ("JBU (16×)", exp1["jbu"], METHOD_COLORS["jbu"], ""),
        ("CNMF (16×)", exp1["cnmf"], METHOD_COLORS["cnmf"], ""),
        ("Guided TV (16×)", exp1["guided_tv"], METHOD_COLORS["guided_tv"], ""),
        ("SIREN (16×)", exp1["siren_inr"], METHOD_COLORS["siren_inr"], ""),
        ("X-FUSE (16×)", exp1["xfuse"], METHOD_COLORS["xfuse"], ""),
        ("Mean (0-Shot)", exp2["mean_prior"], "#bdbdbd", "//"),
        ("Poly-OLS (0-Shot)", exp2["poly_ols"], "#7f7f7f", "//"),
        ("PatchRidge (0-Shot)", exp2["patch_ridge"], "#17becf", "//"),
        ("X-FUSE (0-Shot)", exp2["xfuse_zeroshot"], METHOD_COLORS["xfuse"], "//"),
    ]
    pos_z = np.arange(len(rows_z))
    for idx_z, (_, m_z, col_z, hatch_z) in enumerate(rows_z):
        bar_ax.barh(
            idx_z,
            m_z["support_iou"]["mean"],
            xerr=m_z["support_iou"]["std"],
            height=0.72,
            capsize=2.2,
            color=col_z,
            hatch=hatch_z,
            edgecolor="black",
            linewidth=0.55,
        )
    bar_ax.axhline(6.5, color="black", linestyle=":", linewidth=1.0)
    bar_ax.set_yticks(pos_z)
    bar_ax.set_yticklabels([r[0] for r in rows_z], fontsize=7.6)
    bar_ax.invert_yaxis()
    bar_ax.set_xlim(0.0, 1.02)
    bar_ax.set_xlabel("Crystalline Support IoU (N=4)")
    bar_ax.set_title("H. Real Zn13X Summary\n(Guided 16× vs 0-Shot)")
    bar_ax.grid(axis="x", alpha=0.25, linestyle="--")

    p2 = fig_dir / "referee_fig2_zn13x_real_validation.png"
    fig2.savefig(p2, dpi=300)
    fig2.savefig(p2.with_suffix(".pdf"))
    plt.close(fig2)
    saved.append(p2)

    # ── Figure 3: Visual Comparison on 16x Moffat Oriented Lamellar Colonies ──
    lam = vis_cache["lamellar_16x"]
    fig3, axes3 = plt.subplots(2, 4, figsize=(13.6, 6.8), constrained_layout=True)
    phase_cmap = matplotlib.colors.ListedColormap(["#1b1e23", "#2b8cbe", "#e34a33"])

    axes3[0, 0].imshow(lam["hr_image"], cmap="gray", vmin=0.0, vmax=1.0)
    axes3[0, 0].set_title("A. HR Micrograph (Matched μ, σ, |∇I|)")
    axes3[0, 0].axis("off")

    coarse_rgb = np.zeros((*lam["coarse_guide"].shape[1:], 3), dtype=np.float32)
    coarse_rgb[..., 2] = np.clip(lam["coarse_guide"][0], 0.0, 1.0)
    coarse_rgb[..., 0] = np.clip(lam["coarse_guide"][1], 0.0, 1.0)
    axes3[0, 1].imshow(coarse_rgb, interpolation="nearest")
    axes3[0, 1].set_title("B. 16× Moffat Guide (14×14)")
    axes3[0, 1].axis("off")

    axes3[0, 2].imshow(
        lam["gt_labels"], cmap=phase_cmap, vmin=0, vmax=2, interpolation="nearest"
    )
    axes3[0, 2].set_title("C. Ground-Truth Colony Labels")
    axes3[0, 2].axis("off")

    axes3[0, 3].imshow(
        lam["otsu_mask"], cmap=phase_cmap, vmin=0, vmax=2, interpolation="nearest"
    )
    axes3[0, 3].set_title(f"D. Multi-Otsu (mIoU={lam['otsu_iou']:.3f})")
    axes3[0, 3].axis("off")

    axes3[1, 0].imshow(
        lam["siren_mask"], cmap=phase_cmap, vmin=0, vmax=2, interpolation="nearest"
    )
    axes3[1, 0].set_title(f"E. SIREN INR (mIoU={lam['siren_iou']:.3f})")
    axes3[1, 0].axis("off")

    axes3[1, 1].imshow(
        lam["guided_tv_mask"], cmap=phase_cmap, vmin=0, vmax=2, interpolation="nearest"
    )
    axes3[1, 1].set_title(f"F. Guided TV (mIoU={lam['guided_tv_iou']:.3f})")
    axes3[1, 1].axis("off")

    axes3[1, 2].imshow(
        lam["xfuse_mask"], cmap=phase_cmap, vmin=0, vmax=2, interpolation="nearest"
    )
    axes3[1, 2].set_title(f"G. X-FUSE (mIoU={lam['xfuse_iou']:.3f})")
    axes3[1, 2].axis("off")

    im_u = axes3[1, 3].imshow(lam["xfuse_unc"], cmap="inferno", vmin=0.0, vmax=0.8)
    axes3[1, 3].set_title("H. X-FUSE Combined Uncertainty")
    axes3[1, 3].axis("off")
    fig3.colorbar(im_u, ax=axes3[1, 3], fraction=0.046, pad=0.03)

    p3 = fig_dir / "referee_fig3_lamellar_microstructure_visual.png"
    fig3.savefig(p3, dpi=300)
    fig3.savefig(p3.with_suffix(".pdf"))
    plt.close(fig3)
    saved.append(p3)

    # ── Figure 4: Poisson Spectral Cube, Stage Drift, AUSE, & Cross-Specimen LAC ─
    fig4, axes4 = plt.subplots(2, 2, figsize=(12.4, 8.8), constrained_layout=True)

    # Panel A: Poisson Spectral Cube Sweep
    pois = results["exp5_poisson_and_drift"]["poisson_spectral_sweep"]
    doses = [row["mean_counts_per_coarse_px"] for row in pois]
    pois_specs = [
        ("nmf_bicubic", "NMF + Bicubic", METHOD_COLORS["bicubic"], "o"),
        ("nmf_jbu", "NMF + JBU", METHOD_COLORS["jbu"], "s"),
        ("nmf_guided_tv", "NMF + Guided TV", METHOD_COLORS["guided_tv"], "^"),
        ("nmf_siren_inr", "NMF + SIREN INR", METHOD_COLORS["siren_inr"], "P"),
        ("xfuse_poisson", "X-FUSE Direct Poisson", METHOD_COLORS["xfuse"], "*"),
    ]
    for p_key, p_lbl, p_col, p_mk in pois_specs:
        means = [row["methods"][p_key]["miou"]["mean"] for row in pois]
        stds = [row["methods"][p_key]["miou"]["std"] for row in pois]
        axes4[0, 0].errorbar(
            doses,
            means,
            yerr=stds,
            label=p_lbl,
            color=p_col,
            marker=p_mk,
            linewidth=2.3 if "xfuse" in p_key else 1.5,
            markersize=7.5 if "xfuse" in p_key else 5.5,
            capsize=3,
        )
    axes4[0, 0].set_title("A. 16-Bin Low-Count Poisson Spectral Unmixing (4× Moffat)")
    axes4[0, 0].set_xlabel("Mean Photon / Count Budget per Coarse Pixel")
    axes4[0, 0].set_ylabel("Foreground mIoU (mean ± s.d., N=3)")
    axes4[0, 0].set_xticks(doses)
    axes4[0, 0].set_ylim(0.50, 1.01)
    axes4[0, 0].grid(alpha=0.25, linestyle="--")
    axes4[0, 0].legend(loc="lower right", frameon=True)

    # Panel B: Correlative Stage Drift Sweep
    drift = results["exp5_poisson_and_drift"]["stage_drift_sweep"]
    shifts = [row["injected_shift_hr_px"] for row in drift]
    drift_specs = [
        ("bicubic", "Bicubic (Unaligned)", METHOD_COLORS["bicubic"], "o", "-"),
        ("jbu", "JBU (Unaligned)", METHOD_COLORS["jbu"], "s", "-"),
        ("cnmf", "CNMF (Unaligned)", METHOD_COLORS["cnmf"], "D", "-"),
        ("guided_tv", "Guided TV (Unaligned)", METHOD_COLORS["guided_tv"], "^", "-"),
        ("siren_inr", "SIREN INR (Unaligned)", METHOD_COLORS["siren_inr"], "P", "-"),
        ("xfuse_unaligned", "X-FUSE (align='none')", "#e377c2", "o", "--"),
        (
            "xfuse_aligned",
            "X-FUSE (Differentiable Align)",
            METHOD_COLORS["xfuse"],
            "*",
            "-",
        ),
    ]
    for d_key, d_lbl, d_col, d_mk, d_ls in drift_specs:
        means = [row["methods"][d_key]["miou"]["mean"] for row in drift]
        stds = [row["methods"][d_key]["miou"]["std"] for row in drift]
        axes4[0, 1].errorbar(
            shifts,
            means,
            yerr=stds,
            label=d_lbl,
            color=d_col,
            marker=d_mk,
            linestyle=d_ls,
            linewidth=2.3 if d_key == "xfuse_aligned" else 1.5,
            markersize=7.5 if d_key == "xfuse_aligned" else 5.5,
            capsize=3,
        )
    axes4[0, 1].set_title("B. Correlative Stage Drift Robustness (4× Moffat)")
    axes4[0, 1].set_xlabel("Injected Guide Drift Along +X (HR Pixels)")
    axes4[0, 1].set_ylabel("Foreground mIoU (mean ± s.d., N=3)")
    axes4[0, 1].set_xticks(shifts)
    axes4[0, 1].set_ylim(0.50, 1.01)
    axes4[0, 1].grid(alpha=0.25, linestyle="--")
    axes4[0, 1].legend(loc="lower left", fontsize=7.4, frameon=True)

    # Panel C: Zero-Label Single-Specimen AUSE (Lower = Better Error Ranking)
    ause_lam = results["exp6_uncertainty_and_conformal"]["zero_label_ause_lamellar_8x"]
    u_keys = ["bicubic", "jbu", "cnmf", "guided_tv", "siren_inr", "xfuse"]
    u_lbls = ["Bicubic", "JBU", "CNMF", "Guided TV", "SIREN INR", "X-FUSE"]
    u_means = [ause_lam[k]["mean"] for k in u_keys]
    u_stds = [ause_lam[k]["std"] for k in u_keys]
    u_cols = [METHOD_COLORS[k] for k in u_keys]
    axes4[1, 0].bar(
        np.arange(len(u_keys)),
        u_means,
        yerr=u_stds,
        capsize=3,
        color=u_cols,
        edgecolor="black",
        linewidth=0.6,
    )
    axes4[1, 0].set_xticks(np.arange(len(u_keys)))
    axes4[1, 0].set_xticklabels(u_lbls, rotation=20, ha="right")
    axes4[1, 0].set_ylabel("AUSE ↓ (Lower = Better Error Ranking, N=3)")
    axes4[1, 0].set_title("C. Zero-Label Uncertainty Ranking (8× Lamellar AUSE)")
    axes4[1, 0].grid(axis="y", alpha=0.25, linestyle="--")

    # Panel D: Cross-Specimen LAC Conformal Calibration (Specimen A -> Specimen B)
    conf = results["exp6_uncertainty_and_conformal"]["cross_specimen_lac_lamellar_8x"]
    nom = [row["nominal_coverage"] for row in conf]
    emp_m = [row["empirical_coverage"]["mean"] for row in conf]
    emp_s = [row["empirical_coverage"]["std"] for row in conf]
    sz_m = [row["mean_set_size"]["mean"] for row in conf]
    sz_s = [row["mean_set_size"]["std"] for row in conf]

    ax_cov = axes4[1, 1]
    ax_sz = ax_cov.twinx()
    ax_sz.spines["top"].set_visible(False)

    x_c = np.arange(len(conf))
    w_c = 0.34
    b1 = ax_cov.bar(
        x_c - w_c / 2,
        emp_m,
        width=w_c,
        yerr=emp_s,
        capsize=3,
        color="#2b8cbe",
        edgecolor="black",
        linewidth=0.6,
        label="Empirical Coverage P(Y ∈ C(X))",
    )
    (l_nom,) = ax_cov.plot(
        x_c - w_c / 2,
        nom,
        "k--o",
        linewidth=1.3,
        markersize=5,
        label="Nominal Target (1 − α)",
    )
    b2 = ax_sz.bar(
        x_c + w_c / 2,
        sz_m,
        width=w_c,
        yerr=sz_s,
        capsize=3,
        color="#fdbb84",
        edgecolor="black",
        linewidth=0.6,
        label="Mean Set Size E[|C(X)|] (K=3)",
    )
    ax_cov.set_xticks(x_c)
    ax_cov.set_xticklabels([f"1−α = {n:.2f}\n(α={1 - n:.2f})" for n in nom])
    ax_cov.set_ylim(0.65, 1.03)
    ax_sz.set_ylim(0.8, 2.6)
    ax_cov.set_ylabel("Empirical Coverage on Unseen Specimen B")
    ax_sz.set_ylabel("Mean Prediction Set Size |C(X)|")
    ax_cov.set_title("D. Cross-Specimen LAC Calibration (Specimen A → B, N=3)")
    ax_cov.grid(axis="y", alpha=0.25, linestyle="--")
    handles = [b1, l_nom, b2]
    labels = [
        "Empirical Coverage (left)",
        "Nominal Target 1 − α (left)",
        "Mean Set Size |C(X)| (right)",
    ]
    ax_cov.legend(handles, labels, loc="upper right", fontsize=7.5, frameon=True)

    p4 = fig_dir / "referee_fig4_poisson_drift_conformal.png"
    fig4.savefig(p4, dpi=300)
    fig4.savefig(p4.with_suffix(".pdf"))
    plt.close(fig4)
    saved.append(p4)

    if artifact_dir is not None and artifact_dir.exists():
        for p_img in saved:
            (artifact_dir / p_img.name).write_bytes(p_img.read_bytes())

    return saved


def _save_state(
    output_dir: Path, results: dict[str, Any], vis_cache: dict[str, Any]
) -> None:
    out_json = output_dir / "referee_evaluation_results.json"
    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, dict[str, Any]] = {}
    for group_k, group_d in vis_cache.items():
        scalars[group_k] = {}
        for k, v in group_d.items():
            if isinstance(v, np.ndarray):
                arrays[f"{group_k}__{k}"] = v
            else:
                scalars[group_k][k] = v
    results["_vis_scalars"] = scalars
    out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if arrays:
        np.savez_compressed(output_dir / "vis_cache.npz", **arrays)


def _load_state(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    out_json = output_dir / "referee_evaluation_results.json"
    if not out_json.exists():
        return {}, {}
    results: dict[str, Any] = json.loads(out_json.read_text(encoding="utf-8"))
    scalars = results.get("_vis_scalars", {})
    vis_cache: dict[str, Any] = {gk: dict(gd) for gk, gd in scalars.items()}
    npz_path = output_dir / "vis_cache.npz"
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as data:
            for full_k in data.files:
                gk, item_k = full_k.split("__", 1)
                vis_cache.setdefault(gk, {})[item_k] = data[full_k]
    return results, vis_cache


def run_referee_evaluation(
    output_dir: Path,
    artifact_dir: Path | None = None,
    backbone: str = "dinov2_vits14_reg",
    pretrained: bool = True,
    stride: int = 7,
    seeds: tuple[int, ...] = (42, 43, 44),
    device: str = "cpu",
    resume: bool = False,
    stages: set[str] | None = None,
) -> dict[str, Any]:
    """Execute the complete multi-seed referee benchmark & figure generation suite."""
    t_suite_start = time.perf_counter()
    _install_ram_backbone_cache()
    output_dir.mkdir(parents=True, exist_ok=True)
    norm = "preserve_counts"
    active = stages or {"1", "2", "3", "4", "5a", "5b", "6", "plot"}

    if active != {"plot"}:
        _ = _xf_backbones.load_backbone(
            backbone, stride=stride, device=device, pretrained=pretrained
        )

    results, vis_cache = _load_state(output_dir) if resume else ({}, {})
    results["protocol"] = {
        "backbone": backbone,
        "pretrained": pretrained,
        "stride": stride,
        "seeds": list(seeds),
        "n_seeds": len(seeds),
        "beam_profile": "moffat",
        "guide_normalization": norm,
        "device": device,
        "semi_synthetic_datasets_excluded_from_real_claims": ["morgis_001"],
    }

    zn_pairs = load_bundled_zn13x_pairs()
    assert len(zn_pairs) >= 2, "Expected bundled Zn13X Slice 4 & Slice 9 pairs."
    s4, s9 = zn_pairs[0], zn_pairs[1]
    crop_sz = 224

    # ── Experiment 1: Real Synchrotron Diamond DIAD Zn13X Guided 16x Hold-Out SR ─
    if ({"1", "1a", "1b"} & active) and not (
        resume and "exp1_zn13x_guided_16x_holdout" in results
    ):
        print(
            "[1/6] Running Real Diamond DIAD Zn13X 16x Moffat Hold-Out SR "
            "(N=4 ROIs)...",
            flush=True,
        )
        p05_s4, p99_s4 = np.percentile(s4.guide_map[0], [5, 99])
        zn_rois = [(192, 1024), (640, 640), (448, 512), (512, 768)]
        roi_subset = (
            {0, 1}
            if active == {"1a"}
            else ({2, 3} if active == {"1b"} else {0, 1, 2, 3})
        )
        exp1_part: dict[str, Any] = dict(results.get("_exp1_partial", {}))
        for roi_idx, (y0, x0) in enumerate(zn_rois):
            if roi_idx not in roi_subset or (resume and str(roi_idx) in exp1_part):
                continue
            hr_sub = s4.hr_image[y0 : y0 + crop_sz, x0 : x0 + crop_sz].astype(
                np.float32
            )
            gt_112 = np.clip(
                (
                    s4.guide_map[
                        0, y0 // 2 : (y0 + crop_sz) // 2, x0 // 2 : (x0 + crop_sz) // 2
                    ]
                    - p05_s4
                )
                / max(float(p99_s4 - p05_s4), 1e-6),
                0.0,
                1.0,
            ).astype(np.float32)
            _, g_lr_hold, _ = make_experimental_holdout(
                np.zeros_like(gt_112),
                gt_112[None],
                factor=8,
                beam_profile="moffat",
            )

            t0 = time.perf_counter()
            r_otsu = otsu_threshold_baseline(
                hr_sub,
                g_lr_hold,
                mode="multilabel",
                guide_normalization=norm,
                calibrate=False,
            )
            t_otsu = time.perf_counter() - t0

            t0 = time.perf_counter()
            r_bic = bicubic_upsample_baseline(
                g_lr_hold,
                (crop_sz, crop_sz),
                mode="multilabel",
                guide_normalization=norm,
                calibrate=False,
            )
            t_bic = time.perf_counter() - t0

            t0 = time.perf_counter()
            r_jbu = jbu_guided_baseline(
                hr_sub,
                g_lr_hold,
                mode="multilabel",
                guide_normalization=norm,
                calibrate=False,
            )
            t_jbu = time.perf_counter() - t0

            t0 = time.perf_counter()
            r_cnmf = cnmf_pansharpen_baseline(
                hr_sub,
                g_lr_hold,
                mode="multilabel",
                n_iters=35,
                device=device,
                guide_normalization=norm,
                calibrate=False,
            )
            t_cnmf = time.perf_counter() - t0

            t0 = time.perf_counter()
            r_gtv = guided_variational_sr_baseline(
                hr_sub,
                g_lr_hold,
                mode="multilabel",
                n_steps=60,
                device=device,
                guide_normalization=norm,
                calibrate=False,
            )
            t_gtv = time.perf_counter() - t0

            t0 = time.perf_counter()
            r_inr = coordinate_inr_baseline(
                hr_sub,
                g_lr_hold,
                mode="multilabel",
                hidden_dim=48,
                epochs=65,
                seed=seeds[0],
                device=device,
                guide_normalization=norm,
                calibrate=False,
            )
            t_inr = time.perf_counter() - t0

            _set_all_seeds(seeds[0])
            m_zn = XFuse(
                backbone=backbone,
                stride=stride,
                dtype=torch.float32,
                loss_fn="forward_psf_mse",
                interpolation_mode="cross_attn",
                fusion_method="spatial_mlp",
                guide_residual=True,
                tv_weight=0.01,
                learn_psf=True,
                pretrained=pretrained,
                lr=5e-3,
                device=device,
            )
            t0 = time.perf_counter()
            r_xf = m_zn.fit_predict(
                hr_sub,
                g_lr_hold,
                epochs=35,
                mode="multilabel",
                calibrate_to_guide=False,
                tta_shifts=(),
                tta_flips=True,
                guide_normalization=norm,
            )
            t_xf = time.perf_counter() - t0

            roi_rec: dict[str, dict[str, float]] = {}
            for m_k, rep_m, lat_m in [
                ("otsu", r_otsu, t_otsu),
                ("bicubic", r_bic, t_bic),
                ("jbu", r_jbu, t_jbu),
                ("cnmf", r_cnmf, t_cnmf),
                ("guided_tv", r_gtv, t_gtv),
                ("siren_inr", r_inr, t_inr),
                ("xfuse", r_xf, t_xf),
            ]:
                c_met = _continuous_metrics_zn13x(rep_m.probabilities, gt_112)
                roi_rec[m_k] = {**c_met, "latency_s": lat_m}

            if roi_idx == 0:
                xf_c = roi_rec["xfuse"]
                vis_cache.setdefault("zn13x", {}).update(
                    {
                        "s4_hr": hr_sub,
                        "s4_holdout_lr": g_lr_hold[0],
                        "s4_gt_112": gt_112,
                        "s4_xfuse": r_xf.probabilities[0, 0].detach().cpu().numpy(),
                        "s4_xfuse_iou": xf_c["support_iou"],
                        "s4_xfuse_psnr": xf_c["psnr_db"],
                    }
                )
            exp1_part[str(roi_idx)] = roi_rec
            results["_exp1_partial"] = exp1_part
            _save_state(output_dir, results, vis_cache)
            print(
                f"  [1/6] ROI {roi_idx + 1}/{len(zn_rois)} done (xfuse={t_xf:.1f}s)",
                flush=True,
            )

        if len(exp1_part) == len(zn_rois):
            zn_guided_acc: dict[str, dict[str, list[float]]] = {
                k: {
                    "pearson_r": [],
                    "psnr_db": [],
                    "mae": [],
                    "support_iou": [],
                    "latency_s": [],
                }
                for k in METHOD_KEYS
            }
            for r_i in range(len(zn_rois)):
                for m_k in METHOD_KEYS:
                    for met_name, met_val in exp1_part[str(r_i)][m_k].items():
                        zn_guided_acc[m_k][met_name].append(float(met_val))
            results["exp1_zn13x_guided_16x_holdout"] = {
                "rois_yx": [list(r) for r in zn_rois],
                "crop_size_hr": crop_sz,
                "measured_xrdct_grid": 112,
                "holdout_moffat_grid": 14,
                "effective_resolution_gap_vs_xct": 16,
                "methods": {
                    k: {m_n: _mean_std(vals) for m_n, vals in m_dict.items()}
                    for k, m_dict in zn_guided_acc.items()
                },
            }
            results.pop("_exp1_partial", None)
            _save_state(output_dir, results, vis_cache)

    # ── Experiment 2: Real Zn13X Cross-Slice Zero-Shot Transfer (coarse_guide=None) ─
    if ({"2", "2a", "2b"} & active) and not (
        resume and "exp2_zn13x_cross_slice_zeroshot" in results
    ):
        print(
            "[2/6] Running Real Diamond DIAD Zn13X Zero-Shot Cross-Slice "
            "Transfer (N=4 pairs)...",
            flush=True,
        )
        zs_rois = [(192, 1024), (640, 640)]
        zs_directions = [("slice4_to_slice9", s4, s9), ("slice9_to_slice4", s9, s4)]
        zs_methods = ("mean_prior", "poly_ols", "patch_ridge", "xfuse_zeroshot")
        dir_subset = {0} if active == {"2a"} else ({1} if active == {"2b"} else {0, 1})
        exp2_part: dict[str, Any] = dict(results.get("_exp2_partial", {}))

        for dir_idx, (dir_name, src_s, tgt_s) in enumerate(zs_directions):
            if dir_idx not in dir_subset:
                continue
            p05_src, p99_src = np.percentile(src_s.guide_map[0], [5, 99])
            span_src = max(float(p99_src - p05_src), 1e-6)
            for roi_idx, (y0, x0) in enumerate(zs_rois):
                pkey = f"{dir_idx}_{roi_idx}"
                if resume and pkey in exp2_part:
                    continue
                src_hr = src_s.hr_image[y0 : y0 + crop_sz, x0 : x0 + crop_sz].astype(
                    np.float32
                )
                tgt_hr = tgt_s.hr_image[y0 : y0 + crop_sz, x0 : x0 + crop_sz].astype(
                    np.float32
                )
                src_gt = np.clip(
                    (
                        src_s.guide_map[
                            0,
                            y0 // 2 : (y0 + crop_sz) // 2,
                            x0 // 2 : (x0 + crop_sz) // 2,
                        ]
                        - p05_src
                    )
                    / span_src,
                    0.0,
                    1.0,
                ).astype(np.float32)
                tgt_gt = np.clip(
                    (
                        tgt_s.guide_map[
                            0,
                            y0 // 2 : (y0 + crop_sz) // 2,
                            x0 // 2 : (x0 + crop_sz) // 2,
                        ]
                        - p05_src
                    )
                    / span_src,
                    0.0,
                    1.0,
                ).astype(np.float32)

                t0 = time.perf_counter()
                pred_mean = np.full_like(tgt_hr, float(src_gt.mean()))
                lat_mean = time.perf_counter() - t0

                t0 = time.perf_counter()
                pred_poly = _poly_ols_cross_slice(src_hr, src_gt, tgt_hr)
                lat_poly = time.perf_counter() - t0

                t0 = time.perf_counter()
                pred_ridge = _patch_ridge_cross_slice(src_hr, src_gt, tgt_hr)
                lat_ridge = time.perf_counter() - t0

                _set_all_seeds(seeds[0])
                m_zs = XFuse(
                    backbone=backbone,
                    stride=stride,
                    dtype=torch.float32,
                    loss_fn="forward_psf_mse",
                    interpolation_mode="cross_attn",
                    fusion_method="spatial_mlp",
                    guide_residual=False,
                    tv_weight=0.02,
                    learn_psf=True,
                    pretrained=pretrained,
                    lr=1e-2,
                    device=device,
                )
                m_zs.fit(
                    src_hr,
                    src_gt[None],
                    epochs=35,
                    tta_shifts=(),
                    tta_flips=True,
                    guide_normalization=norm,
                )
                t0 = time.perf_counter()
                rep_zs = m_zs.predict(
                    tgt_hr,
                    coarse_guide=None,
                    mode="multilabel",
                    calibrate_to_guide=False,
                )
                lat_xf_zs = time.perf_counter() - t0
                pred_xf_zs = rep_zs.probabilities[0, 0].detach().cpu().numpy()

                pair_rec: dict[str, dict[str, float]] = {}
                for zk, p_arr, l_val in [
                    ("mean_prior", pred_mean, lat_mean),
                    ("poly_ols", pred_poly, lat_poly),
                    ("patch_ridge", pred_ridge, lat_ridge),
                    ("xfuse_zeroshot", pred_xf_zs, lat_xf_zs),
                ]:
                    cm = _continuous_metrics_zn13x(p_arr, tgt_gt)
                    pair_rec[zk] = {**cm, "predict_latency_s": l_val}

                if dir_idx == 0 and roi_idx == 0:
                    cm_poly = pair_rec["poly_ols"]
                    cm_xf = pair_rec["xfuse_zeroshot"]
                    vis_cache.setdefault("zn13x", {}).update(
                        {
                            "s9_gt_112": tgt_gt,
                            "s9_poly": pred_poly,
                            "s9_poly_r": cm_poly["pearson_r"],
                            "s9_poly_iou": cm_poly["support_iou"],
                            "s9_xfuse_zs": pred_xf_zs,
                            "s9_xfuse_r": cm_xf["pearson_r"],
                            "s9_xfuse_iou": cm_xf["support_iou"],
                            "direction": dir_name,
                        }
                    )
                exp2_part[pkey] = pair_rec
                results["_exp2_partial"] = exp2_part
                _save_state(output_dir, results, vis_cache)
                print(f"  [2/6] pair {pkey} done", flush=True)

        if len(exp2_part) == 4:
            zs_acc: dict[str, dict[str, list[float]]] = {
                k: {
                    "pearson_r": [],
                    "psnr_db": [],
                    "mae": [],
                    "support_iou": [],
                    "predict_latency_s": [],
                }
                for k in zs_methods
            }
            for d_i in range(2):
                for r_i in range(2):
                    for zk in zs_methods:
                        for mn, mv in exp2_part[f"{d_i}_{r_i}"][zk].items():
                            zs_acc[zk][mn].append(float(mv))
            results["exp2_zn13x_cross_slice_zeroshot"] = {
                "transfer_pairs": [
                    f"{d}@{r[0]}_{r[1]}" for d, _, _ in zs_directions for r in zs_rois
                ],
                "unguided_target_slice": True,
                "inapplicable_guided_only_baselines": [
                    "bicubic",
                    "jbu",
                    "cnmf",
                    "guided_tv",
                    "siren_inr",
                ],
                "methods": {
                    k: {mn: _mean_std(vals) for mn, vals in mdict.items()}
                    for k, mdict in zs_acc.items()
                },
            }
            results.pop("_exp2_partial", None)
            _save_state(output_dir, results, vis_cache)

    # ── Experiment 3: Regime A — Piecewise-Constant Intensity Cartoon (4x, 8x, 16x) ─
    if ({"3", "3a", "3b"} & active) and not (
        resume and "exp3_regime_a_cartoon_sweep" in results
    ):
        car_factors = (
            (4, 8) if active == {"3a"} else ((16,) if active == {"3b"} else (4, 8, 16))
        )
        print(
            f"[3/6] Running Regime A (Piecewise Intensity Cartoon, "
            f"factors={car_factors} Moffat, N=3 seeds)...",
            flush=True,
        )
        cartoon_size = 160
        exp3_part: dict[str, Any] = dict(results.get("_exp3_partial", {}))
        for factor in car_factors:
            if resume and str(factor) in exp3_part:
                continue
            per_m: dict[str, dict[str, list[float]]] = {
                k: {"miou": [], "bf1": [], "latency_s": []} for k in METHOD_KEYS
            }
            for s in seeds:
                ph = generate_correlative_phantom_2d(
                    size=cartoon_size,
                    downsample_factor=factor,
                    phase_b_contrast=0.35,
                    hr_noise_std=0.03,
                    poisson_counts=300.0,
                    beam_profile="moffat",
                    seed=s,
                )
                reps, times = _run_all_7_multiclass(
                    ph.hr_image,
                    ph.coarse_guide,
                    backbone=backbone,
                    pretrained=pretrained,
                    stride=stride,
                    epochs=35,
                    tv_weight=0.005,
                    seed=s,
                    device=device,
                )
                for k in METHOD_KEYS:
                    met = compute_segmentation_metrics(
                        reps[k].hard_mask[0], ph.gt_labels
                    )
                    per_m[k]["miou"].append(met["iou"])
                    per_m[k]["bf1"].append(met["boundary_f1"])
                    per_m[k]["latency_s"].append(times[k])

            exp3_part[str(factor)] = {
                "downsample_factor": factor,
                "image_size": cartoon_size,
                "methods": {
                    k: {mn: _mean_std(vals) for mn, vals in mdict.items()}
                    for k, mdict in per_m.items()
                },
            }
            results["_exp3_partial"] = exp3_part
            _save_state(output_dir, results, vis_cache)
            print(f"  [3/6] factor={factor}x done", flush=True)

        if all(str(f) in exp3_part for f in (4, 8, 16)):
            results["exp3_regime_a_cartoon_sweep"] = [
                exp3_part["4"],
                exp3_part["8"],
                exp3_part["16"],
            ]
            results.pop("_exp3_partial", None)
            _save_state(output_dir, results, vis_cache)

    # ── Experiment 4: Regime B1 — Oriented Iso-Intensity Lamellar Colonies (8x, 16x) ─
    if ({"4", "4a", "4b"} & active) and not (
        resume and "exp4_regime_b_lamellar_sweep" in results
    ):
        lam_factors = (
            (8,) if active == {"4a"} else ((16,) if active == {"4b"} else (8, 16))
        )
        print(
            f"[4/6] Running Regime B1 (Oriented Iso-Intensity Lamellae, "
            f"factors={lam_factors} Moffat, N=3 seeds)...",
            flush=True,
        )
        lam_size = 224
        partial_rows: dict[str, Any] = dict(results.get("_exp4_partial_rows", {}))
        ause_lam_8x: dict[str, list[float]] = {
            k: [] for k in METHOD_KEYS if k != "otsu"
        }
        for factor in lam_factors:
            if resume and str(factor) in partial_rows:
                continue
            per_m = {k: {"miou": [], "bf1": [], "latency_s": []} for k in METHOD_KEYS}
            colony_stats_list: list[dict[str, float]] = []
            for s_idx, s in enumerate(seeds):
                ph_lam = generate_oriented_lamellar_phantom_2d(
                    size=lam_size,
                    downsample_factor=factor,
                    beam_profile="moffat",
                    seed=s,
                )
                gy, gx = np.gradient(ph_lam.hr_image)
                grad_mag = np.hypot(gy, gx)
                m1 = ph_lam.gt_labels == 1
                m2 = ph_lam.gt_labels == 2
                colony_stats_list.append(
                    {
                        "colony_a_mean": float(ph_lam.hr_image[m1].mean()),
                        "colony_b_mean": float(ph_lam.hr_image[m2].mean()),
                        "colony_a_std": float(ph_lam.hr_image[m1].std()),
                        "colony_b_std": float(ph_lam.hr_image[m2].std()),
                        "colony_a_grad": float(grad_mag[m1].mean()),
                        "colony_b_grad": float(grad_mag[m2].mean()),
                    }
                )
                reps, times = _run_all_7_multiclass(
                    ph_lam.hr_image,
                    ph_lam.coarse_guide,
                    backbone=backbone,
                    pretrained=pretrained,
                    stride=stride,
                    epochs=40,
                    tv_weight=0.01,
                    seed=s,
                    device=device,
                )
                for k in METHOD_KEYS:
                    met = compute_segmentation_metrics(
                        reps[k].hard_mask[0], ph_lam.gt_labels
                    )
                    per_m[k]["miou"].append(met["iou"])
                    per_m[k]["bf1"].append(met["boundary_f1"])
                    per_m[k]["latency_s"].append(times[k])
                    if factor == 8 and k != "otsu":
                        sc = sparsification_curve(
                            reps[k].hard_mask[0],
                            ph_lam.gt_labels,
                            reps[k].combined_uncertainty[0],
                        )
                        ause_lam_8x[k].append(float(sc["ause"]))

                if factor == 16 and s_idx == 0:
                    vis_cache["lamellar_16x"] = {
                        "hr_image": ph_lam.hr_image,
                        "coarse_guide": ph_lam.coarse_guide,
                        "gt_labels": ph_lam.gt_labels,
                        "otsu_mask": reps["otsu"].hard_mask[0].cpu().numpy(),
                        "otsu_iou": per_m["otsu"]["miou"][0],
                        "siren_mask": reps["siren_inr"].hard_mask[0].cpu().numpy(),
                        "siren_iou": per_m["siren_inr"]["miou"][0],
                        "guided_tv_mask": reps["guided_tv"].hard_mask[0].cpu().numpy(),
                        "guided_tv_iou": per_m["guided_tv"]["miou"][0],
                        "xfuse_mask": reps["xfuse"].hard_mask[0].cpu().numpy(),
                        "xfuse_iou": per_m["xfuse"]["miou"][0],
                        "xfuse_unc": reps["xfuse"]
                        .combined_uncertainty[0]
                        .cpu()
                        .numpy(),
                    }
                print(
                    f"  [4/6] factor={factor}x seed={s} done "
                    f"(xfuse={times['xfuse']:.1f}s)",
                    flush=True,
                )

            partial_rows[str(factor)] = {
                "downsample_factor": factor,
                "image_size": lam_size,
                "colony_intensity_parity": {
                    stat_k: round(
                        float(np.mean([cs[stat_k] for cs in colony_stats_list])), 4
                    )
                    for stat_k in colony_stats_list[0]
                },
                "methods": {
                    k: {mn: _mean_std(vals) for mn, vals in mdict.items()}
                    for k, mdict in per_m.items()
                },
            }
            results["_exp4_partial_rows"] = partial_rows
            if factor == 8:
                results["_ause_lam_8x"] = {
                    k: _mean_std(vals) for k, vals in ause_lam_8x.items()
                }
            _save_state(output_dir, results, vis_cache)

        if "8" in partial_rows and "16" in partial_rows:
            results["exp4_regime_b_lamellar_sweep"] = [
                partial_rows["8"],
                partial_rows["16"],
            ]
            results.pop("_exp4_partial_rows", None)
            _save_state(output_dir, results, vis_cache)

    # ── Experiment 5a: Low-Count 16-Bin Poisson Spectral Cube ─────────────────
    spec_size = 128
    if "5a" in active and not (
        resume
        and "exp5_poisson_and_drift" in results
        and "poisson_spectral_sweep" in results["exp5_poisson_and_drift"]
    ):
        print(
            "[5a/6] Running Low-Count 16-Bin Poisson Spectral Cube Sweep "
            "(N=3 seeds)...",
            flush=True,
        )
        poisson_rows: list[dict[str, Any]] = []
        pois_keys = (
            "nmf_bicubic",
            "nmf_jbu",
            "nmf_guided_tv",
            "nmf_siren_inr",
            "xfuse_poisson",
        )
        for dose in (20.0, 45.0, 100.0):
            p_acc: dict[str, dict[str, list[float]]] = {
                k: {"miou": [], "bf1": []} for k in pois_keys
            }
            for s in seeds:
                ph_sp = generate_spectral_cube_phantom_2d(
                    size=spec_size,
                    downsample_factor=4,
                    n_spectral_bins=16,
                    mean_counts=dose,
                    beam_profile="moffat",
                    seed=s,
                )
                nmf_lr = _nmf_unmix_spectral_cube(
                    ph_sp.spectral_cube_lr, n_components=2, n_iters=50, seed=s
                )
                r_nb = bicubic_upsample_baseline(
                    nmf_lr,
                    (spec_size, spec_size),
                    mode="multiclass",
                    guide_normalization=norm,
                )
                r_nj = jbu_guided_baseline(
                    ph_sp.hr_image, nmf_lr, mode="multiclass", guide_normalization=norm
                )
                r_ntv = guided_variational_sr_baseline(
                    ph_sp.hr_image,
                    nmf_lr,
                    mode="multiclass",
                    n_steps=60,
                    device=device,
                    guide_normalization=norm,
                )
                r_nsir = coordinate_inr_baseline(
                    ph_sp.hr_image,
                    nmf_lr,
                    mode="multiclass",
                    hidden_dim=48,
                    epochs=65,
                    seed=s,
                    device=device,
                    guide_normalization=norm,
                )
                _set_all_seeds(s)
                m_sp = XFuse(
                    backbone=backbone,
                    stride=stride,
                    dtype=torch.float32,
                    loss_fn="forward_psf_poisson",
                    interpolation_mode="cross_attn",
                    fusion_method="spatial_mlp",
                    spectral_channels=16,
                    guide_channels=2,
                    learn_psf=True,
                    guide_residual=True,
                    tv_weight=0.008,
                    pretrained=pretrained,
                    lr=1e-2,
                    device=device,
                )
                r_xf_sp = m_sp.fit_predict(
                    ph_sp.hr_image,
                    ph_sp.spectral_cube_lr,
                    epochs=38,
                    mode="multiclass",
                    tta_shifts=(),
                    tta_flips=True,
                    spectral_unmix=True,
                    n_components=2,
                )
                for pk, rep_p in [
                    ("nmf_bicubic", r_nb),
                    ("nmf_jbu", r_nj),
                    ("nmf_guided_tv", r_ntv),
                    ("nmf_siren_inr", r_nsir),
                    ("xfuse_poisson", r_xf_sp),
                ]:
                    pm = _perm_invariant_fg_metrics(rep_p.hard_mask[0], ph_sp.gt_labels)
                    p_acc[pk]["miou"].append(pm["iou"])
                    p_acc[pk]["bf1"].append(pm["boundary_f1"])

            poisson_rows.append(
                {
                    "mean_counts_per_coarse_px": dose,
                    "n_spectral_bins": 16,
                    "downsample_factor": 4,
                    "image_size": spec_size,
                    "methods": {
                        pk: {mn: _mean_std(vals) for mn, vals in pdict.items()}
                        for pk, pdict in p_acc.items()
                    },
                }
            )
        results.setdefault("exp5_poisson_and_drift", {})["poisson_spectral_sweep"] = (
            poisson_rows
        )
        _save_state(output_dir, results, vis_cache)

    # ── Experiment 5b: Correlative Stage Drift Sweep ──────────────────────────
    if ({"5b", "5b1", "5b2"} & active) and not (
        resume
        and "exp5_poisson_and_drift" in results
        and "stage_drift_sweep" in results["exp5_poisson_and_drift"]
    ):
        shifts_list = (0.0, 2.0, 4.0, 5.0)
        run_shifts = (
            (0.0, 2.0)
            if active == {"5b1"}
            else ((4.0, 5.0) if active == {"5b2"} else shifts_list)
        )
        print(
            f"[5b/6] Running Correlative Stage Drift Sweep (shifts={run_shifts} HR px, "
            "N=3 seeds)...",
            flush=True,
        )
        drift_keys = (
            "bicubic",
            "jbu",
            "cnmf",
            "guided_tv",
            "siren_inr",
            "xfuse_unaligned",
            "xfuse_aligned",
        )
        exp5b_part: dict[str, Any] = dict(results.get("_exp5b_partial", {}))
        for shift_px in run_shifts:
            sh_key = f"{shift_px:.1f}"
            if resume and sh_key in exp5b_part:
                continue
            sh_acc: dict[str, dict[str, list[float]]] = {
                k: {"miou": [], "bf1": []} for k in drift_keys
            }
            est_sh: list[float] = []
            err_sh: list[float] = []

            for s in seeds:
                ph_d = generate_correlative_phantom_2d(
                    size=spec_size,
                    downsample_factor=4,
                    phase_b_contrast=0.35,
                    hr_noise_std=0.025,
                    poisson_counts=300.0,
                    beam_profile="moffat",
                    seed=s + 200,
                )
                m_un = XFuse(
                    backbone=backbone,
                    stride=stride,
                    dtype=torch.float32,
                    loss_fn="compositional_kl",
                    interpolation_mode="cross_attn",
                    fusion_method="spatial_mlp",
                    align_guide="none",
                    learn_psf=False,
                    tv_weight=0.005,
                    pretrained=pretrained,
                    lr=1e-2,
                    device=device,
                )
                m_al = XFuse(
                    backbone=backbone,
                    stride=stride,
                    dtype=torch.float32,
                    loss_fn="compositional_kl",
                    interpolation_mode="cross_attn",
                    fusion_method="spatial_mlp",
                    align_guide="translation",
                    max_shift_coarse_px=2.0,
                    learn_psf=True,
                    tv_weight=0.005,
                    pretrained=pretrained,
                    lr=1e-2,
                    device=device,
                )
                g_drift = _inject_shifted_moffat_guide(
                    ph_d.gt_phase_channels,
                    factor=4,
                    shift_x_hr_px=shift_px,
                    psf_sigma_hr=2.2,
                )
                r_db = bicubic_upsample_baseline(
                    g_drift,
                    (spec_size, spec_size),
                    mode="multiclass",
                    guide_normalization=norm,
                )
                r_dj = jbu_guided_baseline(
                    ph_d.hr_image, g_drift, mode="multiclass", guide_normalization=norm
                )
                r_dc = cnmf_pansharpen_baseline(
                    ph_d.hr_image,
                    g_drift,
                    mode="multiclass",
                    n_iters=35,
                    device=device,
                    guide_normalization=norm,
                )
                r_dtv = guided_variational_sr_baseline(
                    ph_d.hr_image,
                    g_drift,
                    mode="multiclass",
                    n_steps=60,
                    device=device,
                    guide_normalization=norm,
                )
                r_dsir = coordinate_inr_baseline(
                    ph_d.hr_image,
                    g_drift,
                    mode="multiclass",
                    hidden_dim=48,
                    epochs=65,
                    seed=s,
                    device=device,
                    guide_normalization=norm,
                )

                _set_all_seeds(s)
                r_un = m_un.fit_predict(
                    ph_d.hr_image,
                    g_drift,
                    epochs=35,
                    mode="multiclass",
                    tta_shifts=(),
                    tta_flips=True,
                    guide_normalization=norm,
                    reset_weights=True,
                )

                _set_all_seeds(s)
                r_al = m_al.fit_predict(
                    ph_d.hr_image,
                    g_drift,
                    epochs=42,
                    mode="multiclass",
                    tta_shifts=(),
                    tta_flips=True,
                    guide_normalization=norm,
                    reset_weights=True,
                )
                dx_hat = float(r_al.diagnostics.get("shift_x_hr_px", 0.0))
                est_sh.append(dx_hat)
                err_sh.append(abs(abs(dx_hat) - abs(shift_px)))

                for dk, rep_d in [
                    ("bicubic", r_db),
                    ("jbu", r_dj),
                    ("cnmf", r_dc),
                    ("guided_tv", r_dtv),
                    ("siren_inr", r_dsir),
                    ("xfuse_unaligned", r_un),
                    ("xfuse_aligned", r_al),
                ]:
                    dm = compute_segmentation_metrics(
                        rep_d.hard_mask[0], ph_d.gt_labels
                    )
                    sh_acc[dk]["miou"].append(dm["iou"])
                    sh_acc[dk]["bf1"].append(dm["boundary_f1"])

            exp5b_part[sh_key] = {
                "injected_shift_hr_px": shift_px,
                "image_size": spec_size,
                "estimated_shift_hr_px": _mean_std(est_sh),
                "shift_abs_error_hr_px": _mean_std(err_sh),
                "methods": {
                    dk: {mn: _mean_std(vals) for mn, vals in ddict.items()}
                    for dk, ddict in sh_acc.items()
                },
            }
            results["_exp5b_partial"] = exp5b_part
            _save_state(output_dir, results, vis_cache)
            print(f"  [5b/6] shift={shift_px}px done", flush=True)

        if all(f"{sh:.1f}" in exp5b_part for sh in shifts_list):
            results.setdefault("exp5_poisson_and_drift", {})["stage_drift_sweep"] = [
                exp5b_part[f"{sh:.1f}"] for sh in shifts_list
            ]
            results.pop("_exp5b_partial", None)
            _save_state(output_dir, results, vis_cache)

    # ── Experiment 6: Cross-Specimen Conformal Calibration (A -> Unseen B) ────
    if ({"6", "6a", "6b"} & active) and not (
        resume and "exp6_uncertainty_and_conformal" in results
    ):
        print(
            "[6/6] Running Cross-Specimen Conformal Calibration "
            "(Specimen A -> Specimen B, N=3 pairs)...",
            flush=True,
        )
        alphas = (0.05, 0.10, 0.20)
        exp6_part: dict[str, Any] = dict(results.get("_exp6_partial", {}))

        if ({"6", "6a"} & active) and not (resume and "lam" in exp6_part):
            lac_lam_acc: dict[float, dict[str, list[float]]] = {a: {} for a in alphas}
            raps_lam_acc: dict[float, dict[str, list[float]]] = {a: {} for a in alphas}
            for s in seeds:
                ph_a_lam = generate_oriented_lamellar_phantom_2d(
                    size=160, downsample_factor=8, beam_profile="moffat", seed=s
                )
                ph_b_lam = generate_oriented_lamellar_phantom_2d(
                    size=160, downsample_factor=8, beam_profile="moffat", seed=s + 1000
                )
                _set_all_seeds(s)
                m_cal_lam = XFuse(
                    backbone=backbone,
                    stride=stride,
                    dtype=torch.float32,
                    loss_fn="compositional_kl",
                    interpolation_mode="cross_attn",
                    fusion_method="spatial_mlp",
                    learn_psf=True,
                    tv_weight=0.01,
                    pretrained=pretrained,
                    lr=1e-2,
                    device=device,
                )
                rep_a_lam = m_cal_lam.fit_predict(
                    ph_a_lam.hr_image,
                    ph_a_lam.coarse_guide,
                    epochs=40,
                    mode="multiclass",
                    tta_shifts=(),
                    tta_flips=True,
                    guide_normalization=norm,
                )
                rep_b_lam = m_cal_lam.predict(
                    ph_b_lam.hr_image,
                    coarse_guide=None,
                    mode="multiclass",
                    calibrate_to_guide=False,
                )
                gt_a_lam_t = torch.from_numpy(ph_a_lam.gt_labels).unsqueeze(0)
                gt_b_lam_t = torch.from_numpy(ph_b_lam.gt_labels).unsqueeze(0)
                for a in alphas:
                    for cal_obj, acc_d in [
                        (ConformalCalibrator(alpha=a, method="lac"), lac_lam_acc),
                        (
                            ConformalCalibrator(
                                alpha=a,
                                method="raps",
                                k_reg=1,
                                lambda_reg=0.03,
                                randomized=False,
                                fit_temperature=True,
                                spatial_block_size=4,
                            ),
                            raps_lam_acc,
                        ),
                    ]:
                        cal_obj.calibrate_labels(rep_a_lam.probabilities, gt_a_lam_t)
                        ps_b, _ = cal_obj.predict_sets_and_intervals(
                            rep_b_lam.probabilities, guide_lr=None, mode="multiclass"
                        )
                        st = empirical_conformal_coverage(ps_b, gt_b_lam_t)
                        st["temperature"] = float(cal_obj.temperature)
                        for sk, sv in st.items():
                            acc_d[a].setdefault(sk, []).append(sv)
            exp6_part["lam"] = {
                "lac": [
                    {
                        "alpha": a,
                        "nominal_coverage": round(1.0 - a, 4),
                        **{sk: _mean_std(vals) for sk, vals in lac_lam_acc[a].items()},
                    }
                    for a in alphas
                ],
                "raps": [
                    {
                        "alpha": a,
                        "nominal_coverage": round(1.0 - a, 4),
                        **{sk: _mean_std(vals) for sk, vals in raps_lam_acc[a].items()},
                    }
                    for a in alphas
                ],
            }
            results["_exp6_partial"] = exp6_part
            _save_state(output_dir, results, vis_cache)
            print("  [6/6] lamellar conformal done", flush=True)

        if ({"6", "6b"} & active) and not (resume and "car" in exp6_part):
            lac_car_acc: dict[float, dict[str, list[float]]] = {a: {} for a in alphas}
            raps_car_acc: dict[float, dict[str, list[float]]] = {a: {} for a in alphas}
            for s in seeds:
                ph_a_car = generate_correlative_phantom_2d(
                    size=160,
                    downsample_factor=8,
                    phase_b_contrast=0.35,
                    hr_noise_std=0.03,
                    poisson_counts=300.0,
                    beam_profile="moffat",
                    seed=s,
                )
                ph_b_car = generate_correlative_phantom_2d(
                    size=160,
                    downsample_factor=8,
                    phase_b_contrast=0.35,
                    hr_noise_std=0.03,
                    poisson_counts=300.0,
                    beam_profile="moffat",
                    seed=s + 1000,
                )
                _set_all_seeds(s)
                m_cal_car = XFuse(
                    backbone=backbone,
                    stride=stride,
                    dtype=torch.float32,
                    loss_fn="compositional_kl",
                    interpolation_mode="cross_attn",
                    fusion_method="spatial_mlp",
                    learn_psf=True,
                    tv_weight=0.005,
                    pretrained=pretrained,
                    lr=1e-2,
                    device=device,
                )
                rep_a_car = m_cal_car.fit_predict(
                    ph_a_car.hr_image,
                    ph_a_car.coarse_guide,
                    epochs=35,
                    mode="multiclass",
                    tta_shifts=(),
                    tta_flips=True,
                    guide_normalization=norm,
                )
                rep_b_car = m_cal_car.predict(
                    ph_b_car.hr_image,
                    coarse_guide=None,
                    mode="multiclass",
                    calibrate_to_guide=False,
                )
                gt_a_car_t = torch.from_numpy(ph_a_car.gt_labels).unsqueeze(0)
                gt_b_car_t = torch.from_numpy(ph_b_car.gt_labels).unsqueeze(0)
                for a in alphas:
                    for cal_obj, acc_d in [
                        (ConformalCalibrator(alpha=a, method="lac"), lac_car_acc),
                        (
                            ConformalCalibrator(
                                alpha=a,
                                method="raps",
                                k_reg=1,
                                lambda_reg=0.03,
                                randomized=False,
                                fit_temperature=True,
                                spatial_block_size=4,
                            ),
                            raps_car_acc,
                        ),
                    ]:
                        cal_obj.calibrate_labels(rep_a_car.probabilities, gt_a_car_t)
                        ps_b, _ = cal_obj.predict_sets_and_intervals(
                            rep_b_car.probabilities, guide_lr=None, mode="multiclass"
                        )
                        st = empirical_conformal_coverage(ps_b, gt_b_car_t)
                        st["temperature"] = float(cal_obj.temperature)
                        for sk, sv in st.items():
                            acc_d[a].setdefault(sk, []).append(sv)
            exp6_part["car"] = {
                "lac": [
                    {
                        "alpha": a,
                        "nominal_coverage": round(1.0 - a, 4),
                        **{sk: _mean_std(vals) for sk, vals in lac_car_acc[a].items()},
                    }
                    for a in alphas
                ],
                "raps": [
                    {
                        "alpha": a,
                        "nominal_coverage": round(1.0 - a, 4),
                        **{sk: _mean_std(vals) for sk, vals in raps_car_acc[a].items()},
                    }
                    for a in alphas
                ],
            }
            results["_exp6_partial"] = exp6_part
            _save_state(output_dir, results, vis_cache)
            print("  [6/6] cartoon conformal done", flush=True)

        if "lam" in exp6_part and "car" in exp6_part:
            results["exp6_uncertainty_and_conformal"] = {
                "zero_label_ause_lamellar_8x": results.get("_ause_lam_8x", {}),
                "cross_specimen_lac_lamellar_8x": exp6_part["lam"]["lac"],
                "cross_specimen_raps_lamellar_8x": exp6_part["lam"]["raps"],
                "cross_specimen_lac_cartoon_8x": exp6_part["car"]["lac"],
                "cross_specimen_raps_cartoon_8x": exp6_part["car"]["raps"],
            }
            results.pop("_exp6_partial", None)
            _save_state(output_dir, results, vis_cache)

    results["last_batch_wall_clock_s"] = round(time.perf_counter() - t_suite_start, 2)
    if "plot" in active:
        fig_paths = _render_figures(
            results,
            vis_cache=vis_cache,
            fig_dir=output_dir / "figures",
            artifact_dir=artifact_dir,
        )
        results["generated_figures"] = [str(p) for p in fig_paths]
        _save_state(output_dir, results, vis_cache)

    out_json = output_dir / "referee_evaluation_results.json"
    print(
        f"Saved state to {out_json} in {results['last_batch_wall_clock_s']:.1f}s.",
        flush=True,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic multi-seed referee evaluation & figure generation."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/outputs/referee_evaluation"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Optional directory into which rendered 300-DPI PNG figures are copied.",
    )
    parser.add_argument("--backbone", default="dinov2_vits14_reg")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--stride", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from completed experiment stages in output-dir.",
    )
    parser.add_argument(
        "--stages",
        default=None,
        help="Optional comma-separated stage subset (e.g. '1,2' or '3,4' or '6,plot').",
    )
    args = parser.parse_args()
    stage_set = (
        {s.strip() for s in args.stages.split(",") if s.strip()}
        if args.stages
        else None
    )

    run_referee_evaluation(
        output_dir=args.output_dir,
        artifact_dir=args.artifact_dir,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        stride=args.stride,
        device=args.device,
        resume=bool(args.resume),
        stages=stage_set,
    )


if __name__ == "__main__":
    main()
