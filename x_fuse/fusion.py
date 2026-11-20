"""Compositional physics-guided foundation-model feature fusion engine
``CorrelativeFuser`` (aliased as ``XFuse``).

v1.0 Compositional Architecture:
- ``self.backbone`` (``HighResBackbone``): Frozen ViT feature extractor with
  optional ``ApositionalFeatureDebiaser`` and ``CrossAttentionFeatureUpsampler``
  (``interpolation_mode="cross_attn"``) or ``JointBilateralUpsampler`` (``"jbu"``).
- ``self.head`` (``LinearProbeHead | SpatialConvMLPHead`` or
  ``WindowedGuideCrossAttentionHead``):
  Composable fusion/prediction head coupled to ``PhysicsForwardModel`` (supporting
  Poisson shot-noise deviance NLL, simplex compositional KL, forward-PSF BCE/MSE,
  and raw 3D spectral/diffraction cube NMF unmixing).
- ``self.guide_aligner`` (``DifferentiableGuideAligner``): Translation, rigid,
  affine, or cubic B-spline deformable scan-distortion registration.
- ``self.calibrator`` (``ConformalCalibrator``): Optional finite-sample Split
  Conformal Prediction calibrator producing ``(1 - alpha)`` phase sets.
"""

from __future__ import annotations

import zlib
from collections import OrderedDict
from collections.abc import Sequence
from functools import partial
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import HighResBackbone, Interpolation, LayerSelection, MinInputSizeSpec
from .debias import ApositionalFeatureDebiaser
from .heads import (
    FusionOptions,
    LearnedChannelGating as LearnedChannelGating,
    LinearProbeHead,
    SpatialConvMLPHead,
    SpatialCrossAttention as SpatialCrossAttention,
    WindowedGuideCrossAttentionHead,
    XRDCrossAttention as XRDCrossAttention,
)
from .loaders import detrend_radial_cupping
from .physics import GuideNormalization, LossOptions, PhysicsForwardModel
from .registration import AlignMode, DifferentiableGuideAligner
from .tiling import run_tiled_2d
from .transforms import (
    build_default_tta,
    ensure_spatial_divisible,
    normalize_structural,
)
from .uncertainty import (
    ConformalCalibrator,
    ConformalMethod,
    UncertaintyReport,
    WelfordAccumulator,
    _safe_quantile,
    build_uncertainty_report,
    compute_hard_phase_mask,
    compute_posterior_uncertainty,
    match_coarse_priors,
)
from .upsamplers import (
    CrossAttentionFeatureUpsampler,
    ForwardPSFDownsampler,
    PSFProfile,
)
from .utils import guide_to_tensor
from .vit_patch import AttentionOptions

PoreMaskSpec = (
    np.ndarray | torch.Tensor | Literal["auto_otsu", "auto_otsu_fringe_guarded"] | None
)


def _otsu_threshold_1d(values: np.ndarray, n_bins: int = 256) -> float:
    """Compute exact 1D Otsu threshold maximizing inter-class variance."""
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return float(finite[0]) if finite.size == 1 else 0.0
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi - lo < 1e-12:
        return lo
    hist, bin_edges = np.histogram(finite, bins=n_bins, range=(lo, hi))
    prob = hist.astype(np.float64) / max(float(hist.sum()), 1.0)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    sigma_b2 = np.where(
        denom > 1e-12, ((mu_t * omega - mu) ** 2) / np.maximum(denom, 1e-12), -1.0
    )
    best_idx = int(np.argmax(sigma_b2))
    return float(centers[best_idx])


def _tie_homogeneity_filter_2d(img_2d: np.ndarray, alpha_vox2: float) -> np.ndarray:
    """Apply Paganin/Tie-homogeneity Fourier damping ``1 / (1 + alpha * |k|^2)``
    with reflect padding to suppress 1–2 px Fresnel phase-contrast overshoots
    along pore walls before Otsu binarization.
    """
    if alpha_vox2 <= 0.0:
        return img_2d
    h, w = img_2d.shape
    pad_y = min(8, max(1, h // 4))
    pad_x = min(8, max(1, w // 4))
    padded = np.pad(
        img_2d.astype(np.float32), ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect"
    )
    ph, pw = padded.shape
    ky = (2.0 * np.pi * np.fft.fftfreq(ph)).astype(np.float32)[:, None]
    kx = (2.0 * np.pi * np.fft.rfftfreq(pw)).astype(np.float32)[None, :]
    filt = 1.0 / (1.0 + float(alpha_vox2) * (ky * ky + kx * kx))
    reg = np.fft.irfft2(np.fft.rfft2(padded) * filt, s=(ph, pw)).astype(np.float32)
    return reg[pad_y : pad_y + h, pad_x : pad_x + w]


def segment_pores_otsu(
    high_res: np.ndarray | torch.Tensor,
    *,
    tie_homogeneity_alpha_vox2: float = 0.0,
    detrend_cupping: bool = False,
    roi_mask: np.ndarray | None = None,
    dark_pores: bool = True,
) -> np.ndarray:
    """Derive a boolean structural void/pore mask ``M_pore`` from a 2D ``(H, W)``
    micrograph or 3D ``(Z, Y, X)`` tomogram without blurring ``high_res`` itself.

    When ``tie_homogeneity_alpha_vox2 > 0`` (e.g. ``0.6``), applies a Fourier
    Tie-homogeneity phase-retrieval filter solely to compute a Fresnel-fringe-free
    segmentation threshold, preventing bright/dark Ram-Lak edge overshoots from
    punching false pinholes in solid walls or leaving 1-px chemical shells in pores.
    """
    arr = (
        high_res.detach().cpu().float().numpy()
        if isinstance(high_res, torch.Tensor)
        else np.asarray(high_res, dtype=np.float32)
    )
    if arr.ndim not in (2, 3):
        raise ValueError(
            f"segment_pores_otsu expects 2D (H, W) or 3D (Z, Y, X), got {arr.shape}."
        )

    finite_mask = np.isfinite(arr)
    eff_roi = (
        (np.asarray(roi_mask, dtype=bool) & finite_mask)
        if roi_mask is not None
        else finite_mask
    )
    finite_vals = arr[finite_mask]
    if finite_vals.size == 0:
        return np.zeros(arr.shape, dtype=bool)
    f_min = float(np.min(finite_vals))
    f_max = float(np.max(finite_vals))
    f_med = float(np.median(finite_vals))
    arr_work = np.nan_to_num(arr, nan=f_med, posinf=f_max, neginf=f_min)

    if detrend_cupping:
        arr_work = detrend_radial_cupping(arr_work, valid_mask=eff_roi)

    alpha = float(max(0.0, tie_homogeneity_alpha_vox2))
    if alpha > 0.0:
        if arr_work.ndim == 2:
            arr_work = _tie_homogeneity_filter_2d(arr_work, alpha)
        else:
            arr_work = np.stack(
                [
                    _tie_homogeneity_filter_2d(arr_work[z], alpha)
                    for z in range(arr_work.shape[0])
                ],
                axis=0,
            )

    sample_vals = arr_work[eff_roi] if np.any(eff_roi) else arr_work
    thresh = _otsu_threshold_1d(sample_vals)
    pore_bool = (arr_work < thresh) if dark_pores else (arr_work > thresh)
    if roi_mask is not None:
        pore_bool = pore_bool & np.asarray(roi_mask, dtype=bool)
    return np.asarray(pore_bool, dtype=bool)


def _tensor_content_digest(x: torch.Tensor) -> tuple[tuple[int, ...], int]:
    """Compute a fast strided CRC32 content signature over up to 4096 samples."""
    flat = x.detach().float().reshape(-1)
    n_el = int(flat.numel())
    if n_el == 0:
        return tuple(x.shape), 0
    step = max(1, n_el // 4096)
    sub = flat[::step].cpu().numpy()
    return tuple(x.shape), zlib.crc32(sub.tobytes())


def compute_channel_correlation_gates(
    features: torch.Tensor,
    guide_map: torch.Tensor,
    loss_fn: LossOptions = "pearson",
    top_k: int | None = None,
) -> torch.Tensor:
    """Vectorized per-channel relevance gates ``(M, C)`` between ``features``
    ``(B, C, H, W)`` and ``guide_map`` ``(1, M, H_lr, W_lr)`` without Python loops.
    """
    bsz, channels, height, width = features.shape
    if top_k is None:
        top_k = max(1, channels // 4)
    top_k = max(1, min(int(top_k), channels))

    if guide_map.dim() == 2:
        guide_map = guide_map.unsqueeze(0).unsqueeze(0)
    elif guide_map.dim() == 3:
        guide_map = guide_map.unsqueeze(0)

    guide_resized = F.interpolate(
        guide_map.float().to(features.device),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    if guide_resized.shape[0] == 1 and bsz > 1:
        guide_resized = guide_resized.expand(bsz, -1, -1, -1)

    n_guides = guide_resized.shape[1]
    feat_mat = features.float().permute(1, 0, 2, 3).reshape(channels, -1)
    guide_mat = guide_resized.permute(1, 0, 2, 3).reshape(n_guides, -1)
    n_pts = feat_mat.shape[1]

    if loss_fn == "pearson":
        if n_pts <= 1:
            scores = torch.zeros(
                (n_guides, channels), device=features.device, dtype=torch.float32
            )
        else:
            f_centered = feat_mat - feat_mat.mean(dim=1, keepdim=True)
            g_centered = guide_mat - guide_mat.mean(dim=1, keepdim=True)
            f_norm = f_centered.norm(dim=1, keepdim=True)
            g_norm = g_centered.norm(dim=1, keepdim=True)
            denom = g_norm @ f_norm.T
            cov = g_centered @ f_centered.T
            scores = torch.where(
                denom > 1e-12, cov / denom.clamp(min=1e-12), torch.zeros_like(cov)
            )
            scores = torch.nan_to_num(scores, nan=0.0).clamp(0.0, 1.0)
    else:
        g_min = guide_mat.min(dim=1, keepdim=True).values
        g_max = guide_mat.max(dim=1, keepdim=True).values
        g_normed = (guide_mat - g_min) / (g_max - g_min + 1e-8)
        f_min = feat_mat.min(dim=1, keepdim=True).values
        f_max = feat_mat.max(dim=1, keepdim=True).values
        f_normed = ((feat_mat - f_min) / (f_max - f_min + 1e-8)).clamp(1e-6, 1.0 - 1e-6)
        log_f = torch.log(f_normed)
        log_1mf = torch.log(1.0 - f_normed)
        bce = -(g_normed @ log_f.T + (1.0 - g_normed) @ log_1mf.T) / max(n_pts, 1)
        scores = (1.0 - bce / 0.693).clamp(0.0, 1.0)

    topk_vals, topk_idx = scores.topk(top_k, dim=1)
    thresholds = topk_vals[:, -1:]
    mask = torch.zeros_like(scores)
    mask.scatter_(1, topk_idx, 1.0)
    ties_above_thresh = (scores >= thresholds) & (thresholds > 0)
    gates = torch.where(
        ties_above_thresh.sum(dim=1, keepdim=True) == top_k,
        scores * ties_above_thresh.float(),
        scores * mask,
    )
    return gates.to(features.dtype)


class XRDFusionMethod(nn.Module):
    """Applies aligned TTA transforms to the coarse physical guide map and
    modulates dense ViT features via correlation gating, linear probe gating,
    spatial Conv-MLP gating, or windowed cross-attention.
    """

    def __init__(
        self,
        xrd_img: torch.Tensor,
        transform: list[partial],
        require_grad: bool = False,
        learned_gating: nn.Module | None = None,
        spatial_attention: nn.Module | None = None,
        spatial_mlp: nn.Module | None = None,
        loss_fn: LossOptions = "pearson",
    ) -> None:
        super().__init__()
        self.require_grad = require_grad
        self.xrd_image = xrd_img
        self.transform = transform
        self.learned_gating_module = learned_gating
        self.spatial_attention_module = spatial_attention
        self.spatial_mlp_module = spatial_mlp
        self.loss_fn: LossOptions = loss_fn
        self._tr_cache_key: tuple[Any, ...] | None = None
        self._tr_cache_val: torch.Tensor | None = None

    def get_tr(self, hr_hw: tuple[int, int] | None = None) -> torch.Tensor:
        """Apply all forward TTA transforms to the guide tensor as a batch.
        When ``hr_hw`` is supplied and differs from ``xrd_image``'s spatial shape,
        shifts/flips are applied in HR pixel coordinates before pooling back to LR
        so 1-pixel HR shifts equal ``1 / S`` coarse pixels (not ``S`` HR px).
        """
        base = self.xrd_image
        if base.ndim == 2:
            base = base.unsqueeze(0).unsqueeze(0)
        elif base.ndim == 3:
            base = base.unsqueeze(0)
        if len(self.transform) == 0:
            return base.unsqueeze(0)

        lr_hw = (int(base.shape[-2]), int(base.shape[-1]))
        cache_key = (
            int(base.data_ptr()),
            tuple(base.shape),
            len(self.transform),
            hr_hw,
        )
        if (
            self._tr_cache_key == cache_key
            and self._tr_cache_val is not None
            and self._tr_cache_val.device == base.device
        ):
            return self._tr_cache_val

        use_subpixel_hr = hr_hw is not None and hr_hw != lr_hw
        src = (
            F.interpolate(
                base.float(), size=hr_hw, mode="bilinear", align_corners=False
            )
            if use_subpixel_hr
            else base
        )
        tr_list = []
        for tr_fn in self.transform:
            t_out = tr_fn(src)
            if t_out.ndim == 3:
                t_out = t_out.unsqueeze(0)
            if use_subpixel_hr:
                t_out = F.adaptive_avg_pool2d(t_out, output_size=lr_hw).to(base.dtype)
            tr_list.append(t_out)
        stacked = torch.stack(tr_list)
        self._tr_cache_key = cache_key
        self._tr_cache_val = stacked
        return stacked

    def __translen__(self, transform: list[partial]) -> int:
        return len(transform)

    def forward_xrd(
        self,
        x: torch.Tensor,
        idx: int,
        xrd_fusion_method: FusionOptions,
        top_k: int | None = None,
        hr_hw: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        tr_xrd = self.get_tr(hr_hw=hr_hw)[idx]
        if xrd_fusion_method == "gating":
            return self._direct_correlation_gating(x, tr_xrd, top_k=top_k)
        if xrd_fusion_method == "learned_gating":
            return self._learned_channel_gating(x, tr_xrd)
        if xrd_fusion_method == "spatial_mlp":
            if self.spatial_mlp_module is None:
                self.spatial_mlp_module = SpatialConvMLPHead(
                    feat_dim=x.shape[1], loss_fn=self.loss_fn
                ).to(device=x.device, dtype=x.dtype)
            return self.spatial_mlp_module(x, tr_xrd)
        if xrd_fusion_method == "attention":
            return self._spatial_attention(x, tr_xrd)
        raise ValueError(f"Unknown fusion method '{xrd_fusion_method}'.")

    def _direct_correlation_gating(
        self,
        features: torch.Tensor,
        xrd_map: torch.Tensor,
        top_k: int | None = None,
    ) -> torch.Tensor:
        gates_mc = compute_channel_correlation_gates(
            features, xrd_map, loss_fn=self.loss_fn, top_k=top_k
        )
        gates = gates_mc.amax(dim=0)
        return features * gates.view(1, -1, 1, 1)

    def _learned_channel_gating(
        self, features: torch.Tensor, xrd_map: torch.Tensor
    ) -> torch.Tensor:
        if self.learned_gating_module is None:
            self.learned_gating_module = LinearProbeHead(feat_dim=features.shape[1]).to(
                device=features.device, dtype=features.dtype
            )
        return self.learned_gating_module(features, xrd_map)

    def _spatial_attention(
        self, features: torch.Tensor, xrd_map: torch.Tensor
    ) -> torch.Tensor:
        n_guides = (
            xrd_map.shape[1]
            if xrd_map.ndim == 4
            else (xrd_map.shape[0] if xrd_map.ndim == 3 else 1)
        )
        if self.spatial_attention_module is None:
            self.spatial_attention_module = WindowedGuideCrossAttentionHead(
                feat_dim=features.shape[1], num_heads=8, guide_channels=n_guides
            ).to(device=features.device, dtype=features.dtype)
        return self.spatial_attention_module(features, xrd_map)


GuideFusionMethod = XRDFusionMethod


class CorrelativeFuser(nn.Module):
    """Compositional correlative fusion solver combining a ``HighResBackbone``,
    optional ``ApositionalFeatureDebiaser``, ``CrossAttentionFeatureUpsampler`` /
    ``JointBilateralUpsampler``, pluggable ``FusionHead`` + ``PhysicsForwardModel``,
    ``DifferentiableGuideAligner`` (including cubic B-splines), and finite-sample
    ``ConformalCalibrator``.
    """

    def __init__(
        self,
        dino_name: str | None = None,
        xrd_fuse_method: FusionOptions | None = None,
        stride: int = 4,
        pca_dim: int = -1,
        track_grad: bool = False,
        dtype: torch.dtype | int | None = None,
        loss_fn: LossOptions | None = None,
        guide_channels: int = 1,
        interpolation_mode: Interpolation = "nearest-exact",
        pretrained: bool = True,
        lr: float = 1e-3,
        *,
        backbone: str | None = None,
        fusion_method: FusionOptions | None = None,
        upsampler: Interpolation | None = None,
        learned_upsampler: bool = False,
        upsampler_orientation: bool = True,
        upsampler_subtoken: bool = True,
        debias_positional: bool = False,
        debias_degree: int = 2,
        layers: LayerSelection = "last",
        align_guide: AlignMode = "none",
        max_shift_coarse_px: float = 1.5,
        bspline_grid_size: int = 6,
        learn_psf: bool = False,
        psf_profile: PSFProfile = "gaussian",
        conformal_alpha: float | None = None,
        conformal_method: ConformalMethod = "lac",
        conformal_fit_temperature: bool = False,
        conformal_spatial_block_size: int = 1,
        spectral_channels: int | None = None,
        tv_weight: float = 1e-3,
        guide_residual: bool = False,
        tile_size: int | None = None,
        tile_overlap: int = 64,
        min_input_size: MinInputSizeSpec = "auto",
        pore_mask: PoreMaskSpec = None,
        pore_homogeneity_alpha_vox2: float = 0.0,
        pore_detrend_cupping: bool = False,
        cache_features: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        resolved_backbone = backbone or dino_name or "dinov2_vits14"
        resolved_fusion: FusionOptions = (
            fusion_method
            or xrd_fuse_method
            or ("spatial_mlp" if guide_residual else "learned_gating")
        )
        resolved_loss: LossOptions = (
            loss_fn
            if loss_fn is not None
            else (
                "forward_psf_bce"
                if (learn_psf or psf_profile != "gaussian" or pore_mask is not None)
                else "bce"
            )
        )
        resolved_interp: Interpolation = upsampler or interpolation_mode
        device = kwargs.get("device", None)
        if dtype is None:
            is_cuda = str(device).startswith("cuda") if device is not None else False
            resolved_dtype: torch.dtype | int = (
                torch.float16 if is_cuda else torch.float32
            )
        else:
            resolved_dtype = dtype

        self.backbone_extractor = HighResBackbone(
            resolved_backbone,
            stride=stride,
            pca_dim=pca_dim,
            dtype=resolved_dtype,
            track_grad=track_grad,
            pretrained=pretrained,
            interpolation_mode=resolved_interp,
            layers=layers,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            debias_positional=debias_positional,
            debias_degree=debias_degree,
            learned_upsampler=learned_upsampler,
            upsampler_orientation=upsampler_orientation,
            upsampler_subtoken=upsampler_subtoken,
            min_input_size=min_input_size,
            **kwargs,
        )

        self.backbone_name: str = resolved_backbone
        self.fusion_method: FusionOptions = resolved_fusion
        self.xrd_fuse_method: FusionOptions = resolved_fusion
        self.loss_fn: LossOptions = resolved_loss
        self.guide_channels: int = max(1, int(guide_channels))
        self.spectral_channels: int | None = spectral_channels
        self.lr: float = float(lr)
        self.align_guide: AlignMode = align_guide
        self.max_shift_coarse_px: float = float(max_shift_coarse_px)
        self.bspline_grid_size: int = int(bspline_grid_size)
        self.learn_psf: bool = bool(learn_psf)
        self.psf_profile: PSFProfile = psf_profile
        self.learned_upsampler: bool = bool(learned_upsampler)
        self.upsampler_orientation: bool = bool(upsampler_orientation)
        self.upsampler_subtoken: bool = bool(upsampler_subtoken)
        self.tv_weight: float = float(tv_weight)
        self.guide_residual: bool = bool(guide_residual)
        self.pore_mask_spec: PoreMaskSpec = pore_mask
        self.pore_homogeneity_alpha_vox2: float = float(pore_homogeneity_alpha_vox2)
        self.pore_detrend_cupping: bool = bool(pore_detrend_cupping)
        self._active_pore_mask_hr: torch.Tensor | None = None
        self.cache_features: bool = bool(cache_features)
        self.conformal_alpha: float | None = conformal_alpha
        self.conformal_method: ConformalMethod = conformal_method
        self.conformal_fit_temperature: bool = bool(conformal_fit_temperature)
        self.conformal_spatial_block_size: int = max(
            1, int(conformal_spatial_block_size)
        )

        self.calibrator: ConformalCalibrator | None = (
            ConformalCalibrator(
                alpha=conformal_alpha,
                method=conformal_method,
                fit_temperature=self.conformal_fit_temperature,
                spatial_block_size=self.conformal_spatial_block_size,
            )
            if conformal_alpha is not None
            else None
        )

        self.learned_gating: LinearProbeHead | None = None
        self.spatial_mlp: SpatialConvMLPHead | None = None
        self.spatial_attention: WindowedGuideCrossAttentionHead | None = None
        self.guide_aligner: DifferentiableGuideAligner = DifferentiableGuideAligner(
            mode=align_guide,
            max_shift_coarse_px=max_shift_coarse_px,
            control_grid_size=bspline_grid_size,
        )
        self.fusion_optimizer: torch.optim.Optimizer | None = None
        self.align_optimizer: torch.optim.Optimizer | None = None
        self._init_fusion_heads(
            self.guide_channels,
            device=device,
            spectral_channels=spectral_channels,
        )

        self.xrd_fusion_module: XRDFusionMethod | None = None
        self._unmixed_guide_lr: torch.Tensor | None = None
        self._fit_guide_ptr: int | None = None
        self._fit_hr_sig: tuple[tuple[int, ...], int] | None = None
        self._tta_cache: OrderedDict[
            tuple[Any, ...], tuple[str, list[torch.Tensor]]
        ] = OrderedDict()
        self._max_cache_entries: int = 8
        self._last_loss_history: list[float] = []

    # ── Clean Compositional Forwarding Properties (100% Back-Compat) ─────────

    @property
    def backbone(self) -> HighResBackbone:
        return self.backbone_extractor

    @property
    def debiaser(self) -> ApositionalFeatureDebiaser:
        return self.backbone_extractor.debiaser

    @property
    def debias_positional(self) -> bool:
        return self.backbone_extractor.debiaser.enabled

    @debias_positional.setter
    def debias_positional(self, value: bool) -> None:
        new_val = bool(value)
        if self.backbone_extractor.debiaser.enabled != new_val:
            self.backbone_extractor.debiaser.enabled = new_val
            self.clear_feature_cache()

    @property
    def head(self) -> nn.Module | None:
        if self.xrd_fuse_method == "learned_gating":
            return self.learned_gating
        if self.xrd_fuse_method == "spatial_mlp":
            return self.spatial_mlp
        if self.xrd_fuse_method == "attention":
            return self.spatial_attention
        return None

    @property
    def dinov2(self) -> nn.Module:
        return self.backbone_extractor.dinov2

    @property
    def feat_dim(self) -> int:
        return self.backbone_extractor.feat_dim

    @property
    def original_patch_size(self) -> int:
        return self.backbone_extractor.original_patch_size

    @property
    def stride(self) -> tuple[int, int]:
        return self.backbone_extractor.stride

    @property
    def tile_size(self) -> int | None:
        return self.backbone_extractor.tile_size

    @tile_size.setter
    def tile_size(self, value: int | None) -> None:
        self.backbone_extractor.tile_size = value
        self.clear_feature_cache()

    @property
    def tile_overlap(self) -> int:
        return self.backbone_extractor.tile_overlap

    @tile_overlap.setter
    def tile_overlap(self, value: int) -> None:
        self.backbone_extractor.tile_overlap = int(value)
        self.clear_feature_cache()

    @property
    def min_input_size(self) -> MinInputSizeSpec:
        return self.backbone_extractor.min_input_size

    @min_input_size.setter
    def min_input_size(self, value: MinInputSizeSpec) -> None:
        self.backbone_extractor.min_input_size = value
        self.clear_feature_cache()

    @property
    def interpolation_mode(self) -> Interpolation:
        return self.backbone_extractor.interpolation_mode

    @interpolation_mode.setter
    def interpolation_mode(self, value: Interpolation) -> None:
        self.backbone_extractor.interpolation_mode = value
        self.clear_feature_cache()

    @property
    def dtype(self) -> torch.dtype:
        return self.backbone_extractor.dtype

    @property
    def track_grad(self) -> bool:
        return self.backbone_extractor.track_grad

    @property
    def transforms(self) -> list[partial]:
        return self.backbone_extractor.transforms

    @property
    def inverse_transforms(self) -> list[partial]:
        return self.backbone_extractor.inverse_transforms

    @property
    def layer_indices(self) -> list[int] | None:
        return self.backbone_extractor.layer_indices

    def set_transforms(
        self, transforms: list[partial], inv_transforms: list[partial]
    ) -> None:
        self.backbone_extractor.set_transforms(transforms, inv_transforms)
        self.clear_feature_cache()

    def get_n_patches(self, img_h: int, img_w: int) -> tuple[int, int]:
        return self.backbone_extractor.get_n_patches(img_h, img_w)

    def get_transformed_input_batch(
        self, x: torch.Tensor, transforms: list[partial]
    ) -> torch.Tensor:
        return self.backbone_extractor.get_transformed_input_batch(x, transforms)

    def extract_patch_tokens(
        self, transformed_img: torch.Tensor, attn_choice: AttentionOptions = "none"
    ) -> torch.Tensor:
        return self.backbone_extractor.extract_patch_tokens(
            transformed_img, attn_choice=attn_choice
        )

    def extract_patch_grid(
        self, transformed_img: torch.Tensor, attn_choice: AttentionOptions = "none"
    ) -> torch.Tensor:
        return self.backbone_extractor.extract_patch_grid(
            transformed_img, attn_choice=attn_choice
        )

    def _upsample_patch_map(
        self,
        patch_map: torch.Tensor,
        target_hw: tuple[int, int],
        guidance_img: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.backbone_extractor._upsample_patch_map(
            patch_map, target_hw, guidance_img=guidance_img
        )

    def set_model_stride(
        self, dino_model: nn.Module, stride_l: int, verbose: bool = False
    ) -> None:
        self.backbone_extractor.set_model_stride(
            dino_model, stride_l=stride_l, verbose=verbose
        )
        self.clear_feature_cache()

    # ── Head & Optimizer Initialization ──────────────────────────────────────

    def clear_feature_cache(self) -> None:
        """Release any cached frozen TTA feature tensors."""
        if hasattr(self, "_tta_cache"):
            self._tta_cache.clear()

    def _active_physics(self) -> PhysicsForwardModel | None:
        if self.xrd_fuse_method == "learned_gating" and self.learned_gating is not None:
            return self.learned_gating.physics
        if self.xrd_fuse_method == "spatial_mlp" and self.spatial_mlp is not None:
            return self.spatial_mlp.physics
        if self.xrd_fuse_method == "attention" and self.spatial_attention is not None:
            return self.spatial_attention.physics
        return None

    def _active_downsampler(self) -> ForwardPSFDownsampler | None:
        phys = self._active_physics()
        return phys.psf_downsampler if phys is not None else None

    def unmixed_spectra(self) -> torch.Tensor | None:
        """Return learned ``(M, E)`` spectral basis when unmixing is active."""
        phys = self._active_physics()
        if phys is not None and phys.spectral_dict is not None:
            return phys.spectral_dict.spectra().detach()
        return None

    def _init_fusion_heads(
        self,
        guide_channels: int,
        device: str | torch.device | None = None,
        spectral_channels: int | None = None,
    ) -> None:
        self.guide_channels = max(1, int(guide_channels))
        self.spectral_channels = spectral_channels
        self.learned_gating = None
        self.spatial_mlp = None
        self.spatial_attention = None

        extra_params: list[nn.Parameter] = []
        if self.learned_upsampler and self.interpolation_mode == "cross_attn":
            extra_params = list(
                self.backbone_extractor.cross_attn_upsampler.parameters()
            )

        if self.xrd_fuse_method == "learned_gating":
            self.learned_gating = LinearProbeHead(
                feat_dim=self.feat_dim,
                loss_fn=self.loss_fn,
                out_channels=self.guide_channels,
                learn_psf=self.learn_psf,
                spectral_channels=spectral_channels,
                psf_profile=self.psf_profile,
            )
            if device is not None:
                self.learned_gating = self.learned_gating.to(device)
            self.fusion_optimizer = torch.optim.Adam(
                list(self.learned_gating.parameters()) + extra_params, lr=self.lr
            )
        elif self.xrd_fuse_method == "spatial_mlp":
            self.spatial_mlp = SpatialConvMLPHead(
                feat_dim=self.feat_dim,
                out_channels=self.guide_channels,
                loss_fn=self.loss_fn,
                learn_psf=self.learn_psf,
                spectral_channels=spectral_channels,
                tv_weight=self.tv_weight,
                guide_residual=self.guide_residual,
                psf_profile=self.psf_profile,
                orientation_aware=self.upsampler_orientation,
            )
            if device is not None:
                self.spatial_mlp = self.spatial_mlp.to(device)
            self.fusion_optimizer = torch.optim.Adam(
                list(self.spatial_mlp.parameters()) + extra_params, lr=self.lr
            )
        elif self.xrd_fuse_method == "attention":
            self.spatial_attention = WindowedGuideCrossAttentionHead(
                feat_dim=self.feat_dim,
                num_heads=8,
                guide_channels=self.guide_channels,
                loss_fn=self.loss_fn,
                learn_psf=self.learn_psf,
                spectral_channels=spectral_channels,
                psf_profile=self.psf_profile,
            )
            if device is not None:
                self.spatial_attention = self.spatial_attention.to(device)
            self.fusion_optimizer = torch.optim.Adam(
                list(self.spatial_attention.parameters()) + extra_params, lr=self.lr
            )

        self.guide_aligner = DifferentiableGuideAligner(
            mode=self.align_guide,
            max_shift_coarse_px=self.max_shift_coarse_px,
            control_grid_size=self.bspline_grid_size,
        )
        if device is not None:
            self.guide_aligner = self.guide_aligner.to(device)
        align_params = list(self.guide_aligner.parameters())
        self.align_optimizer = (
            torch.optim.Adam(align_params, lr=max(self.lr * 2.0, 2e-2))
            if align_params
            else None
        )

    def reset_trainable_weights(self) -> None:
        """Reset all specimen-specific trainable head, upsampler, aligner, and
        physics parameters (plus Adam momentum state) before fitting a new sample.
        """
        dev = next(self.dinov2.parameters()).device
        if self.learned_upsampler and self.interpolation_mode == "cross_attn":
            self.backbone_extractor.cross_attn_upsampler = (
                CrossAttentionFeatureUpsampler(
                    feat_dim=self.feat_dim,
                    key_dim=16,
                    radius=2,
                    learned=True,
                    orientation_aware=self.upsampler_orientation,
                    subtoken_offsets=self.upsampler_subtoken,
                ).to(dev)
            )
        self._init_fusion_heads(
            self.guide_channels,
            device=dev,
            spectral_channels=self.spectral_channels,
        )
        self.guide_aligner.reset_parameters()
        align_params = list(self.guide_aligner.parameters())
        self.align_optimizer = (
            torch.optim.Adam(align_params, lr=max(self.lr * 2.0, 2e-2))
            if align_params
            else None
        )
        phys = self._active_physics()
        if phys is not None:
            phys.reset_specimen_state()
        if self.xrd_fusion_module is not None:
            self._bind_guide_only(self.xrd_fusion_module.xrd_image)

    def _bind_guide_only(self, guide_tensor: torch.Tensor) -> None:
        """Update the bound guide tensor in ``xrd_fusion_module`` without
        re-initializing trained head parameters (used during multi-image inference).
        """
        if guide_tensor.ndim == 2:
            guide_tensor = guide_tensor.unsqueeze(0).unsqueeze(0)
        elif guide_tensor.ndim == 3:
            guide_tensor = guide_tensor.unsqueeze(0)
        dev = next(self.dinov2.parameters()).device
        if len(self.transforms) == 0:
            fwd, inv = build_default_tta(shift_dists=[1], include_flips=True)
            self.set_transforms(fwd, inv)
        self.xrd_fusion_module = XRDFusionMethod(
            xrd_img=guide_tensor.to(dev),
            transform=self.transforms,
            require_grad=self.track_grad,
            learned_gating=self.learned_gating,
            spatial_attention=self.spatial_attention,
            spatial_mlp=self.spatial_mlp,
            loss_fn=self.loss_fn,
        )

    def set_guide_transforms(
        self,
        guide_tensor: torch.Tensor,
        fwd: list[partial],
        inv: list[partial],
        *,
        n_components: int | None = None,
        spectral_unmix: bool | None = None,
    ) -> None:
        """Bind a ``(1, M, H_lr, W_lr)`` physical guide tensor (or 3D spectral
        cube ``(1, E, H_lr, W_lr)`` when ``spectral_unmix=True``) and TTA views.
        """
        self.set_transforms(fwd, inv)
        if guide_tensor.ndim == 2:
            guide_tensor = guide_tensor.unsqueeze(0).unsqueeze(0)
        elif guide_tensor.ndim == 3:
            guide_tensor = guide_tensor.unsqueeze(0)

        dev = next(self.dinov2.parameters()).device
        g_dev = guide_tensor.to(dev)
        if spectral_unmix is None:
            spectral_unmix = (
                self.spectral_channels is not None
                and int(g_dev.shape[1]) == self.spectral_channels
            )

        if spectral_unmix:
            spec_ch = int(g_dev.shape[1])
            m_comp = int(n_components or self.guide_channels)
            if self.spectral_channels != spec_ch or self.guide_channels != m_comp:
                self._init_fusion_heads(m_comp, device=dev, spectral_channels=spec_ch)
            phys = self._active_physics()
            if phys is not None and phys.spectral_dict is not None:
                self._unmixed_guide_lr = phys.spectral_dict.initialize_from_cube(g_dev)
                self._fit_guide_ptr = int(g_dev.data_ptr())
        else:
            self._unmixed_guide_lr = None
            self._fit_guide_ptr = None
            n_guides = int(g_dev.shape[1])
            if n_guides != self.guide_channels or self.spectral_channels is not None:
                self._init_fusion_heads(n_guides, device=dev, spectral_channels=None)

        self.xrd_fusion_module = XRDFusionMethod(
            xrd_img=g_dev,
            transform=self.transforms,
            require_grad=self.track_grad,
            learned_gating=self.learned_gating,
            spatial_attention=self.spatial_attention,
            spatial_mlp=self.spatial_mlp,
            loss_fn=self.loss_fn,
        )

    def set_xrd_transforms(
        self, xrd_tensor: torch.Tensor, fwd: list[partial], inv: list[partial]
    ) -> None:
        """Backward-compatible alias for ``set_guide_transforms``."""
        self.set_guide_transforms(xrd_tensor, fwd, inv)

    # ── TTA Feature Caching & Extraction ─────────────────────────────────────

    def _transform_signature(self) -> tuple[Any, ...]:
        sig = []
        for tr in self.transforms:
            fn_name = getattr(getattr(tr, "func", tr), "__name__", "fn")
            kw = tuple(sorted((getattr(tr, "keywords", None) or {}).items()))
            sig.append((fn_name, str(kw)))
        return tuple(sig)

    def _cache_key(self, x: torch.Tensor, want_hr: bool) -> tuple[Any, ...]:
        shp, crc = _tensor_content_digest(x)
        deb = self.backbone_extractor.debiaser
        return (
            shp,
            crc,
            str(x.device),
            str(self.dtype),
            self.stride,
            self._transform_signature(),
            self.interpolation_mode,
            deb.enabled,
            deb.mode,
            deb.degree,
            deb.r2_threshold,
            self.tile_size,
            self.backbone_extractor.resolved_min_input_size,
            want_hr,
        )

    def _extract_tta_views(
        self, x: torch.Tensor, want_hr: bool
    ) -> tuple[str, list[torch.Tensor]]:
        """Extract frozen ViT features per TTA view with optional LRU caching,
        single-pass full-FOV debiasing, and autograd preservation for
        ``learned_upsampler=True``.
        """
        _, img_h, img_w = x.shape
        is_tiled = self.tile_size is not None and max(img_h, img_w) > self.tile_size
        train_up = self.learned_upsampler and self.interpolation_mode == "cross_attn"
        can_cache = (
            self.cache_features and not self.track_grad and not (train_up and is_tiled)
        )
        cache_hr = want_hr and not train_up and (img_h * img_w <= 512 * 512)
        key = self._cache_key(x, cache_hr)
        if can_cache and key in self._tta_cache:
            self._tta_cache.move_to_end(key)
            return self._tta_cache[key]

        with torch.no_grad():
            x_in = x.to(self.dtype) if self.dtype != torch.float32 else x
            img_batch = self.get_transformed_input_batch(x_in, self.transforms)
        n_transforms = len(self.transforms)

        if is_tiled and self.tile_size is not None:
            tile_sz = self.tile_size
            hr_views: list[torch.Tensor] = []
            for i in range(n_transforms):
                tr_img = img_batch[i]

                def _single_tr_tile(tile_chw: torch.Tensor) -> torch.Tensor:
                    t_in = tile_chw.unsqueeze(0)
                    _, th, tw = tile_chw.shape
                    with torch.no_grad():
                        perm = self.extract_patch_grid(t_in, attn_choice="none")
                    return self._upsample_patch_map(
                        perm.float(), (th, tw), guidance_img=t_in
                    )

                hr_v = run_tiled_2d(
                    _single_tr_tile,
                    tr_img,
                    tile_size=tile_sz,
                    overlap=self.tile_overlap,
                    patch_size=self.original_patch_size,
                    stride=self.stride[0],
                )
                # Full-FOV debiasing after stitching prevents tile-boundary seams
                hr_views.append(self.backbone_extractor.debiaser(hr_v))
            if can_cache and (img_h * img_w <= 512 * 512):
                self._tta_cache[key] = ("hr", hr_views)
                while len(self._tta_cache) > self._max_cache_entries:
                    self._tta_cache.popitem(last=False)
            return ("hr", hr_views)

        out_views: list[torch.Tensor] = []
        kind = "hr" if cache_hr else "patch"
        for i in range(n_transforms):
            transformed_img = img_batch[i].unsqueeze(0)
            with torch.no_grad():
                permuted = self.extract_patch_grid(transformed_img, attn_choice="none")
                debiased = self.backbone_extractor.debiaser(permuted.float())
            if cache_hr:
                with torch.no_grad():
                    hr_f = self._upsample_patch_map(
                        debiased, (img_h, img_w), guidance_img=transformed_img
                    )
                out_views.append(hr_f)
            else:
                out_views.append(debiased)

        result = (kind, out_views)
        if can_cache:
            self._tta_cache[key] = result
            while len(self._tta_cache) > self._max_cache_entries:
                self._tta_cache.popitem(last=False)
        return result

    @torch.no_grad()
    def forward_raw_features(
        self, x: torch.Tensor, attn_choice: AttentionOptions = "none"
    ) -> torch.Tensor:
        """Extract dense TTA-averaged ViT features ``(1, C, H, W)`` without guide
        modulation (delegating directly to ``self.backbone_extractor``).
        """
        return self.backbone_extractor.forward_sequential(x, attn_choice=attn_choice)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        attn_choice: AttentionOptions = "none",
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Standard ``nn.Module.forward`` entrypoint: runs guide-modulated TTA
        fusion if a guide has been bound, else returns raw TTA ViT features.
        """
        if self.xrd_fusion_module is not None:
            return self.forward_sequential(x, attn_choice=attn_choice, top_k=top_k)
        return self.forward_raw_features(x, attn_choice=attn_choice)

    @torch.no_grad()
    def forward_sequential(
        self,
        x: torch.Tensor,
        attn_choice: AttentionOptions = "none",
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Run TTA-ensembled feature extraction + guide fusion."""
        mean_feats, _ = self.forward_with_variance(
            x, attn_choice=attn_choice, top_k=top_k
        )
        return mean_feats

    @torch.no_grad()
    def forward_with_variance(
        self,
        x: torch.Tensor,
        attn_choice: AttentionOptions = "none",
        top_k: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run TTA-ensembled feature extraction + guide fusion while tracking
        online Welford spatial variance across transforms in ``O(1)`` extra memory.
        """
        assert self.xrd_fusion_module is not None, (
            "Call set_guide_transforms / set_xrd_transforms before forward."
        )
        self.eval()

        x_in = x.to(self.dtype) if self.dtype != torch.float32 else x.clone()
        img_batch = self.get_transformed_input_batch(x_in, self.transforms)
        _, img_h, img_w = x_in.shape

        welford = WelfordAccumulator()
        n_transforms = len(self.transforms)

        for i in range(n_transforms):
            transformed_img = img_batch[i].unsqueeze(0)
            permuted = self.extract_patch_grid(transformed_img, attn_choice=attn_choice)
            permuted = self.backbone_extractor.debiaser(permuted)
            fused_img = self.xrd_fusion_module.forward_xrd(
                permuted,
                i,
                self.xrd_fuse_method,
                top_k=top_k,
                hr_hw=(img_h, img_w),
            )
            full_size = self._upsample_patch_map(
                fused_img, (img_h, img_w), guidance_img=transformed_img
            )
            inverted = self.inverse_transforms[i](full_size)
            welford.update(inverted)

        mean_f32, var_f32 = welford.finalize()
        return mean_f32.to(self.dtype), var_f32

    # ── Porous-Media Void Clamping Helpers ──────────────────────────────────

    def _resolve_pore_mask_2d(
        self,
        hr_source: np.ndarray | torch.Tensor,
        padded_hw: tuple[int, int],
        orig_hw: tuple[int, int],
        spec_override: PoreMaskSpec = None,
        device: torch.device | None = None,
    ) -> torch.Tensor | None:
        eff_spec = spec_override if spec_override is not None else self.pore_mask_spec
        if eff_spec is None:
            return None
        dev = device if device is not None else next(self.dinov2.parameters()).device
        orig_h, orig_w = orig_hw
        pad_h, pad_w = padded_hw

        if isinstance(eff_spec, str):
            if eff_spec == "auto_otsu_fringe_guarded":
                alpha = (
                    self.pore_homogeneity_alpha_vox2
                    if self.pore_homogeneity_alpha_vox2 > 0.0
                    else 0.6
                )
            elif eff_spec == "auto_otsu":
                alpha = self.pore_homogeneity_alpha_vox2
            else:
                raise ValueError(
                    f"Unsupported pore_mask='{eff_spec}'; expected 'auto_otsu', "
                    "'auto_otsu_fringe_guarded', a boolean mask, or None."
                )
            hr_np = (
                hr_source.detach().cpu().float().numpy()
                if isinstance(hr_source, torch.Tensor)
                else np.asarray(hr_source, dtype=np.float32)
            )
            while hr_np.ndim > 3 and hr_np.shape[0] == 1:
                hr_np = hr_np[0]
            if hr_np.ndim == 3:
                if hr_np.shape[0] in (1, 3, 4) and tuple(hr_np.shape[1:]) == (
                    orig_h,
                    orig_w,
                ):
                    hr_2d = (
                        hr_np[1] if hr_np.shape[0] == 3 else np.mean(hr_np[:3], axis=0)
                    )
                elif hr_np.shape[-1] in (1, 3, 4):
                    hr_2d = np.mean(hr_np[..., :3], axis=-1)
                else:
                    hr_2d = hr_np[hr_np.shape[0] // 2]
            else:
                hr_2d = hr_np
            mask_np = segment_pores_otsu(
                hr_2d,
                tie_homogeneity_alpha_vox2=alpha,
                detrend_cupping=self.pore_detrend_cupping,
            )
        else:
            raw_m = (
                eff_spec.detach().cpu().numpy()
                if isinstance(eff_spec, torch.Tensor)
                else np.asarray(eff_spec)
            )
            while raw_m.ndim > 2 and raw_m.shape[0] == 1:
                raw_m = raw_m[0]
            if raw_m.ndim == 3 and raw_m.shape[-1] in (1, 3, 4):
                raw_m = np.mean(raw_m[..., :3], axis=-1)
            elif raw_m.ndim == 3 and raw_m.shape[0] in (3, 4):
                raw_m = np.mean(raw_m[:3], axis=0)
            if raw_m.ndim != 2:
                raise ValueError(
                    f"Expected 2D pore_mask for 2D fusion, got shape {raw_m.shape}."
                )
            mask_np = raw_m > 0.5

        m_t = torch.from_numpy(np.ascontiguousarray(mask_np, dtype=np.float32))[
            None, None, ...
        ].to(dev)
        if tuple(m_t.shape[-2:]) != (orig_h, orig_w):
            m_t = F.interpolate(m_t, size=(orig_h, orig_w), mode="nearest")
        if (pad_h, pad_w) != (orig_h, orig_w):
            m_t = F.pad(m_t, (0, pad_w - orig_w, 0, pad_h - orig_h), mode="replicate")
        return m_t > 0.5

    def _apply_pore_mask_to_logits_and_guide(
        self,
        logits: torch.Tensor,
        guide_lr: torch.Tensor,
        pore_mask_hr: torch.Tensor | None,
        valid_hr_hw: tuple[int, int] | None = None,
        allow_guide_porosity_cap: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pore_mask_hr is None:
            return logits, guide_lr
        pm = pore_mask_hr.to(device=logits.device, dtype=torch.bool)
        if pm.shape[-2:] != logits.shape[-2:]:
            pm = (
                F.interpolate(pm.float(), size=tuple(logits.shape[-2:]), mode="nearest")
                > 0.5
            )

        phys = self._active_physics()
        if phys is not None and phys.loss_fn == "compositional_kl":
            phys._force_bg_for_pores = True
            phys._resolved_bg_simplex = True

        clamped_logits = torch.where(pm, torch.full_like(logits, -30.0), logits)

        if (
            allow_guide_porosity_cap
            and self.spectral_channels is None
            and self.loss_fn != "forward_psf_poisson"
            and (self.guide_channels >= 2 or self.loss_fn == "compositional_kl")
        ):
            solid_hr = (~pm).float()
            if valid_hr_hw is not None:
                solid_hr = solid_hr[:, :, : valid_hr_hw[0], : valid_hr_hw[1]]
            lr_hw = (int(guide_lr.shape[-2]), int(guide_lr.shape[-1]))
            ds = self._active_downsampler()
            with torch.no_grad():
                if self.align_guide != "none":
                    solid_hr = self.guide_aligner(solid_hr, lr_hw).clamp(0.0, 1.0)
                if ds is not None:
                    f_solid_lr = ds(
                        solid_hr.expand(-1, ds.n_channels, -1, -1), lr_hw
                    ).mean(dim=1, keepdim=True)
                else:
                    f_solid_lr = F.adaptive_avg_pool2d(solid_hr, output_size=lr_hw)
                g_sum = guide_lr.sum(dim=1, keepdim=True)
                scale = torch.where(
                    g_sum > f_solid_lr + 1e-4,
                    (f_solid_lr / g_sum.clamp(min=1e-6)).clamp(0.0, 1.0),
                    torch.ones_like(g_sum),
                )
            guide_lr = guide_lr * scale
        return clamped_logits, guide_lr

    def _enforce_pore_probs(
        self,
        probs: torch.Tensor,
        pore_mask_hr: torch.Tensor | None,
        is_multiclass: bool,
    ) -> torch.Tensor:
        if pore_mask_hr is None:
            return probs
        pm = pore_mask_hr.to(device=probs.device, dtype=torch.bool)
        if pm.shape[-2:] != probs.shape[-2:]:
            pm = (
                F.interpolate(pm.float(), size=tuple(probs.shape[-2:]), mode="nearest")
                > 0.5
            )
        if is_multiclass:
            if probs.shape[1] == self.guide_channels + 1:
                bg = torch.where(pm, torch.ones_like(probs[:, :1]), probs[:, :1])
                fg = torch.where(pm, torch.zeros_like(probs[:, 1:]), probs[:, 1:])
                return torch.cat([bg, fg], dim=1)
            bg_new = pm.to(dtype=probs.dtype)
            fg_new = torch.where(pm, torch.zeros_like(probs), probs)
            return torch.cat([bg_new, fg_new], dim=1)
        return torch.where(pm, torch.zeros_like(probs), probs)

    # ── Optimization Step ────────────────────────────────────────────────────

    def train_fusion_step(
        self,
        x: torch.Tensor,
        update_aligner: bool = True,
        valid_hr_hw: tuple[int, int] | None = None,
        loss_mask_lr: torch.Tensor | None = None,
        pore_mask_hr: torch.Tensor | None = None,
    ) -> float:
        """Optimize the active fusion head (+ optional trainable cross-attention
        upsampler, ``DifferentiableGuideAligner``, learnable PSF, and spectral
        dictionary) across all TTA views in canonical HR coordinate space while
        keeping the ViT frozen.
        """
        assert self.xrd_fuse_method in (
            "learned_gating",
            "spatial_mlp",
            "attention",
        ), "train_fusion_step requires 'learned_gating', 'spatial_mlp', or 'attention'."
        assert (
            self.xrd_fusion_module is not None and self.fusion_optimizer is not None
        ), "Call set_guide_transforms / set_xrd_transforms before training."

        self.train()
        self.dinov2.eval()

        eff_pm = pore_mask_hr if pore_mask_hr is not None else self._active_pore_mask_hr
        _, img_h, img_w = x.shape
        want_hr = (
            self.loss_fn
            in (
                "forward_psf_bce",
                "forward_psf_mse",
                "forward_psf_poisson",
                "compositional_kl",
            )
            or self.xrd_fuse_method == "spatial_mlp"
            or self.align_guide != "none"
            or (self.learned_upsampler and self.interpolation_mode == "cross_attn")
            or self.spectral_channels is not None
            or valid_hr_hw is not None
            or loss_mask_lr is not None
            or eff_pm is not None
        )
        kind, cached_views = self._extract_tta_views(x, want_hr=want_hr)
        img_batch: torch.Tensor | None = None
        if want_hr and (kind == "patch" or self.xrd_fuse_method == "spatial_mlp"):
            x_in = x.to(self.dtype) if self.dtype != torch.float32 else x
            with torch.no_grad():
                img_batch = self.get_transformed_input_batch(x_in, self.transforms)

        n_transforms = len(self.transforms)
        total_loss = 0.0
        self.fusion_optimizer.zero_grad()
        if self.align_optimizer is not None:
            self.align_optimizer.zero_grad()

        use_align = self.align_guide != "none"
        aligner_arg = self.guide_aligner if use_align else None
        canon_guide = self.xrd_fusion_module.xrd_image
        if canon_guide.dim() == 2:
            canon_guide = canon_guide.unsqueeze(0).unsqueeze(0)
        elif canon_guide.dim() == 3:
            canon_guide = canon_guide.unsqueeze(0)
        lr_hw = (int(canon_guide.shape[-2]), int(canon_guide.shape[-1]))

        params: list[nn.Parameter] = []
        if self.learned_upsampler and self.interpolation_mode == "cross_attn":
            params.extend(
                list(self.backbone_extractor.cross_attn_upsampler.parameters())
            )

        tr_guides = (
            self.xrd_fusion_module.get_tr(hr_hw=(img_h, img_w))
            if self.xrd_fuse_method == "attention"
            else None
        )

        for i in range(n_transforms):
            tr_guidance_img = (
                img_batch[i].unsqueeze(0) if img_batch is not None else None
            )

            if self.xrd_fuse_method in ("learned_gating", "spatial_mlp"):
                head_mod: LinearProbeHead | SpatialConvMLPHead = (
                    self.learned_gating  # type: ignore[assignment]
                    if self.xrd_fuse_method == "learned_gating"
                    else self.spatial_mlp
                )
                assert head_mod is not None
                if want_hr:
                    if kind == "hr":
                        feat_in = cached_views[i]
                    else:
                        assert tr_guidance_img is not None
                        feat_in = self._upsample_patch_map(
                            cached_views[i].float(),
                            (img_h, img_w),
                            guidance_img=tr_guidance_img,
                        )
                else:
                    feat_in = cached_views[i]

                logits_tr = head_mod.predict_logits(
                    feat_in, guidance_hr=tr_guidance_img
                )
                if logits_tr.shape[-2:] != (img_h, img_w):
                    logits_tr = F.interpolate(
                        logits_tr,
                        size=(img_h, img_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                logits_canon = self.inverse_transforms[i](logits_tr)
                if isinstance(head_mod, SpatialConvMLPHead):
                    logits_canon = head_mod.apply_guide_residual(
                        logits_canon, canon_guide, valid_hr_hw=valid_hr_hw
                    )
                    logits_canon, target_guide = (
                        self._apply_pore_mask_to_logits_and_guide(
                            logits_canon,
                            canon_guide,
                            eff_pm,
                            valid_hr_hw=valid_hr_hw,
                        )
                    )
                    loss = head_mod.compute_loss_from_logits(
                        logits_canon,
                        xrd_map=target_guide,
                        coarse_target=target_guide,
                        aligner=aligner_arg,
                        guidance_hr=x.unsqueeze(0),
                        valid_hr_hw=valid_hr_hw,
                        loss_mask_lr=loss_mask_lr,
                    )
                else:
                    logits_canon, target_guide = (
                        self._apply_pore_mask_to_logits_and_guide(
                            logits_canon,
                            canon_guide,
                            eff_pm,
                            valid_hr_hw=valid_hr_hw,
                        )
                    )
                    loss = head_mod.compute_loss_from_logits(
                        logits_canon,
                        xrd_map=target_guide,
                        coarse_target=target_guide,
                        aligner=aligner_arg,
                        valid_hr_hw=valid_hr_hw,
                        loss_mask_lr=loss_mask_lr,
                    )
                if i == 0:
                    params.extend(list(head_mod.parameters()))
            else:
                assert self.spatial_attention is not None and tr_guides is not None
                feat_in = (
                    self._upsample_patch_map(
                        cached_views[i].float(),
                        (img_h, img_w),
                        guidance_img=tr_guidance_img,
                    )
                    if (want_hr and kind == "patch" and tr_guidance_img is not None)
                    else cached_views[i]
                )
                if valid_hr_hw is not None and valid_hr_hw != (img_h, img_w):
                    oh, ow = valid_hr_hw
                    if feat_in.shape[-2:] != (img_h, img_w):
                        feat_in = F.interpolate(
                            feat_in.float(),
                            size=(img_h, img_w),
                            mode="bilinear",
                            align_corners=False,
                        )
                    feat_canon = self.inverse_transforms[i](feat_in)[:, :, :oh, :ow]
                    logits_canon = self.spatial_attention.predict_logits(
                        feat_canon, canon_guide
                    )
                    pm_crop = eff_pm[:, :, :oh, :ow] if eff_pm is not None else None
                    logits_canon, target_guide = (
                        self._apply_pore_mask_to_logits_and_guide(
                            logits_canon,
                            canon_guide,
                            pm_crop,
                            valid_hr_hw=None,
                        )
                    )
                    loss = self.spatial_attention.compute_loss_from_logits(
                        logits_canon,
                        guide_map=target_guide,
                        coarse_target=target_guide,
                        aligner=aligner_arg,
                        valid_hr_hw=None,
                        loss_mask_lr=loss_mask_lr,
                    )
                else:
                    tr_guide_coarse = tr_guides[i]
                    if tr_guide_coarse.dim() == 3:
                        tr_guide_coarse = tr_guide_coarse.unsqueeze(0)
                    logits_tr = self.spatial_attention.predict_logits(
                        feat_in, tr_guide_coarse
                    )
                    if logits_tr.shape[-2:] != (img_h, img_w):
                        logits_tr = F.interpolate(
                            logits_tr,
                            size=(img_h, img_w),
                            mode="bilinear",
                            align_corners=False,
                        )
                    logits_canon = self.inverse_transforms[i](logits_tr)
                    logits_canon, target_guide = (
                        self._apply_pore_mask_to_logits_and_guide(
                            logits_canon,
                            canon_guide,
                            eff_pm,
                            valid_hr_hw=valid_hr_hw,
                        )
                    )
                    loss = self.spatial_attention.compute_loss_from_logits(
                        logits_canon,
                        guide_map=target_guide,
                        coarse_target=target_guide,
                        aligner=aligner_arg,
                        valid_hr_hw=valid_hr_hw,
                        loss_mask_lr=loss_mask_lr,
                    )
                if i == 0:
                    params.extend(list(self.spatial_attention.parameters()))

            if use_align and update_aligner:
                eff_hr_hw = valid_hr_hw if valid_hr_hw is not None else (img_h, img_w)
                loss = loss + self.guide_aligner.regularization_loss(
                    lr_hw=lr_hw, device=loss.device, hr_hw=eff_hr_hw
                )

            scaled_loss = loss / max(n_transforms, 1)
            scaled_loss.backward()
            total_loss += float(loss.detach().cpu().item())

        if params:
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        self.fusion_optimizer.step()
        if use_align and update_aligner and self.align_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.guide_aligner.parameters()), max_norm=1.0
            )
            self.align_optimizer.step()
        return total_loss / max(n_transforms, 1)

    # ── Diagnostics & Calibrated Inference ───────────────────────────────────

    def _build_diagnostics(
        self, hr_hw: tuple[int, int], lr_hw: tuple[int, int]
    ) -> dict[str, Any]:
        up_fac = self.backbone_extractor.small_fov_upscale_factor(hr_hw[0], hr_hw[1])
        diag: dict[str, Any] = {
            "backbone": self.backbone_name,
            "fusion_method": self.xrd_fuse_method,
            "layers": (
                list(self.layer_indices) if self.layer_indices is not None else "last"
            ),
            "feat_dim": self.feat_dim,
            "interpolation_mode": self.interpolation_mode,
            "debias_positional": self.debias_positional,
        }
        if up_fac > 1:
            diag["small_fov_upscale_factor"] = up_fac
        if self._active_pore_mask_hr is not None:
            pm_crop = self._active_pore_mask_hr[:, :, : hr_hw[0], : hr_hw[1]]
            diag["pore_mask_active"] = True
            diag["porosity_fraction"] = round(float(pm_crop.float().mean().item()), 4)
        diag.update(self.guide_aligner.diagnostics(hr_hw=hr_hw, lr_hw=lr_hw))
        phys = self._active_physics()
        if phys is not None:
            diag.update(phys.diagnostics(hr_hw=hr_hw, lr_hw=lr_hw))
        if self.calibrator is not None and self.calibrator.q_label is not None:
            diag["conformal_alpha"] = self.calibrator.alpha
            diag["conformal_q_label"] = round(float(self.calibrator.q_label), 4)
        if self._last_loss_history:
            diag["initial_loss"] = round(self._last_loss_history[0], 5)
            diag["final_loss"] = round(self._last_loss_history[-1], 5)
        return diag

    def _logits_from_hr_view(
        self,
        hr_feats: torch.Tensor,
        transformed_img: torch.Tensor | None,
        tr_guide: torch.Tensor,
        patch_debiased: torch.Tensor | None = None,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if self.xrd_fuse_method == "learned_gating" and self.learned_gating:
            return self.learned_gating.predict_logits(hr_feats)
        if self.xrd_fuse_method == "spatial_mlp" and self.spatial_mlp:
            return self.spatial_mlp.predict_logits(
                hr_feats, guidance_hr=transformed_img
            )
        if self.xrd_fuse_method == "attention" and self.spatial_attention:
            return self.spatial_attention.predict_logits(hr_feats, tr_guide)
        gate_src = patch_debiased if patch_debiased is not None else hr_feats
        gates_mc = compute_channel_correlation_gates(
            gate_src.float(),
            tr_guide,
            loss_fn=self.loss_fn,
            top_k=top_k,
        ).float()
        feat_std = (hr_feats - hr_feats.mean(dim=(-2, -1), keepdim=True)) / (
            hr_feats.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        )
        w_norm = gates_mc / gates_mc.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return torch.einsum("mc,bchw->bmhw", w_norm, feat_std)

    @torch.no_grad()
    def _predict_logits_and_var_tile(
        self,
        x_tile: torch.Tensor,
        guide_lr: torch.Tensor,
        top_k: int | None = None,
        *,
        _allow_cache: bool = True,
        _override_tr_guide: torch.Tensor | None = None,
        valid_hr_hw: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        _, img_h, img_w = x_tile.shape
        n_transforms = len(self.transforms)
        canon_unpad = (
            valid_hr_hw is not None
            and valid_hr_hw != (img_h, img_w)
            and self.xrd_fuse_method in ("attention", "gating")
        )

        logit_welford = WelfordAccumulator()
        prob_welford = WelfordAccumulator()

        use_cached = (
            _allow_cache
            and self.cache_features
            and self.xrd_fuse_method != "gating"
            and (img_h * img_w <= 512 * 512)
            and not (self.tile_size is not None and max(img_h, img_w) > self.tile_size)
        )
        cached_kind = ""
        cached_views: list[torch.Tensor] = []
        if use_cached:
            cached_kind, cached_views = self._extract_tta_views(x_tile, want_hr=True)

        need_img_batch = (
            not use_cached
            or cached_kind != "hr"
            or self.xrd_fuse_method == "spatial_mlp"
        )
        img_batch: torch.Tensor | None = None
        if need_img_batch:
            x_in = (
                x_tile.to(self.dtype) if self.dtype != torch.float32 else x_tile.clone()
            )
            img_batch = self.get_transformed_input_batch(x_in, self.transforms)

        tr_guides = (
            self.xrd_fusion_module.get_tr(hr_hw=(img_h, img_w))
            if (
                not canon_unpad
                and _override_tr_guide is None
                and self.xrd_fusion_module is not None
            )
            else None
        )

        for i in range(n_transforms):
            transformed_img = (
                img_batch[i].unsqueeze(0) if img_batch is not None else None
            )
            patch_debiased: torch.Tensor | None = None
            if use_cached and cached_kind == "hr":
                hr_feats = cached_views[i]
            elif use_cached and cached_kind == "patch":
                assert transformed_img is not None
                patch_debiased = cached_views[i].float()
                hr_feats = self._upsample_patch_map(
                    patch_debiased, (img_h, img_w), guidance_img=transformed_img
                )
            else:
                assert transformed_img is not None
                permuted = self.extract_patch_grid(transformed_img, attn_choice="none")
                patch_debiased = (
                    self.backbone_extractor.debiaser(permuted.float())
                    if _allow_cache
                    else permuted.float()
                )
                hr_feats = self._upsample_patch_map(
                    patch_debiased, (img_h, img_w), guidance_img=transformed_img
                )

            if canon_unpad and valid_hr_hw is not None:
                oh, ow = valid_hr_hw
                feat_canon = self.inverse_transforms[i](hr_feats)[:, :, :oh, :ow]
                canon_g = (
                    _override_tr_guide if _override_tr_guide is not None else guide_lr
                )
                logits_valid = self._logits_from_hr_view(
                    feat_canon,
                    transformed_img=None,
                    tr_guide=canon_g,
                    patch_debiased=None,
                    top_k=top_k,
                )
                inv_logits = F.pad(
                    logits_valid, (0, img_w - ow, 0, img_h - oh), mode="replicate"
                )
            else:
                tr_g = (
                    _override_tr_guide
                    if _override_tr_guide is not None
                    else (tr_guides[i] if tr_guides is not None else guide_lr)
                )
                logits_view = self._logits_from_hr_view(
                    hr_feats,
                    transformed_img=transformed_img,
                    tr_guide=tr_g,
                    patch_debiased=patch_debiased,
                    top_k=top_k,
                )
                inv_logits = self.inverse_transforms[i](logits_view)
            logit_welford.update(inv_logits)
            prob_welford.update(torch.sigmoid(inv_logits))

        mean_logits, _ = logit_welford.finalize()
        _, epistemic_prob_var = prob_welford.finalize()
        return mean_logits, epistemic_prob_var

    @torch.no_grad()
    def _predict_logits_and_var_maybe_tiled(
        self,
        x: torch.Tensor,
        guide_lr: torch.Tensor,
        top_k: int | None = None,
        *,
        override_tr_guide: torch.Tensor | None = None,
        valid_hr_hw: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        _, img_h, img_w = x.shape
        if self.tile_size is None or max(img_h, img_w) <= self.tile_size:
            return self._predict_logits_and_var_tile(
                x,
                guide_lr,
                top_k=top_k,
                _allow_cache=True,
                _override_tr_guide=override_tr_guide,
                valid_hr_hw=valid_hr_hw,
            )

        # Seam-free full-FOV debiasing + TTA Welford loop ('attention' & 'gating' safe)
        if (
            img_h * img_w <= 1024 * 1024
            or self.debias_positional
            or self.xrd_fuse_method in ("attention", "gating")
        ):
            canon_unpad = (
                valid_hr_hw is not None
                and valid_hr_hw != (img_h, img_w)
                and self.xrd_fuse_method in ("attention", "gating")
            )
            cached_views: list[torch.Tensor] | None = None
            if self.cache_features and (img_h * img_w <= 512 * 512):
                _, cached_views = self._extract_tta_views(x, want_hr=True)

            x_in = x.to(self.dtype) if self.dtype != torch.float32 else x.clone()
            tr_guides = (
                self.xrd_fusion_module.get_tr(hr_hw=(img_h, img_w))
                if (
                    not canon_unpad
                    and override_tr_guide is None
                    and self.xrd_fusion_module is not None
                )
                else None
            )
            logit_welford = WelfordAccumulator()
            prob_welford = WelfordAccumulator()

            for i in range(len(self.transforms)):
                tr_img = self.transforms[i](x_in)
                if cached_views is not None:
                    hr_feats = cached_views[i]
                else:

                    def _single_tr_tile(tile_chw: torch.Tensor) -> torch.Tensor:
                        t_in = tile_chw.unsqueeze(0)
                        _, th, tw = tile_chw.shape
                        perm = self.extract_patch_grid(t_in, attn_choice="none")
                        return self._upsample_patch_map(
                            perm.float(), (th, tw), guidance_img=t_in
                        )

                    hr_v = run_tiled_2d(
                        _single_tr_tile,
                        tr_img,
                        tile_size=self.tile_size,
                        overlap=self.tile_overlap,
                        patch_size=self.original_patch_size,
                        stride=self.stride[0],
                    )
                    hr_feats = self.backbone_extractor.debiaser(hr_v)

                if canon_unpad and valid_hr_hw is not None:
                    oh, ow = valid_hr_hw
                    feat_canon = self.inverse_transforms[i](hr_feats)[:, :, :oh, :ow]
                    canon_g = (
                        override_tr_guide if override_tr_guide is not None else guide_lr
                    )
                    logits_valid = self._logits_from_hr_view(
                        feat_canon,
                        transformed_img=None,
                        tr_guide=canon_g,
                        top_k=top_k,
                    )
                    inv_logits = F.pad(
                        logits_valid, (0, img_w - ow, 0, img_h - oh), mode="replicate"
                    )
                else:
                    tr_g = (
                        override_tr_guide
                        if override_tr_guide is not None
                        else (tr_guides[i] if tr_guides is not None else guide_lr)
                    )
                    logits_view = self._logits_from_hr_view(
                        hr_feats,
                        transformed_img=tr_img.unsqueeze(0),
                        tr_guide=tr_g,
                        top_k=top_k,
                    )
                    inv_logits = self.inverse_transforms[i](logits_view)
                logit_welford.update(inv_logits)
                prob_welford.update(torch.sigmoid(inv_logits))

            mean_logits, _ = logit_welford.finalize()
            _, epistemic_prob_var = prob_welford.finalize()
            return mean_logits, epistemic_prob_var

        n_ch = self.guide_channels

        def _tile_pred(tile_chw: torch.Tensor) -> torch.Tensor:
            ml, vp = self._predict_logits_and_var_tile(
                tile_chw,
                guide_lr,
                top_k=top_k,
                _allow_cache=False,
                _override_tr_guide=override_tr_guide,
            )
            return torch.cat([ml, vp], dim=1)

        packed = run_tiled_2d(
            _tile_pred,
            x,
            tile_size=self.tile_size,
            overlap=self.tile_overlap,
            patch_size=self.original_patch_size,
            stride=self.stride[0],
        )
        return packed[:, :n_ch], packed[:, n_ch:]

    def _effective_reference_guide(self, guide_lr: torch.Tensor) -> torch.Tensor:
        """For spectral-cube models, return component-level coarse reference maps,
        projecting out-of-sample spectral cubes onto the learned basis on the fly.
        """
        g = guide_lr
        if g.ndim == 2:
            g = g.unsqueeze(0).unsqueeze(0)
        elif g.ndim == 3:
            g = g.unsqueeze(0)

        if (
            self.spectral_channels is not None
            and g.shape[1] == self.spectral_channels
            and g.shape[1] != self.guide_channels
        ):
            if (
                self._unmixed_guide_lr is not None
                and self._fit_guide_ptr is not None
                and int(guide_lr.data_ptr()) == self._fit_guide_ptr
                and self._unmixed_guide_lr.shape[-2:] == g.shape[-2:]
            ):
                abund = self._unmixed_guide_lr
            else:
                phys = self._active_physics()
                if phys is not None and phys.spectral_dict is not None:
                    abund = phys.spectral_dict.project_cube(g)
                elif self._unmixed_guide_lr is not None:
                    abund = self._unmixed_guide_lr
                else:
                    return g[:, : self.guide_channels]
            ref_norm: GuideNormalization = (
                "global_max" if self.loss_fn == "compositional_kl" else "minmax"
            )
            return guide_to_tensor(
                abund,
                device=guide_lr.device,
                normalization=ref_norm,
            )
        return g

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        guide_lr: torch.Tensor | None = None,
        mode: Literal["multiclass", "multilabel"] = "multilabel",
        calibrate_to_guide: bool = True,
        abstain_threshold: float | None = None,
        top_k: int | None = None,
        *,
        unguided_transfer: bool = False,
        use_aligner: bool = True,
        valid_hr_hw: tuple[int, int] | None = None,
        pore_mask_hr: torch.Tensor | None = None,
    ) -> UncertaintyReport:
        """Produce calibrated high-resolution phase/element probabilities, tri-part
        pixel certainty, and optional Split Conformal ``prediction_sets``.
        """
        self.eval()
        _, img_h, img_w = x.shape
        eff_pm = pore_mask_hr if pore_mask_hr is not None else self._active_pore_mask_hr

        if unguided_transfer:
            assert self.xrd_fusion_module is not None, (
                "Call fit() or set_guide_transforms() before unguided predict()."
            )
            train_ref = self._effective_reference_guide(
                self.xrd_fusion_module.xrd_image
            )
            prior_query = train_ref.mean(dim=(-2, -1), keepdim=True).expand_as(
                train_ref
            )
            mean_logits, epistemic_prob_var = self._predict_logits_and_var_maybe_tiled(
                x,
                prior_query,
                top_k=top_k,
                override_tr_guide=prior_query,
                valid_hr_hw=valid_hr_hw,
            )
            if eff_pm is not None:
                mean_logits, _ = self._apply_pore_mask_to_logits_and_guide(
                    mean_logits,
                    prior_query,
                    eff_pm,
                    valid_hr_hw=valid_hr_hw,
                    allow_guide_porosity_cap=False,
                )
            phys = self._active_physics()
            if self.loss_fn == "compositional_kl" and phys is not None:
                probs = phys.logits_to_simplex_probs(mean_logits, target=None)
            elif mode == "multiclass":
                bg_logit = torch.zeros(
                    (mean_logits.shape[0], 1, img_h, img_w),
                    device=mean_logits.device,
                    dtype=mean_logits.dtype,
                )
                probs = F.softmax(torch.cat([bg_logit, mean_logits], dim=1), dim=1)
            else:
                probs = torch.sigmoid(mean_logits)

            eff_mode: Literal["multiclass", "multilabel"] = (
                "multiclass" if probs.shape[1] >= 2 and mode == "multiclass" else mode
            )
            probs = self._enforce_pore_probs(
                probs, eff_pm, is_multiclass=(eff_mode == "multiclass")
            )
            entropy, margin = compute_posterior_uncertainty(probs, mode=eff_mode)
            epi = (
                (
                    epistemic_prob_var.mean(dim=1)
                    if epistemic_prob_var.ndim == 4
                    else epistemic_prob_var
                )
                .float()
                .to(entropy.device)
            )
            pm_3d: torch.Tensor | None = None
            if eff_pm is not None:
                pm_4d = eff_pm.to(device=entropy.device, dtype=torch.bool)
                if pm_4d.shape[-2:] != entropy.shape[-2:]:
                    pm_4d = (
                        F.interpolate(
                            pm_4d.float(),
                            size=tuple(entropy.shape[-2:]),
                            mode="nearest",
                        )
                        > 0.5
                    )
                pm_3d = pm_4d.squeeze(1) if pm_4d.ndim == 4 else pm_4d
                epi = torch.where(pm_3d, torch.zeros_like(epi), epi)
            epi_ref = epi[~pm_3d] if (pm_3d is not None and (~pm_3d).any()) else epi
            p99 = _safe_quantile(epi_ref, 0.99).clamp(min=1e-6)
            epi_norm = (epi / p99).clamp(0.0, 1.0)
            phys_res = torch.zeros_like(entropy)
            combined = (0.65 * entropy + 0.35 * epi_norm).clamp(0.0, 1.0)
            if pm_3d is not None:
                combined = torch.where(pm_3d, torch.zeros_like(combined), combined)
            unresolved = torch.zeros_like(entropy, dtype=torch.bool)
            hard = compute_hard_phase_mask(probs, mode=eff_mode, channel_dim=1)
            if abstain_threshold is not None:
                abstain_cond = combined > abstain_threshold
                if pm_3d is not None:
                    abstain_cond = abstain_cond & (~pm_3d)
                hard = torch.where(abstain_cond, torch.full_like(hard, -1), hard)
            pred_sets: torch.Tensor | None = None
            conf_hw: torch.Tensor | None = None
            if self.calibrator is not None:
                pred_sets, conf_hw = self.calibrator.predict_sets_and_intervals(
                    probs_hr=probs,
                    guide_lr=None,
                    epistemic_var_hr=epi,
                    mode=eff_mode,
                    pore_mask_hr=eff_pm,
                )
            lr_hw = (int(train_ref.shape[-2]), int(train_ref.shape[-1]))
            diag = self._build_diagnostics((img_h, img_w), lr_hw)
            diag["unguided_transfer"] = True
            return UncertaintyReport(
                probabilities=probs,
                hard_mask=hard,
                aleatoric_entropy=entropy,
                confidence_margin=margin,
                epistemic_variance=epi,
                physical_residual=phys_res,
                combined_uncertainty=combined,
                unresolved_mask=unresolved,
                diagnostics=diag,
                prediction_sets=pred_sets,
                conformal_interval_halfwidth=conf_hw,
            )

        if guide_lr is None:
            assert self.xrd_fusion_module is not None, (
                "Provide guide_lr or call set_guide_transforms first."
            )
            guide_lr = self.xrd_fusion_module.xrd_image
        if guide_lr.ndim == 2:
            guide_lr = guide_lr.unsqueeze(0).unsqueeze(0)
        elif guide_lr.ndim == 3:
            guide_lr = guide_lr.unsqueeze(0)

        ref_guide = self._effective_reference_guide(guide_lr)
        mean_logits, epistemic_prob_var = self._predict_logits_and_var_maybe_tiled(
            x, guide_lr, top_k=top_k, valid_hr_hw=valid_hr_hw
        )
        if self.xrd_fuse_method == "spatial_mlp" and self.spatial_mlp is not None:
            mean_logits = self.spatial_mlp.apply_guide_residual(
                mean_logits, guide_lr, valid_hr_hw=valid_hr_hw
            )

        mean_logits, ref_guide_eff = self._apply_pore_mask_to_logits_and_guide(
            mean_logits,
            ref_guide,
            eff_pm,
            valid_hr_hw=valid_hr_hw,
            allow_guide_porosity_cap=(
                mode == "multiclass" or self.loss_fn == "compositional_kl"
            ),
        )

        phys = self._active_physics()
        if self.loss_fn == "compositional_kl" and phys is not None:
            probs = phys.logits_to_simplex_probs(mean_logits, target=ref_guide_eff)
            if calibrate_to_guide and probs.shape[1] == ref_guide_eff.shape[1] + 1:
                bg_log = torch.zeros_like(mean_logits[:, :1])
                probs = match_coarse_priors(
                    torch.cat([bg_log, mean_logits], dim=1),
                    ref_guide_eff,
                    mode="multiclass",
                )
        elif mode == "multiclass":
            bg_logit = torch.zeros(
                (mean_logits.shape[0], 1, img_h, img_w),
                device=mean_logits.device,
                dtype=mean_logits.dtype,
            )
            full_logits = torch.cat([bg_logit, mean_logits], dim=1)
            probs = (
                match_coarse_priors(full_logits, ref_guide_eff, mode="multiclass")
                if calibrate_to_guide
                else F.softmax(full_logits, dim=1)
            )
        else:
            probs = (
                match_coarse_priors(mean_logits, ref_guide_eff, mode="multilabel")
                if calibrate_to_guide
                else torch.sigmoid(mean_logits)
            )

        out_mode: Literal["multiclass", "multilabel"] = (
            "multiclass" if probs.shape[1] >= 2 and mode == "multiclass" else mode
        )
        probs = self._enforce_pore_probs(
            probs, eff_pm, is_multiclass=(out_mode == "multiclass")
        )

        lr_hw = (int(ref_guide_eff.shape[-2]), int(ref_guide_eff.shape[-1]))
        return build_uncertainty_report(
            probs_hr=probs,
            guide_lr=ref_guide_eff,
            epistemic_var=epistemic_prob_var,
            mode=out_mode,
            abstain_threshold=abstain_threshold,
            downsampler=self._active_downsampler(),
            aligner=(self.guide_aligner if use_aligner else None),
            calibrator=self.calibrator,
            diagnostics=self._build_diagnostics((img_h, img_w), lr_hw),
            pore_mask_hr=eff_pm,
        )

    # ── High-Level 3-Line Pythonic API ───────────────────────────────────────

    def _default_guide_norm(
        self, normalization: GuideNormalization | None
    ) -> GuideNormalization:
        if normalization is not None:
            return normalization
        if self.loss_fn == "forward_psf_poisson":
            return "preserve_counts"
        if self.loss_fn == "compositional_kl":
            return "global_max"
        return "minmax"

    def _prepare_high_level_hr(
        self, hr_image: np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        dev = next(self.dinov2.parameters()).device
        if (
            isinstance(hr_image, torch.Tensor)
            and hr_image.ndim == 3
            and hr_image.shape[0] == 3
        ):
            hr_t = hr_image.to(dev)
        else:
            hr_t = normalize_structural(hr_image, device=dev)
        return ensure_spatial_divisible(
            hr_t, patch_size=self.original_patch_size, stride=self.stride[0]
        )

    @staticmethod
    def _inpaint_holdout_coarse(
        guide_t: torch.Tensor, holdout_mask_lr: torch.Tensor
    ) -> torch.Tensor:
        """Replace held-out coarse pixels with smooth non-held-out neighbor averages
        so neither NMF initialization nor attention queries see calibration targets.
        """
        g = guide_t.clone()
        if g.ndim == 3:
            g = g.unsqueeze(0)
        valid = (
            (~holdout_mask_lr)
            .to(device=g.device, dtype=g.dtype)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        if float(valid.sum().item()) <= 0.0:
            return guide_t
        ch_mean = (g * valid).sum(dim=(-2, -1), keepdim=True) / valid.sum(
            dim=(-2, -1), keepdim=True
        ).clamp(min=1.0)
        cur = torch.where(valid > 0.5, g, ch_mean)
        kernel = torch.ones((1, 1, 3, 3), device=g.device, dtype=g.dtype)
        for _ in range(2):
            v_cnt = F.conv2d(valid, kernel, padding=1)
            g_sum = F.conv2d(
                cur * valid,
                kernel.expand(cur.shape[1], 1, 3, 3),
                padding=1,
                groups=cur.shape[1],
            )
            interp = torch.where(v_cnt > 0.0, g_sum / v_cnt.clamp(min=1.0), ch_mean)
            cur = torch.where(valid > 0.5, g, interp)
        return cur.squeeze(0) if guide_t.ndim == 3 else cur

    def fit(
        self,
        hr_image: np.ndarray | torch.Tensor,
        coarse_guide: np.ndarray | torch.Tensor,
        epochs: int = 25,
        tta_shifts: Sequence[int] = (1,),
        tta_flips: bool = True,
        warmup_fraction: float = 0.30,
        guide_normalization: GuideNormalization | None = None,
        n_components: int | None = None,
        spectral_unmix: bool = False,
        *,
        pore_mask: PoreMaskSpec = None,
        max_steps: int | None = None,
        tol: float = 1e-6,
        reset_weights: bool = True,
    ) -> list[float]:
        """High-level training entrypoint: normalizes and pads ``hr_image`` and
        ``coarse_guide`` (or raw 3D spectral cube when ``spectral_unmix=True``),
        caches frozen ViT TTA features, optimizes the fusion head (+ optional
        co-registration, PSF width, & spectral dictionary), and calibrates
        out-of-sample Split Conformal prediction sets when ``conformal_alpha`` is set.
        """
        dev = next(self.dinov2.parameters()).device
        hr_prepared, (orig_h, orig_w) = self._prepare_high_level_hr(hr_image)
        pad_h, pad_w = int(hr_prepared.shape[-2]), int(hr_prepared.shape[-1])
        self._fit_hr_sig = _tensor_content_digest(hr_prepared[:, :orig_h, :orig_w])
        self._active_pore_mask_hr = self._resolve_pore_mask_2d(
            hr_image,
            padded_hw=(pad_h, pad_w),
            orig_hw=(orig_h, orig_w),
            spec_override=pore_mask,
            device=dev,
        )

        norm_mode = (
            "preserve_counts"
            if spectral_unmix and self.loss_fn == "forward_psf_poisson"
            else (
                "global_max"
                if spectral_unmix
                else self._default_guide_norm(guide_normalization)
            )
        )
        guide_t = guide_to_tensor(coarse_guide, device=dev, normalization=norm_mode)
        lr_hw = (int(guide_t.shape[-2]), int(guide_t.shape[-1]))

        if reset_weights:
            if spectral_unmix:
                self.guide_channels = int(n_components or self.guide_channels)
                self.spectral_channels = int(guide_t.shape[1])
            else:
                self.guide_channels = int(guide_t.shape[1])
                self.spectral_channels = None
            self.reset_trainable_weights()
        else:
            self.guide_aligner.reset_parameters()
            phys_pre = self._active_physics()
            if phys_pre is not None:
                phys_pre.reset_specimen_state()

        cal_mask_lr: torch.Tensor | None = None
        train_mask_lr: torch.Tensor | None = None
        train_guide_t = guide_t
        if self.calibrator is not None and (lr_hw[0] * lr_hw[1] >= 16):
            cal_mask_lr = ConformalCalibrator.make_spatial_holdout_mask(
                lr_hw, holdout_fraction=0.25
            ).to(dev)
            dilated = (
                F.max_pool2d(
                    cal_mask_lr.float().unsqueeze(0).unsqueeze(0),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
                .squeeze(0)
                .squeeze(0)
                > 0.5
            )
            exclude_lr = dilated if int((~dilated).sum().item()) >= 8 else cal_mask_lr
            train_mask_lr = ~exclude_lr
            train_guide_t = self._inpaint_holdout_coarse(guide_t, cal_mask_lr)

        fwd, inv = build_default_tta(
            shift_dists=list(tta_shifts),
            pattern="Neumann",
            include_flips=tta_flips,
        )
        self.set_guide_transforms(
            train_guide_t,
            fwd,
            inv,
            n_components=n_components,
            spectral_unmix=spectral_unmix,
        )
        self.guide_aligner.reset_parameters()

        if self.xrd_fuse_method not in ("learned_gating", "spatial_mlp", "attention"):
            self._bind_guide_only(guide_t)
            self._last_loss_history = []
            return []

        valid_hw = (orig_h, orig_w) if (orig_h, orig_w) != (pad_h, pad_w) else None
        n_epochs = max(1, int(max_steps if max_steps is not None else epochs))
        warmup_steps = (
            int(round(warmup_fraction * n_epochs)) if self.align_guide != "none" else 0
        )
        losses: list[float] = []
        for ep in range(n_epochs):
            loss_val = self.train_fusion_step(
                hr_prepared,
                update_aligner=(ep >= warmup_steps),
                valid_hr_hw=valid_hw,
                loss_mask_lr=train_mask_lr,
                pore_mask_hr=self._active_pore_mask_hr,
            )
            prev_loss = losses[-1] if losses else None
            losses.append(loss_val)
            if (
                tol > 0.0
                and prev_loss is not None
                and ep >= max(warmup_steps + 3, 5)
                and abs(prev_loss - loss_val) < tol
            ):
                break
        self._last_loss_history = losses

        # Bind full un-inpainted guide after training for holdout calibration
        if cal_mask_lr is not None:
            self._bind_guide_only(guide_t)
            phys_post = self._active_physics()
            if (
                spectral_unmix
                and phys_post is not None
                and phys_post.spectral_dict is not None
            ):
                g4 = guide_t.unsqueeze(0) if guide_t.ndim == 3 else guide_t
                self._unmixed_guide_lr = phys_post.spectral_dict.project_cube(g4)
                self._fit_guide_ptr = int(self.xrd_fusion_module.xrd_image.data_ptr())  # type: ignore[union-attr]

        if self.calibrator is not None:
            assert self.xrd_fusion_module is not None
            bound_g = self.xrd_fusion_module.xrd_image
            ref_g = self._effective_reference_guide(bound_g)
            if cal_mask_lr is None:
                cal_mask_lr = ConformalCalibrator.make_spatial_holdout_mask(
                    lr_hw, holdout_fraction=0.25
                ).to(dev)
            mean_logits, epi_var = self._predict_logits_and_var_maybe_tiled(
                hr_prepared, bound_g, valid_hr_hw=valid_hw
            )
            if self.xrd_fuse_method == "spatial_mlp" and self.spatial_mlp is not None:
                mean_logits = self.spatial_mlp.apply_guide_residual(
                    mean_logits, bound_g, valid_hr_hw=valid_hw
                )
            mean_logits = mean_logits[:, :, :orig_h, :orig_w]
            epi_var = epi_var[:, :, :orig_h, :orig_w]
            pm_cal = (
                self._active_pore_mask_hr[:, :, :orig_h, :orig_w]
                if self._active_pore_mask_hr is not None
                else None
            )
            n_g = int(ref_g.shape[1])
            phys_cal = self._active_physics()
            cal_mode: Literal["multiclass", "multilabel"] = (
                "multiclass"
                if n_g >= 2 or self.loss_fn == "compositional_kl"
                else "multilabel"
            )
            mean_logits, ref_g_eff = self._apply_pore_mask_to_logits_and_guide(
                mean_logits,
                ref_g,
                pm_cal,
                valid_hr_hw=None,
                allow_guide_porosity_cap=(cal_mode == "multiclass"),
            )
            if self.loss_fn == "compositional_kl" and phys_cal is not None:
                probs_cal = phys_cal.logits_to_simplex_probs(
                    mean_logits, target=ref_g_eff
                )
                if probs_cal.shape[1] == n_g + 1:
                    bg_l = torch.zeros_like(mean_logits[:, :1])
                    probs_cal = match_coarse_priors(
                        torch.cat([bg_l, mean_logits], dim=1),
                        ref_g_eff,
                        mode="multiclass",
                    )
            elif cal_mode == "multiclass":
                bg_l = torch.zeros_like(mean_logits[:, :1])
                probs_cal = match_coarse_priors(
                    torch.cat([bg_l, mean_logits], dim=1),
                    ref_g_eff,
                    mode="multiclass",
                )
            else:
                probs_cal = match_coarse_priors(
                    mean_logits, ref_g_eff, mode="multilabel"
                )
            probs_cal = self._enforce_pore_probs(
                probs_cal, pm_cal, is_multiclass=(cal_mode == "multiclass")
            )
            self.calibrator.calibrate_coarse(
                probs_hr=probs_cal,
                guide_lr=ref_g_eff,
                epistemic_var_hr=epi_var,
                cal_mask_lr=cal_mask_lr,
                downsampler=self._active_downsampler(),
                aligner=self.guide_aligner,
                mode=cal_mode,
                pore_mask_hr=pm_cal,
            )

        return losses

    @torch.no_grad()
    def predict(
        self,
        hr_image: np.ndarray | torch.Tensor,
        coarse_guide: np.ndarray | torch.Tensor | None = None,
        mode: Literal["auto", "multiclass", "multilabel"] = "auto",
        calibrate_to_guide: bool = True,
        abstain_threshold: float | None = None,
        top_k: int | None = None,
        guide_normalization: GuideNormalization | None = None,
        *,
        pore_mask: PoreMaskSpec = None,
    ) -> UncertaintyReport:
        """High-level inference entrypoint: predicts calibrated probabilities and
        an ``UncertaintyReport`` automatically cropped back to ``hr_image`` shape.
        """
        dev = next(self.dinov2.parameters()).device
        hr_prepared, (orig_h, orig_w) = self._prepare_high_level_hr(hr_image)
        pad_h, pad_w = int(hr_prepared.shape[-2]), int(hr_prepared.shape[-1])
        valid_hw = (orig_h, orig_w) if (orig_h, orig_w) != (pad_h, pad_w) else None
        cur_hr_sig = _tensor_content_digest(hr_prepared[:, :orig_h, :orig_w])
        same_specimen = self._fit_hr_sig is not None and cur_hr_sig == self._fit_hr_sig

        if (
            pore_mask is not None
            or not same_specimen
            or self._active_pore_mask_hr is None
        ):
            resolved_pm = self._resolve_pore_mask_2d(
                hr_image,
                padded_hw=(pad_h, pad_w),
                orig_hw=(orig_h, orig_w),
                spec_override=pore_mask,
                device=dev,
            )
        else:
            resolved_pm = self._active_pore_mask_hr

        guide_t: torch.Tensor | None = None
        unguided_transfer = False
        if coarse_guide is not None:
            norm_mode = (
                "preserve_counts"
                if self.spectral_channels is not None
                and self.loss_fn == "forward_psf_poisson"
                else (
                    "global_max"
                    if self.spectral_channels is not None
                    else self._default_guide_norm(guide_normalization)
                )
            )
            guide_t = guide_to_tensor(coarse_guide, device=dev, normalization=norm_mode)
            if self.xrd_fusion_module is None:
                fwd, inv = build_default_tta(shift_dists=[1], include_flips=True)
                self.set_guide_transforms(guide_t, fwd, inv)
            else:
                g_chk = guide_t.unsqueeze(0) if guide_t.ndim == 3 else guide_t
                if _tensor_content_digest(g_chk) != _tensor_content_digest(
                    self.xrd_fusion_module.xrd_image
                ):
                    self._bind_guide_only(guide_t)
                guide_t = self.xrd_fusion_module.xrd_image
        elif self.xrd_fusion_module is not None:
            if same_specimen or self._fit_hr_sig is None:
                guide_t = self.xrd_fusion_module.xrd_image
            else:
                unguided_transfer = True

        ref_g = (
            self._effective_reference_guide(guide_t) if guide_t is not None else None
        )
        n_guides = (
            int(ref_g.shape[1])
            if (ref_g is not None and ref_g.ndim == 4)
            else self.guide_channels
        )
        resolved_mode: Literal["multiclass", "multilabel"] = (
            (
                "multiclass"
                if (n_guides >= 2 or self.loss_fn == "compositional_kl")
                else "multilabel"
            )
            if mode == "auto"
            else mode
        )
        prev_pm = self._active_pore_mask_hr
        self._active_pore_mask_hr = resolved_pm
        try:
            report = self.predict_with_uncertainty(
                hr_prepared,
                guide_lr=guide_t,
                mode=resolved_mode,
                calibrate_to_guide=calibrate_to_guide,
                abstain_threshold=abstain_threshold,
                top_k=top_k,
                unguided_transfer=unguided_transfer,
                use_aligner=(same_specimen or self._fit_hr_sig is None),
                valid_hr_hw=valid_hw,
                pore_mask_hr=resolved_pm,
            )
            cropped = report.crop_spatial(orig_h, orig_w)
            if ref_g is not None:
                lr_hw = (int(ref_g.shape[-2]), int(ref_g.shape[-1]))
                cropped.diagnostics = self._build_diagnostics((orig_h, orig_w), lr_hw)
        finally:
            self._active_pore_mask_hr = prev_pm
        return cropped

    def fit_predict(
        self,
        hr_image: np.ndarray | torch.Tensor,
        coarse_guide: np.ndarray | torch.Tensor,
        epochs: int = 25,
        mode: Literal["auto", "multiclass", "multilabel"] = "auto",
        tta_shifts: Sequence[int] = (1,),
        tta_flips: bool = True,
        calibrate_to_guide: bool = True,
        abstain_threshold: float | None = None,
        guide_normalization: GuideNormalization | None = None,
        n_components: int | None = None,
        spectral_unmix: bool = False,
        *,
        pore_mask: PoreMaskSpec = None,
        max_steps: int | None = None,
        tol: float = 1e-6,
        reset_weights: bool = True,
    ) -> UncertaintyReport:
        """Fit the fusion model on ``(hr_image, coarse_guide)`` and return the
        calibrated, un-cropped ``UncertaintyReport`` in a single call.
        """
        self.fit(
            hr_image,
            coarse_guide,
            epochs=epochs,
            tta_shifts=tta_shifts,
            tta_flips=tta_flips,
            guide_normalization=guide_normalization,
            n_components=n_components,
            spectral_unmix=spectral_unmix,
            pore_mask=pore_mask,
            max_steps=max_steps,
            tol=tol,
            reset_weights=reset_weights,
        )
        return self.predict(
            hr_image,
            coarse_guide=coarse_guide,
            mode=mode,
            calibrate_to_guide=calibrate_to_guide,
            abstain_threshold=abstain_threshold,
            guide_normalization=guide_normalization,
            pore_mask=pore_mask,
        )


# Backward-compatible class aliases
XFuse = CorrelativeFuser
