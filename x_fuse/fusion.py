"""Physics-guided foundation-model feature fusion modules and the unified
``CorrelativeFuser`` (aliased as ``XFuse``) engine.

Capabilities in v0.3+:
- Single-pass ViT backbone initialization with optional multi-layer feature
  pyramids (``layers="last"`` or ``"multiscale"``).
- Frozen TTA feature caching (15x-35x faster probe optimization across epochs)
  and Hann-windowed overlap tiling (``tile_size``) for large micrographs.
- Vectorized multi-channel correlation/BCE gating across all ``C`` ViT channels
  and ``M`` physical guide channels simultaneously.
- Forward-PSF downsampling loss (``forward_psf_bce`` / ``forward_psf_mse``)
  with optional learnable anisotropic PSF widths (``learn_psf=True``) and
  bounded differentiable sub-pixel drift/rigid/affine co-registration
  (``align_guide``).
- High-level 3-line ``fit`` / ``predict`` / ``fit_predict`` API alongside
  full backward compatibility with low-level hooks.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from functools import partial
from typing import Any, Literal, TypeAlias

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import HighResBackbone, Interpolation, LayerSelection
from .registration import AlignMode, DifferentiableGuideAligner
from .tiling import run_tiled_2d
from .transforms import (
    build_default_tta,
    ensure_spatial_divisible,
    normalize_structural,
)
from .uncertainty import (
    UncertaintyReport,
    WelfordAccumulator,
    build_uncertainty_report,
    match_coarse_priors,
)
from .upsamplers import ForwardPSFDownsampler
from .utils import guide_to_tensor
from .vit_patch import AttentionOptions

FusionOptions: TypeAlias = Literal["gating", "learned_gating", "attention"]
LossOptions: TypeAlias = Literal["pearson", "bce", "forward_psf_bce", "forward_psf_mse"]


class XRDCrossAttention(nn.Module):
    """Cross-attention where ``M`` coarse physical guide channels query the
    dense ``(B, C, H, W)`` ViT feature space."""

    def __init__(
        self,
        feat_dim: int = 384,
        num_heads: int = 8,
        guide_channels: int = 1,
        loss_fn: LossOptions = "bce",
        learn_psf: bool = False,
    ):
        super().__init__()
        while feat_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.guide_channels = guide_channels
        self.loss_fn = loss_fn

        self.xrd_to_query = nn.Linear(guide_channels, feat_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.pred_head = nn.Linear(feat_dim, guide_channels)
        self.psf_downsampler = ForwardPSFDownsampler(
            learnable=learn_psf, n_channels=guide_channels
        )

    def _prepare_guide(
        self, guide_map: torch.Tensor, target_hw: tuple[int, int]
    ) -> torch.Tensor:
        if guide_map.dim() == 2:
            guide_map = guide_map.unsqueeze(0).unsqueeze(0)
        elif guide_map.dim() == 3:
            guide_map = guide_map.unsqueeze(0)
        if guide_map.shape[-2:] != target_hw:
            guide_map = F.interpolate(
                guide_map.float(),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        return guide_map

    def forward(self, features: torch.Tensor, xrd_map: torch.Tensor) -> torch.Tensor:
        """Modulate ``features`` ``(B, C, H, W)`` via guide-queried cross-attention."""
        bsz, channels, height, width = features.shape
        orig_dtype = features.dtype

        guide = self._prepare_guide(xrd_map, (height, width)).to(
            device=features.device, dtype=orig_dtype
        )
        feat_flat = features.reshape(bsz, channels, -1).permute(0, 2, 1)  # (B, H*W, C)
        guide_flat = guide.reshape(guide.shape[0], guide.shape[1], -1).permute(0, 2, 1)

        queries = self.xrd_to_query(guide_flat)  # (B_g, H*W, C)
        if queries.shape[0] == 1 and bsz > 1:
            queries = queries.expand(bsz, -1, -1)

        attended, _ = self.attention(queries, feat_flat, feat_flat)
        return attended.permute(0, 2, 1).reshape(features.shape).to(orig_dtype)

    def predict_logits(
        self, features: torch.Tensor, guide_map: torch.Tensor
    ) -> torch.Tensor:
        attended = self.forward(features.float(), guide_map.float())
        bsz, _, height, width = attended.shape
        logits = self.pred_head(attended.permute(0, 2, 3, 1))
        return logits.permute(0, 3, 1, 2).reshape(
            bsz, self.guide_channels, height, width
        )

    def compute_loss(
        self,
        features: torch.Tensor,
        guide_map: torch.Tensor,
        aligner: DifferentiableGuideAligner | None = None,
    ) -> torch.Tensor:
        if guide_map.dim() == 2:
            guide_map = guide_map.unsqueeze(0).unsqueeze(0)
        elif guide_map.dim() == 3:
            guide_map = guide_map.unsqueeze(0)
        logits = self.predict_logits(features.float(), guide_map.float())
        target = guide_map.float().to(logits.device)
        if target.shape[0] == 1 and logits.shape[0] > 1:
            target = target.expand(logits.shape[0], -1, -1, -1)
        lr_hw = (int(target.shape[-2]), int(target.shape[-1]))
        if aligner is not None and aligner.mode != "none":
            logits = aligner(logits, lr_hw)
        if lr_hw != logits.shape[-2:]:
            pred_coarse = self.psf_downsampler(logits, lr_hw)
            return F.binary_cross_entropy_with_logits(pred_coarse, target)
        return F.binary_cross_entropy_with_logits(logits, target)


SpatialCrossAttention = XRDCrossAttention


class LearnedChannelGating(nn.Module):
    """Linear probe ranking ViT channels by physical guide relevance, supporting:
    - Single- or multi-channel guides (``out_channels = M``),
    - Class-balanced BCE, Pearson correlation, and FeatUp-inspired
      **Forward-PSF partial-volume losses** (``forward_psf_bce``, ``forward_psf_mse``),
    - Optional learnable per-channel anisotropic PSF widths (``learn_psf=True``),
    - Direct calibrated HR logit prediction ``predict_logits(features)`` as well
      as hard top-``K`` channel selection ``select_top_k(k)``.
    """

    def __init__(
        self,
        feat_dim: int = 384,
        loss_fn: LossOptions = "bce",
        out_channels: int = 1,
        learn_psf: bool = False,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.loss_fn = loss_fn
        self.out_channels = out_channels
        self.pred_head = nn.Linear(feat_dim, out_channels)
        self.psf_downsampler = ForwardPSFDownsampler(
            learnable=learn_psf, n_channels=out_channels
        )
        self._channel_mask: torch.Tensor | None = None

    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Predict ``(B, M, H, W)`` phase/element logits directly from features."""
        bsz, _, height, width = features.shape
        feat_nhwc = features.float().permute(0, 2, 3, 1)  # (B, H, W, C)
        head_f32 = F.linear(
            feat_nhwc, self.pred_head.weight.float(), self.pred_head.bias.float()
        )
        return head_f32.permute(0, 3, 1, 2).reshape(
            bsz, self.out_channels, height, width
        )

    def compute_loss_from_logits(
        self,
        logits: torch.Tensor,
        xrd_map: torch.Tensor,
        coarse_target: torch.Tensor | None = None,
        aligner: DifferentiableGuideAligner | None = None,
    ) -> torch.Tensor:
        """Evaluate loss given precomputed HR or patch ``logits`` ``(B, M, H, W)``."""
        bsz, _, height, width = logits.shape
        ref = coarse_target if coarse_target is not None else xrd_map
        if ref.dim() == 2:
            ref = ref.unsqueeze(0).unsqueeze(0)
        elif ref.dim() == 3:
            ref = ref.unsqueeze(0)
        ref = ref.float().to(logits.device)
        if ref.shape[0] == 1 and bsz > 1:
            ref = ref.expand(bsz, -1, -1, -1)

        lr_hw = (int(ref.shape[-2]), int(ref.shape[-1]))
        warped_logits = (
            aligner(logits, lr_hw)
            if (aligner is not None and aligner.mode != "none")
            else logits
        )

        if self.loss_fn in ("forward_psf_bce", "forward_psf_mse"):
            if self.loss_fn == "forward_psf_mse":
                # Apply physical PSF blurring in probability space [0, 1]
                pooled_probs = self.psf_downsampler(torch.sigmoid(warped_logits), lr_hw)
                return F.mse_loss(pooled_probs, ref)
            if (
                aligner is not None
                and aligner.mode != "none"
                or self.psf_downsampler.learnable
            ):
                # Probability-space PSF integration gives exact linear partial-volume
                # gradients into both spatial alignment and PSF kernel width!
                probs_hr = torch.sigmoid(warped_logits)
                pooled_p = self.psf_downsampler(probs_hr, lr_hw).clamp(1e-6, 1.0 - 1e-6)
                n_pos = ref.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
                n_neg = (1.0 - ref).sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
                pos_weight = (n_neg / n_pos).clamp(0.1, 20.0)
                bce_elem = -(
                    pos_weight * ref * torch.log(pooled_p)
                    + (1.0 - ref) * torch.log(1.0 - pooled_p)
                )
                return bce_elem.mean()
            pooled_logits = self.psf_downsampler(warped_logits, lr_hw)
            n_pos = ref.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
            n_neg = (1.0 - ref).sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
            pos_weight = (n_neg / n_pos).clamp(0.1, 20.0)
            return F.binary_cross_entropy_with_logits(
                pooled_logits, ref, pos_weight=pos_weight
            )

        if lr_hw != (height, width):
            ref = F.interpolate(
                ref, size=(height, width), mode="bilinear", align_corners=False
            )

        pred_flat = warped_logits.reshape(bsz * self.out_channels, -1)
        target_flat = ref.reshape(bsz * self.out_channels, -1)

        if self.loss_fn == "bce":
            n_pos = target_flat.sum(dim=-1, keepdim=True).clamp(min=1.0)
            n_neg = (1.0 - target_flat).sum(dim=-1, keepdim=True).clamp(min=1.0)
            pos_weight = (n_neg / n_pos).to(pred_flat.device)
            return F.binary_cross_entropy_with_logits(
                pred_flat, target_flat, pos_weight=pos_weight
            )

        pred_c = pred_flat - pred_flat.mean(dim=-1, keepdim=True)
        target_c = target_flat - target_flat.mean(dim=-1, keepdim=True)
        corr = (pred_c * target_c).sum(dim=-1) / (
            pred_c.norm(dim=-1) * target_c.norm(dim=-1) + 1e-8
        )
        return (1.0 - corr).mean()

    def compute_loss(
        self,
        features: torch.Tensor,
        xrd_map: torch.Tensor,
        coarse_target: torch.Tensor | None = None,
        aligner: DifferentiableGuideAligner | None = None,
    ) -> torch.Tensor:
        """Train the linear probe against either spatially matched ``xrd_map`` or
        un-interpolated ``coarse_target`` via ``ForwardPSFDownsampler``."""
        logits = self.predict_logits(features)
        return self.compute_loss_from_logits(
            logits, xrd_map=xrd_map, coarse_target=coarse_target, aligner=aligner
        )

    def select_top_k(self, k: int) -> None:
        """Derive a hard channel mask keeping top-``k`` channels by max probe weight."""
        k_clamped = max(1, min(int(k), self.feat_dim))
        weights = self.pred_head.weight.data.abs().amax(dim=0)  # (feat_dim,)
        top_k = weights.topk(k_clamped).indices
        mask = torch.zeros(self.feat_dim, device=weights.device)
        mask[top_k] = 1.0
        self._channel_mask = mask

    def forward(self, features: torch.Tensor, _xrd_map: torch.Tensor) -> torch.Tensor:
        """Zero out all channels outside the top-``K`` mask."""
        if self._channel_mask is None:
            return features
        mask = self._channel_mask.to(device=features.device, dtype=features.dtype).view(
            1, -1, 1, 1
        )
        return features * mask


def compute_channel_correlation_gates(
    features: torch.Tensor,
    guide_map: torch.Tensor,
    loss_fn: LossOptions = "pearson",
    top_k: int | None = None,
) -> torch.Tensor:
    """Vectorized per-channel relevance gates ``(M, C)`` between ``features``
    ``(B, C, H, W)`` and ``guide_map`` ``(1, M, H_lr, W_lr)`` without Python loops."""
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
            f_norm = f_centered.norm(dim=1, keepdim=True)  # (C, 1)
            g_norm = g_centered.norm(dim=1, keepdim=True)  # (M, 1)
            denom = g_norm @ f_norm.T  # (M, C)
            cov = g_centered @ f_centered.T  # (M, C)
            scores = torch.where(
                denom > 1e-12, cov / denom.clamp(min=1e-12), torch.zeros_like(cov)
            )
            scores = torch.nan_to_num(scores, nan=0.0).clamp(0.0, 1.0)
    else:
        f_min = feat_mat.min(dim=1, keepdim=True).values
        f_max = feat_mat.max(dim=1, keepdim=True).values
        f_normed = ((feat_mat - f_min) / (f_max - f_min + 1e-8)).clamp(1e-6, 1.0 - 1e-6)
        log_f = torch.log(f_normed)
        log_1mf = torch.log(1.0 - f_normed)
        bce = -(guide_mat @ log_f.T + (1.0 - guide_mat) @ log_1mf.T) / max(n_pts, 1)
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
    fuses it with dense ViT features via correlation gating, learned channel
    gating, or spatial cross-attention."""

    def __init__(
        self,
        xrd_img: torch.Tensor,
        transform: list[partial],
        require_grad: bool = False,
        learned_gating: nn.Module | None = None,
        spatial_attention: nn.Module | None = None,
        loss_fn: LossOptions = "pearson",
    ):
        super().__init__()
        self.require_grad = require_grad
        self.xrd_image = xrd_img
        self.transform = transform
        self.learned_gating_module = learned_gating
        self.spatial_attention_module = spatial_attention
        self.loss_fn = loss_fn

    def get_tr(self) -> torch.Tensor:
        """Apply all forward TTA transforms to the guide tensor as a batch."""
        base = self.xrd_image
        if base.ndim == 2:
            base = base.unsqueeze(0).unsqueeze(0)
        elif base.ndim == 3:
            base = base.unsqueeze(0)
        if len(self.transform) == 0:
            return base.unsqueeze(0) if base.ndim == 4 else base
        tr_list = []
        for tr_fn in self.transform:
            t_out = tr_fn(base)
            if t_out.ndim == 3:
                t_out = t_out.unsqueeze(0)
            tr_list.append(t_out)
        return torch.stack(tr_list)  # (N_transforms, 1, M, H_lr, W_lr)

    def __translen__(self, transform: list[partial]) -> int:
        return len(transform)

    def forward_xrd(
        self,
        x: torch.Tensor,
        idx: int,
        xrd_fusion_method: FusionOptions,
        top_k: int | None = None,
    ) -> torch.Tensor:
        tr_xrd = self.get_tr()[idx]
        if xrd_fusion_method == "gating":
            return self._direct_correlation_gating(x, tr_xrd, top_k=top_k)
        if xrd_fusion_method == "learned_gating":
            return self._learned_channel_gating(x, tr_xrd)
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
            self.learned_gating_module = LearnedChannelGating(
                feat_dim=features.shape[1]
            ).to(device=features.device, dtype=features.dtype)

        _, _, height, width = features.shape
        if xrd_map.dim() == 2:
            xrd_map = xrd_map.unsqueeze(0).unsqueeze(0)
        elif xrd_map.dim() == 3:
            xrd_map = xrd_map.unsqueeze(0)
        xrd_map = F.interpolate(
            xrd_map.to(features.dtype),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
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
            self.spatial_attention_module = XRDCrossAttention(
                feat_dim=features.shape[1], num_heads=8, guide_channels=n_guides
            ).to(device=features.device, dtype=features.dtype)

        _, _, height, width = features.shape
        if xrd_map.dim() == 2:
            xrd_map = xrd_map.unsqueeze(0).unsqueeze(0)
        elif xrd_map.dim() == 3:
            xrd_map = xrd_map.unsqueeze(0)
        xrd_map = F.interpolate(
            xrd_map.to(features.dtype),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return self.spatial_attention_module(features, xrd_map)


GuideFusionMethod = XRDFusionMethod


class CorrelativeFuser(HighResBackbone):
    """Modality-agnostic correlative fusion model combining dense ViT features
    from a high-resolution structural image ``I_HR`` with ``M`` coarse physical
    guide channels ``G_LR``, outputting both gated high-res features and
    calibrated phase/element segmentations with tri-part pixel certainty.

    Supports both a **3-line high-level API** (``fit``, ``predict``, ``fit_predict``)
    and full low-level control (``set_guide_transforms``, ``train_fusion_step``,
    ``predict_with_uncertainty``).
    """

    def __init__(
        self,
        dino_name: str | None = None,
        xrd_fuse_method: FusionOptions | None = None,
        stride: int = 4,
        pca_dim: int = -1,
        track_grad: bool = False,
        dtype: torch.dtype | int = torch.float16,
        loss_fn: LossOptions = "bce",
        guide_channels: int = 1,
        interpolation_mode: Interpolation = "nearest-exact",
        pretrained: bool = True,
        lr: float = 1e-3,
        *,
        backbone: str | None = None,
        fusion_method: FusionOptions | None = None,
        layers: LayerSelection = "last",
        align_guide: AlignMode = "none",
        max_shift_coarse_px: float = 1.5,
        learn_psf: bool = False,
        tile_size: int | None = None,
        tile_overlap: int = 64,
        cache_features: bool = True,
        **kwargs,
    ):
        resolved_backbone = backbone or dino_name or "dinov2_vits14"
        resolved_fusion: FusionOptions = (
            fusion_method or xrd_fuse_method or "learned_gating"
        )
        device = kwargs.get("device", None)
        super().__init__(
            resolved_backbone,
            stride=stride,
            pca_dim=pca_dim,
            dtype=dtype,
            track_grad=track_grad,
            pretrained=pretrained,
            interpolation_mode=interpolation_mode,
            layers=layers,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            **kwargs,
        )
        self.backbone_name: str = resolved_backbone
        self.fusion_method: FusionOptions = resolved_fusion
        self.xrd_fuse_method: FusionOptions = resolved_fusion
        self.loss_fn: LossOptions = loss_fn
        self.guide_channels: int = guide_channels
        self.lr: float = lr
        self.align_guide: AlignMode = align_guide
        self.max_shift_coarse_px: float = float(max_shift_coarse_px)
        self.learn_psf: bool = bool(learn_psf)
        self.cache_features: bool = bool(cache_features)

        self.learned_gating: LearnedChannelGating | None = None
        self.spatial_attention: XRDCrossAttention | None = None
        self.guide_aligner: DifferentiableGuideAligner = DifferentiableGuideAligner(
            mode=align_guide, max_shift_coarse_px=max_shift_coarse_px
        )
        if device is not None:
            self.guide_aligner = self.guide_aligner.to(device)
        self.fusion_optimizer: torch.optim.Optimizer | None = None
        self.align_optimizer: torch.optim.Optimizer | None = None
        self._init_fusion_heads(guide_channels, device=device)

        self.xrd_fusion_module: XRDFusionMethod | None = None
        self._tta_cache: OrderedDict[
            tuple[Any, ...], tuple[str, list[torch.Tensor]]
        ] = OrderedDict()
        self._max_cache_entries: int = 8
        self._last_loss_history: list[float] = []

    def clear_feature_cache(self) -> None:
        """Release any cached frozen TTA feature tensors."""
        self._tta_cache.clear()

    def _init_fusion_heads(
        self, guide_channels: int, device: str | torch.device | None = None
    ) -> None:
        self.guide_channels = guide_channels
        if self.xrd_fuse_method == "learned_gating":
            self.learned_gating = LearnedChannelGating(
                feat_dim=self.feat_dim,
                loss_fn=self.loss_fn,
                out_channels=guide_channels,
                learn_psf=self.learn_psf,
            )
            if device is not None:
                self.learned_gating = self.learned_gating.to(device)
            self.fusion_optimizer = torch.optim.Adam(
                self.learned_gating.parameters(), lr=self.lr
            )
        elif self.xrd_fuse_method == "attention":
            self.spatial_attention = XRDCrossAttention(
                feat_dim=self.feat_dim,
                num_heads=8,
                guide_channels=guide_channels,
                loss_fn=self.loss_fn,
                learn_psf=self.learn_psf,
            )
            if device is not None:
                self.spatial_attention = self.spatial_attention.to(device)
            self.fusion_optimizer = torch.optim.Adam(
                self.spatial_attention.parameters(), lr=self.lr
            )

        self.guide_aligner = DifferentiableGuideAligner(
            mode=self.align_guide, max_shift_coarse_px=self.max_shift_coarse_px
        )
        if device is not None:
            self.guide_aligner = self.guide_aligner.to(device)
        align_params = list(self.guide_aligner.parameters())
        self.align_optimizer = (
            torch.optim.Adam(align_params, lr=max(self.lr * 2.0, 2e-2))
            if align_params
            else None
        )

    def set_guide_transforms(
        self, guide_tensor: torch.Tensor, fwd: list[partial], inv: list[partial]
    ) -> None:
        """Bind a ``(1, M, H_lr, W_lr)`` physical guide tensor and TTA transforms."""
        prev_n_tr = len(self.transforms)
        super().set_transforms(fwd, inv)
        if len(self.transforms) != prev_n_tr:
            self.clear_feature_cache()
        if guide_tensor.ndim == 2:
            guide_tensor = guide_tensor.unsqueeze(0).unsqueeze(0)
        elif guide_tensor.ndim == 3:
            guide_tensor = guide_tensor.unsqueeze(0)

        n_guides = guide_tensor.shape[1]
        if n_guides != self.guide_channels:
            dev = next(self.dinov2.parameters()).device
            self._init_fusion_heads(n_guides, device=dev)

        self.xrd_fusion_module = XRDFusionMethod(
            xrd_img=guide_tensor,
            transform=self.transforms,
            require_grad=self.track_grad,
            learned_gating=self.learned_gating,
            spatial_attention=self.spatial_attention,
            loss_fn=self.loss_fn,
        )

    def set_xrd_transforms(
        self, xrd_tensor: torch.Tensor, fwd: list[partial], inv: list[partial]
    ) -> None:
        """Backward-compatible alias for ``set_guide_transforms``."""
        self.set_guide_transforms(xrd_tensor, fwd, inv)

    def _cache_key(self, x: torch.Tensor, want_hr: bool) -> tuple[Any, ...]:
        flat = x.detach().float().reshape(-1)
        n_el = flat.numel()
        s0 = round(float(flat[0].item()), 5) if n_el > 0 else 0.0
        sm = round(float(flat[n_el // 2].item()), 5) if n_el > 1 else 0.0
        s1 = round(float(flat[-1].item()), 5) if n_el > 2 else 0.0
        mean_v = round(float(flat.mean().item()), 5) if n_el > 0 else 0.0
        return (
            tuple(x.shape),
            str(x.device),
            len(self.transforms),
            self.interpolation_mode,
            self.tile_size,
            want_hr,
            s0,
            sm,
            s1,
            mean_v,
        )

    @torch.no_grad()
    def _extract_tta_views(
        self, x: torch.Tensor, want_hr: bool
    ) -> tuple[str, list[torch.Tensor]]:
        """Extract frozen ViT features per TTA view with optional LRU caching."""
        _, img_h, img_w = x.shape
        # Only cache full HR-upsampled views when spatial footprint is moderate
        cache_hr = want_hr and (img_h * img_w <= 512 * 512)
        key = self._cache_key(x, cache_hr)
        if self.cache_features and key in self._tta_cache:
            self._tta_cache.move_to_end(key)
            return self._tta_cache[key]

        x_in = x.to(self.dtype) if self.dtype != torch.float32 else x
        img_batch = self.get_transformed_input_batch(x_in, self.transforms)
        n_transforms = len(self.transforms)

        if self.tile_size is not None and max(img_h, img_w) > self.tile_size:
            # Tiled HR extraction per TTA view so large micrographs never OOM
            hr_views: list[torch.Tensor] = []
            for i in range(n_transforms):
                tr_img = img_batch[i]

                def _single_tr_tile(tile_chw: torch.Tensor) -> torch.Tensor:
                    t_in = tile_chw.unsqueeze(0)
                    _, th, tw = tile_chw.shape
                    nph, npw = self.get_n_patches(th, tw)
                    tok = self.extract_patch_tokens(t_in, attn_choice="none")
                    perm = (
                        tok.squeeze(0)
                        .reshape(nph, npw, self.feat_dim)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                    )
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
                hr_views.append(hr_v)
            return ("hr", hr_views)

        n_patch_h, n_patch_w = self.get_n_patches(img_h, img_w)
        out_views: list[torch.Tensor] = []
        kind = "hr" if cache_hr else "patch"
        for i in range(n_transforms):
            transformed_img = img_batch[i].unsqueeze(0)
            feats = self.extract_patch_tokens(transformed_img, attn_choice="none")
            permuted = (
                feats.squeeze(0)
                .reshape(n_patch_h, n_patch_w, self.feat_dim)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            if cache_hr:
                hr_f = self._upsample_patch_map(
                    permuted.float(), (img_h, img_w), guidance_img=transformed_img
                )
                out_views.append(hr_f)
            else:
                out_views.append(permuted)

        result = (kind, out_views)
        if self.cache_features:
            self._tta_cache[key] = result
            while len(self._tta_cache) > self._max_cache_entries:
                self._tta_cache.popitem(last=False)
        return result

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

        Returns:
            ``(mean_features, feature_variance)`` each of shape ``(1, C, H, W)``.
        """
        assert self.xrd_fusion_module is not None, (
            "Call set_guide_transforms / set_xrd_transforms before forward."
        )

        x_in = x.to(self.dtype) if self.dtype != torch.float32 else x.clone()
        img_batch = self.get_transformed_input_batch(x_in, self.transforms)
        _, img_h, img_w = x_in.shape
        n_patch_h, n_patch_w = self.get_n_patches(img_h, img_w)

        welford = WelfordAccumulator()
        n_transforms = len(self.transforms)

        for i in range(n_transforms):
            transformed_img = img_batch[i].unsqueeze(0)
            feats = self.extract_patch_tokens(transformed_img, attn_choice=attn_choice)
            channels = int(feats.shape[-1])
            permuted = (
                feats.squeeze(0)
                .reshape(n_patch_h, n_patch_w, channels)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            fused_img = self.xrd_fusion_module.forward_xrd(
                permuted, i, self.xrd_fuse_method, top_k=top_k
            )
            full_size = self._upsample_patch_map(
                fused_img, (img_h, img_w), guidance_img=transformed_img
            )
            inverted = self.inverse_transforms[i](full_size)
            welford.update(inverted)

        mean_f32, var_f32 = welford.finalize()
        return mean_f32.to(self.dtype), var_f32

    def train_fusion_step(self, x: torch.Tensor, update_aligner: bool = True) -> float:
        """Update ``LearnedChannelGating`` or ``XRDCrossAttention`` (plus optional
        ``DifferentiableGuideAligner`` and learnable PSF) for one optimization step
        across all TTA views while keeping the ViT frozen."""
        assert self.xrd_fuse_method in (
            "learned_gating",
            "attention",
        ), "train_fusion_step applies to 'learned_gating' or 'attention'."
        assert (
            self.xrd_fusion_module is not None and self.fusion_optimizer is not None
        ), "Call set_guide_transforms / set_xrd_transforms before training."

        _, img_h, img_w = x.shape
        want_hr = (
            self.loss_fn in ("forward_psf_bce", "forward_psf_mse")
            or self.align_guide != "none"
        )
        kind, cached_views = self._extract_tta_views(x, want_hr=want_hr)
        img_batch: torch.Tensor | None = None
        if want_hr and kind == "patch":
            x_in = x.to(self.dtype) if self.dtype != torch.float32 else x
            with torch.no_grad():
                img_batch = self.get_transformed_input_batch(x_in, self.transforms)

        n_transforms = len(self.transforms)
        total_loss = 0.0
        self.fusion_optimizer.zero_grad()
        if self.align_optimizer is not None:
            self.align_optimizer.zero_grad()

        use_align = self.align_guide != "none"
        canon_guide = self.xrd_fusion_module.xrd_image
        if canon_guide.dim() == 2:
            canon_guide = canon_guide.unsqueeze(0).unsqueeze(0)
        elif canon_guide.dim() == 3:
            canon_guide = canon_guide.unsqueeze(0)
        lr_hw = (int(canon_guide.shape[-2]), int(canon_guide.shape[-1]))

        for i in range(n_transforms):
            tr_guide_coarse = self.xrd_fusion_module.get_tr()[i]
            if tr_guide_coarse.dim() == 3:
                tr_guide_coarse = tr_guide_coarse.unsqueeze(0)

            if self.xrd_fuse_method == "learned_gating":
                assert self.learned_gating is not None
                if want_hr:
                    if kind == "hr":
                        hr_feats = cached_views[i]
                    else:
                        assert img_batch is not None
                        hr_feats = self._upsample_patch_map(
                            cached_views[i].float(),
                            (img_h, img_w),
                            guidance_img=img_batch[i].unsqueeze(0),
                        )
                    if use_align:
                        # Invert TTA back to canonical I_HR frame so drift alignment
                        # is 100% equivariant across flips, shifts, and rotations!
                        logits_tr = self.learned_gating.predict_logits(hr_feats)
                        logits_canon = self.inverse_transforms[i](logits_tr)
                        loss = self.learned_gating.compute_loss_from_logits(
                            logits_canon,
                            xrd_map=canon_guide,
                            coarse_target=canon_guide,
                            aligner=self.guide_aligner,
                        )
                    else:
                        loss = self.learned_gating.compute_loss(
                            hr_feats, tr_guide_coarse, coarse_target=tr_guide_coarse
                        )
                else:
                    loss = self.learned_gating.compute_loss(
                        cached_views[i], tr_guide_coarse
                    )
                params = list(self.learned_gating.parameters())
            else:
                assert self.spatial_attention is not None
                feat_in = cached_views[i]
                loss = self.spatial_attention.compute_loss(
                    feat_in,
                    canon_guide if use_align else tr_guide_coarse,
                    aligner=self.guide_aligner if use_align else None,
                )
                params = list(self.spatial_attention.parameters())

            if use_align and update_aligner:
                loss = loss + self.guide_aligner.regularization_loss(
                    lr_hw=lr_hw, device=loss.device
                )

            scaled_loss = loss / max(n_transforms, 1)
            scaled_loss.backward()
            total_loss += float(loss.detach().cpu().item())

        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        self.fusion_optimizer.step()
        if use_align and update_aligner and self.align_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.guide_aligner.parameters()), max_norm=1.0
            )
            self.align_optimizer.step()
        return total_loss / max(n_transforms, 1)

    def _active_downsampler(self) -> ForwardPSFDownsampler | None:
        if self.xrd_fuse_method == "learned_gating" and self.learned_gating is not None:
            return self.learned_gating.psf_downsampler
        if self.xrd_fuse_method == "attention" and self.spatial_attention is not None:
            return self.spatial_attention.psf_downsampler
        return None

    def _build_diagnostics(
        self, hr_hw: tuple[int, int], lr_hw: tuple[int, int]
    ) -> dict[str, Any]:
        diag: dict[str, Any] = {
            "backbone": self.backbone_name,
            "fusion_method": self.xrd_fuse_method,
            "layers": (
                list(self.layer_indices) if self.layer_indices is not None else "last"
            ),
            "feat_dim": self.feat_dim,
            "interpolation_mode": self.interpolation_mode,
        }
        diag.update(self.guide_aligner.diagnostics(hr_hw=hr_hw, lr_hw=lr_hw))
        ds = self._active_downsampler()
        if ds is not None:
            diag.update(ds.diagnostics(hr_hw=hr_hw, lr_hw=lr_hw))
        if self._last_loss_history:
            diag["initial_loss"] = round(self._last_loss_history[0], 5)
            diag["final_loss"] = round(self._last_loss_history[-1], 5)
        return diag

    @torch.no_grad()
    def _predict_logits_and_var_tile(
        self, x_tile: torch.Tensor, guide_lr: torch.Tensor, top_k: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_in = x_tile.to(self.dtype) if self.dtype != torch.float32 else x_tile.clone()
        img_batch = self.get_transformed_input_batch(x_in, self.transforms)
        _, img_h, img_w = x_in.shape
        n_patch_h, n_patch_w = self.get_n_patches(img_h, img_w)

        logit_welford = WelfordAccumulator()
        prob_welford = WelfordAccumulator()

        for i in range(len(self.transforms)):
            transformed_img = img_batch[i].unsqueeze(0)
            feats = self.extract_patch_tokens(transformed_img, attn_choice="none")
            permuted = (
                feats.squeeze(0)
                .reshape(n_patch_h, n_patch_w, self.feat_dim)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            hr_feats = self._upsample_patch_map(
                permuted.float(), (img_h, img_w), guidance_img=transformed_img
            )

            if self.xrd_fuse_method == "learned_gating" and self.learned_gating:
                logits_view = self.learned_gating.predict_logits(hr_feats)
            elif self.xrd_fuse_method == "attention" and self.spatial_attention:
                tr_g = (
                    self.xrd_fusion_module.get_tr()[i]
                    if self.xrd_fusion_module is not None
                    else guide_lr
                )
                logits_view = self.spatial_attention.predict_logits(hr_feats, tr_g)
            else:
                tr_g = (
                    self.xrd_fusion_module.get_tr()[i]
                    if self.xrd_fusion_module is not None
                    else guide_lr
                )
                gates_mc = compute_channel_correlation_gates(
                    permuted.float(), tr_g, loss_fn=self.loss_fn, top_k=top_k
                ).float()
                feat_std = (hr_feats - hr_feats.mean(dim=(-2, -1), keepdim=True)) / (
                    hr_feats.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
                )
                w_norm = gates_mc / gates_mc.sum(dim=1, keepdim=True).clamp(min=1e-6)
                logits_view = torch.einsum("mc,bchw->bmhw", w_norm, feat_std)

            inv_logits = self.inverse_transforms[i](logits_view)
            logit_welford.update(inv_logits)
            prob_welford.update(torch.sigmoid(inv_logits))

        mean_logits, _ = logit_welford.finalize()
        _, epistemic_prob_var = prob_welford.finalize()
        return mean_logits, epistemic_prob_var

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        guide_lr: torch.Tensor | None = None,
        mode: Literal["multiclass", "multilabel"] = "multilabel",
        calibrate_to_guide: bool = True,
        abstain_threshold: float | None = None,
        top_k: int | None = None,
    ) -> UncertaintyReport:
        """Produce calibrated high-resolution phase/element probabilities and a
        full ``UncertaintyReport`` (chemical entropy, TTA epistemic variance, and
        physical forward-model residual) in a single TTA sweep."""
        if guide_lr is None:
            assert self.xrd_fusion_module is not None, (
                "Provide guide_lr or call set_guide_transforms first."
            )
            guide_lr = self.xrd_fusion_module.xrd_image
        if guide_lr.ndim == 2:
            guide_lr = guide_lr.unsqueeze(0).unsqueeze(0)
        elif guide_lr.ndim == 3:
            guide_lr = guide_lr.unsqueeze(0)

        _, img_h, img_w = x.shape
        if (
            self.tile_size is not None
            and max(img_h, img_w) > self.tile_size
            and self.xrd_fuse_method == "learned_gating"
        ):
            n_ch = self.guide_channels

            def _tile_pred(tile_chw: torch.Tensor) -> torch.Tensor:
                ml, vp = self._predict_logits_and_var_tile(
                    tile_chw, guide_lr, top_k=top_k
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
            mean_logits = packed[:, :n_ch]
            epistemic_prob_var = packed[:, n_ch:]
        else:
            mean_logits, epistemic_prob_var = self._predict_logits_and_var_tile(
                x, guide_lr, top_k=top_k
            )

        if mode == "multiclass":
            bg_logit = torch.zeros(
                (mean_logits.shape[0], 1, img_h, img_w),
                device=mean_logits.device,
                dtype=mean_logits.dtype,
            )
            full_logits = torch.cat([bg_logit, mean_logits], dim=1)
            probs = (
                match_coarse_priors(full_logits, guide_lr, mode="multiclass")
                if calibrate_to_guide
                else F.softmax(full_logits, dim=1)
            )
        else:
            probs = (
                match_coarse_priors(mean_logits, guide_lr, mode="multilabel")
                if calibrate_to_guide
                else torch.sigmoid(mean_logits)
            )

        lr_hw = (int(guide_lr.shape[-2]), int(guide_lr.shape[-1]))
        return build_uncertainty_report(
            probs_hr=probs,
            guide_lr=guide_lr,
            epistemic_var=epistemic_prob_var,
            mode=mode,
            abstain_threshold=abstain_threshold,
            downsampler=self._active_downsampler(),
            aligner=self.guide_aligner,
            diagnostics=self._build_diagnostics((img_h, img_w), lr_hw),
        )

    # ── High-Level 3-Line Pythonic API (v0.4+) ────────────────────────────────

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
        if (
            self.tile_size is not None
            and max(hr_t.shape[-2], hr_t.shape[-1]) > self.tile_size
        ):
            return hr_t, (int(hr_t.shape[-2]), int(hr_t.shape[-1]))
        return ensure_spatial_divisible(
            hr_t, patch_size=self.original_patch_size, stride=self.stride[0]
        )

    def fit(
        self,
        hr_image: np.ndarray | torch.Tensor,
        coarse_guide: np.ndarray | torch.Tensor,
        epochs: int = 25,
        tta_shifts: Sequence[int] = (1,),
        tta_flips: bool = True,
        warmup_fraction: float = 0.30,
    ) -> list[float]:
        """High-level training entrypoint: normalizes and pads ``hr_image`` and
        ``coarse_guide``, caches frozen ViT TTA features, and optimizes the
        fusion head (+ optional co-registration & PSF width).
        """
        dev = next(self.dinov2.parameters()).device
        hr_prepared, _ = self._prepare_high_level_hr(hr_image)
        guide_t = guide_to_tensor(coarse_guide, device=dev, normalize=True)
        fwd, inv = build_default_tta(
            shift_dists=list(tta_shifts),
            pattern="Neumann",
            include_flips=tta_flips,
        )
        self.set_guide_transforms(guide_t, fwd, inv)

        if self.xrd_fuse_method not in ("learned_gating", "attention"):
            self._last_loss_history = []
            return []

        n_epochs = max(1, int(epochs))
        warmup_steps = (
            int(round(warmup_fraction * n_epochs)) if self.align_guide != "none" else 0
        )
        losses: list[float] = []
        for ep in range(n_epochs):
            loss_val = self.train_fusion_step(
                hr_prepared, update_aligner=(ep >= warmup_steps)
            )
            losses.append(loss_val)
        self._last_loss_history = losses
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
    ) -> UncertaintyReport:
        """High-level inference entrypoint: predicts calibrated probabilities and
        an ``UncertaintyReport`` automatically cropped back to ``hr_image`` shape.
        """
        dev = next(self.dinov2.parameters()).device
        hr_prepared, (orig_h, orig_w) = self._prepare_high_level_hr(hr_image)
        guide_t: torch.Tensor | None = None
        if coarse_guide is not None:
            guide_t = guide_to_tensor(coarse_guide, device=dev, normalize=True)
            if self.xrd_fusion_module is None:
                fwd, inv = build_default_tta(shift_dists=[1], include_flips=True)
                self.set_guide_transforms(guide_t, fwd, inv)
        elif self.xrd_fusion_module is not None:
            guide_t = self.xrd_fusion_module.xrd_image

        n_guides = (
            guide_t.shape[1]
            if (guide_t is not None and guide_t.ndim == 4)
            else self.guide_channels
        )
        resolved_mode: Literal["multiclass", "multilabel"] = (
            ("multiclass" if n_guides >= 2 else "multilabel")
            if mode == "auto"
            else mode
        )
        report = self.predict_with_uncertainty(
            hr_prepared,
            guide_lr=guide_t,
            mode=resolved_mode,
            calibrate_to_guide=calibrate_to_guide,
            abstain_threshold=abstain_threshold,
            top_k=top_k,
        )
        cropped = report.crop_spatial(orig_h, orig_w)
        if guide_t is not None:
            lr_hw = (int(guide_t.shape[-2]), int(guide_t.shape[-1]))
            cropped.diagnostics = self._build_diagnostics((orig_h, orig_w), lr_hw)
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
        )
        return self.predict(
            hr_image,
            coarse_guide=coarse_guide,
            mode=mode,
            calibrate_to_guide=calibrate_to_guide,
            abstain_threshold=abstain_threshold,
        )


# Backward-compatible class alias
XFuse = CorrelativeFuser
