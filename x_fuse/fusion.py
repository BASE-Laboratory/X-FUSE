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

from collections import OrderedDict
from collections.abc import Sequence
from functools import partial
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import HighResBackbone, Interpolation, LayerSelection
from .heads import (
    FusionOptions,
    LearnedChannelGating as LearnedChannelGating,
    LinearProbeHead,
    SpatialConvMLPHead,
    SpatialCrossAttention as SpatialCrossAttention,
    WindowedGuideCrossAttentionHead,
    XRDCrossAttention as XRDCrossAttention,
)
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
    UncertaintyReport,
    WelfordAccumulator,
    build_uncertainty_report,
    match_coarse_priors,
)
from .upsamplers import ForwardPSFDownsampler
from .utils import guide_to_tensor
from .vit_patch import AttentionOptions


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

    def get_tr(self) -> torch.Tensor:
        """Apply all forward TTA transforms to the guide tensor as a batch."""
        base = self.xrd_image
        if base.ndim == 2:
            base = base.unsqueeze(0).unsqueeze(0)
        elif base.ndim == 3:
            base = base.unsqueeze(0)
        if len(self.transform) == 0:
            return base.unsqueeze(0)
        tr_list = []
        for tr_fn in self.transform:
            t_out = tr_fn(base)
            if t_out.ndim == 3:
                t_out = t_out.unsqueeze(0)
            tr_list.append(t_out)
        return torch.stack(tr_list)

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
        dtype: torch.dtype | int = torch.float16,
        loss_fn: LossOptions = "bce",
        guide_channels: int = 1,
        interpolation_mode: Interpolation = "nearest-exact",
        pretrained: bool = True,
        lr: float = 1e-3,
        *,
        backbone: str | None = None,
        fusion_method: FusionOptions | None = None,
        upsampler: Interpolation | None = None,
        learned_upsampler: bool = False,
        debias_positional: bool = False,
        debias_degree: int = 2,
        layers: LayerSelection = "last",
        align_guide: AlignMode = "none",
        max_shift_coarse_px: float = 1.5,
        bspline_grid_size: int = 6,
        learn_psf: bool = False,
        conformal_alpha: float | None = None,
        conformal_method: Literal["lac", "aps"] = "lac",
        spectral_channels: int | None = None,
        tv_weight: float = 1e-3,
        tile_size: int | None = None,
        tile_overlap: int = 64,
        cache_features: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        resolved_backbone = backbone or dino_name or "dinov2_vits14"
        resolved_fusion: FusionOptions = (
            fusion_method or xrd_fuse_method or "learned_gating"
        )
        resolved_interp: Interpolation = upsampler or interpolation_mode
        device = kwargs.get("device", None)

        self.backbone_extractor = HighResBackbone(
            resolved_backbone,
            stride=stride,
            pca_dim=pca_dim,
            dtype=dtype,
            track_grad=track_grad,
            pretrained=pretrained,
            interpolation_mode=resolved_interp,
            layers=layers,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            debias_positional=debias_positional,
            debias_degree=debias_degree,
            learned_upsampler=learned_upsampler,
            **kwargs,
        )

        self.backbone_name: str = resolved_backbone
        self.fusion_method: FusionOptions = resolved_fusion
        self.xrd_fuse_method: FusionOptions = resolved_fusion
        self.loss_fn: LossOptions = loss_fn
        self.guide_channels: int = max(1, int(guide_channels))
        self.spectral_channels: int | None = spectral_channels
        self.lr: float = float(lr)
        self.align_guide: AlignMode = align_guide
        self.max_shift_coarse_px: float = float(max_shift_coarse_px)
        self.bspline_grid_size: int = int(bspline_grid_size)
        self.learn_psf: bool = bool(learn_psf)
        self.learned_upsampler: bool = bool(learned_upsampler)
        self.debias_positional: bool = bool(debias_positional)
        self.tv_weight: float = float(tv_weight)
        self.cache_features: bool = bool(cache_features)
        self.conformal_alpha: float | None = conformal_alpha
        self.conformal_method: Literal["lac", "aps"] = conformal_method

        self.calibrator: ConformalCalibrator | None = (
            ConformalCalibrator(alpha=conformal_alpha, method=conformal_method)
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
        self._tta_cache: OrderedDict[
            tuple[Any, ...], tuple[str, list[torch.Tensor]]
        ] = OrderedDict()
        self._max_cache_entries: int = 8
        self._last_loss_history: list[float] = []

    # ── Clean Compositional Forwarding Properties (100% Back-Compat) ─────────

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

    @property
    def tile_overlap(self) -> int:
        return self.backbone_extractor.tile_overlap

    @tile_overlap.setter
    def tile_overlap(self, value: int) -> None:
        self.backbone_extractor.tile_overlap = int(value)

    @property
    def interpolation_mode(self) -> Interpolation:
        return self.backbone_extractor.interpolation_mode

    @interpolation_mode.setter
    def interpolation_mode(self, value: Interpolation) -> None:
        self.backbone_extractor.interpolation_mode = value

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

    # ── Head & Optimizer Initialization ──────────────────────────────────────

    def clear_feature_cache(self) -> None:
        """Release any cached frozen TTA feature tensors."""
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

    def set_guide_transforms(
        self,
        guide_tensor: torch.Tensor,
        fwd: list[partial],
        inv: list[partial],
        *,
        n_components: int | None = None,
        spectral_unmix: bool = False,
    ) -> None:
        """Bind a ``(1, M, H_lr, W_lr)`` physical guide tensor (or 3D spectral
        cube ``(1, E, H_lr, W_lr)`` when ``spectral_unmix=True``) and TTA views.
        """
        prev_n_tr = len(self.transforms)
        self.set_transforms(fwd, inv)
        if len(self.transforms) != prev_n_tr:
            self.clear_feature_cache()
        if guide_tensor.ndim == 2:
            guide_tensor = guide_tensor.unsqueeze(0).unsqueeze(0)
        elif guide_tensor.ndim == 3:
            guide_tensor = guide_tensor.unsqueeze(0)

        dev = next(self.dinov2.parameters()).device
        if spectral_unmix:
            spec_ch = int(guide_tensor.shape[1])
            m_comp = int(n_components or self.guide_channels)
            if self.spectral_channels != spec_ch or self.guide_channels != m_comp:
                self._init_fusion_heads(m_comp, device=dev, spectral_channels=spec_ch)
            phys = self._active_physics()
            if phys is not None and phys.spectral_dict is not None:
                self._unmixed_guide_lr = phys.spectral_dict.initialize_from_cube(
                    guide_tensor.to(dev)
                )
        else:
            self._unmixed_guide_lr = None
            n_guides = int(guide_tensor.shape[1])
            if n_guides != self.guide_channels or self.spectral_channels is not None:
                self._init_fusion_heads(n_guides, device=dev, spectral_channels=None)

        self.xrd_fusion_module = XRDFusionMethod(
            xrd_img=guide_tensor.to(dev),
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
            self.debias_positional,
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
        train_up = self.learned_upsampler and self.interpolation_mode == "cross_attn"
        cache_hr = want_hr and not train_up and (img_h * img_w <= 512 * 512)
        key = self._cache_key(x, cache_hr)
        if self.cache_features and key in self._tta_cache:
            self._tta_cache.move_to_end(key)
            return self._tta_cache[key]

        x_in = x.to(self.dtype) if self.dtype != torch.float32 else x
        img_batch = self.get_transformed_input_batch(x_in, self.transforms)
        n_transforms = len(self.transforms)

        if self.tile_size is not None and max(img_h, img_w) > self.tile_size:
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
                out_views.append(self.backbone_extractor.debiaser(permuted))

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
            permuted = self.backbone_extractor.debiaser(permuted)
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

    # ── Optimization Step ────────────────────────────────────────────────────

    def train_fusion_step(self, x: torch.Tensor, update_aligner: bool = True) -> float:
        """Optimize the active fusion head (+ optional trainable cross-attention
        upsampler, ``DifferentiableGuideAligner``, learnable PSF, and spectral
        dictionary) across all TTA views while keeping the ViT frozen.
        """
        assert self.xrd_fuse_method in (
            "learned_gating",
            "spatial_mlp",
            "attention",
        ), "train_fusion_step requires 'learned_gating', 'spatial_mlp', or 'attention'."
        assert (
            self.xrd_fusion_module is not None and self.fusion_optimizer is not None
        ), "Call set_guide_transforms / set_xrd_transforms before training."

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

        for i in range(n_transforms):
            tr_guide_coarse = self.xrd_fusion_module.get_tr()[i]
            if tr_guide_coarse.dim() == 3:
                tr_guide_coarse = tr_guide_coarse.unsqueeze(0)
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
                        hr_feats = cached_views[i]
                    else:
                        assert tr_guidance_img is not None
                        hr_feats = self._upsample_patch_map(
                            cached_views[i].float(),
                            (img_h, img_w),
                            guidance_img=tr_guidance_img,
                        )
                    if use_align:
                        logits_tr = head_mod.predict_logits(
                            hr_feats, guidance_hr=tr_guidance_img
                        )
                        logits_canon = self.inverse_transforms[i](logits_tr)
                        if isinstance(head_mod, SpatialConvMLPHead):
                            loss = head_mod.compute_loss_from_logits(
                                logits_canon,
                                xrd_map=canon_guide,
                                coarse_target=canon_guide,
                                aligner=self.guide_aligner,
                                guidance_hr=x.unsqueeze(0),
                            )
                        else:
                            loss = head_mod.compute_loss_from_logits(
                                logits_canon,
                                xrd_map=canon_guide,
                                coarse_target=canon_guide,
                                aligner=self.guide_aligner,
                            )
                    else:
                        if isinstance(head_mod, SpatialConvMLPHead):
                            loss = head_mod.compute_loss(
                                hr_feats,
                                tr_guide_coarse,
                                coarse_target=tr_guide_coarse,
                                guidance_hr=tr_guidance_img,
                            )
                        else:
                            loss = head_mod.compute_loss(
                                hr_feats,
                                tr_guide_coarse,
                                coarse_target=tr_guide_coarse,
                            )
                else:
                    loss = head_mod.compute_loss(cached_views[i], tr_guide_coarse)
                if i == 0:
                    params.extend(list(head_mod.parameters()))
            else:
                assert self.spatial_attention is not None
                feat_in = (
                    self._upsample_patch_map(
                        cached_views[i].float(),
                        (img_h, img_w),
                        guidance_img=tr_guidance_img,
                    )
                    if (want_hr and kind == "patch" and tr_guidance_img is not None)
                    else cached_views[i]
                )
                if use_align:
                    logits_tr = self.spatial_attention.predict_logits(
                        feat_in, tr_guide_coarse
                    )
                    logits_canon = self.inverse_transforms[i](logits_tr)
                    loss = self.spatial_attention.physics.compute_loss_from_logits(
                        logits_canon,
                        guide_map=canon_guide,
                        coarse_target=canon_guide,
                        aligner=self.guide_aligner,
                    )
                else:
                    loss = self.spatial_attention.compute_loss(
                        feat_in, tr_guide_coarse, aligner=None
                    )
                if i == 0:
                    params.extend(list(self.spatial_attention.parameters()))

            if use_align and update_aligner:
                loss = loss + self.guide_aligner.regularization_loss(
                    lr_hw=lr_hw, device=loss.device
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
            elif self.xrd_fuse_method == "spatial_mlp" and self.spatial_mlp:
                logits_view = self.spatial_mlp.predict_logits(
                    hr_feats, guidance_hr=transformed_img
                )
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
                    self.backbone_extractor.debiaser(permuted.float()),
                    tr_g,
                    loss_fn=self.loss_fn,
                    top_k=top_k,
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

    def _effective_reference_guide(self, guide_lr: torch.Tensor) -> torch.Tensor:
        """For spectral-cube models, return component-level coarse reference maps."""
        if (
            self._unmixed_guide_lr is not None
            and guide_lr.shape[1] != self.guide_channels
        ):
            return guide_to_tensor(
                self._unmixed_guide_lr,
                device=guide_lr.device,
                normalization="global_max",
            )
        return guide_lr

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
        """Produce calibrated high-resolution phase/element probabilities, tri-part
        pixel certainty, and optional Split Conformal ``prediction_sets``.
        """
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
        _, img_h, img_w = x.shape
        if (
            self.tile_size is not None
            and max(img_h, img_w) > self.tile_size
            and self.xrd_fuse_method in ("learned_gating", "spatial_mlp")
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

        phys = self._active_physics()
        if self.loss_fn == "compositional_kl" and phys is not None:
            probs = phys.logits_to_simplex_probs(mean_logits, target=ref_guide)
            if calibrate_to_guide and probs.shape[1] == ref_guide.shape[1] + 1:
                bg_log = torch.zeros_like(mean_logits[:, :1])
                probs = match_coarse_priors(
                    torch.cat([bg_log, mean_logits], dim=1),
                    ref_guide,
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
                match_coarse_priors(full_logits, ref_guide, mode="multiclass")
                if calibrate_to_guide
                else F.softmax(full_logits, dim=1)
            )
        else:
            probs = (
                match_coarse_priors(mean_logits, ref_guide, mode="multilabel")
                if calibrate_to_guide
                else torch.sigmoid(mean_logits)
            )

        lr_hw = (int(ref_guide.shape[-2]), int(ref_guide.shape[-1]))
        return build_uncertainty_report(
            probs_hr=probs,
            guide_lr=ref_guide,
            epistemic_var=epistemic_prob_var,
            mode="multiclass" if probs.shape[1] >= 2 and mode == "multiclass" else mode,
            abstain_threshold=abstain_threshold,
            downsampler=self._active_downsampler(),
            aligner=self.guide_aligner,
            calibrator=self.calibrator,
            diagnostics=self._build_diagnostics((img_h, img_w), lr_hw),
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
        guide_normalization: GuideNormalization | None = None,
        n_components: int | None = None,
        spectral_unmix: bool = False,
    ) -> list[float]:
        """High-level training entrypoint: normalizes and pads ``hr_image`` and
        ``coarse_guide`` (or raw 3D spectral cube when ``spectral_unmix=True``),
        caches frozen ViT TTA features, optimizes the fusion head (+ optional
        co-registration, PSF width, & spectral dictionary), and calibrates
        Split Conformal prediction sets when ``conformal_alpha`` is configured.
        """
        dev = next(self.dinov2.parameters()).device
        hr_prepared, _ = self._prepare_high_level_hr(hr_image)
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
        fwd, inv = build_default_tta(
            shift_dists=list(tta_shifts),
            pattern="Neumann",
            include_flips=tta_flips,
        )
        self.set_guide_transforms(
            guide_t,
            fwd,
            inv,
            n_components=n_components,
            spectral_unmix=spectral_unmix,
        )

        if self.xrd_fuse_method not in ("learned_gating", "spatial_mlp", "attention"):
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

        if self.calibrator is not None:
            ref_g = self._effective_reference_guide(guide_t)
            lr_hw = (int(ref_g.shape[-2]), int(ref_g.shape[-1]))
            cal_mask = ConformalCalibrator.make_spatial_holdout_mask(
                lr_hw, holdout_fraction=0.25
            )
            mean_logits, epi_var = self._predict_logits_and_var_tile(
                hr_prepared, guide_t
            )
            n_g = int(ref_g.shape[1])
            cal_mode: Literal["multiclass", "multilabel"] = (
                "multiclass"
                if n_g >= 2 or self.loss_fn == "compositional_kl"
                else "multilabel"
            )
            if cal_mode == "multiclass":
                bg_l = torch.zeros_like(mean_logits[:, :1])
                probs_cal = match_coarse_priors(
                    torch.cat([bg_l, mean_logits], dim=1), ref_g, mode="multiclass"
                )
            else:
                probs_cal = match_coarse_priors(mean_logits, ref_g, mode="multilabel")
            self.calibrator.calibrate_coarse(
                probs_hr=probs_cal,
                guide_lr=ref_g,
                epistemic_var_hr=epi_var,
                cal_mask_lr=cal_mask,
                downsampler=self._active_downsampler(),
                aligner=self.guide_aligner,
                mode=cal_mode,
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
    ) -> UncertaintyReport:
        """High-level inference entrypoint: predicts calibrated probabilities and
        an ``UncertaintyReport`` automatically cropped back to ``hr_image`` shape.
        """
        dev = next(self.dinov2.parameters()).device
        hr_prepared, (orig_h, orig_w) = self._prepare_high_level_hr(hr_image)
        guide_t: torch.Tensor | None = None
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
        elif self.xrd_fusion_module is not None:
            guide_t = self.xrd_fusion_module.xrd_image

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
        report = self.predict_with_uncertainty(
            hr_prepared,
            guide_lr=guide_t,
            mode=resolved_mode,
            calibrate_to_guide=calibrate_to_guide,
            abstain_threshold=abstain_threshold,
            top_k=top_k,
        )
        cropped = report.crop_spatial(orig_h, orig_w)
        if ref_g is not None:
            lr_hw = (int(ref_g.shape[-2]), int(ref_g.shape[-1]))
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
        guide_normalization: GuideNormalization | None = None,
        n_components: int | None = None,
        spectral_unmix: bool = False,
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
        )
        return self.predict(
            hr_image,
            coarse_guide=coarse_guide,
            mode=mode,
            calibrate_to_guide=calibrate_to_guide,
            abstain_threshold=abstain_threshold,
            guide_normalization=guide_normalization,
        )


# Backward-compatible class aliases
XFuse = CorrelativeFuser
