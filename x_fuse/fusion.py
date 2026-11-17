"""Physics-guided foundation-model feature fusion modules and the unified
``CorrelativeFuser`` (aliased as ``XFuse``) engine.

Upgrades over v0.1:
- Single-pass ViT backbone initialization (no double ``torch.hub`` loading).
- Vectorized multi-channel correlation/BCE gating across all ``C`` ViT channels
  and ``M`` physical guide channels simultaneously.
- Forward-PSF downsampling loss (``forward_psf_bce`` / ``forward_psf_mse``)
  supervising sharp HR phase predictions against coarse physical measurements.
- Device-safe and trainable ``XRDCrossAttention`` / ``SpatialCrossAttention``.
- Extract-once multi-channel fusion + online Welford TTA epistemic variance +
  optional pure-PyTorch Joint Bilateral Upsampling (JBU).
"""

from functools import partial
from typing import Literal, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import HighResBackbone, Interpolation
from .uncertainty import (
    UncertaintyReport,
    WelfordAccumulator,
    build_uncertainty_report,
    match_coarse_priors,
)
from .upsamplers import ForwardPSFDownsampler
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
    ):
        super().__init__()
        # Fall back cleanly when feat_dim % num_heads != 0 (e.g. ViT-T 192 / 8)
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
        self.psf_downsampler = ForwardPSFDownsampler()

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
        bsz, channels, height, width = attended.shape
        logits = self.pred_head(attended.permute(0, 2, 3, 1))
        return logits.permute(0, 3, 1, 2).reshape(
            bsz, self.guide_channels, height, width
        )

    def compute_loss(
        self, features: torch.Tensor, guide_map: torch.Tensor
    ) -> torch.Tensor:
        if guide_map.dim() == 2:
            guide_map = guide_map.unsqueeze(0).unsqueeze(0)
        elif guide_map.dim() == 3:
            guide_map = guide_map.unsqueeze(0)
        logits = self.predict_logits(features.float(), guide_map.float())
        target = guide_map.float().to(logits.device)
        if target.shape[0] == 1 and logits.shape[0] > 1:
            target = target.expand(logits.shape[0], -1, -1, -1)
        if target.shape[-2:] != logits.shape[-2:]:
            pred_coarse = self.psf_downsampler(logits, target.shape[-2:])
            return F.binary_cross_entropy_with_logits(pred_coarse, target)
        return F.binary_cross_entropy_with_logits(logits, target)


SpatialCrossAttention = XRDCrossAttention


class LearnedChannelGating(nn.Module):
    """Linear probe ranking ViT channels by physical guide relevance, supporting:
    - Single- or multi-channel guides (``out_channels = M``),
    - Class-balanced BCE, Pearson correlation, and FeatUp-inspired
      **Forward-PSF partial-volume losses** (``forward_psf_bce``, ``forward_psf_mse``),
    - Direct calibrated HR logit prediction ``predict_logits(features)`` as well
      as hard top-``K`` channel selection ``select_top_k(k)``.
    """

    def __init__(
        self,
        feat_dim: int = 384,
        loss_fn: LossOptions = "bce",
        out_channels: int = 1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.loss_fn = loss_fn
        self.out_channels = out_channels
        self.pred_head = nn.Linear(feat_dim, out_channels)
        self.psf_downsampler = ForwardPSFDownsampler()
        self._channel_mask: torch.Tensor | None = None

    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Predict ``(B, M, H, W)`` phase/element logits directly from features."""
        bsz, channels, height, width = features.shape
        feat_nhwc = features.float().permute(0, 2, 3, 1)  # (B, H, W, C)
        head_f32 = F.linear(
            feat_nhwc, self.pred_head.weight.float(), self.pred_head.bias.float()
        )
        return head_f32.permute(0, 3, 1, 2).reshape(
            bsz, self.out_channels, height, width
        )

    def compute_loss(
        self,
        features: torch.Tensor,
        xrd_map: torch.Tensor,
        coarse_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Train the linear probe against either spatially matched ``xrd_map`` or
        un-interpolated ``coarse_target`` via ``ForwardPSFDownsampler``."""
        bsz, _, height, width = features.shape
        logits = self.predict_logits(features)  # (B, M, H, W)

        ref = coarse_target if coarse_target is not None else xrd_map
        if ref.dim() == 2:
            ref = ref.unsqueeze(0).unsqueeze(0)
        elif ref.dim() == 3:
            ref = ref.unsqueeze(0)
        ref = ref.float().to(logits.device)
        if ref.shape[0] == 1 and bsz > 1:
            ref = ref.expand(bsz, -1, -1, -1)

        if self.loss_fn in ("forward_psf_bce", "forward_psf_mse"):
            pooled_logits = self.psf_downsampler(logits, ref.shape[-2:])
            if self.loss_fn == "forward_psf_mse":
                return F.mse_loss(torch.sigmoid(pooled_logits), ref)
            n_pos = ref.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
            n_neg = (1.0 - ref).sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
            pos_weight = (n_neg / n_pos).clamp(0.1, 20.0)
            return F.binary_cross_entropy_with_logits(
                pooled_logits, ref, pos_weight=pos_weight
            )

        if ref.shape[-2:] != (height, width):
            ref = F.interpolate(
                ref, size=(height, width), mode="bilinear", align_corners=False
            )

        pred_flat = logits.reshape(bsz * self.out_channels, -1)
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
    # Flatten across batch and spatial positions:
    # feat_mat: (C, N), guide_mat: (M, N) where N = B * H * W
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
        # Vectorized BCE channel scoring
        f_min = feat_mat.min(dim=1, keepdim=True).values
        f_max = feat_mat.max(dim=1, keepdim=True).values
        f_normed = ((feat_mat - f_min) / (f_max - f_min + 1e-8)).clamp(1e-6, 1.0 - 1e-6)
        log_f = torch.log(f_normed)  # (C, N)
        log_1mf = torch.log(1.0 - f_normed)  # (C, N)
        # Mean BCE(f_c, g_m) = -1/N * [g_m @ log_f.T + (1 - g_m) @ log_1mf.T]
        bce = -(guide_mat @ log_f.T + (1.0 - guide_mat) @ log_1mf.T) / max(n_pts, 1)
        scores = (1.0 - bce / 0.693).clamp(0.0, 1.0)

    # Zero all but top_k channels per guide channel (breaking ties deterministically
    # to keep exact k channels when scores are distinct)
    topk_vals, topk_idx = scores.topk(top_k, dim=1)
    thresholds = topk_vals[:, -1:]
    # If threshold > 0, keep scores >= threshold; if ties push count > top_k on distinct
    # random inputs or threshold == 0, scatter exact top_k indices when threshold > 0
    mask = torch.zeros_like(scores)
    mask.scatter_(1, topk_idx, 1.0)
    # Preserve original behaviour for all-zero BCE gates: 0 * 1 = 0 anyway!
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
        )  # (M, C)
        # Aggregate across M guide channels via max-envelope if M > 1
        gates = gates_mc.amax(dim=0)  # (C,)
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
    """

    def __init__(
        self,
        dino_name: str,
        xrd_fuse_method: FusionOptions = "learned_gating",
        stride: int = 4,
        pca_dim: int = -1,
        track_grad: bool = False,
        dtype: torch.dtype | int = torch.float16,
        loss_fn: LossOptions = "bce",
        guide_channels: int = 1,
        interpolation_mode: Interpolation = "nearest-exact",
        pretrained: bool = True,
        lr: float = 1e-3,
        **kwargs,
    ):
        device = kwargs.get("device", None)
        super().__init__(
            dino_name,
            stride=stride,
            pca_dim=pca_dim,
            dtype=dtype,
            track_grad=track_grad,
            pretrained=pretrained,
            interpolation_mode=interpolation_mode,
            **kwargs,
        )
        self.xrd_fuse_method: FusionOptions = xrd_fuse_method
        self.loss_fn: LossOptions = loss_fn
        self.guide_channels: int = guide_channels
        self.lr: float = lr

        self.learned_gating: LearnedChannelGating | None = None
        self.spatial_attention: XRDCrossAttention | None = None
        self.fusion_optimizer: torch.optim.Optimizer | None = None
        self._init_fusion_heads(guide_channels, device=device)

        self.xrd_fusion_module: XRDFusionMethod | None = None

    def _init_fusion_heads(
        self, guide_channels: int, device: str | torch.device | None = None
    ) -> None:
        self.guide_channels = guide_channels
        if self.xrd_fuse_method == "learned_gating":
            self.learned_gating = LearnedChannelGating(
                feat_dim=self.feat_dim,
                loss_fn=self.loss_fn,
                out_channels=guide_channels,
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
            )
            if device is not None:
                self.spatial_attention = self.spatial_attention.to(device)
            self.fusion_optimizer = torch.optim.Adam(
                self.spatial_attention.parameters(), lr=self.lr
            )

    def set_guide_transforms(
        self, guide_tensor: torch.Tensor, fwd: list[partial], inv: list[partial]
    ) -> None:
        """Bind a ``(1, M, H_lr, W_lr)`` physical guide tensor and TTA transforms."""
        super().set_transforms(fwd, inv)
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
        channels = (
            self.n_heads + self.feat_dim if attn_choice != "none" else self.feat_dim
        )
        n_patch_h, n_patch_w = self.get_n_patches(img_h, img_w)

        welford = WelfordAccumulator()
        n_transforms = len(self.transforms)

        for i in range(n_transforms):
            transformed_img = img_batch[i].unsqueeze(0)
            out_dict = self.dinov2.forward_feats_attn(  # type: ignore[operator]
                transformed_img, None, attn_choice
            )
            if attn_choice != "none":
                feats = torch.cat(
                    (out_dict["x_norm_patchtokens"], out_dict["x_patchattn"]), dim=-1
                )
            else:
                feats = out_dict["x_norm_patchtokens"]

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

    def train_fusion_step(self, x: torch.Tensor) -> float:
        """Update ``LearnedChannelGating`` or ``XRDCrossAttention`` weights for
        one optimization step across all TTA views while keeping the ViT frozen."""
        assert self.xrd_fuse_method in (
            "learned_gating",
            "attention",
        ), "train_fusion_step applies to 'learned_gating' or 'attention'."
        assert (
            self.xrd_fusion_module is not None and self.fusion_optimizer is not None
        ), "Call set_guide_transforms / set_xrd_transforms before training."

        x_in = x.to(self.dtype) if self.dtype != torch.float32 else x
        _, img_h, img_w = x_in.shape
        n_patch_h, n_patch_w = self.get_n_patches(img_h, img_w)

        with torch.no_grad():
            img_batch = self.get_transformed_input_batch(x_in, self.transforms)

        n_transforms = len(self.transforms)
        total_loss = 0.0
        self.fusion_optimizer.zero_grad()

        for i in range(n_transforms):
            with torch.no_grad():
                transformed_img = img_batch[i].unsqueeze(0)
                out_dict = self.dinov2.forward_feats_attn(  # type: ignore[operator]
                    transformed_img, None, "none"
                )
                features = out_dict["x_norm_patchtokens"].squeeze(0)
                permuted = (
                    features.reshape(n_patch_h, n_patch_w, self.feat_dim)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                )

            tr_guide_coarse = self.xrd_fusion_module.get_tr()[i]
            if tr_guide_coarse.dim() == 3:
                tr_guide_coarse = tr_guide_coarse.unsqueeze(0)

            if self.xrd_fuse_method == "learned_gating":
                assert self.learned_gating is not None
                if self.loss_fn in ("forward_psf_bce", "forward_psf_mse"):
                    hr_feats = self._upsample_patch_map(
                        permuted.float(), (img_h, img_w), guidance_img=transformed_img
                    )
                    loss = self.learned_gating.compute_loss(
                        hr_feats, tr_guide_coarse, coarse_target=tr_guide_coarse
                    )
                else:
                    loss = self.learned_gating.compute_loss(permuted, tr_guide_coarse)
                params = self.learned_gating.parameters()
            else:
                assert self.spatial_attention is not None
                loss = self.spatial_attention.compute_loss(permuted, tr_guide_coarse)
                params = self.spatial_attention.parameters()

            scaled_loss = loss / max(n_transforms, 1)
            scaled_loss.backward()
            total_loss += float(loss.detach().cpu().item())

        torch.nn.utils.clip_grad_norm_(list(params), max_norm=1.0)
        self.fusion_optimizer.step()
        return total_loss / max(n_transforms, 1)

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

        x_in = x.to(self.dtype) if self.dtype != torch.float32 else x.clone()
        img_batch = self.get_transformed_input_batch(x_in, self.transforms)
        _, img_h, img_w = x_in.shape
        n_patch_h, n_patch_w = self.get_n_patches(img_h, img_w)

        logit_welford = WelfordAccumulator()
        prob_welford = WelfordAccumulator()

        for i in range(len(self.transforms)):
            transformed_img = img_batch[i].unsqueeze(0)
            out_dict = self.dinov2.forward_feats_attn(  # type: ignore[operator]
                transformed_img, None, "none"
            )
            permuted = (
                out_dict["x_norm_patchtokens"]
                .squeeze(0)
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
                ).float()  # (M, C)
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

        if mode == "multiclass":
            # Add implicit background logit = 0 so M foreground phases + 1 background
            # compete cleanly via softmax!
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

        return build_uncertainty_report(
            probs_hr=probs,
            guide_lr=guide_lr,
            epistemic_var=epistemic_prob_var,
            mode=mode,
            abstain_threshold=abstain_threshold,
        )


# Backward-compatible class alias
XFuse = CorrelativeFuser
