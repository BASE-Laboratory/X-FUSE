"""Composable high-resolution fusion and phase-prediction heads for X-FUSE:

1. ``LinearProbeHead`` (aliased as ``LearnedChannelGating``): Fast pixel-wise
   linear projection + channel ranking supervised via ``PhysicsForwardModel``.
2. ``SpatialConvMLPHead`` (``fusion_method="spatial_mlp"``): Nonlinear bottleneck
   MLP + ``3x3`` depthwise-separable spatial context residual + multi-scale HR
   texture skip connection + optional edge-aware Total Variation regularization.
3. ``WindowedGuideCrossAttentionHead`` (aliased as ``XRDCrossAttention`` /
   ``SpatialCrossAttention``): Memory-safe ``O(H * W)`` pooled-key cross-attention
   replacing the legacy ``O((H * W)^2)`` global attention bottleneck so
   ``512x512+`` micrographs run in ``< 80 MB`` RAM.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Literal, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics import LossOptions, PhysicsForwardModel
from .registration import DifferentiableGuideAligner
from .upsamplers import (
    ForwardPSFDownsampler,
    PSFProfile,
    extract_multiscale_texture_descriptors,
    extract_structure_tensor_field,
)

FusionOptions: TypeAlias = Literal[
    "gating", "learned_gating", "spatial_mlp", "attention"
]

_HR_DESC_CACHE: OrderedDict[
    tuple[bytes, tuple[int, ...], str, bool],
    tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
    ],
] = OrderedDict()
_HR_DESC_CACHE_MAX = 16


def _get_hr_descriptors(
    g_dev: torch.Tensor, orientation_aware: bool
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
]:
    """Extract (or fetch from bounded LRU cache when ``not g_dev.requires_grad``)
    ``(tex_8ch, spin2_4ch, coh_2ch, gy_edge, gx_edge)`` for ``g_dev``.
    """
    g_f32 = g_dev.float()
    can_cache = not g_f32.requires_grad and g_f32.device.type == "cpu"
    key: tuple[bytes, tuple[int, ...], str, bool] | None = None
    if can_cache:
        g_c = g_f32.detach().contiguous()
        digest = hashlib.blake2b(g_c.numpy().tobytes(), digest_size=12).digest()
        key = (digest, tuple(g_c.shape), bool(orientation_aware))  # type: ignore[assignment]
        cached = _HR_DESC_CACHE.get(key)  # type: ignore[arg-type]
        if cached is not None:
            _HR_DESC_CACHE.move_to_end(key)  # type: ignore[arg-type]
            return cached

    desc = extract_multiscale_texture_descriptors(g_f32)
    edge_sig = desc[:, [0, 5, 7]]
    gy = (
        (edge_sig[:, :, 1:, :] - edge_sig[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
    )
    gx = (
        (edge_sig[:, :, :, 1:] - edge_sig[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
    )
    spin2: torch.Tensor | None = None
    coh: torch.Tensor | None = None
    if orientation_aware:
        spin2 = extract_structure_tensor_field(g_f32)
        coh = torch.cat(
            [
                torch.hypot(spin2[:, 0:1], spin2[:, 1:2]),
                torch.hypot(spin2[:, 2:3], spin2[:, 3:4]),
            ],
            dim=1,
        )
        sy = (spin2[:, :, 1:, :] - spin2[:, :, :-1, :]).norm(dim=1, keepdim=True)
        sx = (spin2[:, :, :, 1:] - spin2[:, :, :, :-1]).norm(dim=1, keepdim=True)
        gy = gy + 0.5 * sy
        gx = gx + 0.5 * sx

    out = (desc, spin2, coh, gy, gx)
    if can_cache and key is not None:
        _HR_DESC_CACHE[key] = out
        if len(_HR_DESC_CACHE) > _HR_DESC_CACHE_MAX:
            _HR_DESC_CACHE.popitem(last=False)
    return out


def edge_aware_tv_loss(
    logits: torch.Tensor,
    guidance_hr: torch.Tensor | None = None,
    edge_beta: float = 10.0,
    simplex: bool = False,
    orientation_aware: bool = True,
) -> torch.Tensor:
    """Anisotropic edge-aware Total Variation prior on HR ``logits`` ``(B, M, H, W)``.
    Smooths noisy sub-patch predictions inside homogeneous grains while relaxing
    smoothing across strong structural intensity, texture, or spin-2 orientation
    boundaries in ``I_HR``.
    """
    if simplex:
        bg = torch.zeros_like(logits[:, :1], dtype=torch.float32)
        probs = F.softmax(torch.cat([bg, logits.float()], dim=1), dim=1)
    else:
        probs = torch.sigmoid(logits.float())
    dy = (probs[:, :, 1:, :] - probs[:, :, :-1, :]).abs()
    dx = (probs[:, :, :, 1:] - probs[:, :, :, :-1]).abs()

    if guidance_hr is not None:
        if guidance_hr.ndim == 3:
            guidance_hr = guidance_hr.unsqueeze(0)
        g_dev = guidance_hr.to(logits.device)
        _, _, _, gy, gx = _get_hr_descriptors(g_dev, orientation_aware)
        wy = torch.exp(-edge_beta * gy)
        wx = torch.exp(-edge_beta * gx)
        return (wy * dy).mean() + (wx * dx).mean()

    return dy.mean() + dx.mean()


class LinearProbeHead(nn.Module):
    """Linear probe ranking ViT channels by physical guide relevance and
    predicting calibrated HR logits supervised through ``PhysicsForwardModel``.
    """

    def __init__(
        self,
        feat_dim: int = 384,
        loss_fn: LossOptions = "bce",
        out_channels: int = 1,
        learn_psf: bool = False,
        spectral_channels: int | None = None,
        psf_profile: PSFProfile = "gaussian",
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.loss_fn: LossOptions = loss_fn
        self.out_channels = max(1, int(out_channels))
        self.pred_head = nn.Linear(self.feat_dim, self.out_channels)
        self.physics = PhysicsForwardModel(
            n_channels=self.out_channels,
            loss_fn=loss_fn,
            learn_psf=learn_psf,
            spectral_channels=spectral_channels,
            psf_profile=psf_profile,
        )
        self._channel_mask: torch.Tensor | None = None

    @property
    def psf_downsampler(self) -> ForwardPSFDownsampler:
        return self.physics.psf_downsampler

    def predict_logits(
        self, features: torch.Tensor, guidance_hr: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict ``(B, M, H, W)`` phase/element logits directly from features."""
        bsz, _, height, width = features.shape
        feat_nhwc = features.float().permute(0, 2, 3, 1)
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
        valid_hr_hw: tuple[int, int] | None = None,
        loss_mask_lr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.physics.compute_loss_from_logits(
            logits,
            guide_map=xrd_map,
            coarse_target=coarse_target,
            aligner=aligner,
            valid_hr_hw=valid_hr_hw,
            loss_mask_lr=loss_mask_lr,
        )

    def compute_loss(
        self,
        features: torch.Tensor,
        xrd_map: torch.Tensor,
        coarse_target: torch.Tensor | None = None,
        aligner: DifferentiableGuideAligner | None = None,
        valid_hr_hw: tuple[int, int] | None = None,
        loss_mask_lr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.predict_logits(features)
        return self.compute_loss_from_logits(
            logits,
            xrd_map=xrd_map,
            coarse_target=coarse_target,
            aligner=aligner,
            valid_hr_hw=valid_hr_hw,
            loss_mask_lr=loss_mask_lr,
        )

    def get_channel_importance(self) -> torch.Tensor:
        """Return ``(C,)`` channel importance aggregated across all ``M`` outputs."""
        return self.pred_head.weight.detach().float().norm(dim=0)

    def select_top_k(self, k: int) -> None:
        scores = self.pred_head.weight.data.abs().amax(dim=0)
        k_clamped = min(max(int(k), 1), self.feat_dim)
        topk_idx = scores.topk(k_clamped).indices
        mask = torch.zeros(self.feat_dim, device=scores.device, dtype=torch.float32)
        mask[topk_idx] = 1.0
        self._channel_mask = mask

    def forward(
        self, features: torch.Tensor, _xrd_map: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._channel_mask is None:
            return features
        mask = self._channel_mask.to(device=features.device, dtype=features.dtype).view(
            1, -1, 1, 1
        )
        return features * mask


class SpatialConvMLPHead(nn.Module):
    """Nonlinear spatial fusion head (``fusion_method="spatial_mlp"``) combining
    ViT semantic features with multi-scale HR texture & anisotropy coherence
    descriptors via a bottleneck residual depthwise-separable Conv-MLP block and
    optional spin-2 edge-aware TV prior.
    """

    def __init__(
        self,
        feat_dim: int = 384,
        hidden_dim: int = 64,
        out_channels: int = 1,
        loss_fn: LossOptions = "forward_psf_bce",
        learn_psf: bool = False,
        spectral_channels: int | None = None,
        tv_weight: float = 1e-3,
        guide_residual: bool = False,
        psf_profile: PSFProfile = "gaussian",
        orientation_aware: bool = True,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_channels = max(1, int(out_channels))
        self.loss_fn: LossOptions = loss_fn
        self.tv_weight = float(tv_weight)
        self.guide_residual = bool(guide_residual)
        self.orientation_aware = bool(orientation_aware)

        n_groups = 8 if (self.hidden_dim > 8 and self.hidden_dim % 8 == 0) else 1
        self.feat_norm = nn.GroupNorm(1, self.feat_dim)
        self.in_proj = nn.Conv2d(self.feat_dim, self.hidden_dim, kernel_size=1)
        self.tex_norm = nn.GroupNorm(8, 8)
        self.tex_stem = nn.Sequential(
            nn.Conv2d(8, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1),
        )
        self.coh_stem = nn.Conv2d(2, self.hidden_dim, kernel_size=1)
        nn.init.zeros_(self.coh_stem.weight)
        if self.coh_stem.bias is not None:
            nn.init.zeros_(self.coh_stem.bias)
        self.norm1 = nn.GroupNorm(n_groups, self.hidden_dim)
        self.dw_conv = nn.Conv2d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=3,
            padding=1,
            groups=self.hidden_dim,
        )
        self.pw_conv = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.norm2 = nn.GroupNorm(n_groups, self.hidden_dim)
        self.out_proj = nn.Conv2d(self.hidden_dim * 2, self.out_channels, kernel_size=1)
        self.pixel_proj = nn.Conv2d(8, self.out_channels, kernel_size=1)
        nn.init.zeros_(self.pixel_proj.weight)
        if self.pixel_proj.bias is not None:
            nn.init.zeros_(self.pixel_proj.bias)
        self.coh_proj = nn.Conv2d(2, self.out_channels, kernel_size=1)
        nn.init.zeros_(self.coh_proj.weight)
        if self.coh_proj.bias is not None:
            nn.init.zeros_(self.coh_proj.bias)
        self.guide_alpha: nn.Parameter | None = None
        if self.guide_residual:
            self.guide_alpha = nn.Parameter(torch.ones(1, self.out_channels, 1, 1))
            nn.init.zeros_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)

        self.physics = PhysicsForwardModel(
            n_channels=self.out_channels,
            loss_fn=loss_fn,
            learn_psf=learn_psf,
            spectral_channels=spectral_channels,
            psf_profile=psf_profile,
        )

    @property
    def psf_downsampler(self) -> ForwardPSFDownsampler:
        return self.physics.psf_downsampler

    def _encode_hidden(
        self, features: torch.Tensor, guidance_hr: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        mod_dtype = self.in_proj.weight.dtype
        bsz, _, height, width = features.shape
        if not features.requires_grad and self.feat_norm.weight is not None:
            with torch.no_grad():
                x_hat = F.group_norm(
                    features.float(), 1, None, None, self.feat_norm.eps
                )
            w_eff = self.in_proj.weight.float() * self.feat_norm.weight.float().view(
                1, -1, 1, 1
            )
            b_eff = self.in_proj.bias.float() if self.in_proj.bias is not None else None
            if self.feat_norm.bias is not None:
                b_shift = (
                    self.in_proj.weight.float().squeeze(-1).squeeze(-1)
                    @ self.feat_norm.bias.float()
                )
                b_eff = b_shift if b_eff is None else (b_eff + b_shift)
            f_h = F.conv2d(x_hat, w_eff, b_eff).to(mod_dtype)
        else:
            f_h = self.in_proj(self.feat_norm(features.to(mod_dtype)))
        tex_n: torch.Tensor | None = None
        coh: torch.Tensor | None = None
        if guidance_hr is not None:
            if guidance_hr.ndim == 3:
                guidance_hr = guidance_hr.unsqueeze(0)
            g_resized = (
                F.interpolate(
                    guidance_hr.float(),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
                if guidance_hr.shape[-2:] != (height, width)
                else guidance_hr.float()
            )
            g_dev = g_resized.to(features.device)
            tex_f, _, coh_raw, _, _ = _get_hr_descriptors(g_dev, self.orientation_aware)
            tex = tex_f.to(mod_dtype)
            if tex.shape[0] == 1 and bsz > 1:
                tex = tex.expand(bsz, -1, -1, -1)
            if tex.shape[0] * height * width > 1:
                tex_n = self.tex_norm(tex)
            else:
                tex_n = tex * 0.0 + self.tex_norm.bias.to(mod_dtype).view(1, -1, 1, 1)
            t_h = self.tex_stem(tex_n)
            if self.orientation_aware and coh_raw is not None:
                coh_m = coh_raw.to(mod_dtype)
                coh = (
                    coh_m.expand(bsz, -1, -1, -1)
                    if (coh_m.shape[0] == 1 and bsz > 1)
                    else coh_m
                )
                t_h = t_h + self.coh_stem(coh)
        else:
            t_h = torch.zeros_like(f_h)

        h = F.gelu(self.norm1(f_h + t_h))
        res = self.pw_conv(F.gelu(self.dw_conv(h)))
        return F.gelu(self.norm2(h + res)), t_h, tex_n, coh

    def predict_logits(
        self, features: torch.Tensor, guidance_hr: torch.Tensor | None = None
    ) -> torch.Tensor:
        h_out, t_h, tex_n, coh = self._encode_hidden(features, guidance_hr=guidance_hr)
        logits = self.out_proj(torch.cat([h_out, t_h], dim=1))
        if tex_n is not None:
            logits = logits + self.pixel_proj(tex_n)
        if coh is not None:
            logits = logits + self.coh_proj(coh)
        return logits.float()

    def apply_guide_residual(
        self,
        logits_canon: torch.Tensor,
        canon_guide: torch.Tensor | None,
        valid_hr_hw: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Anchor canonical structural logits to the smooth coarse-guide prior so
        macroscopic intra-grain chemical zonation invisible in ``I_HR`` is preserved
        while ``SpatialConvMLPHead`` learns high-frequency boundary deconvolution.
        """
        if not self.guide_residual or canon_guide is None or self.guide_alpha is None:
            return logits_canon
        g = canon_guide.float().to(logits_canon.device)
        if g.ndim == 2:
            g = g.unsqueeze(0).unsqueeze(0)
        elif g.ndim == 3:
            g = g.unsqueeze(0)
        if g.shape[1] != self.out_channels:
            if (
                self.physics.spectral_dict is not None
                and g.shape[1] == self.physics.spectral_dict.spectral_channels
            ):
                g = self.physics.spectral_dict.project_cube(g)
            else:
                g = g[:, : self.out_channels]
        full_h, full_w = int(logits_canon.shape[-2]), int(logits_canon.shape[-1])
        oh, ow = valid_hr_hw if valid_hr_hw is not None else (full_h, full_w)
        g_hr = F.interpolate(g, size=(oh, ow), mode="bilinear", align_corners=False)
        if self.loss_fn == "compositional_kl":
            bg_hr = (1.0 - g_hr.sum(dim=1, keepdim=True)).clamp(1e-4, 1.0)
            prior = torch.log(g_hr.clamp(1e-4, 1.0) / bg_hr)
        else:
            prior = torch.logit(g_hr.clamp(1e-4, 1.0 - 1e-4))
        scaled_prior = self.guide_alpha.float().to(logits_canon.device) * prior
        if (oh, ow) != (full_h, full_w):
            scaled_prior = F.pad(
                scaled_prior, (0, full_w - ow, 0, full_h - oh), mode="replicate"
            )
        return logits_canon.float() + scaled_prior

    def compute_loss_from_logits(
        self,
        logits: torch.Tensor,
        xrd_map: torch.Tensor,
        coarse_target: torch.Tensor | None = None,
        aligner: DifferentiableGuideAligner | None = None,
        guidance_hr: torch.Tensor | None = None,
        valid_hr_hw: tuple[int, int] | None = None,
        loss_mask_lr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base_loss = self.physics.compute_loss_from_logits(
            logits,
            guide_map=xrd_map,
            coarse_target=coarse_target,
            aligner=aligner,
            valid_hr_hw=valid_hr_hw,
            loss_mask_lr=loss_mask_lr,
        )
        tv_logits = (
            logits[:, :, : valid_hr_hw[0], : valid_hr_hw[1]]
            if valid_hr_hw is not None
            else logits
        )
        tv_guidance = (
            guidance_hr[..., : valid_hr_hw[0], : valid_hr_hw[1]]
            if (guidance_hr is not None and valid_hr_hw is not None)
            else guidance_hr
        )
        if self.tv_weight > 0.0 and min(tv_logits.shape[-2], tv_logits.shape[-1]) >= 4:
            base_loss = base_loss + self.tv_weight * edge_aware_tv_loss(
                tv_logits,
                guidance_hr=tv_guidance,
                simplex=(self.loss_fn == "compositional_kl"),
                orientation_aware=self.orientation_aware,
            )
        return base_loss

    def compute_loss(
        self,
        features: torch.Tensor,
        xrd_map: torch.Tensor,
        coarse_target: torch.Tensor | None = None,
        aligner: DifferentiableGuideAligner | None = None,
        guidance_hr: torch.Tensor | None = None,
        valid_hr_hw: tuple[int, int] | None = None,
        loss_mask_lr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.predict_logits(features, guidance_hr=guidance_hr)
        logits = self.apply_guide_residual(logits, xrd_map, valid_hr_hw=valid_hr_hw)
        return self.compute_loss_from_logits(
            logits,
            xrd_map=xrd_map,
            coarse_target=coarse_target,
            aligner=aligner,
            guidance_hr=guidance_hr,
            valid_hr_hw=valid_hr_hw,
            loss_mask_lr=loss_mask_lr,
        )

    def forward(
        self,
        features: torch.Tensor,
        xrd_map: torch.Tensor | None = None,
        guidance_hr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_feat = self.in_proj.weight.detach().float()[:, :, 0, 0]
        gates = w_feat.norm(dim=0)
        gates = gates / gates.max().clamp(min=1e-8)
        return features * gates.to(device=features.device, dtype=features.dtype).view(
            1, -1, 1, 1
        )


class WindowedGuideCrossAttentionHead(nn.Module):
    """Memory-safe ``O(H * W)`` pooled-key cross-attention head replacing the
    legacy ``O((H * W)^2)`` global attention bottleneck.

    Caps the spatial key/value token grid to at most ``max_kv_grid x max_kv_grid``
    (default ``16 x 16 = 256`` tokens) via adaptive average pooling and blends
    the attended semantic tokens with a local ``3x3`` residual feature skip,
    running comfortably on ``512x512``–``2048x2048`` micrographs in ``< 80 MB`` RAM.
    """

    def __init__(
        self,
        feat_dim: int = 384,
        num_heads: int = 8,
        guide_channels: int = 1,
        loss_fn: LossOptions = "bce",
        learn_psf: bool = False,
        spectral_channels: int | None = None,
        max_kv_grid: int = 16,
        psf_profile: PSFProfile = "gaussian",
    ) -> None:
        super().__init__()
        while feat_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.feat_dim = int(feat_dim)
        self.num_heads = int(num_heads)
        self.guide_channels = max(1, int(guide_channels))
        self.loss_fn: LossOptions = loss_fn
        self.max_kv_grid = max(4, int(max_kv_grid))

        self.feat_norm = nn.GroupNorm(1, self.feat_dim)
        self.xrd_to_query = nn.Linear(self.guide_channels, self.feat_dim)
        self.feat_query_proj = nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=1)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.feat_dim,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=0.05,
        )
        self.local_refine = nn.Conv2d(
            self.feat_dim, self.feat_dim, kernel_size=3, padding=1, groups=self.feat_dim
        )
        self.pred_head = nn.Linear(self.feat_dim, self.guide_channels)
        self.physics = PhysicsForwardModel(
            n_channels=self.guide_channels,
            loss_fn=loss_fn,
            learn_psf=learn_psf,
            spectral_channels=spectral_channels,
            psf_profile=psf_profile,
        )

    @property
    def psf_downsampler(self) -> ForwardPSFDownsampler:
        return self.physics.psf_downsampler

    def _prepare_guide(
        self, guide_map: torch.Tensor, target_hw: tuple[int, int]
    ) -> torch.Tensor:
        if guide_map.dim() == 2:
            guide_map = guide_map.unsqueeze(0).unsqueeze(0)
        elif guide_map.dim() == 3:
            guide_map = guide_map.unsqueeze(0)
        g = guide_map.float()
        # Handle spectral cube inputs by projecting/averaging to guide_channels
        if g.shape[1] != self.guide_channels:
            if self.physics.spectral_dict is not None and (
                g.shape[1] == self.physics.spectral_dict.spectral_channels
            ):
                spec = self.physics.spectral_dict.spectra().detach().to(g.device)
                g = torch.einsum("behw,me->bmhw", g, spec)
            else:
                g = g[:, : self.guide_channels]
        if g.shape[-2:] != target_hw:
            g = F.interpolate(g, size=target_hw, mode="bilinear", align_corners=False)
        return g

    def forward(self, features: torch.Tensor, xrd_map: torch.Tensor) -> torch.Tensor:
        """Modulate ``features`` ``(B, C, H, W)`` in ``O(H * W * K_pool)`` memory."""
        bsz, channels, height, width = features.shape
        orig_dtype = features.dtype
        mod_dtype = self.feat_query_proj.weight.dtype
        feat_raw = features.to(mod_dtype)
        feat_f = (
            self.feat_norm(feat_raw)
            if feat_raw.shape[0] * height * width > 1
            else feat_raw
        )

        guide = self._prepare_guide(xrd_map, (height, width)).to(
            device=features.device, dtype=mod_dtype
        )
        guide_flat = guide.reshape(guide.shape[0], guide.shape[1], -1).permute(0, 2, 1)
        q_guide = self.xrd_to_query(guide_flat)
        if q_guide.shape[0] == 1 and bsz > 1:
            q_guide = q_guide.expand(bsz, -1, -1)

        q_feat = (
            self.feat_query_proj(feat_f).reshape(bsz, channels, -1).permute(0, 2, 1)
        )
        queries = q_feat + q_guide  # (B, H*W, C)

        # Cap key/value tokens via spatial adaptive pooling -> strictly O(H*W)!
        kv_h = min(height, self.max_kv_grid)
        kv_w = min(width, self.max_kv_grid)
        kv_spatial = (
            F.adaptive_avg_pool2d(feat_f, output_size=(kv_h, kv_w))
            if (kv_h, kv_w) != (height, width)
            else feat_f
        )
        kv_flat = kv_spatial.reshape(bsz, channels, -1).permute(0, 2, 1)

        attended, _ = self.attention(queries, kv_flat, kv_flat, need_weights=False)
        attended_2d = attended.permute(0, 2, 1).reshape(bsz, channels, height, width)
        out = feat_f + 0.5 * attended_2d + 0.5 * self.local_refine(feat_f)
        return out.to(orig_dtype)

    def predict_logits(
        self, features: torch.Tensor, guide_map: torch.Tensor
    ) -> torch.Tensor:
        mod_dtype = self.pred_head.weight.dtype
        attended = self.forward(features.to(mod_dtype), guide_map)
        bsz, _, height, width = attended.shape
        logits = self.pred_head(attended.permute(0, 2, 3, 1)).float()
        return logits.permute(0, 3, 1, 2).reshape(
            bsz, self.guide_channels, height, width
        )

    def compute_loss_from_logits(
        self,
        logits: torch.Tensor,
        guide_map: torch.Tensor,
        coarse_target: torch.Tensor | None = None,
        aligner: DifferentiableGuideAligner | None = None,
        valid_hr_hw: tuple[int, int] | None = None,
        loss_mask_lr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.physics.compute_loss_from_logits(
            logits,
            guide_map=guide_map,
            coarse_target=coarse_target if coarse_target is not None else guide_map,
            aligner=aligner,
            valid_hr_hw=valid_hr_hw,
            loss_mask_lr=loss_mask_lr,
        )

    def compute_loss(
        self,
        features: torch.Tensor,
        guide_map: torch.Tensor,
        coarse_target: torch.Tensor | None = None,
        aligner: DifferentiableGuideAligner | None = None,
        valid_hr_hw: tuple[int, int] | None = None,
        loss_mask_lr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.predict_logits(features.float(), guide_map.float())
        return self.compute_loss_from_logits(
            logits,
            guide_map=guide_map,
            coarse_target=coarse_target,
            aligner=aligner,
            valid_hr_hw=valid_hr_hw,
            loss_mask_lr=loss_mask_lr,
        )


# Backward-compatible class aliases
LearnedChannelGating = LinearProbeHead
XRDCrossAttention = WindowedGuideCrossAttentionHead
SpatialCrossAttention = WindowedGuideCrossAttentionHead
