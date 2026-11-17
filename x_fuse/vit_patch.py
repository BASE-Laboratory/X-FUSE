"""Vision Transformer stride, positional-encoding interpolation, and attention
extraction utilities.

Adapted from HR-Dv2 (https://github.com/tldr-group/HR-Dv2, MIT License) and
FeatUp (https://github.com/mhamilton723/FeatUp, MIT License).
"""

import math
import os
from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import torch
import torch.nn.functional as F

AttentionOptions: TypeAlias = Literal["q", "k", "v", "o", "none"]

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
    else:
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False


def get_qkvo_per_head(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    x_a: torch.Tensor,
    which: AttentionOptions,
    drop_fn: Callable,
    n_heads: int,
) -> torch.Tensor:
    """Extract per-head ``q``, ``k``, ``v``, or attention-output ``o`` summaries."""
    prims, mapping = [q, k, v], "qkv"
    if which in ("q", "k", "v"):
        prim = prims[mapping.index(which)]
        return prim.sum(dim=-1)
    if which == "o":
        if x_a.shape == q.shape:
            return drop_fn(x_a).sum(dim=-1)
        bsz, n_tok, _, _ = q.shape
        if x_a.shape == (bsz, n_tok, n_heads, n_tok):
            return drop_fn(x_a[:, 0, :, :]).permute(0, 2, 1).contiguous()
        raise ValueError(f"Unexpected x_a shape {tuple(x_a.shape)} for option 'o'")
    raise ValueError(f"Invalid attention option: {which}")


def drop_add_residual_stochastic_depth(
    x: torch.Tensor,
    residual_func: Callable[[torch.Tensor], torch.Tensor],
    sample_drop_ratio: float = 0.0,
) -> torch.Tensor:
    bsz, _, _ = x.shape
    sample_subset_size = max(int(bsz * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(bsz, device=x.device))[:sample_subset_size]
    x_subset = x[brange]
    residual = residual_func(x_subset)
    x_flat = x.flatten(1)
    residual_flat = residual.flatten(1)
    residual_scale_factor = bsz / sample_subset_size
    x_plus_residual = torch.index_add(
        x_flat, 0, brange, residual_flat.to(dtype=x.dtype), alpha=residual_scale_factor
    )
    return x_plus_residual.view_as(x)


class Patch:
    """Monkey-patch helpers enabling custom ViT stride and attention hooks."""

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: tuple[int, int]) -> Callable:
        def interpolate_pos_encoding(
            self, x: torch.Tensor, w: int, h: int
        ) -> torch.Tensor:
            previous_dtype = x.dtype
            npatch = x.shape[1] - 1
            n_orig = self.pos_embed.shape[1] - 1
            if npatch == n_orig and w == h:
                return self.pos_embed
            pos_embed = self.pos_embed.float()
            class_pos_embed = pos_embed[:, 0]
            patch_pos_embed = pos_embed[:, 1:]
            dim = x.shape[-1]
            w0: float = 1 + (w - patch_size) // stride_hw[1]
            h0: float = 1 + (h - patch_size) // stride_hw[0]
            assert w0 * h0 == npatch, (
                f"Grid size mismatch for {h}x{w} (patch={patch_size}, "
                f"stride={stride_hw}): got {h0}x{w0}={h0 * w0}, expected {npatch}"
            )
            w0, h0 = w0 + 0.1, h0 + 0.1
            side = int(math.sqrt(n_orig))
            patch_pos_embed = F.interpolate(
                patch_pos_embed.reshape(1, side, side, dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(n_orig), h0 / math.sqrt(n_orig)),
                mode="bicubic",
                align_corners=False,
                recompute_scale_factor=False,
            )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
                previous_dtype
            )

        return interpolate_pos_encoding

    @staticmethod
    def _fix_mem_eff_attn() -> Callable:
        def forward(
            self,
            x: torch.Tensor,
            attn_bias=None,
            attn_choice: AttentionOptions = "none",
        ) -> torch.Tensor:
            if not XFORMERS_AVAILABLE:
                if attn_bias is not None:
                    raise AssertionError("xFormers is required for nested tensors")
                if attn_choice == "none" and hasattr(self, "_original_forward"):
                    return self._original_forward(x)
                bsz, n_tok, channels = x.shape
                head_dim = channels // self.num_heads
                qkv = self.qkv(x).reshape(bsz, n_tok, 3, self.num_heads, head_dim)
                q, k, v = qkv.unbind(2)
                attn_out = F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                ).transpose(1, 2)
                out = self.proj_drop(self.proj(attn_out.reshape(bsz, n_tok, channels)))
                if attn_choice != "none":
                    n_h = getattr(self, "num_heads", getattr(self, "n_heads", 6))
                    extra = get_qkvo_per_head(
                        q, k, v, attn_out, attn_choice, self.attn_drop, n_h
                    )
                    out = torch.cat((out, extra), dim=-1)
                return out

            bsz, n_tok, channels = x.shape
            qkv = self.qkv(x).reshape(
                bsz, n_tok, 3, self.num_heads, channels // self.num_heads
            )
            q, k, v = unbind(qkv, 2)
            x_attn = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
            out = self.proj_drop(self.proj(x_attn.reshape([bsz, n_tok, channels])))
            if attn_choice != "none":
                n_h = getattr(self, "num_heads", getattr(self, "n_heads", 6))
                to_append = get_qkvo_per_head(
                    q, k, v, x_attn, attn_choice, self.attn_drop, n_h
                )
                out = torch.cat((out, to_append), dim=-1)
            return out

        return forward

    @staticmethod
    def _fix_block_forward_dino() -> Callable:
        def forward(
            self, x: torch.Tensor, attn_choice: AttentionOptions = "none"
        ) -> torch.Tensor:
            def attn_residual_func(inp: torch.Tensor) -> torch.Tensor:
                return self.attn(self.norm1(inp))

            def ffn_residual_func(inp: torch.Tensor) -> torch.Tensor:
                return self.mlp(self.norm2(inp))

            if attn_choice != "none":
                n_h = self.attn.num_heads
                ax = self.attn(self.norm1(x), attn_choice=attn_choice)
                xa, a = ax[:, :, :-n_h], ax[:, :, -n_h:]
                x = x + xa
                x = x + ffn_residual_func(x)
                return torch.cat((x, a), dim=-1)

            drop_ratio = getattr(self, "sample_drop_ratio", 0.0)
            if self.training and drop_ratio > 0.1:
                x = drop_add_residual_stochastic_depth(
                    x, residual_func=attn_residual_func, sample_drop_ratio=drop_ratio
                )
                x = drop_add_residual_stochastic_depth(
                    x, residual_func=ffn_residual_func, sample_drop_ratio=drop_ratio
                )
            else:
                x = x + attn_residual_func(x)
                x = x + ffn_residual_func(x)
            return x

        return forward

    @staticmethod
    def _fix_block_forward_dv2() -> Callable:
        def forward(
            self, x: torch.Tensor, attn_choice: AttentionOptions = "none"
        ) -> torch.Tensor:
            def attn_residual_func(inp: torch.Tensor) -> torch.Tensor:
                return self.ls1(self.attn(self.norm1(inp)))

            def ffn_residual_func(inp: torch.Tensor) -> torch.Tensor:
                return self.ls2(self.mlp(self.norm2(inp)))

            if attn_choice != "none":
                n_h = self.attn.num_heads
                ax = self.attn(self.norm1(x), attn_choice=attn_choice)
                xa, a = ax[:, :, :-n_h], ax[:, :, -n_h:]
                x = x + self.ls1(xa)
                x = x + ffn_residual_func(x)
                return torch.cat((x, a), dim=-1)

            drop_ratio = getattr(self, "sample_drop_ratio", 0.0)
            if self.training and drop_ratio > 0.1:
                x = drop_add_residual_stochastic_depth(
                    x, residual_func=attn_residual_func, sample_drop_ratio=drop_ratio
                )
                x = drop_add_residual_stochastic_depth(
                    x, residual_func=ffn_residual_func, sample_drop_ratio=drop_ratio
                )
            else:
                x = x + attn_residual_func(x)
                x = x + ffn_residual_func(x)
            return x

        return forward

    @staticmethod
    def _add_new_forward_features_dino() -> Callable:
        def forward_feats_attn(
            self, x, masks=None, attn_choice: AttentionOptions = "none"
        ):
            x = self.prepare_tokens(x)
            for i, blk in enumerate(self.blocks):
                if i < len(self.blocks) - 1:
                    x = blk(x)
                else:
                    x = blk(x, attn_choice=attn_choice)

            if attn_choice != "none":
                x_feats = x[:, :, : -self.num_heads]
                x_attn = x[:, :, -self.num_heads :]
            else:
                x_feats = x

            n_reg = getattr(self, "num_register_tokens", 0)
            x_norm = self.norm(x_feats)
            out_dict = {
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_regtokens": x_norm[:, 1 : n_reg + 1],
                "x_norm_patchtokens": x_norm[:, n_reg + 1 :],
                "x_prenorm": x,
                "masks": masks,
            }
            if attn_choice != "none":
                out_dict["x_patchattn"] = x_attn[:, n_reg + 1 :]
            return out_dict

        return forward_feats_attn

    @staticmethod
    def _add_new_forward_features_dv2() -> Callable:
        def forward_feats_attn(
            self, x, masks=None, attn_choice: AttentionOptions = "none"
        ):
            x = self.prepare_tokens_with_masks(x, masks)
            for i, blk in enumerate(self.blocks):
                if i < len(self.blocks) - 1:
                    x = blk(x)
                else:
                    x = blk(x, attn_choice=attn_choice)

            if attn_choice != "none":
                x_feats = x[:, :, : -self.num_heads]
                x_attn = x[:, :, -self.num_heads :]
            else:
                x_feats = x

            n_reg = getattr(self, "num_register_tokens", 0)
            x_norm = self.norm(x_feats)
            out_dict = {
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_regtokens": x_norm[:, 1 : n_reg + 1],
                "x_norm_patchtokens": x_norm[:, n_reg + 1 :],
                "x_prenorm": x,
                "masks": masks,
            }
            if attn_choice != "none":
                out_dict["x_patchattn"] = x_attn[:, n_reg + 1 :]
            return out_dict

        return forward_feats_attn

    @staticmethod
    def _add_new_forward_features_dv3(model: Any) -> Callable:
        def forward_feats_attn(
            self, x, masks=None, attn_choice: AttentionOptions = "none"
        ):
            return model.forward_features(x)

        return forward_feats_attn

    @staticmethod
    def _add_new_forward_features_vit() -> Callable:
        def forward_feats_attn(
            self, x, masks=None, attn_choice: AttentionOptions = "none"
        ):
            _, _, w, h = x.shape
            x = self.patch_embed(x)
            if x.ndim == 4:
                # timm dynamic_img_size returns (B, H_p, W_p, C)
                x = x.flatten(1, 2)
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            if hasattr(self, "interpolate_pos_encoding"):
                x = x + self.interpolate_pos_encoding(x, w, h)

            for i, blk in enumerate(self.blocks):
                if i < len(self.blocks) - 1:
                    x = blk(x)
                else:
                    x = blk(x, attn_choice=attn_choice)

            if attn_choice != "none":
                x_feats = x[:, :, : -self.num_heads]
                x_attn = x[:, :, -self.num_heads :]
            else:
                x_feats = x

            n_reg = getattr(self, "num_register_tokens", 0)
            x_norm = self.norm(x_feats)
            out_dict = {
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_regtokens": x_norm[:, 1 : n_reg + 1],
                "x_norm_patchtokens": x_norm[:, n_reg + 1 :],
                "x_prenorm": x,
                "masks": masks,
            }
            if attn_choice != "none":
                out_dict["x_patchattn"] = x_attn[:, n_reg + 1 :]
            return out_dict

        return forward_feats_attn


class TorchPCA:
    """Fast GPU/CPU low-rank PCA on PyTorch tensors (adapted from FeatUp)."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_: torch.Tensor | None = None
        self.components_: torch.Tensor | None = None
        self.singular_values_: torch.Tensor | None = None

    def fit(self, x: torch.Tensor) -> "TorchPCA":
        self.mean_ = x.mean(dim=0)
        unbiased = x - self.mean_.unsqueeze(0)
        _, s_vals, v_mat = torch.pca_lowrank(
            unbiased, q=self.n_components, center=False, niter=4
        )
        self.components_ = v_mat.T
        self.singular_values_ = s_vals
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.mean_ is not None and self.components_ is not None
        t0 = x.float() - self.mean_.to(x.device).float().unsqueeze(0)
        return t0 @ self.components_.to(x.device).float().T


def torch_pca(
    feature_img: torch.Tensor,
    dim: int = 3,
    fit_pca: TorchPCA | None = None,
    max_samples: int = 20000,
) -> torch.Tensor:
    """Project a ``(C, H, W)`` or ``(1, C, H, W)`` feature map to ``(H*W, dim)``."""
    if feature_img.ndim == 4:
        feature_img = feature_img.squeeze(0)
    channels, height, width = feature_img.shape
    n_pix = height * width
    flat = feature_img.reshape(channels, n_pix).permute(1, 0).float()

    if max_samples is not None and n_pix > max_samples:
        generator = torch.Generator(device=flat.device).manual_seed(0)
        indices = torch.randperm(n_pix, generator=generator, device=flat.device)[
            :max_samples
        ]
        sample = flat[indices]
    else:
        sample = flat

    if fit_pca is None:
        fit_pca = TorchPCA(dim).fit(sample)
    return fit_pca.transform(flat)
