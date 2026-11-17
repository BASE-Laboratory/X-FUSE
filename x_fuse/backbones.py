"""Self-contained Vision Transformer backbone loader and dense TTA feature
extractor (replacing the external HR-Dv2 dependency and fixing double loading).
"""

import re
from collections.abc import Sequence
from functools import partial
from types import MethodType
from typing import Any, Literal, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from torch.nn.modules.utils import _pair

from .debias import ApositionalFeatureDebiaser
from .tiling import run_tiled_2d
from .transforms import iden_partial
from .upsamplers import CrossAttentionFeatureUpsampler, JointBilateralUpsampler
from .utils import get_alibi_model, get_dv3_model
from .vit_patch import AttentionOptions, Patch

Interpolation: TypeAlias = Literal[
    "nearest",
    "linear",
    "bilinear",
    "bicubic",
    "trilinear",
    "area",
    "nearest-exact",
    "jbu",
    "cross_attn",
]
LayerSelection: TypeAlias = Literal["last", "multiscale"] | Sequence[int]


def infer_vit_params(
    dino_name: str, model: nn.Module | None = None
) -> tuple[int, int, int]:
    """Infer ``(feat_dim, patch_size, n_heads)`` from a model instance or name."""
    if model is not None:
        inner: Any = getattr(model, "model", model)
        embed_dim = getattr(inner, "embed_dim", getattr(inner, "num_features", None))
        n_heads = getattr(inner, "num_heads", None)
        if n_heads is None and hasattr(inner, "blocks") and len(inner.blocks) > 0:
            n_heads = getattr(inner.blocks[0].attn, "num_heads", 6)
        patch_size = getattr(model, "patch_size", None)
        if patch_size is None and hasattr(inner, "patch_embed"):
            ps = getattr(inner.patch_embed, "patch_size", 16)
            patch_size = ps[0] if isinstance(ps, (tuple, list)) else int(ps)
        if embed_dim is not None and patch_size is not None and n_heads is not None:
            return int(embed_dim), int(patch_size), int(n_heads)

    feat_dim_lookup = {
        "t": 192,
        "s": 384,
        "b": 768,
        "l": 1024,
        "h": 1280,
        "g": 1536,
        "7b": 4096,
    }
    n_heads_lookup = {
        "t": 3,
        "s": 6,
        "b": 12,
        "l": 16,
        "h": 16,
        "g": 16,
        "7b": 32,
    }
    for segment in dino_name.split("_"):
        m = re.match(r"^vit(7b|[tsblhg])(\d+)", segment)
        if m:
            arch, patch_sz = m.group(1), int(m.group(2))
            return feat_dim_lookup[arch], patch_sz, n_heads_lookup[arch]

    if "tiny" in dino_name:
        return 192, 16, 3
    if "base" in dino_name:
        return 768, 16, 12
    if "large" in dino_name:
        return 1024, 16, 16
    return 384, 16, 6


def load_backbone(
    dino_name: str,
    stride: int,
    device: str | torch.device | None = None,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """Single-pass factory loading DINOv2, DINOv3, ALiBi, NoPE, or ``timm`` ViTs."""
    if "backbone" in kwargs and isinstance(kwargs["backbone"], nn.Module):
        return kwargs["backbone"]

    dev_str = str(device) if device is not None else "cpu"

    if dino_name.startswith("alibi:") and "model_path" not in kwargs:
        # Instantiate an un-checkpointed ALiBi-wrapped timm ViT directly!
        from .alibi import MODEL_LIST, AlibiVitWrapper

        spec = dino_name.split("alibi:", 1)[1] or MODEL_LIST[1]
        return AlibiVitWrapper(
            model_identifier=spec,
            stride=stride,
            device=dev_str,
            pretrained=pretrained,
        )

    if "alibi" in dino_name or "nope" in dino_name:
        model_path = kwargs["model_path"]
        if "dv3" in dino_name or "dinov3" in dino_name:
            lib_path = kwargs["lib_path"]
            return get_dv3_model(
                dino_name, model_path, lib_path, device=dev_str, stride=stride
            )
        return get_alibi_model(dino_name, model_path, device=dev_str, stride=stride)

    if dino_name.startswith("timm:"):
        timm_id = dino_name.split("timm:", 1)[1]
        return create_model(timm_id, pretrained=pretrained, dynamic_img_size=True)

    if "dinov2" in dino_name:
        return torch.hub.load("facebookresearch/dinov2", dino_name)

    if "dinov3" in dino_name:
        lib_path = kwargs.get("lib_path")
        chk_path = kwargs.get("chk_path")
        assert (chk_path is not None) and (lib_path is not None), (
            "Must supply local lib_path and chk_path for DINOv3!"
        )
        return torch.hub.load(lib_path, dino_name, source="local", weights=chk_path)

    if "dino" in dino_name:
        return torch.hub.load("facebookresearch/dino:main", dino_name)

    if "deit" in dino_name:
        return create_model(
            "deit_small_patch16_224", pretrained=pretrained, dynamic_img_size=True
        )
    if "384" in dino_name:
        return create_model(
            "vit_small_patch16_384", pretrained=pretrained, dynamic_img_size=True
        )
    return create_model(
        "vit_small_patch16_224", pretrained=pretrained, dynamic_img_size=True
    )


class HighResBackbone(nn.Module):
    """Self-contained dense ViT feature extractor supporting sub-patch stride,
    invertible TTA ensembles, JBU or ``AnyUp``/``JAFAR``-style cross-attention
    upsampling, and apositional coordinate-ramp debiasing."""

    def __init__(
        self,
        dino_name: str,
        stride: int = 4,
        pca_dim: int = -1,
        dtype: torch.dtype | int = torch.float32,
        track_grad: bool = False,
        pretrained: bool = True,
        interpolation_mode: Interpolation = "nearest-exact",
        layers: LayerSelection = "last",
        tile_size: int | None = None,
        tile_overlap: int = 64,
        debias_positional: bool = False,
        debias_degree: int = 2,
        learned_upsampler: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dino_name = dino_name
        device = kwargs.pop("device", None)

        self.dinov2: nn.Module = load_backbone(
            dino_name, stride=stride, device=device, pretrained=pretrained, **kwargs
        )
        self.dinov2.eval()

        feat, patch, n_heads = infer_vit_params(dino_name, self.dinov2)
        if "dinov2" not in dino_name:
            dino_any: Any = self.dinov2
            if not hasattr(dino_any, "num_heads"):
                dino_any.num_heads = n_heads
            if not hasattr(dino_any, "num_register_tokens"):
                dino_any.num_register_tokens = 0

        self.original_patch_size: int = patch
        self.original_stride = _pair(patch)
        self.base_feat_dim: int = feat
        self.n_heads: int = n_heads
        self.n_register_tokens: int = getattr(self.dinov2, "num_register_tokens", 0)
        self.layers_spec: LayerSelection = layers
        self._blocks_ref, self.layer_indices = self._resolve_layer_selection(
            self.dinov2, layers
        )
        self.feat_dim: int = (
            feat * len(self.layer_indices) if self.layer_indices is not None else feat
        )
        self.tile_size: int | None = tile_size
        self.tile_overlap: int = int(tile_overlap)

        self.stride = _pair(stride)
        self.set_model_stride(self.dinov2, patch)
        self.set_model_stride(self.dinov2, stride)

        self.transforms: list[partial] = []
        self.inverse_transforms: list[partial] = []
        self.interpolation_mode: Interpolation = interpolation_mode
        self.jbu_upsampler = JointBilateralUpsampler(
            guidance_dim=3, radius=2, sigma_spatial=1.5, sigma_range=0.15, learned=False
        )
        self.cross_attn_upsampler = CrossAttentionFeatureUpsampler(
            feat_dim=self.feat_dim,
            key_dim=16,
            radius=2,
            learned=learned_upsampler,
        )
        self.debiaser = ApositionalFeatureDebiaser(
            enabled=debias_positional, degree=debias_degree
        )
        self.pca_dim = pca_dim
        self.do_pca = pca_dim > 3

        if isinstance(dtype, int):
            dtype = torch.float16 if dtype == 16 else torch.float32
        self.dtype = dtype

        if dtype != torch.float32:
            self.dinov2 = self.dinov2.to(dtype)
        if device is not None:
            self.dinov2 = self.dinov2.to(device)
            self.jbu_upsampler = self.jbu_upsampler.to(device)
            self.cross_attn_upsampler = self.cross_attn_upsampler.to(device)
            self.debiaser = self.debiaser.to(device)

        self.track_grad = track_grad
        self.patch_last_block(self.dinov2, dino_name)

    @staticmethod
    def _resolve_layer_selection(
        model: nn.Module, layers: LayerSelection
    ) -> tuple[Any | None, list[int] | None]:
        if layers == "last":
            return None, None
        blocks: Any = getattr(model, "blocks", None)
        if blocks is None and hasattr(model, "model"):
            blocks = getattr(model.model, "blocks", None)
        if blocks is None or len(blocks) == 0:
            return None, None
        n_blocks = len(blocks)
        if layers == "multiscale":
            indices = sorted(
                {
                    max(0, min(n_blocks - 1, round((k + 1) * n_blocks / 4) - 1))
                    for k in range(4)
                }
            )
            return blocks, indices
        if isinstance(layers, Sequence) and not isinstance(layers, str):
            resolved: list[int] = []
            for raw_idx in layers:
                idx = int(raw_idx)
                if idx < 0:
                    idx += n_blocks
                if not (0 <= idx < n_blocks):
                    raise ValueError(
                        f"Layer index {raw_idx} out of bounds for ViT "
                        f"with {n_blocks} blocks."
                    )
                if idx not in resolved:
                    resolved.append(idx)
            if not resolved:
                raise ValueError(
                    "layers sequence must contain at least one block index."
                )
            return blocks, resolved
        raise ValueError(f"Unsupported layers specification: {layers!r}")

    def get_model_params(self, dino_name: str) -> tuple[int, int, int]:
        return infer_vit_params(dino_name, getattr(self, "dinov2", None))

    def set_model_stride(
        self, dino_model: nn.Module, stride_l: int, verbose: bool = False
    ) -> None:
        dino_any: Any = dino_model
        new_stride_pair = _pair(stride_l)
        self.stride = new_stride_pair

        # Handle wrapped ALiBi / DINOv3 models whose ViT lives at .model
        if hasattr(dino_any, "model") and hasattr(dino_any.model, "patch_embed"):
            dino_any.model.patch_embed.proj.stride = new_stride_pair
            dino_any.stride = stride_l
            if verbose:
                print(f"Setting wrapped stride to ({stride_l},{stride_l})")
            return

        if hasattr(dino_any, "patch_embed") and hasattr(dino_any.patch_embed, "proj"):
            dino_any.patch_embed.proj.stride = new_stride_pair
            if hasattr(dino_any.patch_embed, "strict_img_size"):
                dino_any.patch_embed.strict_img_size = False
            if self.dino_name.startswith("timm:"):
                pe_mod = dino_any.patch_embed

                def _subpatch_timm_pe_forward(
                    self_pe: Any, inp: torch.Tensor
                ) -> torch.Tensor:
                    proj_out = self_pe.proj(inp)
                    if getattr(self_pe, "flatten", True):
                        proj_out = proj_out.flatten(2).transpose(1, 2)
                    else:
                        fmt = str(getattr(self_pe, "output_fmt", ""))
                        if "NHWC" in fmt:
                            proj_out = proj_out.permute(0, 2, 3, 1)
                    return self_pe.norm(proj_out)

                pe_mod.forward = MethodType(_subpatch_timm_pe_forward, pe_mod)
            # For non-timm DINO/DINOv2 models, patch interpolate_pos_encoding
            elif hasattr(dino_any, "pos_embed") and dino_any.pos_embed is not None:
                patched_pos = MethodType(
                    Patch._fix_pos_enc(self.original_patch_size, new_stride_pair),
                    dino_any,
                )
                dino_any.interpolate_pos_encoding = patched_pos
        if verbose:
            print(f"Setting stride to ({stride_l},{stride_l})")

    def patch_last_block(self, dino_model: nn.Module, dino_name: str) -> None:
        dino_any: Any = dino_model
        if hasattr(dino_any, "forward_feats_attn"):
            return

        if "alibi" in dino_name or "nope" in dino_name:

            def forward_feats_attn(self_model, x, masks=None, attn_choice="none"):
                feats = self_model.forward_features(x)
                feats = feats.permute(0, 2, 1)
                return {"x_norm_patchtokens": feats, "masks": masks}

            dino_any.forward_feats_attn = MethodType(forward_feats_attn, dino_any)
            return

        if dino_name.startswith("timm:"):

            def forward_feats_timm(self_model, x, masks=None, attn_choice="none"):
                feats = self_model.forward_features(x)
                n_prefix = getattr(self_model, "num_prefix_tokens", 1)
                patch_tokens = feats[:, n_prefix:, :]
                return {"x_norm_patchtokens": patch_tokens, "masks": masks}

            dino_any.forward_feats_attn = MethodType(forward_feats_timm, dino_any)
            return

        final_block = dino_any.blocks[-1]
        attn_block = final_block.attn
        attn_block._original_forward = attn_block.forward
        if "dinov3" not in dino_name:
            attn_block.forward = MethodType(Patch._fix_mem_eff_attn(), attn_block)

        if "dinov2" in dino_name:
            final_block.forward = MethodType(
                Patch._fix_block_forward_dv2(), final_block
            )
            dino_any.forward_feats_attn = MethodType(
                Patch._add_new_forward_features_dv2(), dino_any
            )
        elif "dinov3" in dino_name:
            dino_any.forward_feats_attn = MethodType(
                Patch._add_new_forward_features_dv3(dino_any), dino_any
            )
        elif "dino" in dino_name:
            for blk in dino_any.blocks:
                blk.forward = MethodType(Patch._fix_block_forward_dino(), blk)
                blk.attn._original_forward = blk.attn.forward
                blk.attn.forward = MethodType(Patch._fix_mem_eff_attn(), blk.attn)
            dino_any.forward_feats_attn = MethodType(
                Patch._add_new_forward_features_dino(), dino_any
            )
        else:
            for blk in dino_any.blocks:
                blk.forward = MethodType(Patch._fix_block_forward_dino(), blk)
                blk.attn._original_forward = blk.attn.forward
                blk.attn.forward = MethodType(Patch._fix_mem_eff_attn(), blk.attn)
            dino_any.forward_feats_attn = MethodType(
                Patch._add_new_forward_features_vit(), dino_any
            )

    def get_n_patches(self, img_h: int, img_w: int) -> tuple[int, int]:
        stride_l = self.stride[0]
        n_patch_h = 1 + (img_h - self.original_patch_size) // stride_l
        n_patch_w = 1 + (img_w - self.original_patch_size) // stride_l
        return n_patch_h, n_patch_w

    def set_transforms(
        self, transforms: list[partial], inv_transforms: list[partial]
    ) -> None:
        assert len(transforms) == len(inv_transforms), (
            "Each transform must have an inverse!"
        )
        self.transforms = list(transforms)
        self.inverse_transforms = list(inv_transforms)

    def _upsample_patch_map(
        self,
        patch_map: torch.Tensor,
        target_hw: tuple[int, int],
        guidance_img: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pm = self.debiaser(patch_map)
        if self.interpolation_mode == "cross_attn" and guidance_img is not None:
            return self.cross_attn_upsampler(pm, guidance_img)
        if self.interpolation_mode == "jbu" and guidance_img is not None:
            return self.jbu_upsampler(pm, guidance_img)
        mode = (
            "bilinear"
            if self.interpolation_mode in ("jbu", "cross_attn")
            else self.interpolation_mode
        )
        kwargs = {"align_corners": False} if mode in ("bilinear", "bicubic") else {}
        return F.interpolate(pm, size=target_hw, mode=mode, **kwargs)

    @torch.no_grad()
    def get_transformed_input_batch(
        self, x: torch.Tensor, transforms: list[partial]
    ) -> torch.Tensor:
        if len(transforms) == 0:
            self.transforms = [iden_partial]
            self.inverse_transforms = [iden_partial]
            transforms = self.transforms

        img_list: list[torch.Tensor] = [tr_fn(x) for tr_fn in transforms]
        if img_list[0].ndim == 3:
            img_batch = torch.stack(img_list)
        else:
            img_batch = torch.cat(img_list, dim=0)

        if self.dtype != torch.float32:
            img_batch = img_batch.to(self.dtype)
        return img_batch

    def extract_patch_tokens(
        self,
        transformed_img: torch.Tensor,
        attn_choice: AttentionOptions = "none",
    ) -> torch.Tensor:
        """Extract single-layer or multi-scale standardized patch tokens
        ``(B, N_patches, feat_dim [+ n_heads])``.
        """
        if self._blocks_ref is None or self.layer_indices is None:
            out_dict = self.dinov2.forward_feats_attn(  # type: ignore[operator]
                transformed_img, None, attn_choice
            )
            if attn_choice != "none" and "x_patchattn" in out_dict:
                return torch.cat(
                    (out_dict["x_norm_patchtokens"], out_dict["x_patchattn"]), dim=-1
                )
            return out_dict["x_norm_patchtokens"]

        captured: dict[int, torch.Tensor] = {}
        handles = []

        def _make_hook(b_idx: int):
            def _hook(_mod: nn.Module, _inp: Any, out: Any) -> None:
                tok = out[0] if isinstance(out, tuple) else out
                captured[b_idx] = tok

            return _hook

        try:
            for idx in self.layer_indices:
                handles.append(
                    self._blocks_ref[idx].register_forward_hook(_make_hook(idx))
                )
            out_dict = self.dinov2.forward_feats_attn(  # type: ignore[operator]
                transformed_img, None, attn_choice
            )
        finally:
            for h in handles:
                h.remove()

        n_patches = int(out_dict["x_norm_patchtokens"].shape[1])
        layer_tensors: list[torch.Tensor] = []
        for idx in self.layer_indices:
            raw_tok = captured[idx][:, -n_patches:, : self.base_feat_dim]
            norm_tok = F.layer_norm(raw_tok.float(), (self.base_feat_dim,)).to(
                raw_tok.dtype
            )
            layer_tensors.append(norm_tok)

        multi_feats = torch.cat(layer_tensors, dim=-1)
        if attn_choice != "none" and "x_patchattn" in out_dict:
            multi_feats = torch.cat((multi_feats, out_dict["x_patchattn"]), dim=-1)
        return multi_feats

    @torch.no_grad()
    def _forward_single_tile(
        self, x: torch.Tensor, attn_choice: AttentionOptions = "none"
    ) -> torch.Tensor:
        if self.dtype != torch.float32:
            x = x.to(self.dtype)
        img_batch = self.get_transformed_input_batch(x, self.transforms)
        _, img_h, img_w = x.shape
        n_patch_h, n_patch_w = self.get_n_patches(img_h, img_w)

        out_feature_img: torch.Tensor | None = None
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
            full_size = self._upsample_patch_map(
                permuted, (img_h, img_w), guidance_img=transformed_img
            )
            inverted = self.inverse_transforms[i](full_size)
            if out_feature_img is None:
                out_feature_img = torch.zeros_like(inverted)
            out_feature_img = out_feature_img + inverted

        assert out_feature_img is not None
        return out_feature_img / max(n_transforms, 1)

    @torch.no_grad()
    def forward_sequential(
        self, x: torch.Tensor, attn_choice: AttentionOptions = "none"
    ) -> torch.Tensor:
        """Extract dense TTA-averaged high-resolution features ``(1, C, H, W)``,
        automatically tiling when ``self.tile_size`` is set.
        """
        if (
            self.tile_size is not None
            and max(x.shape[-2], x.shape[-1]) > self.tile_size
        ):
            return run_tiled_2d(
                lambda tile: self._forward_single_tile(tile, attn_choice=attn_choice),
                x,
                tile_size=self.tile_size,
                overlap=self.tile_overlap,
                patch_size=self.original_patch_size,
                stride=self.stride[0],
            )
        return self._forward_single_tile(x, attn_choice=attn_choice)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, attn_choice: AttentionOptions = "none"
    ) -> torch.Tensor:
        return self.forward_sequential(x, attn_choice=attn_choice)


# Backward-compatible alias
HighResDV2 = HighResBackbone
