"""Command-line interface for X-FUSE environment diagnostics, open dataset
discovery, synthetic smoke checks, and batch correlative fusion (`xfuse fuse`).
"""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from . import __version__
from .backbones import Interpolation, LayerSelection
from .datasets import list_open_datasets
from .fusion import XFuse
from .heads import FusionOptions
from .io import load_microscopy_file, save_uncertainty_report
from .physics import LossOptions
from .registration import AlignMode
from .segment import compute_segmentation_metrics, jbu_guided_baseline
from .synthetic import generate_correlative_phantom_2d
from .uncertainty import expected_calibration_error


def _cmd_info() -> int:
    payload = {
        "x_fuse_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        ),
    }
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_datasets(as_json: bool = False) -> int:
    registry = list_open_datasets()
    if as_json:
        print(
            json.dumps(
                {
                    k: {
                        "title": v.title,
                        "hr_modality": v.hr_modality,
                        "guide_modality": v.guide_modality,
                        "doi": v.doi,
                        "zenodo_record_id": v.zenodo_record_id,
                    }
                    for k, v in registry.items()
                },
                indent=2,
            )
        )
        return 0

    for key, ds in registry.items():
        print(f"- {key} (DOI: {ds.doi})")
        print(f"    Title: {ds.title}")
        print(f"    Modalities: {ds.hr_modality}  +  {ds.guide_modality}")
    return 0


def _cmd_demo(
    output_path: Path | None = None,
    dino_name: str = "timm:vit_tiny_patch16_224",
    pretrained: bool = False,
    epochs: int = 12,
    device: str = "cpu",
) -> int:
    phantom = generate_correlative_phantom_2d(
        size=64, downsample_factor=4, phase_b_contrast=0.35, seed=42
    )
    jbu_rep = jbu_guided_baseline(
        phantom.hr_image, phantom.coarse_guide, mode="multiclass"
    )
    jbu_metrics = compute_segmentation_metrics(jbu_rep.hard_mask[0], phantom.gt_labels)

    model = XFuse(
        backbone=dino_name,
        fusion_method="learned_gating",
        stride=8,
        dtype=torch.float32,
        loss_fn="forward_psf_bce",
        interpolation_mode="jbu",
        pretrained=pretrained,
        lr=1e-2,
        device=device,
    )
    rep = model.fit_predict(
        phantom.hr_image,
        phantom.coarse_guide,
        epochs=epochs,
        mode="multiclass",
        tta_shifts=(1,),
        tta_flips=True,
    )
    pred_mask = rep.hard_mask[0].cpu().numpy()
    probs = rep.probabilities.cpu()
    xf_metrics = compute_segmentation_metrics(pred_mask, phantom.gt_labels)
    ece = expected_calibration_error(probs, phantom.gt_labels)

    summary = {
        "dino_name": dino_name,
        "pretrained": pretrained,
        "jbu_miou": round(jbu_metrics["iou"], 4),
        "xfuse_miou": round(xf_metrics["iou"], 4),
        "xfuse_ece": round(ece, 4),
        "mean_uncertainty": round(float(rep.combined_uncertainty[0].mean().item()), 4),
    }
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return 0


def _cmd_fuse(
    hr_path: Path,
    guide_paths: Sequence[Path],
    output_path: Path,
    backbone: str = "dinov2_vits14",
    fusion_method: FusionOptions = "learned_gating",
    loss_fn: LossOptions = "forward_psf_bce",
    upsampler: Interpolation = "jbu",
    learned_upsampler: bool = False,
    debias_positional: bool = False,
    conformal_alpha: float | None = None,
    spectral_unmix: bool = False,
    n_components: int | None = None,
    layers: Literal["last", "multiscale"] = "last",
    align_guide: AlignMode = "none",
    learn_psf: bool = False,
    tile_size: int | None = None,
    stride: int = 4,
    epochs: int = 25,
    pretrained: bool = True,
    device: str = "cpu",
    pixel_size_um: float | None = None,
) -> int:
    if stride < 1:
        raise ValueError(f"--stride must be >= 1, got {stride}.")
    if epochs < 1:
        raise ValueError(f"--epochs must be >= 1, got {epochs}.")
    if conformal_alpha is not None and not (0.0 < conformal_alpha < 1.0):
        raise ValueError(
            f"--conformal-alpha must lie strictly in (0, 1), got {conformal_alpha}."
        )
    if tile_size is not None and tile_size < 16:
        raise ValueError(f"--tile-size must be >= 16, got {tile_size}.")
    if not hr_path.exists():
        raise FileNotFoundError(f"High-resolution input file not found: {hr_path}")
    for gp in guide_paths:
        if not gp.exists():
            raise FileNotFoundError(f"Coarse guide file not found: {gp}")

    hr_arr, _ = load_microscopy_file(hr_path)
    g_list = []
    names = []
    for gp in guide_paths:
        g_arr, _ = load_microscopy_file(gp)
        if g_arr.ndim == 2:
            g_list.append(g_arr)
            names.append(gp.stem)
        elif g_arr.ndim == 3:
            for idx in range(g_arr.shape[0]):
                g_list.append(g_arr[idx])
                names.append(f"{gp.stem}_{idx}")
        else:
            raise ValueError(f"Unsupported guide array shape {g_arr.shape} from {gp}.")

    guide_stack = np.stack(g_list, axis=0).astype(np.float32)
    spec_ch = int(guide_stack.shape[0]) if spectral_unmix else None
    g_ch = (
        int(n_components or 2) if spectral_unmix else max(1, int(guide_stack.shape[0]))
    )
    if spectral_unmix:
        names = [f"component_{i}" for i in range(g_ch)]

    layer_sel: LayerSelection = layers
    model = XFuse(
        backbone=backbone,
        fusion_method=fusion_method,
        stride=stride,
        dtype=torch.float32,
        loss_fn=loss_fn,
        guide_channels=g_ch,
        spectral_channels=spec_ch,
        interpolation_mode=upsampler,
        learned_upsampler=learned_upsampler,
        debias_positional=debias_positional,
        conformal_alpha=conformal_alpha,
        pretrained=pretrained,
        lr=8e-3,
        layers=layer_sel,
        align_guide=align_guide,
        learn_psf=learn_psf,
        tile_size=tile_size,
        device=device,
    )
    rep = model.fit_predict(
        hr_arr,
        guide_stack,
        epochs=epochs,
        mode="auto",
        spectral_unmix=spectral_unmix,
        n_components=n_components,
    )
    ch_names = (
        ["background"] + names
        if rep.probabilities.shape[1] == len(names) + 1
        else names
    )
    saved = save_uncertainty_report(
        rep, output_path, channel_names=ch_names, pixel_size_um=pixel_size_um
    )
    out_summary = {
        "saved_report": str(saved),
        "hr_shape": list(hr_arr.shape),
        "guide_shape": list(guide_stack.shape),
        "has_conformal_sets": rep.prediction_sets is not None,
        "diagnostics": rep.diagnostics,
    }
    print(json.dumps(out_summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xfuse",
        description=(
            "X-FUSE: Physics-guided foundation-model fusion and uncertainty "
            "quantification for correlative materials imaging."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="Print runtime Python, PyTorch, and CUDA diagnostics.")

    p_ds = sub.add_parser(
        "datasets", help="List registered open-access correlative datasets (Zenodo)."
    )
    p_ds.add_argument(
        "--json", action="store_true", help="Emit dataset registry as JSON."
    )

    p_demo = sub.add_parser(
        "demo", help="Run a self-contained synthetic 2D correlative smoke check."
    )
    p_demo.add_argument("--dino-name", default="timm:vit_tiny_patch16_224")
    p_demo.add_argument("--pretrained", action="store_true")
    p_demo.add_argument("--epochs", type=int, default=12)
    p_demo.add_argument("--device", default="cpu")
    p_demo.add_argument("--output", type=Path, default=None)

    p_fuse = sub.add_parser(
        "fuse",
        help=(
            "Run 2D correlative fusion on micrograph files and export "
            "OME-TIFF/HDF5/Zarr."
        ),
    )
    p_fuse.add_argument(
        "--hr", type=Path, required=True, help="High-res structural file."
    )
    p_fuse.add_argument(
        "--guide",
        type=Path,
        nargs="+",
        required=True,
        help="One or more coarse physical guide files (or 3D spectral cube).",
    )
    p_fuse.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output report path (.ome.tif, .h5, or .zarr).",
    )
    p_fuse.add_argument("--backbone", default="dinov2_vits14")
    p_fuse.add_argument(
        "--fusion-method",
        choices=["gating", "learned_gating", "spatial_mlp", "attention"],
        default="learned_gating",
    )
    p_fuse.add_argument(
        "--loss",
        choices=[
            "bce",
            "pearson",
            "forward_psf_bce",
            "forward_psf_mse",
            "forward_psf_poisson",
            "compositional_kl",
        ],
        default="forward_psf_bce",
    )
    p_fuse.add_argument(
        "--upsampler",
        choices=[
            "jbu",
            "cross_attn",
            "bilinear",
            "bicubic",
            "nearest",
            "nearest-exact",
            "area",
        ],
        default="jbu",
    )
    p_fuse.add_argument(
        "--learned-upsampler",
        action="store_true",
        help="Jointly fine-tune CrossAttentionFeatureUpsampler projections.",
    )
    p_fuse.add_argument(
        "--debias-positional",
        action="store_true",
        help="Project out linear/quadratic positional biases from ViT features.",
    )
    p_fuse.add_argument(
        "--conformal-alpha",
        type=float,
        default=None,
        help="Optional miscoverage level alpha in (0, 1) (e.g. 0.10 for 90%% sets).",
    )
    p_fuse.add_argument(
        "--spectral-unmix",
        action="store_true",
        help="Unmix a raw 3D spectral/diffraction cube via NMF + forward model.",
    )
    p_fuse.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Number of unmixed components when --spectral-unmix is set.",
    )
    p_fuse.add_argument("--layers", choices=["last", "multiscale"], default="last")
    p_fuse.add_argument(
        "--align-guide",
        choices=["none", "translation", "rigid", "affine", "bspline"],
        default="none",
    )
    p_fuse.add_argument("--learn-psf", action="store_true")
    p_fuse.add_argument("--tile-size", type=int, default=None)
    p_fuse.add_argument("--stride", type=int, default=4)
    p_fuse.add_argument("--epochs", type=int, default=25)
    p_fuse.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable downloading pretrained weights (useful for offline testing).",
    )
    p_fuse.add_argument("--device", default="cpu")
    p_fuse.add_argument("--pixel-size-um", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "datasets":
        return _cmd_datasets(as_json=bool(args.json))
    if args.command == "demo":
        if args.epochs < 1:
            raise ValueError(f"--epochs must be >= 1, got {args.epochs}.")
        return _cmd_demo(
            output_path=args.output,
            dino_name=args.dino_name,
            pretrained=bool(args.pretrained),
            epochs=args.epochs,
            device=args.device,
        )
    if args.command == "fuse":
        return _cmd_fuse(
            hr_path=args.hr,
            guide_paths=args.guide,
            output_path=args.output,
            backbone=args.backbone,
            fusion_method=args.fusion_method,
            loss_fn=args.loss,
            upsampler=args.upsampler,
            learned_upsampler=bool(args.learned_upsampler),
            debias_positional=bool(args.debias_positional),
            conformal_alpha=args.conformal_alpha,
            spectral_unmix=bool(args.spectral_unmix),
            n_components=args.n_components,
            layers=args.layers,
            align_guide=args.align_guide,
            learn_psf=bool(args.learn_psf),
            tile_size=args.tile_size,
            stride=args.stride,
            epochs=args.epochs,
            pretrained=not bool(args.no_pretrained),
            device=args.device,
            pixel_size_um=args.pixel_size_um,
        )
    return _cmd_info()


if __name__ == "__main__":
    raise SystemExit(main())
