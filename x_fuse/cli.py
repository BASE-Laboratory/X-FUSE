"""Command-line interface for X-FUSE environment diagnostics, open dataset
discovery, and quick synthetic correlative verification.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Sequence
from pathlib import Path

import torch

from . import __version__
from .datasets import list_open_datasets
from .fusion import CorrelativeFuser
from .segment import compute_segmentation_metrics, jbu_guided_baseline
from .synthetic import generate_correlative_phantom_2d
from .transforms import (
    build_default_tta,
    ensure_spatial_divisible,
    normalize_structural,
)
from .uncertainty import expected_calibration_error
from .utils import guide_to_tensor


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

    hr_t, (oh, ow) = ensure_spatial_divisible(
        normalize_structural(phantom.hr_image, device=device),
        patch_size=16,
        stride=8,
    )
    guide_t = guide_to_tensor(phantom.coarse_guide, device=device, normalize=True)
    fwd, inv = build_default_tta(shift_dists=[1], pattern="Neumann")

    fuser = CorrelativeFuser(
        dino_name=dino_name,
        xrd_fuse_method="learned_gating",
        stride=8,
        dtype=torch.float32,
        loss_fn="forward_psf_bce",
        guide_channels=2,
        interpolation_mode="jbu",
        pretrained=pretrained,
        lr=1e-2,
        device=device,
    )
    fuser.set_guide_transforms(guide_t, fwd, inv)
    for _ in range(epochs):
        fuser.train_fusion_step(hr_t)

    rep = fuser.predict_with_uncertainty(
        hr_t, guide_lr=guide_t, mode="multiclass", calibrate_to_guide=True
    )
    pred_mask = rep.hard_mask[0, :oh, :ow].cpu().numpy()
    probs = rep.probabilities[:, :, :oh, :ow].cpu()
    xf_metrics = compute_segmentation_metrics(pred_mask, phantom.gt_labels)
    ece = expected_calibration_error(probs, phantom.gt_labels)

    summary = {
        "dino_name": dino_name,
        "pretrained": pretrained,
        "jbu_miou": round(jbu_metrics["iou"], 4),
        "xfuse_miou": round(xf_metrics["iou"], 4),
        "xfuse_ece": round(ece, 4),
        "mean_uncertainty": round(
            float(rep.combined_uncertainty[0, :oh, :ow].mean().item()), 4
        ),
    }
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "datasets":
        return _cmd_datasets(as_json=bool(args.json))
    if args.command == "demo":
        return _cmd_demo(
            output_path=args.output,
            dino_name=args.dino_name,
            pretrained=bool(args.pretrained),
            epochs=args.epochs,
            device=args.device,
        )
    return _cmd_info()


if __name__ == "__main__":
    raise SystemExit(main())
