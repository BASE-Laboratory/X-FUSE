"""Command-line interface for X-FUSE environment diagnostics, open dataset
discovery & download, synthetic smoke checks, 2D correlative fusion (`xfuse fuse`),
and sparse-slice 3D volumetric fusion (`xfuse fuse-3d`).
"""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from . import __version__
from .active import suggest_next_slices
from .backbones import Interpolation, LayerSelection
from .datasets import fetch_open_dataset, list_open_datasets
from .fusion import PoreMaskSpec, XFuse
from .heads import FusionOptions
from .io import export_for_openimpala, load_microscopy_file, save_uncertainty_report
from .physics import LossOptions
from .registration import AlignMode
from .segment import compute_segmentation_metrics, jbu_guided_baseline
from .synthetic import generate_correlative_phantom_2d
from .uncertainty import ConformalMethod, expected_calibration_error
from .upsamplers import PSFProfile
from .volumetric import (
    VolumeSliceAccessor,
    VolumetricFuser,
    VolumetricUncertaintyReport,
)


def _parse_pore_mask_arg(raw_spec: str | None) -> PoreMaskSpec:
    if raw_spec is None:
        return None
    if raw_spec == "auto_otsu":
        return "auto_otsu"
    if raw_spec == "auto_otsu_fringe_guarded":
        return "auto_otsu_fringe_guarded"
    pm_path = Path(raw_spec)
    if not pm_path.exists():
        raise FileNotFoundError(
            f"--pore-mask '{raw_spec}' must be 'auto_otsu', "
            "'auto_otsu_fringe_guarded', or an existing mask file."
        )
    arr, _ = load_microscopy_file(pm_path)
    return arr > 0.5


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


def _cmd_datasets(
    as_json: bool = False,
    fetch_key: str | None = None,
    dest_root: Path | None = None,
) -> int:
    if fetch_key is not None:
        paths = fetch_open_dataset(fetch_key, dest_root=dest_root)
        payload = {
            "dataset_key": fetch_key,
            "downloaded_files": [str(p) for p in paths],
        }
        print(json.dumps(payload, indent=2))
        return 0

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
        fusion_method="spatial_mlp",
        stride=8,
        dtype=torch.float32,
        loss_fn="forward_psf_bce",
        interpolation_mode="cross_attn",
        guide_residual=True,
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
    fusion_method: FusionOptions = "spatial_mlp",
    loss_fn: LossOptions = "forward_psf_bce",
    upsampler: Interpolation = "cross_attn",
    learned_upsampler: bool = False,
    debias_positional: bool = False,
    conformal_alpha: float | None = None,
    conformal_method: ConformalMethod = "raps",
    conformal_block_size: int = 8,
    spectral_unmix: bool = False,
    n_components: int | None = None,
    layers: Literal["last", "multiscale"] = "last",
    align_guide: AlignMode = "none",
    learn_psf: bool = False,
    psf_profile: PSFProfile = "gaussian",
    guide_residual: bool = True,
    tv_weight: float = 0.01,
    tile_size: int | None = None,
    stride: int = 4,
    epochs: int = 25,
    pretrained: bool = True,
    device: str = "cpu",
    pixel_size_um: float | None = None,
    pore_mask_raw: str | None = None,
) -> int:
    if stride < 1:
        raise ValueError(f"--stride must be >= 1, got {stride}.")
    if epochs < 1:
        raise ValueError(f"--epochs must be >= 1, got {epochs}.")
    if conformal_alpha is not None and not (0.0 < conformal_alpha < 1.0):
        raise ValueError(
            f"--conformal-alpha must lie strictly in (0, 1), got {conformal_alpha}."
        )
    if conformal_block_size < 1:
        raise ValueError(
            f"--conformal-block-size must be >= 1, got {conformal_block_size}."
        )
    if tile_size is not None and tile_size < 16:
        raise ValueError(f"--tile-size must be >= 16, got {tile_size}.")
    if tv_weight < 0.0:
        raise ValueError(f"--tv-weight must be >= 0, got {tv_weight}.")
    if n_components is not None and n_components < 1:
        raise ValueError(f"--n-components must be >= 1, got {n_components}.")
    if pixel_size_um is not None and pixel_size_um <= 0.0:
        raise ValueError(f"--pixel-size-um must be > 0, got {pixel_size_um}.")
    if not hr_path.exists():
        raise FileNotFoundError(f"High-resolution input file not found: {hr_path}")
    for gp in guide_paths:
        if not gp.exists():
            raise FileNotFoundError(f"Coarse guide file not found: {gp}")

    pore_spec = _parse_pore_mask_arg(pore_mask_raw)
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
        conformal_method=conformal_method,
        conformal_fit_temperature=True,
        conformal_spatial_block_size=conformal_block_size,
        pretrained=pretrained,
        lr=8e-3,
        layers=layer_sel,
        align_guide=align_guide,
        learn_psf=learn_psf,
        psf_profile=psf_profile,
        guide_residual=guide_residual,
        tv_weight=tv_weight,
        tile_size=tile_size,
        pore_mask=pore_spec,
        device=device,
    )
    rep = model.fit_predict(
        hr_arr,
        guide_stack,
        epochs=epochs,
        mode="auto",
        spectral_unmix=spectral_unmix,
        n_components=n_components,
        pore_mask=pore_spec,
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


def _save_volumetric_report(
    rep: VolumetricUncertaintyReport, output_path: Path
) -> Path:
    return save_uncertainty_report(rep, output_path)


def _parse_guide_slice_spec(spec: str) -> tuple[int, Path]:
    if ":" not in spec:
        raise ValueError(
            f"Invalid --guide-slice '{spec}'. Expected '<z_index>:<file_path>'."
        )
    z_str, path_str = spec.split(":", 1)
    z_idx = int(z_str)
    if z_idx < 0:
        raise ValueError(f"Slice index must be >= 0, got {z_idx} in '{spec}'.")
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Sparse guide slice file not found: {p}")
    return z_idx, p


def _clean_openimpala_stem(output_path: Path, export_dir: Path) -> str:
    stem = output_path.stem
    if stem.lower().endswith(".ome"):
        stem = stem[:-4]
    if not stem:
        stem = "sample"
    cand_tif = (export_dir / f"{stem}.tif").resolve()
    if cand_tif == output_path.resolve():
        stem = f"{stem}_openimpala"
    return stem


def _cmd_fuse_3d(
    hr_volume_path: Path,
    guide_slice_specs: Sequence[str],
    output_path: Path,
    backbone: str = "dinov2_vits14",
    loss_fn: LossOptions = "compositional_kl",
    axial_psf_sigma: float = 1.5,
    learn_axial_psf: bool | None = None,
    use_triplane: bool = True,
    suggest_next: int = 0,
    stride: int = 4,
    epochs: int = 12,
    pretrained: bool = True,
    device: str = "cpu",
    pore_mask_raw: str | None = None,
    export_openimpala_dir: Path | None = None,
    conformal_alpha: float | None = None,
    conformal_method: ConformalMethod = "raps",
    openimpala_phase: int = 0,
    openimpala_direction: str = "x",
) -> int:
    if stride < 1:
        raise ValueError(f"--stride must be >= 1, got {stride}.")
    if epochs < 1:
        raise ValueError(f"--epochs must be >= 1, got {epochs}.")
    if axial_psf_sigma < 0.0:
        raise ValueError(f"--axial-psf-sigma must be >= 0, got {axial_psf_sigma}.")
    if suggest_next < 0:
        raise ValueError(f"--suggest-next must be >= 0, got {suggest_next}.")
    if conformal_alpha is not None and not (0.0 < conformal_alpha < 1.0):
        raise ValueError(
            f"--conformal-alpha must lie strictly in (0, 1), got {conformal_alpha}."
        )
    if not hr_volume_path.exists():
        raise FileNotFoundError(f"3D HR volume not found: {hr_volume_path}")

    eff_learn_axial = (
        (axial_psf_sigma > 0.0) if learn_axial_psf is None else bool(learn_axial_psf)
    )
    pore_spec = _parse_pore_mask_arg(pore_mask_raw)
    accessor = VolumeSliceAccessor(hr_volume_path)
    z_dim = accessor.shape[0]
    sparse_guides: dict[int, np.ndarray | torch.Tensor] = {}
    n_ch: int | None = None
    for spec in guide_slice_specs:
        z_idx, g_path = _parse_guide_slice_spec(spec)
        if z_idx >= z_dim:
            raise ValueError(f"Guide slice z={z_idx} exceeds volume depth Z={z_dim}.")
        g_arr, _ = load_microscopy_file(g_path)
        if g_arr.ndim == 2:
            g_arr = g_arr[None, ...]
        elif g_arr.ndim != 3:
            raise ValueError(f"Expected 2D or 3D guide slice, got {g_arr.shape}.")
        if n_ch is None:
            n_ch = int(g_arr.shape[0])
        elif int(g_arr.shape[0]) != n_ch:
            raise ValueError(
                "Inconsistent guide channels across slices "
                f"({g_arr.shape[0]} vs {n_ch})."
            )
        sparse_guides[z_idx] = g_arr.astype(np.float32)

    assert n_ch is not None
    fuser = XFuse(
        backbone=backbone,
        fusion_method="spatial_mlp",
        stride=stride,
        dtype=torch.float32,
        loss_fn=loss_fn,
        interpolation_mode="cross_attn",
        guide_channels=n_ch,
        learn_psf=True,
        guide_residual=True,
        conformal_alpha=conformal_alpha,
        conformal_method=conformal_method,
        pore_mask=pore_spec,
        pretrained=pretrained,
        lr=8e-3,
        device=device,
    )
    vf = VolumetricFuser(fuser, tta_shifts=None, tta_flips=False)
    vf.fit_sparse_slices(
        accessor,
        sparse_guides,
        epochs=epochs,
        pore_mask=pore_spec,
        depth_psf_sigma=axial_psf_sigma,
        learn_depth_psf=eff_learn_axial,
        triplane_consistency_weight=0.15 if use_triplane else 0.0,
    )
    vol_mode: Literal["triplane", "2.5d_slab"] = (
        "triplane" if use_triplane else "2.5d_slab"
    )
    pred_mode: Literal["multiclass", "multilabel"] = (
        "multiclass" if (n_ch >= 2 or loss_fn == "compositional_kl") else "multilabel"
    )
    rep = vf.predict_volume(
        accessor,
        mode=vol_mode,
        sparse_guides=sparse_guides,
        pred_mode=pred_mode,
        pore_mask=pore_spec,
    )
    saved = _save_volumetric_report(rep, output_path)
    recommended: list[int] = []
    if suggest_next > 0:
        plan = suggest_next_slices(
            rep, acquired_indices=sorted(sparse_guides.keys()), k=int(suggest_next)
        )
        recommended = list(plan.recommended_indices)

    summary: dict[str, Any] = {
        "saved_report": str(saved),
        "volume_shape_zyx": list(accessor.shape),
        "acquired_slices": sorted(sparse_guides.keys()),
        "z_slice_consistency": round(float(rep.z_slice_consistency), 4),
        "has_conformal_sets": rep.prediction_sets is not None,
        "suggested_next_slices": recommended,
        "diagnostics": rep.diagnostics,
    }
    if export_openimpala_dir is not None:
        imp_name = _clean_openimpala_stem(output_path, export_openimpala_dir)
        imp_tif, imp_in = export_for_openimpala(
            rep,
            export_openimpala_dir,
            name=imp_name,
            active_phase=openimpala_phase,
            direction=openimpala_direction,
        )
        summary["openimpala_tiff"] = str(imp_tif)
        summary["openimpala_inputs"] = str(imp_in)
    print(json.dumps(summary, indent=2))
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
        "datasets",
        help="List or fetch registered open-access correlative datasets (Zenodo).",
    )
    p_ds.add_argument(
        "--json", action="store_true", help="Emit dataset registry as JSON."
    )
    p_ds.add_argument(
        "--fetch",
        default=None,
        metavar="DATASET_KEY",
        help="Download and checksum-verify a registered Zenodo dataset.",
    )
    p_ds.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Optional destination directory when --fetch is used.",
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
        default="spatial_mlp",
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
        default="cross_attn",
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
        "--conformal-method",
        choices=["lac", "aps", "raps"],
        default="raps",
        help="Conformal nonconformity score when --conformal-alpha is set.",
    )
    p_fuse.add_argument(
        "--conformal-block-size",
        type=int,
        default=8,
        help="Spatial block stride B for correlated-pixel conformal calibration.",
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
    p_fuse.add_argument(
        "--psf-profile",
        choices=["gaussian", "moffat", "core_halo"],
        default="gaussian",
        help="Probe PSF radial profile (gaussian, heavy-tailed moffat, or core_halo).",
    )
    p_fuse.add_argument(
        "--guide-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Anchor SpatialConvMLPHead logits to smooth coarse-guide prior.",
    )
    p_fuse.add_argument("--tv-weight", type=float, default=0.01)
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
    p_fuse.add_argument(
        "--pore-mask",
        default=None,
        help="Optional void mask file, 'auto_otsu', or 'auto_otsu_fringe_guarded'.",
    )

    p_f3d = sub.add_parser(
        "fuse-3d",
        help="Run sparse-slice 3D volumetric fusion & active slice steering.",
    )
    p_f3d.add_argument(
        "--hr-volume",
        type=Path,
        required=True,
        help="3D structural volume (.npy, .tif, .h5, .zarr, or slice directory).",
    )
    p_f3d.add_argument(
        "--guide-slice",
        nargs="+",
        required=True,
        metavar="Z:PATH",
        help="One or more sparse 2D chemical guide slices formatted as '<z>:<path>'.",
    )
    p_f3d.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output 3D report path (.h5, .zarr, .npz, or .tif).",
    )
    p_f3d.add_argument("--backbone", default="dinov2_vits14")
    p_f3d.add_argument(
        "--loss",
        choices=[
            "forward_psf_bce",
            "forward_psf_mse",
            "forward_psf_poisson",
            "compositional_kl",
        ],
        default="compositional_kl",
    )
    p_f3d.add_argument("--axial-psf-sigma", type=float, default=1.5)
    p_f3d.add_argument(
        "--learn-axial-psf",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Learn axial depth-PSF sigma jointly (defaults to True when sigma > 0).",
    )
    p_f3d.add_argument(
        "--triplane",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable orthogonal XY/XZ/YZ tri-plane inference and consistency.",
    )
    p_f3d.add_argument(
        "--suggest-next",
        type=int,
        default=0,
        help="Recommend next K unvisited Z-slices for active beamtime steering.",
    )
    p_f3d.add_argument(
        "--conformal-alpha",
        type=float,
        default=None,
        help="Optional miscoverage level alpha in (0, 1) for 3D conformal bounds.",
    )
    p_f3d.add_argument(
        "--conformal-method",
        choices=["lac", "aps", "raps"],
        default="raps",
        help="Conformal nonconformity score when --conformal-alpha is set.",
    )
    p_f3d.add_argument(
        "--pore-mask",
        default=None,
        help="Optional 3D void mask file, 'auto_otsu', or 'auto_otsu_fringe_guarded'.",
    )
    p_f3d.add_argument(
        "--export-openimpala-dir",
        type=Path,
        default=None,
        help="Optional directory to export OpenImpala uint8 .tif + AMReX .inputs.",
    )
    p_f3d.add_argument(
        "--openimpala-phase",
        type=int,
        default=0,
        help="Active phase ID for OpenImpala tortuosity deck (default: 0 = pore).",
    )
    p_f3d.add_argument(
        "--openimpala-direction",
        choices=["x", "y", "z"],
        default="x",
        help="Transport axis ('x', 'y', or 'z') for OpenImpala tortuosity deck.",
    )
    p_f3d.add_argument("--stride", type=int, default=4)
    p_f3d.add_argument("--epochs", type=int, default=12)
    p_f3d.add_argument("--no-pretrained", action="store_true")
    p_f3d.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "datasets":
        return _cmd_datasets(
            as_json=bool(args.json),
            fetch_key=args.fetch,
            dest_root=args.dest,
        )
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
            conformal_method=args.conformal_method,
            conformal_block_size=args.conformal_block_size,
            spectral_unmix=bool(args.spectral_unmix),
            n_components=args.n_components,
            layers=args.layers,
            align_guide=args.align_guide,
            learn_psf=bool(args.learn_psf),
            psf_profile=args.psf_profile,
            guide_residual=bool(args.guide_residual),
            tv_weight=args.tv_weight,
            tile_size=args.tile_size,
            stride=args.stride,
            epochs=args.epochs,
            pretrained=not bool(args.no_pretrained),
            device=args.device,
            pixel_size_um=args.pixel_size_um,
            pore_mask_raw=args.pore_mask,
        )
    if args.command == "fuse-3d":
        return _cmd_fuse_3d(
            hr_volume_path=args.hr_volume,
            guide_slice_specs=args.guide_slice,
            output_path=args.output,
            backbone=args.backbone,
            loss_fn=args.loss,
            axial_psf_sigma=args.axial_psf_sigma,
            learn_axial_psf=args.learn_axial_psf,
            use_triplane=bool(args.triplane),
            suggest_next=args.suggest_next,
            stride=args.stride,
            epochs=args.epochs,
            pretrained=not bool(args.no_pretrained),
            device=args.device,
            pore_mask_raw=args.pore_mask,
            export_openimpala_dir=args.export_openimpala_dir,
            conformal_alpha=args.conformal_alpha,
            conformal_method=args.conformal_method,
            openimpala_phase=args.openimpala_phase,
            openimpala_direction=args.openimpala_direction,
        )
    return _cmd_info()


if __name__ == "__main__":
    raise SystemExit(main())
