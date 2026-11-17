# X-FUSE: Physics-Guided Foundation-Model Feature Fusion & Uncertainty-Aware Segmentation for Correlative Materials Imaging

[![CI](https://github.com/BASE-Laboratory/X-FUSE/actions/workflows/ci.yml/badge.svg)](https://github.com/BASE-Laboratory/X-FUSE/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Typed](https://img.shields.io/badge/typing-PEP%20561-informational.svg)](x_fuse/py.typed)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**X-FUSE** bridges the ubiquitous resolution-vs.-specificity trade-off in materials characterization. Across electron, X-ray, and neutron imaging, researchers routinely acquire a **high-resolution structural image** ($I_{\text{HR}}$, fast & sharp, but chemically ambiguous) alongside a **coarser or sparse physical/chemical guide modality** ($G_{\text{LR}}$, chemically or crystallographically specific, but 4×–32× coarser or restricted to sparse 2D slices).

Unlike purely geometric foundation models (such as SAM/SAM2) that segment contrast boundaries without knowing chemical identity—or naive interpolation of coarse chemical maps—**X-FUSE** extracts dense sub-patch Vision Transformer features from $I_{\text{HR}}$, fuses $M$ physical guide channels simultaneously via a differentiable forward point-spread-function (PSF) operator, and outputs **calibrated high-resolution phase segmentations paired with pixel/voxel-level certainty maps**.

---

## Supported Correlative Modality Pairs

| High-Resolution Structural Modality ($I_{\text{HR}}$) | Coarse / Sparse Physical Guide ($G_{\text{LR}}$) | Resolution Gap | Open Benchmark Key (`x_fuse.datasets`) |
| :--- | :--- | :--- | :--- |
| **Synchrotron XCT** (attenuation / phase contrast) | **XRD-CT** Bragg peak / Rietveld phase maps | 5×–20× (or sparse 2D slices in 3D) | Bundled `Zn13X` & `MOR-GIS` (`data/samples/`) |
| **SEM / BSE-SEM** | **Multi-Element EDS / EDX** & **EBSD** | 4×–16× interaction-volume blur | `edxia_cement_sem_eds`, `refractory_alloy_bse_eds_ebsd` |
| **BF-TEM / HAADF-STEM** | **STEM-EDS** & **Core-Loss EELS** | 4×–16× dose-limited coarse step | `ti_rcca_stem_eds`, `cuzn_stem_eels` |
| **White-Beam Neutron Imaging / CT** | **ToF Bragg-Edge** crystallographic phase / strain | 4×–12× flux-limited detector binning | `isis_imat_bragg_edge` |

---

## Key Capabilities (v0.2.0)

1. **100% Self-Contained Pure-PyTorch Package**: Zero vendored external sub-repos or custom C++/CUDA compilation required. Includes native sub-patch ViT stride & positional-encoding patching (`DINOv2`, `DINOv3`, `ALiBi`, `NoPE`, and `timm` ViTs), invertible sub-patch test-time augmentation (TTA), and pure-PyTorch progressive **Joint Bilateral Upsampling (JBU)**.
2. **Multi-Channel Extract-Once Fusion & Forward-PSF Supervision**: Extracts dense HR ViT features once and fuses $M$ elemental/phase channels (`Ca, Si, Al, Fe, ...`) simultaneously, supervising sharp HR phase predictions via $\mathcal{D}(\hat{P}_{\text{HR}}) \approx G_{\text{LR}}$ rather than training against blurry upsampled targets.
3. **Tri-Part Pixel & Voxel Certainty Maps (`UncertaintyReport`)**:
   - **Chemical / Aleatoric Ambiguity**: Prior-calibrated posterior entropy and top-1 vs. top-2 probability margin.
   - **Spatial / Epistemic Variance**: Online Welford variance across TTA shifts/flips and orthogonal 2.5D planes in $O(1)$ extra memory.
   - **Physical Forward-Model Residual**: $|\mathcal{D}(\hat{P}_{\text{HR}}) - G_{\text{LR}}|$, automatically flagging **unresolved regions** where $I_{\text{HR}}$ lacks structural contrast for a real chemical gradient.
4. **SuRVoS-Inspired 2.5D Volumetric Mode (`VolumetricFuser`)**:
   - Zero-overhead `(z - Δz, z, z + Δz)` 3-slice RGB ViT input packing,
   - **Sparse-slice 2D → 3D chemical propagation** (train on a few 2D XRD-CT or FIB-EDS slices, segment the full 3D tomogram),
   - Orthogonal `XY + XZ + YZ` **tri-plane fusion** with inter-plane epistemic variance.

---

## Installation

Using [`uv`](https://docs.astral.sh/uv/) (recommended):

```bash
git clone https://github.com/BASE-Laboratory/X-FUSE.git
cd X-FUSE

# CPU + dev + synthetic PoreSpy benchmarks (Linux / macOS)
uv sync --extra cpu --extra dev --extra synthetic --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu

# Or GPU (CUDA 12.1 on Linux)
uv sync --extra cuda --extra dev --extra synthetic
```

See [INSTALL.md](INSTALL.md) for optional extras (`sam`, `notebooks`) and platform details.

---

## Quickstart

### 1. 2D Multi-Channel Correlative Fusion + Certainty Map

```python
import torch
from x_fuse import (
    CorrelativeFuser,
    build_default_tta,
    ensure_spatial_divisible,
    generate_correlative_phantom_2d,
    guide_to_tensor,
    normalize_structural,
)

# 1. Load or synthesize paired HR structural image + M-channel coarse physical guide
phantom = generate_correlative_phantom_2d(size=128, downsample_factor=8)
hr_tensor = normalize_structural(phantom.hr_image)           # float32-native (3, H, W)
hr_padded, (h, w) = ensure_spatial_divisible(hr_tensor, patch_size=14, stride=4)
guide_tensor = guide_to_tensor(phantom.coarse_guide)         # (1, M, H_lr, W_lr)

# 2. Initialize self-contained fuser (supports dinov2_vits14, dinov2_vitb14_reg, timm:*, etc.)
fwd_tta, inv_tta = build_default_tta(shift_dists=[1, 2], pattern="Neumann", include_flips=True)
fuser = CorrelativeFuser(
    dino_name="dinov2_vits14",
    xrd_fuse_method="learned_gating",
    loss_fn="forward_psf_bce",
    interpolation_mode="jbu",
    stride=4,
    dtype=torch.float32,
)
fuser.set_guide_transforms(guide_tensor, fwd_tta, inv_tta)

# 3. Fit lightweight linear probe against coarse physical measurements
for _ in range(20):
    loss = fuser.train_fusion_step(hr_padded)

# 4. Predict HR phase masks + calibrated pixel-level UncertaintyReport
report = fuser.predict_with_uncertainty(hr_padded, guide_lr=guide_tensor, mode="multiclass")
phase_mask = report.hard_mask[0, :h, :w]                     # (H, W) int64 phase IDs
certainty = 1.0 - report.combined_uncertainty[0, :h, :w]     # (H, W) pixel certainty in [0, 1]
unresolved = report.unresolved_mask[0, :h, :w]               # (H, W) bool physical mismatch flag
```

### 2. SuRVoS-Style 2.5D Sparse-Slice → 3D Tomogram Propagation

```python
from x_fuse import CorrelativeFuser, VolumetricFuser, generate_correlative_phantom_3d

phantom_3d = generate_correlative_phantom_3d(shape=(32, 64, 64), downsample_factor=4)
base = CorrelativeFuser("dinov2_vits14", loss_fn="forward_psf_bce", stride=4, dtype=torch.float32)
vol_fuser = VolumetricFuser(base)

# Train on just 3 sparse 2D chemical/diffraction slices out of Z=32!
sparse_slices = {4: phantom_3d.coarse_guide_3d[:, 1], 16: phantom_3d.coarse_guide_3d[:, 4], 28: phantom_3d.coarse_guide_3d[:, 7]}
vol_fuser.fit_sparse_slices(phantom_3d.hr_volume, sparse_guides=sparse_slices, epochs=15, delta=1)

# Propagate across the entire 3D volume via orthogonal XY + XZ + YZ tri-planes
vol_report = vol_fuser.predict_volume(phantom_3d.hr_volume, mode="triplane", pred_mode="multiclass")
print(vol_report.hard_mask.shape, "Z-consistency:", vol_report.z_slice_consistency)
```

### 3. Reproducible Benchmark Suite & Open Datasets

```bash
# Run synthetic super-resolution, invisible-phase residual, sparsification & 2.5D suite
uv run python -m benchmarks.run_synthetic_suite --quick

# Run full unit test suite
uv run pytest -m "not gpu" --cov=x_fuse -v
```

Fetch registered open correlative datasets from Zenodo (`edxia` SEM-EDS, HAADF-STEM-EDS/EELS, ISIS IMAT Bragg-Edge):

```python
from x_fuse import fetch_open_dataset, list_open_datasets

print(list(list_open_datasets().keys()))
# paths = fetch_open_dataset("ti_rcca_stem_eds")
```

---

## Command-Line Interface & Tutorials

Once installed, the `xfuse` CLI provides instant environment diagnostics, Zenodo dataset discovery, and a self-contained synthetic correlative smoke check:

```bash
xfuse info
xfuse datasets
xfuse demo --epochs 15
```

Interactive step-by-step notebooks (stored with outputs stripped) are available in [`notebooks/`](notebooks/):
- `notebooks/01_correlative_2d_and_uncertainty.ipynb` — 2D multi-channel correlative fusion, JBU comparison, and tri-part pixel uncertainty.
- `notebooks/02_volumetric_25d_propagation.ipynb` — SuRVoS-style 2.5D slab packing, sparse $z$-slice training, and 3D tri-plane propagation.

---

## Repository Layout

```text
X-FUSE/
├── x_fuse/
│   ├── backbones.py      # Self-contained ViT loader & dense sub-patch TTA extractor
│   ├── fusion.py         # CorrelativeFuser (XFuse), Forward-PSF gating & cross-attention
│   ├── upsamplers.py     # Pure-PyTorch scale-dilated JBU & ForwardPSFDownsampler
│   ├── uncertainty.py    # Online Welford TTA variance, entropy, physical residual, ECE, AUSE
│   ├── volumetric.py     # SuRVoS-style 2.5D slice packing, sparse 2D->3D & tri-plane fusion
│   ├── segment.py        # Sign-aligned PCA+Otsu, JBU/bicubic/KMeans baselines, BF1, SAM2 hook
│   ├── synthetic.py      # Controlled 2D & 3D multi-phase correlative phantom generators
│   ├── datasets.py       # Bundled sample locator, Zenodo fetcher & experimental hold-out tool
│   ├── loaders.py        # XCT/XRD-CT HDF5/NeXus, SEM-EDS hypermaps & generic TIFF pairs
│   ├── transforms.py     # Invertible sub-patch TTA & float32-native structural normalization
│   ├── vit_patch.py      # ViT stride / pos-embed patching & TorchPCA
│   └── cli.py            # `xfuse` CLI entrypoint
├── benchmarks/           # Reproducible quantitative benchmark CLI & summary generators
├── notebooks/            # Clean, output-stripped tutorial notebooks (2D + 2.5D)
├── data/samples/         # Bundled Zn13X & MOR-GIS sample correlative slices
├── tests/                # Comprehensive CPU & CUDA GPU pytest suite (>=80% coverage gate)
└── Makefile              # Developer shortcuts: sync, format, lint, typecheck, test, build
```

## Contributing, Citation & License

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for developer setup (`make lint`, `make typecheck`, `make test`) and [`CITATION.cff`](CITATION.cff) for software citation metadata. Released under the [MIT License](LICENSE) by the **BASE Laboratory**. Self-contained ViT patching, ALiBi attention, and JBU modules acknowledge inspiration and adapted components from [`HR-Dv2`](https://github.com/tldr-group/HR-Dv2), [`dino-saw`](https://github.com/tldr-group/dino-saw), and [`FeatUp`](https://github.com/mhamilton723/FeatUp).
