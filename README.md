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

## Key Capabilities (v0.4.0)

1. **100% Self-Contained Pure-PyTorch Package & 3-Line API**: Zero vendored external sub-repos or custom C++/CUDA compilation required. Run end-to-end normalized, padded, and calibrated correlative fusion in three lines via `XFuse(...).fit_predict(hr_image, coarse_guide)`.
2. **Multi-Scale ViT Pyramid, Megapixel Tiling & TTA Caching**: Concatenate shallow boundary tokens with deep semantic tokens (`layers="multiscale"`), cache frozen TTA features across training epochs (`~10x–20x` speedup), and tile megapixel micrographs seamlessly via raised-cosine Hann blending (`tile_size=...`).
3. **Differentiable Sub-Pixel Drift & Anisotropic PSF Learning**: Jointly compensate sample stage/beam drift (`align_guide="translation" | "rigid" | "affine"`) in the canonical HR frame while learning per-channel Gaussian interaction-volume PSF widths (`learn_psf=True`).
4. **Tri-Part Certainty & Active Beamtime Steering**:
   - **Chemical / Aleatoric Ambiguity**: Prior-calibrated posterior entropy and top-1 vs. top-2 probability margin.
   - **Spatial / Epistemic Variance**: Online Welford variance across TTA views and orthogonal 2.5D planes in $O(1)$ extra memory.
   - **Physical Forward-Model Residual**: $|\mathcal{D}(\hat{P}_{\text{HR}}) - G_{\text{LR}}|$, flagging contrast-invisible phases, paired with `suggest_next_slices` and `suggest_2d_rois` for live experiment steering.
5. **Out-of-Core SuRVoS 2.5D Mode & Scientific I/O**: Lazy HDF5 / Zarr / TIFF-stack slice streaming (`VolumeSliceAccessor`), sparse-slice 2D → 3D propagation (`VolumetricFuser`), OME-TIFF / HDF5 / Zarr report export (`save_uncertainty_report`), and RosettaSciIO vendor microscopy readers (`.emd`, `.dm3`/`.dm4`, `.bcf`, `.hspy`).

---

## Installation

Using [`uv`](https://docs.astral.sh/uv/) (recommended):

```bash
git clone https://github.com/BASE-Laboratory/X-FUSE.git
cd X-FUSE

# CPU + microscopy I/O + dev + docs extras (Linux / macOS)
uv sync --extra cpu --extra io --extra dev --extra docs \
  --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu

# Or GPU (CUDA 12.1 on Linux)
uv sync --extra cuda --extra io --extra dev
```

See [INSTALL.md](INSTALL.md) for optional extras (`io`, `docs`, `synthetic`, `sam`, `notebooks`) and platform details.

---

## Quickstart

### 1. 3-Line 2D Multi-Channel Correlative Fusion + Drift/PSF Recovery

```python
from x_fuse import XFuse, generate_correlative_phantom_2d, save_uncertainty_report, suggest_2d_rois

phantom = generate_correlative_phantom_2d(size=128, downsample_factor=8)
model = XFuse(
    backbone="dinov2_vits14",
    layers="multiscale",
    align_guide="translation",
    learn_psf=True,
    stride=4,
)
report = model.fit_predict(phantom.hr_image, phantom.coarse_guide, epochs=25, mode="multiclass")

phase_mask = report.hard_mask[0]                             # (H, W) int64 phase IDs
certainty = 1.0 - report.combined_uncertainty[0]             # (H, W) pixel certainty in [0, 1]
rois = suggest_2d_rois(report, roi_size=(32, 32), k=2)       # Targeted re-acquisition windows
save_uncertainty_report(report, "fused_report.ome.tif", pixel_size_um=0.05)
```

*(Low-level hooks `fuser.set_guide_transforms(...)`, `fuser.train_fusion_step(...)`, and `fuser.predict_with_uncertainty(...)` remain 100% supported for custom research loops.)*

### 2. SuRVoS-Style 2.5D Sparse-Slice → 3D Tomogram + Active Beamtime Steering

```python
from x_fuse import XFuse, VolumetricFuser, generate_correlative_phantom_3d, suggest_next_slices

phantom_3d = generate_correlative_phantom_3d(shape=(32, 64, 64), downsample_factor=4)
vol_fuser = VolumetricFuser(XFuse(backbone="dinov2_vits14", stride=4))

# Train on just 3 sparse 2D chemical/diffraction slices out of Z=32 (supports lazy HDF5/Zarr!)
sparse_slices = {4: phantom_3d.coarse_guide_3d[:, 1], 16: phantom_3d.coarse_guide_3d[:, 4], 28: phantom_3d.coarse_guide_3d[:, 7]}
vol_fuser.fit_sparse_slices(phantom_3d.hr_volume, sparse_guides=sparse_slices, epochs=15, delta=1)

# Propagate across the entire 3D volume via orthogonal XY + XZ + YZ tri-planes
vol_report = vol_fuser.predict_volume(phantom_3d.hr_volume, mode="triplane", pred_mode="multiclass")
plan = suggest_next_slices(vol_report, acquired_indices=list(sparse_slices), k=3, min_spacing=4)
print("Next optimal z-slices to scan:", plan.recommended_indices)
```

### 3. Reproducible Paper Benchmarks, Figures & Open Datasets

```bash
# Run synthetic super-resolution, invisible-phase residual, sparsification & 2.5D suite
uv run python -m benchmarks.run_synthetic_suite --quick

# Run paper ablation suite + vector PDF/PNG figure generation (multi-scale, drift/PSF, active 3D, Zn13X/MOR-GIS)
uv run python -m benchmarks.run_paper_benchmarks --quick --output-dir benchmarks/outputs/paper_quick

# Run full unit test suite (>=82% coverage gate)
uv run pytest -m "not gpu" --cov=x_fuse --cov-fail-under=82 -v
```

Fetch registered open correlative datasets from Zenodo (`edxia` SEM-EDS, HAADF-STEM-EDS/EELS, ISIS IMAT Bragg-Edge):

```python
from x_fuse import fetch_open_dataset, list_open_datasets

print(list(list_open_datasets().keys()))
# paths = fetch_open_dataset("ti_rcca_stem_eds")
```

---

## Command-Line Interface & Tutorials

Once installed, the `xfuse` CLI provides environment diagnostics, Zenodo dataset discovery, synthetic smoke checks, and zero-code batch correlative fusion:

```bash
xfuse info
xfuse datasets
xfuse demo --epochs 15
xfuse fuse --hr data/samples/Zn13X/high_4.tiff --guide data/samples/Zn13X/low_4.tiff --output zn13x.ome.tif
```

Interactive step-by-step notebooks (stored with outputs stripped) are available in [`notebooks/`](notebooks/):
- `notebooks/01_correlative_2d_and_uncertainty.ipynb` — 2D multi-channel correlative fusion, JBU comparison, and tri-part pixel uncertainty.
- `notebooks/02_volumetric_25d_propagation.ipynb` — SuRVoS-style 2.5D slab packing, sparse $z$-slice training, and 3D tri-plane propagation.
- `notebooks/03_active_acquisition_and_registration.ipynb` — 3-line `XFuse.fit_predict`, sub-pixel drift/PSF recovery, active experiment steering, and OME-TIFF/HDF5 export.

---

## Repository Layout

```text
X-FUSE/
├── x_fuse/
│   ├── backbones.py      # Multi-layer ViT loader, sub-patch TTA & tiled extractor
│   ├── fusion.py         # XFuse / CorrelativeFuser, 3-line API, TTA cache & gating
│   ├── registration.py   # Differentiable sub-pixel translation/rigid/affine aligner
│   ├── tiling.py         # Hann-windowed overlapping 2D tile inference engine
│   ├── upsamplers.py     # Scale-dilated JBU & learnable anisotropic ForwardPSFDownsampler
│   ├── uncertainty.py    # Online Welford TTA variance, entropy, physical residual, ECE, AUSE
│   ├── volumetric.py     # Lazy HDF5/Zarr 2.5D slice streaming, sparse 2D->3D & tri-plane
│   ├── active.py         # Active sparse-slice (`suggest_next_slices`) & 2D ROI steering
│   ├── io.py             # OME-TIFF / HDF5 / Zarr report I/O & RosettaSciIO vendor readers
│   ├── segment.py        # Sign-aligned PCA+Otsu, JBU/bicubic/KMeans baselines, BF1, SAM2
│   ├── synthetic.py      # Controlled 2D & 3D multi-phase correlative phantom generators
│   ├── datasets.py       # Bundled sample locator, Zenodo fetcher & experimental hold-out tool
│   ├── loaders.py        # XCT/XRD-CT HDF5/NeXus, SEM-EDS hypermaps & generic TIFF pairs
│   ├── transforms.py     # Invertible sub-patch TTA & float32-native structural normalization
│   ├── vit_patch.py      # ViT stride / pos-embed patching & TorchPCA
│   └── cli.py            # `xfuse` CLI (`info`, `datasets`, `demo`, `fuse`)
├── docs/                 # MkDocs Material documentation (theory, modality recipes, API)
├── benchmarks/           # Synthetic & paper ablation runners, Zenodo adapters & PDF plotters
├── notebooks/            # Clean, output-stripped tutorial notebooks (01, 02, 03)
├── data/samples/         # Bundled Zn13X & MOR-GIS sample correlative slices
├── tests/                # Comprehensive CPU & CUDA GPU pytest suite (>=82% coverage gate)
└── Makefile              # Developer shortcuts: sync, format, lint, typecheck, docs-check, test
```

## Contributing, Citation & License

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for developer setup (`make lint`, `make typecheck`, `make test`) and [`CITATION.cff`](CITATION.cff) for software citation metadata. Released under the [MIT License](LICENSE) by the **BASE Laboratory**. Self-contained ViT patching, ALiBi attention, and JBU modules acknowledge inspiration and adapted components from [`HR-Dv2`](https://github.com/tldr-group/HR-Dv2), [`dino-saw`](https://github.com/tldr-group/dino-saw), and [`FeatUp`](https://github.com/mhamilton723/FeatUp).
