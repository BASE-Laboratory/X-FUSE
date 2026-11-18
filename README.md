# X-FUSE: Physics-Guided Foundation-Model Feature Fusion & Uncertainty-Aware Segmentation for Correlative Materials Imaging

[![CI](https://github.com/BASE-Laboratory/X-FUSE/actions/workflows/ci.yml/badge.svg)](https://github.com/BASE-Laboratory/X-FUSE/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Typed](https://img.shields.io/badge/typing-PEP%20561-informational.svg)](x_fuse/py.typed)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**X-FUSE** bridges the ubiquitous resolution-vs.-specificity trade-off in materials characterization. Across electron, X-ray, and neutron imaging, researchers routinely acquire a **high-resolution structural image** ($I_{\text{HR}}$, fast & sharp, but chemically ambiguous) alongside a **coarser or sparse physical/chemical guide modality** ($G_{\text{LR}}$, chemically or crystallographically specific, but 4×–32× coarser or restricted to sparse 2D slices).

Unlike purely geometric foundation models (such as SAM/SAM2) that segment contrast boundaries without knowing chemical identity—or naive interpolation of coarse chemical maps—**X-FUSE** extracts dense sub-patch Vision Transformer features from $I_{\text{HR}}$, fuses $M$ physical guide channels simultaneously via a differentiable forward point-spread-function (PSF) operator, and outputs **prior-matched high-resolution phase segmentations paired with pixel/voxel-level certainty maps and optional Split Conformal prediction sets**.

---

## Supported Correlative Modality Pairs

| High-Resolution Structural Modality ($I_{\text{HR}}$) | Coarse / Sparse Physical Guide ($G_{\text{LR}}$) | Resolution Gap | Open Benchmark Key (`x_fuse.datasets`) |
| :--- | :--- | :--- | :--- |
| **Synchrotron XCT** (attenuation / phase contrast) | **XRD-CT** Bragg peak / Rietveld phase maps | 5×–20× (or sparse 2D slices in 3D) | Bundled `Zn13X` & `MOR-GIS` (`data/samples/`) |
| **SEM / BSE-SEM** | **Multi-Element EDS / EDX** & **EBSD** | 4×–16× interaction-volume blur | `edxia_cement_sem_eds`, `refractory_alloy_bse_eds_ebsd` |
| **BF-TEM / HAADF-STEM** | **STEM-EDS** & **Core-Loss EELS** | 4×–16× dose-limited coarse step | `ti_rcca_stem_eds`, `cuzn_stem_eels` |
| **White-Beam Neutron Imaging / CT** | **ToF Bragg-Edge** crystallographic phase / strain | 4×–12× flux-limited detector binning | `isis_imat_bragg_edge` |

---

## Key Capabilities (v1.0.0)

1. **Quantitative Physics Forward Model & Raw 3D Spectral Cube Unmixing**: Train directly against low-dose counting statistics via Poisson shot-noise deviance NLL (`loss_fn="forward_psf_poisson"`), enforce exact simplex conservation $\sum_k P_k = 1$ (`loss_fn="compositional_kl"`), or pass raw 3D `(E, H_lr, W_lr)` EDS / EELS / XRD spectral cubes directly (`spectral_unmix=True`) via differentiable SPA-seeded NMF (`SpectralUnmixingDictionary`).
2. **Texture-Aware Cross-Attention Upsampling & Apositional Feature Debiasing**: Separate iso-intensity textured phases (e.g., lamellar precipitates vs. homogeneous matrix) with `CrossAttentionFeatureUpsampler` (`interpolation_mode="cross_attn"`) + `SpatialConvMLPHead` (`fusion_method="spatial_mlp"`), and eliminate ViT coordinate-ramp shortcut learning on macroscopic chemical gradients via `ApositionalFeatureDebiaser` (`debias_positional=True`).
3. **Finite-Sample Split Conformal Prediction & Tri-Part Certainty**: Pair **aleatoric entropy**, **online $O(1)$-memory Welford TTA / tri-plane epistemic variance**, and **forward physical PSF residual** with distribution-free $(1 - \alpha)$ Split Conformal phase prediction sets (`conformal_alpha=0.10`, `ConformalCalibrator`) and active beamtime steering (`suggest_next_slices`, `suggest_2d_rois`).
4. **Differentiable Rigid, Affine & Cubic B-Spline Registration**: Jointly compensate rigid stage drift or nonlinear SEM/STEM charging scan-distortion (`align_guide="translation" | "rigid" | "affine" | "bspline"`) with thin-plate bending energy regularization and Jacobian diffeomorphism diagnostics, alongside learnable per-channel PSF widths (`learn_psf=True`).
5. **Megapixel Tiling, Lazy 2.5D Slice Streaming & Scientific I/O**: Pure-PyTorch package (pretrained DINOv2/ViT weights resolved reproducibly via `timm` or offline with `pretrained=False`), content-hashed frozen TTA feature caching across optimization steps, Hann-windowed 2D tiling (`tile_size=...`), lazy HDF5 / Zarr / TIFF-stack 2.5D input slice streaming (`VolumeSliceAccessor`, `VolumetricFuser`), OME-TIFF / HDF5 / Zarr report export, and RosettaSciIO vendor readers (`.emd`, `.dm3`/`.dm4`, `.bcf`, `.hspy`).

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

### 1. 3-Line 2D Multi-Channel Correlative Fusion + Conformal Prediction Sets

```python
from x_fuse import XFuse, generate_correlative_phantom_2d, save_uncertainty_report, suggest_2d_rois

phantom = generate_correlative_phantom_2d(size=128, downsample_factor=8)
model = XFuse(
    backbone="dinov2_vits14",
    interpolation_mode="cross_attn",
    fusion_method="spatial_mlp",
    debias_positional=True,
    layers="multiscale",
    align_guide="translation",
    learn_psf=True,
    conformal_alpha=0.10,
    stride=4,
)
report = model.fit_predict(phantom.hr_image, phantom.coarse_guide, epochs=25, mode="multiclass")

phase_mask = report.hard_mask[0]                             # (H, W) int64 phase IDs
phase_sets_90 = report.prediction_sets[0]                    # (K, H, W) 90% conformal phase sets
certainty = 1.0 - report.combined_uncertainty[0]             # (H, W) pixel certainty in [0, 1]
rois = suggest_2d_rois(report, roi_size=(32, 32), k=2)       # Targeted re-acquisition windows
save_uncertainty_report(report, "fused_report.h5", pixel_size_um=0.05)
```

*(Low-level hooks `fuser.set_guide_transforms(...)`, `fuser.train_fusion_step(...)`, and `fuser.predict_with_uncertainty(...)` remain 100% supported for custom research loops.)*

### 2. SuRVoS-Style 2.5D Sparse-Slice → 3D Tomogram + Active Beamtime Steering

```python
from x_fuse import XFuse, VolumetricFuser, generate_correlative_phantom_3d, suggest_next_slices

phantom_3d = generate_correlative_phantom_3d(shape=(32, 64, 64), downsample_factor=4)
vol_fuser = VolumetricFuser(XFuse(backbone="dinov2_vits14", stride=4))

# Train on just 3 sparse 2D chemical/diffraction slices out of Z=32 (streams 2D slabs lazily from HDF5/Zarr!)
sparse_slices = {4: phantom_3d.coarse_guide_3d[:, 1], 16: phantom_3d.coarse_guide_3d[:, 4], 28: phantom_3d.coarse_guide_3d[:, 7]}
vol_fuser.fit_sparse_slices(phantom_3d.hr_volume, sparse_guides=sparse_slices, epochs=15, delta=1)

# Propagate across the 3D volume via 2.5D slabs or orthogonal XY + XZ + YZ tri-planes
vol_report = vol_fuser.predict_volume(phantom_3d.hr_volume, mode="triplane", pred_mode="multiclass")
plan = suggest_next_slices(vol_report, acquired_indices=list(sparse_slices), k=3, min_spacing=4)
print("Next optimal z-slices to scan:", plan.recommended_indices)
```

### 3. Reproducible Paper Benchmarks, Competing Baselines & Open Datasets

*(Note: `--quick` runs a fast offline CPU smoke check with unpretrained `timm:vit_tiny_patch16_224`; pass `--backbone dinov2_vitb14_reg --pretrained --device cuda` for full pretrained evaluations.)*

```bash
# Run synthetic super-resolution, invisible-phase residual, sparsification & 2.5D smoke suite
uv run python -m benchmarks.run_synthetic_suite --quick

# Run v1.0 paper benchmark + competing field baselines (Multi-Otsu, CNMF, Gram-Schmidt, Guided TV, SIREN INR) + PDF figures
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m benchmarks.run_paper_benchmarks --quick --output-dir benchmarks/outputs/paper_quick

# Run full unit test suite (>=85% coverage gate)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run pytest -m "not gpu" --cov=x_fuse --cov-fail-under=85 -v
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
xfuse fuse --hr data/samples/Zn13X/high_4.tiff --guide data/samples/Zn13X/low_4.tiff \
  --upsampler cross_attn --fusion-method spatial_mlp --conformal-alpha 0.10 --output zn13x.h5
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
│   ├── backbones.py      # Multi-layer ViT loader, ALiBi/NoPE support, TTA & tiled extractor
│   ├── debias.py         # Apositional polynomial feature debiaser (`ApositionalFeatureDebiaser`)
│   ├── heads.py          # LinearProbeHead, SpatialConvMLPHead & O(HW) WindowedGuideCrossAttention
│   ├── physics.py        # Poisson NLL, simplex KL & 3D spectral cube NMF (`PhysicsForwardModel`)
│   ├── fusion.py         # Compositional CorrelativeFuser (`XFuse`), 3-line API & TTA cache
│   ├── registration.py   # Differentiable translation/rigid/affine & cubic B-spline aligner
│   ├── tiling.py         # Hann-windowed overlapping 2D tile inference engine
│   ├── upsamplers.py     # CrossAttentionFeatureUpsampler, scale-dilated JBU & ForwardPSFDownsampler
│   ├── uncertainty.py    # Split ConformalCalibrator, Welford TTA variance, entropy, residual, ECE
│   ├── volumetric.py     # Lazy HDF5/Zarr 2.5D slice streaming, sparse 2D->3D & tri-plane
│   ├── active.py         # Active sparse-slice (`suggest_next_slices`) & 2D ROI steering
│   ├── io.py             # OME-TIFF / HDF5 / Zarr report I/O & RosettaSciIO vendor readers
│   ├── segment.py        # Sign-aligned PCA+Otsu, JBU/bicubic/KMeans baselines, BF1, SAM2
│   ├── synthetic.py      # 2D/3D multiphase, textured iso-intensity, gradient & spectral phantoms
│   ├── datasets.py       # Bundled sample locator, Zenodo fetcher & experimental hold-out tool
│   ├── loaders.py        # XCT/XRD-CT HDF5/NeXus, SEM-EDS hypermaps & generic TIFF pairs
│   ├── transforms.py     # Invertible sub-patch TTA & float32-native structural normalization
│   ├── vit_patch.py      # ViT stride / pos-embed patching & TorchPCA
│   └── cli.py            # `xfuse` CLI (`info`, `datasets`, `demo`, `fuse`)
├── docs/                 # MkDocs Material documentation (theory, modality recipes, API)
├── benchmarks/           # Paper benchmarks, CNMF/GS/TV/INR field baselines & PDF plotters
├── notebooks/            # Clean, output-stripped tutorial notebooks (01, 02, 03)
├── data/samples/         # Bundled Zn13X & MOR-GIS sample correlative slices
├── tests/                # Comprehensive CPU & CUDA GPU pytest suite (>=85% coverage gate)
└── Makefile              # Developer shortcuts: sync, format, lint, typecheck, docs-check, test
```

## Contributing, Citation & License

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for developer setup (`make lint`, `make typecheck`, `make test`) and [`CITATION.cff`](CITATION.cff) for software citation metadata. Released under the [MIT License](LICENSE) by the **BASE Laboratory**. Self-contained ViT patching, ALiBi attention, and feature upsampling modules acknowledge inspiration and adapted components from [`HR-Dv2`](https://github.com/tldr-group/HR-Dv2), [`dino-saw`](https://github.com/tldr-group/dino-saw), [`FeatUp`](https://github.com/mhamilton723/FeatUp), and [`JAFAR`](https://github.com/PaulCouairon/JAFAR).
