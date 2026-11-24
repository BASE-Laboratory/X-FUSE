# X-FUSE

**Physics-Guided Foundation-Model Feature Fusion & Uncertainty-Aware Segmentation for Correlative Materials Imaging**

Across electron, X-ray, and neutron characterization, experimentalists face a fundamental trade-off between **spatial resolution** and **chemical or crystallographic specificity**:

- **High-resolution structural modalities ($I_{\text{HR}}$)** — such as synchrotron XCT, SEM-BSE, BF-TEM / HAADF-STEM, and white-beam neutron radiography — capture sharp nanoscale-to-microscale morphology rapidly, but suffer from overlapping attenuation or atomic-number contrast.
- **Coarse or sparse physical guide modalities ($G_{\text{LR}}$)** — such as XRD-CT, SEM/STEM-EDS, EELS, EBSD, and time-of-flight Bragg-edge imaging — resolve exact elemental or phase identity, but are $4\times\text{–}32\times$ coarser or restricted to a handful of sparse 2D slices due to interaction-volume blur, dose limits, or photon/neutron flux.

**X-FUSE** bridges these modalities in a **pure-PyTorch package with zero custom C++/CUDA compilation or vendored sub-repositories** (pretrained DINOv2/ViT weights are resolved reproducibly via `timm` and cached locally, or bypassed offline with `pretrained=False`). It extracts dense, multi-scale sub-patch Vision Transformer features from $I_{\text{HR}}$, fits a physics-informed fusion head against $G_{\text{LR}}$ through a differentiable forward point-spread-function (PSF) operator and sub-pixel drift aligner, and outputs **prior-matched super-resolved phase segmentations paired with pixel/voxel-level certainty maps and optional Split Conformal prediction sets**.

---

## Core Capabilities

1. **3-Line High-Level API (`XFuse.fit_predict`)**: Automatic structural normalization, spatial padding/cropping, content-hashed frozen TTA feature caching, Spin-2 `CrossAttentionFeatureUpsampler` + `SpatialConvMLPHead`, `ApositionalFeatureDebiaser`, and Split Conformal RAPS/APS/LAC calibration in three lines of Python.
2. **Quantitative Forward Physics, Porous-Media Void Clamping & 3D Spectral Cubes**: Train directly via `forward_psf_poisson`, `compositional_kl`, or `forward_psf_bce` with Gaussian / Moffat / Core+Halo PSFs, Fresnel-fringe-guarded structural void clamping (`pore_mask="auto_otsu_fringe_guarded"`), and raw 3D `(E, H_lr, W_lr)` spectral cube unmixing (`spectral_unmix=True`).
3. **Megapixel Tiling, Out-of-Core Synchrotron Streaming & Differentiable Registration**: Hann-windowed 2D overlapping tile inference (`tile_size=...`), out-of-core Savu HDF5 / DAWN NeXus / Zarr streaming (`load_xct_subvolume`, `load_diad_xrdct`, `VolumeSliceAccessor`), axial depth-PSF 3D deconvolution, and sub-pixel `translation` / `rigid` / `affine` / `bspline` registration.
4. **Tri-Part Certainty & 1D Voronoi Active Beamtime Steering**: Separate **chemical entropy**, **online Welford TTA / tri-plane epistemic variance**, and **forward physical PSF residual**, and schedule optimal sparse $z$-slices (`suggest_next_slices`) or 2D re-scan windows (`suggest_2d_rois`).
5. **Standardized Scientific I/O & OME-Zarr v0.4 Export**: Export and reload uncertainty reports as calibrated **OME-Zarr v0.4** (`.ome.zarr` via `export_report_ome_zarr` / `export_volume_ome_zarr` on `basecoat`), **OME-TIFF** (`.ome.tif`), **HDF5/NeXus** (`.h5`), **Zarr** (`.zarr`), or **NumPy** (`.npz`), including mode-downsampled conformal phase envelopes (`phase_lower`, `phase_upper`) for Layer-3 solvers, and read vendor `.emd`, `.dm3`/`.dm4`, `.bcf`, and `.hspy` archives (`x-fuse[io]`).

---

## Installation

Using [`uv`](https://docs.astral.sh/uv/) (recommended):

```bash
git clone https://github.com/BASE-Laboratory/X-FUSE.git
cd X-FUSE

# Install core package + microscopy I/O + developer & docs extras (CPU)
uv sync --extra cpu --extra io --extra dev --extra docs \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu

# Or on a Linux CUDA workstation:
uv sync --extra cuda --extra io --extra dev
```

Or with `pip`:

```bash
pip install -e ".[io,dev]"
```

---

## 3-Line Quickstart

```python
from x_fuse import XFuse, generate_correlative_phantom_2d, save_uncertainty_report

phantom = generate_correlative_phantom_2d(size=128, downsample_factor=8)
model = XFuse(
    backbone="dinov2_vits14",
    interpolation_mode="cross_attn",
    fusion_method="spatial_mlp",
    loss_fn="forward_psf_bce",
    layers="multiscale",
    align_guide="translation",
    learn_psf=True,
)
report = model.fit_predict(phantom.hr_image, phantom.coarse_guide, epochs=25)

save_uncertainty_report(report, "fused_output.ome.tif", pixel_size_um=0.05)
```

Continue to [Mathematical Theory](theory.md) for the forward-physics and uncertainty formulation, [Benchmarks & Regimes](benchmarks.md) for empirical operating regimes, [Modality Recipes](modalities.md) for domain-specific configurations, or [Tutorials & CLI](tutorials.md) for batch workflows.
