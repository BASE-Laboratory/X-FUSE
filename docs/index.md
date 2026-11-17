# X-FUSE

**Physics-Guided Foundation-Model Feature Fusion & Uncertainty-Aware Segmentation for Correlative Materials Imaging**

Across electron, X-ray, and neutron characterization, experimentalists face a fundamental trade-off between **spatial resolution** and **chemical or crystallographic specificity**:

- **High-resolution structural modalities ($I_{\text{HR}}$)** — such as synchrotron XCT, SEM-BSE, BF-TEM / HAADF-STEM, and white-beam neutron radiography — capture sharp nanoscale-to-microscale morphology rapidly, but suffer from overlapping attenuation or atomic-number contrast.
- **Coarse or sparse physical guide modalities ($G_{\text{LR}}$)** — such as XRD-CT, SEM/STEM-EDS, EELS, EBSD, and time-of-flight Bragg-edge imaging — resolve exact elemental or phase identity, but are $4\times\text{–}32\times$ coarser or restricted to a handful of sparse 2D slices due to interaction-volume blur, dose limits, or photon/neutron flux.

**X-FUSE** bridges these modalities in a **100% self-contained pure-PyTorch package**. It extracts dense, multi-scale sub-patch Vision Transformer features from $I_{\text{HR}}$, fits a physics-informed fusion head against $G_{\text{LR}}$ through a differentiable forward point-spread-function (PSF) operator and sub-pixel drift aligner, and outputs **super-resolved phase segmentations paired with pixel/voxel-level certainty maps**.

---

## Core Capabilities

1. **3-Line High-Level API (`XFuse.fit_predict`)**: Automatic structural normalization, spatial padding/cropping, frozen TTA feature caching, and multi-channel calibration in three lines of Python.
2. **Megapixel Tiling & Out-of-Core 3D Streaming**: Raised-cosine Hann-windowed 2D overlapping tile inference (`tile_size=...`) and lazy HDF5 / Zarr / TIFF-stack 2.5D slice streaming (`VolumeSliceAccessor`) for gigavoxel synchrotron tomograms.
3. **Multi-Scale ViT Pyramid & Differentiable Co-Registration**: Concatenate early boundary tokens with deep semantic tokens (`layers="multiscale"`), jointly refine sub-pixel guide drift (`align_guide="translation" | "rigid" | "affine"`), and learn per-channel anisotropic interaction-volume PSF widths (`learn_psf=True`).
4. **Tri-Part Certainty & Active Beamtime Steering**: Separate **chemical entropy**, **online Welford TTA / tri-plane epistemic variance**, and **forward physical PSF residual**, and automatically schedule optimal sparse $z$-slices (`suggest_next_slices`) or 2D re-scan windows (`suggest_2d_rois`).
5. **Standardized Scientific I/O**: Export and reload uncertainty reports as multi-channel **OME-TIFF** (`.ome.tif`), **HDF5/NeXus** (`.h5`), or **Zarr** (`.zarr`), and read vendor `.emd`, `.dm3`/`.dm4`, `.bcf`, and `.hspy` archives via optional RosettaSciIO (`x-fuse[io]`).

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
    layers="multiscale",
    align_guide="translation",
    learn_psf=True,
)
report = model.fit_predict(phantom.hr_image, phantom.coarse_guide, epochs=25)

save_uncertainty_report(report, "fused_output.ome.tif", pixel_size_um=0.05)
```

Continue to [Mathematical Theory](theory.md) for the forward-physics and uncertainty formulation, [Modality Recipes](modalities.md) for domain-specific configurations, or [Tutorials & CLI](tutorials.md) for batch workflows.
