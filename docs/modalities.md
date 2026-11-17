# Modality Recipes

X-FUSE uses a single modality-neutral interface across X-ray, electron, and neutron characterization. Below are recommended parameter recipes for common laboratory and beamline setups.

---

## 1. Synchrotron XCT + XRD-CT (2D & Sparse-Slice 2.5D)

- **Physical Regime**: High-resolution attenuation/phase-contrast XCT ($I_{\text{HR}}$, $0.5\text{–}2\,\mu\text{m}$ voxel) paired with pencil-beam XRD-CT ($G_{\text{LR}}$, $5\times\text{–}20\times$ coarser in-plane or acquired on only $3\text{–}25$ sparse $z$-slices).
- **Recommended Setup**:
  - Enable `VolumetricFuser` with `slab_thickness=3` (`2.5d_slab`) for training on sparse XRD-CT slices and `mode="triplane"` for full-volume 3D propagation.
  - Stream large HDF5/NeXus or Zarr reconstructions lazily without loading the entire 3D tomogram into RAM.

```python
import h5py
from x_fuse import XFuse, VolumetricFuser, suggest_next_slices

base = XFuse(backbone="dinov2_vits14", layers="multiscale", stride=4)
vol_fuser = VolumetricFuser(base)

with h5py.File("xct_tomogram.h5", "r") as hf:
    xct_lazy = hf["entry/reconstruction/data"]  # Streamed slice-by-slice!
    vol_fuser.fit_sparse_slices(xct_lazy, sparse_guides={10: xrd_z10, 50: xrd_z50}, epochs=20)
    vol_report = vol_fuser.predict_volume(xct_lazy, mode="triplane")

# Recommend the next 3 XRD-CT z-slices to scan during live beamtime:
plan = suggest_next_slices(vol_report, acquired_indices=[10, 50], k=3, min_spacing=8)
print("Next XRD-CT slices:", plan.recommended_indices)
```

---

## 2. SEM-BSE + Multi-Element EDS / EBSD

- **Physical Regime**: Backscattered electron micrograph ($I_{\text{HR}}$, $5\text{–}20\,\text{nm}$ pixel) paired with $15\text{–}20\,\text{kV}$ EDS elemental maps ($G_{\text{LR}}$, $\sim 1\text{–}2\,\mu\text{m}$ pear-shaped interaction volume) where higher-energy lines (e.g., Fe-K$\alpha$, Ca-K$\alpha$) have wider effective excitation volumes than low-energy lines (Mg-K$\alpha$, Si-K$\alpha$).
- **Recommended Setup**:
  - Set `learn_psf=True` so `ForwardPSFDownsampler` learns per-element Gaussian interaction-volume widths automatically.
  - Set `tile_size=512` for large-area stitched SEM mosaics ($2048^2\text{–}8192^2$).

```python
from x_fuse import XFuse

model = XFuse(
    backbone="dinov2_vitb14_reg",
    layers="multiscale",
    learn_psf=True,
    align_guide="translation",
    tile_size=512,
    tile_overlap=64,
)
report = model.fit_predict(bse_mosaic, eds_element_stack, epochs=30, mode="multiclass")
print("Learned per-element PSF sigmas:", report.diagnostics["psf_sigma_yx_hr_px"])
```

---

## 3. HAADF-STEM + Dose-Limited STEM-EDS / Core-Loss EELS

- **Physical Regime**: Rapid, low-dose HAADF survey image ($I_{\text{HR}}$) paired with a coarse-step STEM-EDS or EELS spectrum image ($G_{\text{LR}}$) susceptible to **sample stage drift** during long dwell times.
- **Recommended Setup**:
  - Load `.emd`, `.dm4`, or `.hspy` vendor files directly via `load_microscopy_file` (`pip install x-fuse[io]`).
  - Set `align_guide="rigid"` or `"translation"` to compensate sub-pixel specimen drift between the HAADF survey and chemical scan.
  - Run `suggest_2d_rois` on the resulting `UncertaintyReport` to target high-uncertainty precipitates for fine-step EELS re-acquisition.

```python
from x_fuse import XFuse, load_microscopy_file, suggest_2d_rois

haadf, _ = load_microscopy_file("survey_haadf.dm4")
ti_map, _ = load_microscopy_file("eds_Ti.emd")
mo_map, _ = load_microscopy_file("eds_Mo.emd")

model = XFuse(backbone="dinov2_vits14", align_guide="rigid", learn_psf=True)
report = model.fit_predict(haadf, [ti_map, mo_map], epochs=25)

rois = suggest_2d_rois(report, roi_size=(64, 64), k=3)
```

---

## 4. White-Beam Neutron Imaging + ToF Bragg-Edge Mapping

- **Physical Regime**: High-flux white-beam neutron attenuation radiograph/tomogram ($I_{\text{HR}}$) fused with wavelength-resolved Time-of-Flight (ToF) Bragg-edge phase fraction or strain maps ($G_{\text{LR}}$, e.g., fcc/bcc Iron phase transformation or multi-metal powder phantoms).
- **Recommended Setup**:
  - Pay special attention to `report.physical_residual` and `report.unresolved_mask`: crystallographic texture or phase transitions (such as martensitic vs. austenitic steel) can have nearly identical white-beam attenuation in $I_{\text{HR}}$, which X-FUSE's forward-PSF residual explicitly flags rather than hallucinating false sharp boundaries.
