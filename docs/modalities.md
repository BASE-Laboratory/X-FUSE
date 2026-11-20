# Modality Recipes

X-FUSE uses a single modality-neutral interface across X-ray, electron, and neutron characterization. Below are recommended parameter recipes for common laboratory and beamline setups.

---

## 1. Synchrotron XCT + XRD-CT (2D & Sparse-Slice 2.5D)

- **Physical Regime**: High-resolution attenuation/phase-contrast XCT ($I_{\text{HR}}$, $0.5\text{–}2\,\mu\text{m}$ voxel) paired with pencil-beam XRD-CT ($G_{\text{LR}}$, $5\times\text{–}20\times$ coarser in-plane or acquired on only $3\text{–}25$ sparse $z$-slices).
- **Recommended Setup**:
  - Enable `VolumetricFuser` with `delta=1` (`2.5d_slab`, 3-slice `[z-1, z, z+1]` context) for training on sparse XRD-CT slices and `mode="triplane"` for full-volume 3D propagation.
  - Stream large HDF5/NeXus or Zarr reconstructions lazily without loading the entire 3D tomogram into RAM.

```python
import h5py
from x_fuse import XFuse, VolumetricFuser, export_for_openimpala, suggest_next_slices

base = XFuse(
    backbone="dinov2_vits14",
    layers="multiscale",
    stride=4,
    pore_mask="auto_otsu_fringe_guarded",
    conformal_alpha=0.10,
    conformal_method="raps",
)
vol_fuser = VolumetricFuser(base)

with h5py.File("xct_tomogram.h5", "r") as hf:
    xct_lazy = hf["entry/reconstruction/data"]  # Streamed slice-by-slice!
    vol_fuser.fit_sparse_slices(
        xct_lazy,
        sparse_guides={10: xrd_z10, 50: xrd_z50},
        epochs=20,
        pore_mask="auto_otsu_fringe_guarded",
        depth_psf_sigma=1.8,
        learn_depth_psf=True,
        triplane_consistency_weight=0.15,
    )
    vol_report = vol_fuser.predict_volume(
        xct_lazy,
        mode="triplane",
        sparse_guides={10: xrd_z10, 50: xrd_z50},
        pred_mode="multiclass",
        pore_mask="auto_otsu_fringe_guarded",
    )

# Export multi-phase uint8 geometry + conformal envelopes for OpenImpala AMReX PDE transport:
export_for_openimpala(vol_report, "./openimpala_geom", name="zeolite_pellet", active_phase=0)

# Recommend the next 3 XRD-CT z-slices to scan during live beamtime (1D Voronoi steering):
plan = suggest_next_slices(vol_report, acquired_indices=[10, 50], k=3, min_spacing=8)
print("Next XRD-CT slices:", plan.recommended_indices)
```

---

## 2. SEM-BSE + Multi-Element EDS / EBSD

- **Physical Regime**: Backscattered electron micrograph ($I_{\text{HR}}$, $5\text{–}20\,\text{nm}$ pixel) paired with $15\text{–}20\,\text{kV}$ EDS elemental maps ($G_{\text{LR}}$, $\sim 1\text{–}2\,\mu\text{m}$ pear-shaped interaction volume) where a narrow primary beam core is surrounded by a wide backscattered X-ray excitation skirt, and higher-energy lines (e.g., Fe-K$\alpha$, Ca-K$\alpha$) excite wider volumes than low-energy lines (Mg-K$\alpha$, Si-K$\alpha$).
- **Recommended Setup**:
  - Set `psf_profile="core_halo"` and `learn_psf=True` so `ForwardPSFDownsampler` models both the narrow primary beam core and wide backscattered excitation halo per element.
  - Set `loss_fn="compositional_kl"` with `guide_residual=True` when fusing calibrated wt%/at% EDS maps on the simplex.
  - Set `tile_size=512` for large-area stitched SEM mosaics ($2048^2\text{–}8192^2$).

```python
from x_fuse import XFuse

model = XFuse(
    backbone="dinov2_vitb14_reg",
    interpolation_mode="cross_attn",
    fusion_method="spatial_mlp",
    layers="multiscale",
    loss_fn="compositional_kl",
    psf_profile="core_halo",
    learn_psf=True,
    guide_residual=True,
    align_guide="translation",
    tile_size=512,
    tile_overlap=64,
)
report = model.fit_predict(bse_mosaic, eds_element_stack, epochs=30, mode="multiclass")
print("Learned per-element PSF sigmas:", report.diagnostics["psf_sigma_yx_hr_px"])
```

---

## 3. HAADF-STEM + Dose-Limited STEM-EDS / Core-Loss EELS

- **Physical Regime**: Rapid, low-dose HAADF survey image ($I_{\text{HR}}$) paired with a coarse-step STEM-EDS or 3D core-loss EELS spectrum image ($G_{\text{LR}}$) susceptible to **sample stage drift** and small nanoscale fields of view (`<112 px`).
- **Recommended Setup**:
  - Load `.emd`, `.dm4`, or `.hspy` vendor files directly via `load_microscopy_file` (`pip install x-fuse[io]`).
  - For raw 3D `(E, H_lr, W_lr)` EELS/EDS cubes, enable `loss_fn="forward_psf_poisson"`, `spectral_unmix=True`, and `guide_residual=True`; `min_input_size="auto"` automatically bicubic-upscales small nanoscale survey crops before ViT extraction to prevent token starvation.
  - Set `align_guide="rigid"` or `"translation"` to compensate sub-pixel specimen drift, and run `suggest_2d_rois` to target high-uncertainty precipitates for fine-step EELS re-acquisition.

```python
from x_fuse import XFuse, load_microscopy_file, suggest_2d_rois

haadf, _ = load_microscopy_file("survey_haadf.dm4")
ti_map, _ = load_microscopy_file("eds_Ti.emd")
mo_map, _ = load_microscopy_file("eds_Mo.emd")

model = XFuse(
    backbone="dinov2_vits14",
    interpolation_mode="cross_attn",
    fusion_method="spatial_mlp",
    loss_fn="forward_psf_bce",
    align_guide="rigid",
    psf_profile="moffat",
    learn_psf=True,
    guide_residual=True,
    conformal_alpha=0.10,
    conformal_method="raps",
)
report = model.fit_predict(haadf, [ti_map, mo_map], epochs=25)

rois = suggest_2d_rois(report, roi_size=(64, 64), k=3)
```

---

## 4. White-Beam Neutron Imaging + ToF Bragg-Edge Mapping

- **Physical Regime**: High-flux white-beam neutron attenuation radiograph/tomogram ($I_{\text{HR}}$) fused with wavelength-resolved Time-of-Flight (ToF) Bragg-edge phase fraction or strain maps ($G_{\text{LR}}$, e.g., fcc/bcc Iron phase transformation or multi-metal powder phantoms).
- **Recommended Setup**:
  - Pay special attention to `report.physical_residual` and `report.unresolved_mask`: crystallographic texture or phase transitions (such as martensitic vs. austenitic steel) can have nearly identical white-beam attenuation in $I_{\text{HR}}$, which X-FUSE's forward-PSF residual explicitly flags rather than hallucinating false sharp boundaries.
