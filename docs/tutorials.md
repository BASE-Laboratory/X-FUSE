# Tutorials & Command-Line Interface

## Command-Line Interface (`xfuse`)

Installing `x-fuse` registers the `xfuse` command-line tool for environment verification, Zenodo dataset discovery, synthetic smoke testing, and zero-code batch correlative fusion.

### 1. Environment Diagnostics & Open Datasets

```bash
# Print JSON runtime summary (Python, PyTorch, CUDA availability)
xfuse info

# List or download registered open-access correlative datasets on Zenodo
xfuse datasets
xfuse datasets --json
xfuse datasets --fetch edxia_cement_sem_eds --dest ./zenodo_cache
```

### 2. Self-Contained Synthetic Smoke Benchmark

```bash
xfuse demo --epochs 15 --output demo_metrics.json
```

### 3. Batch 2D & Sparse-Slice 3D Correlative Fusion (`xfuse fuse` / `xfuse fuse-3d`)

Fuse a high-resolution 2D structural micrograph (`.tif`, `.png`, `.npy`, `.emd`, `.dm4`, `.hspy`) with one or more coarse physical guide maps and export a calibrated multi-channel report to **OME-TIFF**, **HDF5**, or **Zarr**:

```bash
xfuse fuse \
  --hr data/samples/Zn13X/high_4.tiff \
  --guide data/samples/Zn13X/low_4.tiff \
  --output zn13x_fused.ome.tif \
  --backbone dinov2_vits14 \
  --fusion-method spatial_mlp \
  --upsampler cross_attn \
  --psf-profile moffat \
  --learn-psf \
  --guide-residual \
  --pore-mask auto_otsu_fringe_guarded \
  --conformal-alpha 0.10 \
  --conformal-method raps \
  --pixel-size-um 0.5
```

Or propagate sparse 2D chemical/diffraction slices across a 3D structural tomogram with axial interaction-volume PSF deconvolution, active next-slice beamtime steering, and `OpenImpala` AMReX multi-phase geometry export:

```bash
xfuse fuse-3d \
  --hr-volume xct_tomogram.h5 \
  --guide-slice 4:xrd_z04.tif 28:xrd_z28.tif \
  --output fused_3d_report.h5 \
  --pore-mask auto_otsu_fringe_guarded \
  --axial-psf-sigma 1.8 \
  --learn-axial-psf \
  --triplane \
  --suggest-next 3 \
  --export-openimpala-dir ./openimpala_geom
```

---

## Reproducible Paper Benchmarks, Ablations & LaTeX Tables

X-FUSE ships reproducible benchmark suites and a deterministic LaTeX table generator under `benchmarks/`:

```bash
# 1. Core synthetic super-resolution, contrast-to-zero failure, ECE/AUSE & 2.5D suite
uv run python -m benchmarks.run_synthetic_suite --quick

# 2. Referee operating-regime evaluation (Regime A / B1 / C, Zn13X, Poisson & RAPS UQ)
uv run python -m benchmarks.run_referee_evaluation --output-dir benchmarks/outputs/referee_evaluation

# 3. Staged v1.0.0 architectural, volumetric active-steering & tri-modality Zenodo ablations
uv run python -m benchmarks.run_design_ablations --output-dir benchmarks/outputs/design_ablations

# 4. Multi-length K11 DIAD Zn-13X XCT + XRD-CT coupling, VoDaSuRe gap & OpenImpala export
uv run python -m benchmarks.run_zn13x_multiscale_couplings --output-dir benchmarks/outputs/zn13x_multiscale

# 5. Export ready-to-compile booktabs LaTeX manuscript tables from benchmark JSONs
uv run python -m benchmarks.export_latex_tables --out-dir benchmarks/outputs/tables
```

---

## Interactive Jupyter Notebooks

Three clean, output-stripped notebooks are included in the repository's `notebooks/` directory:

1. **`01_correlative_2d_and_uncertainty.ipynb`** — 2D multi-channel correlative fusion, classical JBU comparison, prior calibration, and tri-part pixel uncertainty (`aleatoric_entropy`, `epistemic_variance`, `physical_residual`).
2. **`02_volumetric_25d_propagation.ipynb`** — SuRVoS-inspired `(z - Δz, z, z + Δz)` 2.5D slab packing, sparse $z$-slice probe fitting, and orthogonal `XY + XZ + YZ` tri-plane propagation.
3. **`03_active_acquisition_and_registration.ipynb`** — 3-line `XFuse.fit_predict` API, sub-pixel guide drift & anisotropic PSF recovery, active sparse-slice & 2D ROI steering (`suggest_next_slices`, `suggest_2d_rois`), and OME-TIFF / HDF5 report export.
