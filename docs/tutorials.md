# Tutorials & Command-Line Interface

## Command-Line Interface (`xfuse`)

Installing `x-fuse` registers the `xfuse` command-line tool for environment verification, Zenodo dataset discovery, synthetic smoke testing, and zero-code batch correlative fusion.

### 1. Environment Diagnostics & Open Datasets

```bash
# Print JSON runtime summary (Python, PyTorch, CUDA availability)
xfuse info

# List registered open-access correlative datasets on Zenodo
xfuse datasets
xfuse datasets --json
```

### 2. Self-Contained Synthetic Smoke Benchmark

```bash
xfuse demo --epochs 15 --output demo_metrics.json
```

### 3. Batch Correlative Fusion (`xfuse fuse`)

Fuse a high-resolution structural micrograph (`.tif`, `.png`, `.npy`, `.emd`, `.dm4`, `.hspy`) with one or more coarse physical guide maps and export a calibrated multi-channel report to **OME-TIFF**, **HDF5**, or **Zarr**:

```bash
xfuse fuse \
  --hr data/samples/Zn13X/high_4.tiff \
  --guide data/samples/Zn13X/low_4.tiff \
  --output zn13x_fused.ome.tif \
  --backbone dinov2_vits14 \
  --layers multiscale \
  --align-guide translation \
  --learn-psf \
  --pixel-size-um 0.5
```

---

## Reproducible Paper Benchmarks & Figures

X-FUSE ships two reproducible benchmark suites under `benchmarks/`:

```bash
# 1. Core synthetic super-resolution, contrast-to-zero failure, ECE/AUSE & 2.5D suite
uv run python -m benchmarks.run_synthetic_suite --quick

# 2. Full paper ablation suite + publication PDF/PNG figure generator
#    (Multi-scale ViT ablation, sub-pixel drift/PSF recovery, active beamtime curve,
#     bundled Zn13X & tiled MOR-GIS synchrotron benchmarks)
uv run python -m benchmarks.run_paper_benchmarks --quick --output-dir benchmarks/outputs/paper_quick
```

On a GPU workstation with internet access, run the full pretrained DINOv2 + Zenodo hold-out evaluation:

```bash
uv run python -m benchmarks.run_paper_benchmarks \
  --backbone dinov2_vitb14_reg \
  --pretrained \
  --device cuda \
  --download-zenodo \
  --output-dir benchmarks/outputs/paper_full
```

---

## Interactive Jupyter Notebooks

Three clean, output-stripped notebooks are included in the repository's `notebooks/` directory:

1. **`01_correlative_2d_and_uncertainty.ipynb`** — 2D multi-channel correlative fusion, classical JBU comparison, prior calibration, and tri-part pixel uncertainty (`aleatoric_entropy`, `epistemic_variance`, `physical_residual`).
2. **`02_volumetric_25d_propagation.ipynb`** — SuRVoS-inspired `(z - Δz, z, z + Δz)` 2.5D slab packing, sparse $z$-slice probe fitting, and orthogonal `XY + XZ + YZ` tri-plane propagation.
3. **`03_active_acquisition_and_registration.ipynb`** — 3-line `XFuse.fit_predict` API, sub-pixel guide drift & anisotropic PSF recovery, active sparse-slice & 2D ROI steering (`suggest_next_slices`, `suggest_2d_rois`), and OME-TIFF / HDF5 report export.
