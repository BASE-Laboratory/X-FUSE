# AGENTS.md — `X-FUSE` (Layer 2: Image Tools)

Read [`ECOSYSTEM.md`](https://github.com/BASE-Laboratory/basecoat/blob/main/ECOSYSTEM.md) before modifying repository boundaries, I/O contracts, or cross-package dependencies.

## Architectural Role & Licence
- **Layer:** Layer 2 — Image tools (`BASE-Laboratory` 4-layer software architecture).
- **Licence:** `MIT` (`Copyright (c) 2025-2026 Calum Green and BASE Laboratory`, plus MIT third-party attributions in `LICENSE`). Keep `LICENSE` intact. `X-FUSE` may import `basecoat` (`BSD-3-Clause`), **never** copy code from `X-FUSE` into `basecoat`.
- **Owns:** Physics-guided foundation-model feature fusion (`XFuse`, `CorrelativeFuser`, `VolumetricFuser`), differentiable sub-pixel/B-spline guide alignment (`DifferentiableGuideAligner`), split/RAPS conformal calibration & uncertainty quantification (`UncertaintyReport`, `VolumetricUncertaintyReport`), active slice/ROI acquisition (`suggest_next_slices`, `suggest_2d_rois`), and multimodal microscopy/diffraction loaders.
- **Delegates to `basecoat` (Layer 1):** Calibrated `Volume` / `LazyVolume` / `VolumeSet` containers, OME-Zarr v0.4 multiscales + mode-downsampled phase/conformal label export (`export_report_ome_zarr`, `export_volume_ome_zarr`), conservative ring/cupping filters (`remove_rings_polar_conservative`, `detrend_radial_cupping`), out-of-core Savu `(X, Z, Y)` HDF5 slicing (`open_savu_h5`), standard TIFF/NPY volume I/O, SHA-256 dataset fetching (`fetch_file`), and AST layer import enforcement (`python -m basecoat.layers`).
- **Banned imports (`[tool.basecoat.layers]`):** `multimodal_benchmark`, `xct_research`, `porousdiff_workflows`, `openimpala`, `sms`, `openlsrx`, `braggtrack`, `porousdiff`, `numcodecs`.

## Transition Notes (`v1.0.x` -> `v1.1.0`)
- `export_for_openimpala` is deprecated in favour of `export_volume_ome_zarr` and `export_report_ome_zarr` (which write calibrated primary imagery/chemical maps plus mode-downsampled `phase`, `phase_lower`, and `phase_upper` labels and `uncertainties/*` sidecars in a single `.ome.zarr` store) and will be removed once `OpenImpala v5.1` reads `.ome.zarr` natively.
- Specimen-specific Zn-13X scan ID tables (`DIAD_ZN13X_50UM_SCAN_IDS`, `DIAD_ZN13X_25UM_SCAN_IDS`), `load_diad_xct_zn13x`, and `+47.5 deg` stage defaults were removed from `x_fuse.loaders` in PR `#10` (`24d5502`) and live exclusively in `BASE-Laboratory/MultimodalBenchmark` (Layer 4).
- `OPEN_DATASETS`, `list_open_datasets()`, and `fetch_open_dataset()` in `x_fuse.datasets` have no hardcoded scan IDs and emit `DeprecationWarning` pending migration to Layer 4 (`MultimodalBenchmark`).
