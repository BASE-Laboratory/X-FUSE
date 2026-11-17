# Changelog

All notable changes to **X-FUSE** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-11-17

### Added
- **3-Line High-Level API (`XFuse.fit` / `predict` / `fit_predict`)**: Modality-neutral initialization (`backbone`, `fusion_method`, `layers`, `align_guide`, `learn_psf`, `tile_size`) with automatic percentile structural normalization, spatial divisibility padding, and exact output cropping (`UncertaintyReport.crop_spatial`).
- **Frozen TTA Caching, Hann Tiling & Out-of-Core 3D Streaming**: LRU caching of frozen multi-view ViT TTA features across training epochs (`~10x–20x` probe speedup), raised-cosine Hann-windowed overlapping 2D tile inference (`x_fuse.tiling`), and lazy HDF5 / Zarr / TIFF-directory 2.5D slice streaming (`VolumeSliceAccessor` in `x_fuse.volumetric`).
- **Multi-Scale ViT Pyramid (`layers="multiscale"`)**: Forward-hook token extraction concatenating LayerNorm-stabilized shallow, mid, and deep ViT blocks for crisp sub-patch phase boundary localization.
- **Differentiable Sub-Pixel Registration & Learnable Anisotropic PSF**: Canonical-frame TTA-equivariant `DifferentiableGuideAligner` (`translation`, `rigid`, `affine` with $\tanh$-bounded shifts and $L_2$ identity regularization) paired with per-channel learnable Gaussian interaction-volume widths (`ForwardPSFDownsampler(learnable=True)`).
- **Active Experiment Steering (`x_fuse.active`)**: `suggest_next_slices` for sparse-slice 3D tomography scheduling and integral-image NMS `suggest_2d_rois` for targeted 2D re-acquisition.
- **Standardized Scientific I/O & `xfuse fuse` CLI (`x_fuse.io`, `x_fuse.cli`)**: Export and reload 2D/3D uncertainty reports in `.ome.tif`, `.h5`, and `.zarr`, load vendor `.emd`/`.dm3`/`.dm4`/`.bcf`/`.hspy` archives via optional RosettaSciIO (`x-fuse[io]`), and run batch fusion from the terminal via `xfuse fuse`.
- **Paper Benchmark Suite, Plotters & MkDocs Documentation**: Added `benchmarks/dataset_adapters.py`, `benchmarks/run_paper_benchmarks.py`, `benchmarks/plot_paper_figures.py`, Tutorial `03_active_acquisition_and_registration.ipynb`, and strict MkDocs Material documentation (`mkdocs.yml`, `docs/`, `.github/workflows/docs.yml`).

## [0.2.0] - 2026-11-17

### Added
- **Self-Contained Pure-PyTorch Architecture**: Added `x_fuse.transforms` (invertible shift/flip/rotation TTA), `x_fuse.vit_patch` (sub-patch ViT stride & positional-encoding patching with native PyTorch SDPA fallback + `TorchPCA`), and `x_fuse.upsamplers` (`ForwardPSFDownsampler` + scale-dilated `JointBilateralUpsampler`).
- **Modality-Agnostic Multi-Channel Fusion (`CorrelativeFuser`)**: Extract structural ViT features once and fuse `M` physical guide channels simultaneously via vectorized correlation gating, trainable `SpatialCrossAttention`, or forward-PSF-supervised `LearnedChannelGating`.
- **Tri-Part Pixel/Voxel Certainty (`x_fuse.uncertainty`)**: Prior-matching probability calibration (`match_coarse_priors`), normalized posterior entropy, online $O(1)$-memory `WelfordAccumulator` epistemic variance, forward-PSF physical discrepancy maps, ECE, Brier score, and sparsification/AUSE curves.
- **SuRVoS-Style 2.5D Volumetric Mode (`x_fuse.volumetric`)**: Adjacent `(z - Δz, z, z + Δz)` RGB slice packing (`2.5d_slab`), sparse z-slice probe training (`fit_sparse_slices`), and orthogonal `XY + XZ + YZ` `triplane` fusion with inter-plane epistemic variance.
- **Segmentation Baselines & Optional SAM2 Seeding (`x_fuse.segment`)**: Classical bicubic, edge-guided JBU, unsupervised DINO + K-Means, sign-aligned PCA + Otsu, and high-certainty-seeded `refine_with_sam2`.
- **Open Dataset Registry, Synthetic Phantoms & CLI**: Added `x_fuse.datasets` (Zenodo correlative registry + MD5-verified fetcher + `make_experimental_holdout`), `x_fuse.synthetic` (2D/3D multiphase phantoms), `benchmarks/run_synthetic_suite.py`, and the `xfuse` CLI (`xfuse info`, `xfuse datasets`, `xfuse demo`).
- **CI/CD & Packaging**: Multi-Python/OS GitHub Actions matrix, Ruff + Mypy enforcement, wheel smoke verification, OIDC PyPI release workflow, `CITATION.cff`, `CONTRIBUTING.md`, and stripped tutorial notebooks.

### Fixed
- Eliminated duplicate ViT backbone loading during `XFuse` / `CorrelativeFuser` initialization.
- Fixed 8-bit `uint8` quantization loss in `xrd_to_tensor` (`guide_to_tensor` now preserves `float32` dynamic range).
- Fixed inverse transform composition order `(i1 ∘ i2)` when combining shift + flip/rotation TTA transforms.
- Fixed non-deterministic file ordering and unclosed HDF5 handles in `x_fuse.loaders` and `x_fuse.utils.load_xrd_h5`.

### Removed
- Removed vendored `FeatUp/` and `HR-Dv2/` sub-repositories and custom CUDA C++ compilation requirements.

## [0.1.0] - 2026-05-06

### Added
- Initial research prototype for 2D synchrotron XCT + XRD-CT feature fusion (`XFuse`, `alibi`, `loaders`).
