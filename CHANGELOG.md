# Changelog

All notable changes to **X-FUSE** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-17

### Added
- **Quantitative Forward Physics & 3D Spectral Cube Unmixing (`x_fuse.physics`)**: Added `PhysicsForwardModel` supporting Poisson shot-noise deviance NLL (`loss_fn="forward_psf_poisson"`) with learnable gain/background for low-dose EDS/EELS/XRF/Bragg-edge counts, simplex mass/atomic-fraction conservation (`loss_fn="compositional_kl"`), raw 3D `(E, H_lr, W_lr)` hyperspectral/diffraction cube unmixing via SPA-seeded multiplicative NMF (`SpectralUnmixingDictionary`, `spectral_unmix=True`), and physical count/simplex preserving `GuideNormalization` modes in `guide_to_tensor`.
- **Texture-Aware Cross-Attention Upsampler & Spatial MLP Head (`x_fuse.upsamplers`, `x_fuse.heads`)**: Added `JAFAR`/`AnyUp`-inspired `CrossAttentionFeatureUpsampler` (`interpolation_mode="cross_attn"`, conditioned on 8 multiscale structural & local texture descriptors `extract_multiscale_texture_descriptors`) and residual `SpatialConvMLPHead` (`fusion_method="spatial_mlp"`) with edge-weighted Total Variation regularization (`edge_aware_tv_loss`) to separate iso-intensity textured phases.
- **Apositional Feature Debiasing (`x_fuse.debias`)**: Added `ApositionalFeatureDebiaser` (`debias_positional=True`) projecting ViT patch channels onto the orthogonal complement of bivariate Legendre spatial polynomials (`build_orthonormal_spatial_basis`, `measure_positional_r2`) to prevent coordinate-ramp shortcut learning on macroscopic chemical gradients, plus direct `"alibi:<timm_id>"` loading and explicit `dino-saw` attribution in `x_fuse.alibi`.
- **Split Conformal Prediction & Cubic B-Spline Registration (`x_fuse.uncertainty`, `x_fuse.registration`)**: Added `ConformalCalibrator` (`"lac"` & `"aps"` score quantiles + `calibrate_coarse` zero-label physical calibration) producing finite-sample `(1 - alpha)` boolean `prediction_sets` and `conformal_interval_halfwidth` on `UncertaintyReport` (`empirical_conformal_coverage`), plus non-rigid cubic B-spline scan-distortion alignment (`align_guide="bspline"`) with thin-plate bending energy and minimum Jacobian determinant diagnostics.
- **Competing Field Baselines & Stress-Test Phantoms (`benchmarks/baselines.py`, `x_fuse.synthetic`)**: Added `otsu_threshold_baseline` (multi-Otsu 1D intensity thresholding), `gram_schmidt_pansharpen_baseline`, `cnmf_pansharpen_baseline` (Coupled NMF), `guided_variational_sr_baseline` (edge-weighted TV solver), and `coordinate_inr_baseline` (`SIREN`), plus `generate_textured_isointensity_phantom_2d`, `generate_macroscopic_gradient_phantom_2d`, and `generate_spectral_cube_phantom_2d`.

### Changed
- **Compositional `CorrelativeFuser` Architecture & Deterministic `timm` DINOv2 (`x_fuse.fusion`, `x_fuse.backbones`)**: Refactored `CorrelativeFuser` (`XFuse`) into a compositional `nn.Module` with `reset_trainable_weights()`, canonical unpadded TTA evaluation, sub-pixel `grid_sample` TTA shifts, content-hashed `crc32` TTA caching, `0.5 * factor` default PSF ratio, and deterministic `timm` DINOv2 loading (`pretrained=False` offline-safe).
- **Memory-Safe $O(HW)$ Cross-Attention (`WindowedGuideCrossAttentionHead`)**: Replaced dense $O((HW)^2)$ pixel-to-pixel attention in `XRDCrossAttention` with $O(HW \cdot H_{\text{LR}} W_{\text{LR}})$ HR-pixel-to-coarse-guide cross-attention capable of scaling cleanly to $512 \times 512+$ tiles.
- **Global 3D Contrast & Wheel-Safe Data Paths (`x_fuse.volumetric`, `x_fuse.datasets`)**: Preserved global 3D intensity scaling across `VolumetricFuser` slices (`value_range=(vmin, vmax)`), added per-slice guide resolution for 3D `"attention"`/`"gating"` heads, fixed multi-channel `multilabel` `hard_mask`, updated `isis_imat_bragg_edge` Zenodo DOI to `10.5281/zenodo.4273969`, and routed wheel-mode downloads to `$XFUSE_DATA_DIR` / `~/.cache/x_fuse`.
- Removed top-level `cv2` and `matplotlib` imports from `x_fuse.utils` and debug `print` side effects from `x_fuse.alibi`.

## [0.4.0] - 2026-11-17

### Added
- **3-Line High-Level API (`XFuse.fit` / `predict` / `fit_predict`)**: Modality-neutral initialization (`backbone`, `fusion_method`, `layers`, `align_guide`, `learn_psf`, `tile_size`) with automatic percentile structural normalization, spatial divisibility padding, and exact output cropping (`UncertaintyReport.crop_spatial`).
- **Frozen TTA Caching, Hann Tiling & Lazy 2.5D Slice Streaming**: LRU caching of frozen multi-view ViT TTA features across training epochs, raised-cosine Hann-windowed overlapping 2D tile inference (`x_fuse.tiling`), and lazy HDF5 / Zarr / TIFF-directory 2.5D input slice streaming (`VolumeSliceAccessor` in `x_fuse.volumetric`).
- **Multi-Scale ViT Pyramid (`layers="multiscale"`)**: Forward-hook token extraction concatenating LayerNorm-stabilized shallow, mid, and deep ViT blocks for crisp sub-patch phase boundary localization.
- **Differentiable Sub-Pixel Registration & Learnable Anisotropic PSF**: Canonical-frame TTA-equivariant `DifferentiableGuideAligner` (`translation`, `rigid`, `affine` with $\tanh$-bounded shifts and $L_2$ identity regularization) paired with per-channel learnable Gaussian interaction-volume widths (`ForwardPSFDownsampler(learnable=True)`).
- **Paper Benchmark Suite, Plotters & MkDocs Documentation**: Added `benchmarks/dataset_adapters.py`, `benchmarks/run_paper_benchmarks.py`, `benchmarks/plot_paper_figures.py`, Tutorial `03_active_acquisition_and_registration.ipynb`, and strict MkDocs Material documentation (`mkdocs.yml`, `docs/`, `.github/workflows/docs.yml`).

## [0.3.0] - 2026-11-16

### Added
- **SuRVoS-Style 2.5D Volumetric Mode (`x_fuse.volumetric`)**: Adjacent `(z - Δz, z, z + Δz)` 3-slice packing (`2.5d_slab`), sparse $z$-slice head training (`fit_sparse_slices`), and orthogonal `XY + XZ + YZ` `triplane` fusion with inter-plane epistemic variance.
- **Active Experiment Steering & Scientific I/O (`x_fuse.active`, `x_fuse.io`)**: `suggest_next_slices` for sparse-slice 3D tomography scheduling, integral-image NMS `suggest_2d_rois` for targeted 2D re-acquisition, OME-TIFF / HDF5 / Zarr uncertainty report serialization, RosettaSciIO vendor readers (`x-fuse[io]`), and `xfuse fuse` CLI batch execution.

## [0.2.0] - 2026-11-15

### Added
- **Self-Contained Pure-PyTorch Architecture**: Added `x_fuse.transforms` (invertible shift/flip/rotation TTA), `x_fuse.vit_patch` (sub-patch ViT stride & positional-encoding patching with native PyTorch SDPA fallback + `TorchPCA`), and `x_fuse.upsamplers` (`ForwardPSFDownsampler` + scale-dilated `JointBilateralUpsampler`).
- **Modality-Agnostic Multi-Channel Fusion (`CorrelativeFuser`)**: Extract structural ViT features once and fuse `M` physical guide channels simultaneously via vectorized correlation gating, trainable `SpatialCrossAttention`, or forward-PSF-supervised `LearnedChannelGating`.
- **Tri-Part Pixel/Voxel Certainty (`x_fuse.uncertainty`)**: Prior-matching probability adjustment (`match_coarse_priors`), normalized posterior entropy, online $O(1)$-memory `WelfordAccumulator` epistemic variance, forward-PSF physical discrepancy maps, ECE, Brier score, and sparsification/AUSE curves.
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
