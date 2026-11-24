# Changelog

All notable changes to **X-FUSE** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Native 2D & 3D OME-Zarr v0.4 Report Export & Physical Amplitude Calibration (`x_fuse.io`, `x_fuse.fusion`, `x_fuse.volumetric`, `x_fuse.cli`)**: Added `export_report_ome_zarr` and automatic `.ome.zarr` routing in `save_uncertainty_report` writing calibrated OME-NGFF v0.4 `VolumeSet` stores (`basecoat v0.3.0+`) with named chemical channel metadata (`--channel-names`), per-guide physical intensity scale conservation (`guide_physical_scales` & `reconstruct_physical_maps`), conformal phase envelopes (`phase_lower`, `phase_upper`), `uncertainties/*` sidecars, and optional coarse benchmark grid evaluation sidecars (`--eval-grid`, `--eval-slice-z`).
- **`basecoat` Layer-2 Volumetric & Filter Integration (`x_fuse.io`, `x_fuse.loaders`, `x_fuse.volumetric`)**: Added `export_volume_ome_zarr`, out-of-core Savu `(X, Z, Y)` slicing via `basecoat.open_savu_h5`, conservative ring/cupping filters via `basecoat`, `.zenodo.json` archive metadata, `AGENTS.md`, and `[tool.basecoat.layers]` AST layer-import enforcement.

### Changed
- **Deprecated Legacy Layer-4 Coupling Helpers (`x_fuse.io`, `x_fuse.datasets`)**: Deprecated `export_for_openimpala` in favour of `export_volume_ome_zarr` / `export_report_ome_zarr`, stripped hardcoded DIAD Zn-13X scan IDs from `OPEN_DATASETS["diad_zn13x_xrdct"]`, and marked `list_open_datasets()` / `fetch_open_dataset()` deprecated pending migration to Layer 4 (`MultimodalBenchmark`).

### Removed
- **Specimen-Specific DIAD Loader Constants (`x_fuse.loaders`)**: Removed `DIAD_ZN13X_50UM_SCAN_IDS`, `DIAD_ZN13X_25UM_SCAN_IDS`, `load_diad_xct_zn13x`, and hardcoded `+47.5°` stage motor-zero defaults from Layer 2 (`scan_ids` and `correct_motor_offset_deg` are now passed explicitly by Layer 4 workflows).

## [1.0.0] - 2026-11-20

### Added
- **Fringe-Guarded Forward-PSF Porous-Media Physics & `OpenImpala` Export (`x_fuse.fusion`, `x_fuse.volumetric`, `x_fuse.io`)**: Added `pore_mask` (`"auto_otsu"`, `"auto_otsu_fringe_guarded"` with Fourier Tie-homogeneity phase-fringe damping `segment_pores_otsu`, or explicit void masks) clamping void logits prior to `ForwardPSFDownsampler` to deconvolve partial-volume pore dilution with zero void leakage, `.npz` uncertainty report round-tripping, `.h5`/`.nxs`/`.npz` microscopy file loading (`load_microscopy_file`), and `export_for_openimpala` generating uncompressed `uint8` 3D multi-phase TIFFs, conformal lower/upper transport envelopes (for both `multiclass` and `multilabel` models), and AMReX `ParmParse` `.inputs` decks matching `OpenLSR-X`.
- **Out-of-Core Synchrotron Savu/DAWN Loaders & 1D Voronoi Active Steering (`x_fuse.loaders`, `x_fuse.datasets`, `x_fuse.active`)**: Added out-of-core post-flip HDF5 subvolume slicing (`load_xct_subvolume`, `load_diad_xct_zn13x`), multi-reflection DAWN NeXus peak aggregation (`all_sum`, `all_mean`, `positive_sum`, `median_norm_sum`, `low_q`, `high_q`, `nearest_q`, `single_peak`) + `+47.5°` `gts_cs_theta` motor-zero correction + cleaned-bundle ingestion (`load_xrdct_peak_maps`, `aggregate_peak_maps`, `load_diad_xrdct`), conservative synchrotron ring/cupping cleanup (`detrend_radial_cupping`, `remove_polar_rings_conservative`), Zenodo `diad_zn13x_xrdct` (`13329639`) / `diad_zn13x_powder_xrd` (`13329670`) entries with offline cache fallback, and 1D Voronoi-integrated active slice steering (`suggest_next_slices`).
- **Spin-2 Orientation-Equivariant Sub-Token Upsampling (`x_fuse.upsamplers`, `x_fuse.heads`)**: Added continuous sub-token cross-attention with center-symmetric $D_4$-exact coordinate tie-breaking across non-integer scale ratios (`CrossAttentionFeatureUpsampler`), spin-2 structure-tensor coherence descriptors (`extract_multiscale_texture_descriptors(include_orientation=True)`), direct $1 \times 1$ `pixel_proj` / `coh_proj` structural skip connections, and count-safe `guide_residual=True` anchoring in `SpatialConvMLPHead`.
- **Heavy-Tailed Forward PSF Physics & Small-FOV ViT Auto-Scaling (`x_fuse.upsamplers`, `x_fuse.datasets`, `x_fuse.backbones`)**: Added differentiable Lorentzian/Moffat (`psf_profile="moffat"`) and two-Gaussian SEM backscattered `psf_profile="core_halo"` forward operators in `ForwardPSFDownsampler` and `make_experimental_holdout`, plus automatic small-FOV bicubic ViT pre-upscaling (`min_input_size="auto"`) in `HighResBackbone.extract_patch_grid` to prevent token starvation on nanoscale STEM-EELS ROIs.
- **3D Axial Interaction-Volume Deconvolution & Localized Reaction Phantom (`x_fuse.volumetric`, `x_fuse.synthetic`)**: Added learnable 1D axial depth-PSF slab deconvolution (`depth_psf_sigma`, `learn_depth_psf=True`) and unguided-slice tri-plane consistency regularization (`triplane_consistency_weight`) in `VolumetricFuser.fit_sparse_slices`, alongside `generate_localized_reaction_phantom_3d` for active beamtime steering benchmarks.
- **Spatial-Block Temperature-Scaled RAPS Conformal Calibration (`x_fuse.uncertainty`)**: Upgraded `ConformalCalibrator` with Romano et al. (2020) strict deterministic APS/RAPS bijection (`method="raps"`, `k_reg`, `lambda_reg`), strided $B \times B$ block-center decorrelation (`spatial_block_size`), held-out NLL temperature scaling (`fit_temperature=True`), and void-aware solid-phase conformal calibration (`pore_mask_hr`).
- **Expanded `xfuse` CLI, Tri-Modality Zenodo Adapters & LaTeX Table Exporter (`x_fuse.cli`, `benchmarks/`)**: Added `xfuse fuse-3d` (`--pore-mask`, `--conformal-alpha`, `--conformal-method`, `--export-openimpala-dir`, `--openimpala-phase`, `--openimpala-direction`), `xfuse datasets --fetch`, and `xfuse fuse` flags (`--psf-profile`, `--guide-residual`, `--tv-weight`, `--conformal-method`, `--conformal-block-size`, `--pore-mask`); updated `edxia_cement_sem_eds` to Zenodo record `3862958` (`10.5281/zenodo.3862958`, Bruker Esprit wt% CSVs); added `benchmarks/run_zn13x_multiscale_couplings.py`, `benchmarks/export_latex_tables.py`, and `docs/benchmarks.md`.
- **Quantitative Forward Physics & 3D Spectral Cube Unmixing (`x_fuse.physics`)**: Added `PhysicsForwardModel` supporting Poisson shot-noise deviance NLL (`loss_fn="forward_psf_poisson"`) with learnable gain/background for low-dose EDS/EELS/XRF/Bragg-edge counts, simplex mass/atomic-fraction conservation (`loss_fn="compositional_kl"`), raw 3D `(E, H_lr, W_lr)` hyperspectral/diffraction cube unmixing via SPA-seeded multiplicative NMF (`SpectralUnmixingDictionary`, `spectral_unmix=True`), and physical count/simplex preserving `GuideNormalization` modes in `guide_to_tensor`.
- **Texture-Aware Cross-Attention Upsampler & Spatial MLP Head (`x_fuse.upsamplers`, `x_fuse.heads`)**: Added `JAFAR`/`AnyUp`-inspired `CrossAttentionFeatureUpsampler` (`interpolation_mode="cross_attn"`, conditioned on 8 multiscale structural & local texture descriptors `extract_multiscale_texture_descriptors`) and residual `SpatialConvMLPHead` (`fusion_method="spatial_mlp"`) with edge-weighted Total Variation regularization (`edge_aware_tv_loss`) to separate iso-intensity textured phases.
- **Apositional Feature Debiasing (`x_fuse.debias`)**: Added `ApositionalFeatureDebiaser` (`debias_positional=True`) projecting ViT patch channels onto the orthogonal complement of bivariate Legendre spatial polynomials (`build_orthonormal_spatial_basis`, `measure_positional_r2`) to prevent coordinate-ramp shortcut learning on macroscopic chemical gradients, plus direct `"alibi:<timm_id>"` loading and explicit `dino-saw` attribution in `x_fuse.alibi`.
- **Split Conformal Prediction & Cubic B-Spline Registration (`x_fuse.uncertainty`, `x_fuse.registration`)**: Added `ConformalCalibrator` (`"lac"` & `"aps"` score quantiles + `calibrate_coarse` zero-label physical calibration) producing finite-sample `(1 - alpha)` boolean `prediction_sets` and `conformal_interval_halfwidth` on `UncertaintyReport` (`empirical_conformal_coverage`), plus non-rigid cubic B-spline scan-distortion alignment (`align_guide="bspline"`) with thin-plate bending energy and minimum Jacobian determinant diagnostics.
- **Competing Field Baselines & Stress-Test Phantoms (`benchmarks/baselines.py`, `x_fuse.synthetic`)**: Added `otsu_threshold_baseline` (multi-Otsu 1D intensity thresholding), `gram_schmidt_pansharpen_baseline`, `cnmf_pansharpen_baseline` (Coupled NMF), `guided_variational_sr_baseline` (edge-weighted TV solver), and `coordinate_inr_baseline` (`SIREN`), plus `generate_textured_isointensity_phantom_2d`, `generate_macroscopic_gradient_phantom_2d`, and `generate_spectral_cube_phantom_2d`.

### Changed
- **Compositional `CorrelativeFuser` Architecture & Deterministic `timm` DINOv2 (`x_fuse.fusion`, `x_fuse.backbones`)**: Refactored `CorrelativeFuser` (`XFuse`) into a compositional `nn.Module` with `reset_trainable_weights()`, canonical unpadded TTA evaluation, sub-pixel `grid_sample` TTA shifts, content-hashed `crc32` TTA caching, `0.5 * factor` default PSF ratio, and deterministic `timm` DINOv2 loading (`pretrained=False` offline-safe).
- **Memory-Safe $O(HW)$ Cross-Attention (`WindowedGuideCrossAttentionHead`)**: Replaced dense $O((HW)^2)$ pixel-to-pixel attention in `XRDCrossAttention` with $O(HW \cdot H_{\text{LR}} W_{\text{LR}})$ HR-pixel-to-coarse-guide cross-attention capable of scaling cleanly to $512 \times 512+$ tiles.
- **Global 3D Contrast & Wheel-Safe Data Paths (`x_fuse.volumetric`, `x_fuse.datasets`)**: Preserved global 3D intensity scaling across `VolumetricFuser` slices (`value_range=(vmin, vmax)`), added per-slice guide resolution for 3D `"attention"`/`"gating"`/`"spatial_mlp"` heads, fixed multi-channel `multilabel` `hard_mask`, updated `isis_imat_bragg_edge` Zenodo DOI to `10.5281/zenodo.4273969`, and routed wheel-mode downloads to `$XFUSE_DATA_DIR` / `~/.cache/x_fuse`.
- Removed unused `opencv-python-headless` core dependency (~59 MB lighter wheel footprint), top-level `matplotlib` imports from `x_fuse.utils`, and debug `print` side effects from `x_fuse.alibi`.

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
