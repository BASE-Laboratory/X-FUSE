# API Reference

## High-Level & Compositional 2D Fusion (`x_fuse.fusion`, `x_fuse.heads`)

::: x_fuse.fusion
    options:
      members:
        - CorrelativeFuser
        - XFuse
        - segment_pores_otsu
        - compute_channel_correlation_gates

::: x_fuse.heads
    options:
      members:
        - LinearProbeHead
        - SpatialConvMLPHead
        - WindowedGuideCrossAttentionHead
        - edge_aware_tv_loss

---

## Quantitative Forward Physics & Spectral Unmixing (`x_fuse.physics`)

::: x_fuse.physics
    options:
      members:
        - PhysicsForwardModel
        - SpectralUnmixingDictionary
        - poisson_deviance_loss

---

## 2.5D Volumetric Propagation & Lazy Streaming (`x_fuse.volumetric`)

::: x_fuse.volumetric
    options:
      members:
        - VolumeSliceAccessor
        - VolumetricFuser
        - VolumetricUncertaintyReport
        - pack_25d_slice
        - z_slice_consistency_score

---

## Multi-Layer Backbones, Positional Debiasing & Tiling (`x_fuse.backbones`, `x_fuse.debias`, `x_fuse.tiling`)

::: x_fuse.backbones
    options:
      members:
        - HighResBackbone
        - load_backbone

::: x_fuse.debias
    options:
      members:
        - ApositionalFeatureDebiaser
        - build_orthonormal_spatial_basis
        - measure_positional_r2

::: x_fuse.tiling
    options:
      members:
        - generate_tile_boxes
        - hann_window_2d
        - run_tiled_2d

---

## Differentiable Co-Registration & Feature Upsamplers (`x_fuse.registration`, `x_fuse.upsamplers`)

::: x_fuse.registration
    options:
      members:
        - DifferentiableGuideAligner

::: x_fuse.upsamplers
    options:
      members:
        - CrossAttentionFeatureUpsampler
        - ForwardPSFDownsampler
        - JointBilateralUpsampler
        - extract_multiscale_texture_descriptors
        - joint_bilateral_upsample

---

## Conformal Calibration, Uncertainty & Active Acquisition (`x_fuse.uncertainty`, `x_fuse.active`)

::: x_fuse.uncertainty
    options:
      members:
        - ConformalCalibrator
        - UncertaintyReport
        - WelfordAccumulator
        - match_coarse_priors
        - compute_posterior_uncertainty
        - compute_bald_mutual_information
        - compute_physical_residual
        - build_uncertainty_report
        - empirical_conformal_coverage
        - expected_calibration_error
        - brier_score
        - sparsification_curve

::: x_fuse.active
    options:
      members:
        - ActiveSliceRecommendation
        - ActiveROI2D
        - suggest_next_slices
        - suggest_2d_rois

---

## Scientific I/O, Synchrotron Loaders & Open Datasets (`x_fuse.io`, `x_fuse.loaders`, `x_fuse.datasets`)

::: x_fuse.io
    options:
      members:
        - export_volume_ome_zarr
        - export_for_openimpala
        - save_uncertainty_report
        - load_uncertainty_report
        - load_microscopy_file

::: x_fuse.loaders
    options:
      members:
        - load_xct_subvolume
        - load_diad_xrdct
        - load_xrdct_peak_maps
        - aggregate_peak_maps
        - detrend_radial_cupping
        - remove_polar_rings_conservative

::: x_fuse.datasets
    options:
      members:
        - OpenCorrelativeDataset
        - list_open_datasets
        - fetch_open_dataset
        - get_sample_dataset_path
        - make_experimental_holdout

---

## Synthetic Correlative Phantoms (`x_fuse.synthetic`)

::: x_fuse.synthetic
    options:
      members:
        - generate_correlative_phantom_2d
        - generate_oriented_lamellar_phantom_2d
        - generate_correlative_phantom_3d
        - generate_localized_reaction_phantom_3d
        - generate_spectral_cube_phantom_2d
