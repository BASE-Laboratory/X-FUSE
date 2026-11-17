# API Reference

## High-Level & Core 2D Fusion (`x_fuse.fusion`)

::: x_fuse.fusion
    options:
      members:
        - CorrelativeFuser
        - XFuse
        - LearnedChannelGating
        - SpatialCrossAttention
        - compute_channel_correlation_gates

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

## Multi-Layer Backbones & Tiling (`x_fuse.backbones`, `x_fuse.tiling`)

::: x_fuse.backbones
    options:
      members:
        - HighResBackbone
        - load_backbone

::: x_fuse.tiling
    options:
      members:
        - generate_tile_boxes
        - hann_window_2d
        - run_tiled_2d

---

## Differentiable Co-Registration & Forward PSF (`x_fuse.registration`, `x_fuse.upsamplers`)

::: x_fuse.registration
    options:
      members:
        - DifferentiableGuideAligner

::: x_fuse.upsamplers
    options:
      members:
        - ForwardPSFDownsampler
        - JointBilateralUpsampler
        - joint_bilateral_upsample

---

## Uncertainty Calibration & Active Acquisition (`x_fuse.uncertainty`, `x_fuse.active`)

::: x_fuse.uncertainty
    options:
      members:
        - UncertaintyReport
        - WelfordAccumulator
        - match_coarse_priors
        - compute_posterior_uncertainty
        - compute_bald_mutual_information
        - compute_physical_residual
        - build_uncertainty_report
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

## Scientific I/O & Open Datasets (`x_fuse.io`, `x_fuse.datasets`)

::: x_fuse.io
    options:
      members:
        - save_uncertainty_report
        - load_uncertainty_report
        - load_microscopy_file

::: x_fuse.datasets
    options:
      members:
        - OpenCorrelativeDataset
        - list_open_datasets
        - fetch_open_dataset
        - get_sample_dataset_path
        - make_experimental_holdout
