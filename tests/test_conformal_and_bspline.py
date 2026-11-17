"""Statistical coverage and diffeomorphic registration tests for Split Conformal
Prediction (`ConformalCalibrator`) and cubic B-spline guide alignment
(`DifferentiableGuideAligner(mode="bspline")`).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from x_fuse import (
    ConformalCalibrator,
    DifferentiableGuideAligner,
    XFuse,
    empirical_conformal_coverage,
    generate_spectral_cube_phantom_2d,
    save_uncertainty_report,
)


@pytest.mark.parametrize("method", ["lac", "aps"])
@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_split_conformal_finite_sample_marginal_coverage_guarantee(
    method: str, alpha: float
) -> None:
    rng = np.random.default_rng(1234)
    n_classes = 3
    h, w = 40, 40  # 1600 pixels total (800 cal, 800 test per trial)

    coverages = []
    for _trial in range(15):
        logits = rng.normal(0.0, 1.4, size=(1, n_classes, h, w)).astype(np.float32)
        probs = torch.softmax(torch.from_numpy(logits), dim=1)
        # Draw true labels directly from the predictive categorical distribution
        flat_p = probs[0].permute(1, 2, 0).reshape(-1, n_classes).numpy()
        cum = np.cumsum(flat_p, axis=-1)
        u = rng.random((flat_p.shape[0], 1))
        labels = (u > cum).sum(axis=-1).reshape(1, h, w)
        targets = torch.from_numpy(labels.astype(np.int64))

        perm = rng.permutation(h * w)
        cal_mask = np.zeros(h * w, dtype=bool)
        cal_mask[perm[:800]] = True
        cal_mask_2d = torch.from_numpy(cal_mask.reshape(1, h, w))
        test_mask_2d = ~cal_mask_2d

        cal = ConformalCalibrator(
            alpha=alpha,
            method="lac" if method == "lac" else "aps",
        )
        cal.calibrate_labels(probs, targets, cal_mask=cal_mask_2d)
        sets, _ = cal.predict_sets_and_intervals(probs)
        stats = empirical_conformal_coverage(
            sets[0][:, test_mask_2d[0]], targets[0][test_mask_2d[0]]
        )
        coverages.append(stats["empirical_coverage"])

    mean_cov = float(np.mean(coverages))
    assert mean_cov >= (1.0 - alpha) - 0.025


def test_zero_label_conformal_in_xfuse_and_hdf5_zarr_serialization(
    tmp_path: Path,
) -> None:
    ph = generate_spectral_cube_phantom_2d(
        size=48, downsample_factor=4, n_spectral_bins=8, mean_counts=80.0, seed=3
    )
    coarse_2ch = ph.gt_phase_channels.reshape(2, 12, 4, 12, 4).mean(axis=(2, 4))

    model = XFuse(
        backbone="timm:vit_tiny_patch16_224",
        pretrained=False,
        stride=8,
        dtype=torch.float32,
        loss_fn="forward_psf_bce",
        conformal_alpha=0.10,
        lr=1e-2,
    )
    rep = model.fit_predict(
        ph.hr_image,
        coarse_2ch,
        epochs=6,
        mode="multiclass",
        tta_shifts=(),
        tta_flips=False,
    )
    assert rep.prediction_sets is not None
    assert rep.prediction_sets.dtype == torch.bool
    assert rep.prediction_sets.shape == (1, 3, 48, 48)
    assert rep.conformal_interval_halfwidth is not None
    assert rep.conformal_interval_halfwidth.shape == (1, 2, 48, 48)
    assert rep.diagnostics.get("conformal_alpha") == pytest.approx(0.10)

    h5_path = save_uncertainty_report(rep, tmp_path / "conformal_out.h5")
    with h5py.File(h5_path, "r") as f:
        grp = f["xfuse_report"]
        assert "prediction_sets" in grp
        assert "conformal_interval_halfwidth" in grp
        assert grp["prediction_sets"].shape == (3, 48, 48)


def test_bspline_deformable_guide_aligner_diffeomorphic_recovery() -> None:
    ph_warped = generate_spectral_cube_phantom_2d(
        size=64,
        downsample_factor=4,
        n_spectral_bins=6,
        mean_counts=200.0,
        bspline_warp_px=2.5,
        seed=88,
    )
    # Collapse spectral bins into two clean warped coarse bands for registration test
    warped_guide = np.stack(
        [
            ph_warped.spectral_cube_lr[:3].sum(axis=0),
            ph_warped.spectral_cube_lr[3:].sum(axis=0),
        ],
        axis=0,
    )
    warped_guide /= max(float(warped_guide.max()), 1e-6)

    gt_hr = torch.from_numpy(ph_warped.gt_phase_channels).unsqueeze(0).float()
    target_lr = torch.from_numpy(warped_guide).unsqueeze(0).float()
    lr_hw = (16, 16)

    aligner = DifferentiableGuideAligner(
        mode="bspline",
        max_shift_coarse_px=1.5,
        control_grid_size=5,
        reg_weight=5e-4,
    )
    opt = torch.optim.Adam(aligner.parameters(), lr=5e-2)

    def _eval_loss() -> torch.Tensor:
        warped = aligner(gt_hr, lr_hw)
        pooled = torch.nn.functional.adaptive_avg_pool2d(warped, lr_hw)
        # Scale-normalized MSE
        p_norm = pooled / pooled.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        t_norm = target_lr / target_lr.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        return torch.nn.functional.mse_loss(
            p_norm, t_norm
        ) + aligner.regularization_loss(lr_hw, gt_hr.device)

    init_loss = float(_eval_loss().item())
    for _ in range(35):
        opt.zero_grad()
        loss = _eval_loss()
        loss.backward()
        opt.step()
    final_loss = float(_eval_loss().item())

    assert final_loss < init_loss
    min_jac = aligner.min_jacobian_determinant((64, 64), lr_hw, gt_hr.device)
    assert min_jac > 0.1
    diag = aligner.diagnostics((64, 64), lr_hw)
    assert diag["align_mode"] == "bspline"
    assert "bspline_min_jacobian_det" in diag
