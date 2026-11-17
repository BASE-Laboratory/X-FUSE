"""Unit tests for differentiable co-registration (`DifferentiableGuideAligner`),
learnable per-channel anisotropic PSF (`ForwardPSFDownsampler`), and out-of-core
lazy 3D HDF5 / Zarr / TIFF-directory streaming (`VolumeSliceAccessor`).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from PIL import Image
from scipy.ndimage import shift as nd_shift

from x_fuse import (
    DifferentiableGuideAligner,
    ForwardPSFDownsampler,
    VolumeSliceAccessor,
    VolumetricFuser,
    XFuse,
    generate_correlative_phantom_2d,
    generate_correlative_phantom_3d,
    make_experimental_holdout,
)


@pytest.mark.parametrize("mode", ["none", "translation", "rigid", "affine"])
def test_differentiable_guide_aligner_modes_and_diagnostics(mode: str):
    aligner = DifferentiableGuideAligner(
        mode=mode,  # type: ignore[arg-type]
        max_shift_coarse_px=1.5,
        max_rotation_deg=5.0,
    )
    x = torch.rand(1, 2, 32, 32, requires_grad=True)
    warped = aligner(x, lr_hw=(8, 8))
    assert warped.shape == x.shape

    reg = aligner.regularization_loss(lr_hw=(8, 8), device=x.device)
    assert reg.ndim == 0 and float(reg.item()) >= 0.0

    diag = aligner.diagnostics(hr_hw=(32, 32), lr_hw=(8, 8))
    assert diag["align_mode"] == mode
    assert "shift_x_hr_px" in diag and "rotation_deg" in diag

    if mode != "none":
        loss = warped.mean() + reg
        loss.backward()
        assert aligner.raw_shift is not None and aligner.raw_shift.grad is not None


def test_differentiable_aligner_invalid_mode_and_shift_recovery():
    with pytest.raises(ValueError, match="Invalid align mode"):
        DifferentiableGuideAligner(mode="projective")  # type: ignore[arg-type]

    # Direct synthetic test: aligner + ForwardPSF recovers a +3.0 HR-px X shift
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 64),
        torch.linspace(-1.0, 1.0, 64),
        indexing="ij",
    )
    hr_blob = ((xx**2 + yy**2) < 0.35**2).float().unsqueeze(0).unsqueeze(0)
    shifted_np = nd_shift(
        hr_blob[0, 0].numpy(), shift=(0.0, 3.0), order=1, mode="nearest"
    )
    down = ForwardPSFDownsampler(psf_sigma_ratio=0.35, learnable=False)
    target_lr = down(
        torch.from_numpy(shifted_np).unsqueeze(0).unsqueeze(0), target_size=(16, 16)
    )

    aligner = DifferentiableGuideAligner(
        mode="translation", max_shift_coarse_px=1.5, reg_weight=1e-5
    )
    opt = torch.optim.Adam(aligner.parameters(), lr=0.1)
    init_loss = float(
        (down(aligner(hr_blob, (16, 16)), (16, 16)) - target_lr)
        .detach()
        .abs()
        .mean()
        .item()
    )

    for _ in range(45):
        opt.zero_grad()
        pred_lr = down(aligner(hr_blob, (16, 16)), (16, 16))
        loss = (pred_lr - target_lr).square().mean() + aligner.regularization_loss(
            (16, 16), hr_blob.device
        )
        loss.backward()
        opt.step()

    final_loss = float(
        (down(aligner(hr_blob, (16, 16)), (16, 16)) - target_lr)
        .detach()
        .abs()
        .mean()
        .item()
    )
    diag = aligner.diagnostics(hr_hw=(64, 64), lr_hw=(16, 16))
    assert final_loss < 0.35 * init_loss
    assert 2.0 <= float(diag["shift_x_hr_px"]) <= 4.0
    assert abs(float(diag["shift_y_hr_px"])) < 0.5


def test_learnable_anisotropic_psf_downsampler():
    psf = ForwardPSFDownsampler(psf_sigma_ratio=0.35, learnable=True, n_channels=2)
    hr_probs = torch.rand(1, 2, 48, 48, requires_grad=True)
    out = psf(hr_probs, target_size=(12, 12))
    assert out.shape == (1, 2, 12, 12)

    out.square().mean().backward()
    assert psf.log_sigma_ratio is not None
    assert psf.log_sigma_ratio.grad is not None
    assert torch.any(psf.log_sigma_ratio.grad.abs() > 0)

    # Explicit anisotropic tuple psf_sigma
    out_tuple = psf(hr_probs.detach(), target_size=(12, 12), psf_sigma=(1.4, 0.5))
    assert out_tuple.shape == (1, 2, 12, 12)

    diag = psf.diagnostics(hr_hw=(48, 48), lr_hw=(12, 12))
    assert diag["learnable_psf"] is True
    sigmas = diag["psf_sigma_yx_hr_px"]
    assert isinstance(sigmas, list) and len(sigmas) == 2


def test_end_to_end_xfuse_with_drift_and_learnable_psf():
    ph = generate_correlative_phantom_2d(size=64, downsample_factor=4, seed=19)
    gt_ch = np.stack(
        [
            (ph.gt_labels == 1).astype(np.float32),
            (ph.gt_labels == 2).astype(np.float32),
        ],
        axis=0,
    )
    shifted_gt = np.stack(
        [nd_shift(c, (0.0, 2.5), order=1, mode="nearest") for c in gt_ch]
    )
    _, drifted_guide, _ = make_experimental_holdout(
        ph.hr_image, shifted_gt, factor=4, psf_sigma=1.5
    )

    model = XFuse(
        backbone="timm:vit_tiny_patch16_224",
        stride=8,
        pretrained=False,
        align_guide="translation",
        learn_psf=True,
        lr=1e-2,
    )
    rep = model.fit_predict(ph.hr_image, drifted_guide, epochs=10, tta_shifts=(1,))
    assert rep.probabilities.shape == (1, 3, 64, 64)
    assert rep.diagnostics["align_mode"] == "translation"
    assert rep.diagnostics["learnable_psf"] is True
    assert "psf_sigma_yx_hr_px" in rep.diagnostics


def test_lazy_volume_slice_accessor_hdf5_zarr_and_tiff_dir(tmp_path: Path):
    zarr = pytest.importorskip("zarr")
    ph3d = generate_correlative_phantom_3d(
        shape=(8, 32, 32), downsample_factor=4, seed=5
    )
    vol_np = ph3d.hr_volume

    # 1. HDF5 lazy dataset
    h5_path = tmp_path / "tomo.h5"
    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset("recon", data=vol_np)

    # 2. Zarr array
    zarr_path = tmp_path / "tomo.zarr"
    zroot = zarr.open_group(str(zarr_path), mode="w")
    if hasattr(zroot, "create_array"):
        zarr_ds = zroot.create_array("recon", data=vol_np)
    else:
        zarr_ds = zroot.create_dataset("recon", data=vol_np)

    # 3. Sorted 2D TIFF slice directory
    slice_dir = tmp_path / "slices"
    slice_dir.mkdir()
    for z in range(vol_np.shape[0]):
        Image.fromarray(vol_np[z]).save(slice_dir / f"slice_{z:03d}.tiff")

    acc_mem = VolumeSliceAccessor(vol_np)
    assert not acc_mem.is_lazy

    with h5py.File(h5_path, "r") as hf:
        acc_h5 = VolumeSliceAccessor(hf["recon"])
        acc_zr = VolumeSliceAccessor(zarr_ds)
        acc_dir = VolumeSliceAccessor(slice_dir)

        for acc in (acc_h5, acc_zr, acc_dir):
            assert acc.is_lazy
            assert acc.shape == (8, 32, 32)
            assert np.allclose(acc.get_plane(0, 3), vol_np[3], atol=1e-5)
            assert np.allclose(acc.get_plane(1, 10), vol_np[:, 10, :], atol=1e-5)
            assert np.allclose(acc.get_plane(2, 15), vol_np[:, :, 15], atol=1e-5)

        # Fit & predict on lazy HDF5 dataset without loading full array
        base = XFuse(
            backbone="timm:vit_tiny_patch16_224",
            stride=8,
            pretrained=False,
            dtype=torch.float32,
        )
        vf = VolumetricFuser(base, tta_shifts=(1,), tta_flips=False)
        sparse = {1: ph3d.coarse_guide_3d[:, 0], 6: ph3d.coarse_guide_3d[:, 1]}
        losses = vf.fit_sparse_slices(hf["recon"], sparse_guides=sparse, epochs=3)
        assert len(losses) == 3

        rep_h5 = vf.predict_volume(hf["recon"], mode="triplane")
        rep_np = vf.predict_volume(vol_np, mode="triplane")
        assert rep_h5.hard_mask.shape == (8, 32, 32)
        assert torch.allclose(rep_h5.probabilities, rep_np.probabilities, atol=1e-4)

    with pytest.raises(IndexError):
        acc_mem.get_plane(0, 99)
    with pytest.raises(ValueError, match="axis must be"):
        acc_mem.get_plane(4, 0)
