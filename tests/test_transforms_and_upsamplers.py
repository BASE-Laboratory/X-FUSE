"""Unit tests for invertible TTA transforms, float-native normalization,
pure-PyTorch Joint Bilateral Upsampling (JBU), Forward-PSF downsampling, and
bug-fixed float/loader utilities.
"""

import numpy as np
import pytest
import torch

from x_fuse.loaders import load_eds_text_hypermap, load_xrdct_phase
from x_fuse.transforms import (
    build_default_tta,
    combine_transforms,
    ensure_spatial_divisible,
    get_rotation_transforms,
    get_shift_transforms,
    normalize_structural,
)
from x_fuse.upsamplers import (
    ForwardPSFDownsampler,
    JointBilateralUpsampler,
    joint_bilateral_upsample,
)
from x_fuse.utils import guide_to_tensor, xrd_to_tensor


def test_combined_shift_and_flip_exact_invertibility():
    """Verify (f2 o f1)^-1 = i1 o i2 holds even for non-commuting shift + flip/rot."""
    torch.manual_seed(7)
    img = torch.randn(1, 3, 24, 32)

    fwd_s, inv_s = get_shift_transforms([1, 3], pattern="Moore")
    fwd_r, inv_r = get_rotation_transforms()
    # Combine shifts with 90-deg rotations (non-commuting on square tensor)
    sq = torch.randn(1, 3, 24, 24)
    fwd, inv = combine_transforms(fwd_s, fwd_r, inv_s, inv_r)
    for f_fn, i_fn in zip(fwd, inv):
        recovered = i_fn(f_fn(sq))
        assert torch.allclose(recovered, sq, atol=1e-6)

    # Also test default shift + flip TTA on rectangular image
    fwd_tta, inv_tta = build_default_tta(
        shift_dists=[1, 2], pattern="Neumann", include_flips=True
    )
    for f_fn, i_fn in zip(fwd_tta, inv_tta):
        recovered = i_fn(f_fn(img))
        assert torch.allclose(recovered, img, atol=1e-6)


def test_normalize_structural_preserves_float_precision():
    """Ensure subtle 32-bit float steps smaller than 1/255 are not quantized away."""
    ramp = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    out = normalize_structural(ramp, imagenet_norm=False)
    assert out.shape == (3, 64, 64)
    assert out.dtype == torch.float32
    # Every pixel along the ramp must remain strictly increasing (4096 > 256 levels)
    diffs = torch.diff(out[0].flatten())
    assert torch.all(diffs > 0)
    assert len(torch.unique(out[0])) == 64 * 64


def test_ensure_spatial_divisible():
    x = torch.randn(3, 50, 53)
    padded, orig_hw = ensure_spatial_divisible(x, patch_size=16, stride=4)
    assert orig_hw == (50, 53)
    assert (padded.shape[-2] - 16) % 4 == 0
    assert (padded.shape[-1] - 16) % 4 == 0


def test_xrd_and_guide_to_tensor_float_and_multichannel():
    """Regression test: float [0, 1] maps must NOT be truncated to zeros."""
    arr_f32 = np.linspace(0.0, 0.85, 16 * 16, dtype=np.float32).reshape(16, 16)
    t = xrd_to_tensor(arr_f32)
    assert t.shape == (1, 1, 16, 16)
    assert t.dtype == torch.float32
    assert float(t.min()) == pytest.approx(0.0, abs=1e-6)
    assert float(t.max()) == pytest.approx(1.0, abs=1e-6)
    assert len(torch.unique(t)) == 256

    multi = np.stack([arr_f32, 1.0 - arr_f32, np.zeros_like(arr_f32)], axis=0)
    tm = guide_to_tensor(multi)
    assert tm.shape == (1, 3, 16, 16)
    # Zero-variance channel safely maps to all zeros without NaNs
    assert not torch.isnan(tm).any()
    assert torch.all(tm[0, 2] == 0.0)


def test_forward_psf_downsampler_conserves_mean_fraction():
    """Physical forward operator should preserve global phase volume fraction."""
    ds = ForwardPSFDownsampler(psf_sigma_ratio=0.35)
    hr = torch.zeros(1, 2, 64, 64)
    hr[:, 0, 16:48, 16:48] = 1.0  # 25% area
    hr[:, 1, :32, :] = 1.0  # 50% area
    lr = ds(hr, (8, 8))
    assert lr.shape == (1, 2, 8, 8)
    assert float(lr[:, 0].mean()) == pytest.approx(0.25, abs=0.02)
    assert float(lr[:, 1].mean()) == pytest.approx(0.50, abs=0.02)


def test_jbu_preserves_sharp_step_edge_better_than_bilinear():
    """Guided JBU should snap a blurry coarse boundary to a sharp HR step edge."""
    hr = torch.zeros(1, 1, 32, 32)
    hr[:, :, :, 16:] = 1.0

    coarse = torch.zeros(1, 1, 4, 4)
    coarse[:, :, :, 2:] = 1.0

    jbu_out = joint_bilateral_upsample(
        coarse, hr, radius=3, sigma_spatial=2.5, sigma_range=0.08, n_iters=3
    )
    bilin_out = torch.nn.functional.interpolate(
        coarse, size=(32, 32), mode="bilinear", align_corners=False
    )

    jbu_err = float((jbu_out - hr).abs().mean())
    bilin_err = float((bilin_out - hr).abs().mean())
    assert jbu_err < bilin_err

    # Also verify learned=True mode supports backward gradients
    learned_jbu = JointBilateralUpsampler(guidance_dim=1, radius=2, learned=True)
    up_l = learned_jbu(coarse, hr)
    loss = (up_l - hr).square().mean()
    loss.backward()
    assert learned_jbu.log_temp.grad is not None


def test_load_xrdct_phase_sorted_order_and_eds_text(tmp_path):
    """Verify multi-slice NeXus loader sorts numerically and EDS text loader works."""
    import h5py

    phase_dir = tmp_path / "phase"
    phase_dir.mkdir()
    # Create zn_slice_2.nxs, zn_slice_10.nxs, zn_slice_1.nxs out of order
    for idx in (10, 1, 2):
        with h5py.File(phase_dir / f"zn_slice_{idx}.nxs", "w") as hf:
            hf.create_dataset(
                "data", data=np.full((8, 8), float(idx), dtype=np.float32)
            )
    vols = load_xrdct_phase(str(phase_dir), phases=["Zn"], shape=(3, 8, 8))
    assert list(vols[0][:, 0, 0]) == [1.0, 2.0, 10.0]

    eds_dir = tmp_path / "eds"
    eds_dir.mkdir()
    np.savetxt(eds_dir / "sample_BSE.txt", np.ones((6, 8)) * 0.5, delimiter=",")
    np.savetxt(eds_dir / "sample_Ca.txt", np.ones((6, 8)), delimiter=",")
    np.savetxt(eds_dir / "sample_Si.txt", np.full((6, 8), 2.0), delimiter=",")
    bse, stack, names = load_eds_text_hypermap(
        eds_dir, prefix="sample", elements=["Ca", "Si"], delimiter=","
    )
    assert bse.shape == (6, 8)
    assert stack.shape == (2, 6, 8)
    assert names == ["Ca", "Si"]
