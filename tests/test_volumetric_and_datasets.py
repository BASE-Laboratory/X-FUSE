"""Unit tests for SuRVoS-inspired 2.5D volumetric fusion (sparse-slice 2D -> 3D
propagation and orthogonal tri-plane inference) plus sample/open dataset tools.
"""

import numpy as np
import torch

from x_fuse.datasets import (
    get_sample_dataset_path,
    list_open_datasets,
    make_experimental_holdout,
)
from x_fuse.fusion import CorrelativeFuser
from x_fuse.loaders import load_correlative_tiff_pair
from x_fuse.synthetic import generate_correlative_phantom_3d
from x_fuse.volumetric import (
    VolumetricFuser,
    pack_25d_slice,
    z_slice_consistency_score,
)


def test_pack_25d_slice_axes_and_boundary_clamping():
    vol = np.arange(6 * 8 * 10, dtype=np.float32).reshape(6, 8, 10)
    s_z0 = pack_25d_slice(vol, index=0, axis=0, delta=1)
    assert s_z0.shape == (3, 8, 10)
    # Reflection at index=0 with delta=1 gives [vol[1], vol[0], vol[1]]
    assert np.array_equal(s_z0[0], vol[1])
    assert np.array_equal(s_z0[1], vol[0])
    assert np.array_equal(s_z0[2], vol[1])

    s_y = pack_25d_slice(vol, index=7, axis=1, delta=2)
    assert s_y.shape == (3, 6, 10)

    s_x = pack_25d_slice(vol, index=4, axis=2, delta=1)
    assert s_x.shape == (3, 6, 8)


def test_volumetric_fuser_sparse_slices_and_triplane():
    phantom_3d = generate_correlative_phantom_3d(
        shape=(16, 32, 32), downsample_factor=4, seed=21
    )
    fuser = CorrelativeFuser(
        dino_name="timm:vit_tiny_patch16_224",
        xrd_fuse_method="learned_gating",
        stride=8,
        dtype=torch.float32,
        loss_fn="forward_psf_bce",
        guide_channels=2,
        pretrained=False,
        device="cpu",
    )
    vol_fuser = VolumetricFuser(fuser, tta_shifts=[], tta_flips=False)

    # Provide only 2 sparse 2D chemical slices out of Z=16!
    sparse_guides = {
        3: phantom_3d.coarse_guide_3d[:, 0],  # (2, 8, 8)
        11: phantom_3d.coarse_guide_3d[:, 2],
    }
    losses = vol_fuser.fit_sparse_slices(
        phantom_3d.hr_volume, sparse_guides=sparse_guides, epochs=3, delta=1
    )
    assert len(losses) == 3
    assert all(np.isfinite(losses))

    rep_slab = vol_fuser.predict_volume(
        phantom_3d.hr_volume,
        mode="2.5d_slab",
        delta=1,
        sparse_guides=sparse_guides,
        pred_mode="multiclass",
    )
    assert rep_slab.probabilities.shape == (3, 16, 32, 32)
    assert rep_slab.hard_mask.shape == (16, 32, 32)
    assert 0.0 <= rep_slab.z_slice_consistency <= 1.0

    rep_tri = vol_fuser.predict_volume(
        phantom_3d.hr_volume,
        mode="triplane",
        delta=1,
        coarse_guide_3d=phantom_3d.coarse_guide_3d,
        pred_mode="multiclass",
    )
    assert rep_tri.probabilities.shape == (3, 16, 32, 32)
    assert rep_tri.triplane_variance.shape == (16, 32, 32)
    assert float(rep_tri.triplane_variance.max()) >= 0.0
    assert z_slice_consistency_score(rep_tri.hard_mask) >= 0.0


def test_bundled_samples_and_experimental_holdout():
    zn_dir = get_sample_dataset_path("Zn13X")
    mor_dir = get_sample_dataset_path("MOR-GIS")
    assert (zn_dir / "high_4.tiff").exists()
    assert (mor_dir / "xct_12.png").exists()

    hr, guide, names = load_correlative_tiff_pair(
        zn_dir / "high_4.tiff", [zn_dir / "low_4.tiff"]
    )
    assert hr.ndim == 2
    assert guide.ndim == 3 and guide.shape[0] == 1
    assert names == ["low_4"]

    hr_c, coarse, gt = make_experimental_holdout(
        np.random.rand(66, 70).astype(np.float32),
        np.random.rand(2, 66, 70).astype(np.float32),
        factor=8,
        poisson_counts=100.0,
        seed=4,
    )
    assert hr_c.shape == (64, 64)
    assert gt.shape == (2, 64, 64)
    assert coarse.shape == (2, 8, 8)

    registry = list_open_datasets()
    assert "edxia_cement_sem_eds" in registry
    assert "isis_imat_bragg_edge" in registry
