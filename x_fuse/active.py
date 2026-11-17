"""Uncertainty-driven active acquisition & experiment steering for correlative
tomography and microscopy (e.g., selecting optimal sparse XRD-CT / FIB-EDS
z-slices or targeted high-dose 2D STEM-EELS / WDS sub-regions during beamtime).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .uncertainty import UncertaintyReport
from .volumetric import VolumetricUncertaintyReport


@dataclass(frozen=True)
class ActiveSliceRecommendation:
    """Ranked sparse-slice acquisition plan for 3D correlative tomography."""

    recommended_indices: list[int]
    slice_scores: np.ndarray  # (Z,) composite acquisition utility in [0, 1]
    acquired_indices: list[int]
    expected_uncertainty_gain: float


@dataclass(frozen=True)
class ActiveROI2D:
    """High-uncertainty / high-residual 2D region of interest ``(y0, x0, y1, x1)``."""

    y0: int
    x0: int
    y1: int
    x1: int
    score: float
    mean_uncertainty: float
    mean_physical_residual: float


def suggest_next_slices(
    vol_report: VolumetricUncertaintyReport,
    acquired_indices: Sequence[int] = (),
    k: int = 3,
    min_spacing: int = 4,
    weights: tuple[float, float, float] = (0.45, 0.45, 0.10),
) -> ActiveSliceRecommendation:
    """Recommend the next ``k`` 2D chemical/diffraction slices along ``Z`` to
    acquire in a 3D structural tomogram.

    Combines:
    1. Slice-mean **tri-plane / TTA epistemic variance** (structural view disagreement),
    2. Slice-mean **posterior chemical entropy** (phase boundary ambiguity), and
    3. Dynamic **spatial diversity bonus** (distance from nearest already-acquired
       or newly scheduled slice) so beamtime isn't spent on redundant adjacent slices.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")

    epi_zyx = vol_report.triplane_variance.detach().cpu().float().numpy()
    ent_zyx = vol_report.aleatoric_entropy.detach().cpu().float().numpy()
    z_dim = int(epi_zyx.shape[0])

    epi_z = epi_zyx.mean(axis=(1, 2))
    ent_z = ent_zyx.mean(axis=(1, 2))

    def _norm01(arr: np.ndarray) -> np.ndarray:
        lo, hi = float(np.min(arr)), float(np.max(arr))
        if hi - lo < 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - lo) / (hi - lo)).astype(np.float32)

    epi_norm = _norm01(epi_z)
    ent_norm = _norm01(ent_z)
    w_epi, w_ent, w_dist = weights
    intrinsic = w_epi * epi_norm + w_ent * ent_norm

    known = sorted({int(i) for i in acquired_indices if 0 <= int(i) < z_dim})
    selected: list[int] = []
    all_z = np.arange(z_dim, dtype=np.float32)

    # Record initial score profile before greedy updates
    if known:
        init_dist = np.min(
            np.abs(all_z[:, None] - np.asarray(known, dtype=np.float32)[None, :]),
            axis=1,
        )
        init_dist_norm = _norm01(init_dist)
    else:
        init_dist_norm = np.ones(z_dim, dtype=np.float32)
    initial_scores = (intrinsic + w_dist * init_dist_norm).astype(np.float32)

    max_picks = min(int(k), max(0, z_dim - len(known)))
    for _ in range(max_picks):
        occupied = known + selected
        if occupied:
            occ_arr = np.asarray(occupied, dtype=np.float32)
            min_d = np.min(np.abs(all_z[:, None] - occ_arr[None, :]), axis=1)
            dist_bonus = _norm01(min_d)
        else:
            min_d = np.full(z_dim, float(z_dim), dtype=np.float32)
            dist_bonus = np.ones(z_dim, dtype=np.float32)

        step_scores = intrinsic + w_dist * dist_bonus
        # Prefer slices at least min_spacing away; relax to >= 1 if volume is tight
        valid = min_d >= max(1, int(min_spacing))
        if not np.any(valid):
            valid = min_d >= 1.0
        if not np.any(valid):
            break

        masked_scores = np.where(valid, step_scores, -np.inf)
        best_idx = int(np.argmax(masked_scores))
        selected.append(best_idx)

    raw_unc_z = 0.5 * epi_norm + 0.5 * ent_norm
    if selected:
        gain = float(np.mean(raw_unc_z[selected]))
    else:
        gain = 0.0

    return ActiveSliceRecommendation(
        recommended_indices=selected,
        slice_scores=initial_scores,
        acquired_indices=known,
        expected_uncertainty_gain=round(gain, 4),
    )


def _box_sum_2d(arr: np.ndarray, win_h: int, win_w: int) -> np.ndarray:
    """Exact O(H*W) sliding-window sum using a padded 2D integral image."""
    padded = np.pad(arr.astype(np.float64), ((1, 0), (1, 0)), mode="constant")
    ii = padded.cumsum(axis=0).cumsum(axis=1)
    return (
        ii[win_h:, win_w:]
        - ii[:-win_h, win_w:]
        - ii[win_h:, :-win_w]
        + ii[:-win_h, :-win_w]
    )


def suggest_2d_rois(
    report: UncertaintyReport,
    roi_size: tuple[int, int] = (64, 64),
    k: int = 3,
    residual_weight: float = 0.5,
) -> list[ActiveROI2D]:
    """Identify up to ``k`` non-overlapping 2D sub-windows ``(y0, x0, y1, x1)``
    with maximal combined uncertainty and forward physical residual for targeted
    high-dose re-acquisition (e.g., fine-step STEM-EELS or WDS).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")

    unc = report.combined_uncertainty[0].detach().cpu().float().numpy()
    res = report.physical_residual[0].detach().cpu().float().numpy()
    height, width = int(unc.shape[0]), int(unc.shape[1])
    win_h = max(1, min(int(roi_size[0]), height))
    win_w = max(1, min(int(roi_size[1]), width))
    area = float(win_h * win_w)

    w_r = float(np.clip(residual_weight, 0.0, 1.0))
    priority_map = (1.0 - w_r) * unc + w_r * res

    mean_prio = _box_sum_2d(priority_map, win_h, win_w) / area
    mean_unc = _box_sum_2d(unc, win_h, win_w) / area
    mean_res = _box_sum_2d(res, win_h, win_w) / area

    work = mean_prio.copy()
    rois: list[ActiveROI2D] = []
    out_h, out_w = work.shape

    for _ in range(int(k)):
        flat_idx = int(np.argmax(work))
        best_val = float(work.ravel()[flat_idx])
        if not np.isfinite(best_val):
            break
        y0, x0 = divmod(flat_idx, out_w)
        rois.append(
            ActiveROI2D(
                y0=int(y0),
                x0=int(x0),
                y1=int(y0 + win_h),
                x1=int(x0 + win_w),
                score=round(best_val, 4),
                mean_uncertainty=round(float(mean_unc[y0, x0]), 4),
                mean_physical_residual=round(float(mean_res[y0, x0]), 4),
            )
        )
        # Suppress overlapping candidate origins so ROIs are spatially disjoint
        sup_y0 = max(0, y0 - win_h + 1)
        sup_y1 = min(out_h, y0 + win_h)
        sup_x0 = max(0, x0 - win_w + 1)
        sup_x1 = min(out_w, x0 + win_w)
        work[sup_y0:sup_y1, sup_x0:sup_x1] = -np.inf

    return rois
