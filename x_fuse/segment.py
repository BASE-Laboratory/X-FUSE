"""Downstream phase/element segmentation heads, classical & unsupervised baselines,
segmentation boundary/overlap metrics, and optional certainty-aware SAM2 refinement.
"""

from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.optimize import linear_sum_assignment
from skimage.filters import threshold_otsu
from sklearn.cluster import MiniBatchKMeans

from .uncertainty import (
    UncertaintyReport,
    build_uncertainty_report,
    match_coarse_priors,
)
from .upsamplers import joint_bilateral_upsample
from .vit_patch import TorchPCA, torch_pca


def do_single_pca(features: torch.Tensor, n_components: int = 3) -> np.ndarray:
    """Project a ``(1, C, H, W)`` or ``(C, H, W)`` feature tensor to ``(H, W, K)``
    min-max normalized ``[0, 1]`` PCA components."""
    if features.ndim == 4:
        features = features.squeeze(0)
    _, height, width = features.shape
    proj = torch_pca(features, dim=n_components)  # (H*W, K)
    arr = proj.detach().cpu().float().numpy().reshape(height, width, n_components)
    mn = arr.min(axis=(0, 1), keepdims=True)
    mx = arr.max(axis=(0, 1), keepdims=True)
    span = np.where((mx - mn) > 1e-8, mx - mn, 1.0)
    return ((arr - mn) / span).astype(np.float32)


def rescale_pca(pca_flat: torch.Tensor | np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Reshape flattened ``(H*W, K)`` PCA output to ``(H, W, K)`` in ``[0, 1]``."""
    arr = (
        pca_flat.detach().cpu().float().numpy()
        if isinstance(pca_flat, torch.Tensor)
        else np.asarray(pca_flat, dtype=np.float32)
    )
    k = arr.shape[-1]
    img = arr.reshape(hw[0], hw[1], k)
    mn = img.min(axis=(0, 1), keepdims=True)
    mx = img.max(axis=(0, 1), keepdims=True)
    span = np.where((mx - mn) > 1e-8, mx - mn, 1.0)
    return ((img - mn) / span).astype(np.float32)


def pca_otsu_segment(
    features: torch.Tensor,
    guide_lr: torch.Tensor | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment a gated ``(1, C, H, W)`` feature map via 1st-component PCA + Otsu,
    automatically resolving the arbitrary ``+-1`` eigenvector sign flip by
    correlating against ``guide_lr``.

    Returns:
        ``(binary_mask, pc1_score)`` of shape ``(H, W)`` in ``{0, 1}`` and ``[0, 1]``.
    """
    if features.ndim == 4:
        features = features.squeeze(0)
    _, height, width = features.shape

    flat = features.reshape(features.shape[0], -1).permute(1, 0).float()
    if float(flat.std()) < 1e-8:
        zeros = np.zeros((height, width), dtype=np.float32)
        return zeros.astype(np.int64), zeros

    pca = TorchPCA(n_components=1).fit(flat)
    pc1 = pca.transform(flat).reshape(height, width)

    if guide_lr is not None:
        g_t = (
            guide_lr.detach().float()
            if isinstance(guide_lr, torch.Tensor)
            else torch.from_numpy(np.asarray(guide_lr, dtype=np.float32))
        )
        if g_t.ndim == 2:
            g_t = g_t.unsqueeze(0).unsqueeze(0)
        elif g_t.ndim == 3:
            g_t = g_t.unsqueeze(0)
        g_hr = (
            F.interpolate(
                g_t[:, :1].to(pc1.device),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )
        pc_c = pc1 - pc1.mean()
        g_c = g_hr - g_hr.mean()
        cov = float((pc_c * g_c).sum().item())
        if cov < 0.0:
            pc1 = -pc1

    pc1_np = pc1.detach().cpu().numpy().astype(np.float32)
    mn, mx = float(np.min(pc1_np)), float(np.max(pc1_np))
    if mx - mn < 1e-8:
        return np.zeros((height, width), dtype=np.int64), np.zeros_like(pc1_np)

    pc1_norm = (pc1_np - mn) / (mx - mn)
    thresh = float(threshold_otsu(pc1_norm))
    mask = (pc1_norm >= thresh).astype(np.int64)
    return mask, pc1_norm


# ── Classical & Unsupervised Baselines for Quantitative Benchmarking ──────────


def _to_4d_guide(guide_lr: torch.Tensor | np.ndarray) -> torch.Tensor:
    g = (
        guide_lr.detach().float()
        if isinstance(guide_lr, torch.Tensor)
        else torch.from_numpy(np.asarray(guide_lr, dtype=np.float32))
    )
    if g.ndim == 2:
        return g.unsqueeze(0).unsqueeze(0)
    if g.ndim == 3:
        return g.unsqueeze(0)
    return g


def bicubic_upsample_baseline(
    guide_lr: torch.Tensor | np.ndarray,
    target_hw: tuple[int, int],
    mode: Literal["multiclass", "multilabel"] = "multilabel",
    calibrate: bool = True,
) -> UncertaintyReport:
    """Baseline 1: Direct bicubic interpolation of the coarse physical guide ``G_LR``
    without using the high-resolution structural image ``I_HR``."""
    g = _to_4d_guide(guide_lr)
    up = F.interpolate(g, size=target_hw, mode="bicubic", align_corners=False).clamp(
        0.0, 1.0
    )
    if calibrate:
        peak = up.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        logits = 12.0 * (up / peak - 0.5)
        if mode == "multiclass":
            bg_l = torch.zeros_like(logits[:, :1])
            probs = match_coarse_priors(
                torch.cat([bg_l, logits], dim=1), g, mode="multiclass"
            )
        else:
            probs = match_coarse_priors(logits, g, mode="multilabel")
    elif mode == "multiclass":
        bg = (1.0 - up.sum(dim=1, keepdim=True)).clamp(0.0, 1.0)
        probs = torch.cat([bg, up], dim=1)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
    else:
        probs = up
    return build_uncertainty_report(probs_hr=probs, guide_lr=g, mode=mode)


def jbu_guided_baseline(
    hr_image: torch.Tensor | np.ndarray,
    guide_lr: torch.Tensor | np.ndarray,
    radius: int = 3,
    sigma_spatial: float = 2.0,
    sigma_range: float = 0.12,
    n_iters: int = 2,
    mode: Literal["multiclass", "multilabel"] = "multilabel",
    calibrate: bool = True,
) -> UncertaintyReport:
    """Baseline 2: Classical non-ML Joint Bilateral Upsampling (JBU) of ``G_LR``
    guided directly by raw pixel intensities of ``I_HR`` (as used in SEM-EDS JBF
    literature, e.g. Georget et al., Cem. Concr. Res. 2021)."""
    g = _to_4d_guide(guide_lr)
    hr = (
        hr_image.detach().float()
        if isinstance(hr_image, torch.Tensor)
        else torch.from_numpy(np.asarray(hr_image, dtype=np.float32))
    )
    if hr.ndim == 2:
        hr = hr.unsqueeze(0).unsqueeze(0)
    elif hr.ndim == 3:
        hr = hr.unsqueeze(0)

    up = joint_bilateral_upsample(
        g.to(hr.device),
        hr,
        radius=radius,
        sigma_spatial=sigma_spatial,
        sigma_range=sigma_range,
        n_iters=n_iters,
    ).clamp(0.0, 1.0)

    if calibrate:
        peak = up.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        logits = 12.0 * (up / peak - 0.5)
        if mode == "multiclass":
            bg_l = torch.zeros_like(logits[:, :1])
            probs = match_coarse_priors(
                torch.cat([bg_l, logits], dim=1), g.to(hr.device), mode="multiclass"
            )
        else:
            probs = match_coarse_priors(logits, g.to(hr.device), mode="multilabel")
    else:
        if mode == "multiclass":
            bg = (1.0 - up.sum(dim=1, keepdim=True)).clamp(0.0, 1.0)
            probs = torch.cat([bg, up], dim=1)
            probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
        else:
            probs = up

    return build_uncertainty_report(probs_hr=probs, guide_lr=g.to(hr.device), mode=mode)


def dino_kmeans_baseline(
    features: torch.Tensor,
    guide_lr: torch.Tensor | np.ndarray,
    n_clusters: int | None = None,
    seed: int = 0,
) -> UncertaintyReport:
    """Baseline 3: Unsupervised ViT feature clustering (K-Means) with post-hoc
    Hungarian assignment matching clusters to coarse physical guide channels."""
    if features.ndim == 4:
        features = features.squeeze(0)
    channels, height, width = features.shape
    g = _to_4d_guide(guide_lr)
    n_guides = g.shape[1]
    k_clusters = n_clusters if n_clusters is not None else (n_guides + 1)

    feat_np = (
        features.detach().cpu().float().permute(1, 2, 0).reshape(-1, channels).numpy()
    )
    # Standardize per channel so zero-signal background features stay compact
    f_mean = feat_np.mean(axis=0, keepdims=True)
    f_std = np.maximum(feat_np.std(axis=0, keepdims=True), 1e-6)
    feat_normed = (feat_np - f_mean) / f_std

    km = MiniBatchKMeans(
        n_clusters=k_clusters, random_state=seed, batch_size=2048, n_init=3
    )
    labels = km.fit_predict(feat_normed).reshape(height, width)

    # Compute soft cluster probabilities via negative squared distance to centroids
    dists2 = np.mean(
        (feat_normed[:, None, :] - km.cluster_centers_[None, :, :]) ** 2, axis=-1
    )
    logits_k = -4.0 * dists2.reshape(height, width, k_clusters)
    exp_k = np.exp(logits_k - logits_k.max(axis=-1, keepdims=True))
    cluster_probs = exp_k / np.maximum(exp_k.sum(axis=-1, keepdims=True), 1e-8)

    # Align clusters to M foreground guide channels via Pearson correlation
    g_hr = (
        F.interpolate(g, size=(height, width), mode="bilinear", align_corners=False)
        .squeeze(0)
        .cpu()
        .numpy()
    )  # (M, H, W)
    cost = np.zeros((n_guides, k_clusters), dtype=np.float32)
    for m_idx in range(n_guides):
        gm = g_hr[m_idx].ravel()
        gm_c = gm - gm.mean()
        gm_n = np.linalg.norm(gm_c) + 1e-8
        for k_idx in range(k_clusters):
            ck = (labels == k_idx).astype(np.float32).ravel()
            ck_c = ck - ck.mean()
            corr = float(np.dot(gm_c, ck_c) / (gm_n * (np.linalg.norm(ck_c) + 1e-8)))
            cost[m_idx, k_idx] = -corr

    row_ind, col_ind = linear_sum_assignment(cost)
    matched_probs = np.zeros((1, n_guides, height, width), dtype=np.float32)
    for r, c in zip(row_ind, col_ind, strict=False):
        matched_probs[0, r] = cluster_probs[:, :, c]

    probs_t = torch.from_numpy(matched_probs).to(features.device)
    mode: Literal["multiclass", "multilabel"] = (
        "multilabel" if n_guides == 1 else "multiclass"
    )
    if mode == "multiclass":
        bg = (1.0 - probs_t.sum(dim=1, keepdim=True)).clamp(0.0, 1.0)
        probs_t = torch.cat([bg, probs_t], dim=1)
        probs_t = probs_t / probs_t.sum(dim=1, keepdim=True).clamp(min=1e-6)

    return build_uncertainty_report(
        probs_hr=probs_t, guide_lr=g.to(features.device), mode=mode
    )


# ── Segmentation & Boundary Evaluation Metrics ────────────────────────────────


def _extract_boundary(mask: np.ndarray, dilation_px: int = 1) -> np.ndarray:
    b = mask.astype(bool)
    if not b.any() or b.all():
        return np.zeros_like(b)
    struct = np.ones((3, 3), dtype=bool)
    dil = binary_dilation(b, structure=struct, iterations=max(1, dilation_px))
    ero = binary_erosion(b, structure=struct, iterations=max(1, dilation_px))
    return dil ^ ero


def compute_segmentation_metrics(
    pred_mask: np.ndarray | torch.Tensor,
    gt_mask: np.ndarray | torch.Tensor,
    boundary_tolerance_px: int = 2,
) -> dict[str, float]:
    """Compute pixel accuracy, macro foreground IoU, Dice, and boundary F1 score
    (BF1) between predicted and ground-truth 2D or 3D masks.

    Pixels marked ``-1`` in ``pred_mask`` (abstained via uncertainty thresholding)
    are excluded from retained evaluation and reported via ``coverage``.
    """
    pred = (
        pred_mask.detach().cpu().numpy()
        if isinstance(pred_mask, torch.Tensor)
        else np.asarray(pred_mask)
    )
    gt = (
        gt_mask.detach().cpu().numpy()
        if isinstance(gt_mask, torch.Tensor)
        else np.asarray(gt_mask)
    )
    if pred.shape != gt.shape:
        pred = np.squeeze(pred)
        gt = np.squeeze(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    valid = pred >= 0
    coverage = float(valid.mean())
    if not valid.any():
        return {
            "coverage": 0.0,
            "accuracy": 0.0,
            "iou": 0.0,
            "dice": 0.0,
            "boundary_f1": 0.0,
        }

    p_valid = pred[valid]
    g_valid = gt[valid]
    accuracy = float((p_valid == g_valid).mean())

    fg_classes = [int(c) for c in np.unique(gt) if c > 0]
    if not fg_classes:
        fg_classes = [1]

    ious = []
    dices = []
    bf1s = []
    for cls in fg_classes:
        p_c = p_valid == cls
        g_c = g_valid == cls
        inter = int((p_c & g_c).sum())
        union = int((p_c | g_c).sum())
        denom_d = int(p_c.sum() + g_c.sum())
        ious.append(inter / union if union > 0 else 1.0)
        dices.append((2.0 * inter) / denom_d if denom_d > 0 else 1.0)

        if pred.ndim == 2:
            pb = _extract_boundary(pred == cls, dilation_px=1)
            gb = _extract_boundary(gt == cls, dilation_px=1)
            if not pb.any() and not gb.any():
                bf1s.append(1.0)
            elif not pb.any() or not gb.any():
                bf1s.append(0.0)
            else:
                tol = np.ones((3, 3), dtype=bool)
                gb_tol = binary_dilation(
                    gb, structure=tol, iterations=max(1, boundary_tolerance_px)
                )
                pb_tol = binary_dilation(
                    pb, structure=tol, iterations=max(1, boundary_tolerance_px)
                )
                prec = float((pb & gb_tol).sum()) / float(pb.sum())
                rec = float((gb & pb_tol).sum()) / float(gb.sum())
                bf1s.append(
                    (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                )

    return {
        "coverage": coverage,
        "accuracy": accuracy,
        "iou": float(np.mean(ious)),
        "dice": float(np.mean(dices)),
        "boundary_f1": float(np.mean(bf1s)) if bf1s else float(np.mean(dices)),
    }


# ── Optional Certainty-Guided SAM2 Refinement ─────────────────────────────────


def load_sam2_predictor(
    model_cfg: str, checkpoint_path: str, device: str = "cpu"
) -> Any:
    """Lazy loader for ``SAM2ImagePredictor`` (requires ``pip install x-fuse[sam]``)."""
    try:
        from sam2.build_sam import build_sam2  # type: ignore[import-not-found]
        from sam2.sam2_image_predictor import (  # type: ignore[import-not-found]
            SAM2ImagePredictor,
        )
    except ImportError as exc:
        raise ImportError(
            "SAM2 is an optional dependency. Install with `pip install x-fuse[sam]` "
            "(or `uv sync --extra sam`) to use `load_sam2_predictor`."
        ) from exc
    sam_model = build_sam2(model_cfg, checkpoint_path, device=device)
    return SAM2ImagePredictor(sam_model)


def refine_with_sam2(
    hr_image_rgb_uint8: np.ndarray,
    report: UncertaintyReport,
    predictor: Any,
    target_class: int = 1,
    certainty_quantile: float = 0.80,
    max_points: int = 12,
    seed: int = 0,
) -> np.ndarray:
    """Prompt a SAM2 predictor using **only high-certainty core pixels** (low
    composite uncertainty and non-unresolved physical residual) so ambiguous
    phase boundaries are snapped without poisoning prompts with partial-volume noise.
    """
    hard = report.hard_mask[0].detach().cpu().numpy()
    unc = report.combined_uncertainty[0].detach().cpu().numpy()
    unresolved = report.unresolved_mask[0].detach().cpu().numpy()

    certain_cutoff = float(np.quantile(unc, 1.0 - certainty_quantile))
    reliable = (unc <= certain_cutoff) & (~unresolved)

    pos_yx = np.argwhere((hard == target_class) & reliable)
    neg_yx = np.argwhere((hard != target_class) & (hard >= 0) & reliable)
    if len(pos_yx) == 0:
        pos_yx = np.argwhere(hard == target_class)
    if len(pos_yx) == 0:
        return np.zeros_like(hard, dtype=np.int64)

    rng = np.random.default_rng(seed)
    if len(pos_yx) > max_points:
        pos_yx = pos_yx[rng.choice(len(pos_yx), size=max_points, replace=False)]
    if len(neg_yx) > max_points:
        neg_yx = neg_yx[rng.choice(len(neg_yx), size=max_points, replace=False)]

    pos_xy = pos_yx[:, ::-1].astype(np.float32)
    pts_list = [pos_xy]
    lbl_list = [np.ones(len(pos_xy), dtype=np.int32)]
    if len(neg_yx) > 0:
        neg_xy = neg_yx[:, ::-1].astype(np.float32)
        pts_list.append(neg_xy)
        lbl_list.append(np.zeros(len(neg_xy), dtype=np.int32))

    point_coords = np.concatenate(pts_list, axis=0)
    point_labels = np.concatenate(lbl_list, axis=0)

    all_fg = np.argwhere(hard == target_class)
    y0, x0 = all_fg.min(axis=0)
    y1, x1 = all_fg.max(axis=0)
    box = np.array([x0, y0, x1, y1], dtype=np.float32)

    predictor.set_image(hr_image_rgb_uint8)
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box[None, :],
        multimask_output=True,
    )
    best = int(np.argmax(scores))
    return (masks[best] > 0.5).astype(np.int64)
