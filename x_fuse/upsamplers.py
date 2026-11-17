"""Pure-PyTorch Joint Bilateral Upsampler (JBU) and Physical Forward-PSF
Downsampler harvested from FeatUp principles without custom CUDA extensions.

Supports both:
1. Edge-aware upsampling of dense ViT features or coarse physical maps
   conditioned on a high-resolution structural micrograph/tomogram ``I_HR``.
2. Differentiable physical downsampling ``D(P_HR) -> G_LR`` for partial-volume
   forward-model supervision and physical residual uncertainty scoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_gaussian_kernel_2d(
    sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    radius = max(1, int(round(2.5 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g1d = torch.exp(-(coords**2) / (2.0 * max(sigma, 1e-4) ** 2))
    g2d = torch.outer(g1d, g1d)
    g2d = g2d / g2d.sum().clamp(min=1e-12)
    return g2d.view(1, 1, 2 * radius + 1, 2 * radius + 1)


class ForwardPSFDownsampler(nn.Module):
    """Differentiable physical forward operator ``D(P_HR) -> G_LR`` modelling
    probe/interaction-volume blur followed by coarse detector pixel integration."""

    def __init__(self, psf_sigma_ratio: float = 0.35):
        super().__init__()
        self.psf_sigma_ratio = psf_sigma_ratio

    def forward(
        self,
        hr_map: torch.Tensor,
        target_size: tuple[int, int],
        psf_sigma: float | None = None,
    ) -> torch.Tensor:
        """Downsample ``hr_map`` ``(B, M, H, W)`` to ``(H_lr, W_lr)``."""
        bsz, channels, h_hr, w_hr = hr_map.shape
        h_lr, w_lr = target_size
        if (h_hr, w_hr) == (h_lr, w_lr):
            return hr_map

        scale_h = h_hr / max(h_lr, 1)
        scale_w = w_hr / max(w_lr, 1)
        sigma = (
            psf_sigma
            if psf_sigma is not None
            else self.psf_sigma_ratio * 0.5 * (scale_h + scale_w)
        )

        x = hr_map.float()
        if sigma >= 0.4:
            kernel = _make_gaussian_kernel_2d(sigma, x.device, x.dtype)
            pad = kernel.shape[-1] // 2
            x_padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
            weight = kernel.expand(channels, 1, -1, -1)
            x = F.conv2d(x_padded, weight, groups=channels)

        pooled = F.adaptive_avg_pool2d(x, output_size=(h_lr, w_lr))
        return pooled.to(hr_map.dtype)


class JointBilateralUpsampler(nn.Module):
    """Pure-PyTorch Joint Bilateral Upsampler (JBU) using ``F.unfold``.

    Can run either:
    - **Training-free (`learned=False`)**: Classical Gaussian spatial + intensity
      bilateral kernel conditioned on ``guidance`` (works zero-shot on ViT features
      or as a strong non-ML baseline upsampling ``G_LR`` directly!).
    - **Learned (`learned=True`)**: FeatUp-inspired learned 1x1 range projection
      and log-temperature parameter.
    """

    def __init__(
        self,
        guidance_dim: int = 3,
        key_dim: int = 16,
        radius: int = 2,
        dilation: int = 1,
        sigma_spatial: float = 1.5,
        sigma_range: float = 0.15,
        learned: bool = False,
    ):
        super().__init__()
        self.guidance_dim = guidance_dim
        self.key_dim = key_dim
        self.radius = radius
        self.dilation = max(1, int(dilation))
        self.diameter = 2 * radius + 1
        self.learned = learned
        self.range_proj: nn.Sequential | None = None

        if learned:
            self.sigma_spatial = nn.Parameter(torch.tensor(float(sigma_spatial)))
            self.log_temp = nn.Parameter(
                torch.tensor(float(-2.0 * torch.log(torch.tensor(sigma_range))))
            )
            self.range_proj = nn.Sequential(
                nn.Conv2d(guidance_dim, key_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(key_dim, key_dim, kernel_size=1),
            )
        else:
            self.register_buffer("sigma_spatial", torch.tensor(float(sigma_spatial)))
            self.register_buffer(
                "inv_two_sigma_r2",
                torch.tensor(1.0 / (2.0 * max(sigma_range, 1e-4) ** 2)),
            )

    def _spatial_weights(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        coords = torch.arange(-self.radius, self.radius + 1, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        dist2 = yy.square() + xx.square()
        sigma_s = (
            self.sigma_spatial.to(device=device, dtype=dtype).abs().clamp(min=1e-3)
        )
        w_spatial = torch.exp(-dist2 / (2.0 * sigma_s.square()))
        return w_spatial.reshape(1, self.diameter * self.diameter, 1, 1)

    def _compute_kernel(self, guidance: torch.Tensor) -> torch.Tensor:
        """Return normalized ``(B, K*K, H, W)`` bilateral weights."""
        bsz, _, height, width = guidance.shape
        g_float = guidance.float()
        if self.learned and self.range_proj is not None:
            proj = self.range_proj(g_float)
            temp = self.log_temp.exp().clamp(1e-3, 1e4)
        else:
            # Normalize guidance to [0, 1] per sample so sigma_range is scale-invariant
            g_min = g_float.flatten(1).min(dim=1).values.view(bsz, 1, 1, 1)
            g_max = g_float.flatten(1).max(dim=1).values.view(bsz, 1, 1, 1)
            proj = (g_float - g_min) / (g_max - g_min).clamp(min=1e-6)
            inv_buf = self.inv_two_sigma_r2
            assert isinstance(inv_buf, torch.Tensor)
            temp = inv_buf.to(device=g_float.device, dtype=torch.float32)

        k_ch = proj.shape[1]
        pad = self.radius * self.dilation
        padded_g = F.pad(proj, [pad] * 4, mode="reflect")
        patches_g = F.unfold(
            padded_g, kernel_size=self.diameter, dilation=self.dilation
        ).reshape(bsz, k_ch, self.diameter * self.diameter, height, width)
        diff2 = (patches_g - proj.unsqueeze(2)).square().mean(dim=1)  # (B, K*K, H, W)
        log_range = -temp * diff2
        spatial = self._spatial_weights(g_float.device, torch.float32)
        weights = torch.softmax(log_range, dim=1) * spatial
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return weights

    def forward(self, source: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        """Upsample ``source`` ``(B, C, H_s, W_s)`` to ``guidance`` resolution."""
        if guidance.ndim == 3:
            guidance = guidance.unsqueeze(0)
        if source.ndim == 3:
            source = source.unsqueeze(0)

        bsz, src_ch, _, _ = source.shape
        _, _, target_h, target_w = guidance.shape
        if guidance.shape[0] == 1 and bsz > 1:
            guidance = guidance.expand(bsz, -1, -1, -1)

        weights = self._compute_kernel(guidance)  # (B, K*K, H, W)
        hr_init = F.interpolate(
            source.float(),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        pad = self.radius * self.dilation
        padded_src = F.pad(hr_init, [pad] * 4, mode="reflect")

        # Process in channel chunks so memory stays tiny even for C=1024/1536 ViTs!
        chunk_size = 64
        out_chunks = []
        for c_start in range(0, src_ch, chunk_size):
            c_end = min(c_start + chunk_size, src_ch)
            sub = padded_src[:, c_start:c_end]
            c_sub = c_end - c_start
            unfolded = F.unfold(
                sub, kernel_size=self.diameter, dilation=self.dilation
            ).reshape(bsz, c_sub, self.diameter * self.diameter, target_h, target_w)
            filtered = (unfolded * weights.unsqueeze(1)).sum(dim=2)
            out_chunks.append(filtered)

        result = torch.cat(out_chunks, dim=1)
        return result.to(source.dtype)


def joint_bilateral_upsample(
    coarse_map: torch.Tensor,
    hr_guidance: torch.Tensor,
    radius: int = 3,
    sigma_spatial: float = 2.0,
    sigma_range: float = 0.12,
    n_iters: int = 2,
) -> torch.Tensor:
    """Training-free scale-dilated Joint Bilateral Upsampling (JBU) baseline for
    super-resolving a coarse physical map ``G_LR`` across large (4x-32x)
    resolution gaps using structural image ``I_HR``."""
    if hr_guidance.ndim == 3:
        hr_guidance = hr_guidance.unsqueeze(0)
    if coarse_map.ndim == 3:
        coarse_map = coarse_map.unsqueeze(0)

    scale = max(
        hr_guidance.shape[-2] / max(coarse_map.shape[-2], 1),
        hr_guidance.shape[-1] / max(coarse_map.shape[-1], 1),
    )
    # First pass uses scale-matched dilation (~half-cell reach) to snap the coarse
    # bilinear ramp without jumping across separate nearby features.
    wide_dilation = max(1, int(round(0.6 * scale / max(radius, 1))))
    jbu_wide = JointBilateralUpsampler(
        guidance_dim=hr_guidance.shape[1],
        radius=radius,
        dilation=wide_dilation,
        sigma_spatial=sigma_spatial,
        sigma_range=sigma_range,
        learned=False,
    ).to(coarse_map.device)
    out = jbu_wide(coarse_map, hr_guidance)

    if n_iters > 1:
        jbu_fine = JointBilateralUpsampler(
            guidance_dim=hr_guidance.shape[1],
            radius=radius,
            dilation=1,
            sigma_spatial=sigma_spatial,
            sigma_range=sigma_range,
            learned=False,
        ).to(coarse_map.device)
        for _ in range(n_iters - 1):
            out = jbu_fine(out, hr_guidance)
    return out
