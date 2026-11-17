"""Pure-PyTorch Joint Bilateral Upsampler (JBU) and Physical Forward-PSF
Downsampler harvested from FeatUp principles without custom CUDA extensions.

Supports both:
1. Edge-aware upsampling of dense ViT features or coarse physical maps
   conditioned on a high-resolution structural micrograph/tomogram ``I_HR``.
2. Differentiable physical downsampling ``D(P_HR) -> G_LR`` for partial-volume
   forward-model supervision and physical residual uncertainty scoring.
"""

import math

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


def _make_anisotropic_gaussian_kernels(
    sigma_yx: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build differentiable ``(C, 1, 2R+1, 2R+1)`` separable Gaussian kernels
    from ``sigma_yx`` of shape ``(C, 2)``.
    """
    max_sig = float(sigma_yx.detach().max().item())
    radius = max(1, int(math.ceil(2.5 * max(max_sig, 0.4))))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    sig_y = sigma_yx[:, 0:1].to(device=device, dtype=dtype).clamp(min=0.15)  # (C, 1)
    sig_x = sigma_yx[:, 1:2].to(device=device, dtype=dtype).clamp(min=0.15)  # (C, 1)
    gy = torch.exp(-(coords.unsqueeze(0) ** 2) / (2.0 * sig_y.square()))  # (C, K)
    gx = torch.exp(-(coords.unsqueeze(0) ** 2) / (2.0 * sig_x.square()))  # (C, K)
    g2d = gy.unsqueeze(2) * gx.unsqueeze(1)  # (C, K, K)
    g2d = g2d / g2d.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
    return g2d.unsqueeze(1)  # (C, 1, K, K)


class ForwardPSFDownsampler(nn.Module):
    """Differentiable physical forward operator ``D(P_HR) -> G_LR`` modelling
    probe/interaction-volume blur followed by coarse detector pixel integration.

    When ``learnable=True``, optimizes per-channel anisotropic PSF blur ratios
    ``(sigma_ratio_y, sigma_ratio_x)`` directly from coarse guide supervision.
    """

    def __init__(
        self,
        psf_sigma_ratio: float = 0.35,
        learnable: bool = False,
        n_channels: int = 1,
    ):
        super().__init__()
        self.psf_sigma_ratio = float(psf_sigma_ratio)
        self.learnable = bool(learnable)
        self.n_channels = max(1, int(n_channels))
        if self.learnable:
            init_log = math.log(max(self.psf_sigma_ratio, 0.05))
            self.log_sigma_ratio = nn.Parameter(
                torch.full((self.n_channels, 2), init_log, dtype=torch.float32)
            )
        else:
            self.register_parameter("log_sigma_ratio", None)

    def _effective_sigma_yx(
        self,
        channels: int,
        scale_h: float,
        scale_w: float,
        device: torch.device,
    ) -> torch.Tensor:
        assert self.log_sigma_ratio is not None
        ratios = self.log_sigma_ratio.to(device=device, dtype=torch.float32).exp()
        ratios = ratios.clamp(0.05, 1.50)
        if ratios.shape[0] == 1 and channels > 1:
            ratios = ratios.expand(channels, -1)
        elif ratios.shape[0] != channels:
            ratios = ratios[:1].expand(channels, -1)
        scales = torch.tensor([[scale_h, scale_w]], device=device, dtype=torch.float32)
        return ratios * scales  # (C, 2)

    def forward(
        self,
        hr_map: torch.Tensor,
        target_size: tuple[int, int],
        psf_sigma: float | tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Downsample ``hr_map`` ``(B, M, H, W)`` to ``(H_lr, W_lr)``."""
        _, channels, h_hr, w_hr = hr_map.shape
        h_lr, w_lr = target_size
        if (h_hr, w_hr) == (h_lr, w_lr):
            return hr_map

        scale_h = h_hr / max(h_lr, 1)
        scale_w = w_hr / max(w_lr, 1)
        x = hr_map.float()

        if self.learnable and psf_sigma is None and self.log_sigma_ratio is not None:
            sigma_yx = self._effective_sigma_yx(channels, scale_h, scale_w, x.device)
            weight = _make_anisotropic_gaussian_kernels(
                sigma_yx, device=x.device, dtype=torch.float32
            )
            pad = weight.shape[-1] // 2
            x_padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
            x = F.conv2d(x_padded, weight, groups=channels)
        elif isinstance(psf_sigma, tuple):
            sig_t = torch.tensor(
                [[float(psf_sigma[0]), float(psf_sigma[1])]],
                device=x.device,
                dtype=torch.float32,
            ).expand(channels, -1)
            if float(sig_t.max().item()) >= 0.25:
                weight = _make_anisotropic_gaussian_kernels(
                    sig_t, device=x.device, dtype=torch.float32
                )
                pad = weight.shape[-1] // 2
                x_padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
                x = F.conv2d(x_padded, weight, groups=channels)
        else:
            sigma = (
                float(psf_sigma)
                if psf_sigma is not None
                else self.psf_sigma_ratio * 0.5 * (scale_h + scale_w)
            )
            if sigma >= 0.4:
                kernel = _make_gaussian_kernel_2d(sigma, x.device, x.dtype)
                pad = kernel.shape[-1] // 2
                x_padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
                weight = kernel.expand(channels, 1, -1, -1)
                x = F.conv2d(x_padded, weight, groups=channels)

        pooled = F.adaptive_avg_pool2d(x, output_size=(h_lr, w_lr))
        return pooled.to(hr_map.dtype)

    @torch.no_grad()
    def diagnostics(
        self, hr_hw: tuple[int, int], lr_hw: tuple[int, int]
    ) -> dict[str, object]:
        """Return effective PSF blur sigmas in HR pixels."""
        scale_h = hr_hw[0] / max(lr_hw[0], 1)
        scale_w = hr_hw[1] / max(lr_hw[1], 1)
        if not self.learnable or self.log_sigma_ratio is None:
            sig = round(self.psf_sigma_ratio * 0.5 * (scale_h + scale_w), 4)
            return {"learnable_psf": False, "psf_sigma_hr_px": sig}
        sig_yx = self._effective_sigma_yx(
            self.n_channels, scale_h, scale_w, self.log_sigma_ratio.device
        ).cpu()
        return {
            "learnable_psf": True,
            "psf_sigma_yx_hr_px": [
                [round(float(row[0].item()), 4), round(float(row[1].item()), 4)]
                for row in sig_yx
            ],
        }


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
