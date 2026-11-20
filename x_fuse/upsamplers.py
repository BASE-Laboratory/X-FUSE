"""Pure-PyTorch Joint Bilateral Upsampler (JBU) and Physical Forward-PSF
Downsampler harvested from FeatUp principles without custom CUDA extensions.

Supports both:
1. Edge-aware upsampling of dense ViT features or coarse physical maps
   conditioned on a high-resolution structural micrograph/tomogram ``I_HR``.
2. Differentiable physical downsampling ``D(P_HR) -> G_LR`` for partial-volume
   forward-model supervision and physical residual uncertainty scoring.
"""

import math
from typing import Literal, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F

PSFProfile: TypeAlias = Literal["gaussian", "moffat", "core_halo"]


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
    radius_override: int | None = None,
) -> torch.Tensor:
    """Build differentiable ``(C, 1, 2R+1, 2R+1)`` separable Gaussian kernels
    from ``sigma_yx`` of shape ``(C, 2)``.
    """
    max_sig = float(sigma_yx.detach().max().item())
    radius = (
        max(1, int(radius_override))
        if radius_override is not None
        else min(28, max(1, int(math.ceil(2.5 * max(max_sig, 0.4)))))
    )
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    sig_y = sigma_yx[:, 0:1].to(device=device, dtype=dtype).clamp(min=0.15)  # (C, 1)
    sig_x = sigma_yx[:, 1:2].to(device=device, dtype=dtype).clamp(min=0.15)  # (C, 1)
    gy = torch.exp(-(coords.unsqueeze(0) ** 2) / (2.0 * sig_y.square()))  # (C, K)
    gx = torch.exp(-(coords.unsqueeze(0) ** 2) / (2.0 * sig_x.square()))  # (C, K)
    g2d = gy.unsqueeze(2) * gx.unsqueeze(1)  # (C, K, K)
    g2d = g2d / g2d.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
    return g2d.unsqueeze(1)  # (C, 1, K, K)


def _make_anisotropic_moffat_kernels(
    sigma_yx: torch.Tensor,
    beta: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build FWHM-matched differentiable ``(C, 1, 2R+1, 2R+1)`` 2D Moffat kernels
    ``(1 + y^2 / alpha_y^2 + x^2 / alpha_x^2)^(-beta)`` with heavy diffraction wings.
    """
    max_sig = float(sigma_yx.detach().max().item())
    radius = min(28, max(2, int(math.ceil(3.5 * max(max_sig, 0.4)))))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")  # (K, K)

    sig_y = sigma_yx[:, 0].to(device=device, dtype=dtype).clamp(min=0.15)
    sig_x = sigma_yx[:, 1].to(device=device, dtype=dtype).clamp(min=0.15)
    beta_c = beta.to(device=device, dtype=dtype).clamp(1.15, 10.0)
    # Match Gaussian FWHM (2.3548 * sigma) to Moffat FWHM
    # (2 * alpha * sqrt(2^(1/beta) - 1))
    denom_fwhm = (torch.pow(2.0, 1.0 / beta_c) - 1.0).clamp(min=1e-3).sqrt()
    alpha_y = (1.17741 * sig_y / denom_fwhm).clamp(min=0.15).view(-1, 1, 1)
    alpha_x = (1.17741 * sig_x / denom_fwhm).clamp(min=0.15).view(-1, 1, 1)

    r2 = (yy.unsqueeze(0) / alpha_y).square() + (xx.unsqueeze(0) / alpha_x).square()
    k2d = torch.pow(1.0 + r2, -beta_c.view(-1, 1, 1))
    k2d = k2d / k2d.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
    return k2d.unsqueeze(1)


class ForwardPSFDownsampler(nn.Module):
    """Differentiable physical forward operator ``D(P_HR) -> G_LR`` modelling
    probe/interaction-volume blur followed by coarse detector pixel integration.

    Supports three physical beam/interaction profiles via ``psf_profile``:
    - ``"gaussian"``: Standard separable 2D Gaussian probe.
    - ``"moffat"``: Heavy-tailed Lorentzian/Moffat synchrotron/Airy diffraction wings
      with optional per-channel learnable power-law index ``beta``.
    - ``"core_halo"``: Two-Gaussian primary-probe + wide secondary-fluorescence /
      Bremsstrahlung / backscattered interaction skirt mixture.
    """

    def __init__(
        self,
        psf_sigma_ratio: float = 0.5,
        learnable: bool = False,
        n_channels: int = 1,
        psf_profile: PSFProfile = "gaussian",
    ):
        super().__init__()
        if psf_profile not in ("gaussian", "moffat", "core_halo"):
            raise ValueError(
                f"Unsupported psf_profile={psf_profile!r}; expected 'gaussian', "
                "'moffat', or 'core_halo'."
            )
        self.psf_sigma_ratio = float(psf_sigma_ratio)
        self.learnable = bool(learnable)
        self.n_channels = max(1, int(n_channels))
        self.psf_profile: PSFProfile = psf_profile

        if self.learnable:
            init_log = math.log(max(self.psf_sigma_ratio, 0.05))
            self.log_sigma_ratio = nn.Parameter(
                torch.full((self.n_channels, 2), init_log, dtype=torch.float32)
            )
        else:
            self.register_parameter("log_sigma_ratio", None)

        if self.learnable and self.psf_profile == "moffat":
            self.raw_moffat_beta: nn.Parameter | None = nn.Parameter(
                torch.full((self.n_channels,), -1.18, dtype=torch.float32)
            )
        else:
            self.register_parameter("raw_moffat_beta", None)

        if self.learnable and self.psf_profile == "core_halo":
            self.raw_halo_logit: nn.Parameter | None = nn.Parameter(
                torch.full((self.n_channels,), -0.60, dtype=torch.float32)
            )
            self.raw_halo_scale: nn.Parameter | None = nn.Parameter(
                torch.full((self.n_channels,), -0.40, dtype=torch.float32)
            )
        else:
            self.register_parameter("raw_halo_logit", None)
            self.register_parameter("raw_halo_scale", None)

    def _expand_per_channel(
        self, vec: torch.Tensor, channels: int, device: torch.device
    ) -> torch.Tensor:
        v = vec.to(device=device, dtype=torch.float32)
        if v.shape[0] == channels:
            return v
        if v.shape[0] == 1 and channels > 1:
            return v.expand(channels, *v.shape[1:])
        return v[:1].expand(channels, *v.shape[1:])

    def _effective_moffat_beta(
        self, channels: int, device: torch.device
    ) -> torch.Tensor:
        if self.learnable and self.raw_moffat_beta is not None:
            raw = self._expand_per_channel(self.raw_moffat_beta, channels, device)
            return 1.2 + 6.8 * torch.sigmoid(raw)
        return torch.full((channels,), 2.5, device=device, dtype=torch.float32)

    def _effective_halo_params(
        self, channels: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.learnable
            and self.raw_halo_logit is not None
            and self.raw_halo_scale is not None
        ):
            w_h = 0.45 * torch.sigmoid(
                self._expand_per_channel(self.raw_halo_logit, channels, device)
            )
            k_h = 2.0 + 4.0 * torch.sigmoid(
                self._expand_per_channel(self.raw_halo_scale, channels, device)
            )
            return w_h, k_h
        return (
            torch.full((channels,), 0.18, device=device, dtype=torch.float32),
            torch.full((channels,), 3.2, device=device, dtype=torch.float32),
        )

    def _effective_sigma_yx(
        self,
        channels: int,
        scale_h: float,
        scale_w: float,
        device: torch.device,
    ) -> torch.Tensor:
        assert self.log_sigma_ratio is not None
        ratios = self._expand_per_channel(self.log_sigma_ratio, channels, device).exp()
        ratios = ratios.clamp(0.05, 1.50)
        scales = torch.tensor([[scale_h, scale_w]], device=device, dtype=torch.float32)
        return ratios * scales  # (C, 2)

    def _build_profile_kernel(
        self, sigma_yx: torch.Tensor, channels: int, device: torch.device
    ) -> torch.Tensor:
        if self.psf_profile == "moffat":
            beta = self._effective_moffat_beta(channels, device)
            return _make_anisotropic_moffat_kernels(
                sigma_yx, beta, device=device, dtype=torch.float32
            )
        if self.psf_profile == "core_halo":
            w_halo, k_halo = self._effective_halo_params(channels, device)
            halo_sigma = sigma_yx * k_halo.unsqueeze(-1)
            max_h_sig = float(halo_sigma.detach().max().item())
            rad = min(28, max(2, int(math.ceil(2.5 * max(max_h_sig, 0.4)))))
            k_core = _make_anisotropic_gaussian_kernels(
                sigma_yx, device=device, dtype=torch.float32, radius_override=rad
            )
            k_skirt = _make_anisotropic_gaussian_kernels(
                halo_sigma, device=device, dtype=torch.float32, radius_override=rad
            )
            w_view = w_halo.view(channels, 1, 1, 1)
            mix = (1.0 - w_view) * k_core + w_view * k_skirt
            return mix / mix.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
        return _make_anisotropic_gaussian_kernels(
            sigma_yx, device=device, dtype=torch.float32
        )

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
            weight = self._build_profile_kernel(sigma_yx, channels, x.device)
            pad = weight.shape[-1] // 2
            pad_mode = "reflect" if min(h_hr, w_hr) > pad else "replicate"
            x_padded = F.pad(x, (pad, pad, pad, pad), mode=pad_mode)
            x = F.conv2d(x_padded, weight, groups=channels)
        elif isinstance(psf_sigma, tuple):
            sig_t = torch.tensor(
                [[float(psf_sigma[0]), float(psf_sigma[1])]],
                device=x.device,
                dtype=torch.float32,
            ).expand(channels, -1)
            if float(sig_t.max().item()) >= 0.25:
                weight = self._build_profile_kernel(sig_t, channels, x.device)
                pad = weight.shape[-1] // 2
                pad_mode = "reflect" if min(h_hr, w_hr) > pad else "replicate"
                x_padded = F.pad(x, (pad, pad, pad, pad), mode=pad_mode)
                x = F.conv2d(x_padded, weight, groups=channels)
        else:
            sigma = (
                float(psf_sigma)
                if psf_sigma is not None
                else self.psf_sigma_ratio * 0.5 * (scale_h + scale_w)
            )
            if sigma >= 0.4:
                if self.psf_profile == "gaussian":
                    kernel = _make_gaussian_kernel_2d(sigma, x.device, x.dtype)
                    weight = kernel.expand(channels, 1, -1, -1)
                else:
                    sig_t = torch.full(
                        (channels, 2), sigma, device=x.device, dtype=torch.float32
                    )
                    weight = self._build_profile_kernel(sig_t, channels, x.device).to(
                        x.dtype
                    )
                pad = weight.shape[-1] // 2
                pad_mode = "reflect" if min(h_hr, w_hr) > pad else "replicate"
                x_padded = F.pad(x, (pad, pad, pad, pad), mode=pad_mode)
                x = F.conv2d(x_padded, weight, groups=channels)

        pooled = F.adaptive_avg_pool2d(x, output_size=(h_lr, w_lr))
        return pooled.to(hr_map.dtype)

    @torch.no_grad()
    def diagnostics(
        self, hr_hw: tuple[int, int], lr_hw: tuple[int, int]
    ) -> dict[str, object]:
        """Return effective PSF blur sigmas and profile parameters in HR pixels."""
        scale_h = hr_hw[0] / max(lr_hw[0], 1)
        scale_w = hr_hw[1] / max(lr_hw[1], 1)
        dev = (
            self.log_sigma_ratio.device
            if self.log_sigma_ratio is not None
            else torch.device("cpu")
        )
        info: dict[str, object] = {"psf_profile": self.psf_profile}
        if not self.learnable or self.log_sigma_ratio is None:
            sig = round(self.psf_sigma_ratio * 0.5 * (scale_h + scale_w), 4)
            info["learnable_psf"] = False
            info["psf_sigma_hr_px"] = sig
        else:
            sig_yx = self._effective_sigma_yx(
                self.n_channels, scale_h, scale_w, dev
            ).cpu()
            info["learnable_psf"] = True
            info["psf_sigma_yx_hr_px"] = [
                [round(float(row[0].item()), 4), round(float(row[1].item()), 4)]
                for row in sig_yx
            ]
        if self.psf_profile == "moffat":
            betas = self._effective_moffat_beta(self.n_channels, dev).cpu()
            info["moffat_beta"] = [round(float(b.item()), 4) for b in betas]
        elif self.psf_profile == "core_halo":
            w_h, k_h = self._effective_halo_params(self.n_channels, dev)
            info["halo_weight"] = [round(float(w.item()), 4) for w in w_h.cpu()]
            info["halo_scale_ratio"] = [round(float(k.item()), 4) for k in k_h.cpu()]
        return info


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


def extract_multiscale_texture_descriptors(guidance: torch.Tensor) -> torch.Tensor:
    """Extract 8 ``D4``-equivariant HR structural & local-texture descriptor
    channels ``(B, 8, H, W)`` from ``guidance`` ``(B, C_g, H, W)``:
    ``[norm_intensity, grad_mag, grad_energy_3x3, lap_energy_3x3, mean_3x3,
    std_3x3, mean_7x7, std_7x7]``.

    Crucially, ``grad_energy_3x3``, ``lap_energy_3x3``, ``std_3x3``, and ``std_7x7``
    are invariant to both ``90 deg`` TTA rotations/flips and sub-pixel precipitate
    phase shifts, cleanly resolving boundaries between **iso-intensity textured
    phases** where classical grayscale bilateral kernels are completely blind!
    """
    if guidance.ndim == 3:
        guidance = guidance.unsqueeze(0)
    bsz, _, height, width = guidance.shape
    gray = guidance.float().mean(dim=1, keepdim=True)
    g_min = gray.flatten(1).min(dim=1).values.view(bsz, 1, 1, 1)
    g_max = gray.flatten(1).max(dim=1).values.view(bsz, 1, 1, 1)
    norm_g = (gray - g_min) / (g_max - g_min).clamp(min=1e-6)

    # Sobel gradient magnitude & pooled high-frequency texture energies
    mode1 = "reflect" if min(height, width) > 1 else "replicate"
    p1 = F.pad(norm_g, (1, 1, 1, 1), mode=mode1)
    ky = (
        torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            device=norm_g.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        / 4.0
    )
    kx = ky.transpose(-1, -2)
    klap = (
        torch.tensor(
            [[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]],
            device=norm_g.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        / 4.0
    )
    grad_y = F.conv2d(p1, ky)
    grad_x = F.conv2d(p1, kx)
    grad_mag = (grad_y.square() + grad_x.square() + 1e-8).sqrt()
    grad_energy3 = F.avg_pool2d(
        F.pad(grad_mag, (1, 1, 1, 1), mode=mode1), kernel_size=3, stride=1
    )
    lap_abs = F.conv2d(p1, klap).abs()
    lap_energy3 = F.avg_pool2d(
        F.pad(lap_abs, (1, 1, 1, 1), mode=mode1), kernel_size=3, stride=1
    )

    # Local 3x3 and 7x7 mean & texture energy (standard deviation)
    mean3 = F.avg_pool2d(p1, kernel_size=3, stride=1)
    sq3 = F.avg_pool2d(F.pad(norm_g.square(), (1, 1, 1, 1), mode=mode1), 3, 1)
    std3 = (sq3 - mean3.square()).clamp(min=0.0).sqrt()

    pad7 = 3 if min(height, width) > 3 else 1
    mode7 = "reflect" if min(height, width) > pad7 else "replicate"
    k7 = 2 * pad7 + 1
    mean7 = F.avg_pool2d(F.pad(norm_g, [pad7] * 4, mode=mode7), k7, 1)
    sq7 = F.avg_pool2d(F.pad(norm_g.square(), [pad7] * 4, mode=mode7), k7, 1)
    std7 = (sq7 - mean7.square()).clamp(min=0.0).sqrt()

    desc = torch.cat(
        [norm_g, grad_mag, grad_energy3, lap_energy3, mean3, std3, mean7, std7], dim=1
    )
    return desc


def extract_structure_tensor_field(
    guidance: torch.Tensor,
    sigmas: tuple[float, float] = (1.5, 3.5),
    eps: float = 1e-4,
) -> torch.Tensor:
    """Extract normalized multiscale **spin-2 structure-tensor** orientation fields
    ``(B, 2 * len(sigmas), H, W)`` from ``guidance`` ``(B, C_g, H, W)``.

    At each Gaussian integration scale ``sigma``, computes normalized spin-2 moments
    ``(G_sigma * [I_x^2 - I_y^2, 2 I_x I_y]) / (G_sigma * (I_x^2 + I_y^2) + eps)``.

    Properties:
    1. **Breaks iso-intensity orientation ambiguity**: Alternating bright/dark lamellae
       at ``theta`` vs. ``theta + 90 deg`` share identical scalar mean and variance,
       yet map to antipodal spin-2 vectors ``+s`` and ``-s`` with constant value across
       both bright and dark stripes inside each colony.
    2. **Exact ``O(2)`` / ``D_4`` equivariance**: Under any mirror flip or ``90 deg``
       rotation, ``(s_1, s_2)`` undergoes an orthogonal sign/swap transformation, so
       pairwise distances ``||s(p) - s(q)||_2^2`` and local coherence ``||s(p)||_2``
       are exact ``D_4`` invariants.
    """
    if guidance.ndim == 3:
        guidance = guidance.unsqueeze(0)
    gray = guidance.float().mean(dim=1, keepdim=True)
    g_mean = gray.mean(dim=(-2, -1), keepdim=True)
    g_std = gray.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp(min=1e-4)
    norm_g = (gray - g_mean) / g_std

    height, width = int(norm_g.shape[-2]), int(norm_g.shape[-1])
    pad_mode = "reflect" if min(height, width) > 1 else "replicate"
    g_pad = F.pad(norm_g, (1, 1, 1, 1), mode=pad_mode)

    sobel_x = (
        torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=norm_g.device,
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        / 8.0
    )
    sobel_y = sobel_x.transpose(-1, -2).contiguous()

    ix = F.conv2d(g_pad, sobel_x)
    iy = F.conv2d(g_pad, sobel_y)

    j_diff = ix.square() - iy.square()
    j_cross = 2.0 * ix * iy
    j_trace = ix.square() + iy.square()
    raw_tensor = torch.cat([j_diff, j_cross, j_trace], dim=1)  # (B, 3, H, W)

    fields: list[torch.Tensor] = []
    for sigma in sigmas:
        kern = _make_gaussian_kernel_2d(
            float(sigma), device=norm_g.device, dtype=torch.float32
        )
        rad = kern.shape[-1] // 2
        b_mode = "reflect" if min(height, width) > rad else "replicate"
        padded = F.pad(raw_tensor, [rad] * 4, mode=b_mode)
        smoothed = F.conv2d(padded, kern.expand(3, 1, -1, -1), groups=3)
        denom = smoothed[:, 2:3].clamp(min=0.0) + float(eps)
        s_vec = (smoothed[:, 0:2] / denom).clamp(-1.0, 1.0)
        fields.append(s_vec)

    return torch.cat(fields, dim=1)


class CrossAttentionFeatureUpsampler(nn.Module):
    """``JAFAR``/``AnyUp``-inspired local HR-query cross-attention upsampler running
    in ``O(H * W * K^2)`` memory without custom CUDA kernels.

    Why this improves over classical ``JointBilateralUpsampler``:
    1. Queries ``Q_HR(y, x)`` combine multi-scale HR structural + local **texture
       energy** descriptors (``std_3x3``, ``std_7x7``, ``Laplacian``) with optional
       ``D_4``-equivariant **spin-2 structure-tensor** orientation fields
       (``orientation_aware=True``), cleanly separating both textured vs. smooth
       phases and differently oriented iso-intensity lamellar colonies.
    2. When ``subtoken_offsets=True``, performs direct ``3 x 3`` coarse-token
       neighborhood cross-attention conditioned on continuous sub-token coordinate
       offsets ``Delta_{pq}`` before fine HR boundary snapping, eliminating
       ``14-16 px`` bilinear pre-blur across grain boundaries.
    3. Works both **zero-shot (``learned=False``)** and **end-to-end trainable
       (``learned=True``)** under the physical forward-PSF loss.
    """

    def __init__(
        self,
        feat_dim: int | None = None,
        key_dim: int = 16,
        radius: int = 2,
        sigma_spatial: float = 2.0,
        temperature: float = 0.08,
        learned: bool = False,
        orientation_aware: bool = True,
        subtoken_offsets: bool = True,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.key_dim = int(key_dim)
        self.radius = int(radius)
        self.diameter = 2 * self.radius + 1
        self.learned = bool(learned)
        self.orientation_aware = bool(orientation_aware)
        self.subtoken_offsets = bool(subtoken_offsets)

        if learned:
            self.sigma_spatial = nn.Parameter(torch.tensor(float(sigma_spatial)))
            self.subtoken_sigma = nn.Parameter(torch.tensor(0.85))
            self.log_inv_temp = nn.Parameter(
                torch.tensor(math.log(1.0 / max(temperature, 1e-3)))
            )
            self.raw_spin2_weight = nn.Parameter(torch.tensor(0.85))
            self.hr_query_stem: nn.Sequential | None = nn.Sequential(
                nn.Conv2d(8, self.key_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(self.key_dim, self.key_dim, kernel_size=1),
            )
            self.vit_key_proj: nn.Conv2d | None = (
                nn.Conv2d(feat_dim, self.key_dim, kernel_size=1)
                if feat_dim is not None
                else None
            )
            if self.vit_key_proj is not None:
                nn.init.zeros_(self.vit_key_proj.weight)
                if self.vit_key_proj.bias is not None:
                    nn.init.zeros_(self.vit_key_proj.bias)
            self.subtoken_mlp: nn.Sequential | None = nn.Sequential(
                nn.Linear(3, 8),
                nn.GELU(),
                nn.Linear(8, 1),
            )
            last_lin = self.subtoken_mlp[-1]
            assert isinstance(last_lin, nn.Linear)
            nn.init.zeros_(last_lin.weight)
            nn.init.zeros_(last_lin.bias)
        else:
            self.register_buffer("sigma_spatial", torch.tensor(float(sigma_spatial)))
            self.register_buffer("subtoken_sigma", torch.tensor(0.85))
            self.register_buffer(
                "log_inv_temp",
                torch.tensor(math.log(1.0 / max(temperature, 1e-3))),
            )
            self.register_buffer("raw_spin2_weight", torch.tensor(1.35))
            self.hr_query_stem = None
            self.vit_key_proj = None
            self.subtoken_mlp = None

    def _spatial_log_prior(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        coords = torch.arange(-self.radius, self.radius + 1, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        dist2 = yy.square() + xx.square()
        sig = self.sigma_spatial.to(device=device, dtype=dtype).abs().clamp(min=0.2)
        return (-dist2 / (2.0 * sig.square())).reshape(
            1, self.diameter * self.diameter, 1, 1
        )

    def _encode_qk(
        self,
        desc: torch.Tensor,
        vit_feats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.learned and self.hr_query_stem is not None:
            q_map = F.normalize(self.hr_query_stem(desc), dim=1)
            k_map = q_map
            if (
                vit_feats is not None
                and self.vit_key_proj is not None
                and vit_feats.shape[1] == self.vit_key_proj.in_channels
            ):
                k_map = F.normalize(q_map + self.vit_key_proj(vit_feats), dim=1)
            return q_map, k_map

        ch_weights = torch.tensor(
            [0.45, 0.35, 1.8, 1.8, 1.5, 2.5, 1.2, 2.5],
            device=desc.device,
            dtype=torch.float32,
        ).view(1, 8, 1, 1)
        q_map = desc * ch_weights
        return q_map, q_map

    def _spin2_scale(self, device: torch.device) -> torch.Tensor:
        raw = self.raw_spin2_weight.to(device=device, dtype=torch.float32)
        return F.softplus(raw)

    def _compute_attention_weights(
        self,
        hr_desc: torch.Tensor,
        source_hr: torch.Tensor,
        dilation: int,
        spin2_hr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, _, height, width = hr_desc.shape
        q_map, k_map = self._encode_qk(hr_desc, vit_feats=source_hr)

        pad = self.radius * dilation
        pad_mode = "reflect" if min(height, width) > pad else "replicate"
        k_pad = F.pad(k_map, [pad] * 4, mode=pad_mode)
        k_patches = F.unfold(
            k_pad, kernel_size=self.diameter, dilation=dilation
        ).reshape(bsz, q_map.shape[1], self.diameter * self.diameter, height, width)

        inv_temp = (
            self.log_inv_temp.to(device=hr_desc.device, dtype=torch.float32)
            .exp()
            .clamp(0.5, 500.0)
        )
        if self.learned and self.hr_query_stem is not None:
            sim = (k_patches * q_map.unsqueeze(2)).sum(dim=1) * inv_temp
        else:
            diff2 = (k_patches - q_map.unsqueeze(2)).square().mean(dim=1)
            sim = -inv_temp * diff2

        if spin2_hr is not None:
            s_pad = F.pad(spin2_hr, [pad] * 4, mode=pad_mode)
            s_patches = F.unfold(
                s_pad, kernel_size=self.diameter, dilation=dilation
            ).reshape(
                bsz, spin2_hr.shape[1], self.diameter * self.diameter, height, width
            )
            s_diff2 = (s_patches - spin2_hr.unsqueeze(2)).square().mean(dim=1)
            w_s2 = self._spin2_scale(hr_desc.device)
            if self.learned and self.hr_query_stem is not None:
                sim = sim - w_s2 * s_diff2 * inv_temp.clamp(max=25.0)
            else:
                sim = sim - w_s2 * inv_temp * s_diff2

        logits = sim + self._spatial_log_prior(hr_desc.device, torch.float32)
        return torch.softmax(logits, dim=1)  # (B, K*K, H, W)

    def _subtoken_coarse_attend(
        self,
        source_lr: torch.Tensor,
        hr_desc: torch.Tensor,
        spin2_hr: torch.Tensor | None,
    ) -> torch.Tensor:
        """Attend each HR pixel ``(y, x)`` directly over its ``3 x 3`` nearest
        coarse ViT tokens using continuous sub-token offsets ``Delta_{pq}``.
        """
        bsz, src_ch, h_s, w_s = source_lr.shape
        _, _, target_h, target_w = hr_desc.shape
        dev = hr_desc.device

        lr_desc = F.interpolate(hr_desc, size=(h_s, w_s), mode="area")
        q_hr, _ = self._encode_qk(hr_desc, vit_feats=None)
        _, k_lr = self._encode_qk(lr_desc, vit_feats=source_lr.float())

        y_hr = torch.arange(target_h, device=dev, dtype=torch.float32)
        x_hr = torch.arange(target_w, device=dev, dtype=torch.float32)
        y_in_lr = (y_hr + 0.5) * (float(h_s) / float(target_h))
        x_in_lr = (x_hr + 0.5) * (float(w_s) / float(target_w))

        # Break exact-integer coordinate ties symmetrically toward image center for D_4
        eps_y = 1e-6 * torch.sign(0.5 * float(h_s) - y_in_lr)
        eps_x = 1e-6 * torch.sign(0.5 * float(w_s) - x_in_lr)
        u0 = torch.floor(y_in_lr + eps_y).long().clamp(0, h_s - 1)
        v0 = torch.floor(x_in_lr + eps_x).long().clamp(0, w_s - 1)
        lin_idx = (u0.unsqueeze(1) * w_s + v0.unsqueeze(0)).reshape(-1)

        dy_sub = y_in_lr - (u0.float() + 0.5)
        dx_sub = x_in_lr - (v0.float() + 0.5)

        shifts = torch.tensor([-1.0, 0.0, 1.0], device=dev, dtype=torch.float32)
        du, dv = torch.meshgrid(shifts, shifts, indexing="ij")
        delta_y = du.reshape(9, 1, 1) - dy_sub.view(1, target_h, 1)
        delta_x = dv.reshape(9, 1, 1) - dx_sub.view(1, 1, target_w)
        dist2 = delta_y.square() + delta_x.square()

        sig_sub = (
            self.subtoken_sigma.to(device=dev, dtype=torch.float32).abs().clamp(min=0.2)
        )
        spatial_prior = (-dist2 / (2.0 * sig_sub.square())).unsqueeze(0)
        if self.learned and self.subtoken_mlp is not None:
            abs_y = delta_y.abs().expand(9, target_h, target_w)
            abs_x = delta_x.abs().expand(9, target_h, target_w)
            geom = torch.stack([dist2, abs_y + abs_x, abs_y * abs_x], dim=-1)
            spatial_prior = spatial_prior + self.subtoken_mlp(geom).squeeze(
                -1
            ).unsqueeze(0)

        k_lr_pad = F.pad(k_lr, [1, 1, 1, 1], mode="replicate")
        k_lr_unf = F.unfold(k_lr_pad, kernel_size=3).reshape(
            bsz, k_lr.shape[1], 9, h_s * w_s
        )
        k_at_hr = k_lr_unf.index_select(3, lin_idx).reshape(
            bsz, k_lr.shape[1], 9, target_h, target_w
        )

        inv_temp = (
            self.log_inv_temp.to(device=dev, dtype=torch.float32)
            .exp()
            .clamp(0.5, 500.0)
        )
        if self.learned and self.hr_query_stem is not None:
            sim = (k_at_hr * q_hr.unsqueeze(2)).sum(dim=1) * inv_temp
        else:
            diff2 = (k_at_hr - q_hr.unsqueeze(2)).square().mean(dim=1)
            sim = -inv_temp * diff2

        if spin2_hr is not None:
            spin2_lr = F.interpolate(spin2_hr, size=(h_s, w_s), mode="area")
            s_lr_pad = F.pad(spin2_lr, [1, 1, 1, 1], mode="replicate")
            s_lr_unf = F.unfold(s_lr_pad, kernel_size=3).reshape(
                bsz, spin2_hr.shape[1], 9, h_s * w_s
            )
            s_at_hr = s_lr_unf.index_select(3, lin_idx).reshape(
                bsz, spin2_hr.shape[1], 9, target_h, target_w
            )
            s_diff2 = (s_at_hr - spin2_hr.unsqueeze(2)).square().mean(dim=1)
            w_s2 = self._spin2_scale(dev)
            if self.learned and self.hr_query_stem is not None:
                sim = sim - w_s2 * s_diff2 * inv_temp.clamp(max=25.0)
            else:
                sim = sim - w_s2 * inv_temp * s_diff2

        attn = torch.softmax(sim + spatial_prior, dim=1)

        src_pad = F.pad(source_lr.float(), [1, 1, 1, 1], mode="replicate")
        need_grad = torch.is_grad_enabled() and (
            source_lr.requires_grad or attn.requires_grad
        )
        out_sub = torch.zeros(
            (bsz, src_ch, target_h, target_w), device=dev, dtype=torch.float32
        )
        k_idx = 0
        for du_i in range(3):
            u_slice = src_pad[:, :, du_i : du_i + h_s, :]
            u_hr = u_slice.index_select(2, u0)
            for dv_i in range(3):
                patch_lr = u_hr[:, :, :, dv_i : dv_i + w_s]
                patch_hr = patch_lr.index_select(3, v0)
                if need_grad:
                    out_sub = out_sub + patch_hr * attn[:, k_idx : k_idx + 1]
                else:
                    out_sub.addcmul_(patch_hr, attn[:, k_idx : k_idx + 1])
                k_idx += 1
        return out_sub

    def forward(
        self,
        source: torch.Tensor,
        guidance: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Upsample ``source`` ``(B, C, H_s, W_s)`` to ``guidance`` ``(B, C_g, H, W)``
        (or optional ``output_size``) via multi-scale texture + spin-2 cross-attention.
        """
        if guidance.ndim == 3:
            guidance = guidance.unsqueeze(0)
        if source.ndim == 3:
            source = source.unsqueeze(0)
        if output_size is not None and guidance.shape[-2:] != tuple(output_size):
            guidance = F.interpolate(
                guidance.float(),
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        bsz, src_ch, h_s, w_s = source.shape
        _, _, target_h, target_w = guidance.shape
        if guidance.shape[0] == 1 and bsz > 1:
            guidance = guidance.expand(bsz, -1, -1, -1)

        hr_desc = extract_multiscale_texture_descriptors(guidance)
        spin2_hr = (
            extract_structure_tensor_field(guidance) if self.orientation_aware else None
        )

        def _apply_pass(inp: torch.Tensor, dil: int) -> torch.Tensor:
            attn = self._compute_attention_weights(
                hr_desc, inp, dilation=dil, spin2_hr=spin2_hr
            )
            pad = self.radius * dil
            pad_mode = "reflect" if min(target_h, target_w) > pad else "replicate"
            padded = F.pad(inp.float(), [pad] * 4, mode=pad_mode)
            pass_grad = torch.is_grad_enabled() and (
                inp.requires_grad or attn.requires_grad
            )
            out_acc = torch.zeros(
                (bsz, src_ch, target_h, target_w),
                device=inp.device,
                dtype=torch.float32,
            )
            p_idx = 0
            for dy_i in range(self.diameter):
                y0 = dy_i * dil
                row_view = padded[:, :, y0 : y0 + target_h, :]
                for dx_i in range(self.diameter):
                    x0 = dx_i * dil
                    shifted = row_view[:, :, :, x0 : x0 + target_w]
                    if pass_grad:
                        out_acc = out_acc + shifted * attn[:, p_idx : p_idx + 1]
                    else:
                        out_acc.addcmul_(shifted, attn[:, p_idx : p_idx + 1])
                    p_idx += 1
            return out_acc

        if self.subtoken_offsets and (h_s != target_h or w_s != target_w):
            out = self._subtoken_coarse_attend(source, hr_desc, spin2_hr)
            out = _apply_pass(out, 1)
            return out.to(source.dtype)

        hr_init = F.interpolate(
            source.float(),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        scale = max(target_h / max(h_s, 1), target_w / max(w_s, 1))
        wide_dilation = max(1, int(round(0.55 * scale / max(self.radius, 1))))
        out = _apply_pass(hr_init, wide_dilation)
        if wide_dilation > 1:
            out = _apply_pass(out, 1)
        return out.to(source.dtype)
