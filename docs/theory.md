# Mathematical Theory

X-FUSE formulates correlative super-resolution segmentation as **physically constrained inverse inference** conditioned on dense self-supervised structural features.

---

## 1. Dense Sub-Patch Multi-Layer ViT Feature Extraction

Given a structural micrograph $I_{\text{HR}} \in \mathbb{R}^{H \times W}$, X-FUSE modifies the patch-embedding convolution stride of a pretrained Vision Transformer (DINOv2, DINOv3, ALiBi, NoPE, or `timm` ViT) from patch size $p$ (typically $14$ or $16$) down to sub-patch stride $s \in \{2, 4, 7, 8\}$ while interpolating positional embeddings accordingly.

When `layers="multiscale"` is enabled, patch tokens are extracted across shallow, mid-level, and deep transformer blocks $\ell \in \mathcal{L}$ (e.g. quarter, half, three-quarter, and final depth), each individually LayerNorm-stabilized and channel-concatenated:

$$
F_{\text{raw}}(x) = \bigoplus_{\ell \in \mathcal{L}} \mathrm{LN}\!\left(T^{(\ell)}(x)\right) \in \mathbb{R}^{(\sum_\ell D_\ell) \times H_f \times W_f}
$$

Shallow ViT blocks preserve sharp high-frequency pore, grain, and precipitate edges, while deeper blocks capture multi-grain textural context.

### Feature Upsampling: Scale-Dilated JBU & Texture-Aware Cross-Attention

Rather than nearest-neighbor or bilinear upsampling from the stride-$s$ token grid $(H_f, W_f)$ to $(H, W)$, X-FUSE provides two complementary high-resolution feature upsamplers:

1. **Scale-Dilated Joint Bilateral Upsampling (`interpolation_mode="jbu"`)**: Pure-PyTorch progressive bilateral filtering conditioned on normalized structural intensity $g = I_{\text{HR}}$:
   $$
   w_{\text{JBU}}(p, q) = \exp\!\left(-\frac{\|p - q\|_2^2}{2\sigma_s^2}\right) \exp\!\left(-\frac{\|g(p) - g(q)\|_2^2}{2\sigma_r^2}\right)
   $$
2. **Spin-2 Orientation-Aware & Continuous Sub-Token Cross-Attention (`interpolation_mode="cross_attn"`, `CrossAttentionFeatureUpsampler`)**: Inspired by continuous coordinate-query upsamplers (`JAFAR` / `AnyUp`), each pixel $p \in \{1, \dots, H\} \times \{1, \dots, W\}$ extracts an 8-channel multiscale texture descriptor $\tau(p)$ augmented—when `orientation_aware=True`—by a 4-channel multiscale **spin-2 structure-tensor director field** $\mathbf{s}(p) = (c_1 \cos 2\theta_1, c_1 \sin 2\theta_1, c_2 \cos 2\theta_2, c_2 \sin 2\theta_2)$ derived from smoothed outer products $J_\sigma = G_\sigma * (\nabla I_{\text{HR}} \nabla I_{\text{HR}}^\top)$ with local coherence $c = \frac{\lambda_1 - \lambda_2}{\lambda_1 + \lambda_2 + \epsilon}$. Because grain boundaries and lamellae carry headless orientation ($\theta \equiv \theta + \pi$), encoding $2\theta$ makes $\theta$ and $\theta + \pi$ identical while orthogonal colonies ($\theta + \pi/2$) negate both spin-2 components ($\cos 2(\theta + \pi/2) = -\cos 2\theta$), cleanly separating differently oriented iso-intensity colonies! When `subtoken_offsets=True`, continuous sub-token offsets $\Delta_{pq} = (\Delta y_{pq}, \Delta x_{pq}) \in [-1.5, 1.5]^2$ enter both via isotropic radial decay and an exact $D_4$-equivariant learned bilinear distance penalty.

### Apositional Feature Debiasing (`debias_positional=True`)

Pretrained Vision Transformers with learned absolute positional embeddings or RoPE encode linear and quadratic $(y, x)$ coordinate ramps in dedicated feature channels (`dino-saw`, Docherty et al., 2026). On materials exhibiting macroscopic chemical gradients (diffusion couples, battery electrodes, thermal-barrier coatings), a probe can overfit to image coordinates $(y, x)$ instead of microstructure. `ApositionalFeatureDebiaser` projects each feature channel onto the orthogonal complement of normalized degree-$d$ bivariate Legendre polynomials $\mathcal{B}_d(y, x)$, driving coordinate $R^2 \to 0$ on *any* standard ViT without custom re-pretraining.

---

## 2. Quantitative Forward Physics, Heavy-Tailed PSFs & B-Spline Registration

Naive correlative fusion upsamples the coarse chemical guide $G_{\text{LR}}$ to $(H, W)$ and trains against blurry targets—actively penalizing crisp high-resolution phase boundaries.

Instead, X-FUSE predicts high-resolution phase abundances $\hat{P}_{\text{HR}} \in [0, 1]^{M \times H \times W}$ via a `LinearProbeHead`, nonlinear `SpatialConvMLPHead` (`fusion_method="spatial_mlp"`), or $O(HW)$ `WindowedGuideCrossAttentionHead` (`fusion_method="attention"`), warps them into the guide frame via $\mathcal{W}_\theta$ (`DifferentiableGuideAligner`), and applies an anisotropic forward PSF operator $\mathcal{D}_\Sigma$ (`ForwardPSFDownsampler`):

$$
\hat{G}_{\text{LR}} = \mathcal{D}_\Sigma\!\left(\mathcal{W}_\theta(\hat{P}_{\text{HR}})\right) = \mathrm{AvgPool}_{r_y, r_x}\!\left(K_{\Sigma}^{(m)} * \mathcal{W}_\theta(\hat{P}_{\text{HR}})_m\right)
$$

To match real electron, X-ray, and neutron microprobes without Gaussian inverse crime, `ForwardPSFDownsampler` supports three learnable per-channel PSF families (`psf_profile`):

- **`"gaussian"`**: Separable anisotropic Gaussian $K_{\sigma_y, \sigma_x}(\Delta y, \Delta x) \propto \exp\!\left(-\frac{\Delta y^2}{2\sigma_y^2} - \frac{\Delta x^2}{2\sigma_x^2}\right)$.
- **`"moffat"`**: Non-separable heavy-tailed Lorentzian/Moffat probe skirt $K_{\alpha_y, \alpha_x, \beta}(\Delta y, \Delta x) \propto \left(1 + \frac{\Delta y^2}{\alpha_y^2} + \frac{\Delta x^2}{\alpha_x^2}\right)^{-\beta}$ with learnable tail index $\beta > 1.05$.
- **`"core_halo"`**: Double-Gaussian primary beam core plus secondary fluorescence / Monte Carlo backscattered interaction halo $(1 - w_h) K_{\sigma_y, \sigma_x}^{\text{core}} + w_h K_{\eta \sigma_y, \eta \sigma_x}^{\text{halo}}$ with learnable halo fraction $w_h \in (0, 0.5)$ and width ratio $\eta > 1$.

`PhysicsForwardModel` pairs these forward operators with quantitative counting-statistics objectives:

- **Poisson Shot-Noise Deviance (`loss_fn="forward_psf_poisson"`)**: Exact maximum-likelihood negative log-likelihood for low-dose STEM-EDS, EELS, XRF, and time-of-flight Bragg-edge photon/neutron counts $C_{\text{LR}} \sim \mathrm{Poisson}(s_m \hat{G}_{\text{LR}, m} + b_m)$ with learnable per-channel gain $s_m > 0$ and background $b_m \ge 0$:
  $$
  \mathcal{L}_{\text{Poisson}}(\hat{\Lambda}, C) = 2 \sum_{m, u} \left[\hat{\Lambda}_{m,u} - C_{m,u} + C_{m,u} \log\!\left(\frac{\max(C_{m,u}, \epsilon)}{\hat{\Lambda}_{m,u}}\right)\right]
  $$
- **Simplex Mass/Atomic-Fraction Conservation (`loss_fn="compositional_kl"`)**: Enforces $\hat{P}_{\text{HR}} = \mathrm{softmax}(Z_{\text{HR}}) \in \Delta^{K-1}$ ($\sum_{k=1}^K \hat{P}_{k}(x) = 1$ at every pixel) supervised via forward-PSF Kullback–Leibler divergence.
- **3D Spectral / Diffraction Cube Unmixing (`SpectralUnmixingDictionary`, `spectral_unmix=True`)**: Accepts raw $(E, H_{\text{LR}}, W_{\text{LR}})$ EDS/EELS/XRD cubes directly without external pre-fitting, jointly learning non-negative endmember spectra $S \in \mathbb{R}_{\ge 0}^{M \times E}$ ($\sum_e S_{m,e} = 1$, initialized via Successive Projection Algorithm + multiplicative NMF) alongside super-resolved spatial abundances: $\hat{Y}_{e, u} = \sum_{m=1}^M \alpha_m \hat{G}_{\text{LR}, m, u} S_{m, e} + b_e$.
- **Cubic B-Spline Non-Rigid Registration (`align_guide="bspline"`)**: Compensates nonlinear SEM/STEM charging drift and scan-coil hysteresis via a bicubic control-point displacement field regularized by thin-plate bending energy $\|\nabla^2 u\|_2^2$ and monitored via minimum Jacobian determinant $\min \det J_{\mathcal{W}}$ for diffeomorphic invertibility.
- **Fringe-Guarded Porous-Media Void Clamping (`pore_mask="auto_otsu_fringe_guarded"`)**: In porous catalysts and electrodes, structural voids ($M_{\text{pore}} = 1$) contain zero solid phase while coarse diffraction/EDS voxels suffer partial-volume pore dilution. `segment_pores_otsu` damps 1–2 px Fresnel propagation phase-contrast overshoots solely during threshold estimation via a Fourier Tie-homogeneity filter $\mathcal{F}^{-1}\{\mathcal{F}(I_{\text{HR}}) / (1 + \alpha \|\mathbf{k}\|_2^2)\}$ (without blurring $I_{\text{HR}}$) and clamps void logits prior to $\mathcal{D}_\Sigma$, deconvolving solid-framework stoichiometry with zero void leakage.

---

## 3. Prior Calibration, Tri-Part Uncertainty & Split Conformal Sets

After fitting, X-FUSE calibrates high-resolution posteriors via `match_coarse_priors` so spatial means match physical coarse phase fractions, and decomposes pixel/voxel uncertainty into:

1. **Aleatoric / Chemical Ambiguity ($U_{\text{alea}}$)**: Normalized Shannon entropy $-\frac{1}{\log K} \sum_{k=1}^{K} p_k(x) \log p_k(x) \in [0, 1]$.
2. **Epistemic / View Disagreement ($U_{\text{epi}}$)**: Computed in $O(1)$ auxiliary memory via an online **Welford accumulator** across invertible sub-patch TTA views (and orthogonal `XY`, `XZ`, `YZ` planes in 2.5D mode).
3. **Physical Forward-Model Residual ($U_{\text{phys}}$)**: Upsampled $|\mathcal{D}_\Sigma(\mathcal{W}_\theta(\hat{P}_{\text{HR}})) - G_{\text{LR}}|$, automatically flagging **structurally invisible phases** carrying zero contrast in $I_{\text{HR}}$.

### Temperature-Scaled Regularized Adaptive Conformal Prediction (`ConformalCalibrator`)

When `conformal_alpha` $\alpha \in (0, 1)$ is specified (or calibrated against sparse reference labels via `ConformalCalibrator.calibrate_labels`), X-FUSE supports Least Ambiguous Set-Valued Classifiers (`"lac"`), Adaptive Prediction Sets (`"aps"`), and **Regularized Adaptive Prediction Sets (`"raps"`, Angelopoulos et al., ICLR 2021)** with post-hoc NLL temperature scaling (`fit_temperature=True`) on spatially disjoint checkerboard blocks (`spatial_block_size` $b$):

$$
s_{\text{RAPS}}(x, y) = \sum_{j=1}^{r(x, y) - 1} \tilde{\pi}_{(j)}(x; T) + u \cdot \tilde{\pi}_{(r(x, y))}(x; T) + \lambda_{\text{reg}} \max\!\left(0,\, r(x, y) - k_{\text{reg}}\right)
$$

where $\tilde{\pi}(x; T) = \mathrm{softmax}(\log p(x) / T)$ is fit on half of the $b \times b$ spatial blocks and the conformal quantile $\hat{q}_{1-\alpha}$ is evaluated at finite-sample effective sample size $N_{\text{eff}} = \lfloor N_{\text{cal}} / b^2 \rfloor$ on the complementary blocks. This yields compact singleton sets inside homogeneous grain interiors while expanding cleanly to 2-phase sets along diffuse interphase boundaries.

---

## 4. 1D Voronoi-Integrated Active Sparse-Slice & ROI Steering

During synchrotron or electron-microscopy beamtime, `suggest_next_slices` ranks candidate unmeasured $z$-slices in a 3D tomogram by combining local axial-correlation-discounted uncertainty $\hat{U}(c \mid \mathcal{S}) = (w_{\text{epi}} \tilde{U}_{\text{epi}}(c) + w_{\text{alea}} \tilde{U}_{\text{alea}}(c))(1 - e^{-d_{\mathcal{S}}(c)/\ell_z})$ with **1D Voronoi distance-reduction integration** across the full axial stack:

$$
\mathcal{A}(c \mid \mathcal{S}) = 0.55\, \hat{U}(c \mid \mathcal{S}) + \frac{0.45}{Z_{\max}} \sum_{z=1}^{Z} \left(\hat{U}(z \mid \mathcal{S}) + w_{\text{dist}}\right) \max\!\left(0,\, d_{\mathcal{S}}(z) - |z - c|\right)
$$

where $d_{\mathcal{S}}(z) = \min_{s \in \mathcal{S}} |z - s|$, subject to minimum slice spacing $\Delta z_{\min}$. This immediately discovers localized reaction fronts on initial scouts while naturally bisecting unvisited bulk intervals as the acquisition budget grows. Similarly, `suggest_2d_rois` evaluates exact $O(HW)$ integral-image sliding-window sums over $(1 - w_r) U_{\text{comb}} + w_r U_{\text{phys}}$ with non-maximum suppression to schedule non-overlapping 2D re-acquisition windows.

---

## 5. 3D Axial Interaction-Volume Deconvolution & Tri-Plane Consistency

In correlative 3D tomography (`VolumetricFuser`), a 2D chemical or diffraction map acquired at nominal slice $z_k$ (e.g., polished FIB-SEM-EDS or thick-beam XRD-CT) integrates signal across an axial interaction depth $\sigma_z > 0$ rather than an infinitesimally thin plane. When `depth_psf_sigma > 0`, `VolumetricFuser.fit_sparse_slices` supervises an axial stencil $z_k + \delta$ ($\delta \in \{-R, \dots, R\}$) weighted by a normalized Gaussian depth kernel $w_\delta \propto \exp(-\delta^2 / 2\sigma_z^2)$ (optionally learning $\sigma_z$ jointly via `learn_depth_psf=True`):

$$
\hat{P}_{\text{slab}}(z_k) = \sum_{\delta=-R}^{R} w_\delta(\sigma_z)\, \hat{P}_{\text{HR}}(z_k + \delta), \qquad \mathcal{L}_{\text{3D}} = \mathcal{L}_{\text{phys}}\!\left(\mathcal{D}_\Sigma(\hat{P}_{\text{slab}}(z_k)),\, G_{\text{LR}}(z_k)\right) + \lambda_{\text{tri}} \mathcal{L}_{\text{JSD}}(P_{XY}, P_{XZ}, P_{YZ})
$$

When `triplane_consistency_weight` $\lambda_{\text{tri}} > 0$, randomly sampled unguided interior voxels are regularized via symmetric Jensen–Shannon divergence $\mathcal{L}_{\text{JSD}}$ across orthogonal `XY`, `XZ`, and `YZ` 2.5D slice predictions, suppressing slice-to-slice striping between sparse chemical acquisitions.

