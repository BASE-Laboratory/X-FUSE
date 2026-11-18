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
2. **Local HR-Query Cross-Attention Upsampling (`interpolation_mode="cross_attn"`, `CrossAttentionFeatureUpsampler`)**: Inspired by continuous coordinate-query upsamplers (`JAFAR` / `AnyUp`), each pixel $p \in \{1, \dots, H\} \times \{1, \dots, W\}$ extracts an 8-channel multiscale structural & texture descriptor $\tau(p)$ (capturing $3\times 3$ and $7\times 7$ local standard deviations, Sobel gradient magnitude, and Laplacian response) and attends over its $3 \times 3$ neighborhood of coarse ViT tokens with sub-token relative coordinate offsets $\Delta_{pq}$. Unlike scalar intensity JBU, `cross_attn` cleanly separates **iso-intensity phases** (e.g., fine eutectic/precipitate lamellae sharing identical mean grayscale with an adjacent homogeneous matrix).

### Apositional Feature Debiasing (`debias_positional=True`)

Pretrained Vision Transformers with learned absolute positional embeddings or RoPE encode linear and quadratic $(y, x)$ coordinate ramps in dedicated feature channels (`dino-saw`, Docherty et al., 2026). On materials exhibiting macroscopic chemical gradients (diffusion couples, battery electrodes, thermal-barrier coatings), a probe can overfit to image coordinates $(y, x)$ instead of microstructure. `ApositionalFeatureDebiaser` projects each feature channel onto the orthogonal complement of normalized degree-$d$ bivariate Legendre polynomials $\mathcal{B}_d(y, x)$, driving coordinate $R^2 \to 0$ on *any* standard ViT without custom re-pretraining.

---

## 2. Quantitative Forward Physics, Spectral Unmixing & B-Spline Registration

Naive correlative fusion upsamples the coarse chemical guide $G_{\text{LR}}$ to $(H, W)$ and trains against blurry targets—actively penalizing crisp high-resolution phase boundaries.

Instead, X-FUSE predicts high-resolution phase abundances $\hat{P}_{\text{HR}} \in [0, 1]^{M \times H \times W}$ via a `LinearProbeHead`, nonlinear `SpatialConvMLPHead` (`fusion_method="spatial_mlp"`), or $O(HW)$ `WindowedGuideCrossAttentionHead` (`fusion_method="attention"`), warps them into the guide frame via $\mathcal{W}_\theta$ (`DifferentiableGuideAligner`), and applies an anisotropic forward PSF operator $\mathcal{D}_\Sigma$ (`ForwardPSFDownsampler`):

$$
\hat{G}_{\text{LR}} = \mathcal{D}_\Sigma\!\left(\mathcal{W}_\theta(\hat{P}_{\text{HR}})\right) = \mathrm{AvgPool}_{r_y, r_x}\!\left(K_{\sigma_y^{(m)}, \sigma_x^{(m)}} * \mathcal{W}_\theta(\hat{P}_{\text{HR}})_m\right)
$$

`PhysicsForwardModel` supports quantitative counting-statistics objectives:

- **Poisson Shot-Noise Deviance (`loss_fn="forward_psf_poisson"`)**: Exact maximum-likelihood negative log-likelihood for low-dose STEM-EDS, EELS, XRF, and time-of-flight Bragg-edge photon/neutron counts $C_{\text{LR}} \sim \mathrm{Poisson}(s_m \hat{G}_{\text{LR}, m} + b_m)$ with learnable per-channel gain $s_m > 0$ and background $b_m \ge 0$:
  $$
  \mathcal{L}_{\text{Poisson}}(\hat{\Lambda}, C) = 2 \sum_{m, u} \left[\hat{\Lambda}_{m,u} - C_{m,u} + C_{m,u} \log\!\left(\frac{\max(C_{m,u}, \epsilon)}{\hat{\Lambda}_{m,u}}\right)\right]
  $$
- **Simplex Mass/Atomic-Fraction Conservation (`loss_fn="compositional_kl"`)**: Enforces $\hat{P}_{\text{HR}} = \mathrm{softmax}(Z_{\text{HR}}) \in \Delta^{K-1}$ ($\sum_{k=1}^K \hat{P}_{k}(x) = 1$ at every pixel) supervised via forward-PSF Kullback–Leibler divergence.
- **3D Spectral / Diffraction Cube Unmixing (`SpectralUnmixingDictionary`, `spectral_unmix=True`)**: Accepts raw $(E, H_{\text{LR}}, W_{\text{LR}})$ EDS/EELS/XRD cubes directly without external pre-fitting, jointly learning non-negative endmember spectra $S \in \mathbb{R}_{\ge 0}^{M \times E}$ ($\sum_e S_{m,e} = 1$, initialized via Successive Projection Algorithm + multiplicative NMF) alongside super-resolved spatial abundances: $\hat{Y}_{e, u} = \sum_{m=1}^M \alpha_m \hat{G}_{\text{LR}, m, u} S_{m, e} + b_e$.
- **Cubic B-Spline Non-Rigid Registration (`align_guide="bspline"`)**: Compensates nonlinear SEM/STEM charging drift and scan-coil hysteresis via a bicubic control-point displacement field regularized by thin-plate bending energy $\|\nabla^2 u\|_2^2$ and monitored via minimum Jacobian determinant $\min \det J_{\mathcal{W}}$ for diffeomorphic invertibility.

---

## 3. Prior Calibration, Tri-Part Uncertainty & Split Conformal Sets

After fitting, X-FUSE calibrates high-resolution posteriors via `match_coarse_priors` so spatial means match physical coarse phase fractions, and decomposes pixel/voxel uncertainty into:

1. **Aleatoric / Chemical Ambiguity ($U_{\text{alea}}$)**: Normalized Shannon entropy $-\frac{1}{\log K} \sum_{k=1}^{K} p_k(x) \log p_k(x) \in [0, 1]$.
2. **Epistemic / View Disagreement ($U_{\text{epi}}$)**: Computed in $O(1)$ auxiliary memory via an online **Welford accumulator** across invertible sub-patch TTA views (and orthogonal `XY`, `XZ`, `YZ` planes in 2.5D mode).
3. **Physical Forward-Model Residual ($U_{\text{phys}}$)**: Upsampled $|\mathcal{D}_\Sigma(\mathcal{W}_\theta(\hat{P}_{\text{HR}})) - G_{\text{LR}}|$, automatically flagging **structurally invisible phases** carrying zero contrast in $I_{\text{HR}}$.

### Finite-Sample Split Conformal Prediction (`ConformalCalibrator`)

When `conformal_alpha` $\alpha \in (0, 1)$ is specified (or calibrated against sparse reference labels via `ConformalCalibrator.calibrate_labels`), X-FUSE computes finite-sample order-statistic thresholds $\hat{q}_{1-\alpha} = s_{(\lceil (N_{\text{cal}}+1)(1-\alpha) \rceil)}$ using Least Ambiguous Set-Valued Classifiers (`"lac"`) or Adaptive Prediction Sets (`"aps"`), returning boolean `prediction_sets` $\mathcal{C}_{1-\alpha}(x) \subseteq \{1, \dots, K\}$ with distribution-free marginal coverage $\mathbb{P}(Y_{\text{test}} \in \mathcal{C}_{1-\alpha}(X_{\text{test}})) \ge 1 - \alpha$ under exchangeability, alongside normalized coarse residual intervals `conformal_interval_halfwidth`.

---

## 4. Active Sparse-Slice & ROI Acquisition Utility

During synchrotron or electron-microscopy beamtime, `suggest_next_slices` ranks candidate unmeasured $z$-slices in a 3D tomogram using a greedy diversity-penalized acquisition score:

$$
\mathcal{A}(z \mid \mathcal{S}) = w_{\text{epi}} \tilde{U}_{\text{epi}}(z) + w_{\text{alea}} \tilde{U}_{\text{alea}}(z) + w_{\text{dist}} \widetilde{\min_{s \in \mathcal{S}} |z - s|}
$$

subject to a minimum slice spacing $\Delta z_{\min}$, preventing redundant adjacent scans while targeting slices with maximal orthogonal-plane disagreement and phase boundary ambiguity. Similarly, `suggest_2d_rois` evaluates exact $O(HW)$ integral-image sliding-window sums over $(1 - w_r) U_{\text{comb}} + w_r U_{\text{phys}}$ with non-maximum suppression to schedule non-overlapping 2D re-acquisition windows.
