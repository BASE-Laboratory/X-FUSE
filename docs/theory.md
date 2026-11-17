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

### Scale-Dilated Joint Bilateral Upsampling (JBU)

Rather than nearest-neighbor or bilinear upsampling from the stride-$s$ token grid $(H_f, W_f)$ to $(H, W)$, X-FUSE applies pure-PyTorch progressive **scale-dilated Joint Bilateral Upsampling** conditioned on the normalized structural intensity $g = I_{\text{HR}}$:

$$
w_{\text{JBU}}(p, q) = \exp\!\left(-\frac{\|p - q\|_2^2}{2\sigma_s^2}\right) \exp\!\left(-\frac{\|g(p) - g(q)\|_2^2}{2\sigma_r^2}\right)
$$

Across $\log_2(H / H_f)$ dyadic stages with dilation proportional to the remaining upsampling factor, token features snap cleanly onto sub-pixel structural boundaries without block artifacts.

---

## 2. Forward-PSF Partial-Volume Supervision & Differentiable Registration

Naive correlative fusion upsamples the coarse chemical guide $G_{\text{LR}} \in [0, 1]^{M \times H_{\text{LR}} \times W_{\text{LR}}}$ to $(H, W)$ and trains against blurry targets—actively penalizing crisp high-resolution phase boundaries.

Instead, X-FUSE predicts high-resolution phase probabilities $\hat{P}_{\text{HR}} = \sigma(Z_{\text{HR}}) \in [0, 1]^{M \times H \times W}$ first, warps them into the guide's coordinate frame via a bounded differentiable spatial transformer $\mathcal{W}_\theta$ (`DifferentiableGuideAligner`), and applies a per-channel anisotropic forward point-spread-function operator $\mathcal{D}_\Sigma$ (`ForwardPSFDownsampler`):

$$
\hat{G}_{\text{LR}} = \mathcal{D}_\Sigma\!\left(\mathcal{W}_\theta(\hat{P}_{\text{HR}})\right) = \mathrm{AvgPool}_{r_y, r_x}\!\left(K_{\sigma_y^{(m)}, \sigma_x^{(m)}} * \mathcal{W}_\theta(\hat{P}_{\text{HR}})_m\right)
$$

where:

- $\mathcal{W}_\theta$ parameterizes sub-pixel translation, 2D rigid motion $(\theta, t_x, t_y)$, or unconstrained affine warps in the canonical $I_{\text{HR}}$ frame (ensuring exact equivariance across invertible TTA flips, shifts, and rotations), bounded via $\tanh$ to prevent degenerate drift.
- $K_{\sigma_y^{(m)}, \sigma_x^{(m)}}$ is a separable 2D Gaussian PSF kernel with optionally learnable per-channel log-widths (`learn_psf=True`), modeling element-dependent X-ray interaction volumes (e.g., high-energy Fe-K$\alpha$ vs. soft Si-K$\alpha$) or beam astigmatism.
- Training minimizes binary cross-entropy or Huber loss directly at coarse measurement resolution plus weak $L_2$ warp regularization:

$$
\mathcal{L}(\phi, \theta, \Sigma) = \mathcal{L}_{\text{phys}}\!\left(\hat{G}_{\text{LR}}, G_{\text{LR}}\right) + \lambda_{\text{reg}} \|\theta - \theta_{\text{id}}\|_2^2
$$

---

## 3. Prior Calibration & Tri-Part Uncertainty Decomposition

After probe fitting, X-FUSE calibrates high-resolution class posteriors via `match_coarse_priors`, adjusting channel log-odds biases $b \in \mathbb{R}^K$ so the spatial mean of $\hat{P}_{\text{HR}}$ matches physical coarse phase fractions in $G_{\text{LR}}$.

Every pixel (2D) or voxel (3D) receives three complementary uncertainty maps:

1. **Aleatoric / Chemical Ambiguity ($U_{\text{alea}}$)**: Normalized Shannon entropy across $K$ phases:
   $$
   U_{\text{alea}}(x) = -\frac{1}{\log K} \sum_{k=1}^{K} p_k(x) \log p_k(x) \in [0, 1]
   $$
2. **Epistemic / View Disagreement ($U_{\text{epi}}$)**: Computed in $O(1)$ auxiliary memory via an online **Welford accumulator** across invertible sub-patch TTA views $t \in \{1, \dots, T\}$ (and across orthogonal `XY`, `XZ`, `YZ` planes in 2.5D volumetric mode).
3. **Physical Forward-Model Residual ($U_{\text{phys}}$)**: Coarse discrepancy projected onto the HR grid:
   $$
   U_{\text{phys}} = \mathrm{Upsample}\!\left(\frac{1}{M}\sum_{m=1}^{M} \left|\mathcal{D}_\Sigma(\mathcal{W}_\theta(\hat{P}_{\text{HR}}))_m - G_{\text{LR}, m}\right|\right)
   $$
   High $U_{\text{phys}}$ automatically flags **structurally invisible phases**—regions where $G_{\text{LR}}$ detects a real chemical or crystallographic change that carries zero contrast in $I_{\text{HR}}$.

---

## 4. Active Sparse-Slice & ROI Acquisition Utility

During synchrotron or electron-microscopy beamtime, `suggest_next_slices` ranks candidate unmeasured $z$-slices in a 3D tomogram using a greedy diversity-penalized acquisition score:

$$
\mathcal{A}(z \mid \mathcal{S}) = w_{\text{epi}} \tilde{U}_{\text{epi}}(z) + w_{\text{alea}} \tilde{U}_{\text{alea}}(z) + w_{\text{dist}} \widetilde{\min_{s \in \mathcal{S}} |z - s|}
$$

subject to a minimum slice spacing $\Delta z_{\min}$, preventing redundant adjacent scans while targeting slices with maximal orthogonal-plane disagreement and phase boundary ambiguity. Similarly, `suggest_2d_rois` evaluates exact $O(HW)$ integral-image sliding-window sums over $(1 - w_r) U_{\text{comb}} + w_r U_{\text{phys}}$ with non-maximum suppression to schedule non-overlapping 2D re-acquisition windows.
