# Benchmarks & Operating Regimes

X-FUSE is evaluated across both synthetic controlled-physics stress tests and real open-access correlative synchrotron, STEM, and SEM datasets. A central principle of X-FUSE is **honest operating-regime characterization**: foundation-model feature fusion is not a silver bullet for every micrograph, and knowing *when* classical edge-guided filtering suffices—or when structural contrast is physically absent—is essential for trustworthy scientific deployment.

---

## 1. Formal Operating Regime Taxonomy

| Regime | Structural vs. Chemical Relationship | Representative Materials Examples | Recommended Method | Quantitative Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **Regime A — Contrast-Complete Piecewise-Smooth** | Distinct phases exhibit clean, monotonic 1D intensity separation in $I_{\text{HR}}$ (`SNR > 10`) | High-$Z$ intermetallics vs. pores/matrix in BSE or attenuation XCT | Classical **Multi-Otsu** or **JBU** (or `XFuse` when conformal UQ / drift alignment is needed) | Otsu, Guided TV, and X-FUSE all reach `>0.97` mIoU (`4×–8×`); classical filters run in `<0.1s` |
| **Regime B1 — Iso-Intensity Multi-Scale / Oriented Texture** | Distinct chemical phases share identical mean grayscale intensity ($\Delta\mu \approx 0$) and pixel variance, differing only in **sub-PSF spatial morphology or lamellar orientation** | Pearlitic / $\alpha+\beta$ Ti lamellar colonies, oriented eutectic precipitates, spinodal textures | **X-FUSE (`cross_attn` + `spatial_mlp`)** | At `16×` Moffat blur, **X-FUSE achieves `0.8908` mIoU (`0.9317` BF1)** vs. `0.7276` Guided TV (`+16.3 pp`), `0.6662` SIREN INR (`+22.5 pp`), `0.6186` JBU (`+27.2 pp`), and `0.3342` Otsu (`+55.7 pp`) |
| **Regime B2 — Heavy-Tailed Probe / Axial 3D Blur & Drift** | Sharp $I_{\text{HR}}$ paired with Lorentzian/Moffat synchrotron wings, SEM backscattered `core_halo` skirts, sub-pixel stage drift, or thick axial 3D interaction volumes | Synchrotron XRD-CT (`Zn13X`), 15–20 kV SEM-EDS (`edxia LC3`), sparse-slice 3D tomography | **X-FUSE (`psf_profile="moffat" | "core_halo"`, `learn_psf=True`, `align_guide=...`, `VolumetricFuser`)** | `+8.8 pp` mIoU from `core_halo` PSF modeling; `+19.1 pp` mIoU from differentiable drift alignment at `4 px` shift; `+6.8 pp` 3D mIoU from learnable axial depth-PSF deconvolution |
| **Regime C — Contrast-Incomplete / Invisible Zonation** | Chemical variation or crystallographic polymorph transition produces **zero contrast** ($\Delta\mu = 0$, $\Delta\text{texture} = 0$) in $I_{\text{HR}}$ | Macroscopic solid-solution diffusion profiles, polymorphs with identical mass attenuation | **X-FUSE (`guide_residual=True`, `debias_positional=True`, `ConformalCalibrator`)** | Smooth coarse chemical zonation is preserved via `guide_residual=True` without ViT positional ramp artifacts (`R^2 -> 0.0`), while `physical_residual` and RAPS prediction sets widen to flag unresolved sub-PSF boundaries |

---

## 2. Registered Open Correlative Datasets (`x_fuse.datasets`)

All open benchmark datasets can be listed via `xfuse datasets` and fetched with MD5 checksum verification via `fetch_open_dataset(key)` or `xfuse datasets --fetch <key>`:

| Dataset Key | Structural Modality ($I_{\text{HR}}$) | Coarse Physical Guide ($G_{\text{LR}}$) | DOI / Source |
| :--- | :--- | :--- | :--- |
| Bundled `Zn13X` | Diamond DIAD Synchrotron XCT (`2560×2560`) | Synchrotron XRD-CT Bragg peak slices (`4` & `9`) | Bundled in `data/samples/Zn13X/` |
| `diad_zn13x_xrdct` | Diamond K11 DIAD Multi-Length XCT (`1.625–2.6 µm`) | Paired `50 µm` (20-slice) & `25 µm` (4-slice) NeXus XRD-CT (`Na, Zn`) | [`10.5281/zenodo.13329639`](https://doi.org/10.5281/zenodo.13329639) |
| `diad_zn13x_powder_xrd` | Synchrotron Powder Reference Standards | Pure-Phase `13X` & `Zn-13X` Powder Diffraction Profiles | [`10.5281/zenodo.13329670`](https://doi.org/10.5281/zenodo.13329670) |
| `ti_rcca_stem_eds` | Nanoscale HAADF-STEM (`Fig3a` 8Al & `Fig3b` 10Al) | 5-Element STEM-EDS (`Ti, Mo, Ta, Cr, Al`) | [`10.5281/zenodo.21770333`](https://doi.org/10.5281/zenodo.21770333) |
| `cuzn_stem_eels` | Simultaneous HAADF-STEM survey | 3D Core-Loss EELS Spectrum Image (`Cu-L2,3`, `Zn-L2,3`, HyperSpy `.hspy`) | [`10.5281/zenodo.21632101`](https://doi.org/10.5281/zenodo.21632101) |
| `edxia_cement_sem_eds` | EPFL `LC3` Cement SEM-BSE (`BSE Wt_.csv`) | 6-Element Calibrated wt% EDS (`Ca, Si, Al, Fe, Mg, S`; Georget et al., 2021) | [`10.5281/zenodo.3862958`](https://doi.org/10.5281/zenodo.3862958) |
| `refractory_alloy_bse_eds_ebsd` | Multiphase Alloy SEM-BSE | Correlative EDS + EBSD phase maps | [`10.5281/zenodo.19182505`](https://doi.org/10.5281/zenodo.19182505) |
| `isis_imat_bragg_edge` | ISIS IMAT White-Beam Neutron Projections | Multi-Metal Rod Phantom (`Al, Fe, Cu, Ni, Zn`) | [`10.5281/zenodo.4273969`](https://doi.org/10.5281/zenodo.4273969) |

*(Note: Bundled `MOR-GIS` slices in `data/samples/MOR-GIS/` pair real XCT zeolite micrographs with semi-synthetic percentile reference maps for megapixel tiling smoke checks and are excluded from real chemical hold-out claims.)*

---

## 3. Multi-Length Synchrotron XCT + XRD-CT Coupling & OME-Zarr Handoff (`Zn-13X`)

Using the open Diamond Light Source K11 DIAD `Zn-13X` dataset (`10.5281/zenodo.13329639`), `benchmarks/run_zn13x_multiscale_couplings.py` validates X-FUSE across real dual-beamsize synchrotron hardware and porous-media simulation handoff:

- **Real Dual-Beamsize `50 µm → 25 µm` Hardware Audit (`4` Paired Axial Slices)**: Correcting the `+47.5°` `gts_cs_theta` soft-motor zero offset on `50 µm` scans (`43345→43324`, `43395→43322`, `43398→43323`, `43399→43325`) and aggregating all fitted Bragg reflections lifts cross-resolution $f_{\text{Zn}}$ Pearson $r$ from **`0.3417`** (`MAE = 0.1818`, legacy single-peak `0°`) $\rightarrow$ **`0.8678`** (`+47.5°` fix) $\rightarrow$ **`0.9291`** (`all_sum` + `+47.5°`) $\rightarrow$ **`0.9343`** (`positive_sum` + `+47.5°`, `MAE = 0.0387`, **`+59.3 pp`**).
- **Chemical `VoDaSuRe` Synthetic-vs.-Real Domain Gap**: Synthetic `2×` Moffat downsampling of the `25 µm` XRD-CT scans yields mean $r = \mathbf{0.9810}$, whereas real `50 µm` dual-beamsize hardware yields $r = \mathbf{0.9343}$ — quantifying a real synchrotron hardware gap of **$\Delta r = 0.0468$** (peaking at `0.1124` on low-Zn reaction-tail slice `43398→43323`).
- **Synchrotron XCT Artifact Cleanup, Fringe-Guarded `pore_mask`, & OME-Zarr Export**: Combining `detrend_radial_cupping` (for segmentation thresholding only), `remove_polar_rings_conservative`, and Fourier Tie-homogeneity phase-fringe damping (`pore_mask="auto_otsu_fringe_guarded"`) improves corrupted pore-mask IoU (`0.6567 → 0.7018`), enforces **`0.0` void chemical leakage** inside detected pores, and exports calibrated OME-Zarr v0.4 multi-phase geometries + `(1-\alpha)` conformal lower/upper phase-envelope labels + uncertainty sidecars via `export_volume_ome_zarr` (with legacy AMReX `.inputs` compatibility via `export_for_openimpala`).
- **`MultimodalBenchmark` Band-Limited Task `T3` (`k <= 0.43 cyc/px`, `~58 µm`) & Two-Track Cupping Audit**: Under raised-cosine Fourier band-limiting to the `25 µm` XRD-CT SB-FRC cutoff (`0.43 cyc/px`), **X-FUSE (`v1.0.0`) achieves reduced $\chi^2_\nu = \mathbf{0.9287}$** (at the `1.0` diffraction noise floor) and **band-limited RMSE = `0.3619` (`r = 0.9965`)** vs. `CNMF` ($\chi^2_\nu = 5.07$, `RMSE = 0.8581`), `Bicubic` ($\chi^2_\nu = 13.78$, `RMSE = 1.4085`), `Guided-TV` ($\chi^2_\nu = 23.41$), and `JBU` ($\chi^2_\nu = 25.43$), while confirming that 7-term radial cupping subtraction is strongly confounded with core–shell `Zn` chemistry ($r(\Delta_{\text{cup}}, \text{Zn}) = +0.5493$) whereas conservative polar de-ringing preserves radial chemical contrast ($r = -0.0857$).

---

## 4. Reproducing Benchmarks & Manuscript LaTeX Tables

```bash
# 1. Run 3-seed referee operating-regime & real Zn13X evaluation + 300-DPI figures
uv run python -m benchmarks.run_referee_evaluation \
  --output-dir benchmarks/outputs/referee_evaluation

# 2. Run 3-seed v1.0.0 architectural, volumetric active-steering & Zenodo EM ablations
uv run python -m benchmarks.run_design_ablations \
  --output-dir benchmarks/outputs/design_ablations

# 3. Run multi-length K11 DIAD Zn-13X XCT + XRD-CT coupling & OpenImpala export study
uv run python -m benchmarks.run_zn13x_multiscale_couplings \
  --output-dir benchmarks/outputs/zn13x_multiscale

# 4. Deterministically export booktabs LaTeX tables (.tex) from JSON results
uv run python -m benchmarks.export_latex_tables \
  --design-json benchmarks/outputs/design_ablations/design_ablations_results.json \
  --referee-json benchmarks/outputs/referee_evaluation/referee_evaluation_results.json \
  --out-dir benchmarks/outputs/tables
```
