# Installation Guide

`x-fuse` (`v0.2.0+`) is a **100% self-contained pure-PyTorch package** with no external sub-repositories or custom C++/CUDA extensions to compile. It runs cleanly on Linux, macOS (Intel & Apple Silicon), and Windows.

## Prerequisites

- **Python 3.10+** (`3.10`, `3.11`, or `3.12`)
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** (recommended) or standard `pip`

---

## Quick Installation with `uv`

### 1. Linux / macOS (CPU)

```bash
git clone https://github.com/BASE-Laboratory/X-FUSE.git
cd X-FUSE

uv sync --extra cpu --extra dev --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu
```

### 2. Linux (CUDA 12.1 GPU)

```bash
git clone https://github.com/BASE-Laboratory/X-FUSE.git
cd X-FUSE

uv sync --extra cuda --extra dev
```

### 3. Standard `pip` Installation

```bash
pip install -e ".[dev,synthetic]"
```

---

## Optional Dependency Extras

| Extra | Command (`uv sync ...`) | Included Packages / Purpose |
| :--- | :--- | :--- |
| `io` | `--extra io` | `tifffile`, `zarr`, `rosettasciio` for OME-TIFF/Zarr export & vendor `.emd`/`.dm4`/`.bcf`/`.hspy` readers |
| `dev` | `--extra dev` | `pytest`, `pytest-cov`, `pytest-benchmark`, `ruff`, `mypy`, `nbformat`, `nbstripout`, `pre-commit` |
| `docs` | `--extra docs` | `mkdocs`, `mkdocs-material`, `mkdocstrings[python]` for building documentation |
| `synthetic` | `--extra synthetic` | `porespy` for classical PoreSpy blob/sphere generators |
| `sam` | `--extra sam` | `sam2` for certainty-guided SAM2 boundary refinement (`refine_with_sam2`) |
| `notebooks` | `--extra notebooks` | `jupyterlab`, `ipykernel`, `ipywidgets` for interactive notebooks |
| `cuda` | `--extra cuda` | CUDA-enabled PyTorch & `xformers` on Linux x86_64 |
| `cpu` | `--extra cpu` | Explicit CPU wheel marker |

---

## Verify Your Setup

```bash
# 1. Verify package import & version
uv run python -c "import x_fuse; print('x-fuse version:', x_fuse.__version__)"

# 2. Run the CPU unit test suite with >=82% coverage gate
uv run pytest -m "not gpu" --cov=x_fuse --cov-fail-under=82 -v

# 3. Run the quick synthetic & paper ablation benchmarks
uv run python -m benchmarks.run_synthetic_suite --quick
uv run python -m benchmarks.run_paper_benchmarks --quick --output-dir /tmp/xfuse_paper_quick

# 4. Check Ruff formatting/lint, Mypy types, and strict MkDocs build
uv run ruff check x_fuse tests benchmarks
uv run mypy x_fuse
NO_MKDOCS_2_WARNING=1 uv run mkdocs build --strict
```
