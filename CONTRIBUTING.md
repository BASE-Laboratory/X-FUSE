# Contributing to X-FUSE

Thank you for contributing to **X-FUSE**! This guide summarizes the local development setup, quality checks, and conventions for adding new correlative imaging modalities or fusion methods.

## 1. Development Environment Setup

We use [`uv`](https://docs.astral.sh/uv/) for fast, reproducible dependency management:

```bash
git clone https://github.com/BASE-Laboratory/X-FUSE.git
cd X-FUSE
uv sync --extra dev
uv run pre-commit install
```

## 2. Running Linting, Type-Checking, and Tests

Use the root `Makefile` (or invoke `uv run` directly):

```bash
make format          # Auto-format code and sort imports with Ruff
make lint            # Check Ruff lint & formatting rules
make typecheck       # Run static type checking on x_fuse/ with Mypy
make test            # Run CPU unit tests with >=85% coverage gate
make benchmark-quick # Run offline synthetic & sample smoke suite
make build           # Build wheel + sdist packages
```

If you are working on an NVIDIA GPU workstation or HPC node, also run:

```bash
make test-gpu
```

## 3. Notebook Hygiene

Tutorial notebooks live in `notebooks/` and **must be committed with all cell outputs stripped** (enforced automatically by `nbstripout` in `pre-commit` and verified by CI):

```bash
uv run nbstripout notebooks/*.ipynb
```

 Avoid hardcoding cluster-specific paths (such as `/dls/...` or `/scratch/...`); use `x_fuse.get_sample_dataset_path()` or `x_fuse.fetch_open_dataset()` instead.

## 4. Adding a New Correlative Modality Loader

1. Add a self-contained loader function in `x_fuse/loaders.py` returning float32 NumPy arrays with documented axis ordering (`(H, W)`, `(M, H, W)`, or `(Z, Y, X)`).
2. Export the loader in `x_fuse/__init__.py`.
3. Add deterministic unit tests using temporary synthetic files (`tmp_path`) in `tests/test_loaders.py`.
4. If an open-access benchmark exists on Zenodo, register its record ID and DOI in `OPEN_DATASETS` inside `x_fuse/datasets.py`.
