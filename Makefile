.PHONY: help sync format lint typecheck test test-gpu benchmark-quick build clean

help: ## Show available developer targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

sync: ## Install package and developer extras via uv
	uv sync --extra dev

format: ## Format code and sort imports with Ruff
	uv run ruff format x_fuse tests benchmarks
	uv run ruff check --fix x_fuse tests benchmarks

lint: ## Check formatting, lint rules, and stripped notebooks
	uv run ruff check x_fuse tests benchmarks
	uv run ruff format --check x_fuse tests benchmarks

typecheck: ## Run static type checking with Mypy
	uv run mypy x_fuse

test: ## Run CPU unit test suite with coverage enforcement
	uv run pytest -m "not gpu" --cov=x_fuse --cov-report=term-missing --cov-fail-under=80 -v

test-gpu: ## Run CUDA GPU integration tests (requires NVIDIA GPU)
	uv run pytest -m gpu -v

benchmark-quick: ## Run offline synthetic & sample smoke benchmark
	uv run python -m benchmarks.run_synthetic_suite --quick

build: ## Build wheel and sdist distributions
	uv build

clean: ## Remove build, test, and linter caches
	rm -rf dist build *.egg-info .pytest_cache .benchmarks .ruff_cache .mypy_cache .coverage coverage.xml htmlcov benchmarks/outputs
