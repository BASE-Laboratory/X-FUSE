## Summary
- 

## Verification Checklist
- [ ] `make lint` (`ruff check` + `ruff format --check`) passes
- [ ] `make typecheck` (`mypy x_fuse`) passes
- [ ] `make test` (`pytest -m "not gpu"` with `>=85%` coverage) passes
- [ ] Any modified notebooks in `notebooks/` have outputs stripped (`nbstripout`)
