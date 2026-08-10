# Project 6 — kappa-measure

Instrument-and-calibration project: when is a geometry measurement resolvable
at all, and what does a practitioner do about it?

- **Report (primary channel):** `report.html` — open directly from disk.
- **API/table contract (frozen, v0.1.0):** `SCHEMA.md`. Owner: A. Plesner.
  Dependent projects (1, 5) pin the schema, not the numbers.
- **Package:** `src/vmf_measure/` — add to `sys.path` to import
  (`import vmf_measure`).
- **Results:** `results/` (frozen-schema CSVs).

## Scripts

```bash
uv run python projects/kappa-measure/scripts/validate_surface.py    # D1 MC validation
uv run python projects/kappa-measure/scripts/audit_unequal_n.py     # scoped audit
uv run python projects/kappa-measure/scripts/make_null_quantiles.py # null table
uv run python projects/kappa-measure/scripts/make_report.py         # render report.html
```

## Tests

```bash
cd projects/kappa-measure/tests && uv run python -m unittest discover -s . -v
```
