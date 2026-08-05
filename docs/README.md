# vMF sampling write-up

- [`assets/`](assets/) — source publication figures in SVG and high-resolution PNG
- [`tables/`](tables/) — machine-readable summary and headline tables
- [`benchmark_report.html`](benchmark_report.html) — older standalone report, retained for reference

The published post has moved to
`../aplesner.github.io/blog/2026-07-23-faster-vmf-sampling/`; it is not retained
as a second HTML copy in this repository. Regenerate its source figures and
tables after fetching a cluster run:

```bash
scripts/cluster_submit.sh --target tikgpu10
scripts/cluster_fetch.sh RUN_ID
uv run python scripts/make_blog.py \
  --run measurements/runs/20260723T094321Z-tikgpu10 \
  --comparison-run measurements/runs/20260723T160232Z-arton10
```
