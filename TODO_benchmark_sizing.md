# Benchmark re-measurement (post sizing fix)

**Status: optimized PyTorch CPU re-measurements completed on both CPUs (2026-07-23).**

All items from the original TODO are implemented:

- **Calibrated per-call sizes** — `benchmark.py` now calibrates `num_samples`
  per (backend, dtype, dim) so one `sample()` call takes ~2 s
  (`--target-call-time`), capped at 1.2 GB of output per call
  (`--mem-budget-gb`), with a 5 s repeat-to-fill timing window as before.
  The calibrated n is recorded in the CSV `num_samples` column (throughput =
  `num_samples / time_s` is size-independent at steady state, so per-backend
  sizes still compare apples-to-apples). `--num-samples 5000` reproduces the
  old fixed-size behavior.
- **dim=2 / dim=3 closed forms** — all five samplers (`numpy`, `numpy_hh`,
  `torch`, `torch_hh`, and `scipy`) now dispatch to a von Mises path at dim 2
  and the inverse-CDF path at dim 3. `ScipyvMF` previously lacked the special
  cases that real `scipy.stats.vonmises_fisher` ships (`_rvs_2d`/`_rvs_3d`),
  so its dim-2/3 numbers understated scipy; it now mirrors them faithfully
  (dim 2: 13.6 M/s, matching real scipy; dim 3: 17.4 vs 20.7 M/s). The repo
  backends measure at parity: numpy 13.4/18–21 M/s, torch f32 ~33 M/s at dim 3.
- **PyTorch CPU path** — sample batches stay row-major for dense multiplication,
  direction normalization and radial scaling are in-place, and the Householder
  rank-one update uses fused `addmm` rather than allocating an outer-product
  tensor. On the newer CPU this improves PyTorch QR by 5.18× at dimension 2048
  and PyTorch Householder by roughly 4–6% at larger dimensions.
- **Single Householder reflection** — the hh backends drop the second flip;
  `u = (e1 − mu)/‖e1 − mu‖` maps e1 → mu directly (det = −1 is irrelevant for
  vMF; the antipodal special case is gone). Also fixed a pre-existing crash:
  tuple-shaped `num_samples` reshaped before rotation.
- **Report** — the published HTML has moved to
  `../aplesner.github.io/blog/2026-07-23-faster-vmf-sampling/`; source SVG/PNG
  figures under `docs/assets/` use the newer AMD EPYC run and the older Intel
  Xeon run as a hardware-generalization appendix. Raw data lives in
  `measurements/runs/{20260723T094321Z-tikgpu10,20260723T160232Z-arton10}`.
  `docs/benchmark_report.html` is retained as the older standalone report.

## How to re-run

Use the project cluster helpers; they isolate every run by ID, so CSVs from
separate methodologies cannot be mixed accidentally:

```bash
scripts/cluster_submit.sh
scripts/cluster_status.sh JOB_ID
scripts/cluster_fetch.sh RUN_ID
uv run python scripts/make_blog.py --run measurements/runs/RUN_ID
```

The current Slurm array has 39 tasks: 13 dimensions (including dim 3) times
`{numpy, scipy, torch}`. It measures float64 with three seeds and a 2 s timing
window. Both `arton10` and CPU-only `tikgpu10` targets are supported; see
`scripts/CLUSTER.md` for paths and overrides.
