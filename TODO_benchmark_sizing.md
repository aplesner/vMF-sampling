# Benchmark re-measurement (post sizing fix)

**Status: code changes done (2026-07-13) — re-measurement pending.**

All items from the original TODO are implemented:

- **Calibrated per-call sizes** — `benchmark.py` now calibrates `num_samples`
  per (backend, dtype, dim) so one `sample()` call takes ~2 s
  (`--target-call-time`), capped at 1.2 GB of output per call
  (`--mem-budget-gb`), with a 5 s repeat-to-fill timing window as before.
  The calibrated n is recorded in the CSV `num_samples` column (throughput =
  `num_samples / time_s` is size-independent at steady state, so per-backend
  sizes still compare apples-to-apples). `--num-samples 5000` reproduces the
  old fixed-size behavior.
- **dim=2 / dim=3 closed forms** — all four samplers (`numpy`, `numpy_hh`,
  `torch`, `torch_hh`) now dispatch to a von Mises path at dim 2 and the
  inverse-CDF path at dim 3, like scipy. Measured locally: dim 2 at parity
  with scipy (~13.6 M/s both), dim 3 within ~10% (18.5 vs 20.7 M/s).
- **Single Householder reflection** — the hh backends drop the second flip;
  `u = (e1 − mu)/‖e1 − mu‖` maps e1 → mu directly (det = −1 is irrelevant for
  vMF; the antipodal special case is gone). Also fixed a pre-existing crash:
  tuple-shaped `num_samples` reshaped before rotation.
- **Report** — now a standalone `docs/benchmark_report.html` (self-contained,
  blog-ready) with the method text updated and an honest "Updates since these
  measurements" section. Its numbers still come from the old methodology and
  must be refreshed after the re-run.

## How to re-run (all backends, including scipy — hardware may have changed)

1. On the cluster, start from a **clean output dir** — `benchmark.py` appends
   to existing CSVs, which would mix methodologies:

   ```bash
   rm -f /itet-stor/$USER/net_scratch/jobs/vmf-sampling/benchmarks/benchmark_*.csv
   sbatch scripts/benchmark.slurm   # 42 array tasks = 14 dims (incl. dim 3) × {numpy, scipy, torch}
   ```

2. Copy the per-dim CSVs over `measurements/benchmark_*_dim*.csv`, then merge:

   ```bash
   uv run python scripts/merge_benchmarks.py
   ```

3. Re-run `benchmark_analysis.ipynb` and refresh the numbers/figures in
   `docs/benchmark_report.html`.

Notes:

- Each row now takes roughly 8–12 s (calibration probe + 5 s window), so a
  full sweep is slower than the old fixed-5000 runs but still well inside the
  2 h slurm limit per (backend, dim) task.
- Expect low/mid-dim throughputs to rise ~1.6–3× (dim ≤ 64) purely from the
  sizing fix; high-dim numbers and the hh-vs-QR ratios should be stable.
