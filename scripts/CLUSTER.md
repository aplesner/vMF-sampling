# tik42x benchmark workflow

Cluster data lives under
`/itet-stor/aplesner/net_scratch/vmf-sampling`; the small backed-up home is not
used for environments or outputs.

```bash
scripts/cluster_sync.sh --dry-run  # preview one-way rsync
scripts/cluster_submit.sh --target arton10   # original CPU run
scripts/cluster_submit.sh --target tikgpu10  # CPU-only run; requests no GPU
scripts/cluster_status.sh JOB_ID             # squeue + sacct
scripts/cluster_fetch.sh RUN_ID             # retrieve CSVs and environment records
uv run python scripts/make_blog.py \
  --run measurements/runs/TIKGPU10_RUN_ID \
  --comparison-run measurements/runs/ARTON10_RUN_ID
```

Environment overrides: `VMF_CLUSTER_HOST`, `VMF_CLUSTER_USER`,
`VMF_REMOTE_ROOT`, `VMF_RUN_ID`, and `VMF_BENCH_TARGET`. Both supported targets
use 12 CPUs and serialize the array (`%1`) so timings are not contaminated by
another benchmark task from the same run. The `tikgpu10` target selects its GPU
partition but requests no GPU resources.

The sync destination is disposable and uses `rsync --delete`; edit locally,
not in the remote copy.
