---
name: vmf-cluster
description: Syncs this vMF-sampling repository to tik42x net_scratch, submits and monitors the canonical CPU-only sampling benchmark arrays on arton10 or tikgpu10, and fetches completed sampling data. Use for the established sampler throughput sweep; use vmf-kappa and separate Slurm jobs for CPU/CUDA kappa-estimator benchmarks.
compatibility: Requires ssh and rsync locally, SSH host alias tik42x, and Slurm plus uv on tik42x.
---

# vMF sampling cluster workflow

This skill covers the canonical CPU-only **sampling** sweep. It does not define
the newer kappa-estimation CPU/CUDA benchmark. For that work, load
`/skill:vmf-kappa` and create separate Slurm files so the archived sampling
workflow remains reproducible.

Run all commands from the repository root. Never put environments or benchmark
outputs in `/home/aplesner`; use `/itet-stor/aplesner/net_scratch`.

## Connectivity

```bash
ssh -o BatchMode=yes tik42x 'hostname && whoami'
```

The defaults can be overridden with `VMF_CLUSTER_HOST`, `VMF_CLUSTER_USER`, and
`VMF_REMOTE_ROOT`.

## Sync

Preview or perform the one-way source sync:

```bash
scripts/cluster_sync.sh --dry-run
scripts/cluster_sync.sh
```

The destination is
`/itet-stor/aplesner/net_scratch/vmf-sampling/repo`. Generated measurements,
local environments, and Git metadata are excluded. The sync uses `--delete`, so
do not keep untracked work in the remote repo.

## Submit

```bash
scripts/cluster_submit.sh --target arton10
scripts/cluster_submit.sh --target tikgpu10
# To submit code that was already synced:
scripts/cluster_submit.sh --target tikgpu10 --no-sync
```

Record the printed `RUN_ID` and `JOB_ID`. Both targets reserve 12 CPUs and run
only one benchmark task at a time. A complete sweep is 39 tasks: 13 dimensions
across the NumPy, SciPy, and PyTorch backend groups. `tikgpu10` uses the
`disco.all.phd` partition but deliberately requests no GPU; never add `--gres`
for the CPU comparison.

## Monitor and diagnose

```bash
scripts/cluster_status.sh JOB_ID
ssh tik42x 'tail -100 /itet-stor/aplesner/net_scratch/vmf-sampling/slurm/benchmark_JOBID_TASK.log'
```

If a task fails, inspect its log and `sacct` state before resubmitting. Never
silently combine partial runs or CSVs from different run IDs.

## Fetch and render the post

After all 39 tasks complete:

```bash
scripts/cluster_fetch.sh RUN_ID
uv run python scripts/make_blog.py \
  --run measurements/runs/TIKGPU10_RUN_ID \
  --comparison-run measurements/runs/ARTON10_RUN_ID
```

Confirm that the fetched run contains 39 `benchmark_*.csv` files and 273 rows.
The renderer writes publication figures and tables under `docs/`. The published
HTML lives only in the neighboring website repository. Reader-facing output
must use CPU models, not cluster hostnames or run IDs; the blog skill contains
the full editorial and synchronization rules.

## Archived complete runs

Do not overwrite these canonical data sets:

- `20260723T094321Z-tikgpu10`: newer AMD EPYC 7742 primary blog data
- `20260723T160232Z-arton10`: older Intel Xeon E5-2690 v2 generalization data

Use `/skill:vmf-blog` after fetching or selecting benchmark data.
