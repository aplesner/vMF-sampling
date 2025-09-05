# W&B Sweep Instructions

This document explains how to run the vMF sampling sweep using Weights & Biases.

## Prerequisites

1. Install wandb: `pip install wandb`
2. Login to wandb: `wandb login`
3. Make sure you have the required dependencies (torch, numpy, scipy, pydantic, pyyaml)
4. Ensure `python3` is available in your PATH

## Running the Sweep

### 1. Initialize the Sweep

```bash
python3 -m wandb sweep wandb_sweep.yaml
```

This will create a sweep and return a sweep ID like `your-entity/vmf-sampling/sweep-id`

### 2. Run Sweep Agents

The sweep now uses a function-based approach that avoids Python interpreter issues:

```bash
# Run a single agent
python3 -m wandb agent your-entity/vmf-sampling/sweep-id

# Run multiple agents in parallel (recommended)
python3 -m wandb agent your-entity/vmf-sampling/sweep-id &
python3 -m wandb agent your-entity/vmf-sampling/sweep-id &
python3 -m wandb agent your-entity/vmf-sampling/sweep-id &
```

Each agent will call `python3 run_sweep.py` directly, avoiding the `/usr/bin/env python` issue.

### 3. Monitor Results

Visit your W&B dashboard to monitor the sweep progress:
- Performance comparisons across implementations
- Timing analysis by dimension
- Device performance (CPU vs GPU for torch)

## Sweep Configuration

The sweep tests:
- **Dimensions**: 3, 16, 64, 256, 512, 1024
- **Implementations**: numpy, scipy, torch
- **Devices**: auto, cpu, cuda (torch only)
- **Kappa values**: 1.0, 5.0, 10.0, 20.0
- **Fixed samples**: 2000 per run

## Expected Results

- Total combinations: 6 dims × 3 impls × 3 devices × 4 kappa = 216 runs
- Invalid combinations (non-torch with cuda) are automatically skipped
- Metric optimization: minimizing `benchmark/mean_time`

## Individual Config Testing

You can test individual configurations using the main experiment script:

```bash
# Test with base config (loads configs/base.yaml by default)
python run_experiment.py

# Test specific YAML configs
python run_experiment.py --config configs/numpy.yaml
python run_experiment.py --config configs/torch_cpu.yaml
python run_experiment.py --config configs/torch_gpu.yaml
```

## Available Configuration Files

All configs are now in YAML format in the `configs/` directory:

- `configs/base.yaml` - Default configuration (dim=3, numpy)
- `configs/numpy.yaml` - NumPy implementation (dim=1024)
- `configs/scipy.yaml` - SciPy implementation (dim=1024)
- `configs/torch.yaml` - PyTorch implementation (dim=1024, auto device)
- `configs/torch_cpu.yaml` - PyTorch CPU-specific (dim=512)
- `configs/torch_gpu.yaml` - PyTorch GPU-specific (dim=512)

## Profiling Individual Runs

For detailed profiling of specific configurations:

```bash
python profile_experiment.py configs/torch_gpu.yaml --output-file results/profile_torch_gpu.json
```

## Sweep Script Structure

The `run_sweep.py` script now uses a function-based approach:
1. `wandb.agent()` calls the `main()` function for each sweep run  
2. Loads the base configuration from `configs/base.yaml`
3. Applies parameter overrides from `wandb.config` 
4. Filters invalid combinations (e.g., numpy + cuda)
5. Runs benchmarks and logs results to W&B
6. Returns the mean execution time for sweep optimization

This approach eliminates the Python interpreter issues by running everything within the same Python process.