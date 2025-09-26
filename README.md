# vMF Sampling

Simple von Mises-Fisher (vMF) distribution sampling with PyTorch and NumPy backends.

## Quick Start

### Generate samples
```bash
# Single experiment
python generate_data.py --dimension 1000 --kappa 500 --num_samples 2000

# With wandb logging
python generate_data.py --dimension 1000 --kappa 500 --wandb --wandb_project my-project

# Using config file  
python generate_data.py --config configs/base.yaml
```

### Parameter sweep with Wandb
```bash
# Create and launch sweep (recommended)
python launch_sweep.py

# Manual wandb sweep setup
wandb sweep wandb_sweep.yaml        # Create sweep, get SWEEP_ID
wandb agent SWEEP_ID                # Run sweep agent

# Launch multiple parallel agents
python launch_sweep.py --agents 4

# Monitor results at https://wandb.ai
```

### Local sweep (deprecated)
```bash
# Local sweep without wandb (not recommended)
python run_sweep_simple.py
```

## Wandb Sweep Features

- **Distributed execution**: Run agents across multiple machines
- **Real-time monitoring**: Live dashboard with metrics and logs
- **Hyperparameter optimization**: Grid search, random search, Bayesian optimization
- **Early stopping**: Automatic termination of poorly performing runs
- **Experiment comparison**: Compare results across different parameter combinations
- **Resume capability**: Automatic handling of failed/interrupted experiments

## Configuration Files

- `configs/base.yaml` - Base experiment configuration (all defaults)
- `wandb_sweep.yaml` - Wandb sweep configuration with your parameter ranges:
  - **Dimensions**: [500, 1000, 5000, 10000, 20000, 100000]  
  - **Kappas**: [100, 500, 1000, 5000, 10000, 20000, 50000, 100000]
  - **Total**: 48 combinations (grid search)

## Files

- `generate_data.py` - Generate vMF samples 
- `run_experiment.py` - Single experiment runner (used by wandb sweep)
- `launch_sweep.py` - Helper script to launch wandb sweeps
- `wandb_sweep.yaml` - Wandb sweep configuration
- `src/vmf_sampler.py` - Core sampling implementation
- `src/config.py` - Configuration management with OmegaConf