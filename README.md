# vMF Sampling Suite

A comprehensive von Mises-Fisher (vMF) distribution sampling framework with advanced benchmarking, profiling, and logging capabilities.

## Features

- **Multi-backend Support**: NumPy, PyTorch, and SciPy implementations
- **Advanced Benchmarking**: Using `torch.utils.benchmark` for accurate timing
- **Detailed Profiling**: Line-by-line performance analysis with time breakdown
- **Weights & Biases Integration**: Seamless experiment tracking with wandb
- **CSV Logging**: Structured data export for analysis
- **Modular Architecture**: Clean separation of concerns across multiple modules

## Project Structure

```
vMF-sampling/
├── src/                          # Source code modules
│   ├── __init__.py              # Package initialization
│   ├── vmf_sampler.py           # Core vMF sampling implementation
│   ├── config.py                # Configuration management
│   ├── benchmark.py             # Benchmarking utilities
│   ├── profiler.py              # Enhanced profiling with detailed analysis
│   └── logger.py                # CSV and wandb logging
├── configs/                      # Configuration files
│   ├── config_numpy.json       # NumPy implementation config
│   ├── config_torch.json       # PyTorch implementation config
│   ├── config_scipy.json       # SciPy implementation config
│   ├── config_benchmark.json   # High-performance benchmark config
│   └── config_quick_test.json   # Quick test configuration
├── results/                      # Output directory for results
├── run_experiment.py            # Main execution script
├── .env.template               # Environment variables template
└── .gitignore                  # Git ignore file
```

## Quick Start

### 1. Setup Environment

Copy the environment template and add your wandb API key:

```bash
cp .env.template .env
# Edit .env and add your WANDB_API_KEY
```

### 2. Run Experiments

Basic usage:
```bash
python run_experiment.py configs/config_quick_test.json
```

With wandb logging:
```bash
python run_experiment.py configs/config_torch.json --wandb-project my-vmf-experiments
```

With team/entity:
```bash
python run_experiment.py configs/config_benchmark.json --wandb-project my-project --wandb-entity my-team
```

## Configuration

Configuration files are JSON format with the following parameters:

```json
{
  "dimension": 512,              // Dimension of the sphere
  "kappa": 10.0,              // Concentration parameter
  "num_samples": 1000,        // Number of samples to generate
  "implementation": "torch",   // "numpy", "torch", or "scipy"
  "seed": 42,                 // Random seed (null for random)
  "benchmark": true,          // Run benchmarking
  "profile": true,            // Run detailed profiling
  "output_file": "results/my_results.csv"  // Output file path
}
```

## Example Profiling Output

```
✓ Total profiled time: 0.162380s
✓ Total function calls: 7
✓ Time breakdown:
  - Sampling: 57.4%
  - Rotation: 42.6%
  - Other: 0.0%
✓ Top time-consuming functions:
  1. _sample_uniform_direction: 0.078835s (48.5%)
  2. torch.matmul: 0.037516s (23.1%)
  3. torch._C._linalg.linalg_qr: 0.031636s (19.5%)
  4. sample: 0.011190s (6.9%)
  5. _rotate_samples_torch: 0.002958s (1.8%)
```

## Environment Variables

Set up your `.env` file:

```bash
# Required for wandb integration
WANDB_API_KEY=your_api_key_here

# Optional settings
WANDB_MODE=online              # or "offline" for offline mode
WANDB_ENTITY=your_team_name    # Your wandb team/entity
```