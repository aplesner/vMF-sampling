# vMF Sampling - Minimal Setup

Simplified von Mises-Fisher sampling with clean configuration management and profiling.

## Quick Start

### Basic Generation
```bash
# Use default config (configs/base.yaml)
python generate.py

# Use custom config
python generate.py --config my_config.yaml
```

### Benchmarking
```bash
# Benchmark with default config
python benchmark.py

# Benchmark with custom config  
python benchmark.py --config configs/torch.yaml
```

### Profiling
```bash
# Profile with line_profiler (requires: pip install line_profiler)
python profile.py

# Profile specific implementation
python profile.py --config configs/numpy.yaml
```

## Configuration

The system uses **Pydantic + OmegaConf** for clean config management:

- `configs/base.yaml` - Default values for all parameters
- Custom configs merge with base config automatically
- Type validation and error checking included

### Example Config
```yaml
# my_config.yaml
dimension: 5000
kappa: 2000.0
num_samples: 1000
implementation: "torch"
dtype: "float16"
device: "auto"
```

## Scripts

- **`generate.py`** - Generate vMF samples (22 lines)
- **`benchmark.py`** - Benchmark implementations (21 lines) 
- **`profile.py`** - Line-by-line profiling (18 lines)

## Features

- **Minimal code**: Each script <25 lines
- **Clean config**: Pydantic validation + OmegaConf merging
- **Profiling ready**: Line profiler decorators on key methods
- **Three implementations**: NumPy, SciPy, PyTorch for benchmarking
- **Type safety**: Full type hints and validation

## Dependencies

```bash
pip install pydantic omegaconf torch numpy scipy
pip install line_profiler  # For profiling
```