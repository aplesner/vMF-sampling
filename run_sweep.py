#!/usr/bin/env python3
"""
W&B Sweep script for vMF sampling experiments.
"""

import os
import sys
from pathlib import Path
import wandb

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import VMFConfig, run_benchmark, VMFLogger

def load_env_file(env_file: str = ".env"):
    """Load environment variables from file."""
    env_path = Path(env_file)
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, _, value = line.partition('=')
                    os.environ[key.strip()] = value.strip()

def objective(config):
    """Run vMF sampling experiment with given config parameters."""
    
    # Load base configuration
    try:
        base_config = VMFConfig.from_yaml('configs/base.yaml')
        print(f"Loaded base configuration from configs/base.yaml")
    except Exception as e:
        print(f"Error loading base configuration: {e}")
        return None
    
    # Create config dict and override with sweep parameters
    config_dict = base_config.model_dump()
    
    # Override with sweep parameters
    for key, value in config.items():
        if key in config_dict:
            config_dict[key] = value
            print(f"Override {key}: {value}")
    
    # Filter invalid combinations (torch-specific device settings)
    if config_dict['implementation'] != 'torch' and config_dict['device'] in ['cuda']:
        print(f"Skipping invalid combination: {config_dict['implementation']} with {config_dict['device']}")
        wandb.log({"status": "skipped", "reason": "invalid_device_combination"})
        return None
    
    # Create new config object
    try:
        vmf_config = VMFConfig(**config_dict)
    except Exception as e:
        print(f"Error creating config with sweep parameters: {e}")
        wandb.log({"status": "error", "error": str(e)})
        return None
    
    print("=" * 60)
    print("vMF Sampling Experiment (W&B Sweep)")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Dimension: {vmf_config.mu_dim}")
    print(f"  Kappa: {vmf_config.kappa}")
    print(f"  Samples: {vmf_config.num_samples}")
    print(f"  Implementation: {vmf_config.implementation}")
    print(f"  Seed: {vmf_config.seed}")
    print(f"  Device: {vmf_config.device}")
    print(f"  W&B Project: {vmf_config.wandb_project}")
    print(f"  W&B Offline: {vmf_config.wandb_offline}")
    
    # Initialize logger
    logger = VMFLogger(config=vmf_config)
    
    print("Running Benchmarks...")
    timing_results = {}
    
    try:
        timing_results = run_benchmark(vmf_config)
        print(f"✓ Mean time: {timing_results['mean_time']:.6f}s")
        print(f"✓ Median time: {timing_results['median_time']:.6f}s")
        print(f"✓ Std dev: {timing_results['std']:.6f}s")
        print(f"✓ Device: {timing_results['device']}")
        
        # Log results to our logger
        logger.log_experiment(vmf_config, timing_results)
        print(f"✓ Results logged successfully")
        
        # Return the main metric for sweep optimization
        return timing_results['mean_time']
        
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        wandb.log({"status": "failed", "error": str(e)})
        return None
    finally:
        logger.finish()

def main():
    """Main function for sweep agent."""
    # Load environment variables
    load_env_file()
    
    with wandb.init() as run:
        mean_time = objective(run.config)
        if mean_time is not None:
            # Log the main optimization metric
            run.log({"benchmark/mean_time": mean_time})
            print(f"\n✓ Experiment completed - Mean time: {mean_time:.6f}s")
        else:
            print(f"\n✗ Experiment failed or skipped")

if __name__ == "__main__":
    main()