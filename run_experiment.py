#!/usr/bin/env python3
"""
Main script for running vMF sampling experiments.

This script provides a command-line interface for running von Mises-Fisher
sampling experiments with configurable parameters, benchmarking, profiling,
and logging capabilities.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import VMFConfig, run_benchmark, VMFLogger

# Load environment variables from .env file if it exists
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


def main():
    """
    Main execution function that accepts config file as input.
    """
    parser = argparse.ArgumentParser(
        description='Run vMF sampling experiments with benchmarking and profiling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --config configs/numpy.yaml
        """
    )
    
    parser.add_argument('--config', 
                       help='Path to configuration file (JSON or YAML)')
    parser.add_argument('--env-file', 
                       default='.env',
                       help='Path to environment file (default: .env)')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_env_file(args.env_file)
    
    # Load base.yaml configuration
    config = VMFConfig.from_yaml('configs/base.yaml')

    # If additional config file is provided, load and override
    if args.config:
        config_path = Path(args.config)
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = config.from_yaml(args.config)
        elif config_path.suffix.lower() == '.json':
            config = config.from_json(args.config)
        else:
            raise ValueError(f"Configuration file must be .json or .yaml but got {config_path.suffix}")

        print(f"Loaded configuration from {args.config}")

    
    # Initialize logger
    logger = VMFLogger(config=config)

    print("=" * 60)
    print("vMF Sampling Experiment")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Dimension: {config.dimension}")
    print(f"  Kappa: {config.kappa}")
    print(f"  Samples: {config.num_samples}")
    print(f"  Implementation: {config.implementation}")
    print(f"  Seed: {config.seed}")
    print(f"  Device: {config.device}")
    print(f"  W&B Project: {config.wandb_project}")
    print(f"  W&B Offline: {config.wandb_offline}")
    
    print(f"{'='*60}")
    
    print("Running Benchmarks...")
    timing_results = {}
    
    timing_results = run_benchmark(config)
    print(f"✓ Mean time: {timing_results['mean_time']:.6f}s")
    print(f"✓ Median time: {timing_results['median_time']:.6f}s")
    print(f"✓ Std dev: {timing_results['std']:.6f}s")
    print(f"✓ Device: {timing_results['device']}")
    # except Exception as e:
    #     print(f"✗ Benchmark failed: {e}")
    #     timing_results = {}
    
    # Log results
    try:
        logger.log_experiment(config, timing_results)
        print(f"✓ Results logged successfully")
    except Exception as e:
        print(f"✗ Logging failed: {e}")
    
    logger.finish()
    
    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()