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

from src import VMFConfig, run_benchmark, run_profiling, VMFLogger

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
  python run_experiment.py configs/config_numpy.json
  python run_experiment.py configs/config_torch.json --wandb-project my-vmf-project
  python run_experiment.py configs/config_scipy.json --wandb-entity my-team
        """
    )
    
    parser.add_argument('config', 
                       help='Path to JSON configuration file')
    parser.add_argument('--wandb-project', 
                       help='wandb project name (optional)')
    parser.add_argument('--wandb-entity',
                       help='wandb entity/team name (optional)')
    parser.add_argument('--env-file', 
                       default='.env',
                       help='Path to environment file (default: .env)')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_env_file(args.env_file)
    
    # Load configuration
    try:
        config = VMFConfig.from_json(args.config)
        print(f"Loaded configuration from {args.config}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Initialize logger
    logger = VMFLogger(
        config.output_file, 
        args.wandb_project,
        args.wandb_entity
    )
    logger.set_offline_mode(config.offline)
    
    # Get dimensions and implementations to loop over
    dimensions = config.get_dimensions()
    implementations = config.implementations
    
    total_experiments = len(dimensions) * len(implementations)
    current_experiment = 0
    
    print("=" * 60)
    print("vMF Sampling Experiment Suite")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Dimensions: {dimensions}")
    print(f"  Kappa: {config.kappa}")
    print(f"  Samples: {config.num_samples}")
    print(f"  Implementations: {[impl.value for impl in implementations]}")
    print(f"  Seed: {config.seed}")
    print(f"  Device: {config.device}")
    print(f"  Output file: {config.output_file}")
    print(f"  Benchmark: {config.benchmark}")
    print(f"  Profile: {config.profile}")
    print(f"  Offline mode: {config.offline}")
    print(f"  Total experiments: {total_experiments}")
    
    # Loop over all combinations
    for dim in dimensions:
        for impl in implementations:
            current_experiment += 1
            print(f"\n{'='*60}")
            print(f"Experiment {current_experiment}/{total_experiments}: dim={dim}, impl={impl.value}")
            print(f"{'='*60}")
            
            # Create a temporary config for this specific experiment
            temp_config = VMFConfig({
                'mu_dim': dim,
                'kappa': config.kappa,
                'num_samples': config.num_samples,
                'implementation': impl.value,
                'seed': config.seed,
                'benchmark': config.benchmark,
                'profile': config.profile,
                'output_file': config.output_file,
                'device': config.device,
                'offline': config.offline
            })
            
            # Run benchmark
            timing_results = {}
            if temp_config.benchmark:
                print("Running Benchmarks...")
                try:
                    timing_results = run_benchmark(temp_config)
                    print(f"✓ Mean time: {timing_results['mean_time']:.6f}s")
                    print(f"✓ Median time: {timing_results['median_time']:.6f}s")
                    print(f"✓ Std dev: {timing_results['std']:.6f}s")
                    print(f"✓ Device: {timing_results['device']}")
                except Exception as e:
                    print(f"✗ Benchmark failed: {e}")
                    timing_results = {'benchmark_error': str(e)}
            
            # Run profiling
            profile_results = {}
            if temp_config.profile:
                print("Running Detailed Profiling...")
                try:
                    profile_results = run_profiling(temp_config)
                    print(f"✓ Total profiled time: {profile_results.get('total_time', 0):.6f}s")
                    print(f"✓ Total function calls: {profile_results.get('total_calls', 0)}")
                    
                    # Print time breakdown
                    if 'time_breakdown' in profile_results:
                        breakdown = profile_results['time_breakdown']
                        print(f"✓ Time breakdown:")
                        print(f"  - Sampling: {breakdown.get('sampling_percentage', 0):.1f}%")
                        print(f"  - Rotation: {breakdown.get('rotation_percentage', 0):.1f}%")
                        print(f"  - Other: {breakdown.get('other_percentage', 0):.1f}%")
                        
                except Exception as e:
                    print(f"✗ Profiling failed: {e}")
                    profile_results = {'profiling_error': str(e)}
            
            # Log results
            try:
                logger.log_experiment(temp_config, timing_results, profile_results)
                print(f"✓ Results logged for dim={dim}, impl={impl.value}")
            except Exception as e:
                print(f"✗ Logging failed: {e}")
    
    # Finish logging
    print("\n" + "=" * 60)
    print("Finalizing Results...")
    print("=" * 60)
    try:
        logger.finish()
        print(f"✓ All results logged successfully")
    except Exception as e:
        print(f"✗ Final logging failed: {e}")
    
    print("\n" + "=" * 60)
    print("Experiment Suite Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()