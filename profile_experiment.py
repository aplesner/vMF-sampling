#!/usr/bin/env python3
"""
Profiling script for vMF sampling experiments.

This script provides detailed line-by-line profiling of vMF sampling
implementations to identify performance bottlenecks.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import VMFConfig, run_profiling

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
    Main profiling function that accepts config file as input.
    """
    parser = argparse.ArgumentParser(
        description='Profile vMF sampling experiments with detailed line-by-line analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python profile_experiment.py base.yaml
  python profile_experiment.py configs/experiment.yaml
  python profile_experiment.py configs/high_dim.json
        """
    )
    
    parser.add_argument('config', 
                       help='Path to configuration file (JSON or YAML)')
    parser.add_argument('--env-file', 
                       default='.env',
                       help='Path to environment file (default: .env)')
    parser.add_argument('--output-file',
                       help='Optional output file for profiling results')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_env_file(args.env_file)
    
    # Load configuration
    try:
        config_path = Path(args.config)
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = VMFConfig.from_yaml(args.config)
        else:
            config = VMFConfig.from_json(args.config)
        print(f"Loaded configuration from {args.config}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    print("=" * 60)
    print("vMF Sampling Profiler")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Dimension: {config.mu_dim}")
    print(f"  Kappa: {config.kappa}")
    print(f"  Samples: {config.num_samples}")
    print(f"  Implementation: {config.implementation}")
    print(f"  Seed: {config.seed}")
    print(f"  Device: {config.device}")
    
    print(f"\n{'='*60}")
    print(f"Running Detailed Profiling: dim={config.mu_dim}, impl={config.implementation}")
    print(f"{'='*60}")
    
    # Run profiling
    try:
        print("Running Detailed Profiling...")
        profile_results = run_profiling(config)
        
        print(f"✓ Total profiled time: {profile_results.get('total_time', 0):.6f}s")
        print(f"✓ Total function calls: {profile_results.get('total_calls', 0)}")
        
        # Print time breakdown
        if 'time_breakdown' in profile_results:
            breakdown = profile_results['time_breakdown']
            print(f"✓ Time breakdown:")
            print(f"  - Sampling: {breakdown.get('sampling_percentage', 0):.1f}%")
            print(f"  - Rotation: {breakdown.get('rotation_percentage', 0):.1f}%")
            print(f"  - Other: {breakdown.get('other_percentage', 0):.1f}%")
        
        # Print detailed profiling stats if available
        if 'detailed_stats' in profile_results:
            print(f"\n{'='*60}")
            print("Detailed Profiling Statistics:")
            print(f"{'='*60}")
            stats = profile_results['detailed_stats']
            for func_name, func_stats in stats.items():
                print(f"{func_name}:")
                print(f"  Calls: {func_stats.get('ncalls', 0)}")
                print(f"  Total time: {func_stats.get('tottime', 0):.6f}s")
                print(f"  Cumulative time: {func_stats.get('cumtime', 0):.6f}s")
                print(f"  Per call: {func_stats.get('percall', 0):.6f}s")
                print()
        
        # Save results if output file specified
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(output_path, 'w') as f:
                json.dump(profile_results, f, indent=2)
            print(f"✓ Profiling results saved to {args.output_file}")
            
    except Exception as e:
        print(f"✗ Profiling failed: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Profiling Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()