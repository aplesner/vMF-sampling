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
import logging

logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import VMFConfig, run_benchmark, VMFLogger
from src.logger import setup_logging


def main():
    """
    Main execution function that accepts config file as input or works with wandb sweep.
    """
    parser = argparse.ArgumentParser(
        description='Run vMF sampling experiments with benchmarking and profiling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --config configs/base.yaml
  wandb sweep wandb_sweep.yaml  # Use wandb for parameter sweeps
        """
    )
    
    parser.add_argument('--config', 
                       help='Path to configuration file (JSON or YAML)')
    
    args = parser.parse_args()
    
    # Check if running as part of wandb sweep
    import wandb
    if wandb.run is not None:
        # Running in wandb sweep - use sweep config
        logger.info("Running as part of wandb sweep")
        
        # Create config from wandb sweep parameters
        sweep_config = dict(wandb.config)
        config = VMFConfig(**sweep_config)
        
        setup_logging(config.verbosity)
        logger.info(f"Sweep configuration: {dict(wandb.config)}")
    else:
        # Running standalone - use config file or default
        if args.config:
            config = VMFConfig.from_config_file(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            config = VMFConfig.from_config_file('configs/base.yaml')
            logger.info("Loaded base configuration from configs/base.yaml")
        
        setup_logging(config.verbosity)
        logger.debug(f"Final configuration: {config}")

    
    # Initialize logger
    experiment_logger = VMFLogger(config=config)
    setup_logging(config.verbosity)

    logger.info("=" * 60)
    logger.info("vMF Sampling Experiment")
    logger.debug("=" * 60)
    logger.debug(f"Configuration:")
    logger.debug(f"  Dimension: {config.dimension}")
    logger.debug(f"  Kappa: {config.kappa}")
    logger.debug(f"  Samples: {config.num_samples}")
    logger.debug(f"  Implementation: {config.implementation}")
    logger.debug(f"  Seed: {config.seed}")
    logger.debug(f"  Device: {config.device}")
    logger.debug(f"  W&B Project: {config.wandb_project}")
    logger.debug(f"  W&B Offline: {config.wandb_offline}")

    logger.info("=" * 60)

    logger.info("Running Benchmarks...")
    timing_results = {}
    
    timing_results = run_benchmark(config)
    logger.info(f"✓ Mean time: {timing_results['mean_time']:.6f}s")
    logger.info(f"✓ Median time: {timing_results['median_time']:.6f}s")
    logger.info(f"✓ Std dev: {timing_results['std']:.6f}s")

    try:
        experiment_logger.log_experiment(config, timing_results)
        logger.info(f"✓ Results logged successfully")
    except Exception as e:
        logger.error(f"✗ Logging failed: {e}")

    experiment_logger.finish()

    logger.info("=" * 60)
    logger.info("Experiment Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()