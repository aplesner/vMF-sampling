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
    
    args = parser.parse_args()
    
    
    # Load base.yaml configuration
    config = VMFConfig.from_yaml('configs/base.yaml')
    setup_logging(config.verbosity)
    logger.info(f"Loaded base configuration from configs/base.yaml. W&B offline mode {config.wandb_offline}.")

    # TODO: This overrides the base config entirely with default values from VMFConfig. Introduce a merge function in VMFConfig to handle this more gracefully.
    # TODO: Remove the json and yaml distinction and have a single from_file method that infers type from suffix.
    # If additional config file is provided, load and override
    if args.config:
        config_path = Path(args.config)
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            override_config = VMFConfig.from_yaml(args.config)
        elif config_path.suffix.lower() == '.json':
            override_config = VMFConfig.from_json(args.config)
        else:
            raise ValueError(f"Configuration file must be .json or .yaml but got {config_path.suffix}")
        
        # Merge base config with override config  
        merged_dict = config.model_dump()
        merged_dict.update(override_config.model_dump())
        config = VMFConfig(**merged_dict)

        logger.debug(f"Loaded configuration from {args.config}")
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