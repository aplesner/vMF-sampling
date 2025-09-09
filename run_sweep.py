import logging

import os
import sys
from pathlib import Path
import wandb

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import VMFConfig, run_benchmark


def setup_logging(level: str) -> None:
    """Configure root logger for the entire application."""

    format_string = "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"

    logging_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=logging_level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True
    )


def objective(config: dict):
    """Run vMF sampling experiment with given config parameters."""
    
    # Filter invalid combinations (torch-specific device settings)
    if config['implementation'] != 'torch' and config['device'] in ['cuda']:
        logging.debug(f"Skipping invalid combination: {config['implementation']} with {config['device']}")
        wandb.log({"status": "skipped", "reason": "invalid_device_combination"})
        return None
    elif config['implementation'] == 'torch':
        # define implementation as torch:device for clarity
        config['implementation'] = f"torch:{config['device']}"
        # update in wandb config
        wandb.config.update({'implementation': config['implementation']})

    # Create new config object
    try:
        vmf_config = VMFConfig(**config)
    except Exception as e:
        logging.debug(f"✗ Error creating config with sweep parameters: {e}")
        wandb.log({"status": "error", "error": str(e)})
        return None

    logging.debug("Running Benchmark...")
    try:
        timing_results = run_benchmark(vmf_config)
        # if verbose debug, print full timing results
        logging.debug(f"✓ Mean time: {timing_results['mean_time']:.6f}s")
        logging.debug(f"✓ Median time: {timing_results['median_time']:.6f}s")
        logging.debug(f"✓ Std dev: {timing_results['std']:.6f}s")
        logging.debug(f"✓ Device: {timing_results['device']}")
        result = {
            'dimension': config['dimension'],  # Use 'dimension' for clearer plotting
            'mean_runtime': timing_results.get('mean_time', None),  # Explicit mean runtime
            'median_runtime': timing_results.get('median_time', None),
            'std_runtime': timing_results.get('std', None),
            'kappa': config['kappa'],
            'num_samples': config['num_samples'],
            'seed': config['seed'],
            **{k: v for k, v in timing_results.items() if k not in ['mean_time', 'median_time', 'std', 'device']}
        }

        # Log results to W&B
        wandb.log(result)
        logging.debug(f"✓ Results logged successfully")

        # Return the main metric for sweep optimization
        return timing_results['mean_time']
        
    except Exception as e:
        logging.debug(f"✗ Benchmark failed: {e}")
        wandb.log({"status": "failed", "error": str(e)})
        return None

def main():
    """Main function for sweep agent."""
    # Load base configuration
    try:
        base_config = VMFConfig.from_yaml('configs/base.yaml')
        logging.debug(f"Loaded base configuration from configs/base.yaml")
    except Exception as e:
        logging.debug(f"✗ Error loading base configuration: {e}")
        return None

    
    with wandb.init() as run:
        # Merge base config with sweep config
        run.config.update(base_config.model_dump(), allow_val_change=False)
        # get logging level from config
        verbosity = run.config.get('verbosity', 'info')
        setup_logging(verbosity)
        logging.info(f"Starting sweep run with ID: {run.id}")

        mean_time = objective(dict(run.config))
        if mean_time is not None:
            # Log the main optimization metric
            run.log({"benchmark/mean_time": mean_time})
            logging.info(f"\n✓ Experiment completed - Mean time: {mean_time:.6f}s")
        else:
            logging.warning(f"\n✗ Experiment failed or skipped - no mean time to log")

if __name__ == "__main__":
    main()