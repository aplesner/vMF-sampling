"""Script that generates vMF samples given configuration arguments."""

import numpy as np
import torch

import argparse
import logging
from pathlib import Path
import os

from src.vmf_sampler import vMF, Implementation
from src.config import VMFConfig

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


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(config: VMFConfig):
    """Main function to generate vMF samples based on config file."""

    logging.debug(f"Using configuration: {config}")

    set_seed(config.seed or 42)
    
    sampler = vMF(
        dim=config.dimension,
        rotation_needed=False,
        kappa=config.kappa,
        seed=config.seed or 42,
        device=config.device,
        use_scipy_implementation=(config.implementation == Implementation.SCIPY)
    )

    if config.random_mean_direction:
        mu = np.random.randn(config.dimension)
        mu = mu / np.linalg.norm(mu)
        if config.implementation == Implementation.TORCH:
            mu = torch.from_numpy(mu).float()
            if config.device == 'cuda' and torch.cuda.is_available():
                mu = mu.cuda()
        sampler.set_mu(mu)

    # Sample values
    samples = sampler.sample(config.num_samples)

    # Save samples to specified output path
    output_path = Path(os.path.join(
        config.output_dir, 
        f"vmf_samples_dim{config.dimension}_kappa{config.kappa}_n{config.num_samples}_rand_mu{config.random_mean_direction}_seed{config.seed}.npy"
    ))

    os.makedirs(config.output_dir, exist_ok=True)

    if isinstance(samples, torch.Tensor):
        samples = samples.cpu()
        output_path = output_path.with_suffix('.pt')
        torch.save(samples, output_path)
    else:
        np.save(output_path, samples)

    logging.info(f"Saved {config.num_samples} samples to {output_path}")


if __name__ == "__main__":
    # Load configuration and set up logging
    parser = argparse.ArgumentParser(
        description='Generate samples from vMF distribution based on config file'
    )
    parser.add_argument('--config', 
                        required=False,
                        help='Path to configuration file (JSON or YAML) to override defaults')
    
    args = parser.parse_args()

    # Load base configuration at `generation_configs/base.yaml`
    base_config = VMFConfig.from_yaml('generation_configs/base.yaml')

    # Override base configuration with any user-specified values
    if args.config:
        user_config = VMFConfig.from_yaml(args.config) if args.config.endswith(('.yaml', '.yml')) else VMFConfig.from_json(args.config)
        config = base_config.model_copy(update=user_config.model_dump(), deep=True)
    else:
        config = base_config

    setup_logging(config.verbosity)

    logging.info("Starting vMF sampling...")

    main(config)