"""Simple vMF sample generation script."""

import argparse
import logging
import numpy as np
import torch
from pathlib import Path

from src.config import Config
from src.vmf_sampler import vMF


def setup_logging(level: str):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def generate_samples(config: Config) -> Path:
    """Generate vMF samples based on configuration."""
    # Set seeds
    if config.seed:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
    
    # Create sampler
    sampler = vMF(
        dim=config.dimension,
        kappa=config.kappa,
        seed=config.seed,
        device=config.device,
        dtype=getattr(torch, config.dtype) if config.implementation == "torch" else getattr(np, config.dtype),
        use_scipy_implementation=(config.implementation == "scipy")
    )
    
    # Set random mean direction if requested
    if config.random_mean_direction:
        mu = np.random.randn(config.dimension).astype(getattr(np, config.dtype))
        mu = mu / np.linalg.norm(mu)
        if config.implementation == "torch":
            mu = torch.from_numpy(mu).to(dtype=getattr(torch, config.dtype))
            if sampler.device.type == 'cuda':
                mu = mu.cuda()
        sampler.set_mu(mu)
    
    # Generate samples
    samples = sampler.sample(config.num_samples)
    
    # Save samples
    Path(config.output_dir).mkdir(exist_ok=True)
    filename = f"vmf_samples_dim{config.dimension}_kappa{config.kappa}_n{config.num_samples}"
    
    if isinstance(samples, torch.Tensor):
        output_path = Path(config.output_dir) / f"{filename}.pt"
        torch.save(samples.cpu(), output_path)
    else:
        output_path = Path(config.output_dir) / f"{filename}.npy"
        np.save(output_path, samples)
    
    logging.info(f"Generated {config.num_samples} samples -> {output_path}")
    return output_path


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate vMF samples')
    parser.add_argument('--config', help='Config file path (optional)')
    args = parser.parse_args()
    
    # Load configuration
    config = Config.load(args.config)
    setup_logging(config.verbosity)
    
    # Generate samples
    generate_samples(config)


if __name__ == "__main__":
    main()