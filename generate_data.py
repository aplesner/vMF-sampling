"""Simple script to generate vMF samples."""

import numpy as np
import torch
import logging
import argparse
import os
from pathlib import Path

from src.vmf_sampler import vMF
from src.config import VMFConfig


def generate_samples(dimension: int, kappa: float, num_samples: int = 1000, 
                    implementation: str = "torch", device: str = "auto", 
                    dtype: str = "float32", output_dir: str = "samples/",
                    seed: int = 42, random_mean_direction: bool = True,
                    enable_wandb: bool = False, wandb_project: str = "vmf-sampling"):
    """Generate vMF samples with specified parameters."""
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create sampler
    sampler = vMF(
        dim=dimension,
        rotation_needed=False,
        kappa=kappa,
        seed=seed,
        device=device,
        dtype=getattr(torch, dtype) if implementation == "torch" else getattr(np, dtype),
        use_scipy_implementation=(implementation == "scipy")
    )
    
    # Set random mean direction if requested
    if random_mean_direction:
        mu = np.random.randn(dimension).astype(getattr(np, dtype))
        mu = mu / np.linalg.norm(mu)
        if implementation == "torch":
            dtype_obj = getattr(torch, dtype)
            mu = torch.from_numpy(mu).to(dtype=dtype_obj)
            if sampler.device.type == 'cuda':
                mu = mu.cuda()
        sampler.set_mu(mu)
    
    # Initialize wandb logging if enabled
    logger = None
    if enable_wandb:
        try:
            # Create temp config for logging
            temp_config = VMFConfig(
                dimension=dimension, kappa=kappa, num_samples=num_samples,
                implementation=implementation, device=device, dtype=dtype,
                output_dir=output_dir, seed=seed, random_mean_direction=random_mean_direction,
                wandb_project=wandb_project, wandb_offline=False
            )
            logger = VMFLogger(temp_config)
            logging.info(f"Wandb logging enabled for project: {wandb_project}")
        except Exception as e:
            logging.warning(f"Failed to initialize wandb: {e}")
            temp_config = None
    else:
        temp_config = None
    
    # Generate samples
    import time
    start_time = time.time()
    samples = sampler.sample(num_samples)
    generation_time = time.time() - start_time
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save samples
    filename = f"vmf_samples_dim{dimension}_kappa{kappa}_n{num_samples}_seed{seed}"
    if isinstance(samples, torch.Tensor):
        output_path = os.path.join(Path(output_dir), f"{filename}.pt")
        torch.save(samples.cpu(), output_path)
    else:
        output_path = os.path.join(Path(output_dir), f"{filename}.npy")
        np.save(output_path, samples)
    
    logging.info(f"Generated {num_samples} samples: dim={dimension}, kappa={kappa} -> {output_path}")
    logging.info(f"Generation time: {generation_time:.4f}s")
    
    # Log to wandb if enabled
    if logger:
        try:
            timing_results = {
                'generation_time': generation_time,
                'samples_per_second': num_samples / generation_time,
                'device': str(sampler.device) if hasattr(sampler, 'device') else 'unknown'
            }
            logger.log_experiment(temp_config, timing_results)
            logger.finish()
            logging.info("Results logged to wandb")
        except Exception as e:
            logging.warning(f"Failed to log to wandb: {e}")
    
    return output_path


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate vMF samples')
    parser.add_argument('--config', help='Config file path (optional)')
    parser.add_argument('--dimension', type=int, help='Dimension')
    parser.add_argument('--kappa', type=float, help='Concentration parameter')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--output_dir', default='samples/', help='Output directory')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_project', default='vmf-sampling', help='Wandb project name')
    
    args = parser.parse_args()
    
    if args.config:
        # Use config file
        config = VMFConfig.from_config_file(args.config)
        setup_logging(config.verbosity)
        generate_samples(
            dimension=config.dimension,
            kappa=config.kappa,
            num_samples=config.num_samples,
            implementation=config.implementation,
            device=config.device,
            dtype=config.dtype,
            output_dir=config.output_dir,
            seed=config.seed or 42,
            random_mean_direction=config.random_mean_direction,
            enable_wandb=args.wandb or not config.wandb_offline,
            wandb_project=args.wandb_project or config.wandb_project
        )
    else:
        # Use command line arguments
        setup_logging("info")
        if not args.dimension or not args.kappa:
            parser.error("Must specify --dimension and --kappa, or use --config")
        
        generate_samples(
            dimension=args.dimension,
            kappa=args.kappa,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            enable_wandb=args.wandb,
            wandb_project=args.wandb_project
        )


if __name__ == "__main__":
    main()