"""vMF sampling profiling script using line_profiler."""

import argparse
import logging
import numpy as np
import torch
try:
    from line_profiler import LineProfiler
    HAS_LINE_PROFILER = True
except ImportError:
    HAS_LINE_PROFILER = False

from src.config import Config
from src.vmf_sampler import vMF


def setup_logging(level: str):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def profile_sampling(config: Config):
    """Profile vMF sampling with line profiler."""
    if not HAS_LINE_PROFILER:
        print("❌ line_profiler not installed. Install with: pip install line_profiler")
        return
    
    # Set seeds
    if config.seed:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
    
    # Create mean direction
    mu = np.random.randn(config.dimension)
    mu = mu / np.linalg.norm(mu)
    
    # Initialize sampler
    sampler = vMF(
        dim=config.dimension,
        kappa=config.kappa,
        seed=config.seed,
        device=config.device,
        dtype=getattr(torch, config.dtype) if config.implementation == "torch" else getattr(np, config.dtype),
        use_scipy_implementation=(config.implementation == "scipy")
    )
    
    # Convert mu for torch implementation
    if config.implementation == "torch":
        mu = torch.from_numpy(mu).to(dtype=getattr(torch, config.dtype))
        if sampler.device.type == 'cuda':
            mu = mu.cuda()
    
    # Create profiler and add functions to profile
    profiler = LineProfiler()
    profiler.add_function(sampler.sample)
    profiler.add_function(sampler.rotate_samples)
    profiler.add_function(sampler._sample_uniform_direction)
    
    # Add implementation-specific methods
    if config.implementation == "torch":
        profiler.add_function(sampler._rotate_samples_torch)
        profiler.add_function(sampler._compute_rotation_matrix_torch)
    elif config.implementation == "scipy":
        profiler.add_function(sampler._rotate_samples_scipy)
        profiler.add_function(sampler._compute_rotation_matrix_scipy)
    else:  # numpy
        profiler.add_function(sampler._rotate_samples_numpy)
        profiler.add_function(sampler._compute_rotation_matrix_numpy)
    
    print(f"🔍 Profiling {config.implementation} implementation...")
    print(f"   Dimension: {config.dimension}, Kappa: {config.kappa}, Samples: {config.num_samples}")
    
    # Run profiling
    profiler.enable_by_count()
    samples = sampler.sample(config.num_samples, mu=mu)
    profiler.disable_by_count()
    
    # Print results
    print(f"\n📊 Profiling Results:")
    profiler.print_stats()
    
    print(f"\n✅ Generated {len(samples)} samples")
    return samples


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Profile vMF sampling')
    parser.add_argument('--config', help='Config file path (optional)')
    args = parser.parse_args()
    
    # Load configuration
    config = Config.load(args.config)
    setup_logging(config.verbosity)
    
    # Run profiling
    profile_sampling(config)


if __name__ == "__main__":
    main()