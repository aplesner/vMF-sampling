"""vMF sampling benchmarking script."""

import argparse
import logging
import numpy as np
import torch
from torch.utils import benchmark

from src.config import Config
from src.vmf_sampler import vMF


def setup_logging(level: str):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def run_benchmark(config: Config) -> dict:
    """Run benchmarking for vMF sampling."""
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
    
    # Define sampling function
    def sample_func():
        return sampler.sample(config.num_samples, mu=mu)
    
    # Run benchmark
    timer = benchmark.Timer(
        stmt='sample_func()',
        globals={'sample_func': sample_func},
        label=f'vMF sampling ({config.implementation})',
        description=f'dim={config.dimension}, kappa={config.kappa}, n={config.num_samples}'
    )
    
    measurement = timer.blocked_autorange(min_run_time=2.0)
    
    results = {
        'mean_time': measurement.mean,
        'median_time': measurement.median,
        'std': float(np.std(measurement.raw_times)),
        'iterations_per_second': len(measurement.times) / 2.0,
        'device': str(sampler.device),
        'implementation': config.implementation,
        'dimension': config.dimension,
        'kappa': config.kappa,
        'num_samples': config.num_samples
    }
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Benchmark vMF sampling')
    parser.add_argument('--config', help='Config file path (optional)')
    args = parser.parse_args()
    
    # Load configuration
    config = Config.load(args.config)
    setup_logging(config.verbosity)
    
    # Run benchmark
    logging.info(f"Running benchmark: {config.implementation}, dim={config.dimension}, kappa={config.kappa}")
    results = run_benchmark(config)
    
    # Print results
    print(f"\nBenchmark Results:")
    print(f"  Implementation: {results['implementation']}")
    print(f"  Device: {results['device']}")
    print(f"  Mean time: {results['mean_time']:.6f}s")
    print(f"  Median time: {results['median_time']:.6f}s")
    print(f"  Std dev: {results['std']:.6f}s")
    print(f"  Iterations/sec: {results['iterations_per_second']:.2f}")


if __name__ == "__main__":
    main()