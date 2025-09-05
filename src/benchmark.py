"""
Benchmarking utilities for vMF sampling.
"""

import numpy as np
import torch
from torch.utils import benchmark
from typing import Dict, Callable

from .vmf_sampler import vMF, Implementation
from .config import VMFConfig

# Global benchmark time setting
BENCHMARK_TIME = 2.0


def run_benchmark(config: VMFConfig) -> Dict[str, float|str]:
    """
    Run benchmarking using torch.utils.benchmark.
    
    Parameters
    ----------
    config : VMFConfig
        Experiment configuration.
        
    Returns
    -------
    dict
        Benchmark timing results.
    """
    # Create mean direction
    np.random.seed(config.seed)
    mu = np.random.randn(config.mu_dim)
    mu = mu / np.linalg.norm(mu)
    
    # Determine device for torch
    if config.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = config.device
    
    # Initialize sampler
    sampler = vMF(
        dim=config.mu_dim,
        kappa=config.kappa,
        seed=config.seed,
        device=device,
        use_scipy_implementation=(config.implementation == Implementation.SCIPY)
    )
    
    if config.implementation == Implementation.TORCH:
        mu = torch.from_numpy(mu).float()
        if device == 'cuda':
            mu = mu.cuda()
    
    # Define sampling function
    def sample_func():
        return sampler.sample(config.num_samples, mu=mu)
    
    # Run benchmark
    timer = benchmark.Timer(
        stmt='sample_func()',
        globals={'sample_func': sample_func},
        label=f'vMF sampling ({config.implementation})',
        description=f'dim={config.mu_dim}, kappa={config.kappa}, n={config.num_samples}'
    )

    measurement = timer.blocked_autorange(
        min_run_time=BENCHMARK_TIME,
    )
    
    return {
        'mean_time': measurement.mean,
        'median_time': measurement.median,
        'std': float(np.std(measurement.raw_times)),
        'iterations per second': len(measurement.times) / BENCHMARK_TIME,
        'device': device
    }
