"""
Enhanced profiling utilities for vMF sampling with detailed line-by-line analysis.
"""

import cProfile
import pstats
from pstats import SortKey
import numpy as np
import torch
from typing import Dict, Any, List, Tuple
import inspect
import linecache

from .vmf_sampler import vMF, Implementation
from .config import VMFConfig


class DetailedProfiler:
    """Enhanced profiler that provides line-by-line analysis."""
    
    def __init__(self, config: VMFConfig):
        self.config = config
        self.profiler = cProfile.Profile(subcalls=True, builtins=True)
        
    def _get_function_lines(self, func_name: str, filename: str) -> List[str]:
        """Get source lines for a function."""
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            return lines
        except:
            return []
    
    def _analyze_function_timing(self, stats: pstats.Stats, 
                                target_functions: List[str]) -> Dict[str, Any]:
        """Analyze timing for specific functions of interest."""
        function_stats = {}
        
        for func_key, func_data in stats.stats.items():
            filename, lineno, func_name = func_key
            full_name = f"{func_name} ({filename}:{lineno})"
            
            # Check if this function is one we're interested in
            if any(target in func_name.lower() for target in target_functions):
                cc, nc, tt, ct = func_data[:4]
                
                function_stats[full_name] = {
                    'total_time': tt,
                    'cumulative_time': ct,
                    'num_calls': nc,
                    'time_per_call': tt / nc if nc > 0 else 0,
                    'filename': filename,
                    'line_number': lineno
                }
        
        return function_stats
    
    def profile_sampling(self) -> Dict[str, Any]:
        """Profile the sampling process with detailed analysis."""
        # Create mean direction
        np.random.seed(self.config.seed)
        mu = np.random.randn(self.config.dimension)
        mu = mu / np.linalg.norm(mu)
        
        # Determine device for torch
        if self.config.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.config.device
        
        # Initialize sampler
        sampler = vMF(
            dim=self.config.dimension,
            kappa=self.config.kappa,
            seed=self.config.seed,
            device=device,
            use_scipy_implementation=(self.config.implementation == Implementation.SCIPY)
        )
        
        if self.config.implementation == Implementation.TORCH:
            mu = torch.from_numpy(mu).float()
            if device == 'cuda':
                mu = mu.cuda()
        
        # Start profiling
        self.profiler.enable()
        
        # Run the sampling
        samples = sampler.sample(self.config.num_samples, mu=mu)
        
        self.profiler.disable()
        
        # Analyze results
        return self._analyze_profile_results()
    
    def _analyze_profile_results(self) -> Dict[str, Any]:
        """Analyze profiling results and extract detailed information."""
        stats = pstats.Stats(self.profiler)
        stats.sort_stats(SortKey.TIME)
        
        # Functions we're particularly interested in
        target_functions = [
            'sample', 'rotate', '_compute_rotation_matrix', 
            '_rotate_samples', '_sample_uniform_direction',
            'beta', 'random', 'linalg', 'qr', 'matmul'
        ]
        
        # Get detailed function analysis
        function_analysis = self._analyze_function_timing(stats, target_functions)
        
        # Get top time consumers
        top_functions = sorted(
            function_analysis.items(),
            key=lambda x: x[1]['total_time'],
            reverse=True
        )[:10]
        
        # Calculate totals
        total_time = sum(data['total_time'] for _, data in function_analysis.items())
        total_calls = sum(data['num_calls'] for _, data in function_analysis.items())
        
        # Categorize functions by component
        sampling_time = 0
        rotation_time = 0
        other_time = 0
        
        for func_name, data in function_analysis.items():
            if any(keyword in func_name.lower() for keyword in ['sample', 'beta', 'random']):
                sampling_time += data['total_time']
            elif any(keyword in func_name.lower() for keyword in ['rotate', 'qr', 'matmul']):
                rotation_time += data['total_time']
            else:
                other_time += data['total_time']
        
        return {
            'total_time': total_time,
            'total_calls': total_calls,
            'sampling_time': sampling_time,
            'rotation_time': rotation_time,
            'other_time': other_time,
            'top_functions': [
                {
                    'name': name,
                    'total_time': data['total_time'],
                    'cumulative_time': data['cumulative_time'],
                    'num_calls': data['num_calls'],
                    'time_per_call': data['time_per_call'],
                    'percentage': (data['total_time'] / total_time * 100) if total_time > 0 else 0
                }
                for name, data in top_functions
            ],
            'time_breakdown': {
                'sampling_percentage': (sampling_time / total_time * 100) if total_time > 0 else 0,
                'rotation_percentage': (rotation_time / total_time * 100) if total_time > 0 else 0,
                'other_percentage': (other_time / total_time * 100) if total_time > 0 else 0
            }
        }


def run_profiling(config: VMFConfig) -> Dict[str, Any]:
    """
    Run detailed profiling analysis.
    
    Parameters
    ----------
    config : VMFConfig
        Experiment configuration.
        
    Returns
    -------
    dict
        Detailed profiling results.
    """
    profiler = DetailedProfiler(config)
    return profiler.profile_sampling()