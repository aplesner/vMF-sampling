"""
Configuration management for vMF sampling experiments.
"""

import json
from typing import Dict, Any
from pathlib import Path

from .vmf_sampler import Implementation


class VMFConfig:
    """Configuration class for vMF sampling experiments."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration from dictionary.
        
        Parameters
        ----------
        config_dict : dict
            Configuration dictionary with keys:
            - mu_dim: int or list, dimension(s) of mu (can be single int or list for sweep)
            - kappa: float, concentration parameter
            - num_samples: int, number of samples to generate
            - implementation: str or list, one of 'scipy', 'numpy', 'torch' (can be single or list)
            - seed: int or None, random seed
            - benchmark: bool, whether to run benchmarks
            - profile: bool, whether to run line-by-line profiling
            - output_file: str, output CSV file path
            - device: str, torch device ('cpu', 'cuda', 'auto')
            - offline: bool, whether to save CSV locally (default True, set False for wandb-only)
        """
        self.mu_dim = config_dict.get('mu_dim', 3)
        self.kappa = config_dict.get('kappa', 1.0)
        self.num_samples = config_dict.get('num_samples', 1000)
        
        # Handle single implementation or list of implementations
        impl_value = config_dict.get('implementation', 'numpy')
        if isinstance(impl_value, list):
            self.implementations = [Implementation(impl) for impl in impl_value]
        else:
            self.implementations = [Implementation(impl_value)]
        # Keep backward compatibility
        self.implementation = self.implementations[0]
        
        self.seed = config_dict.get('seed', None)
        self.benchmark = config_dict.get('benchmark', True)
        self.profile = config_dict.get('profile', False)
        self.output_file = config_dict.get('output_file', 'results/vmf_results.csv')
        self.device = config_dict.get('device', 'auto')
        self.offline = config_dict.get('offline', True)
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        # Validate mu_dim (can be int or list)
        if isinstance(self.mu_dim, list):
            if not all(dim >= 2 for dim in self.mu_dim):
                raise ValueError("All dimensions in mu_dim must be >= 2")
        else:
            if self.mu_dim < 2:
                raise ValueError("mu_dim must be >= 2")
                
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
    
    @classmethod
    def from_json(cls, filepath: str):
        """Load configuration from JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'mu_dim': self.mu_dim,
            'kappa': self.kappa,
            'num_samples': self.num_samples,
            'implementation': self.implementation.value,
            'seed': self.seed,
            'benchmark': self.benchmark,
            'profile': self.profile,
            'output_file': self.output_file,
            'device': self.device,
            'offline': self.offline
        }
    
    def get_dimensions(self):
        """Get list of dimensions to process."""
        if isinstance(self.mu_dim, list):
            return self.mu_dim
        else:
            return [self.mu_dim]
    
    def __repr__(self):
        return f"VMFConfig({self.to_dict()})"