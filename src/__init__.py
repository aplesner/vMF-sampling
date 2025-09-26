"""
vMF Sampling Package

A comprehensive package for von Mises-Fisher distribution sampling with
benchmarking, profiling, and logging capabilities.
"""

from .vmf_sampler import vMF, Implementation
from .config import Config

__version__ = "1.0.0"
__author__ = "Claude Code"

__all__ = [
    'vMF',
    'Implementation', 
    'Config'
]