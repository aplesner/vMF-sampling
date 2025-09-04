"""
vMF Sampling Package

A comprehensive package for von Mises-Fisher distribution sampling with
benchmarking, profiling, and logging capabilities.
"""

from .vmf_sampler import vMF, Implementation
from .config import VMFConfig
from .benchmark import run_benchmark
from .profiler import run_profiling, DetailedProfiler
from .logger import VMFLogger

__version__ = "1.0.0"
__author__ = "Claude Code"

__all__ = [
    'vMF',
    'Implementation', 
    'VMFConfig',
    'run_benchmark',
    'run_profiling',
    'DetailedProfiler',
    'VMFLogger'
]