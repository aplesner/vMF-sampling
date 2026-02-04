"""Profiling decorators and utilities."""

import functools
import time
from typing import Callable, Any

# Global profiling state
PROFILING_ENABLED = False

try:
    from line_profiler import LineProfiler
    HAS_LINE_PROFILER = True
except ImportError:
    HAS_LINE_PROFILER = False


def enable_profiling():
    """Enable profiling decorators."""
    global PROFILING_ENABLED
    PROFILING_ENABLED = True


def disable_profiling():
    """Disable profiling decorators."""
    global PROFILING_ENABLED  
    PROFILING_ENABLED = False


def profile_time(func: Callable) -> Callable:
    """Decorator to profile function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        if not PROFILING_ENABLED:
            return func(*args, **kwargs)
        
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        print(f"⏱️  {func.__name__}: {end_time - start_time:.6f}s")
        return result
    return wrapper


def profile_line(func: Callable) -> Callable:
    """Decorator to mark function for line profiling."""
    # This is mainly a marker - actual line profiling is done externally
    func._profile_line = True
    return func


# Convenience decorator that combines both
def profile(func: Callable) -> Callable:
    """Decorator for both time and line profiling."""
    return profile_line(profile_time(func))