import time

import numpy as np
import scipy
import scipy.stats

# import cProfile
# import pstats
# from pstats import SortKey

import scipy_implementation

def sample_vMF_scipy(mu: np.array, kappa: float, n: int) -> np.ndarray:
    """
    Sample from the von Mises-Fisher distribution using scipy.

    Parameters
    ----------
    mu : np.ndarray
        The mean direction (unit vector).
    kappa : float
        Concentration parameter.
    n : int
        Number of samples to generate.

    Returns
    -------
    np.ndarray
        Samples from the von Mises-Fisher distribution.
    """

    assert kappa > 0, "kappa must be positive"

    samples = scipy.stats.vonmises_fisher.rvs(mu, kappa, size=n)

    return samples


def benchmark_sampling_methods(mu: np.array, kappa: float, n: int) -> None:
    """
    Benchmark the sampling methods for the von Mises-Fisher distribution.

    Parameters
    ----------
    mu : np.ndarray
        The mean direction (unit vector).
    kappa : float
        Concentration parameter.
    n : int
        Number of samples to generate.
    """

    # Benchmark numpy method
    start_time = time.time()
    scipy_implementation.sample_vMF(mu, kappa, n)
    numpy_time = time.time() - start_time

    # Benchmark scipy method
    start_time = time.time()
    sample_vMF_scipy(mu, kappa, n)
    scipy_time = time.time() - start_time

    print(f"NumPy method took {numpy_time:.4f} seconds")
    print(f"SciPy method took {scipy_time:.4f} seconds")


def main():
    # Parameters for the von Mises-Fisher distribution
    mu = np.zeros(2000)
    mu[-1] = 1.0  # Mean direction (unit vector)
    kappa = 10.0  # Concentration parameter
    n = 4000  # Number of samples

    # Benchmark the sampling methods
    benchmark_sampling_methods(mu, kappa, n)


def profile_implementation():
    """
    Profile the implementation of the von Mises-Fisher sampling methods.
    """

    mu = np.zeros(1500)
    mu[-1] = 1.0  # Mean direction (unit vector)
    kappa = 10.0  # Concentration parameter
    n = 2000  # Number of samples
    # pr = cProfile.Profile(subcalls=False, builtins=False)
    # pr.enable()
    
    scipy_implementation.sample_vMF(mu, kappa, n)

    # pr.disable()
    # pr.dump_stats('restats')

    # # Print the profiling results
    # p = pstats.Stats('restats')
    # p.strip_dirs().sort_stats(SortKey.LINE).print_stats()
    # # Profile the scipy implementation
    
if __name__ == "__main__":
    main()
