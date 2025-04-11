import time
import numpy as np
import torch
from typing import Callable, Optional, Tuple, Union, Any
from tqdm import tqdm
import unittest


def _check_random_state(seed: Optional[int]) -> Any:
    """
    Check and return a random number generator.
    
    Parameters
    ----------
    seed : Optional[int]
        Random seed. If None, use the default random state.
        
    Returns
    -------
    random_state : numpy.random.RandomState
        Random number generator.
    """
    assert seed is None or isinstance(seed, (int, np.integer)), (
        f"seed must be None or an integer, got {type(seed)}"
    )
    if seed is None:
        # Use the default random state
        return np.random
    else:
        return np.random.RandomState(seed)


def _sample_uniform_direction(dim: int, size: Union[int, Tuple[int, ...]], random_state: Any) -> np.ndarray:
    """
    Generate uniform directions on the unit sphere.
    
    Parameters
    ----------
    dim : int
        Dimension of the sphere.
    size : int or tuple
        Number of samples.
    random_state : numpy.random.RandomState
        Random number generator.
        
    Returns
    -------
    samples : ndarray, shape=(size, dim)
        Uniform samples from the unit sphere.
    """
    samples_shape = np.append(size, dim)
    samples = random_state.standard_normal(samples_shape)
    samples /= np.linalg.norm(samples, axis=-1, keepdims=True)
    return samples


def rotate_samples_torch(samples: np.ndarray, mu: np.ndarray, dim: int) -> np.ndarray:
    """
    A QR decomposition is used to find the rotation that maps the
    north pole (1, 0,...,0) to the vector mu. This rotation is then
    applied to all samples using PyTorch.
    
    Parameters
    ----------
    samples: array_like, shape = (n_samples, dim)
        Samples to rotate.
    mu : array-like, shape=(dim,)
        Point to parametrise the rotation.
    dim : int
        Dimension of the space.
        
    Returns
    -------
    samples : ndarray, shape=(n_samples, dim)
        Rotated samples.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = torch.tensor(samples, device=device, dtype=torch.float32)
    mu = torch.tensor(mu, device=device, dtype=torch.float32)
    with torch.no_grad():
        base_point = torch.zeros((dim, ), device=device, dtype=torch.float32)
        base_point[0] = 1.
        embedded = torch.cat([mu[None, :], torch.zeros((dim - 1, dim), device=device, dtype=torch.float32)], dim=0)
        embedded = torch.transpose(embedded, 0, 1)

        rotmatrix, _ = torch.linalg.qr(embedded)

        # check if the rotation is correct
        rotated_base_point = torch.mv(rotmatrix, base_point)
        if torch.allclose(rotated_base_point, mu):
            rotsign = 1
        else:
            rotsign = -1

        # apply rotation
        samples = torch.matmul(rotmatrix, samples.T).T * rotsign

        return samples.cpu().numpy()
        

def rotate_samples_numpy(samples: np.ndarray, mu: np.ndarray, dim: int) -> np.ndarray:
    """
    A QR decomposition is used to find the rotation that maps the
    north pole (1, 0,...,0) to the vector mu. This rotation is then
    applied to all samples using NumPy.
    
    Parameters
    ----------
    samples: array_like, shape = (n_samples, dim)
        Samples to rotate.
    mu : array-like, shape=(dim,)
        Point to parametrise the rotation.
    dim : int
        Dimension of the space.
        
    Returns
    -------
    samples : ndarray, shape=(n_samples, dim)
        Rotated samples.
    """    
    base_point = np.zeros((dim, ))
    base_point[0] = 1.
    embedded = np.concatenate([mu[None, :], np.zeros((dim - 1, dim))])

    rotmatrix, _ = np.linalg.qr(np.transpose(embedded))

    rotated_base_point = np.matmul(rotmatrix, base_point[:, None])[:, 0]
    if np.allclose(rotated_base_point, mu):
        rotsign = 1
    else:
        rotsign = -1

    # apply rotation (this is a matrix multiplication)
    samples = np.matmul(rotmatrix, samples.T).T * rotsign

    return samples


def rotate_samples_scipy(samples: np.ndarray, mu: np.ndarray, dim: int) -> np.ndarray:
    """
    A QR decomposition is used to find the rotation that maps the
    north pole (1, 0,...,0) to the vector mu. This rotation is then
    applied to all samples using SciPy's einsum.
    
    Parameters
    ----------
    samples: array_like, shape = (n_samples, dim)
        Samples to rotate.
    mu : array-like, shape=(dim,)
        Point to parametrise the rotation.
    dim : int
        Dimension of the space.
        
    Returns
    -------
    samples : ndarray, shape=(n_samples, dim)
        Rotated samples.
    """
    base_point = np.zeros((dim, ))
    base_point[0] = 1.
    embedded = np.concatenate([mu[None, :], np.zeros((dim - 1, dim))])

    rotmatrix, _ = np.linalg.qr(np.transpose(embedded))

    rotated_base_point = np.matmul(rotmatrix, base_point[:, None])[: ,0]
    if np.allclose(rotated_base_point, mu):
        rotsign = 1
    else:
        rotsign = -1

    # apply rotation
    samples = np.einsum('ij,...j->...i', rotmatrix, samples) * rotsign

    return samples


def sample_vMF(mu: np.ndarray, 
               kappa: float, 
               n: int, 
               rotate_func: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
               seed: Optional[int] = None) -> np.ndarray:
    """
    Generate samples from a n-dimensional von Mises-Fisher distribution
    with mu = [1, 0, ..., 0] and kappa via rejection sampling.
    Samples then have to be rotated towards the desired mean direction mu.
    
    Parameters
    ----------
    mu : np.ndarray, shape=(dim,)
        Mean direction (unit vector).
    kappa : float
        Concentration parameter.
    n : int
        Number of samples to generate.
    rotate_func : Callable[[np.ndarray, np.ndarray, int], np.ndarray]
        Function to rotate samples. Must take samples, mu, and dimension as arguments.
    seed : Optional[int], default=None
        Random seed.
        
    Returns
    -------
    samples : np.ndarray, shape=(n, dim)
        Samples from the von Mises-Fisher distribution.
    """
    random_state = _check_random_state(seed)
    size = n

    dim = mu.shape[0]
    assert dim > 1, "dim must be greater than 1"
    assert kappa > 0, "kappa must be positive"
    # check if mu is a unit vector
    assert np.isclose(np.linalg.norm(mu), 1), "mu must be a unit vector"

    dim_minus_one = dim - 1
    # calculate number of requested samples
    if size is not None:
        if not np.iterable(size):
            size = (size, )
        n_samples = np.prod(size)
    else:
        n_samples = 1

    # calculate envelope for rejection sampler (eq. 4)
    sqrt = np.sqrt(4 * kappa ** 2. + dim_minus_one ** 2)
    envelop_param = (-2 * kappa + sqrt) / dim_minus_one
    if envelop_param == 0:
        # the regular formula suffers from loss of precision for high
        # kappa. This can only be detected by checking for 0 here.
        # Workaround: expansion for sqrt variable
        envelop_param = (dim_minus_one/4 * kappa**-1.
                            - dim_minus_one**3/64 * kappa**-3.)
    # reference step 0
    node = (1. - envelop_param) / (1. + envelop_param)
    correction = (kappa * node + dim_minus_one
                    * (np.log(4) + np.log(envelop_param)
                    - 2 * np.log1p(envelop_param)))
    n_accepted = 0
    x = np.zeros((n_samples, ))
    halfdim = 0.5 * dim_minus_one
    # main loop
    while n_accepted < n_samples:
        # generate candidates acc. to reference step 1
        sym_beta = random_state.beta(halfdim, halfdim,
                                        size=n_samples - n_accepted)
        coord_x = (1 - (1 + envelop_param) * sym_beta) / (
            1 - (1 - envelop_param) * sym_beta)
        # accept or reject: reference step 2
        accept_tol = random_state.random(n_samples - n_accepted)
        criterion = (
            kappa * coord_x
            + dim_minus_one * (np.log((1 + envelop_param - coord_x
            + coord_x * envelop_param) / (1 + envelop_param)))
            - correction) > np.log(accept_tol)
        accepted_iter = np.sum(criterion)
        x[n_accepted:n_accepted + accepted_iter] = coord_x[criterion]
        n_accepted += accepted_iter

    # concatenate x and remaining coordinates: step 3
    coord_rest = _sample_uniform_direction(
        dim=dim_minus_one, 
        size=n_samples,
        random_state=random_state
        )
    coord_rest = np.einsum(
        '...,...i->...i', np.sqrt(1 - x ** 2), coord_rest)
    samples = np.concatenate([x[..., None], coord_rest], axis=1)
    # reshape output to (size, dim)
    if size is not None and len(size) > 1:
        samples = samples.reshape(size + (dim, ))

    # Rotate samples using the provided rotation function
    samples = rotate_func(samples, mu, dim)
    return samples


class TestVMFSampling(unittest.TestCase):
    """
    Unit tests for von Mises-Fisher sampling with different rotation implementations.
    """
    
    def setUp(self):
        """Set up test parameters."""
        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
    def test_rotation_equivalence_small(self):
        """Test that all rotation implementations give equivalent results for small dimensions."""
        dim = 32
        n_samples = 16
        
        # Create a random unit vector as the mean direction
        mu = np.random.randn(dim)
        mu = mu / np.linalg.norm(mu)
        
        kappa = 5.0
        
        # Generate samples using all three rotation methods with the same seed
        samples_numpy = sample_vMF(mu, kappa, n_samples, rotate_samples_numpy, seed=42)
        samples_torch = sample_vMF(mu, kappa, n_samples, rotate_samples_torch, seed=42)
        samples_scipy = sample_vMF(mu, kappa, n_samples, rotate_samples_scipy, seed=42)
        
        # Check that the results are similar (allowing for small numerical differences)
        np.testing.assert_allclose(samples_numpy, samples_torch, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(samples_numpy, samples_scipy, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(samples_torch, samples_scipy, rtol=1e-5, atol=1e-5)
        
    def test_rotation_equivalence_medium(self):
        """Test that all rotation implementations give equivalent results for medium dimensions."""
        dim = 128
        n_samples = 64
        
        # Create a random unit vector as the mean direction
        mu = np.random.randn(dim)
        mu = mu / np.linalg.norm(mu)
        
        kappa = 10.0
        
        # Generate samples using all three rotation methods with the same seed
        samples_numpy = sample_vMF(mu, kappa, n_samples, rotate_samples_numpy, seed=42)
        samples_torch = sample_vMF(mu, kappa, n_samples, rotate_samples_torch, seed=42)
        samples_scipy = sample_vMF(mu, kappa, n_samples, rotate_samples_scipy, seed=42)
        
        # Check that the results are similar (allowing for small numerical differences)
        np.testing.assert_allclose(samples_numpy, samples_torch, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(samples_numpy, samples_scipy, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(samples_torch, samples_scipy, rtol=1e-5, atol=1e-5)
        
    def test_rotation_equivalence_large(self):
        """Test that all rotation implementations give equivalent results for large dimensions."""
        dim = 256
        n_samples = 128
        
        # Create a random unit vector as the mean direction
        mu = np.random.randn(dim)
        mu = mu / np.linalg.norm(mu)
        
        kappa = 20.0
        
        # Generate samples using all three rotation methods with the same seed
        samples_numpy = sample_vMF(mu, kappa, n_samples, rotate_samples_numpy, seed=42)
        samples_torch = sample_vMF(mu, kappa, n_samples, rotate_samples_torch, seed=42)
        samples_scipy = sample_vMF(mu, kappa, n_samples, rotate_samples_scipy, seed=42)
        
        # Check that the results are similar (allowing for small numerical differences)
        np.testing.assert_allclose(samples_numpy, samples_torch, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(samples_numpy, samples_scipy, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(samples_torch, samples_scipy, rtol=1e-4, atol=1e-4)
        
    def test_performance_comparison(self):
        """Compare performance of different rotation implementations."""
        dim = 512
        n_samples = 256
        
        # Create a random unit vector as the mean direction
        mu = np.random.randn(dim)
        mu = mu / np.linalg.norm(mu)
        
        kappa = 15.0
        
        # Time each implementation
        start_time = time.time()
        _ = sample_vMF(mu, kappa, n_samples, rotate_samples_numpy, seed=42)
        numpy_time = time.time() - start_time
        
        start_time = time.time()
        _ = sample_vMF(mu, kappa, n_samples, rotate_samples_torch, seed=42)
        torch_time = time.time() - start_time
        
        start_time = time.time()
        _ = sample_vMF(mu, kappa, n_samples, rotate_samples_scipy, seed=42)
        scipy_time = time.time() - start_time
        
        print(f"\nPerformance comparison for dim={dim}, n_samples={n_samples}:")
        print(f"NumPy implementation: {numpy_time:.4f} seconds")
        print(f"Torch implementation: {torch_time:.4f} seconds")
        print(f"SciPy implementation: {scipy_time:.4f} seconds")
        
        # No assertions here, just for performance comparison


def benchmark_sampling_methods(dim_range: list[int]=(512, 1024, 2048), 
                              n_samples_range: list[int]=(256, ), 
                              kappa: int=10.0):
    """
    Benchmark the different rotation implementations for various dimensions and sample sizes.
    
    Parameters
    ----------
    dim_range : tuple or list
        Range of dimensions to benchmark.
    n_samples_range : tuple or list
        Range of sample sizes to benchmark.
    kappa : float
        Concentration parameter.
    """
    results = {}
    
    for dim in dim_range:
        results[dim] = {}
        
        # Create a random unit vector as the mean direction
        mu = np.random.randn(dim)
        mu = mu / np.linalg.norm(mu)
        
        for n_samples in n_samples_range:
            results[dim][n_samples] = {}
            
            # Time each implementation
            start_time = time.time()
            _ = sample_vMF(mu, kappa, n_samples, rotate_samples_scipy, seed=42)
            scipy_time = time.time() - start_time

            start_time = time.time()
            _ = sample_vMF(mu, kappa, n_samples, rotate_samples_numpy, seed=42)
            numpy_time = time.time() - start_time
            
            start_time = time.time()
            _ = sample_vMF(mu, kappa, n_samples, rotate_samples_torch, seed=42)
            torch_time = time.time() - start_time
                        
            results[dim][n_samples]['scipy'] = scipy_time
            results[dim][n_samples]['numpy'] = numpy_time
            results[dim][n_samples]['torch'] = torch_time
            
            print(f"dim={dim}, n_samples={n_samples}:")
            print(f"  SciPy: {scipy_time:.4f}s")
            print(f"  NumPy: {numpy_time:.4f}s")
            print(f"  Torch: {torch_time:.4f}s")
            
    return results


if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    results = benchmark_sampling_methods()
