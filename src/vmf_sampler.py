"""
Von Mises-Fisher distribution sampler.

This module contains the core vMF sampling class with support for multiple backends
(NumPy, PyTorch, SciPy) and efficient rotation algorithms.
"""

import numpy as np
import torch
from enum import Enum


class Implementation(Enum):
    """Enum for different vMF sampling implementations."""
    SCIPY = "scipy"
    NUMPY = "numpy" 
    TORCH = "torch"


class vMF:
    """
    Von Mises-Fisher distribution sampler.
    
    This class implements efficient sampling from the von Mises-Fisher (vMF) distribution
    using rejection sampling with rotation. The vMF distribution is a probability distribution
    on the unit hypersphere, parameterized by a mean direction μ and concentration parameter κ.
    
    The implementation supports multiple backends (NumPy, PyTorch, SciPy) and can handle
    high-dimensional sampling efficiently on both CPU and GPU.

    The scipy version is a copy of the implementation found in the SciPy library. The numpy and torch versions are custom implementations are based on the SciPy version and meant to make them faster.

    Parameters
    ----------
    dim : int
        Dimension of the hypersphere (dimension of the space).
    kappa : float
        Concentration parameter (κ > 0). Higher values indicate more concentration
        around the mean direction.
    seed : int or None, optional
        Random seed for reproducibility. If None, uses default random state.
    device : str, torch.device or None, optional
        Device for PyTorch computations. If None, auto-selects CUDA if available.
    dtype : torch.dtype or None, optional
        Data type for PyTorch tensors. If None, defaults to torch.float32.
    use_scipy_implementation : bool, optional
        Whether to use SciPy-style rotation implementation. Default is False.
        
    Attributes
    ----------
    dim : int
        Dimension of the space.
    kappa : float
        Concentration parameter.
    mu : np.ndarray or torch.Tensor or None
        Mean direction (unit vector).
    device : torch.device
        Computing device.
    dtype : torch.dtype
        Data type for computations.
    rotmatrix : np.ndarray or torch.Tensor or None
        Rotation matrix for transforming samples.
    rotsign : int or None
        Sign correction for rotation.
        
    Examples
    --------
    >>> # Basic usage
    >>> sampler = vMF(dim=3, kappa=10.0, seed=42)
    >>> mu = np.array([1.0, 0.0, 0.0])
    >>> samples = sampler.sample(1000, mu=mu)
    >>> print(samples.shape)  # (1000, 3)
    
    >>> # Using PyTorch on GPU
    >>> sampler = vMF(dim=512, kappa=5.0, device='cuda')
    >>> mu = torch.randn(512)
    >>> mu = mu / torch.norm(mu)  # normalize to unit vector
    >>> samples = sampler.sample(10000, mu=mu)
    """
    
    def __init__(self, dim: int, kappa: float, seed: int|None = None, 
                 device: str|torch.device|None = None, dtype: torch.dtype|None = None, 
                 use_scipy_implementation: bool = False):

        self.dim = dim
        self.kappa = kappa
        self.seed = seed
        if device is not None:
            if isinstance(device, str):
                self.device = torch.device(device)
            else:
                self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dtype = dtype if dtype is not None else torch.float32

        self.use_scipy_implementation = use_scipy_implementation

        self.rotmatrix = None
        self.rotsign = None
        self.mu = None
        self.random_state = self._random_state(seed)

    def set_seed(self, seed: int | None):
        """
        Set the random seed for reproducibility.
        
        Parameters
        ----------
        seed : int or None
            Random seed. If None, use the default random state.
        """
        if seed is None:
            self.random_state = np.random
        else:
            self.random_state = np.random.RandomState(seed)

    def _verify_mu(self, mu: np.ndarray|torch.Tensor):
        """
        Verify that mu is a unit vector.
        
        Parameters
        ----------
        mu : np.ndarray or torch.Tensor
            Mean direction (unit vector).
            
        Raises
        ------
        ValueError
            If mu is not a unit vector.
        """
        if isinstance(mu, torch.Tensor):
            if mu.dim() != 1 or mu.shape[0] != self.dim:
                raise ValueError(f"mu must be a 1D tensor of shape ({self.dim},)")

            if not torch.isclose(torch.norm(mu), torch.tensor(1.0)):
                raise ValueError("mu must be a unit vector")
        else:
            if mu.ndim != 1 or mu.shape[0] != self.dim:
                raise ValueError(f"mu must be a 1D array of shape ({self.dim},)")

            if not np.isclose(np.linalg.norm(mu), 1.0):
                raise ValueError("mu must be a unit vector")

    def _compute_rotation_matrix_torch(self):
        """Compute rotation matrix using PyTorch."""
        if not isinstance(self.mu, torch.Tensor):
            raise ValueError("mu must be a torch tensor")

        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                base_point = torch.zeros((self.dim, ), device=self.device, dtype=self.dtype)
                base_point[0] = 1.
                embedded = torch.cat([self.mu[None, :], torch.zeros((self.dim - 1, self.dim), device=self.device, dtype=self.dtype)], dim=0)
                embedded = torch.transpose(embedded, 0, 1)

            self.rotmatrix, _ = torch.linalg.qr(embedded)

            # check if the rotation is correct
            rotated_base_point = torch.mv(self.rotmatrix, base_point)
            if torch.allclose(rotated_base_point, self.mu):
                self.rotsign = 1
            else:
                self.rotsign = -1

    def _compute_rotation_matrix_numpy(self):
        """Compute rotation matrix using NumPy."""
        if not isinstance(self.mu, np.ndarray):
            raise ValueError("mu must be a numpy array")

        TORCH_DTYPE_TO_NUMPY_DTYPE = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.float16: np.float16,
        }

        dtype = TORCH_DTYPE_TO_NUMPY_DTYPE.get(self.dtype, np.float32)

        base_point = np.zeros((self.dim, ), dtype=dtype)
        base_point[0] = 1.
        embedded = np.concatenate([self.mu[None, :], np.zeros((self.dim - 1, self.dim), dtype=dtype)])

        self.rotmatrix, _ = np.linalg.qr(np.transpose(embedded))

        rotated_base_point = np.matmul(self.rotmatrix, base_point[:, None])[:, 0]
        if np.allclose(rotated_base_point, self.mu):
            self.rotsign = 1
        else:
            self.rotsign = -1

    def _compute_rotation_matrix_scipy(self):
        """Compute rotation matrix using SciPy."""
        if not isinstance(self.mu, np.ndarray):
            raise ValueError("mu must be a numpy array")

        base_point = np.zeros((self.dim, ))
        base_point[0] = 1.
        embedded = np.concatenate([self.mu[None, :], np.zeros((self.dim - 1, self.dim))])

        self.rotmatrix, _ = np.linalg.qr(np.transpose(embedded))

        rotated_base_point = np.matmul(self.rotmatrix, base_point[:, None])[:, 0]
        if np.allclose(rotated_base_point, self.mu):
            self.rotsign = 1
        else:
            self.rotsign = -1

    def compute_rotation_matrix(self):
        """
        Initialize the rotation matrix and sign based on the mean direction mu.
        """
        if not isinstance(self.mu, (np.ndarray, torch.Tensor)):
            raise ValueError("mu must be a numpy array or torch tensor")

        if isinstance(self.mu, torch.Tensor):
            self._compute_rotation_matrix_torch()
        elif self.use_scipy_implementation:
            self._compute_rotation_matrix_scipy()
        else:
            self._compute_rotation_matrix_numpy()

    def _rotate_samples_torch(self, samples: torch.Tensor) -> torch.Tensor:
        """Rotate samples using PyTorch."""
        if not isinstance(samples, torch.Tensor):
            raise ValueError("samples must be a torch tensor")

        if not isinstance(self.rotmatrix, torch.Tensor):
            self._compute_rotation_matrix_torch()

        assert isinstance(self.rotsign, int)
        assert isinstance(self.rotmatrix, torch.Tensor)

        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                return torch.matmul(self.rotmatrix, samples.T).T * self.rotsign
            
    def _rotate_samples_numpy(self, samples: np.ndarray) -> np.ndarray:
        """Rotate samples using NumPy."""
        if not isinstance(samples, np.ndarray):
            raise ValueError("samples must be a numpy array")

        if not isinstance(self.rotmatrix, np.ndarray):
            self._compute_rotation_matrix_numpy()

        assert isinstance(self.rotsign, int)
        assert isinstance(self.rotmatrix, np.ndarray)

        return np.matmul(self.rotmatrix, samples.T).T * self.rotsign

    def _rotate_samples_scipy(self, samples: np.ndarray) -> np.ndarray:
        """Rotate samples using SciPy einsum."""
        if not isinstance(self.rotmatrix, np.ndarray):
            self._compute_rotation_matrix_scipy()

        assert isinstance(self.rotsign, int)
        assert isinstance(self.rotmatrix, np.ndarray)

        return np.einsum('ij,...j->...i', self.rotmatrix, samples) * self.rotsign

    def rotate_samples(self, samples: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        Rotate samples to align with the mean direction mu.
        
        Parameters
        ----------
        samples : np.ndarray or torch.Tensor
            Samples to rotate.
            
        Returns
        -------
        np.ndarray or torch.Tensor
            Rotated samples.
        """
        if not isinstance(samples, (np.ndarray, torch.Tensor)):
            raise ValueError("samples must be a numpy array or torch tensor")

        # Decide rotation method based on mu type, not samples type
        if isinstance(self.mu, torch.Tensor):
            # Convert samples to torch if needed
            if isinstance(samples, np.ndarray):
                samples_torch = torch.from_numpy(samples).to(self.mu.device).to(self.mu.dtype)
                result = self._rotate_samples_torch(samples_torch)
                return result.cpu().numpy()  # Convert back to numpy
            else:
                return self._rotate_samples_torch(samples)
        else:
            # mu is numpy array
            if isinstance(samples, torch.Tensor):
                samples_numpy = samples.cpu().numpy()
            else:
                samples_numpy = samples
            
            if not self.use_scipy_implementation:
                return self._rotate_samples_numpy(samples_numpy)
            else:
                return self._rotate_samples_scipy(samples_numpy)

    def set_mu(self, mu: np.ndarray | torch.Tensor):
        """Set the mean direction and recompute rotation matrix."""
        self.mu = mu
        self.rotmatrix = None
        self.rotsign = None
        self.compute_rotation_matrix()

    def set_kappa(self, kappa: float):
        """Set the concentration parameter."""
        self.kappa = kappa

    def _random_state(self, seed: int | None):
        """Initialize random number generator."""
        assert seed is None or isinstance(seed, (int, np.integer)), (
            f"seed must be None or an integer, got {type(seed)}"
        )
        if seed is None:
            # Use the default random state
            return np.random.default_rng()
        else:
            return np.random.default_rng(seed)
            
    def _sample_uniform_direction(self, dim: int, size: int) -> np.ndarray:
        """
        Generate uniform directions on the unit sphere.
        
        Parameters
        ----------
        dim : int
            Dimension of the sphere.
        size : int
            Number of samples.
            
        Returns
        -------
        samples : ndarray, shape=(size, dim)
            Uniform samples from the unit sphere.
        """
        samples = self.random_state.standard_normal((size, dim))
        samples /= np.linalg.norm(samples, axis=-1, keepdims=True)
        return samples

    def sample(self, num_samples: int, mu: np.ndarray | torch.Tensor | None = None, 
               kappa: float | None = None) -> np.ndarray | torch.Tensor:
        """
        Sample from the von Mises-Fisher distribution.
        
        Generates samples from the vMF distribution using rejection sampling with
        efficient rotation. The implementation is based on Ulrich (1984) and Wood (1994).
        
        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        mu : np.ndarray or torch.Tensor or None, optional
            Mean direction (unit vector). If provided, overrides the current mu.
            Must be a unit vector of dimension `self.dim`.
        kappa : float or None, optional
            Concentration parameter. If provided, overrides the current kappa.
            Must be positive.
            
        Returns
        -------
        samples : np.ndarray or torch.Tensor
            Generated samples from the vMF distribution. Shape is (num_samples, dim).
            Return type matches the type of mu (or numpy if mu is not provided).
            
        Raises
        ------
        ValueError
            If mu is not provided and not set during initialization.
            If mu is not a unit vector.
            If kappa is not positive.
            If mu type is incompatible with existing mu.
            
        Examples
        --------
        >>> sampler = vMF(dim=3, kappa=5.0)
        >>> mu = np.array([0.0, 0.0, 1.0])  # point north
        >>> samples = sampler.sample(1000, mu=mu)
        >>> print(f"Mean direction: {np.mean(samples, axis=0)}")
        >>> print(f"Sample shape: {samples.shape}")
        
        """
        # Sample from the vMF distribution
        if mu is not None:
            self.set_mu(mu)
        elif self.mu is None:
            raise ValueError("Must provide mu either as an argument or during initialization.")

        assert self.mu is not None, "mu must be set before sampling"

        if kappa is not None:
            self.set_kappa(kappa)
        assert self.kappa > 0, "kappa must be positive"

        dim = self.mu.shape[0]
        assert dim > 1, "dim must be greater than 1"
        assert self.kappa > 0, "kappa must be positive"
        # check if mu is a unit vector
        assert np.isclose(np.linalg.norm(self.mu), 1), "mu must be a unit vector"

        dim_minus_one = dim - 1
        # calculate number of requested samples
        if num_samples is not None:
            n_samples = np.prod(num_samples) if np.iterable(num_samples) else num_samples
        else:
            n_samples = 1

        # calculate envelope for rejection sampler (eq. 4)
        sqrt = np.sqrt(4 * self.kappa ** 2. + dim_minus_one ** 2)
        envelop_param = (-2 * self.kappa + sqrt) / dim_minus_one
        if envelop_param == 0:
            # the regular formula suffers from loss of precision for high
            # kappa. This can only be detected by checking for 0 here.
            # Workaround: expansion for sqrt variable
            envelop_param = (dim_minus_one/4 * self.kappa**-1.
                                - dim_minus_one**3/64 * self.kappa**-3.)
        # reference step 0
        node = (1. - envelop_param) / (1. + envelop_param)
        correction = (self.kappa * node + dim_minus_one
                        * (np.log(4) + np.log(envelop_param)
                        - 2 * np.log1p(envelop_param)))
        n_accepted = 0
        x = np.zeros((n_samples, ))
        halfdim = 0.5 * dim_minus_one
        # main loop
        while n_accepted < n_samples:
            # generate candidates acc. to reference step 1
            sym_beta = self.random_state.beta(halfdim, halfdim,
                                            size=n_samples - n_accepted)
            coord_x = (1 - (1 + envelop_param) * sym_beta) / (
                1 - (1 - envelop_param) * sym_beta)
            # accept or reject: reference step 2
            accept_tol = self.random_state.random(n_samples - n_accepted)
            criterion = (
                self.kappa * coord_x
                + dim_minus_one * (np.log((1 + envelop_param - coord_x
                + coord_x * envelop_param) / (1 + envelop_param)))
                - correction) > np.log(accept_tol)
            accepted_iter = np.sum(criterion)
            x[n_accepted:n_accepted + accepted_iter] = coord_x[criterion]
            n_accepted += accepted_iter

        # concatenate x and remaining coordinates: step 3
        coord_rest = self._sample_uniform_direction(
            dim=dim_minus_one, 
            size=int(n_samples)
            )
        coord_rest = np.einsum(
            '...,...i->...i', np.sqrt(1 - x ** 2), coord_rest)
        samples = np.concatenate([x[..., None], coord_rest], axis=1)
        
        # reshape output to (size, dim)
        if isinstance(num_samples, (list, tuple)) and len(num_samples) > 1:
            samples = samples.reshape(tuple(num_samples) + (dim, ))

        # Rotate samples using the class rotation method
        samples = self.rotate_samples(samples)
        return samples