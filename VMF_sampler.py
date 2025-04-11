import numpy as np
from tqdm import tqdm


class VMFSampler:
    """
    A class for efficient sampling from von Mises-Fisher distributions.
    Precomputes the rotation matrix for a given direction mu, allowing
    for faster sampling with different kappa values.
    """

    def __init__(self, mu):
        """
        Initialize the VMFSampler with a mean direction.

        Parameters
        ----------
        mu : array-like, shape=[dim, ]
            Mean direction (unit vector).
        """
        self.mu = np.array(mu, dtype=np.float64)
        self.dim = self.mu.shape[0]
        
        # Validate inputs
        assert self.dim > 1, "Dimension must be greater than 1"
        assert np.isclose(np.linalg.norm(self.mu), 1), "mu must be a unit vector"
        
        # Precompute rotation matrix
        self.rotmatrix, self.rotsign = self._precompute_rotation()
        
    def _precompute_rotation(self):
        """
        Precompute the rotation matrix Q and sign for the given direction mu.
        
        Returns
        -------
        rotmatrix : ndarray, shape=(dim, dim)
            Rotation matrix.
        rotsign : int
            Sign of the rotation (1 or -1).
        """
        # Base point (north pole)
        base_point = np.zeros(self.dim)
        base_point[0] = 1.
        
        # Compute rotation matrix using QR decomposition
        embedded = np.concatenate([self.mu[None, :], np.zeros((self.dim - 1, self.dim))])
        rotmatrix, _ = np.linalg.qr(np.transpose(embedded))
        
        # Determine the sign of the rotation
        if np.allclose(np.matmul(rotmatrix, base_point[:, None])[:, 0], self.mu):
            rotsign = 1
        else:
            rotsign = -1
            
        return rotmatrix, rotsign
    
    def _check_random_state(self, seed):
        """
        Check and return a random number generator.
        
        Parameters
        ----------
        seed : None or int
            Random seed.
            
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
    
    def _sample_uniform_direction(self, dim, size, random_state):
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
    
    def _rotate_samples(self, samples):
        """
        Apply the precomputed rotation to the samples.
        
        Parameters
        ----------
        samples : ndarray, shape=(n_samples, dim)
            Samples to rotate.
            
        Returns
        -------
        rotated_samples : ndarray, shape=(n_samples, dim)
            Rotated samples.
        """
        # Apply rotation using matrix multiplication
        return np.matmul(samples, self.rotmatrix.T) * self.rotsign
    
    def sample(self, kappa, n, seed=None):
        """
        Sample from the von Mises-Fisher distribution.
        
        Parameters
        ----------
        kappa : float
            Concentration parameter.
        n : int
            Number of samples to generate.
        seed : None or int, optional
            Random seed.
            
        Returns
        -------
        samples : ndarray, shape=(n, dim)
            Samples from the von Mises-Fisher distribution.
        """
        assert kappa > 0, "kappa must be positive"
        
        random_state = self._check_random_state(seed)
        size = n
        
        dim_minus_one = self.dim - 1
        
        # Calculate number of requested samples
        if size is not None:
            if not np.iterable(size):
                size = (size, )
            n_samples = np.prod(size)
        else:
            n_samples = 1
            
        # Calculate envelope for rejection sampler (eq. 4)
        sqrt = np.sqrt(4 * kappa ** 2. + dim_minus_one ** 2)
        envelop_param = (-2 * kappa + sqrt) / dim_minus_one
        
        if envelop_param == 0:
            # Handle precision loss for high kappa
            envelop_param = (dim_minus_one/4 * kappa**-1. - dim_minus_one**3/64 * kappa**-3.)
            
        # Reference step 0
        node = (1. - envelop_param) / (1. + envelop_param)
        correction = (kappa * node + dim_minus_one
                     * (np.log(4) + np.log(envelop_param)
                     - 2 * np.log1p(envelop_param)))
        
        n_accepted = 0
        x = np.zeros((n_samples, ))
        halfdim = 0.5 * dim_minus_one
        
        # Main rejection sampling loop
        with tqdm(desc="Sampling", total=n_samples) as pbar:
            while n_accepted < n_samples:
                # Generate candidates according to reference step 1
                sym_beta = random_state.beta(halfdim, halfdim,
                                            size=n_samples - n_accepted)
                coord_x = (1 - (1 + envelop_param) * sym_beta) / (
                    1 - (1 - envelop_param) * sym_beta)
                    
                # Accept or reject: reference step 2
                accept_tol = random_state.random(n_samples - n_accepted)
                criterion = (
                    kappa * coord_x
                    + dim_minus_one * (np.log((1 + envelop_param - coord_x
                    + coord_x * envelop_param) / (1 + envelop_param)))
                    - correction) > np.log(accept_tol)
                    
                accepted_iter = np.sum(criterion)
                x[n_accepted:n_accepted + accepted_iter] = coord_x[criterion]
                n_accepted += accepted_iter
                
                # Update progress bar
                pbar.update(accepted_iter)
                
        # Concatenate x and remaining coordinates: step 3
        coord_rest = self._sample_uniform_direction(
            dim=dim_minus_one, 
            size=n_samples,
            random_state=random_state
        )
        
        coord_rest = np.einsum(
            '...,...i->...i', np.sqrt(1 - x ** 2), coord_rest)
            
        samples = np.concatenate([x[..., None], coord_rest], axis=1)
        
        # Reshape output to (size, dim)
        if size is not None and len(size) > 1:
            samples = samples.reshape(size + (self.dim, ))
            
        # Apply the precomputed rotation
        samples = self._rotate_samples(samples)
        
        return samples


# Example usage
if __name__ == "__main__":
    import time
    
    # Set up parameters
    dim = 1500
    mu = np.zeros(dim)
    mu[-1] = 1.0  # Mean direction (unit vector)
    mu = mu / np.linalg.norm(mu)  # Ensure it's a unit vector
    
    kappa = 10.0  # Concentration parameter
    n = 1000  # Number of samples
    
    # Initialize the sampler
    print("Initializing VMF sampler...")
    start = time.time()
    vmf_sampler = VMFSampler(mu)
    init_time = time.time() - start
    print(f"Initialization took {init_time:.4f} seconds")
    
    # Sample with the same mu but different kappa values
    print("\nSampling with kappa =", kappa)
    start = time.time()
    samples1 = vmf_sampler.sample(kappa, n)
    sample_time1 = time.time() - start
    print(f"Sampling took {sample_time1:.4f} seconds")
    
    # Sample again with a different kappa
    new_kappa = 20.0
    print(f"\nSampling with kappa = {new_kappa}")
    start = time.time()
    samples2 = vmf_sampler.sample(new_kappa, n)
    sample_time2 = time.time() - start
    print(f"Sampling took {sample_time2:.4f} seconds")
    
    # Compare with the original implementation
    import scipy_implementation
    print("\nComparing with original implementation...")
    start = time.time()
    samples_orig = scipy_implementation.sample_vMF(mu, kappa, n)
    orig_time = time.time() - start
    print(f"Original implementation took {orig_time:.4f} seconds")
    
    # Print speedup
    total_optimized_time = init_time + sample_time1 + sample_time2
    print(f"\nSpeedup for two samplings: {orig_time*2/total_optimized_time:.2f}x")
