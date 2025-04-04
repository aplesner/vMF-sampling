import time

import numpy as np
from tqdm import tqdm

import line_profiler

def _check_random_state(seed):
    assert seed is None or isinstance(seed, (int, np.integer)), (
        f"seed must be None or an integer, got {type(seed)}"
    )
    if seed is None:
        # Use the default random state
        return np.random
    else:
        return np.random.RandomState(seed)
    


def _sample_uniform_direction(dim, size, random_state):
    """
    Private method to generate uniform directions
    Reference: Marsaglia, G. (1972). "Choosing a Point from the Surface of a
               Sphere". Annals of Mathematical Statistics. 43 (2): 645-646.
    """
    samples_shape = np.append(size, dim)
    samples = random_state.standard_normal(samples_shape)
    samples /= np.linalg.norm(samples, axis=-1, keepdims=True)
    return samples

@line_profiler.profile
def _rotate_samples(samples, mu, dim):
    """A QR decomposition is used to find the rotation that maps the
    north pole (1, 0,...,0) to the vector mu. This rotation is then
    applied to all samples.

    Parameters
    ----------
    samples: array_like, shape = [..., n]
    mu : array-like, shape=[n, ]
        Point to parametrise the rotation.

    Returns
    -------
    samples : rotated samples

    """
    base_point = np.zeros((dim, ))
    base_point[0] = 1.
    embedded = np.concatenate([mu[None, :], np.zeros((dim - 1, dim))])
    rotmatrix, _ = np.linalg.qr(np.transpose(embedded)) # This line is responsible for 74% of the time
    if np.allclose(np.matmul(rotmatrix, base_point[:, None])[:, 0], mu):
        rotsign = 1
    else:
        rotsign = -1

    # apply rotation (this is a matrix multiplication)
    # samples_new = np.einsum('ij,...j->...i', rotmatrix, samples) * rotsign # This line is responsible for 26% of the time
    print(rotmatrix.shape, samples.shape)
    samples2 = np.matvec(rotmatrix, samples) * rotsign
    # check if the rotation worked
    # assert np.allclose(samples2, samples_new), "Rotation failed"
    return samples2

# @line_profiler.profile
def sample_vMF(mu: np.array, kappa: float, n: int) -> np.ndarray:
    """
    Generate samples from a n-dimensional von Mises-Fisher distribution
    with mu = [1, 0, ..., 0] and kappa via rejection sampling.
    Samples then have to be rotated towards the desired mean direction mu.
    Reference: https://doi.org/10.1080/03610919408813161
    """

    random_state = _check_random_state(0)
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
        # https://www.wolframalpha.com/input?i=sqrt%284*x%5E2%2Bd%5E2%29
        # e = (-2 * k + sqrt(k**2 + d**2)) / d
        #   ~ (-2 * k + 2 * k + d**2/(4 * k) - d**4/(64 * k**3)) / d
        #   = d/(4 * k) - d**3/(64 * k**3)
        envelop_param = (dim_minus_one/4 * kappa**-1.
                            - dim_minus_one**3/64 * kappa**-3.)
    # reference step 0
    node = (1. - envelop_param) / (1. + envelop_param)
    # t = ln(1 - ((1-x)/(1+x))**2)
    #   = ln(4 * x / (1+x)**2)
    #   = ln(4) + ln(x) - 2*log1p(x)
    correction = (kappa * node + dim_minus_one
                    * (np.log(4) + np.log(envelop_param)
                    - 2 * np.log1p(envelop_param)))
    n_accepted = 0
    x = np.zeros((n_samples, ))
    halfdim = 0.5 * dim_minus_one
    # main loop
    with tqdm(desc="Sampling") as pbar:
        while n_accepted < n_samples:
            # generate candidates acc. to reference step 1
            sym_beta = random_state.beta(halfdim, halfdim,
                                            size=n_samples - n_accepted)
            coord_x = (1 - (1 + envelop_param) * sym_beta) / (
                1 - (1 - envelop_param) * sym_beta)
            # accept or reject: reference step 2
            # reformulation for numerical stability:
            # t = ln(1 - (1-x)/(1+x) * y)
            #   = ln((1 + x - y +x*y)/(1 +x))
            accept_tol = random_state.random(n_samples - n_accepted)
            criterion = (
                kappa * coord_x
                + dim_minus_one * (np.log((1 + envelop_param - coord_x
                + coord_x * envelop_param) / (1 + envelop_param)))
                - correction) > np.log(accept_tol)
            accepted_iter = np.sum(criterion)
            x[n_accepted:n_accepted + accepted_iter] = coord_x[criterion]
            n_accepted += accepted_iter

            # update progress bar
            pbar.update(1)

    # concatenate x and remaining coordinates: step 3
    coord_rest = _sample_uniform_direction(
        dim=dim_minus_one, 
        size=n_accepted,
        random_state=random_state
        )
    coord_rest = np.einsum(
        '...,...i->...i', np.sqrt(1 - x ** 2), coord_rest)
    samples = np.concatenate([x[..., None], coord_rest], axis=1)
    # reshape output to (size, dim)
    if size is not None:
        samples = samples.reshape(size + (dim, ))
    else:
        samples = np.squeeze(samples)

    samples = _rotate_samples(samples, mu, dim) # This line is responsible for 99% of the time
    return samples

def profile_implementation():
    """
    Profile the implementation of the von Mises-Fisher sampling methods.
    """

    mu = np.zeros(1500)
    mu[-1] = 1.0  # Mean direction (unit vector)
    kappa = 10.0  # Concentration parameter
    n = 1000  # Number of samples
    # pr = cProfile.Profile(subcalls=False, builtins=False)
    # pr.enable()
    
    sample_vMF(mu, kappa, n)

    # pr.disable()
    # pr.dump_stats('restats')

    # # Print the profiling results
    # p = pstats.Stats('restats')
    # p.strip_dirs().sort_stats(SortKey.LINE).print_stats()
    # # Profile the scipy implementation
    
if __name__ == "__main__":
    profile_implementation()
