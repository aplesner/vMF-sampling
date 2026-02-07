from __future__ import annotations

import numpy as np

from .vmf_sampler import vMFSampler

DTYPES = [np.float16, np.float32, np.float64]

class NumpyvMFHH(vMFSampler):
    def __init__(
        self,
        dim: int,
        mu: np.ndarray | None = None,
        kappa: float = 10.0,
        seed: int | None = None,
        rotation_needed: bool = True,
        dtype: np.dtype | None = None,
        inplace: bool = True,
    ) -> None:
        if dtype is not None:
            if dtype not in DTYPES:
                raise ValueError(f"dtype must be one of {DTYPES}, got {dtype}")
            self.dtype = dtype
        else:
            self.dtype = np.float64

        super().__init__(dim, mu=mu, kappa=kappa, seed=seed, rotation_needed=rotation_needed)
        self.random_state = np.random.default_rng(seed)
        self.inplace = inplace

    def _verify_mu(self, mu: np.ndarray) -> None:
        if not isinstance(mu, np.ndarray):
            raise ValueError("mu must be a numpy array")
        if not np.isclose(np.linalg.norm(mu), 1.0):
            raise ValueError("mu must be a unit vector")

    def _default_mu(self) -> np.ndarray:
        mu = np.zeros(self.dim, dtype=np.float64)
        mu[0] = 1.0
        return mu

    def _rotation_needed(self, mu: np.ndarray) -> bool:
        return not np.isclose(mu[0], 1.0)

    def _rotate_householder(self, S: np.ndarray) -> np.ndarray:
        """
        Applies the shortest-path rotation mapping x -> y to set S.
        S is an array of shape (n_samples, n_dims).
        """
        x = np.zeros((self.dim,), dtype=self.mu.dtype)
        x[0] = 1.0
        y = self.mu
        # 1. First reflection: x -> -x
        u1 = x / np.linalg.norm(x)
        
        # 2. Second reflection: -x -> y
        v2 = y - (-x)
        u2 = v2 / np.linalg.norm(v2)
        
        # Vectorized application: H(s) = s - 2(u·s)u
        # We apply H1 then H2
        S_temp = S - 2 * np.outer(S @ u1, u1)
        S_rotated = S_temp - 2 * np.outer(S_temp @ u2, u2)
        
        return S_rotated

    def _rotate_householder_inplace(self, S: np.ndarray) -> np.ndarray:
        """
        Applies the shortest-path rotation mapping x -> y to set S.
        S is an array of shape (n_samples, n_dims).
        """
        assert S.ndim == 2
        x = np.zeros((self.dim,), dtype=self.mu.dtype)
        x[0] = 1.0
        # 1. First reflection: x -> -x
        u1 = x

        # 2. Second reflection: -x -> y
        if np.allclose(x, -self.mu):
            u2 = np.zeros_like(x)
            u2[1] = 1.0
        else:
            v2 = self.mu - (-x)
            u2 = v2 / np.linalg.norm(v2)
        
        # Vectorized application: H(s) = s - 2(u·s)u
        # We apply H1 then H2
        S_outer = np.empty_like(S)

        S_dot = S[:, 0].copy()  # We can use the structure of x to make this faster (skipping the dot product)
        np.multiply(2, S_dot, out=S_dot)
        np.outer(S_dot, u1, out=S_outer)
        S -= S_outer
        np.dot(S, u2, out=S_dot)
        np.multiply(2, S_dot, out=S_dot)
        np.outer(S_dot, u2, out=S_outer)
        return S - S_outer

    def _rotate_samples(self, samples: np.ndarray) -> np.ndarray:
        if self.inplace:
            return self._rotate_householder_inplace(samples.copy())
        else:
            return self._rotate_householder(samples)

    def _sample_uniform_direction(self, dim: int, size: int) -> np.ndarray:
        samples = self.random_state.standard_normal((size, dim))
        samples /= np.linalg.norm(samples, axis=-1, keepdims=True)
        return samples

    def _sample(self, num_samples: int | tuple[int, ...]) -> np.ndarray:
        dim = self.dim
        dim_minus_one = dim - 1
        n_samples = np.prod(num_samples) if np.iterable(num_samples) else num_samples

        sqrt = np.sqrt(4 * self.kappa**2 + dim_minus_one**2)
        envelop_param = (-2 * self.kappa + sqrt) / dim_minus_one
        if envelop_param == 0:
            envelop_param = (dim_minus_one / 4 * self.kappa**-1 - dim_minus_one**3 / 64 * self.kappa**-3)

        node = (1.0 - envelop_param) / (1.0 + envelop_param)
        correction = self.kappa * node + dim_minus_one * (
            np.log(4) + np.log(envelop_param) - 2 * np.log1p(envelop_param)
        )

        n_accepted = 0
        x = np.zeros((n_samples,))
        halfdim = 0.5 * dim_minus_one

        while n_accepted < n_samples:
            remaining = n_samples - n_accepted
            sym_beta = self.random_state.beta(halfdim, halfdim, size=remaining)
            coord_x = (1 - (1 + envelop_param) * sym_beta) / (1 - (1 - envelop_param) * sym_beta)
            accept_tol = self.random_state.random(remaining)
            criterion = (
                self.kappa * coord_x
                + dim_minus_one
                * (
                    np.log((1 + envelop_param - coord_x + coord_x * envelop_param) / (1 + envelop_param))
                )
                - correction
                > np.log(accept_tol)
            )
            accepted_iter = int(np.sum(criterion))
            x[n_accepted : n_accepted + accepted_iter] = coord_x[criterion]
            n_accepted += accepted_iter

        coord_rest = self._sample_uniform_direction(dim=dim_minus_one, size=int(n_samples))
        coord_rest = np.einsum("...,...i->...i", np.sqrt(1 - x**2), coord_rest)
        samples = np.concatenate([x[..., None], coord_rest], axis=1)

        if isinstance(num_samples, (list, tuple)) and len(num_samples) > 1:
            samples = samples.reshape(tuple(num_samples) + (dim,))

        if self._rotation_needed(self.mu):
            samples = self._rotate_samples(samples)
        return samples
