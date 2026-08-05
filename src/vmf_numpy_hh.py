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
        make_copy: bool = False,
    ) -> None:
        if dtype is not None:
            if dtype not in DTYPES:
                raise ValueError(f"dtype must be one of {DTYPES}, got {dtype}")
            self.dtype = dtype
        else:
            self.dtype = np.float64

        super().__init__(dim, mu=mu, kappa=kappa, seed=seed, rotation_needed=rotation_needed)
        # PCG64DXSM is the modern, statistically stronger PCG64 variant.  Keep
        # the generator local to this sampler so instances do not share state.
        self.random_state = np.random.Generator(np.random.PCG64DXSM(seed))
        self.inplace = inplace
        self.make_copy = make_copy

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
        Applies a single Householder reflection mapping e1 -> mu to set S.
        S is an array of shape (n_samples, n_dims).

        vMF is rotationally symmetric about mu, so orientation preservation
        (det = +1) is unnecessary; a single reflection (det = -1) suffices
        and is half the work of the previous two-reflection rotation.
        """
        x = np.zeros((self.dim,), dtype=self.mu.dtype)
        x[0] = 1.0
        v = x - self.mu
        u = v / np.linalg.norm(v)

        # Vectorized application: H(s) = s - 2(u·s)u
        return S - 2 * np.outer(S @ u, u)

    def _rotate_householder_inplace(self, S: np.ndarray) -> np.ndarray:
        """
        Applies a single Householder reflection mapping e1 -> mu to set S,
        in place. S is an array of shape (n_samples, n_dims).

        vMF is rotationally symmetric about mu, so orientation preservation
        (det = +1) is unnecessary; a single reflection (det = -1) suffices
        and is half the work of the previous two-reflection rotation.
        """
        assert S.ndim == 2
        x = np.zeros((self.dim,), dtype=self.mu.dtype)
        x[0] = 1.0
        v = x - self.mu
        u = v / np.linalg.norm(v)

        # Vectorized application: H(s) = s - 2(u·s)u
        S_dot = np.empty(S.shape[0], dtype=S.dtype)
        np.dot(S, u, out=S_dot)
        np.multiply(2, S_dot, out=S_dot)
        S_outer = np.empty_like(S)
        np.outer(S_dot, u, out=S_outer)
        np.subtract(S, S_outer, out=S)
        return S

    def _rotate_samples(self, samples: np.ndarray) -> np.ndarray:
        if self.inplace:
            return self._rotate_householder_inplace(samples.copy() if self.make_copy else samples)
        else:
            return self._rotate_householder(samples)

    def _sample_uniform_direction(self, dim: int, size: int) -> np.ndarray:
        samples = self.random_state.standard_normal((size, dim))
        samples /= np.linalg.norm(samples, axis=-1, keepdims=True)
        return samples

    def _sample_dim2(self, num_samples: int | tuple[int, ...], n_samples: int) -> np.ndarray:
        """Closed-form: vMF on the circle reduces to the von Mises distribution."""
        mean_angle = np.arctan2(self.mu[1], self.mu[0])
        angles = self.random_state.vonmises(mean_angle, self.kappa, size=n_samples)
        samples = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        if isinstance(num_samples, (list, tuple)) and len(num_samples) > 1:
            samples = samples.reshape(tuple(num_samples) + (2,))
        return samples

    def _sample_dim3(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Closed-form inverse-CDF for the polar coordinate about the north pole."""
        tiny = 1e-12
        u = self.random_state.random(n_samples)
        u = np.clip(u, tiny, None)
        x = 1.0 + np.log(u + (1.0 - u) * np.exp(-2.0 * self.kappa)) / self.kappa
        t = np.sqrt(np.clip(1.0 - x**2, 0.0, None))
        circ = self._sample_uniform_direction(dim=2, size=n_samples)
        coord_rest = t[..., None] * circ
        return x, coord_rest

    def _sample(self, num_samples: int | tuple[int, ...]) -> np.ndarray:
        dim = self.dim
        dim_minus_one = dim - 1
        n_samples = np.prod(num_samples) if np.iterable(num_samples) else num_samples

        if dim == 2:
            return self._sample_dim2(num_samples, int(n_samples))

        if dim == 3:
            x, coord_rest = self._sample_dim3(int(n_samples))
        else:
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

        if self._rotation_needed(self.mu):
            samples = self._rotate_samples(samples)

        if isinstance(num_samples, (list, tuple)) and len(num_samples) > 1:
            samples = samples.reshape(tuple(num_samples) + (dim,))
        return samples
