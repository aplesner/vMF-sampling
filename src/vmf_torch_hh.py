from __future__ import annotations

import math

import torch

from .vmf_sampler import vMFSampler

DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]

class TorchvMFHH(vMFSampler):
    def __init__(
        self,
        dim: int,
        mu: torch.Tensor | None = None,
        kappa: float = 10.0,
        seed: int | None = None,
        rotation_needed: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        inplace: bool = True,
    ) -> None:
        if dtype is not None:
            if dtype not in DTYPES:
                raise ValueError(f"dtype must be one of {DTYPES}, got {dtype}")
            self.dtype = dtype
        else:
            self.dtype = torch.float32

        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype if dtype is not None else torch.float32
        if seed is not None:
            torch.manual_seed(seed)

        super().__init__(dim, mu=mu, kappa=kappa, seed=seed, rotation_needed=rotation_needed)
        self.inplace = inplace

    def _verify_mu(self, mu: torch.Tensor) -> None:
        if not isinstance(mu, torch.Tensor):
            raise ValueError("mu must be a torch tensor")
        target = torch.tensor(1.0, device=mu.device, dtype=mu.dtype)
        if not torch.isclose(mu.norm(), target):
            raise ValueError("mu must be a unit vector")

    def _default_mu(self) -> torch.Tensor:
        mu = torch.zeros((self.dim,), device=self.device, dtype=self.dtype)
        mu[0] = 1.0
        return mu

    def _rotation_needed(self, mu: torch.Tensor) -> bool:
        return not torch.isclose(mu[0], torch.tensor(1.0, device=mu.device, dtype=mu.dtype))

    def _on_mu_updated(self) -> None:
        if isinstance(self.mu, torch.Tensor):
            self.device = self.mu.device
            self.dtype = self.mu.dtype

    def _rotate_householder(self, S: torch.Tensor) -> torch.Tensor:
        """
        Applies the shortest-path rotation mapping x -> y to set S.
        S is an array of shape (n_samples, n_dims).
        """
        x = torch.zeros((self.dim,), device=self.device, dtype=self.mu.dtype)
        x[0] = 1.0
        y = self.mu
        # 1. First reflection: x -> -x
        u1 = x / torch.linalg.norm(x)

        # 2. Second reflection: -x -> y
        v2 = y - (-x)
        u2 = v2 / torch.linalg.norm(v2)

        # Vectorized application: H(s) = s - 2(u·s)u
        # We apply H1 then H2
        S_temp = S - 2 * torch.outer(S @ u1, u1)
        S_rotated = S_temp - 2 * torch.outer(S_temp @ u2, u2)

        return S_rotated

    def _rotate_householder_inplace(self, S: torch.Tensor) -> torch.Tensor:
        """
        Applies the shortest-path rotation mapping x -> y to set S.
        S is an array of shape (n_samples, n_dims).
        """
        assert S.ndim == 2
        x = torch.zeros((self.dim,), device=self.device, dtype=self.mu.dtype)
        x[0] = 1.0
        # 1. First reflection: x -> -x
        u1 = x

        # 2. Second reflection: -x -> y
        if torch.allclose(x, -self.mu):
            u2 = torch.zeros_like(x)
            u2[1] = 1.0
        else:
            v2 = self.mu - (-x)
            u2 = v2 / torch.linalg.norm(v2)

        # Vectorized application: H(s) = s - 2(u·s)u
        # We apply H1 then H2
        S_dot = S[:, 0].clone()
        S_dot.mul_(2)
        S -= torch.outer(S_dot, u1)
        S_dot = torch.mv(S, u2)
        S_dot.mul_(2)
        return S - torch.outer(S_dot, u2)

    def _rotate_samples(self, samples: torch.Tensor) -> torch.Tensor:
        if self.inplace:
            return self._rotate_householder_inplace(samples.clone())
        return self._rotate_householder(samples)

    def _sample_uniform_direction(self, dim: int, size: int) -> torch.Tensor:
        samples = torch.randn((size, dim), device=self.device, dtype=self.dtype)
        samples = samples / samples.norm(dim=-1, keepdim=True)
        return samples

    def _sample(self, num_samples: int | tuple[int, ...]) -> torch.Tensor:
        dim = self.dim
        dim_minus_one = dim - 1
        n_samples = int(math.prod(num_samples)) if isinstance(num_samples, (list, tuple)) else int(num_samples)

        sqrt = math.sqrt(4 * self.kappa**2 + dim_minus_one**2)
        envelop_param = (-2 * self.kappa + sqrt) / dim_minus_one
        if envelop_param == 0:
            envelop_param = (dim_minus_one / 4 * self.kappa**-1 - dim_minus_one**3 / 64 * self.kappa**-3)

        node = (1.0 - envelop_param) / (1.0 + envelop_param)
        correction = self.kappa * node + dim_minus_one * (
            math.log(4) + math.log(envelop_param) - 2 * math.log1p(envelop_param)
        )

        n_accepted = 0
        x = torch.zeros((n_samples,), device=self.device, dtype=self.dtype)
        halfdim = 0.5 * dim_minus_one
        beta_dist = torch.distributions.Beta(halfdim, halfdim)

        while n_accepted < n_samples:
            remaining = n_samples - n_accepted
            sym_beta = beta_dist.sample((remaining,)).to(device=self.device, dtype=self.dtype)
            coord_x = (1 - (1 + envelop_param) * sym_beta) / (1 - (1 - envelop_param) * sym_beta)
            accept_tol = torch.rand(remaining, device=self.device, dtype=self.dtype)
            criterion = (
                self.kappa * coord_x
                + dim_minus_one
                * (
                    torch.log((1 + envelop_param - coord_x + coord_x * envelop_param) / (1 + envelop_param))
                )
                - correction
                > torch.log(accept_tol)
            )
            accepted_iter = int(criterion.sum().item())
            x[n_accepted : n_accepted + accepted_iter] = coord_x[criterion]
            n_accepted += accepted_iter

        coord_rest = self._sample_uniform_direction(dim=dim_minus_one, size=n_samples)
        coord_rest = torch.einsum("...,...i->...i", torch.sqrt(1 - x**2), coord_rest)
        samples = torch.cat([x[..., None], coord_rest], dim=1)

        if isinstance(num_samples, (list, tuple)) and len(num_samples) > 1:
            samples = samples.reshape(tuple(num_samples) + (dim,))

        if self.rotation_needed:
            samples = self._rotate_samples(samples)
        return samples
