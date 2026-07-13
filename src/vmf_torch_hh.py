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
        make_copy: bool = False,
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
        self.make_copy = make_copy

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
        Applies a single Householder reflection mapping e1 -> mu to set S.
        S is an array of shape (n_samples, n_dims).

        vMF is rotationally symmetric about mu, so orientation preservation
        (det = +1) is unnecessary; a single reflection (det = -1) suffices
        and is half the work of the previous two-reflection rotation.
        """
        x = torch.zeros((self.dim,), device=self.device, dtype=self.mu.dtype)
        x[0] = 1.0
        v = x - self.mu
        u = v / torch.linalg.norm(v)

        # Vectorized application: H(s) = s - 2(u·s)u
        return S - 2 * torch.outer(S @ u, u)

    def _rotate_householder_inplace(self, S: torch.Tensor) -> torch.Tensor:
        """
        Applies a single Householder reflection mapping e1 -> mu to set S,
        in place. S is an array of shape (n_samples, n_dims).

        vMF is rotationally symmetric about mu, so orientation preservation
        (det = +1) is unnecessary; a single reflection (det = -1) suffices
        and is half the work of the previous two-reflection rotation.
        """
        assert S.ndim == 2
        x = torch.zeros((self.dim,), device=self.device, dtype=self.mu.dtype)
        x[0] = 1.0
        v = x - self.mu
        u = v / torch.linalg.norm(v)

        # Vectorized application: H(s) = s - 2(u·s)u
        S_dot = torch.mv(S, u)
        S_dot.mul_(2)
        S.sub_(torch.outer(S_dot, u))
        return S

    def _rotate_samples(self, samples: torch.Tensor) -> torch.Tensor:
        if self.inplace:
            return self._rotate_householder_inplace(samples.clone() if self.make_copy else samples)
        return self._rotate_householder(samples)

    def _sample_uniform_direction(self, dim: int, size: int) -> torch.Tensor:
        samples = torch.randn((size, dim), device=self.device, dtype=self.dtype)
        samples = samples / samples.norm(dim=-1, keepdim=True)
        return samples

    def _sample_dim2(self, num_samples: int | tuple[int, ...], n_samples: int) -> torch.Tensor:
        """Closed-form: vMF on the circle reduces to the von Mises distribution."""
        dist_dtype = torch.float32 if self.dtype in (torch.float16, torch.bfloat16) else self.dtype
        mu_calc = self.mu.to(dtype=dist_dtype)
        mean_angle = torch.atan2(mu_calc[1], mu_calc[0])
        concentration = torch.tensor(self.kappa, device=self.device, dtype=dist_dtype)
        dist = torch.distributions.VonMises(mean_angle, concentration)
        angles = dist.sample((n_samples,)).to(dtype=self.dtype)
        samples = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        if isinstance(num_samples, (list, tuple)) and len(num_samples) > 1:
            samples = samples.reshape(tuple(num_samples) + (2,))
        return samples

    def _sample_dim3(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Closed-form inverse-CDF for the polar coordinate about the north pole."""
        # log/exp are numerically fragile in half precision (tiny underflows to
        # 0, giving log(0) = -inf), so compute in float32 minimum, like the
        # QR-based rotation matrix does elsewhere for float16/bfloat16.
        calc_dtype = torch.float32 if self.dtype in (torch.float16, torch.bfloat16) else self.dtype
        tiny = 1e-12
        u = torch.rand(n_samples, device=self.device, dtype=calc_dtype)
        u = torch.clamp(u, min=tiny)
        x = 1.0 + torch.log(u + (1.0 - u) * math.exp(-2.0 * self.kappa)) / self.kappa
        t = torch.sqrt(torch.clamp(1.0 - x**2, min=0.0))
        x = x.to(dtype=self.dtype)
        t = t.to(dtype=self.dtype)
        circ = self._sample_uniform_direction(dim=2, size=n_samples)
        coord_rest = t.unsqueeze(-1) * circ
        return x, coord_rest

    def _sample(self, num_samples: int | tuple[int, ...]) -> torch.Tensor:
        dim = self.dim
        dim_minus_one = dim - 1
        n_samples = int(math.prod(num_samples)) if isinstance(num_samples, (list, tuple)) else int(num_samples)

        if dim == 2:
            return self._sample_dim2(num_samples, n_samples)

        if dim == 3:
            x, coord_rest = self._sample_dim3(n_samples)
        else:
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

        if self.rotation_needed:
            samples = self._rotate_samples(samples)

        if isinstance(num_samples, (list, tuple)) and len(num_samples) > 1:
            samples = samples.reshape(tuple(num_samples) + (dim,))
        return samples
