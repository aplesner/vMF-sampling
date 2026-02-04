from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

try:
    import torch
except ImportError:
    torch = None
import numpy as np

if torch is not None:
    MU_TYPE = torch.Tensor | np.ndarray
else:
    MU_TYPE = np.ndarray

class vMFSampler(ABC):
    """Abstract base for von Mises-Fisher samplers."""

    def __init__(
        self,
        dim: int,
        mu: MU_TYPE | None = None,
        kappa: float = 10.0,
        seed: int | None = None,
        rotation_needed: bool = True,
    ) -> None:
        if dim <= 1:
            raise ValueError("dim must be greater than 1")
        if kappa <= 0:
            raise ValueError("kappa must be positive")

        self.dim = dim
        self.kappa = float(kappa)
        self.seed = seed
        self.rotation_needed = rotation_needed
        self.mu: MU_TYPE | None = None

        if mu is not None:
            self.set_mu(mu)

    def set_kappa(self, kappa: float) -> None:
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        self.kappa = float(kappa)

    def set_mu(self, mu: MU_TYPE) -> None:
        self._verify_mu(mu)
        self.mu = mu
        self.dim = self._get_dim(mu)
        self.rotation_needed = self._rotation_needed(mu)
        self._on_mu_updated()

    def sample(
        self,
        num_samples: int | tuple[int, ...],
        mu: MU_TYPE | None = None,
        kappa: float | None = None,
        rotation_needed: bool = True,
    ) -> MU_TYPE:
        if mu is not None:
            self.rotation_needed = rotation_needed
            self.set_mu(mu)
        elif self.mu is None:
            self.set_mu(self._default_mu())

        if kappa is not None:
            self.set_kappa(kappa)

        return self._sample(num_samples)

    def _on_mu_updated(self) -> None:
        """Hook for subclasses to reset cached state."""

    @abstractmethod
    def _sample(self, num_samples: int | tuple[int, ...]) -> MU_TYPE:
        raise NotImplementedError

    @abstractmethod
    def _verify_mu(self, mu: MU_TYPE) -> None:
        raise NotImplementedError

    @abstractmethod
    def _default_mu(self) -> MU_TYPE:
        raise NotImplementedError

    @abstractmethod
    def _rotation_needed(self, mu: MU_TYPE) -> bool:
        raise NotImplementedError

    def _get_dim(self, mu: MU_TYPE) -> int:
        return mu.shape[-1]