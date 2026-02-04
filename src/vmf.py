from __future__ import annotations

from typing import Any

import numpy as np
try:
    import torch
except ImportError:  # pragma: no cover - torch is optional
    torch = None

from .vmf_numpy import NumpyvMF
from .vmf_numpy_hh import NumpyvMFHH
from .vmf_scipy import ScipyvMF
from .vmf_torch import TorchvMF

if torch is not None:
    DTYPE_TYPE = torch.dtype | np.dtype
else:
    DTYPE_TYPE = np.dtype


class vMF:
    """Factory class that dispatches to a backend implementation."""

    def __new__(
        cls,
        dim: int,
        mu: Any | None = None,
        kappa: float = 10.0,
        seed: int | None = None,
        rotation_needed: bool = True,
        backend: str | None = None,
        device: Any | None = None,
        dtype: DTYPE_TYPE | None = None,
        use_scipy: bool = False,
    ) -> Any:
        backend_name = backend
        if backend_name is None:
            if torch is not None and isinstance(mu, torch.Tensor):
                backend_name = "torch"
            elif device is not None or dtype is not None:
                backend_name = "torch"
            elif use_scipy:
                backend_name = "scipy"
            else:
                backend_name = "numpy"

        backend_name = backend_name.lower()
        common_kwargs = dict(
            dim=dim,
            mu=mu,
            kappa=kappa,
            seed=seed,
            rotation_needed=rotation_needed,
        )

        if backend_name == "scipy":
            return ScipyvMF(**common_kwargs)
        if backend_name == "torch":
            if torch is None:
                raise ImportError("PyTorch is not installed. Please install PyTorch to use the torch backend.")
            return TorchvMF(**common_kwargs, device=device, dtype=dtype)
        if backend_name == "numpy":
            return NumpyvMF(**common_kwargs)
        if backend_name == "numpy_hh":
            return NumpyvMFHH(**common_kwargs)

        raise ValueError(f"Unknown backend '{backend_name}'. Expected numpy, scipy, or torch.")
