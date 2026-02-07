from __future__ import annotations

from typing import Any, Callable

import numpy as np

from src.vmf import vMF
from src.vmf_numpy import DTYPES as NUMPY_DTYPES
from src.vmf_scipy import DTYPES as SCIPY_DTYPES

try:
    import torch
    from src.vmf_torch import DTYPES as TORCH_DTYPES
except ImportError:
    torch = None
    TORCH_DTYPES = []

DIMS = [2048]
KAPPAS = [50.0]
NUM_SAMPLES = 5_000

NUMPY_DTYPES_SUBSET = [NUMPY_DTYPES[1]]  # float32
SCIPY_DTYPES_SUBSET = [SCIPY_DTYPES[1]]  # float32
TORCH_DTYPES_SUBSET = [TORCH_DTYPES[2]] if TORCH_DTYPES else []


def _make_mu_numpy(dim: int, dtype: np.dtype) -> np.ndarray:
    mu = np.zeros(dim, dtype=dtype)
    mu[-1] = 1.0
    return mu


def _make_mu_torch(dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mu = torch.zeros((dim,), device=device, dtype=dtype)
    mu[-1] = 1.0
    return mu


def _run_backend(
    dtypes: list[Any],
    make_mu: Callable[[int, Any], Any],
    extra_kwargs: Callable[[Any], dict[str, Any]],
) -> None:
    for dtype in dtypes:
        for dim in DIMS:
            mu = make_mu(dim, dtype)
            for kappa in KAPPAS:
                sampler = vMF(
                    dim=dim,
                    mu=mu,
                    kappa=kappa,
                    seed=0,
                    **extra_kwargs(dtype),
                )
                _ = sampler.sample(NUM_SAMPLES)


def main() -> None:
    backends = [
        ("scipy", SCIPY_DTYPES_SUBSET, _make_mu_numpy, lambda dtype: {"backend": "scipy", "dtype": dtype}),
        # ("numpy", NUMPY_DTYPES_SUBSET, _make_mu_numpy, lambda dtype: {"backend": "numpy", "dtype": dtype}),
        # (
        #     "numpy_hh_inplace",
        #     NUMPY_DTYPES_SUBSET,
        #     _make_mu_numpy,
        #     lambda dtype: {"backend": "numpy_hh", "dtype": dtype, "inplace": True},
        # ),
        # (
        #     "numpy_hh",
        #     NUMPY_DTYPES_SUBSET,
        #     _make_mu_numpy,
        #     lambda dtype: {"backend": "numpy_hh", "dtype": dtype, "inplace": False},
        # ),
    ]

    if torch is not None and False:
        for device in [torch.device("cpu"), torch.device("cuda")]:
            if device.type == "cuda" and not torch.cuda.is_available():
                continue
            backends.extend(
                [
                    (
                        f"torch_hh_inplace_{device.type}",
                        TORCH_DTYPES_SUBSET,
                        lambda dim, dtype, d=device: _make_mu_torch(dim, dtype, d),
                        lambda dtype, d=device: {
                            "backend": "torch_hh",
                            "device": d,
                            "dtype": dtype,
                            "inplace": True,
                        },
                    ),
                    (
                        f"torch_hh_{device.type}",
                        TORCH_DTYPES_SUBSET,
                        lambda dim, dtype, d=device: _make_mu_torch(dim, dtype, d),
                        lambda dtype, d=device: {
                            "backend": "torch_hh",
                            "device": d,
                            "dtype": dtype,
                            "inplace": False,
                        },
                    ),
                ]
            )

    for backend_name, dtypes, make_mu, extra_kwargs in backends:
        if torch is None and backend_name.startswith("torch"):
            continue
        print(f"Running benchmark for {backend_name}")
        _run_backend(dtypes, make_mu, extra_kwargs)


if __name__ == "__main__":
    main()
