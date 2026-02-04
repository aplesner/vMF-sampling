from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import tqdm

from src.vmf import vMF
from src.vmf_numpy import DTYPES as NUMPY_DTYPES
from src.vmf_scipy import DTYPES as SCIPY_DTYPES

try:
    import torch
    from src.vmf_torch import DTYPES as TORCH_DTYPES
except ImportError:
    torch = None
    TORCH_DTYPES = []

SEEDS = [0, 1, 2]
DIMS = [750]
KAPPAS = [50.0]
NUM_SAMPLES = 5_000

AGGREGATE_SEEDS = True
DISPLAY_MEAN_STD = True


def _make_mu_numpy(dim: int, dtype: np.dtype) -> np.ndarray:
    mu = np.zeros(dim, dtype=dtype)
    mu[-1] = 1.0
    return mu


def _make_mu_torch(dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mu = torch.zeros((dim,), device=device, dtype=dtype)
    mu[-1] = 1.0
    return mu


def _time_sample(sampler: Any, num_samples: int) -> float:
    start = time.perf_counter()
    _ = sampler.sample(num_samples)
    return time.perf_counter() - start


def _run_backend(
    backend: str,
    dtypes: list[Any],
    make_mu,
    extra_kwargs,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dtype in dtypes:
        for dim in DIMS:
            mu = make_mu(dim, dtype)
            for kappa in KAPPAS:
                for seed in SEEDS:
                    sampler = vMF(
                        dim=dim,
                        mu=mu,
                        kappa=kappa,
                        seed=seed,
                        **extra_kwargs(dtype),
                    )
                    elapsed = _time_sample(sampler, NUM_SAMPLES)
                    rows.append(
                        {
                            "backend": backend,
                            "dtype": str(dtype),
                            "dim": dim,
                            "kappa": kappa,
                            "seed": seed,
                            "time_s": elapsed,
                        }
                    )
    return rows


def _format_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    mean = df["time_s_mean"]
    std = df["time_s_std"]
    formatted = mean.map("{:.6f}".format) + " ± " + std.map("{:.6f}".format)
    return df.assign(time_s=formatted).drop(columns=["time_s_mean", "time_s_std"])


def main() -> None:
    rows: list[dict[str, Any]] = []

    if torch is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = None
    backends = [
        ("numpy", NUMPY_DTYPES, _make_mu_numpy),
        ("numpy_hh", NUMPY_DTYPES, _make_mu_numpy),
        ("scipy", SCIPY_DTYPES, _make_mu_numpy),
        ("torch", TORCH_DTYPES, lambda dim, dtype: _make_mu_torch(dim, dtype, device)),
    ]
    pbar = tqdm.tqdm(total=len(backends))
    for backend, dtypes, make_mu in backends:
        pbar.desc = f"Running benchmark for {backend}"
        if torch is None and backend == "torch":
            pbar.skip()
            continue
        rows.extend(_run_backend(backend, dtypes, make_mu, lambda _dtype: {"backend": backend}))
        pbar.update(1)

    df = pd.DataFrame(rows)
    df = df.set_index(["backend", "dtype", "dim", "kappa", "seed"]).sort_index()

    if not AGGREGATE_SEEDS:
        print(df)
        return

    grouped = df.groupby(level=["backend", "dtype", "dim", "kappa"])
    stats = grouped.agg(time_s_mean=("time_s", "mean"), time_s_std=("time_s", "std"))

    if DISPLAY_MEAN_STD:
        stats = _format_mean_std(stats)
    print(stats)


if __name__ == "__main__":
    main()
