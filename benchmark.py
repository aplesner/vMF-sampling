from __future__ import annotations

import math
import statistics
import timeit
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

SEED = 0
DIMS = [512]
KAPPAS = [50.0]
NUM_SAMPLES = 5_000
TARGET_TIME_S = 5.0
TIMEIT_REPEATS = 3

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


def _time_sample(sampler: vMF, num_samples: int, target_s: float, repeats: int) -> tuple[float, float]:
    assert target_s > 0.1, f"Target time must be greater than 0.1: {target_s}"
    def _run_once() -> None:
        result = sampler.sample(num_samples)
        if torch is not None and isinstance(result, torch.Tensor) and result.is_cuda:
            torch.cuda.synchronize(result.device)

    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    # Warm up
    _run_once()

    # Initial timings. This gives us a rough estimate of the time per call.
    timer = timeit.Timer(_run_once)
    number, total_time = timer.autorange()
    if number < 1 or total_time <= 0:
        raise ValueError(f"Number of calls is less than 1 or total time is less than or equal to 0: {number}, {total_time}")
    per_call = total_time / number

    # Adjust the number of calls to target the desired time.
    number = int(max(1, math.ceil(target_s / per_call)))
    times = timer.repeat(repeat=repeats, number=number)
    per_call_times = [t / number for t in times]
    mean = statistics.mean(per_call_times)
    std = statistics.stdev(per_call_times) if len(per_call_times) > 1 else 0.0
    return mean, std


def _run_backend(
    backend: str,
    dtypes: list[Any],
    make_mu,
    extra_kwargs,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dtype in tqdm.tqdm(dtypes, desc=f"Running benchmark for {backend}"):
        for dim in DIMS:
            mu = make_mu(dim, dtype)
            for kappa in KAPPAS:
                sampler = vMF(
                    dim=dim,
                    mu=mu,
                    kappa=kappa,
                    seed=SEED,
                    **extra_kwargs(dtype),
                )
                mean_s, std_s = _time_sample(sampler, NUM_SAMPLES, TARGET_TIME_S, TIMEIT_REPEATS)
                rows.append(
                    {
                        "backend": backend,
                        "dtype": str(dtype),
                        "dim": dim,
                        "kappa": kappa,
                        "time_s_mean": mean_s,
                        "time_s_std": std_s,
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

    devices: list[torch.device] = []
    if torch is not None:
        devices.append(torch.device("cpu"))
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

    backends = [
        ("numpy", NUMPY_DTYPES, _make_mu_numpy, lambda dtype: {"backend": "numpy", "dtype": dtype}),
        (
            "numpy_hh_inplace",
            NUMPY_DTYPES,
            _make_mu_numpy,
            lambda dtype: {"backend": "numpy_hh", "dtype": dtype, "inplace": True},
        ),
        (
            "numpy_hh",
            NUMPY_DTYPES,
            _make_mu_numpy,
            lambda dtype: {"backend": "numpy_hh", "dtype": dtype, "inplace": False},
        ),
        ("scipy", SCIPY_DTYPES, _make_mu_numpy, lambda dtype: {"backend": "scipy", "dtype": dtype}),
    ]
    if torch is not None:
        for device in devices:
            device_label = device.type
            # (
            #     f"torch_{device_label}",
            #     TORCH_DTYPES,
            #     lambda dim, dtype, d=device: _make_mu_torch(dim, dtype, d),
            #     lambda dtype, d=device: {"backend": "torch", "device": d, "dtype": dtype},
            # ),
            backends.extend(
                [
                    (
                        f"torch_hh_inplace_{device_label}",
                        TORCH_DTYPES,
                        lambda dim, dtype, d=device: _make_mu_torch(dim, dtype, d),
                        lambda dtype, d=device: {
                            "backend": "torch_hh",
                            "device": d,
                            "dtype": dtype,
                            "inplace": True,
                        },
                    ),
                    (
                        f"torch_hh_{device_label}",
                        TORCH_DTYPES,
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
    for backend, dtypes, make_mu, extra_kwargs in backends:
        if torch is None and backend.startswith("torch"):
            continue
        rows.extend(_run_backend(backend, dtypes, make_mu, extra_kwargs))

    df = pd.DataFrame(rows)
    df = df.set_index(["backend", "dtype", "dim", "kappa"]).sort_index()

    if not AGGREGATE_SEEDS:
        print(df)
        return

    if DISPLAY_MEAN_STD:
        df = _format_mean_std(df)
    print(df)


if __name__ == "__main__":
    main()
