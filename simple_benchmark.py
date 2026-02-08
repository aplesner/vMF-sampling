from __future__ import annotations

import argparse
import csv
import math
import timeit
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import tqdm

from src.vmf import vMF
# from src.vmf_numpy import DTYPES as NUMPY_DTYPES
from src.vmf_scipy import DTYPES as SCIPY_DTYPES

try:
    import torch
    from src.vmf_torch import DTYPES as TORCH_DTYPES
except ImportError:
    torch = None
    TORCH_DTYPES = []

DEFAULT_SEEDS = [0, 1]
DEFAULT_DIMS = [2048, 4096]
NUM_SAMPLES = 5_000
KAPPA = 50
NUMPY_DTYPES = [np.float32]
TORCH_DTYPES = [torch.float32]

AGGREGATE_SEEDS = True
DISPLAY_MEAN_STD = True



def _make_mu_numpy(dim: int, dtype: np.dtype) -> np.ndarray:
    mu = np.zeros(dim, dtype=dtype)
    mu[-1] = 1.0
    return mu


def _make_mu_torch(dim: int, dtype: torch.dtype) -> torch.Tensor:
    mu = torch.zeros((dim,), device="cpu", dtype=dtype)
    mu[-1] = 1.0
    return mu

def _make_mu_torch_cuda(dim: int, dtype: torch.dtype) -> torch.Tensor:
    mu = torch.zeros((dim,), device="cuda", dtype=dtype)
    mu[-1] = 1.0
    return mu


def _time_sampling(sampler: vMF) -> float:
    def _run_once() -> None:
        if hasattr(sampler, "rotmatrix") and sampler.rotmatrix is not None:
            sampler.rotmatrix = None
            sampler.rotsign = None
        result = sampler.sample(NUM_SAMPLES)
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
    per_call_time = total_time / number

    return per_call_time


def _run_backend(
    backend: str,
    dtypes: list[Any],
    make_mu,
    extra_kwargs,
    dims: Iterable[int],
    seeds: Iterable[int],
    kappa: float,
    on_row: Callable[[dict[str, Any]], None],
) -> None:
    for dtype in tqdm.tqdm(dtypes, desc=f"Running benchmark for {backend}"):
        for dim in dims:
            mu = make_mu(dim, dtype)
            
            for seed in seeds:
                kwargs = extra_kwargs(dtype)
                sampler = vMF(
                    dim=dim,
                    mu=mu,
                    kappa=kappa,
                    seed=seed,
                    **kwargs,
                )
                time_s = _time_sampling(sampler)
                device = "cpu"
                if "device" in kwargs and kwargs["device"] is not None:
                    device = getattr(kwargs["device"], "type", str(kwargs["device"]))
                on_row(
                    {
                        "backend": backend,
                        "dtype": str(dtype),
                        "dim": dim,
                        "kappa": kappa,
                        "seed": seed,
                        "time_s": time_s,
                        "uses_householder": "hh" in backend,
                        "inplace": "inplace" in backend,
                        "device": device,
                        "num_samples": NUM_SAMPLES,
                    }
                )


def _format_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    mean = df["time_s_mean"]
    std = df["time_s_std"]
    formatted = mean.map("{:.6f}".format) + " ± " + std.map("{:.6f}".format)
    return df.assign(time_s=formatted).drop(columns=["time_s_mean", "time_s_std"])


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def _parse_dims_range(value: str) -> list[int]:
    parts = [int(part) for part in value.split(":") if part]
    if len(parts) not in (2, 3):
        raise ValueError("dims-range must be start:end or start:end:step")
    start, end = parts[0], parts[1]
    step = parts[2] if len(parts) == 3 else 1
    if step <= 0:
        raise ValueError("dims-range step must be positive")
    if end < start:
        raise ValueError("dims-range end must be >= start")
    return list(range(start, end + 1, step))


def _write_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark vMF samplers.")
    parser.add_argument("--dim", type=int, help="Run a single dimension.")
    parser.add_argument("--dims", type=_parse_int_list, help="Comma-separated dimensions.")
    parser.add_argument("--dims-range", type=_parse_dims_range, help="Range start:end(:step).")
    parser.add_argument("--seeds", type=_parse_int_list, default=None, help="Comma-separated seeds.")
    parser.add_argument("--kappa", type=float, default=KAPPA, help="Kappa value.")
    parser.add_argument("--output", type=Path, help="CSV output path for incremental results.")
    parser.add_argument("--summary", action="store_true", help="Print summary table at end.")
    args = parser.parse_args()

    if args.dim is not None:
        dims = [args.dim]
    elif args.dims is not None:
        dims = args.dims
    elif args.dims_range is not None:
        dims = args.dims_range
    else:
        dims = DEFAULT_DIMS

    seeds = args.seeds if args.seeds is not None else DEFAULT_SEEDS
    kappa = args.kappa

    rows: list[dict[str, Any]] | None = [] if (args.summary or args.output is None) else None
    def on_row(row: dict[str, Any]) -> None:
        if args.output is not None:
            _write_csv_row(args.output, row)
        if rows is not None:
            rows.append(row)

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
        ("torch", TORCH_DTYPES, _make_mu_torch, lambda dtype: {"backend": "torch", "dtype": dtype}),
        ("torch_hh", TORCH_DTYPES, _make_mu_torch, lambda dtype: {"backend": "torch_hh", "dtype": dtype, "inplace": False}),
        ("torch_hh_inplace", TORCH_DTYPES, _make_mu_torch, lambda dtype: {"backend": "torch_hh", "dtype": dtype, "inplace": True}),
        ("torch_cuda", TORCH_DTYPES, _make_mu_torch_cuda, lambda dtype: {"backend": "torch", "dtype": dtype, "device": torch.device("cuda")}),
        ("torch_cuda_hh", TORCH_DTYPES, _make_mu_torch_cuda, lambda dtype: {"backend": "torch_hh", "dtype": dtype, "inplace": False, "device": torch.device("cuda")}),
        ("torch_cuda_hh_inplace", TORCH_DTYPES, _make_mu_torch_cuda, lambda dtype: {"backend": "torch_hh", "dtype": dtype, "inplace": True, "device": torch.device("cuda")}),
    ]

    for backend, dtypes, make_mu, extra_kwargs in backends:
        if torch is None and backend.startswith("torch"):
            continue
        _run_backend(backend, dtypes, make_mu, extra_kwargs, dims, seeds, kappa, on_row)

    if rows is None:
        return

    df = pd.DataFrame(rows)
    df = df.set_index(["backend", "dtype", "dim", "kappa", "seed"]).sort_index()

    if not AGGREGATE_SEEDS:
        print(df)
        return

    grouped = df.groupby(level=["backend", "dtype", "dim", "kappa"])
    stats = grouped.agg(
        time_s_mean=("time_s", "mean"),
        time_s_std=("time_s", "std"),
    )

    if DISPLAY_MEAN_STD:
        stats = _format_mean_std(stats)
    print(stats)


if __name__ == "__main__":
    main()
