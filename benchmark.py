from __future__ import annotations

import argparse
import csv
import math
import time
import timeit
from pathlib import Path
from typing import Any, Callable, Iterable

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

DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_DIMS = [2, 3] + [2**power for power in range(2, 10)]
NUM_SAMPLES = 5_000
TARGET_TIME_S = 5.0
KAPPA = 50

# Per-call size calibration (see TODO_benchmark_sizing.md): a fixed NUM_SAMPLES
# under-reports throughput at low/mid dim because fixed per-call overhead
# (rotation setup, allocation, dispatch) dominates. Instead we calibrate the
# number of samples per sample() call, per (backend, dtype, dim), so one call
# takes about TARGET_CALL_TIME_S, capped by a memory budget.
TARGET_CALL_TIME_S = 2.0
MEM_BUDGET_BYTES = 1.2e9
MIN_SAMPLES = 5_000

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


def _time_sample(sampler: vMF, num_samples: int, target_s: float) -> tuple[float, int]:
    assert target_s > 0.1, f"Target time must be greater than 0.1: {target_s}"
    def _run_once() -> None:
        if hasattr(sampler, "rotmatrix") and sampler.rotmatrix is not None:
            sampler.rotmatrix = None
            sampler.rotsign = None
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
    total_time = timer.timeit(number=number)
    per_call_time = total_time / number
    return per_call_time, number


def _itemsize(dtype: Any) -> int:
    if torch is not None and isinstance(dtype, torch.dtype):
        return dtype.itemsize
    return np.dtype(dtype).itemsize


def _calibrate_num_samples(
    sampler: vMF,
    dim: int,
    itemsize: int,
    target_call_s: float,
    mem_budget_bytes: float,
) -> int:
    """Pick a per-call sample count so one sample() call takes ~target_call_s,
    capped so its output stays within mem_budget_bytes."""
    cap = max(MIN_SAMPLES, int(mem_budget_bytes // (dim * itemsize)))
    n_probe = min(50_000, cap)

    def _run_once(n: int) -> None:
        if hasattr(sampler, "rotmatrix") and sampler.rotmatrix is not None:
            sampler.rotmatrix = None
            sampler.rotsign = None
        result = sampler.sample(n)
        if torch is not None and isinstance(result, torch.Tensor) and result.is_cuda:
            torch.cuda.synchronize(result.device)

    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()

    # Warm up.
    _run_once(min(5_000, n_probe))

    # Probe a single call at n_probe to estimate the per-sample rate.
    start = time.perf_counter()
    _run_once(n_probe)
    t = time.perf_counter() - start

    rate = t / n_probe
    n = max(MIN_SAMPLES, min(int(target_call_s / rate), cap))

    # Refinement: the probe-size call's fixed overhead inflates the per-sample
    # rate estimate, so the first n undershoots. Re-time at the candidate n.
    if n > 4 * n_probe:
        start = time.perf_counter()
        _run_once(n)
        t2 = time.perf_counter() - start
        n = max(MIN_SAMPLES, min(int(target_call_s * n / t2), cap))

    return n


def _run_backend(
    backend: str,
    dtypes: list[Any],
    make_mu,
    extra_kwargs,
    dims: Iterable[int],
    seeds: Iterable[int],
    kappa: float,
    on_row: Callable[[dict[str, Any]], None],
    fixed_num_samples: int | None,
    target_call_time_s: float,
    mem_budget_bytes: float,
    target_time_s: float,
) -> None:
    for dtype in tqdm.tqdm(dtypes, desc=f"Running benchmark for {backend}"):
        if torch is not None and backend == "torch_cpu" and dtype in {torch.float16, torch.bfloat16}:
            continue
        for dim in dims:
            mu = make_mu(dim, dtype)
            num_samples = fixed_num_samples
            for seed in seeds:
                kwargs = extra_kwargs(dtype)
                sampler = vMF(
                    dim=dim,
                    mu=mu,
                    kappa=kappa,
                    seed=seed,
                    **kwargs,
                )
                if num_samples is None:
                    itemsize = _itemsize(dtype)
                    num_samples = _calibrate_num_samples(
                        sampler, dim, itemsize, target_call_time_s, mem_budget_bytes
                    )
                time_s, number = _time_sample(sampler, num_samples, target_time_s)
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
                        "number": number,
                        "uses_householder": "hh" in backend,
                        "inplace": "inplace" in backend,
                        "device": device,
                        "num_samples": num_samples,
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


def _dtype_name(dtype: Any) -> str:
    name = str(dtype)
    for candidate in ("float16", "bfloat16", "float32", "float64"):
        if candidate in name:
            return candidate
    return name


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


def _backend_group(name: str) -> str:
    if name.startswith("numpy"):
        return "numpy"
    if name.startswith("scipy"):
        return "scipy"
    if name.startswith("torch"):
        return "torch"
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark vMF samplers.")
    parser.add_argument("--dim", type=int, help="Run a single dimension.")
    parser.add_argument("--dims", type=_parse_int_list, help="Comma-separated dimensions.")
    parser.add_argument("--dims-range", type=_parse_dims_range, help="Range start:end(:step).")
    parser.add_argument("--kappa", type=float, default=KAPPA, help="Kappa value.")
    parser.add_argument(
        "--seeds",
        type=_parse_int_list,
        default=DEFAULT_SEEDS,
        help="Comma-separated random seeds.",
    )
    parser.add_argument("--output", type=Path, help="CSV output path for incremental results.")
    parser.add_argument(
        "--backends",
        type=lambda v: [item.strip() for item in v.split(",") if item.strip()],
        default=None,
        help="Comma-separated backend groups to run: numpy, scipy, torch.",
    )
    parser.add_argument(
        "--dtypes",
        type=lambda v: [item.strip() for item in v.split(",") if item.strip()],
        default=None,
        help="Optional comma-separated dtype filter, e.g. float32,float64.",
    )
    parser.add_argument("--summary", action="store_true", help="Print summary table at end.")
    parser.add_argument(
        "--target-time",
        type=float,
        default=TARGET_TIME_S,
        help="Target total timing window (s) for each benchmark row.",
    )
    parser.add_argument(
        "--target-call-time",
        type=float,
        default=TARGET_CALL_TIME_S,
        help="Target wall time (s) for a single sample() call, used to calibrate num_samples.",
    )
    parser.add_argument(
        "--mem-budget-gb",
        type=float,
        default=MEM_BUDGET_BYTES / 1e9,
        help="Memory budget (GB) for a single sample() call's output array, caps calibrated num_samples.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Skip calibration and use this fixed num_samples for every row (old behavior).",
    )
    args = parser.parse_args()

    if args.dim is not None:
        dims = [args.dim]
    elif args.dims is not None:
        dims = args.dims
    elif args.dims_range is not None:
        dims = args.dims_range
    else:
        dims = DEFAULT_DIMS

    seeds = args.seeds
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
                        f"torch_{device_label}",
                        TORCH_DTYPES,
                        lambda dim, dtype, device=device: _make_mu_torch(dim, dtype, device=device),
                        lambda dtype, device=device: {
                            "backend": "torch",
                            "device": device,
                            "dtype": dtype,
                        },
                    ),
                    (
                        f"torch_hh_inplace_{device_label}",
                        TORCH_DTYPES,
                        lambda dim, dtype, device=device: _make_mu_torch(dim, dtype, device=device),
                        lambda dtype, device=device: {
                            "backend": "torch_hh",
                            "device": device,
                            "dtype": dtype,
                            "inplace": True,
                        },
                    ),
                    (
                        f"torch_hh_{device_label}",
                        TORCH_DTYPES,
                        lambda dim, dtype, device=device: _make_mu_torch(dim, dtype, device=device),
                        lambda dtype, device=device: {
                            "backend": "torch_hh",
                            "device": device,
                            "dtype": dtype,
                            "inplace": False,
                        },
                    ),
                ]
            )
    backend_filter = None
    if args.backends is not None:
        backend_filter = {item.lower() for item in args.backends}

    dtype_filter = set(args.dtypes) if args.dtypes is not None else None
    for backend, dtypes, make_mu, extra_kwargs in backends:
        if torch is None and backend.startswith("torch"):
            continue
        if backend_filter is not None and _backend_group(backend) not in backend_filter:
            continue
        selected_dtypes = dtypes
        if dtype_filter is not None:
            selected_dtypes = [dtype for dtype in dtypes if _dtype_name(dtype) in dtype_filter]
            if not selected_dtypes:
                continue
        _run_backend(
            backend,
            selected_dtypes,
            make_mu,
            extra_kwargs,
            dims,
            seeds,
            kappa,
            on_row,
            args.num_samples,
            args.target_call_time,
            args.mem_budget_gb * 1e9,
            args.target_time,
        )

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
