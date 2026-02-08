from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.vmf import vMF

try:
    import torch
except ImportError:  # pragma: no cover - torch optional
    torch = None


PAIRS = [
    (500, 100),
    (500, 500),
    (500, 1000),
    (500, 5000),
    (500, 10000),
    (500, 20000),
    (500, 50000),
    (500, 100000),
    (1000, 100),
    (1000, 500),
    (1000, 1000),
    (1000, 5000),
    (1000, 10000),
    (1000, 20000),
    (1000, 50000),
    (1000, 100000),
    (5000, 100),
    (5000, 500),
    (5000, 1000),
    (5000, 5000),
    (5000, 10000),
    (5000, 20000),
    (5000, 50000),
    (5000, 100000),
    (10000, 100),
    (10000, 500),
    (10000, 1000),
    (10000, 5000),
    (10000, 10000),
    (10000, 20000),
    (10000, 50000),
    (10000, 100000),
    (20000, 100),
    (20000, 500),
    (20000, 1000),
    (20000, 5000),
    (20000, 10000),
    (20000, 20000),
    (20000, 50000),
    (20000, 100000),
    (100000, 100),
    (100000, 500),
    (100000, 1000),
    (100000, 5000),
    (100000, 10000),
    (100000, 20000),
    (100000, 50000),
    (100000, 100000),
]


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _write_csv_row(path: Path, row: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def _sample_rbar(
    dim: int,
    kappa: float,
    num_samples: int,
    seed: int,
    device: str,
    dtype: str,
) -> float:
    if torch is None:
        raise ImportError("PyTorch is required for torch_hh sampling.")
    torch_device = torch.device(device)
    torch_dtype = getattr(torch, dtype)
    mu = torch.randn(dim, device=torch_device, dtype=torch_dtype)
    mu = mu / torch.linalg.norm(mu)
    sampler = vMF(
        dim=dim,
        mu=mu,
        kappa=kappa,
        seed=seed,
        backend="torch_hh",
        device=torch_device,
        dtype=torch_dtype,
    )
    samples = sampler.sample(num_samples)
    mean_vec = samples.mean(dim=0)
    rbar = torch.linalg.norm(mean_vec).item()
    return float(rbar)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample Rbar for Table 2 pairs using torch_hh.")
    parser.add_argument("--output", type=Path, required=True, help="CSV output for sampled Rbar.")
    parser.add_argument("--num-samples", type=int, default=10_000, help="Samples per pair.")
    parser.add_argument("--seeds", type=_parse_int_list, default=[0, 1, 2, 3, 4], help="Seeds.")
    parser.add_argument(
        "--device",
        default="cuda" if torch is not None and torch.cuda.is_available() else "cpu",
        help="Torch device to use.",
    )
    parser.add_argument("--dtype", default="float32", help="Torch dtype name, e.g. float32.")
    args = parser.parse_args()

    for dim, kappa in PAIRS:
        for seed in args.seeds:
            rbar = _sample_rbar(
                dim=int(dim),
                kappa=float(kappa),
                num_samples=int(args.num_samples),
                seed=int(seed),
                device=args.device,
                dtype=args.dtype,
            )
            _write_csv_row(
                args.output,
                {
                    "p": int(dim),
                    "kappa_true": float(kappa),
                    "seed": int(seed),
                    "num_samples": int(args.num_samples),
                    "rbar": float(rbar),
                },
            )


if __name__ == "__main__":
    main()
