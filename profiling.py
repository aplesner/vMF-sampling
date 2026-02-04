from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.vmf import vMF

try:
    import torch
except ImportError:
    torch = None


class LineTimer:
    def __init__(self, target_files: set[Path]) -> None:
        self.target_files = {path.resolve() for path in target_files}
        self.timings: dict[tuple[Path, int], float] = defaultdict(float)
        self._last_key: tuple[Path, int] | None = None
        self._last_time = 0.0

    def _trace(self, frame, event, arg):  # type: ignore[override]
        if event != "line":
            return self._trace

        now = time.perf_counter()
        if self._last_key is not None:
            self.timings[self._last_key] += now - self._last_time

        filename = Path(frame.f_code.co_filename).resolve()
        if filename in self.target_files:
            self._last_key = (filename, frame.f_lineno)
        else:
            self._last_key = None

        self._last_time = now
        return self._trace

    def __enter__(self) -> "LineTimer":
        self._last_time = time.perf_counter()
        self._last_key = None
        sys.settrace(self._trace)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        sys.settrace(None)
        if self._last_key is not None:
            self.timings[self._last_key] += time.perf_counter() - self._last_time


def _top_lines(timings: dict[tuple[Path, int], float], limit: int = 10) -> list[tuple[Path, int, float]]:
    sorted_items = sorted(timings.items(), key=lambda item: item[1], reverse=True)
    return [(path, lineno, total) for (path, lineno), total in sorted_items[:limit]]


def _line_text(path: Path, lineno: int) -> str:
    lines = path.read_text().splitlines()
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()
    return ""


def _profile_backend(
    name: str,
    sampler: Any,
    target_files: set[Path],
    sample_size: int,
) -> None:
    with LineTimer(target_files) as timer:
        _ = sampler.sample(sample_size)

    print(f"\nBackend: {name}")
    for path, lineno, total in _top_lines(timer.timings, limit=10):
        text = _line_text(path, lineno)
        print(f"{path.name}:{lineno:4d}  {total:10.6f}s  {text}")


def _make_mu_numpy(dim: int) -> np.ndarray:
    mu = np.zeros(dim, dtype=np.float64)
    mu[0] = 1.0
    return mu


def _make_mu_torch(dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mu = torch.zeros((dim,), device=device, dtype=dtype)
    mu[0] = 1.0
    return mu


def _collect_target_files() -> set[Path]:
    base = Path(__file__).resolve().parent / "src"
    return {
        base / "vmf_numpy.py",
        base / "vmf_scipy.py",
        base / "vmf_torch.py",
        base / "vmf_sampler.py",
    }


def _ensure_available() -> set[Path]:
    return {p for p in _collect_target_files() if p.exists()}


def main(args: argparse.Namespace) -> None:
    target_files = _ensure_available()

    mu = _make_mu_numpy(args.dim)
    numpy_sampler = vMF(
        dim=args.dim,
        mu=mu,
        kappa=args.kappa,
        seed=args.seed,
        backend="numpy",
    )
    _profile_backend("numpy", numpy_sampler, target_files, args.sample_size)

    scipy_sampler = vMF(
        dim=args.dim,
        mu=mu,
        kappa=args.kappa,
        seed=args.seed,
        backend="scipy",
    )
    _profile_backend("scipy", scipy_sampler, target_files, args.sample_size)

    if torch is None:
        print("\nBackend: torch (skipped, torch not installed)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    mu_torch = _make_mu_torch(args.dim, device, dtype)
    torch_sampler = vMF(
        dim=args.dim,
        mu=mu_torch,
        kappa=args.kappa,
        seed=args.seed,
        backend="torch",
        device=device,
        dtype=dtype,
    )
    _profile_backend("torch", torch_sampler, target_files, args.sample_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Line timing for vMF sampling backends.")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--kappa", type=float, default=10.0)
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(args)
