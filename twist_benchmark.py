from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import tqdm

DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DIMS = [3, 4, 15, 16]
DTYPES = [np.float32, np.float64]
PROBE_SIZE = 100

AGGREGATE_SEEDS = True
DISPLAY_MEAN_STD = True


def get_householder_rotation(x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Returns a function that applies the Double Householder rotation."""
    if np.allclose(x, y):
        return lambda s: s
    if np.allclose(x, -y):
        # Map x -> -x via a planar rotation with an auxiliary axis.
        u1 = x
        u2 = np.zeros_like(x)
        axis = 1 if x.shape[0] > 1 else 0
        u2[axis] = 1.0
    else:
        u1 = x / np.linalg.norm(x)
        v2 = y + x
        u2 = v2 / np.linalg.norm(v2)

    def rotate(S: np.ndarray) -> np.ndarray:
        is_single = S.ndim == 1
        S_arr = np.atleast_2d(S)
        S_temp = S_arr - 2 * np.outer(S_arr @ u1, u1)
        S_rotated = S_temp - 2 * np.outer(S_temp @ u2, u2)
        return S_rotated.squeeze() if is_single else S_rotated

    return rotate


def get_qr_rotation(x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Returns a function that applies the QR-based rotation."""
    n = len(x)
    M = np.eye(n, dtype=x.dtype)
    M[:, 0] = y
    Q, _ = np.linalg.qr(M)

    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    if np.dot(Q[:, 0], y) < 0:
        Q[:, 0] = -Q[:, 0]
        Q[:, -1] = -Q[:, -1]

    def rotate(S: np.ndarray) -> np.ndarray:
        return np.dot(S, Q.T)

    return rotate


def measure_twist(
    rotation_func: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    probe_size: int,
) -> tuple[float, float]:
    """
    Measures how much a rotation twists the space perpendicular to the rotation plane.
    Returns the mean and std angle of distortion in degrees.
    """
    n = len(x)
    random_vectors = rng.standard_normal((probe_size, n)).astype(x.dtype, copy=False)

    e1 = x / np.linalg.norm(x)
    v2 = y - np.dot(y, e1) * e1
    if np.linalg.norm(v2) < 1e-10:
        e2 = np.zeros_like(x)
        axis = 0 if np.abs(e1[0]) < 0.9 else 1
        axis = axis if axis < e2.shape[0] else 0
        e2[axis] = 1.0
        e2 = e2 - np.dot(e2, e1) * e1
        e2 /= np.linalg.norm(e2)
    else:
        e2 = v2 / np.linalg.norm(v2)

    proj = np.outer(random_vectors @ e1, e1) + np.outer(random_vectors @ e2, e2)
    z = random_vectors - proj
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    z = np.divide(z, norms, out=np.zeros_like(z), where=norms > 0)

    z_rotated = rotation_func(z)
    dot_prods = np.clip(np.sum(z * z_rotated, axis=1), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot_prods))
    return float(angle_deg.mean()), float(angle_deg.std())


def _make_x_y(
    dim: int,
    dtype: np.dtype,
    rng: np.random.Generator,
    y_is_minus_x: bool,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros(dim, dtype=dtype)
    x[0] = 1.0
    if y_is_minus_x:
        y = -x
    else:
        y = rng.standard_normal(dim).astype(dtype, copy=False)
        y /= np.linalg.norm(y)
    return x, y


def _run_method(
    method: str,
    rotator: Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray]],
    dtypes: list[np.dtype],
    dims: Iterable[int],
    seeds: Iterable[int],
    y_is_minus_x: bool,
    on_row: Callable[[dict[str, Any]], None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dtype in tqdm.tqdm(dtypes, desc=f"Running twist benchmark for {method}"):
        for dim in dims:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                x, y = _make_x_y(dim, dtype, rng, y_is_minus_x)
                rotation = rotator(x, y)
                twist_mean, twist_std = measure_twist(rotation, x, y, rng, PROBE_SIZE)
                row = {
                    "method": method,
                    "dtype": str(dtype),
                    "dim": dim,
                    "seed": seed,
                    "mode": "y=-x" if y_is_minus_x else "y=random",
                    "twist_deg": twist_mean,
                    "probe_std_deg": twist_std,
                }
                on_row(row)
                rows.append(row)
    return rows


def _format_mean_std(df: pd.DataFrame, label: str) -> pd.DataFrame:
    mean = df[f"{label}_mean"]
    std = df[f"{label}_std"]
    formatted = mean.map("{:.6f}".format) + " ± " + std.map("{:.6f}".format)
    return df.assign(**{label: formatted}).drop(columns=[f"{label}_mean", f"{label}_std"])


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
    parser = argparse.ArgumentParser(description="Benchmark twist for rotations.")
    parser.add_argument("--seeds", type=lambda v: [int(x) for x in v.split(",") if x], default=None)
    parser.add_argument("--dims", type=lambda v: [int(x) for x in v.split(",") if x], default=None)
    parser.add_argument(
        "--mode",
        choices=["y=-x", "y=random", "both"],
        default="both",
        help="Which y construction to use.",
    )
    parser.add_argument("--output", type=Path, help="CSV output path for incremental results.")
    parser.add_argument("--summary", action="store_true", help="Print summary table at end.")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else DEFAULT_SEEDS
    dims = args.dims if args.dims is not None else DIMS
    modes = [args.mode] if args.mode in ("y=-x", "y=random") else ["y=-x", "y=random"]

    rows: list[dict[str, Any]] | None = [] if (args.summary or args.output is None) else None
    def on_row(row: dict[str, Any]) -> None:
        if args.output is not None:
            _write_csv_row(args.output, row)
        if rows is not None:
            rows.append(row)

    methods = [
        ("householder", get_householder_rotation),
        ("qr", get_qr_rotation),
    ]
    for mode in modes:
        y_is_minus_x = mode == "y=-x"
        for method, rotator in methods:
            _run_method(method, rotator, DTYPES, dims, seeds, y_is_minus_x, on_row)

    if rows is None:
        return

    df = pd.DataFrame(rows)
    df = df.set_index(["method", "dtype", "dim", "seed", "mode"]).sort_index()

    if not AGGREGATE_SEEDS:
        print(df)
        return

    grouped = df.groupby(level=["method", "dtype", "dim", "mode"])
    stats = grouped.agg(
        twist_deg_mean=("twist_deg", "mean"),
        twist_deg_std=("twist_deg", "std"),
        probe_std_deg_mean=("probe_std_deg", "mean"),
        probe_std_deg_std=("probe_std_deg", "std"),
    )

    if DISPLAY_MEAN_STD:
        stats = _format_mean_std(stats, "twist_deg")
        stats = _format_mean_std(stats, "probe_std_deg")
    print(stats)


if __name__ == "__main__":
    main()
