from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
import tqdm

SEEDS = [0, 1, 2]
DIMS = [3, 4, 15, 16]
DTYPES = [np.float32, np.float64]
PROBE_SIZE = 100

Y_IS_MINUS_X = True

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


def _make_x_y(dim: int, dtype: np.dtype, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros(dim, dtype=dtype)
    x[0] = 1.0
    if Y_IS_MINUS_X:
        y = -x
    else:
        y = rng.standard_normal(dim).astype(dtype, copy=False)
        y /= np.linalg.norm(y)
    return x, y


def _run_method(
    method: str,
    rotator: Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray]],
    dtypes: list[np.dtype],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dtype in tqdm.tqdm(dtypes, desc=f"Running twist benchmark for {method}"):
        for dim in DIMS:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                x, y = _make_x_y(dim, dtype, rng)
                rotation = rotator(x, y)
                twist_mean, twist_std = measure_twist(rotation, x, y, rng, PROBE_SIZE)
                rows.append(
                    {
                        "method": method,
                        "dtype": str(dtype),
                        "dim": dim,
                        "seed": seed,
                        "twist_deg": twist_mean,
                        "probe_std_deg": twist_std,
                    }
                )
    return rows


def _format_mean_std(df: pd.DataFrame, label: str) -> pd.DataFrame:
    mean = df[f"{label}_mean"]
    std = df[f"{label}_std"]
    formatted = mean.map("{:.6f}".format) + " ± " + std.map("{:.6f}".format)
    return df.assign(**{label: formatted}).drop(columns=[f"{label}_mean", f"{label}_std"])


def main() -> None:
    rows: list[dict[str, Any]] = []

    methods = [
        ("householder", get_householder_rotation),
        ("qr", get_qr_rotation),
    ]
    for method, rotator in methods:
        rows.extend(_run_method(method, rotator, DTYPES))

    df = pd.DataFrame(rows)
    df = df.set_index(["method", "dtype", "dim", "seed"]).sort_index()

    if not AGGREGATE_SEEDS:
        print(df)
        return

    grouped = df.groupby(level=["method", "dtype", "dim"])
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
