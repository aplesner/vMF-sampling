"""Matched null quantiles for geometry statistics.

Three null models, defined precisely in ``projects/kappa-measure/SCHEMA.md``:

- ``uniform``: iid uniform on S^{p-1}.  Closed-form quantiles for ``r_bar``
  (chi-square approximation, exact E[Rbar^2] = 1/n) and ``mean_cosine``
  (Gaussian); ``kappa_hat`` quantiles follow by monotone transformation.
- ``mean_removed``: iid uniform on S^{p-1}, but each observation has the
  sample-mean direction projected out and is renormalized to the sphere
  before the statistic is computed.  Null for claims about residual
  concentration beyond a single dominant direction.  Monte Carlo only.
- ``covariance_matched``: Gaussian samples whose covariance matches the
  empirical covariance of a supplied dataset, normalized to the sphere.
  Separates directional concentration from anisotropic scale.  Monte Carlo
  only; requires ``samples``.

Every kappa-hat claim must ship with one of these matched nulls: the
finite-sample bias is systematically upward, the same direction as any
"more concentrated than uniform" claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2, norm

from .kappa import mle_kappa

STATISTICS = ("r_bar", "kappa_hat", "mean_cosine")
NULLS = ("uniform", "mean_removed", "covariance_matched")


@dataclass(frozen=True)
class NullQuantile:
    """One null-quantile row; field order matches ``null_quantiles.csv``."""

    statistic: str
    null: str
    p: int
    n: int
    quantile: float
    value: float
    method: str  # "closed_form" | "monte_carlo"
    reps: int | None
    seed: int | None


def _check_choices(statistic: str, null: str) -> None:
    if statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of {STATISTICS}, got {statistic!r}")
    if null not in NULLS:
        raise ValueError(f"null must be one of {NULLS}, got {null!r}")


def _statistic_from_samples(
    statistic: str, draws: NDArray[np.float64], p: int
) -> NDArray[np.float64]:
    """Statistic of draws shaped (..., n, p), reduced over the sample axis."""

    if statistic == "mean_cosine":
        # Fixed reference direction e1; under the uniform null the choice is
        # immaterial.
        return np.mean(draws[..., 0], axis=-1)
    r_bar = np.linalg.norm(np.mean(draws, axis=-2), axis=-1)
    if statistic == "r_bar":
        return r_bar
    return mle_kappa(r_bar, p)


def _uniform_draws(
    rng: np.random.Generator, reps: int, n: int, p: int
) -> NDArray[np.float64]:
    draws = rng.standard_normal((reps, n, p))
    draws /= np.linalg.norm(draws, axis=-1, keepdims=True)
    return draws


def _mean_removed_draws(
    rng: np.random.Generator, reps: int, n: int, p: int
) -> NDArray[np.float64]:
    draws = _uniform_draws(rng, reps, n, p)
    mean_direction = np.mean(draws, axis=-2, keepdims=True)
    norm = np.linalg.norm(mean_direction, axis=-1, keepdims=True)
    mean_direction = mean_direction / np.maximum(norm, np.finfo(np.float64).tiny)
    residual = draws - np.sum(draws * mean_direction, axis=-1, keepdims=True) * mean_direction
    residual_norm = np.linalg.norm(residual, axis=-1, keepdims=True)
    return residual / np.maximum(residual_norm, np.finfo(np.float64).tiny)



def null_quantile(
    statistic: str,
    p: int,
    n: int,
    q: float,
    *,
    null: str = "uniform",
    samples: ArrayLike | None = None,
    reps: int = 2000,
    seed: int | None = None,
) -> NullQuantile:
    """Quantile ``q`` of ``statistic`` under ``null`` at (n, p).

    Uniform-null quantiles are closed form.  ``mean_removed`` and
    ``covariance_matched`` are Monte Carlo with ``reps`` replicates;
    ``covariance_matched`` requires the observed ``samples`` (n, p).
    """

    _check_choices(statistic, null)
    p = int(p)
    n = int(n)
    q = float(q)
    if p < 2 or n < 1:
        raise ValueError("require p >= 2 and n >= 1")
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie in (0, 1)")

    if null == "uniform":
        if statistic == "r_bar":
            value = float(np.sqrt(chi2.ppf(q, p) / (n * p)))
        elif statistic == "mean_cosine":
            value = float(norm.ppf(q) / np.sqrt(n * p))
        else:  # kappa_hat: monotone transform of r_bar
            r_q = float(np.sqrt(chi2.ppf(q, p) / (n * p)))
            value = float(mle_kappa(min(r_q, 1.0 - 1e-15), p))
        return NullQuantile(statistic, null, p, n, q, value, "closed_form", None, None)

    if statistic == "mean_cosine":
        raise ValueError(
            "mean_cosine is only defined under the uniform null; "
            "mean_removed and covariance_matched support r_bar and kappa_hat"
        )

    if null == "covariance_matched" and samples is None:
        raise ValueError("covariance_matched requires the observed samples")
    sample_arr = None
    if null == "covariance_matched":
        sample_arr = np.asarray(samples, dtype=np.float64)
        if sample_arr.shape != (n, p):
            raise ValueError(
                f"samples must have shape (n, p) = {(n, p)}, got {sample_arr.shape}"
            )
        # Precompute the Cholesky factor once for all chunks.
        covariance = np.cov(sample_arr, rowvar=False)
        covariance += 1e-12 * (np.trace(covariance) / p) * np.eye(p)
        chol = np.linalg.cholesky(covariance)

    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    chunk = max(1, int(2e8 / (n * p)))
    values = np.empty(reps)
    done = 0
    while done < reps:
        take = min(chunk, reps - done)
        if null == "mean_removed":
            draws = _mean_removed_draws(rng, take, n, p)
        else:
            draws = rng.standard_normal((take, n, p)) @ chol.T
            draws /= np.linalg.norm(draws, axis=-1, keepdims=True)
        values[done : done + take] = _statistic_from_samples(statistic, draws, p)
        done += take

    value = float(np.quantile(values, q))
    return NullQuantile(statistic, null, p, n, q, value, "monte_carlo", reps, seed)


def null_quantile_table(
    p_values: list[int],
    n_values: list[int],
    quantiles: list[float],
    *,
    statistics: tuple[str, ...] = STATISTICS,
    null: str = "uniform",
    samples: ArrayLike | None = None,
    reps: int = 2000,
    seed: int | None = None,
) -> list[NullQuantile]:
    """Cartesian grid of null quantiles, as rows ready for ``null_quantiles.csv``."""

    rows: list[NullQuantile] = []
    for statistic in statistics:
        for p in p_values:
            for n in n_values:
                for q in quantiles:
                    rows.append(
                        null_quantile(
                            statistic,
                            p,
                            n,
                            q,
                            null=null,
                            samples=samples,
                            reps=reps,
                            seed=seed,
                        )
                    )
    return rows
