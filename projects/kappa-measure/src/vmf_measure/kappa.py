"""Concentration estimation with a refusal mode.

``kappa_hat`` is the user-facing entry point.  By default it operates in
refusal mode: when the closed-form feasibility surface predicts a relative
RMSE above ``target_rel_rmse`` at the supplied ``n``, it returns the status
string ``"not resolvable at this n"`` and no number, rather than shipping a
point estimate whose error is dominated by finite-sample bias.

The solver itself is the safeguarded Newton iteration on the likelihood
score ``A_p(kappa) = Rbar`` (Sra 2012 brackets, bisection fallback), backed
by the log-Bessel path in :mod:`vmf_measure.log_bessel`, which stays finite
at p >= 8192 where SciPy's ``ive`` underflows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .log_bessel import bessel_ratio_ap

REFUSAL_STATUS = "not resolvable at this n"
OK_STATUS = "ok"

DEFAULT_TARGET_REL_RMSE = 0.01


@dataclass(frozen=True)
class KappaEstimate:
    """Result of :func:`kappa_hat` for a single dataset.

    ``status`` is exactly one of ``"ok"`` or ``"not resolvable at this n"``.
    When refused, ``kappa`` is None; ``rel_rmse`` still carries the predicted
    surface value so callers can see how far below the boundary they are.
    """

    status: str
    kappa: float | None
    r_bar: float
    p: int
    n: int | None
    rel_rmse: float | None
    target_rel_rmse: float | None


def mean_resultant_length(samples: ArrayLike, axis: int = -2) -> NDArray[np.float64]:
    """Return ``||mean(samples)||`` along the sample axis."""

    values = np.asarray(samples, dtype=np.float64)
    return np.linalg.norm(np.mean(values, axis=axis), axis=-1)


def _validate_inputs(r_bar: ArrayLike, dimension: ArrayLike) -> tuple[NDArray, NDArray]:
    r = np.asarray(r_bar, dtype=np.float64)
    p = np.asarray(dimension, dtype=np.float64)
    r, p = np.broadcast_arrays(r, p)
    if np.any(~np.isfinite(r)) or np.any((r < 0.0) | (r >= 1.0)):
        raise ValueError("r_bar must be finite and satisfy 0 <= r_bar < 1")
    if np.any(~np.isfinite(p)) or np.any(p < 2.0) or np.any(p != np.floor(p)):
        raise ValueError("dimension must contain integers greater than or equal to 2")
    return r, p


def banerjee_kappa(r_bar: ArrayLike, dimension: ArrayLike) -> NDArray[np.float64]:
    """Banerjee et al. (2005) closed-form approximation (Sra Eq. 4)."""

    r, p = _validate_inputs(r_bar, dimension)
    return r * (p - r * r) / (1.0 - r * r)


def mle_kappa(
    r_bar: ArrayLike,
    dimension: ArrayLike,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_iterations: int = 50,
) -> NDArray[np.float64]:
    """Joint-MLE concentration from the mean resultant length.

    Safeguarded Newton on ``A_p(kappa) = Rbar`` with Sra's analytical
    bracket; out-of-bracket or invalid proposals fall back to bisection.
    Vectorized over broadcast inputs.  Returns 0 where r_bar == 0.
    """

    r, p = _validate_inputs(r_bar, dimension)
    denominator = 1.0 - r * r
    lower = np.maximum(0.0, r * (p - 2.0) / denominator)
    upper = r * p / denominator
    kappa = banerjee_kappa(r, p)
    active = r > 0.0
    converged = ~active

    for _ in range(max_iterations):
        pending = active & ~converged
        if not np.any(pending):
            break

        safe_kappa = np.where(pending, kappa, 1.0)
        ratio = bessel_ratio_ap(safe_kappa, p)
        score = ratio - r
        just_converged = pending & np.isfinite(score) & (
            np.abs(score) <= atol + rtol * np.maximum(r, 1.0)
        )
        converged |= just_converged
        pending &= ~just_converged
        if not np.any(pending):
            break

        lower = np.where(pending & (score < 0.0), kappa, lower)
        upper = np.where(pending & (score >= 0.0), kappa, upper)
        derivative = 1.0 - ratio * ratio - ((p - 1.0) / safe_kappa) * ratio
        proposal = kappa - score / derivative
        use_bisection = (
            ~np.isfinite(proposal)
            | ~np.isfinite(derivative)
            | (derivative <= 0.0)
            | (proposal <= lower)
            | (proposal >= upper)
        )
        proposal = np.where(use_bisection, 0.5 * (lower + upper), proposal)
        kappa = np.where(pending, proposal, kappa)

    return np.where(active, kappa, 0.0)


def kappa_hat(
    r_bar: float,
    p: int,
    n: int | None = None,
    *,
    target_rel_rmse: float = DEFAULT_TARGET_REL_RMSE,
    refuse: bool = True,
) -> KappaEstimate:
    """Estimate concentration for one dataset, with refusal mode as default.

    Parameters
    ----------
    r_bar : float
        Mean resultant length of the dataset, in [0, 1).
    p : int
        Ambient dimension.
    n : int or None
        Sample count ``r_bar`` was computed from.  Required for the
        admissibility check; without it no refusal can occur and the
        estimate is returned with ``rel_rmse=None``.
    target_rel_rmse : float
        Resolvability threshold on the predicted relative RMSE.
    refuse : bool
        When True (default), return status ``"not resolvable at this n"``
        and ``kappa=None`` where the surface exceeds the target.  When
        False, always return the point estimate and let the caller read
        ``rel_rmse``.
    """

    r_value = float(np.asarray(r_bar, dtype=np.float64))
    p_value = int(p)
    _validate_inputs(r_value, p_value)

    estimate = float(mle_kappa(r_value, p_value))

    if n is None:
        return KappaEstimate(
            status=OK_STATUS,
            kappa=estimate,
            r_bar=r_value,
            p=p_value,
            n=None,
            rel_rmse=None,
            target_rel_rmse=None,
        )

    from .admissibility import rel_rmse_surface  # avoid circular import

    n_value = int(n)
    if n_value <= 0:
        raise ValueError("n must be positive")
    surface = float(rel_rmse_surface(r_value, p_value, n_value))

    if refuse and surface > target_rel_rmse:
        return KappaEstimate(
            status=REFUSAL_STATUS,
            kappa=None,
            r_bar=r_value,
            p=p_value,
            n=n_value,
            rel_rmse=surface,
            target_rel_rmse=target_rel_rmse,
        )
    return KappaEstimate(
        status=OK_STATUS,
        kappa=estimate,
        r_bar=r_value,
        p=p_value,
        n=n_value,
        rel_rmse=surface,
        target_rel_rmse=target_rel_rmse,
    )
