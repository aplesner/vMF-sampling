"""Concentration-parameter estimators for the von Mises--Fisher distribution.

The estimators operate on the mean resultant length ``r_bar``.  Computing
``r_bar`` from samples is deliberately separate so estimator benchmarks can
exclude sample generation and reduction costs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, special

LogIv = Callable[[ArrayLike, ArrayLike], NDArray[np.float64]]


@dataclass(frozen=True)
class NewtonResult:
    """Result from the converged, safeguarded Newton estimator."""

    kappa: NDArray[np.float64]
    iterations: NDArray[np.int64]
    converged: NDArray[np.bool_]


def mean_resultant_length(samples: ArrayLike, axis: int = -2) -> NDArray[np.float64]:
    """Return ``||mean(samples)||`` along the sample axis."""

    values = np.asarray(samples)
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


def scipy_log_iv(order: ArrayLike, x: ArrayLike) -> NDArray[np.float64]:
    """Stable SciPy implementation of ``log(I_order(x))`` using scaled ``ive``."""

    order_values, x_values = np.broadcast_arrays(
        np.asarray(order, dtype=np.float64), np.asarray(x, dtype=np.float64)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(special.ive(order_values, x_values)) + np.abs(x_values)


def bessel_ratio_from_logs(
    kappa: ArrayLike,
    dimension: ArrayLike,
    log_iv: LogIv,
) -> NDArray[np.float64]:
    """Compute ``A_p(kappa) = I_{p/2}(kappa) / I_{p/2-1}(kappa)``."""

    kappa_values, p = np.broadcast_arrays(
        np.asarray(kappa, dtype=np.float64), np.asarray(dimension, dtype=np.float64)
    )
    nu = p / 2.0 - 1.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        ratio = np.exp(log_iv(nu + 1.0, kappa_values) - log_iv(nu, kappa_values))
    return np.where(kappa_values == 0.0, 0.0, ratio)


def _newton_step(
    kappa: NDArray[np.float64],
    r_bar: NDArray[np.float64],
    dimension: NDArray[np.float64],
    log_iv: LogIv,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    ratio = bessel_ratio_from_logs(kappa, dimension, log_iv)
    derivative = 1.0 - ratio * ratio - ((dimension - 1.0) / kappa) * ratio
    return kappa - (ratio - r_bar) / derivative, ratio


def sra_kappa(
    r_bar: ArrayLike,
    dimension: ArrayLike,
    log_iv: LogIv = scipy_log_iv,
) -> NDArray[np.float64]:
    """Sra's (2012) approximation: Banerjee initialization and two Newton steps."""

    r, p = _validate_inputs(r_bar, dimension)
    kappa = banerjee_kappa(r, p)
    active = r > 0.0
    safe_kappa = np.where(active, kappa, 1.0)
    for _ in range(2):
        candidate, _ = _newton_step(safe_kappa, r, p, log_iv)
        safe_kappa = np.where(active, candidate, safe_kappa)
    return np.where(active, safe_kappa, 0.0)


def converged_newton_kappa(
    r_bar: ArrayLike,
    dimension: ArrayLike,
    log_iv: LogIv = scipy_log_iv,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_iterations: int = 50,
) -> NewtonResult:
    """Solve the likelihood equation with safeguarded Newton iterations.

    Sra's lower and upper bounds bracket the MLE.  A Newton proposal outside
    that interval, or one based on an invalid derivative, is replaced by a
    bisection step.
    """

    r, p = _validate_inputs(r_bar, dimension)
    denominator = 1.0 - r * r
    lower = np.maximum(0.0, r * (p - 2.0) / denominator)
    upper = r * p / denominator
    kappa = banerjee_kappa(r, p)
    active = r > 0.0
    iterations = np.zeros(r.shape, dtype=np.int64)
    converged = ~active

    for _ in range(max_iterations):
        pending = active & ~converged
        if not np.any(pending):
            break

        safe_kappa = np.where(pending, kappa, 1.0)
        ratio = bessel_ratio_from_logs(safe_kappa, p, log_iv)
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
        iterations += pending

    return NewtonResult(np.where(active, kappa, 0.0), iterations, converged)


def scipy_ratio(kappa: float, dimension: int) -> float:
    """SciPy's scaled-Bessel ratio, used only by the CPU reference solver."""

    if kappa == 0.0:
        return 0.0
    nu = dimension / 2.0 - 1.0
    denominator = special.ive(nu, kappa)
    numerator = special.ive(nu + 1.0, kappa)
    if denominator == 0.0 or not np.isfinite(denominator) or not np.isfinite(numerator):
        return float("nan")
    return float(numerator / denominator)


def scipy_brentq_kappa(r_bar: ArrayLike, dimension: ArrayLike) -> NDArray[np.float64]:
    """High-accuracy CPU reference obtained by bracketing the likelihood score."""

    r, p = _validate_inputs(r_bar, dimension)
    output = np.empty(r.shape, dtype=np.float64)
    for index in np.ndindex(r.shape):
        target = float(r[index])
        dim = int(p[index])
        if target == 0.0:
            output[index] = 0.0
            continue
        upper = target * dim / (1.0 - target * target)
        output[index] = optimize.brentq(
            lambda value: scipy_ratio(value, dim) - target,
            0.0,
            upper,
            xtol=1e-13,
            rtol=4.0 * np.finfo(np.float64).eps,
            maxiter=200,
        )
    return output


def direct_log_likelihood_kappa(
    r_bar: ArrayLike,
    dimension: ArrayLike,
    log_iv: LogIv = scipy_log_iv,
) -> NDArray[np.float64]:
    """Directly maximize the scalar log likelihood with bounded Brent search.

    This is intentionally independent of the likelihood-score root solvers.  It
    is primarily a validation method rather than the expected fastest estimator.
    """

    r, p = _validate_inputs(r_bar, dimension)
    output = np.empty(r.shape, dtype=np.float64)
    tiny = np.sqrt(np.finfo(np.float64).tiny)
    for index in np.ndindex(r.shape):
        target = float(r[index])
        dim = int(p[index])
        if target == 0.0:
            output[index] = 0.0
            continue
        nu = dim / 2.0 - 1.0
        upper = target * dim / (1.0 - target * target)

        def negative_log_likelihood(value: float) -> float:
            log_normalizer_without_constant = nu * np.log(value) - float(
                log_iv(nu, value)
            )
            return -(log_normalizer_without_constant + value * target)

        result = optimize.minimize_scalar(
            negative_log_likelihood,
            bounds=(tiny, upper),
            method="bounded",
            options={"xatol": 1e-11, "maxiter": 500},
        )
        if not result.success:
            raise RuntimeError(f"likelihood optimization failed: {result.message}")
        output[index] = result.x
    return output
