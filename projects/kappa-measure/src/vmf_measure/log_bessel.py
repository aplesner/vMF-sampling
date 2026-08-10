"""NumPy ``log(I_order(x))`` path, valid where SciPy's ``ive`` underflows.

This is a NumPy translation of the same piecewise asymptotic/power-series
algorithm implemented in ``src/torch_log_iv.py`` (itself a port of CUSF's
``iv_log.cpp``).  SciPy's ``special.ive`` underflows to exactly zero for
order ~511 and above, which makes the vMF admissibility formula itself
uncomputable in stock SciPy at p >= 4096.  This implementation is validated
against a 50-digit mpmath reference at orders beyond 4095 (p >= 8192).

Float64 only; the estimators subtract two large, nearby log-Bessel values,
which is poorly suited to reduced precision.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln, logsumexp

_LOG_INV_SQRT_2PI = -0.9189385332046727
_LOG_2 = math.log(2.0)

# Coefficients of u_k(t) = t**k * polynomial_k(t**2), in ascending order.
# Debye polynomials used by the uniform asymptotic expansion (same table as
# CUSF's iv_log.cpp and src/torch_log_iv.py).
_U_COEFFICIENTS = (
    (0.125, -0.2083333333333333),
    (0.0703125, -0.40104166666666663, 0.3342013888888889),
    (0.0732421875, -0.8912109375, 1.8464626736111112, -1.0258125964506173),
    (0.112152099609375, -2.3640869140625, 8.78912353515625, -11.207002616222994, 4.669584423426247),
    (0.2271080017089844, -7.368794359479632, 42.53499874538845, -91.81824154324002, 84.63621767460073, -28.21207255820024),
    (0.5725014209747314, -26.49143048695156, 218.1905117442116, -699.5796273761325, 1059.990452528, -765.2524681411816, 212.5701300392171),
    (1.727727502584457, -108.0909197883947, 1200.902913216352, -5305.646978613403, 11655.39333686453, -13586.55000643414, 8061.722181737309, -1919.457662318407),
    (6.074042001273483, -493.915304773088, 7109.514302489364, -41192.65496889755, 122200.4649830175, -203400.1772804155, 192547.0012325315, -96980.59838863751, 20204.29133096615),
    (24.38052969955606, -2499.83048181121, 45218.76898136273, -331645.1724845636, 1.268365273321625e6, -2.813563226586534e6, 3.763271297656404e6, -2.998015918538107e6, 1.311763614662977e6, -242919.1879005513),
    (110.0171402692467, -13886.08975371704, 308186.4046126624, -2.785618128086455e6, 1.328876716642182e7, -3.756717666076335e7, 6.634451227472903e7, -7.410514821153266e7, 5.095260249266464e7, -1.970681911843223e7, 3.284469853072038e6),
    (551.3358961220206, -84005.43360302409, 2.243768177922449e6, -2.447406272573873e7, 1.420629077975331e8, -4.958897842750303e8, 1.106842816823014e9, -1.621080552108337e9, 1.55359689957058e9, -9.394623596815784e8, 3.255730741857657e8, -4.932925366450996e7),
    (3038.090510922384, -549842.3275722887, 1.739510755397816e7, -2.251056618894153e8, 1.559279864879258e9, -6.563293792619284e9, 1.79542137311556e10, -3.302659974980072e10, 4.128018557975397e10, -3.463204338815878e10, 1.868820750929582e10, -5.866481492051847e9, 8.147890961183121e8),
    (18257.75547429317, -3.871833442572613e6, 1.43157876718889e8, -2.167164983223795e9, 1.763473060683497e10, -8.786707217802327e10, 2.879006499061506e11, -6.453648692453765e11, 1.008158106865382e12, -1.098375156081223e12, 8.192186695485773e11, -3.990961752244665e11, 1.144982377320258e11, -1.467926124769562e10),
)

_POWER_SERIES_TERMS = 41


def _debye_u(index: int, t: NDArray[np.float64]) -> NDArray[np.float64]:
    t2 = t * t
    coefficients = _U_COEFFICIENTS[index - 1]
    polynomial = np.full_like(t, coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        polynomial = polynomial * t2 + coefficient
    return t**index * polynomial


def _mu_expansion(order: NDArray[np.float64], x: NDArray[np.float64], terms: int) -> NDArray[np.float64]:
    mu = 4.0 * order * order
    current = np.ones_like(x)
    total = current
    for c in range(1, terms + 1):
        odd = 2 * c - 1
        current = current * (-(mu - odd * odd) / (c * 8.0 * x))
        total = total + current
    return x + _LOG_INV_SQRT_2PI - 0.5 * np.log(x) + np.log(np.abs(total))


def _uniform_expansion(order: NDArray[np.float64], x: NDArray[np.float64], terms: int) -> NDArray[np.float64]:
    scaled_x = x / order
    root = np.sqrt(1.0 + scaled_x * scaled_x)
    t = 1.0 / root
    eta = root + np.log(scaled_x / (1.0 + root))
    correction = np.ones_like(x)
    order_power = np.ones_like(order)
    for index in range(1, terms + 1):
        order_power = order_power * order
        correction = correction + _debye_u(index, t) / order_power
    return (
        _LOG_INV_SQRT_2PI
        - 0.5 * np.log(order)
        + order * eta
        - 0.25 * np.log1p(scaled_x * scaled_x)
        + np.log(np.abs(correction))
    )


def _power_series(order: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
    log_x2_over_4 = 2.0 * np.log(x) - 2.0 * _LOG_2
    indices = np.arange(_POWER_SERIES_TERMS, dtype=np.float64)
    shape = (_POWER_SERIES_TERMS,) + (1,) * x.ndim
    indices = indices.reshape(shape)
    log_terms = (
        indices * log_x2_over_4[None, ...]
        - gammaln(indices + 1.0)
        - gammaln(order[None, ...] + indices + 1.0)
    )
    return order * (np.log(x) - _LOG_2) + logsumexp(log_terms, axis=0)


def log_iv(order: ArrayLike, x: ArrayLike) -> NDArray[np.float64]:
    """Approximate ``log(I_order(x))`` for non-negative order and argument.

    Inputs are broadcast to a common shape.  Returns NaN for negative order
    or argument, -inf for x == 0 with order != 0, and 0 for x == order == 0.
    """

    order_arr, x_arr = np.broadcast_arrays(
        np.asarray(order, dtype=np.float64), np.asarray(x, dtype=np.float64)
    )
    order_arr = order_arr.copy()
    x_arr = x_arr.copy()

    invalid = (order_arr < 0.0) | (x_arr < 0.0)
    tiny = np.finfo(np.float64).tiny
    safe_x = np.maximum(x_arr, tiny)
    safe_order = np.maximum(order_arr, tiny)
    log_x = np.log(safe_x)
    log_order = np.log(safe_order)

    # The np.where branches below are evaluated unconditionally; divisions by
    # a zero order or logs of zero occur in discarded branches only.
    with np.errstate(divide="ignore", invalid="ignore"):
        return _select_branch(order_arr, x_arr, safe_order, safe_x, log_x, log_order, invalid)


def _select_branch(order_arr, x_arr, safe_order, safe_x, log_x, log_order, invalid):
    result = _power_series(order_arr, safe_x)
    uniform_13 = (safe_x > 19.6931) & (order_arr > 0.7) | (order_arr > 12.6964)
    uniform_9 = (safe_x > 35.9074) & (order_arr > 0.6) | (order_arr > 20.1534)
    uniform_6 = (safe_x > 84.4153) & (order_arr > 0.46) | (order_arr > 56.9971)
    uniform_4 = (safe_x > 274.2377) & (order_arr > 0.3) | (order_arr > 163.6993)
    mu_20 = ((safe_x > 30.0) & (order_arr < 15.3919)) | (
        ((0.5113 * log_x + 0.7939) > log_order) & (safe_x > 59.6925)
    )
    mu_3 = ((safe_x > 1.4e3) & (order_arr < 3.05)) | (
        ((0.6229 * log_x - 3.2318) > log_order) & (order_arr > 3.1)
    )

    # Reverse application reproduces the reference if/else priority.
    result = np.where(uniform_13, _uniform_expansion(order_arr, safe_x, 13), result)
    result = np.where(uniform_9, _uniform_expansion(order_arr, safe_x, 9), result)
    result = np.where(uniform_6, _uniform_expansion(order_arr, safe_x, 6), result)
    result = np.where(uniform_4, _uniform_expansion(order_arr, safe_x, 4), result)
    result = np.where(mu_20, _mu_expansion(order_arr, safe_x, 20), result)
    result = np.where(mu_3, _mu_expansion(order_arr, safe_x, 3), result)
    result = np.where((x_arr == 0.0) & (order_arr == 0.0), 0.0, result)
    result = np.where((x_arr == 0.0) & (order_arr != 0.0), -np.inf, result)
    return np.where(invalid, np.nan, result)


def bessel_ratio_ap(kappa: ArrayLike, dimension: ArrayLike) -> NDArray[np.float64]:
    """``A_p(kappa) = I_{p/2}(kappa) / I_{p/2-1}(kappa)`` via log differences."""

    kappa_arr, p = np.broadcast_arrays(
        np.asarray(kappa, dtype=np.float64), np.asarray(dimension, dtype=np.float64)
    )
    nu = p / 2.0 - 1.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        ratio = np.exp(log_iv(nu + 1.0, kappa_arr) - log_iv(nu, kappa_arr))
    return np.where(kappa_arr == 0.0, 0.0, ratio)


def bessel_ratio_ap_deriv(kappa: ArrayLike, dimension: ArrayLike) -> NDArray[np.float64]:
    """Derivative ``A_p'(kappa) = 1 - A_p^2 - ((p-1)/kappa) A_p``.

    Equals the per-sample Fisher information for kappa.  At kappa = 0 the
    limiting value ``1/p`` is returned.
    """

    kappa_arr, p = np.broadcast_arrays(
        np.asarray(kappa, dtype=np.float64), np.asarray(dimension, dtype=np.float64)
    )
    safe_kappa = np.where(kappa_arr > 0.0, kappa_arr, 1.0)
    ratio = bessel_ratio_ap(safe_kappa, p)
    deriv = 1.0 - ratio * ratio - ((p - 1.0) / safe_kappa) * ratio
    return np.where(kappa_arr > 0.0, deriv, 1.0 / p)
