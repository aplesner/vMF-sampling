"""Numerically stable vMF KL divergence and concentration geometry, valid at m >= 1024.

All quantities are computed in float64 on top of ``src.torch_log_iv`` (the
pure-PyTorch translation of CUSF's ``log I_v(x)``), because SciPy's ``ive``
underflows to exactly zero above order ~511 and is therefore unusable for
m >= 1024 (shared-constraints measurement rule).

Conventions (Project 4 brief, "Metrics"): an S-VAE with latent dimension ``d``
has ``m = d + 1`` ambient coordinates, i.e. the latent lives on S^{m-1}.

Definitions:
    A_m(kappa) = I_{m/2}(kappa) / I_{m/2 - 1}(kappa)   (mean resultant length rho)
    KL(m, kappa) = KL(vMF(mu, kappa) || Uniform(S^{m-1}))
                 = kappa * A_m(kappa) + log C_m(kappa) + log S_{m-1}
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch.special import gammaln

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.torch_log_iv import _U_COEFFICIENTS, torch_log_iv  # noqa: E402

DTYPE = torch.float64


def _debye_coeff_matrix(terms: int, device, dtype) -> torch.Tensor:
    """Padded (terms, max_degree) matrix of the Debye u_k polynomial coeffs."""
    max_deg = max(len(c) for c in _U_COEFFICIENTS[:terms])
    mat = torch.zeros(terms, max_deg, device=device, dtype=dtype)
    for k, coeffs in enumerate(_U_COEFFICIENTS[:terms]):
        mat[k, : len(coeffs)] = torch.tensor(coeffs, device=device, dtype=dtype)
    return mat


def _log_iv_uniform_fast(order: float, x: torch.Tensor, terms: int) -> torch.Tensor:
    """Vectorized Debye uniform expansion of log I_order(x) for scalar order.

    Same approximation as src.torch_log_iv._uniform_expansion but with the
    Horner loops over the u_k polynomials batched into a single loop over
    polynomial degree: O(degree) kernel launches instead of O(terms*degree).
    """
    nu = torch.as_tensor(order, dtype=x.dtype, device=x.device)
    w = x / nu
    root = torch.sqrt(1.0 + w * w)
    t = 1.0 / root
    eta = root + torch.log(w / (1.0 + root))
    # u_k(t) = t^k * poly_k(t^2); evaluate all poly_k in one Horner pass.
    coeffs = _debye_coeff_matrix(terms, x.device, x.dtype)
    t2 = (t * t).unsqueeze(-1)  # (n, 1)
    poly = torch.zeros(x.shape + (terms,), dtype=x.dtype, device=x.device)
    for c in reversed(range(coeffs.shape[1])):
        poly = poly * t2 + coeffs[:, c]
    ks = torch.arange(1, terms + 1, dtype=x.dtype, device=x.device)
    u_k = t.unsqueeze(-1).pow(ks) * poly
    series = 1.0 + (u_k / nu.pow(ks)).sum(dim=-1)
    return nu * eta - 0.5 * torch.log(2.0 * math.pi * nu * root) + torch.log(series)


def log_iv_fast(order: float, x: torch.Tensor) -> torch.Tensor:
    """log I_order(x) for scalar order, evaluating only the needed branches.

    Branch conditions replicate CUSF's priority exactly (see
    src.torch_log_iv.torch_log_iv); with a scalar order most batches need a
    single branch, avoiding ~5x wasted kernel launches in float64.
    """
    o = float(order)
    safe_x = torch.clamp_min(x, torch.finfo(x.dtype).tiny)
    log_x = torch.log(safe_x)
    log_o = math.log(max(o, torch.finfo(x.dtype).tiny))
    # branch ids in override priority order (later overrides earlier)
    branch = torch.zeros_like(safe_x, dtype=torch.long)  # 0 = power series
    branch = torch.where((safe_x > 19.6931) & (o > 0.7) | (o > 12.6964), torch.full_like(branch, 1), branch)
    branch = torch.where((safe_x > 35.9074) & (o > 0.6) | (o > 20.1534), torch.full_like(branch, 2), branch)
    branch = torch.where((safe_x > 84.4153) & (o > 0.46) | (o > 56.9971), torch.full_like(branch, 3), branch)
    branch = torch.where((safe_x > 274.2377) & (o > 0.3) | (o > 163.6993), torch.full_like(branch, 4), branch)
    branch = torch.where(
        ((safe_x > 30.0) & (o < 15.3919)) | (((0.5113 * log_x + 0.7939) > log_o) & (safe_x > 59.6925)),
        torch.full_like(branch, 5), branch)
    branch = torch.where(
        ((safe_x > 1.4e3) & (o < 3.05)) | (((0.6229 * log_x - 3.2318) > log_o) & (o > 3.1)),
        torch.full_like(branch, 6), branch)

    result = torch.empty_like(safe_x)
    for bid, terms in ((1, 13), (2, 9), (3, 6), (4, 4)):
        mask = branch == bid
        if bool(mask.any()):
            result = torch.where(mask, _log_iv_uniform_fast(o, safe_x, terms), result)
    if bool((branch == 0).any()) or bool((branch >= 5).any()):
        # Rare path: power series or mu expansions — defer to the reference.
        slow = torch_log_iv(torch.as_tensor(o, dtype=x.dtype, device=x.device), safe_x)
        result = torch.where(branch >= 5, slow, result)
        result = torch.where(branch == 0, slow, result)
    result = torch.where(x == 0.0, torch.full_like(result, -torch.inf), result)
    return result


def _as_tensor(value, like: torch.Tensor | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        out = value.to(DTYPE)
    else:
        out = torch.tensor(value, dtype=DTYPE)
    if like is not None and out.device != like.device:
        out = out.to(like.device)
    return out


def log_bessel_iv(order, kappa) -> torch.Tensor:
    """log I_order(kappa), float64, finite for order >= 512 where SciPy fails."""
    kappa_t = _as_tensor(kappa)
    order_t = _as_tensor(order, like=kappa_t)
    order_t, kappa_t = torch.broadcast_tensors(order_t, kappa_t)
    return torch_log_iv(order_t, kappa_t)


def log_surface_area(m) -> torch.Tensor:
    """log of the surface area of the unit sphere S^{m-1}: 2 pi^{m/2} / Gamma(m/2)."""
    m_t = _as_tensor(m)
    return math.log(2.0) + (m_t / 2.0) * math.log(math.pi) - gammaln(m_t / 2.0)


def bessel_ratio_A(kappa, m) -> torch.Tensor:
    """A_m(kappa) = I_{m/2}(kappa) / I_{m/2-1}(kappa), with A_m(0) = 0."""
    kappa_t = _as_tensor(kappa)
    m_t = _as_tensor(m, like=kappa_t)
    kappa_t, m_t = torch.broadcast_tensors(kappa_t, m_t)
    nu = m_t / 2.0 - 1.0
    safe_kappa = torch.clamp(kappa_t, min=1e-300)
    log_ratio = log_bessel_iv(nu + 1.0, safe_kappa) - log_bessel_iv(nu, safe_kappa)
    return torch.where(kappa_t > 0.0, torch.exp(log_ratio), torch.zeros_like(kappa_t))


def bessel_ratio_A_prime(kappa, m) -> torch.Tensor:
    """Riccati identity: A_m'(kappa) = 1 - A^2 - ((m-1)/kappa) A.

    At kappa -> 0, A_m(kappa) ~ kappa/m, so A_m'(0) = 1/m.
    """
    kappa_t = _as_tensor(kappa)
    m_t = _as_tensor(m, like=kappa_t)
    kappa_t, m_t = torch.broadcast_tensors(kappa_t, m_t)
    a = bessel_ratio_A(kappa_t, m_t)
    safe_kappa = torch.clamp(kappa_t, min=1e-12)
    deriv = 1.0 - a * a - ((m_t - 1.0) / safe_kappa) * a
    return torch.where(kappa_t > 1e-12, deriv, 1.0 / m_t)


def log_vmf_log_norm(m, kappa) -> torch.Tensor:
    """log C_m(kappa) with C_m(kappa) = kappa^{m/2-1} / ((2pi)^{m/2} I_{m/2-1}(kappa)).

    Limit at kappa -> 0 is -log S_{m-1} (uniform density).
    """
    kappa_t = _as_tensor(kappa)
    m_t = _as_tensor(m, like=kappa_t)
    m_t, kappa_t = torch.broadcast_tensors(m_t, kappa_t)
    nu = m_t / 2.0 - 1.0
    safe_kappa = torch.clamp(kappa_t, min=1e-300)
    log_c = (
        nu * torch.log(safe_kappa)
        - (m_t / 2.0) * math.log(2.0 * math.pi)
        - log_bessel_iv(nu, safe_kappa)
    )
    uniform_limit = -log_surface_area(m_t)
    return torch.where(kappa_t > 0.0, log_c, uniform_limit)


def kl_vmf_uniform(kappa, m) -> torch.Tensor:
    """KL(vMF(mu, kappa) || Uniform(S^{m-1})); independent of mu, 0 at kappa=0.

    Cancellation control.  kappa*A_m(kappa) and log I_{m/2-1}(kappa) are both
    ~kappa, while their contribution to the KL is only O(m + log kappa).  A is
    computed as exp(log I_{m/2} - log I_{m/2-1}), whose true value 1 - (m-1)/(2k)
    + ... becomes unresolvable once (m-1)/(2k) < ulp(k), i.e. k >~ 5e7 sqrt(m);
    the direct KL then degrades by up to (m-1)/2 nats.  For kappa beyond
    kappa* = 8.2e3 * m^(3/4) (where direct error ~k^2*2^-52 meets the tail
    truncation error ~m^3/k^2) we therefore switch to the large-kappa
    asymptotic form including the first 1/kappa correction:
        KL = (m-1)/2 * (log kappa - log 2pi - 1) + log S_{m-1}
             + (m-1)(m-3)/(4 kappa) + O(m^3/kappa^2).
    The worst error at the switch is ~m^1.5/6.7e7 nats (5e-4 at m=1024).
    """
    kappa_t = _as_tensor(kappa)
    m_t = _as_tensor(m, like=kappa_t)
    kappa_t, m_t = torch.broadcast_tensors(kappa_t, m_t)
    log_s = log_surface_area(m_t)
    kl_direct = (
        kappa_t * bessel_ratio_A(kappa_t, m_t)
        + log_vmf_log_norm(m_t, kappa_t)
        + log_s
    )
    switch = 8.2e3 * m_t.pow(0.75)
    kl_asymp = (
        0.5 * (m_t - 1.0) * (torch.log(kappa_t) - math.log(2.0 * math.pi) - 1.0)
        + log_s
        + (m_t - 1.0) * (m_t - 3.0) / (4.0 * kappa_t)
    )
    kl = torch.where(kappa_t > switch, kl_asymp, kl_direct)
    return torch.where(kappa_t > 0.0, kl, torch.zeros_like(kappa_t))


def kl_grad_kappa(kappa, m) -> torch.Tensor:
    """d KL / d kappa = kappa * A_m'(kappa) (Davidson et al. eq. 6, simplified)."""
    kappa_t = _as_tensor(kappa)
    m_t = _as_tensor(m, like=kappa_t)
    kappa_t, m_t = torch.broadcast_tensors(kappa_t, m_t)
    return kappa_t * bessel_ratio_A_prime(kappa_t, m_t)


def invert_A(rho, m, *, tol: float = 1e-12, max_iter: int = 200) -> torch.Tensor:
    """Solve A_m(kappa) = rho for kappa >= 0 by safeguarded Newton (Sra bracket).

    A_m is strictly increasing in kappa, so the inverse is unique.
    """
    rho_t = _as_tensor(rho)
    m_t = _as_tensor(m, like=rho_t)
    rho_t, m_t = torch.broadcast_tensors(rho_t, m_t)
    if torch.any((rho_t < 0.0) | (rho_t >= 1.0)):
        raise ValueError("rho must lie in [0, 1)")
    denom = torch.clamp(1.0 - rho_t * rho_t, min=1e-300)
    lower = rho_t * torch.clamp(m_t - 2.0, min=1e-9) / denom
    upper = rho_t * m_t / denom
    # Banerjee initialisation.
    kappa = rho_t * (m_t - rho_t * rho_t) / denom
    kappa = torch.where(rho_t > 0.0, kappa, torch.zeros_like(kappa))
    active = rho_t > 0.0
    for _ in range(max_iter):
        safe = torch.clamp(kappa, min=1e-300)
        a = bessel_ratio_A(safe, m_t)
        deriv = bessel_ratio_A_prime(safe, m_t)
        step = (a - rho_t) / torch.clamp(deriv, min=1e-300)
        candidate = safe - step
        # Safeguard: keep the iterate inside Sra's bracket.
        candidate = torch.where(
            (candidate <= lower) | (candidate >= upper) | ~torch.isfinite(candidate),
            0.5 * (lower + upper),
            candidate,
        )
        candidate = torch.clamp(candidate, min=lower * (1 - 1e-12), max=upper * (1 + 1e-12))
        converged = torch.abs(candidate - safe) <= tol * torch.clamp(safe, min=1.0)
        kappa = torch.where(active & ~converged, candidate, kappa)
        if bool(torch.all(~active | converged)):
            break
    return torch.where(active, kappa, torch.zeros_like(kappa))


def invert_kl(kl_target, m, *, tol: float = 1e-10, max_iter: int = 200) -> torch.Tensor:
    """Solve KL(m, kappa) = kl_target for kappa by bisection in log-kappa.

    KL is strictly increasing in kappa (d KL / d kappa = kappa * A_m'(kappa) >= 0),
    so the inverse is unique for kl_target > 0.
    """
    target_t = _as_tensor(kl_target)
    m_t = _as_tensor(m, like=target_t)
    target_t, m_t = torch.broadcast_tensors(target_t, m_t)
    if torch.any(target_t < 0.0):
        raise ValueError("kl_target must be >= 0")
    log_lo = torch.full_like(target_t, -30.0)
    log_hi = torch.full_like(target_t, 1.0)
    # Expand the upper bracket until KL exceeds the target.
    for _ in range(200):
        kl_hi = kl_vmf_uniform(torch.exp(log_hi), m_t)
        # Non-finite KL (overflow to inf in kappa*A) counts as "above target".
        need = torch.isfinite(kl_hi) & (kl_hi < target_t)
        if not bool(torch.any(need)):
            break
        log_hi = torch.where(need, log_hi * 2.0 + 1.0, log_hi)
    for _ in range(max_iter):
        log_mid = 0.5 * (log_lo + log_hi)
        kl_mid = kl_vmf_uniform(torch.exp(log_mid), m_t)
        go_right = torch.isfinite(kl_mid) & (kl_mid < target_t)
        log_lo = torch.where(go_right, log_mid, log_lo)
        log_hi = torch.where(go_right, log_hi, log_mid)
    kappa = torch.exp(0.5 * (log_lo + log_hi))
    return torch.where(target_t > 0.0, kappa, torch.zeros_like(target_t))
