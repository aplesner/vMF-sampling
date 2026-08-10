"""Differentiable, per-row batched vMF(μ, κ) sampling for the S-VAE.

Implements Ulrich's (1984) sampler with the Naesseth et al. (2017)
reparameterization through acceptance-rejection, exactly as Davidson et al.
(2018) App. D, vectorized over rows: each batch element has its own
(μ_i, κ_i).  Two differences from Davidson's reference code:

  * the e1 -> μ map uses z = ω μ + sqrt(1-ω²) w_⊥ with w_⊥ uniform on the
    subsphere orthogonal to μ (a normalized Gaussian projection), which is
    exact and differentiable in μ, instead of an explicit Householder;
  * the KL term and the correction-term normalizer come from a single fused
    Bessel evaluation (CUSF/torch log-Bessel via vmf_kl.log_iv_fast), finite
    at m >= 1024 where the scipy-based reference is numerically invalid, with
    analytic κ gradients (d KL/dκ = κ A_m'(κ), d log C_m/dκ = -A_m(κ)).

Gradient of E_{q(z)}[f(z)] w.r.t. (μ, κ):
  pathwise  : ∇f(z) through z(ε, v; μ, κ)                    (autograd)
  correction: f(z) · ∇_κ [ log g(ω|κ) + log|∂h(ε,κ)/∂ε| ]   (score term)

Only κ receives a correction (g and h do not depend on μ; Davidson App. D.2).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vmf_kl import (  # noqa: E402
    bessel_ratio_A,
    kl_grad_kappa,
    kl_vmf_uniform,
    log_iv_fast,
    log_surface_area,
    log_vmf_log_norm,
)


def _work_dtype(kappa: torch.Tensor) -> torch.dtype:
    """float64 everywhere except MPS (which lacks float64); MPS is dev-only."""
    return torch.float32 if kappa.device.type == "mps" else torch.float64


def _device64(kappa: torch.Tensor) -> torch.Tensor:
    if kappa.device.type == "mps":
        return kappa.detach().cpu().double()
    return kappa.detach().to(torch.float64)


class _BesselEvalFn(torch.autograd.Function):
    """One fused evaluation of A_m(κ), log C_m(κ), KL(vMF‖U) with analytic
    κ gradients.  Two log-Bessel evaluations total (orders m/2-1, m/2),
    each restricted to the branch the κ batch actually selects.
    """

    @staticmethod
    def forward(ctx, kappa: torch.Tensor, m: int):
        k = _device64(kappa)
        nu = m / 2.0 - 1.0
        safe_k = torch.clamp_min(k, 1e-300)
        l0 = log_iv_fast(nu, safe_k)
        l1 = log_iv_fast(nu + 1.0, safe_k)
        a = torch.where(k > 0.0, torch.exp(l1 - l0), torch.zeros_like(k))
        log_s = log_surface_area(m).to(k.device)
        log_c = torch.where(
            k > 0.0,
            nu * torch.log(safe_k) - (m / 2.0) * math.log(2.0 * math.pi) - l0,
            -log_s,
        )
        kl_direct = k * a + log_c + log_s
        switch = 8.2e3 * float(m) ** 0.75
        kl_asymp = (
            0.5 * (m - 1.0) * (torch.log(safe_k) - math.log(2.0 * math.pi) - 1.0)
            + log_s
            + (m - 1.0) * (m - 3.0) / (4.0 * safe_k)
        )
        kl = torch.where(k > switch, kl_asymp, kl_direct)
        kl = torch.where(k > 0.0, kl, torch.zeros_like(k))
        # derivatives: dKL/dκ = κ A' ,  d logC/dκ = -A ,  A' = 1 - A² - (m-1)A/κ
        a_prime = torch.where(
            k > 1e-12,
            1.0 - a * a - ((m - 1.0) / torch.clamp_min(k, 1e-12)) * a,
            torch.full_like(k, 1.0 / m),
        )
        ctx.save_for_backward(k * a_prime, -a)
        out = torch.stack([kl, log_c, a], dim=0)
        return out.to(kappa.device, kappa.dtype)

    @staticmethod
    def backward(ctx, gout):
        d_kl, d_logc = ctx.saved_tensors
        g = gout.detach().to(d_kl.device, d_kl.dtype)
        grad = g[0] * d_kl + g[1] * d_logc  # A output is detached (no g[2] path)
        return grad.to(gout.device, gout.dtype), None


def bessel_eval(kappa: torch.Tensor, m: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(KL, log C_m, A_m) with analytic κ gradients; two log-Bessel evals."""
    out = _BesselEvalFn.apply(kappa, m)
    return out[0], out[1], out[2].detach()


def kl_to_uniform(kappa: torch.Tensor, m: int) -> torch.Tensor:
    """KL(vMF(κ)‖U(S^{m-1})), float64 numerics, analytic κ gradient."""
    return _BesselEvalFn.apply(kappa, m)[0]


def _log_norm_for_correction(kappa: torch.Tensor, m: int) -> torch.Tensor:
    return _BesselEvalFn.apply(kappa, m)[1]


def _ulrich_params(kappa: torch.Tensor, m: int):
    """Per-row (b, a, d) constants of Algorithm 2 in Davidson et al. (2018)."""
    m1 = float(m - 1)
    kappa = kappa.to(_work_dtype(kappa))
    root = torch.sqrt(4.0 * kappa * kappa + m1 * m1)
    b = (-2.0 * kappa + root) / m1
    a = (m1 + 2.0 * kappa + root) / 4.0
    d = 4.0 * a * b / (1.0 + b) - m1 * math.log(m1)
    return b, a, d


def _h_transform(eps: torch.Tensor, b: torch.Tensor):
    """ω = h(ε, κ) and |∂h/∂ε|, both differentiable in κ through b."""
    denom = 1.0 - (1.0 - b) * eps
    omega = (1.0 - (1.0 + b) * eps) / denom
    # |∂h/∂ε| = 2b / (1-(1-b)ε)^2   (Davidson App. D.2, eq. 36)
    log_abs_dh = torch.log(2.0 * b) - 2.0 * torch.log(denom)
    return omega, log_abs_dh


def sample_omega(
    kappa: torch.Tensor,
    m: int,
    max_tries: int = 100,
    log_c: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched acceptance-rejection sampling of ω ~ g(ω|κ, m).

    Returns (omega, correction) where ``correction = log g(ω|κ) +
    log|∂h/∂ε|`` (g normalized, incl. log C_m(κ); Davidson App. D.2) is
    differentiable w.r.t. κ on accepted rows.  ω is differentiable in κ
    pathwise.  Acceptance decisions are detached.  Sampling is GPU-native:
    Beta proposals are drawn on-device and accepted-set checks sync at most
    every 4 rounds.
    """
    if m < 2:
        raise ValueError("m must be >= 2")
    shape = kappa.shape
    wd = _work_dtype(kappa)
    kappa64 = kappa.detach().to(wd).reshape(-1)
    b, a, d = _ulrich_params(kappa64, m)
    nu = (m - 1) / 2.0
    beta = torch.distributions.Beta(
        torch.as_tensor(nu, dtype=wd, device=kappa64.device),
        torch.as_tensor(nu, dtype=wd, device=kappa64.device),
    )

    omega = torch.empty_like(kappa64)
    keep_eps = torch.zeros_like(kappa64)
    accepted = torch.zeros_like(kappa64, dtype=torch.bool)
    for tries in range(1, max_tries + 1):
        n_pending = int((~accepted).sum())  # the only host sync per round
        if n_pending == 0:
            break
        if tries == max_tries:
            raise RuntimeError(
                f"vMF rejection sampling did not finish after {max_tries} rounds "
                f"({n_pending}/{kappa64.numel()} rows pending; max kappa "
                f"{float(kappa64.max()):.3g}, m={m})"
            )
        pidx = torch.nonzero(~accepted).squeeze(-1)
        eps = beta.sample((n_pending,))
        u = torch.rand(n_pending, dtype=wd, device=kappa64.device)
        b_p, a_p, d_p = b[pidx], a[pidx], d[pidx]
        om, _ = _h_transform(eps, b_p)
        t = 2.0 * a_p * b_p / (1.0 - (1.0 - b_p) * eps)
        ok = (m - 1) * torch.log(t) - t + d_p >= torch.log(u)
        rows = pidx[ok]
        omega[rows] = om[ok]
        keep_eps[rows] = eps[ok]
        accepted[rows] = True

    # Recompute ω and the correction differentiably at the accepted ε.
    kappa_g = kappa.to(wd).reshape(-1)
    b_g, _, _ = _ulrich_params(kappa_g, m)
    omega_g, log_abs_dh = _h_transform(keep_eps, b_g)
    # log g(ω|κ) including the κ-dependent normalizer log C_m(κ) (Davidson
    # App. D.2: ∇κ log g contributes -A_m(κ)); κ-independent constants drop out.
    if log_c is None:
        log_c = _log_norm_for_correction(kappa, m)
    log_c = log_c.to(wd).reshape(-1)
    omega_sq = (omega_g * omega_g).clamp(max=1.0 - 1e-12)
    log_g = log_c + kappa_g * omega_g + (m - 3) / 2.0 * torch.log1p(-omega_sq)
    correction = log_g + log_abs_dh
    return (
        omega_g.reshape(shape).to(kappa.dtype),
        correction.reshape(shape).to(kappa.dtype),
    )


def sample_vmf(
    mu: torch.Tensor,
    kappa: torch.Tensor,
    log_c: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable samples z ~ vMF(μ_i, κ_i) row-wise.

    mu: (..., m) unit vectors (renormalized defensively).  kappa: (...) >= 0.
    Returns (z, correction): z is differentiable pathwise in (μ, κ);
    ``correction`` multiplies a *detached* per-row objective f(z) to produce
    the score-function part of the κ gradient.
    """
    m = mu.shape[-1]
    mu = mu / mu.norm(dim=-1, keepdim=True).clamp_min(1e-20)
    kappa = kappa.clamp_min(0.0)
    omega, correction = sample_omega(kappa, m, log_c=log_c)

    # w_⊥: Gaussian projected orthogonal to μ and normalized — uniform on the
    # subsphere S^{m-2}_⊥μ.  Kept differentiable: the reparameterization path
    # is z(μ, ε, v) with base measures (ε, v), so ∂w/∂μ is part of the
    # unbiased pathwise gradient.
    v = torch.randn_like(mu)
    v_perp = v - (v * mu).sum(dim=-1, keepdim=True) * mu
    w = v_perp / v_perp.norm(dim=-1, keepdim=True).clamp_min(1e-20)

    omega = omega.unsqueeze(-1)
    z = omega * mu + (1.0 - omega * omega).clamp_min(0.0).sqrt() * w
    return z, correction


def sample_vmf_with_kl(
    mu: torch.Tensor, kappa: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(z, correction, KL) with a single fused Bessel evaluation."""
    m = mu.shape[-1]
    kl, log_c, _ = bessel_eval(kappa, m)
    z, correction = sample_vmf(mu, kappa, log_c=log_c)
    return z, correction, kl


def svae_elbo_terms(
    log_px_given_z: torch.Tensor,
    kappa: torch.Tensor,
    m: int,
    correction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ELBO pieces for a batch.

    log_px_given_z: (n,) per-row log-likelihood (already differentiable).
    Returns (recon_term, kl_term, correction_term) such that the loss is
        loss = -(recon_term + correction_term) + kl_term
    where correction_term = detach(log_px_given_z) * correction carries the
    score-function gradient w.r.t. κ with zero forward value.
    """
    kl = kl_to_uniform(kappa, m)
    correction_term = log_px_given_z.detach() * correction
    return log_px_given_z, kl, correction_term
