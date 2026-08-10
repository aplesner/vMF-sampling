"""Differentiable, per-row batched vMF(μ, κ) sampling for the S-VAE.

Implements Ulrich's (1984) sampler with the Naesseth et al. (2017)
reparameterization through acceptance-rejection, exactly as Davidson et al.
(2018) App. D, but vectorized over rows: each batch element has its own
(μ_i, κ_i).  Two differences from Davidson's reference code:

  * the e1 -> μ map uses z = ω μ + sqrt(1-ω²) w_⊥ with w_⊥ uniform on the
    subsphere orthogonal to μ (a normalized Gaussian projection), which is
    exact and differentiable in μ, instead of an explicit Householder;
  * the KL term comes from projects/hyperspherical-vae/src/vmf_kl.py (CUSF /
    torch log-Bessel), finite and differentiable at m >= 1024 where the
    scipy-based reference is numerically invalid.

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
from vmf_kl import kl_vmf_uniform, log_vmf_log_norm  # noqa: E402


def _work_dtype(kappa: torch.Tensor) -> torch.dtype:
    """float64 everywhere except MPS (which lacks float64); MPS is dev-only."""
    return torch.float32 if kappa.device.type == "mps" else torch.float64


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
    max_tries: int = 200,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched acceptance-rejection sampling of ω ~ g(ω|κ, m).

    Returns (omega, correction) where ``correction = log g(ω|κ) +
    log|∂h/∂ε|`` is differentiable w.r.t. κ on accepted rows.  ω itself is
    differentiable in κ pathwise.  Acceptance decisions are detached.
    """
    if m < 2:
        raise ValueError("m must be >= 2")
    shape = kappa.shape
    wd = _work_dtype(kappa)
    kappa64 = kappa.detach().to(wd).reshape(-1)
    b, a, d = _ulrich_params(kappa64, m)
    nu = (m - 1) / 2.0
    beta = torch.distributions.Beta(nu, nu)

    omega = torch.empty_like(kappa64)
    keep_eps = torch.zeros_like(kappa64)
    accepted = torch.zeros_like(kappa64, dtype=torch.bool)
    tries = 0
    while not bool(accepted.all()):
        if tries >= max_tries:
            raise RuntimeError(
                f"vMF rejection sampling did not finish after {max_tries} rounds "
                f"({int((~accepted).sum())}/{accepted.numel()} rows pending; "
                f"max kappa {float(kappa64.max()):.3g}, m={m})"
            )
        tries += 1
        n = int((~accepted).sum())
        eps = beta.sample((n,)).to(kappa64)
        u = torch.rand(n, dtype=kappa64.dtype, device=kappa64.device)
        b_p, a_p, d_p = b[~accepted], a[~accepted], d[~accepted]
        om, _ = _h_transform(eps, b_p)
        t = 2.0 * a_p * b_p / (1.0 - (1.0 - b_p) * eps)
        ok = (m - 1) * torch.log(t) - t + d_p >= torch.log(u)
        idx = torch.nonzero(~accepted).squeeze(-1)[ok]
        omega[idx] = om[ok]
        keep_eps[idx] = eps[ok]
        accepted[idx] = True

    # Recompute ω and the correction differentiably at the accepted ε.
    kappa_g = kappa.to(wd).reshape(-1)
    b_g, _, _ = _ulrich_params(kappa_g, m)
    omega_g, log_abs_dh = _h_transform(keep_eps, b_g)
    # log g(ω|κ) including the κ-dependent normalizer log C_m(κ) (Davidson
    # App. D.2: ∇κ log g contributes -A_m(κ)); κ-independent constants drop out.
    omega_sq = (omega_g * omega_g).clamp(max=1.0 - 1e-12)
    log_g = (
        _log_norm_for_correction(kappa_g, m)
        + kappa_g * omega_g
        + (m - 3) / 2.0 * torch.log1p(-omega_sq)
    )
    correction = log_g + log_abs_dh
    return (
        omega_g.reshape(shape).to(kappa.dtype),
        correction.reshape(shape).to(kappa.dtype),
    )


def sample_vmf(
    mu: torch.Tensor,
    kappa: torch.Tensor,
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
    omega, correction = sample_omega(kappa, m)

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


def _log_norm_for_correction(kappa: torch.Tensor, m: int) -> torch.Tensor:
    """log C_m(κ) differentiable in κ, float64 even on MPS (via CPU round-trip)."""
    if kappa.device.type == "mps":
        return log_vmf_log_norm(m, kappa.to("cpu", torch.float64)).to(
            kappa.device, torch.float32
        )
    return log_vmf_log_norm(m, kappa.to(torch.float64))


def kl_to_uniform(kappa: torch.Tensor, m: int) -> torch.Tensor:
    """KL(vMF(κ)‖U(S^{m-1})), differentiable in κ, computed in float64.

    On MPS the computation round-trips through CPU float64 (κ is one scalar
    per row, so this is cheap); gradients flow through the casts.
    """
    if kappa.device.type == "mps":
        return kl_vmf_uniform(kappa.to("cpu", torch.float64), m).to(
            kappa.device, kappa.dtype
        )
    return kl_vmf_uniform(kappa, m).to(kappa.dtype)


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
