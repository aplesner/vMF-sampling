"""Device-native batched vMF concentration estimators for PyTorch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

TorchLogIv = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class TorchNewtonResult:
    kappa: torch.Tensor
    iterations: torch.Tensor
    converged: torch.Tensor


def _inputs(r_bar: torch.Tensor, dimension: torch.Tensor | int) -> tuple[torch.Tensor, torch.Tensor]:
    if not r_bar.is_floating_point():
        raise TypeError("r_bar must be floating point")
    p = torch.as_tensor(dimension, dtype=r_bar.dtype, device=r_bar.device)
    return torch.broadcast_tensors(r_bar, p)


def banerjee_kappa_torch(r_bar: torch.Tensor, dimension: torch.Tensor | int) -> torch.Tensor:
    """Device-native Banerjee et al. closed-form approximation."""

    r_bar, p = _inputs(r_bar, dimension)
    return r_bar * (p - r_bar.square()) / (1.0 - r_bar.square())


def bessel_ratio_from_logs_torch(
    kappa: torch.Tensor,
    dimension: torch.Tensor | int,
    log_iv: TorchLogIv,
) -> torch.Tensor:
    p = torch.as_tensor(dimension, dtype=kappa.dtype, device=kappa.device)
    kappa, p = torch.broadcast_tensors(kappa, p)
    order = p / 2.0 - 1.0
    ratio = torch.exp(log_iv(order + 1.0, kappa) - log_iv(order, kappa))
    return torch.where(kappa == 0.0, torch.zeros_like(ratio), ratio)


def sra_kappa_torch(
    r_bar: torch.Tensor,
    dimension: torch.Tensor | int,
    log_iv: TorchLogIv,
) -> torch.Tensor:
    """Sra's Banerjee initialization followed by exactly two Newton steps."""

    r_bar, p = _inputs(r_bar, dimension)
    kappa = banerjee_kappa_torch(r_bar, p)
    active = r_bar > 0.0
    kappa = torch.where(active, kappa, torch.ones_like(kappa))
    for _ in range(2):
        ratio = bessel_ratio_from_logs_torch(kappa, p, log_iv)
        derivative = 1.0 - ratio.square() - ((p - 1.0) / kappa) * ratio
        kappa = torch.where(active, kappa - (ratio - r_bar) / derivative, kappa)
    return torch.where(active, kappa, torch.zeros_like(kappa))


def converged_newton_kappa_torch(
    r_bar: torch.Tensor,
    dimension: torch.Tensor | int,
    log_iv: TorchLogIv,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    max_iterations: int = 50,
) -> TorchNewtonResult:
    """Batched safeguarded Newton solution of the likelihood equation."""

    r_bar, p = _inputs(r_bar, dimension)
    if rtol is None:
        rtol = 1e-5 if r_bar.dtype == torch.float32 else 1e-10
    if atol is None:
        atol = 1e-6 if r_bar.dtype == torch.float32 else 1e-12

    denominator = 1.0 - r_bar.square()
    lower = torch.clamp_min(r_bar * (p - 2.0) / denominator, 0.0)
    upper = r_bar * p / denominator
    kappa = banerjee_kappa_torch(r_bar, p)
    active = r_bar > 0.0
    iterations = torch.zeros_like(kappa, dtype=torch.int64)
    converged = ~active

    for _ in range(max_iterations):
        pending = active & ~converged
        if not bool(torch.any(pending).item()):
            break
        safe_kappa = torch.where(pending, kappa, torch.ones_like(kappa))
        ratio = bessel_ratio_from_logs_torch(safe_kappa, p, log_iv)
        score = ratio - r_bar
        just_converged = pending & torch.isfinite(score) & (
            torch.abs(score) <= atol + rtol * torch.maximum(r_bar, torch.ones_like(r_bar))
        )
        converged = converged | just_converged
        pending = pending & ~just_converged
        if not bool(torch.any(pending).item()):
            break

        lower = torch.where(pending & (score < 0.0), kappa, lower)
        upper = torch.where(pending & (score >= 0.0), kappa, upper)
        derivative = 1.0 - ratio.square() - ((p - 1.0) / safe_kappa) * ratio
        proposal = kappa - score / derivative
        use_bisection = (
            ~torch.isfinite(proposal)
            | ~torch.isfinite(derivative)
            | (derivative <= 0.0)
            | (proposal <= lower)
            | (proposal >= upper)
        )
        proposal = torch.where(use_bisection, 0.5 * (lower + upper), proposal)
        kappa = torch.where(pending, proposal, kappa)
        iterations = iterations + pending.to(torch.int64)

    return TorchNewtonResult(
        torch.where(active, kappa, torch.zeros_like(kappa)), iterations, converged
    )


def direct_log_likelihood_kappa_torch(
    r_bar: torch.Tensor,
    dimension: torch.Tensor | int,
    log_iv: TorchLogIv,
    *,
    iterations: int = 64,
) -> torch.Tensor:
    """Direct batched likelihood maximization by golden-section search.

    This independent objective-based method is useful for validation.  It is
    expected to require many more log-Bessel evaluations than Newton's method.
    """

    r_bar, p = _inputs(r_bar, dimension)
    active = r_bar > 0.0
    lower = torch.full_like(r_bar, torch.finfo(r_bar.dtype).tiny)
    upper = torch.where(active, r_bar * p / (1.0 - r_bar.square()), torch.ones_like(r_bar))
    order = p / 2.0 - 1.0
    inverse_phi = (5.0**0.5 - 1.0) / 2.0

    def objective(kappa: torch.Tensor) -> torch.Tensor:
        return order * torch.log(kappa) - log_iv(order, kappa) + kappa * r_bar

    for _ in range(iterations):
        left = upper - inverse_phi * (upper - lower)
        right = lower + inverse_phi * (upper - lower)
        choose_left = objective(left) >= objective(right)
        upper = torch.where(choose_left, right, upper)
        lower = torch.where(choose_left, lower, left)
    estimate = 0.5 * (lower + upper)
    return torch.where(active, estimate, torch.zeros_like(estimate))
