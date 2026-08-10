"""The headline estimand: learning-rate band width in decades.

Pre-registered definitions (see PREREGISTRATION.md):

  Common grid: 13 points, logspace(1e-5, 1e-1, 13), i.e. 1/3-decade spacing.
  Every arm runs every point, including LRs where LoRA diverges — a
  divergence is data, not a wasted run.

  Band:  W_tau(arm) = log10 max(eta in band) - log10 min(eta in band),
  band = {eta : P(eta) >= P*_arm - tau}, P*_arm = arm's own sweep peak of
  the seed-mean metric. Reported for tau = 1.0 accuracy point (absolute)
  and tau = 2% relative. Grid-censored: W is a multiple of 1/3 decade; a
  one-point band has W = 0 and means "width <= 1/3 decade".

  Divergence (pre-registered): a run is diverged if ANY of
    (a) a NaN/inf loss was encountered,
    (b) mean train loss over the last 10% of steps > loss at init,
    (c) max grad-norm > 1e4.
  Diverged runs count as P = -inf for band purposes. eta_div(arm) is the
  smallest grid LR at which the seed-median run diverges.

  Contrasts (pre-registered): normalization effect = W(DeLoRA) - W(LoRA);
  per-component delta = W(DeLoRA-PC) - W(DeLoRA). DeLoRA-PC - LoRA is NOT
  primary (it confounds published normalization with untested per-component
  scaling). The r=1 null (DeLoRA-PC == DeLoRA exactly) sets the noise floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

GRID = tuple(np.logspace(-5, -1, 13).tolist())  # 1/3-decade spacing, pre-registered

GRAD_NORM_DIV_THRESHOLD = 1e4  # pre-registered divergence criterion (c)


@dataclass
class RunRecord:
    arm: str
    rank: int
    lr: float
    seed: int
    metric: float                  # final val metric (accuracy points); NaN if unusable
    nan_encountered: bool = False
    loss_init: float = math.nan
    loss_final_window: float = math.nan  # mean train loss, last 10% of steps
    grad_norm_max: float = 0.0
    eff_step_run_mean: float = math.nan  # mean effective step over the run

    def diverged(self) -> bool:
        if self.nan_encountered or (self.metric != self.metric):  # NaN
            return True
        if not math.isnan(self.loss_init) and not math.isnan(self.loss_final_window):
            if self.loss_final_window > self.loss_init:
                return True
        return self.grad_norm_max > GRAD_NORM_DIV_THRESHOLD


@dataclass
class Band:
    arm: str
    rank: int
    peak: float = math.nan
    lo: float = math.nan
    hi: float = math.nan
    width_decades: float = math.nan
    n_in_band: int = 0
    eta_div: float = math.nan      # smallest LR with seed-median divergence
    per_lr: dict = field(default_factory=dict)


def seed_mean_by_lr(records: list[RunRecord]) -> dict[float, float]:
    """Seed-mean metric per LR; diverged runs contribute -inf."""
    by_lr: dict[float, list[float]] = {}
    for rec in records:
        val = -math.inf if rec.diverged() else rec.metric
        by_lr.setdefault(rec.lr, []).append(val)
    return {lr: sum(v) / len(v) for lr, v in by_lr.items()}


def eta_diverged(records: list[RunRecord]) -> float:
    """Smallest LR at which >= half of seeds diverge; NaN if never."""
    by_lr: dict[float, list[bool]] = {}
    for rec in records:
        by_lr.setdefault(rec.lr, []).append(rec.diverged())
    for lr in sorted(by_lr):
        flags = by_lr[lr]
        if sum(flags) >= (len(flags) + 1) // 2:
            return lr
    return math.nan


def band_width(
    records: list[RunRecord],
    tau_abs: float = 1.0,
    tau_rel: float = 0.02,
    arm: str = "",
    rank: int = 0,
) -> Band:
    """Band at both pre-registered tolerances; the returned Band uses tau_abs."""
    means = seed_mean_by_lr(records)
    finite = {lr: m for lr, m in means.items() if m > -math.inf}
    out = Band(arm=arm, rank=rank, per_lr=means, eta_div=eta_diverged(records))
    if not finite:
        return out
    peak = max(finite.values())
    out.peak = peak
    in_band = sorted(lr for lr, m in finite.items() if m >= peak - tau_abs)
    out.lo, out.hi = in_band[0], in_band[-1]
    out.n_in_band = len(in_band)
    out.width_decades = math.log10(in_band[-1]) - math.log10(in_band[0])
    return out


def band_width_rel(records: list[RunRecord], tau_rel: float = 0.02) -> float:
    """Width at tau = 2% relative to the arm's own peak."""
    means = seed_mean_by_lr(records)
    finite = {lr: m for lr, m in means.items() if m > -math.inf}
    if not finite:
        return math.nan
    peak = max(finite.values())
    thresh = peak * (1.0 - tau_rel)
    in_band = sorted(lr for lr, m in finite.items() if m >= thresh)
    return math.log10(in_band[-1]) - math.log10(in_band[0])


def reexpress_bands_effstep(records: list[RunRecord], tau_abs: float = 1.0) -> dict[str, float]:
    """Pre-registered primary robustness check (Falsifier 1).

    Re-expresses each arm's band on the effective-step axis: every run's
    x-coordinate becomes its logged mean effective step
    ||dW_{t+1} - dW_t||_F / ||W_0||_F instead of the nominal LR. If the
    arms' bands coincide on this axis, the nominal-LR widening is a
    reparameterization artifact and the claim is dead.

    Returns {arm: width_decades_on_effstep_axis}.
    """
    by_arm: dict[str, dict[float, list[float]]] = {}
    for rec in records:
        if math.isnan(rec.eff_step_run_mean) or rec.eff_step_run_mean <= 0:
            continue
        x = math.log10(rec.eff_step_run_mean)
        val = -math.inf if rec.diverged() else rec.metric
        by_arm.setdefault(rec.arm, {}).setdefault(round(x, 6), []).append(val)

    widths: dict[str, float] = {}
    for arm, by_x in by_arm.items():
        means = {x: sum(v) / len(v) for x, v in by_x.items()}
        finite = {x: m for x, m in means.items() if m > -math.inf}
        if not finite:
            widths[arm] = math.nan
            continue
        peak = max(finite.values())
        in_band = sorted(x for x, m in finite.items() if m >= peak - tau_abs)
        widths[arm] = in_band[-1] - in_band[0]
    return widths
