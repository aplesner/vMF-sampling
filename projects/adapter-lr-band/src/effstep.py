"""Effective-step logging — the pre-registered primary robustness check.

The number-one threat to the band claim is reparameterization: gains that
init at zero let the model absorb a large nominal learning rate by keeping
the scale small. The control is to re-express every run's x-axis in
effective-step units,

    e_t = || dW_{t+1} - dW_t ||_F / || W_0 ||_F

logged per adapted layer per (logged) step, for free. If the LR bands of
the five arms coincide on this axis, the widening is an artifact and the
claim is dead (Falsifier 1 in the brief).

Also logs the mutual-coherence diagnostic (max |<uhat_i, uhat_j>|) — the
free replacement for the thesis's Appendix-A experiment, whose Theorem 1 is
false and must not be tested.
"""

from __future__ import annotations

import torch

from adapters import AdapterLinear


class AdapterLogger:
    """Logs effective-step and coherence metrics for a set of adapters.

    Call ``record(step)`` AFTER ``optimizer.step()``. The first call
    establishes the reference dW and emits an effective step of 0.
    """

    def __init__(self, adapters: list[AdapterLinear], log_every: int = 1):
        self.adapters = adapters
        self.log_every = log_every
        self.w0_norms = [a.base.weight.norm().item() for a in adapters]
        self._prev: list[torch.Tensor] | None = None
        self.records: list[dict] = []

    @torch.no_grad()
    def _snapshot(self) -> list[torch.Tensor]:
        return [a.delta_weight().detach().clone() for a in self.adapters]

    @torch.no_grad()
    def record(self, step: int) -> dict | None:
        if step % self.log_every != 0:
            return None
        snap = self._snapshot()
        if self._prev is None:
            effs = [0.0] * len(self.adapters)
        else:
            effs = [
                ((snap[i] - self._prev[i]).norm() / max(self.w0_norms[i], 1e-12)).item()
                for i in range(len(self.adapters))
            ]
        self._prev = snap

        coh_u = max(a.mutual_coherence()["coh_u"] for a in self.adapters)
        coh_v = max(a.mutual_coherence()["coh_v"] for a in self.adapters)
        gains = [a.gain_stats()["gain_mean"] for a in self.adapters]
        rec = {
            "step": step,
            "eff_step_mean": sum(effs) / len(effs),
            "eff_step_max": max(effs),
            "coh_u_max": coh_u,
            "coh_v_max": coh_v,
            "gain_mean": sum(gains) / len(gains),
        }
        self.records.append(rec)
        return rec

    def summary(self) -> dict:
        """Per-run summary used for the effective-step band re-expression."""
        if len(self.records) < 2:
            return {"eff_step_run_mean": 0.0, "coh_u_final": 0.0, "coh_v_final": 0.0}
        eff = [r["eff_step_mean"] for r in self.records[1:]]  # skip the 0 reference
        last = self.records[-1]
        return {
            "eff_step_run_mean": sum(eff) / len(eff),
            "coh_u_final": last["coh_u_max"],
            "coh_v_final": last["coh_v_max"],
        }
