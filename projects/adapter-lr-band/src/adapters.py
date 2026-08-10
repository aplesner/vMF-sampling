"""Five adapter arms for Project 2 (the learning-rate band).

All arms wrap a frozen ``nn.Linear`` and add a low-rank update branch.
Parameterizations follow ``papers/nLoRA.pdf`` (DoLoRA, the supervised thesis
being re-established) with the corrections and controls required by
``docs/projects/02-adapter-lr-band.md``:

  lora        : dW = (alpha/r) * B A                       (Hu et al. 2022)
  lora-scalar : dW = (alpha/r) * g * B A                   gain control arm
  lora-diag   : dW = (alpha/r) * B diag(s) A               gain control arm
  delora      : dW = (alpha/r) * g * Uhat Vhat             (Bini et al. 2025)
  delora-pc   : dW = (alpha/r) * Uhat diag(s) Vhat         (thesis "DoLoRA")

with  g or s_i = softplus(lambda) - log 2,  lambda initialized to 0
(so dW = 0 at init for every arm). Note the thesis's claim "s_i >= 0" is
wrong: the range of softplus(lambda) - log 2 is (-log 2, +inf). We keep the
thesis's asymmetric parameterization verbatim for replication fidelity; the
sign-symmetric variant is a separate control arm, not this one.

Free exact null: at r = 1, delora-pc and delora are the *identical*
parameterization (a one-component diag(s) IS a single scalar). The unit
test in tests/test_identity_r1.py asserts this.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ARMS = ("lora", "lora-scalar", "lora-diag", "delora", "delora-pc")

LOG2 = math.log(2.0)


def _shifted_softplus(lam: torch.Tensor) -> torch.Tensor:
    """s = softplus(lambda) - log 2. s = 0 at lambda = 0; range (-log 2, inf)."""
    return F.softplus(lam) - LOG2


class AdapterLinear(nn.Module):
    """Frozen base linear plus one of the five adapter arms.

    ``base`` is frozen in-place. The trainable branch computes
    ``y = base(x) + x @ dW.T`` without materializing ``dW`` in the forward
    pass; ``delta_weight()`` materializes it for logging and analysis.
    """

    def __init__(
        self,
        base: nn.Linear,
        arm: str,
        r: int,
        alpha: float | None = None,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")
        if r < 1:
            raise ValueError("rank must be >= 1")

        self.arm = arm
        self.r = r
        self.alpha = float(alpha) if alpha is not None else 2.0 * r  # thesis: alpha = 2r
        self.scaling = self.alpha / r

        for p in base.parameters():
            p.requires_grad_(False)
        self.base = base

        d_out, d_in = base.weight.shape
        dev = base.weight.device
        dt = base.weight.dtype

        # A in R^{r x d_in} (rows are v_i^T), B in R^{d_out x r} (columns are u_i).
        self.A = nn.Parameter(torch.empty(r, d_in, device=dev, dtype=dt))
        self.B = nn.Parameter(torch.empty(d_out, r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5), generator=generator)

        # Lambda: per-component for diag / delora-pc, single scalar for
        # lora-scalar / delora, absent for plain lora.
        if arm in ("lora-diag", "delora-pc"):
            self.lam = nn.Parameter(torch.zeros(r, device=dev, dtype=dt))
        elif arm in ("lora-scalar", "delora"):
            self.lam = nn.Parameter(torch.zeros(1, device=dev, dtype=dt))
        else:
            self.register_parameter("lam", None)

        if arm == "lora":
            # Standard LoRA init: B = 0 guarantees dW = 0.
            nn.init.zeros_(self.B)
        else:
            # All gain-gated arms use random factors with s = 0 at init.
            # (B = 0 AND s = 0 would be a dead parameterization: no gradient
            # reaches either B or lambda.)
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5), generator=generator)

    # -- normalized factors ------------------------------------------------
    def _bhat(self) -> torch.Tensor:
        """Unit-normalized columns of B (uhat_i)."""
        return self.B / self.B.norm(dim=0, keepdim=True).clamp_min(1e-12)

    def _ahat(self) -> torch.Tensor:
        """Unit-normalized rows of A (vhat_i^T)."""
        return self.A / self.A.norm(dim=1, keepdim=True).clamp_min(1e-12)

    # -- effective factors, so that dW = B_eff @ A_eff ----------------------
    def _factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        arm = self.arm
        if arm == "lora":
            return self.scaling * self.B, self.A
        if arm == "lora-scalar":
            return self.scaling * _shifted_softplus(self.lam) * self.B, self.A
        if arm == "lora-diag":
            return self.scaling * (self.B * _shifted_softplus(self.lam)), self.A
        if arm == "delora":
            return self.scaling * _shifted_softplus(self.lam) * self._bhat(), self._ahat()
        # delora-pc
        return self.scaling * (self._bhat() * _shifted_softplus(self.lam)), self._ahat()

    def delta_weight(self) -> torch.Tensor:
        """Materialized update dW in R^{d_out x d_in}."""
        b_eff, a_eff = self._factors()
        return b_eff @ a_eff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.base.weight, self.base.bias)
        b_eff, a_eff = self._factors()
        return out + F.linear(F.linear(x, a_eff), b_eff)

    # -- diagnostics --------------------------------------------------------
    @torch.no_grad()
    def mutual_coherence(self) -> dict[str, float]:
        """max_{i != j} |<uhat_i, uhat_j>| (and same for vhat).

        Free replacement for the thesis's Appendix-A experiment: near-ortho-
        gonality of learned factors is the empirical question that decides
        whether per-component scales buy anything. Returns NaN-free 0.0 at
        r = 1 (no pairs)."""
        if self.r == 1:
            return {"coh_u": 0.0, "coh_v": 0.0}
        if self.arm in ("delora", "delora-pc"):
            u = self._bhat()
            v = self._ahat()
        else:
            u = self.B / self.B.norm(dim=0, keepdim=True).clamp_min(1e-12)
            v = self.A / self.A.norm(dim=1, keepdim=True).clamp_min(1e-12)
        gu = (u.T @ u).abs()
        gv = (v @ v.T).abs()
        gu.fill_diagonal_(0.0)
        gv.fill_diagonal_(0.0)
        return {"coh_u": gu.max().item(), "coh_v": gv.max().item()}

    @torch.no_grad()
    def gain_stats(self) -> dict[str, float]:
        if self.lam is None:
            return {"gain_mean": 1.0, "gain_min": 1.0, "gain_max": 1.0}
        s = _shifted_softplus(self.lam)
        return {
            "gain_mean": s.mean().item(),
            "gain_min": s.min().item(),
            "gain_max": s.max().item(),
        }


def inject_adapters(
    module: nn.Module,
    arm: str,
    r: int,
    alpha: float | None = None,
    name_filter=None,
    seed: int = 0,
) -> list[AdapterLinear]:
    """Replace target ``nn.Linear`` submodules with ``AdapterLinear`` wrappers.

    ``name_filter(name)`` decides which linears to adapt; default adapts
    every ``nn.Linear`` whose name contains "qkv", "proj", "fc1", or "fc2"
    (timm ViT naming: attention qkv/proj + MLP fc1/fc2).
    Returns the list of created adapters (attribute ``module`` paths are
    patched in place).
    """
    if name_filter is None:
        def name_filter(name: str) -> bool:
            return any(k in name for k in ("qkv", "proj", "fc1", "fc2"))

    gen = torch.Generator(device="cpu").manual_seed(seed)
    adapters: list[AdapterLinear] = []
    # Collect first; mutating _modules while walking is unsafe.
    targets = [
        (parent, attr, name, sub)
        for name, sub in module.named_modules()
        if isinstance(sub, nn.Linear) and name_filter(name)
        for parent, attr in [(module.get_submodule(name.rsplit(".", 1)[0]) if "." in name else module,
                              name.rsplit(".", 1)[-1])]
    ]
    for parent, attr, name, sub in targets:
        adapter = AdapterLinear(sub, arm=arm, r=r, alpha=alpha, generator=gen)
        setattr(parent, attr, adapter)
        adapters.append(adapter)
    return adapters


def adapter_parameters(module: nn.Module) -> list[nn.Parameter]:
    """Trainable parameters: adapter A/B/lambda plus classifier heads."""
    params = []
    for name, p in module.named_parameters():
        if any(k in name for k in (".A", ".B", ".lam", "head")):
            p.requires_grad_(True)
            params.append(p)
        else:
            p.requires_grad_(False)
    return params
