"""LARS optimizer (layer-wise adaptive rate scaling) — Tier-1
scale-invariant-optimizer reference arm. Standard formulation:
trust = eta * ||w|| / ||grad + wd*w||, applied to matrix params only.
"""

import torch


class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.9, weight_decay=0.0, eta=0.001):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mom, wd, eta = (group["lr"], group["momentum"],
                                group["weight_decay"], group["eta"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if wd != 0:
                    d_p = d_p.add(p, alpha=wd)
                if p.ndim > 1:  # weight matrices get layer-wise scaling
                    w_norm = p.norm()
                    g_norm = d_p.norm()
                    if w_norm > 0 and g_norm > 0:
                        d_p = d_p.mul(eta * w_norm / g_norm)
                buf = self.state[p].get("momentum_buffer")
                if buf is None:
                    buf = torch.clone(d_p).detach()
                    self.state[p]["momentum_buffer"] = buf
                else:
                    buf.mul_(mom).add_(d_p)
                p.add_(buf, alpha=-lr)
        return loss
