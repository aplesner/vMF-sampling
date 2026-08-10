"""The project's free exact null: DeLoRA-PC == DeLoRA at r = 1.

At rank 1, a one-component diag(s) IS a single scalar, so DeLoRA-PC and
DeLoRA are the identical parameterization. The measured band difference at
r=1 MUST be zero; its spread is the noise floor for the whole estimand.
This test asserts the identity at the implementation level: identical
forward, identical gradients, and identical multi-step optimizer
trajectories given identical raw parameters.
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adapters import AdapterLinear  # noqa: E402

DT = torch.float64  # tight tolerance: identity must hold to machine precision


def make_pair(seed: int = 0, d_out: int = 8, d_in: int = 6):
    gen = torch.Generator().manual_seed(seed)
    base_w = torch.randn(d_out, d_in, generator=gen, dtype=DT)
    base_b = torch.randn(d_out, generator=gen, dtype=DT)
    a = torch.randn(1, d_in, generator=gen, dtype=DT)
    b = torch.randn(d_out, 1, generator=gen, dtype=DT)

    pair = []
    for arm in ("delora", "delora-pc"):
        base = torch.nn.Linear(d_in, d_out, dtype=DT)
        with torch.no_grad():
            base.weight.copy_(base_w)
            base.bias.copy_(base_b)
        m = AdapterLinear(base, arm=arm, r=1)
        with torch.no_grad():
            m.A.copy_(a)
            m.B.copy_(b)
            m.lam.copy_(torch.zeros(1, dtype=DT))
        pair.append(m)
    return pair


class TestRankOneIdentity(unittest.TestCase):
    def test_forward_identical(self):
        de, pc = make_pair()
        x = torch.randn(4, 6, dtype=DT)
        self.assertTrue(torch.equal(de(x), pc(x)))
        self.assertTrue(torch.equal(de.delta_weight(), pc.delta_weight()))

    def test_gradients_identical(self):
        de, pc = make_pair(seed=1)
        x = torch.randn(4, 6, dtype=DT)
        grads = []
        for m in (de, pc):
            m(x).pow(2).sum().backward()
            grads.append((m.A.grad.clone(), m.B.grad.clone(), m.lam.grad.clone()))
        for g_de, g_pc in zip(*grads):
            self.assertTrue(torch.equal(g_de, g_pc))

    def test_trajectory_identical(self):
        """Ten AdamW steps on identical batches: parameters must stay equal."""
        de, pc = make_pair(seed=2)
        torch.manual_seed(7)
        batches = [torch.randn(4, 6, dtype=DT) for _ in range(10)]
        opts = [torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.0)
                for m in (de, pc)]
        for x in batches:
            for m, opt in zip((de, pc), opts):
                opt.zero_grad()
                m(x).pow(2).sum().backward()
                opt.step()
        for p_de, p_pc in zip(de.parameters(), pc.parameters()):
            self.assertTrue(
                torch.equal(p_de, p_pc),
                "delora and delora-pc diverged at r=1 — the exact null is broken")

    def test_parameter_sets_match(self):
        de, pc = make_pair(seed=3)
        names_de = sorted(n for n, _ in de.named_parameters())
        names_pc = sorted(n for n, _ in pc.named_parameters())
        self.assertEqual(names_de, names_pc)


if __name__ == "__main__":
    unittest.main()
