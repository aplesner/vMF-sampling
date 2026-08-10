"""Structural tests for all five arms."""

import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adapters import ARMS, AdapterLinear  # noqa: E402

DT = torch.float64


def make(arm, r, d_out=8, d_in=6, seed=0):
    torch.manual_seed(seed)
    base = torch.nn.Linear(d_in, d_out, dtype=DT)
    return AdapterLinear(base, arm=arm, r=r,
                         generator=torch.Generator().manual_seed(seed))


class TestArms(unittest.TestCase):
    def test_all_arms_zero_update_at_init(self):
        """Every arm must preserve the base model at init (dW = 0)."""
        for arm in ARMS:
            for r in (1, 4):
                m = make(arm, r)
                dw = m.delta_weight()
                self.assertAlmostEqual(dw.abs().max().item(), 0.0, places=12,
                                       msg=f"{arm} r={r} not zero at init")

    def test_forward_equals_materialized_delta(self):
        """The low-rank branch must equal base(x) + x @ dW.T exactly."""
        for arm in ARMS:
            m = make(arm, 4)
            with torch.no_grad():
                m.lam.fill_(0.3) if m.lam is not None else None
            x = torch.randn(5, 6, dtype=DT)
            ref = x @ m.base.weight.T + m.base.bias + x @ m.delta_weight().T
            self.assertTrue(torch.allclose(m(x), ref, atol=1e-12), arm)

    def test_gain_gradient_alive_at_init(self):
        """The gain arms exist to test reparameterization; lambda must
        receive gradient at init even though dW = 0."""
        for arm in ("lora-scalar", "lora-diag", "delora", "delora-pc"):
            m = make(arm, 2)
            m(torch.randn(3, 6, dtype=DT)).pow(2).sum().backward()
            self.assertIsNotNone(m.lam.grad, arm)
            self.assertGreater(m.lam.grad.abs().sum().item(), 0.0, arm)

    def test_shifted_softplus_range(self):
        """Thesis claims s >= 0; actual range is (-log 2, inf). Document it."""
        from adapters import _shifted_softplus
        lam = torch.tensor([-20.0, 0.0, 5.0], dtype=DT)
        s = _shifted_softplus(lam)
        self.assertAlmostEqual(s[1].item(), 0.0, places=15)      # s(0) = 0
        self.assertGreater(s[0].item(), -math.log(2))              # bounded below
        self.assertLess(s[0].item(), 0.0)                          # ...but negative
        self.assertGreater(s[2].item(), 0.0)

    def test_coherence_bounds_and_r1(self):
        m = make("delora-pc", 4)
        with torch.no_grad():
            m.lam.fill_(0.2)
        coh = m.mutual_coherence()
        for v in coh.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
        m1 = make("delora-pc", 1)
        self.assertEqual(m1.mutual_coherence(), {"coh_u": 0.0, "coh_v": 0.0})

    def test_delora_pc_recovers_lora_like_magnitude(self):
        """With per-component gains, delora-pc can represent an arbitrary
        rank-1 direction at arbitrary scale (sanity of normalization path)."""
        m = make("delora-pc", 1)
        x = torch.randn(4, 6, dtype=DT)
        with torch.no_grad():
            m.lam.fill_(2.0)
        dw = m.delta_weight()
        self.assertGreater(dw.norm().item(), 0.0)
        self.assertTrue(torch.allclose(m(x),
                                       x @ m.base.weight.T + m.base.bias + x @ dw.T,
                                       atol=1e-12))


if __name__ == "__main__":
    unittest.main()
