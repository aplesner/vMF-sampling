"""Statistical tests for the differentiable batched vMF sampler."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vmf_kl import bessel_ratio_A, bessel_ratio_A_prime  # noqa: E402
from vmf_vae import kl_to_uniform, sample_omega, sample_vmf  # noqa: E402

torch.manual_seed(0)


def _batched(mu_row: torch.Tensor, kappa: float, n: int) -> tuple[torch.Tensor, ...]:
    m = mu_row.numel()
    mu = mu_row.expand(n, m).clone().requires_grad_(False)
    k = torch.full((n,), kappa, dtype=torch.float64)
    return mu, k


class TestDistribution(unittest.TestCase):
    """E[z] = A_m(kappa) mu; concentration of the resultant length."""

    def _check(self, m: int, kappa: float, n: int = 100_000):
        e1 = torch.zeros(m, dtype=torch.float64)
        e1[0] = 1.0
        mu, k = _batched(e1, kappa, n)
        z, _ = sample_vmf(mu, k)
        mean = z.detach().mean(dim=0)
        a_exact = bessel_ratio_A(kappa, m).item() if kappa > 0 else 0.0
        # MC standard error of the mean resultant length is O(1/sqrt(n));
        # allow 6 sigma plus slack for the orthogonal components.
        self.assertAlmostEqual(mean[0].item(), a_exact, delta=max(6e-3, 0.02 * max(a_exact, 0.1)))
        off_axis = mean[1:].norm().item()
        self.assertLess(off_axis, 6.0 / n**0.5 + 1e-3)

    def test_m3(self):
        for kappa in [0.0, 1.0, 10.0, 100.0]:
            with self.subTest(kappa=kappa):
                self._check(3, kappa)

    def test_m64(self):
        for kappa in [0.0, 5.0, 32.0]:
            with self.subTest(kappa=kappa):
                self._check(64, kappa)

    def test_m512_high_kappa(self):
        self._check(512, 256.0, n=50_000)

    def test_batched_heterogeneous_rows(self):
        # Each row has its own kappa; resultant lengths must match per-row.
        m, n_per = 16, 20_000
        kappas = torch.tensor([0.0, 0.5, 2.0, 8.0, 64.0], dtype=torch.float64)
        e1 = torch.zeros(m, dtype=torch.float64)
        e1[0] = 1.0
        for kappa in kappas:
            mu, k = _batched(e1, float(kappa), n_per)
            z, _ = sample_vmf(mu, k)
            r_bar = z.detach().mean(0)[0].item()
            a = bessel_ratio_A(float(kappa), m).item() if kappa > 0 else 0.0
            self.assertAlmostEqual(r_bar, a, delta=0.02)


class TestGradients(unittest.TestCase):
    """Pathwise + correction estimator vs exact derivatives of E[f(z)]."""

    def _grad_estimate(self, m: int, kappa: float, n: int = 200_000):
        """f(z) = c . z with fixed c; returns dE[f]/dkappa estimate.

        Uses a *shared scalar* kappa expanded across rows, so that the gradient
        of the batch mean equals the Monte Carlo mean of per-sample gradients
        (the estimator whose expectation is dE[f]/dkappa).
        """
        e1 = torch.zeros(m, dtype=torch.float64)
        e1[0] = 1.0
        c = torch.zeros(m, dtype=torch.float64)
        c[0], c[1 % m] = 0.7, 0.7  # not aligned with mu
        c = c / c.norm()
        mu = e1.expand(n, m).clone()
        k = torch.tensor(kappa, dtype=torch.float64, requires_grad=True)
        z, correction = sample_vmf(mu, k.expand(n))
        f = (z * c).sum(-1)
        objective = f + f.detach() * correction
        grad = torch.autograd.grad(objective.mean(), k)[0].item()
        return grad

    def test_kappa_gradient_m3(self):
        # Exact: dE[f]/dkappa = A_3'(kappa) * (c . mu); A' via Riccati.
        for kappa in [1.0, 5.0, 25.0]:
            est = self._grad_estimate(3, kappa)
            exact = bessel_ratio_A_prime(kappa, 3).item() * (0.7 / (0.7**2 + 0.7**2) ** 0.5)
            with self.subTest(kappa=kappa):
                self.assertAlmostEqual(est, exact, delta=max(3e-3, 0.03 * abs(exact)))

    def test_kappa_gradient_m64(self):
        kappa = 16.0
        est = self._grad_estimate(64, kappa, n=100_000)
        exact = bessel_ratio_A_prime(kappa, 64).item() * (0.7 / (0.7**2 + 0.7**2) ** 0.5)
        self.assertAlmostEqual(est, exact, delta=max(3e-3, 0.05 * abs(exact)))

    def test_mu_gradient(self):
        # dE[c.z]/dmu = A_m(kappa) c, with a shared mu leaf expanded over rows.
        m, kappa, n = 8, 4.0, 100_000
        c = torch.randn(m, dtype=torch.float64)
        c = c / c.norm()
        mu = torch.zeros(m, dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            mu[0] = 1.0
        k = torch.full((n,), kappa, dtype=torch.float64)
        z, _ = sample_vmf(mu.expand(n, m), k)
        f = (z * c).sum(-1)
        grad = torch.autograd.grad(f.mean(), mu)[0]
        # E[f] depends on mu only through its direction, so the exact gradient
        # is the tangent projection A_m(kappa) (I - mu mu^T) c.
        e1 = torch.zeros(m, dtype=torch.float64)
        e1[0] = 1.0
        exact = bessel_ratio_A(kappa, m) * (c - (c * e1).sum() * e1)
        self.assertLess((grad - exact).norm().item(), 0.02)

    def test_kl_gradient_positive_and_matches(self):
        k = torch.tensor([0.5, 5.0, 50.0], dtype=torch.float64, requires_grad=True)
        kl = kl_to_uniform(k, 41)
        grad = torch.autograd.grad(kl.sum(), k)[0]
        expected = k.detach() * bessel_ratio_A_prime(k.detach(), 41)
        self.assertTrue(bool(torch.all(grad > 0)))
        self.assertLess(((grad - expected).abs() / expected).max().item(), 1e-6)

    def test_kappa_zero_uniform(self):
        m, n = 32, 50_000
        e1 = torch.zeros(m, dtype=torch.float64)
        e1[0] = 1.0
        mu, k = _batched(e1, 0.0, n)
        z, correction = sample_vmf(mu, k)
        self.assertTrue(bool(torch.isfinite(z).all()))
        self.assertTrue(bool(torch.isfinite(correction).all()))
        self.assertLess(z.detach().mean(0).norm().item(), 0.02)
        self.assertAlmostEqual(kl_to_uniform(k[0], m).item(), 0.0, places=10)


class TestMPSPortability(unittest.TestCase):
    def test_runs_on_mps_if_available(self):
        if not torch.backends.mps.is_available():
            self.skipTest("no MPS")
        m, n = 16, 4096
        mu = torch.randn(n, m, device="mps")
        mu = mu / mu.norm(dim=-1, keepdim=True)
        k = torch.full((n,), 5.0, device="mps")
        z, correction = sample_vmf(mu, k)
        self.assertTrue(bool(torch.isfinite(z).all()))
        norms = z.norm(dim=-1)
        self.assertLess((norms - 1).abs().max().item(), 1e-4)


if __name__ == "__main__":
    unittest.main()
