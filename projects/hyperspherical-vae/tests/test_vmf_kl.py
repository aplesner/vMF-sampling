"""Unit tests for projects/hyperspherical-vae/src/vmf_kl.py."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vmf_kl import (  # noqa: E402
    bessel_ratio_A,
    bessel_ratio_A_prime,
    invert_A,
    invert_kl,
    kl_grad_kappa,
    kl_vmf_uniform,
    log_bessel_iv,
    log_surface_area,
)


class TestAgainstScipy(unittest.TestCase):
    """Cross-checks where SciPy is still valid (m <= 512, moderate kappa)."""

    def test_log_bessel_iv(self):
        from scipy import special
        for m in [3, 6, 11, 41, 128]:
            for kappa in [1e-3, 0.5, 5.0, 50.0, 300.0]:
                order = m / 2.0 - 1.0
                ref = math.log(special.ive(order, kappa)) + kappa
                got = log_bessel_iv(order, kappa).item()
                self.assertAlmostEqual(ref, got, places=9, msg=f"m={m} k={kappa}")

    def test_bessel_ratio(self):
        from scipy import special
        for m in [3, 6, 41, 128]:
            for kappa in [0.5, 5.0, 50.0, 300.0]:
                ref = special.iv(m / 2.0, kappa) / special.iv(m / 2.0 - 1.0, kappa)
                got = bessel_ratio_A(kappa, m).item()
                self.assertAlmostEqual(ref, got, places=10, msg=f"m={m} k={kappa}")

    def test_kl_small_kappa_limit(self):
        # KL -> 0 as kappa -> 0 (uniform posterior matches uniform prior).
        for m in [3, 21, 128, 1024]:
            self.assertAlmostEqual(kl_vmf_uniform(0.0, m).item(), 0.0, places=12)
            self.assertLess(kl_vmf_uniform(1e-6, m).item(), 1e-8)


class TestIdentities(unittest.TestCase):
    def test_riccati_prime_matches_finite_difference(self):
        for m in [3, 21, 128]:
            for kappa in [0.5, 5.0, 50.0]:
                eps = 1e-5
                fd = (bessel_ratio_A(kappa + eps, m) - bessel_ratio_A(kappa - eps, m)).item() / (2 * eps)
                got = bessel_ratio_A_prime(kappa, m).item()
                self.assertAlmostEqual(fd, got, places=4, msg=f"m={m} k={kappa}")

    def test_kl_grad_is_kappa_times_A_prime(self):
        # Davidson et al. eq. 6 simplifies to d KL/d kappa = kappa * A_m'(kappa) >= 0.
        for m in [3, 21, 128]:
            for kappa in [0.5, 5.0, 50.0]:
                eps = 1e-5
                fd = (kl_vmf_uniform(kappa + eps, m) - kl_vmf_uniform(kappa - eps, m)).item() / (2 * eps)
                got = kl_grad_kappa(kappa, m).item()
                self.assertAlmostEqual(fd, got, places=4, msg=f"m={m} k={kappa}")
                self.assertGreaterEqual(got, 0.0)

    def test_A_monotone_in_kappa(self):
        m = 64
        ks = torch.linspace(0.0, 500.0, 2000, dtype=torch.float64)
        a = bessel_ratio_A(ks, m)
        self.assertTrue(bool(torch.all(torch.diff(a) >= 0)))

    def test_kl_monotone_in_rho(self):
        # At fixed m, rho -> KL is increasing; this makes (rho, KL/m) a valid
        # scale-free reparameterization of the rate.
        m = 128
        rhos = torch.linspace(0.01, 0.99, 200, dtype=torch.float64)
        kls = torch.stack([kl_vmf_uniform(invert_A(r, m), m) for r in rhos])
        self.assertTrue(bool(torch.all(torch.diff(kls) > 0)))


class TestInversion(unittest.TestCase):
    def test_invert_A_roundtrip(self):
        for m in [3, 41, 128, 1024, 4096]:
            for rho in [0.05, 0.3, 0.9, 0.999]:
                kappa = invert_A(rho, m)
                got = bessel_ratio_A(kappa, m).item()
                self.assertAlmostEqual(rho, got, places=8, msg=f"m={m} rho={rho}")

    def test_invert_kl_roundtrip(self):
        for m in [3, 41, 129, 1025]:
            for kl in [0.5, 7.28, 33.5, 200.0]:
                kappa = invert_kl(kl, m)
                got = kl_vmf_uniform(kappa, m).item()
                self.assertAlmostEqual(kl, got, places=6, msg=f"m={m} kl={kl}")

    def test_zero(self):
        self.assertEqual(invert_A(0.0, 128).item(), 0.0)
        self.assertEqual(invert_kl(0.0, 128).item(), 0.0)


class TestGateG1Geometry(unittest.TestCase):
    """Properties the restated G1 relies on (see PREREGISTRATION.md)."""

    def test_fixed_rho_kl_linear_in_m(self):
        # KL/m at fixed rho is (near-)constant across m, tightening as m grows.
        rho = 0.3
        per_m = {m: kl_vmf_uniform(invert_A(rho, m), m).item() / m
                 for m in [128, 512, 1024, 4096]}
        values = list(per_m.values())
        self.assertAlmostEqual(values[0], 0.04715, delta=2e-3)
        self.assertLess((max(values) - min(values)) / min(values), 1e-3)

    def test_rising_branch_boundary_rho_star(self):
        # KL(m) at fixed kappa peaks where rho = A_m(kappa) ~ 0.69.
        for kappa in [20.0, 100.0, 400.0]:
            ms = torch.linspace(2.0, 2.0 * kappa, 3000, dtype=torch.float64)
            kl = kl_vmf_uniform(kappa, ms)
            m_star = ms[int(torch.argmax(kl).item())].item()
            self.assertAlmostEqual(m_star / kappa, 0.762, delta=0.02)
            self.assertAlmostEqual(bessel_ratio_A(kappa, m_star).item(), 0.69, delta=0.02)

    def test_high_m_finite(self):
        # The whole point of the restated gate: computable where SciPy fails.
        for m in [1024, 4096, 8192]:
            kappa = invert_A(0.3, m).item()
            self.assertTrue(math.isfinite(kl_vmf_uniform(kappa, m).item()))

    def test_surface_area_matches_known_values(self):
        self.assertAlmostEqual(log_surface_area(2).item(), math.log(2 * math.pi), places=12)
        self.assertAlmostEqual(log_surface_area(3).item(), math.log(4 * math.pi), places=12)


if __name__ == "__main__":
    unittest.main()
