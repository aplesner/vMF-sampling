"""Tests for the NumPy log-Bessel path, including the p >= 8192 regime."""

from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401  (sys.path setup)
import numpy as np
from vmf_measure import bessel_ratio_ap, bessel_ratio_ap_deriv, log_iv


class TestLogIvReference(unittest.TestCase):
    """50-digit mpmath reference at orders beyond SciPy's underflow limit."""

    def test_high_order_matches_mpmath(self):
        from mpmath import besseli, log, mp, mpf

        mp.dps = 50
        # p = 2*order + 2, so order 4095 <-> p = 8192.
        for nu, x in [
            (511.0, 512.0),
            (2047.0, 2048.0),
            (4095.0, 100.0),
            (4095.0, 4096.0),
            (4095.0, 10000.0),
            (8191.0, 8500.0),
            (16383.0, 20000.0),
        ]:
            with self.subTest(nu=nu, x=x):
                reference = float(log(besseli(mpf(nu), mpf(x))))
                got = float(log_iv(nu, x))
                self.assertAlmostEqual(got / reference, 1.0, places=12)

    def test_matches_torch_port(self):
        try:
            import torch

            from src.torch_log_iv import torch_log_iv
        except ImportError:
            self.skipTest("torch not available")
        orders, xs = np.meshgrid(
            [0.0, 0.5, 3.0, 40.0, 511.0, 2047.0, 4095.0, 8191.0],
            [1e-3, 0.5, 10.0, 100.0, 2000.0, 4096.0, 10000.0, 30000.0],
        )
        got = log_iv(orders, xs)
        reference = torch_log_iv(
            torch.tensor(orders), torch.tensor(xs)
        ).numpy()
        finite = np.isfinite(got)
        rel = np.abs(got[finite] - reference[finite]) / np.abs(reference[finite])
        self.assertLess(rel.max(), 1e-8)

    def test_edge_cases(self):
        self.assertEqual(float(log_iv(0.0, 0.0)), 0.0)
        self.assertEqual(float(log_iv(2.0, 0.0)), -np.inf)
        self.assertTrue(np.isnan(log_iv(-1.0, 1.0)))
        self.assertTrue(np.isnan(log_iv(1.0, -1.0)))


class TestBesselRatio(unittest.TestCase):
    def test_limits_and_monotonicity(self):
        p = 512
        self.assertEqual(float(bessel_ratio_ap(0.0, p)), 0.0)
        kappas = np.geomspace(1e-2, 1e4, 200)
        ratios = bessel_ratio_ap(kappas, p)
        self.assertTrue(np.all(np.diff(ratios) > 0.0))
        self.assertTrue(np.all(ratios < 1.0))
        # Small-kappa expansion: A_p(kappa) ~ kappa / p.
        self.assertAlmostEqual(float(bessel_ratio_ap(1e-3, p)) / (1e-3 / p), 1.0, places=4)

    def test_derivative_limit_at_zero(self):
        for p in (2, 512, 8192):
            self.assertAlmostEqual(float(bessel_ratio_ap_deriv(0.0, p)), 1.0 / p, places=12)

    def test_derivative_positive(self):
        kappas = np.geomspace(1e-2, 1e4, 50)
        for p in (8, 512, 8192):
            self.assertTrue(np.all(bessel_ratio_ap_deriv(kappas, p) > 0.0))

    def test_fisher_info_consistency(self):
        # A_p' must equal the numerical derivative of A_p.
        for p, kappa in [(512, 25.6), (512, 1565.7), (8192, 409.6)]:
            h = 1e-5 * kappa
            numeric = (
                float(bessel_ratio_ap(kappa + h, p)) - float(bessel_ratio_ap(kappa - h, p))
            ) / (2 * h)
            self.assertAlmostEqual(
                float(bessel_ratio_ap_deriv(kappa, p)) / numeric, 1.0, places=5
            )


if __name__ == "__main__":
    unittest.main()
