"""Tests for the closed-form admissibility surface (deliverable D1)."""

from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401  (sys.path setup)
import numpy as np
from vmf_measure import (
    admissible,
    amp,
    bessel_ratio_ap,
    min_resolvable_n,
    rel_rmse_surface,
)


class TestSurface(unittest.TestCase):
    def test_calibration_cell(self):
        # Brief calibration: p=512, Rbar=0.85, n/p=0.5 must sit near 0.63%,
        # far below the 1% target the rectangular rule would deny it.
        value = float(rel_rmse_surface(0.85, 512, 256))
        self.assertGreater(value, 0.005)
        self.assertLess(value, 0.0075)

    def test_rectangular_rule_cell_is_admissible(self):
        # Same cell: the rectangular rule demands n/p >= 10 and Rbar > 0.19;
        # the surface admits n/p = 0.5.
        self.assertTrue(bool(admissible(0.85, 512, 256, target_rel_rmse=0.01)))

    def test_decreasing_in_n(self):
        ns = np.geomspace(10, 1e5, 30)
        values = rel_rmse_surface(0.1, 512, ns)
        self.assertTrue(np.all(np.diff(values) < 0.0))

    def test_undefined_at_zero_rbar(self):
        self.assertEqual(float(rel_rmse_surface(0.0, 512, 100)), np.inf)

    def test_explicit_kappa_path(self):
        # Population-Rbar usage: surface at (A_p(kappa), p, n, kappa) must
        # agree with the inverted path.
        p, kappa, n = 512, 51.2, 500
        r = float(bessel_ratio_ap(kappa, p))
        explicit = float(rel_rmse_surface(r, p, n, kappa=kappa))
        inverted = float(rel_rmse_surface(r, p, n))
        self.assertAlmostEqual(explicit / inverted, 1.0, places=6)

    def test_min_resolvable_n_inverts_surface(self):
        for r, p in [(0.05, 512), (0.3, 128), (0.85, 512)]:
            with self.subTest(r=r, p=p):
                n_min = float(min_resolvable_n(r, p, target_rel_rmse=0.01))
                self.assertLessEqual(float(rel_rmse_surface(r, p, n_min)), 0.01)
                self.assertGreater(float(rel_rmse_surface(r, p, n_min / 2)), 0.01)

    def test_min_resolvable_n_refusal_cell(self):
        # The n=70 refusal cell must require far more samples.
        n_min = float(min_resolvable_n(0.05, 512, target_rel_rmse=0.01))
        self.assertGreater(n_min, 1000.0)

    def test_amp_formula(self):
        r, p = 0.5, 100
        expected = 1 + 2 * 0.25 / 0.75 - 2 * 0.25 / (100 - 0.25)
        self.assertAlmostEqual(float(amp(r, p)), expected, places=12)


if __name__ == "__main__":
    unittest.main()
