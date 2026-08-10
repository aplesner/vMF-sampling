"""Tests for the estimator and the refusal mode."""

from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401  (sys.path setup)
import numpy as np
from vmf_measure import (
    OK_STATUS,
    REFUSAL_STATUS,
    banerjee_kappa,
    bessel_ratio_ap,
    kappa_hat,
    mean_resultant_length,
    mle_kappa,
)


class TestMleKappa(unittest.TestCase):
    def test_recovers_population_kappa(self):
        # With population Rbar = A_p(kappa), the solver must return kappa.
        for p, kappa in [(3, 10.0), (512, 25.6), (512, 1565.7), (4096, 204.8), (8192, 409.6)]:
            with self.subTest(p=p, kappa=kappa):
                r = float(bessel_ratio_ap(kappa, p))
                got = float(mle_kappa(r, p))
                self.assertAlmostEqual(got / kappa, 1.0, places=8)

    def test_zero_rbar(self):
        self.assertEqual(float(mle_kappa(0.0, 512)), 0.0)

    def test_agrees_with_repo_solver_where_scipy_finite(self):
        from src.vmf_kappa import converged_newton_kappa, scipy_log_iv

        r = np.array([0.01, 0.1, 0.5, 0.9])
        p = np.array([8, 32, 128, 512])
        reference = converged_newton_kappa(r, p, log_iv=scipy_log_iv).kappa
        got = mle_kappa(r, p)
        np.testing.assert_allclose(got, reference, rtol=1e-8)

    def test_banerjee_close_to_mle_at_high_concentration(self):
        r = 0.85
        got = float(mle_kappa(r, 512))
        approx = float(banerjee_kappa(r, 512))
        self.assertLess(abs(got - approx) / got, 0.05)

    def test_mean_resultant_length(self):
        rng = np.random.Generator(np.random.PCG64DXSM(0))
        samples = rng.standard_normal((50, 16))
        samples /= np.linalg.norm(samples, axis=-1, keepdims=True)
        r = mean_resultant_length(samples)
        self.assertGreaterEqual(r, 0.0)
        self.assertLess(r, 1.0)


class TestRefusalMode(unittest.TestCase):
    def test_refuses_at_low_n(self):
        # kappa/p = 0.05, n = 70: the canonical infeasible cell.
        est = kappa_hat(0.05, 512, 70)
        self.assertEqual(est.status, REFUSAL_STATUS)
        self.assertEqual(est.status, "not resolvable at this n")
        self.assertIsNone(est.kappa)
        self.assertIsNotNone(est.rel_rmse)
        self.assertGreater(est.rel_rmse, est.target_rel_rmse)

    def test_accepts_high_concentration_corner(self):
        # The cell the rectangular rule falsely rejects: p=512, Rbar=0.85,
        # n/p = 0.5 is one of the best-conditioned cells in the program.
        est = kappa_hat(0.85, 512, 256)
        self.assertEqual(est.status, OK_STATUS)
        self.assertIsNotNone(est.kappa)
        self.assertLess(est.rel_rmse, 0.01)

    def test_refuse_false_returns_number(self):
        est = kappa_hat(0.05, 512, 70, refuse=False)
        self.assertEqual(est.status, OK_STATUS)
        self.assertIsNotNone(est.kappa)
        self.assertGreater(est.rel_rmse, 0.01)

    def test_no_n_means_no_check(self):
        est = kappa_hat(0.05, 512)
        self.assertEqual(est.status, OK_STATUS)
        self.assertIsNone(est.rel_rmse)
        self.assertIsNone(est.n)

    def test_rbar_zero_is_uniform_consistent(self):
        est = kappa_hat(0.0, 512, 10)
        self.assertEqual(est.status, REFUSAL_STATUS)  # surface is inf at r=0
        est_ok = kappa_hat(0.0, 512)
        self.assertEqual(est_ok.status, OK_STATUS)
        self.assertEqual(est_ok.kappa, 0.0)


if __name__ == "__main__":
    unittest.main()
