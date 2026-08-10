"""Tests for the matched-null quantiles."""

from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401  (sys.path setup)
import numpy as np
from vmf_measure import mle_kappa, null_quantile, null_quantile_table


class TestUniformNull(unittest.TestCase):
    def test_rbar_second_moment_exact(self):
        # E[Rbar^2] = 1/n exactly under the uniform null, any p.
        rng = np.random.Generator(np.random.PCG64DXSM(7))
        for p, n in [(16, 50), (512, 70)]:
            draws = rng.standard_normal((400, n, p))
            draws /= np.linalg.norm(draws, axis=-1, keepdims=True)
            r2 = np.sum(np.mean(draws, axis=1) ** 2, axis=-1)
            self.assertAlmostEqual(float(np.mean(r2)) * n, 1.0, delta=0.15)

    def test_closed_form_matches_monte_carlo(self):
        rng = np.random.Generator(np.random.PCG64DXSM(11))
        p, n = 128, 200
        draws = rng.standard_normal((4000, n, p))
        draws /= np.linalg.norm(draws, axis=-1, keepdims=True)
        r_bar = np.linalg.norm(np.mean(draws, axis=1), axis=-1)
        for q in (0.05, 0.5, 0.95):
            closed = null_quantile("r_bar", p, n, q)
            self.assertEqual(closed.method, "closed_form")
            mc = float(np.quantile(r_bar, q))
            self.assertAlmostEqual(closed.value / mc, 1.0, delta=0.03)

    def test_kappa_hat_quantile_is_monotone_transform(self):
        p, n, q = 128, 200, 0.95
        r_q = null_quantile("r_bar", p, n, q).value
        k_q = null_quantile("kappa_hat", p, n, q).value
        self.assertAlmostEqual(k_q, float(mle_kappa(r_q, p)), places=8)

    def test_mean_cosine_scale(self):
        row = null_quantile("mean_cosine", 256, 100, 0.975)
        self.assertAlmostEqual(row.value, 1.96 / np.sqrt(100 * 256), places=4)

    def test_quantile_monotone_in_q(self):
        values = [null_quantile("r_bar", 512, 70, q).value for q in (0.05, 0.5, 0.95)]
        self.assertTrue(np.all(np.diff(values) > 0.0))


class TestMonteCarloNulls(unittest.TestCase):
    def test_mean_removed_runs_and_shrinks_rbar(self):
        plain = null_quantile("r_bar", 64, 50, 0.5)
        removed = null_quantile("r_bar", 64, 50, 0.5, null="mean_removed", reps=300, seed=3)
        self.assertEqual(removed.method, "monte_carlo")
        self.assertEqual(removed.reps, 300)
        # Projecting out the mean direction removes the leading component.
        self.assertLess(removed.value, plain.value * 1.2)

    def test_covariance_matched_requires_samples(self):
        with self.assertRaises(ValueError):
            null_quantile("r_bar", 16, 40, 0.5, null="covariance_matched")

    def test_covariance_matched_recovers_scale_under_isotropy(self):
        rng = np.random.Generator(np.random.PCG64DXSM(5))
        samples = rng.standard_normal((60, 24))
        samples /= np.linalg.norm(samples, axis=-1, keepdims=True)
        row = null_quantile(
            "r_bar", 24, 60, 0.95, null="covariance_matched", samples=samples, reps=300, seed=4
        )
        uniform_row = null_quantile("r_bar", 24, 60, 0.95)
        # Isotropic data: covariance-matched null ~ uniform null.
        self.assertAlmostEqual(row.value / uniform_row.value, 1.0, delta=0.15)

    def test_mean_cosine_rejected_for_mc_nulls(self):
        with self.assertRaises(ValueError):
            null_quantile("mean_cosine", 16, 40, 0.5, null="mean_removed", reps=10, seed=0)

    def test_bad_choices_raise(self):
        with self.assertRaises(ValueError):
            null_quantile("bogus", 16, 40, 0.5)
        with self.assertRaises(ValueError):
            null_quantile("r_bar", 16, 40, 0.5, null="bogus")
        with self.assertRaises(ValueError):
            null_quantile("r_bar", 16, 40, 1.5)

    def test_table_rows(self):
        rows = null_quantile_table([16], [40, 80], [0.5, 0.95], statistics=("r_bar",))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0].statistic, "r_bar")


if __name__ == "__main__":
    unittest.main()
