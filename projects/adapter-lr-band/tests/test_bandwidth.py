"""Tests for the band-width estimand, divergence criterion, and the
mandatory effective-step re-expression."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bandwidth import (GRID, RunRecord, band_width, band_width_rel,  # noqa: E402
                       eta_diverged, reexpress_bands_effstep)


def rec(lr, metric, arm="a", seed=0, nan=False, eff=float("nan")):
    return RunRecord(arm=arm, rank=4, lr=lr, seed=seed, metric=metric,
                     nan_encountered=nan, loss_init=2.3,
                     loss_final_window=1.0 if not nan else float("nan"),
                     grad_norm_max=1.0, eff_step_run_mean=eff)


class TestGrid(unittest.TestCase):
    def test_common_grid(self):
        self.assertEqual(len(GRID), 13)
        self.assertAlmostEqual(GRID[0], 1e-5, places=12)
        self.assertAlmostEqual(GRID[-1], 1e-1, places=12)
        # 1/3-decade spacing
        for a, b in zip(GRID, GRID[1:]):
            self.assertAlmostEqual(math.log10(b) - math.log10(a), 1 / 3, places=9)


class TestBandWidth(unittest.TestCase):
    def test_exact_width(self):
        """Peak at 1e-3; within tau=1.0 from 1e-4 to 1e-2 -> W = 2 decades."""
        metrics = {1e-5: 80.0, 1e-4: 90.0, 1e-3: 91.0, 1e-2: 90.5, 1e-1: 70.0}
        recs = [rec(lr, m) for lr, m in metrics.items()]
        b = band_width(recs, tau_abs=1.0)
        self.assertAlmostEqual(b.peak, 91.0)
        self.assertAlmostEqual(b.lo, 1e-4, places=12)
        self.assertAlmostEqual(b.hi, 1e-2, places=12)
        self.assertAlmostEqual(b.width_decades, 2.0, places=9)

    def test_seed_mean_and_divergence_as_data(self):
        """Diverged runs contribute -inf; the peak must not come from them."""
        recs = [
            rec(1e-3, 90.0, seed=0), rec(1e-3, 92.0, seed=1),
            rec(1e-1, 99.0, seed=0, nan=True), rec(1e-1, 99.0, seed=1, nan=True),
        ]
        b = band_width(recs, tau_abs=1.0)
        self.assertAlmostEqual(b.peak, 91.0)          # not 99: those diverged
        self.assertEqual(b.n_in_band, 1)
        self.assertAlmostEqual(b.width_decades, 0.0)  # one-point band
        self.assertAlmostEqual(eta_diverged(recs), 1e-1, places=12)

    def test_loss_above_init_is_divergence(self):
        r = rec(1e-2, 88.0)
        r.loss_final_window = r.loss_init + 0.5  # ended worse than init
        self.assertTrue(r.diverged())

    def test_gradnorm_is_divergence(self):
        r = rec(1e-2, 88.0)
        r.grad_norm_max = 2e4
        self.assertTrue(r.diverged())

    def test_relative_tolerance(self):
        metrics = {1e-4: 100.0, 1e-3: 99.0, 1e-2: 97.9}
        recs = [rec(lr, m) for lr, m in metrics.items()]
        # tau_rel=2% of peak 100 -> threshold 98.0: {1e-4, 1e-3} -> 1 decade
        self.assertAlmostEqual(band_width_rel(recs, 0.02), 1.0, places=9)

    def test_effstep_reexpression(self):
        """Two arms whose nominal bands differ but whose effective steps
        coincide must have equal widths on the effective-step axis."""
        recs = []
        # LoRA: peak 90 at lr=1e-4, eff-step 1e-6; band on eff axis [1e-6,1e-5]
        recs += [rec(1e-4, 90.0, arm="lora", eff=1e-6),
                 rec(1e-3, 89.5, arm="lora", eff=1e-5),
                 rec(1e-2, 80.0, arm="lora", eff=1e-4)]
        # DeLoRA-PC: nominal band much wider, but same eff-step band
        recs += [rec(1e-3, 90.0, arm="delora-pc", eff=1e-6),
                 rec(1e-2, 89.5, arm="delora-pc", eff=1e-5),
                 rec(1e-1, 80.0, arm="delora-pc", eff=1e-4)]
        widths = reexpress_bands_effstep(recs, tau_abs=1.0)
        self.assertAlmostEqual(widths["lora"], 1.0, places=6)
        self.assertAlmostEqual(widths["delora-pc"], 1.0, places=6)
        self.assertAlmostEqual(widths["lora"] - widths["delora-pc"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
