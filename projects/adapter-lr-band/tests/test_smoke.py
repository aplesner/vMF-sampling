"""End-to-end plumbing smoke test: a few training steps per arm on the
synthetic TinyViT path. Verifies effective-step logging, divergence
detection, and CSV serialization. Not a result."""

import math
import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adapters import ARMS  # noqa: E402
from train import train_run  # noqa: E402


class TestSmoke(unittest.TestCase):
    def test_short_run_each_arm(self):
        dev = torch.device("cpu")
        for arm in ARMS:
            rec, steps = train_run(arm, rank=2, lr=1e-3, seed=0,
                                   arch="tinyvit", dataset="synthetic",
                                   epochs=1, device=dev, log_every=1)
            self.assertFalse(rec.nan_encountered, arm)
            self.assertTrue(math.isfinite(rec.metric), arm)
            self.assertGreater(len(steps), 2, arm)
            # Effective-step series must be finite, non-negative, nonzero
            # after init for trainable arms.
            effs = [s["eff_step_mean"] for s in steps]
            self.assertEqual(effs[0], 0.0)
            self.assertTrue(all(math.isfinite(e) and e >= 0 for e in effs), arm)
            self.assertGreater(sum(effs), 0.0, arm)
            self.assertGreater(rec.eff_step_run_mean, 0.0, arm)
            # Coherence logged and bounded.
            self.assertTrue(all(0.0 <= s["coh_u_max"] <= 1.0 for s in steps), arm)

    def test_absurd_lr_flags_divergence(self):
        dev = torch.device("cpu")
        rec, _ = train_run("lora", rank=2, lr=1e4, seed=0,
                           arch="tinyvit", dataset="synthetic",
                           epochs=1, device=dev)
        self.assertTrue(rec.diverged(),
                        "lr=1e4 must trigger the pre-registered divergence "
                        "criterion")


if __name__ == "__main__":
    unittest.main()
