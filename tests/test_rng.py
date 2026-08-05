from __future__ import annotations

import unittest

import numpy as np

from src.vmf_numpy import NumpyvMF
from src.vmf_numpy_hh import NumpyvMFHH
from src.vmf_scipy import ScipyvMF


class NumpyRNGTests(unittest.TestCase):
    def test_numpy_backends_use_pcg64dxsm(self) -> None:
        for sampler_type in (NumpyvMF, NumpyvMFHH):
            sampler = sampler_type(4, seed=7)
            self.assertIsInstance(sampler.random_state.bit_generator, np.random.PCG64DXSM)

    def test_numpy_seed_is_reproducible(self) -> None:
        for sampler_type in (NumpyvMF, NumpyvMFHH):
            left = sampler_type(4, seed=23).sample(128)
            right = sampler_type(4, seed=23).sample(128)
            np.testing.assert_array_equal(left, right)

    def test_scipy_baseline_keeps_legacy_random_state(self) -> None:
        sampler = ScipyvMF(4, seed=7)
        self.assertIsInstance(sampler.random_state, np.random.RandomState)


try:
    import torch

    from src.vmf_torch import TorchvMF
    from src.vmf_torch_hh import TorchvMFHH
except ImportError:  # pragma: no cover - PyTorch is optional
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchRNGTests(unittest.TestCase):
    def test_seed_is_reproducible_in_all_sampling_paths(self) -> None:
        for sampler_type in (TorchvMF, TorchvMFHH):
            for dim in (2, 3, 4):
                left = sampler_type(dim, seed=31, device="cpu").sample(128)
                right = sampler_type(dim, seed=31, device="cpu").sample(128)
                torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_sampler_does_not_modify_global_rng(self) -> None:
        torch.manual_seed(101)
        expected = torch.rand(8)

        torch.manual_seed(101)
        for sampler_type in (TorchvMF, TorchvMFHH):
            for dim in (2, 3, 4):
                sampler_type(dim, seed=5, device="cpu").sample(64)
        actual = torch.rand(8)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_householder_inplace_reuses_sample_storage(self) -> None:
        mu = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
        sampler = TorchvMFHH(4, mu=mu, seed=5, device="cpu", dtype=torch.float64)
        samples = torch.randn(32, 4, dtype=torch.float64)
        expected = sampler._rotate_householder(samples.clone())
        storage = samples.data_ptr()

        actual = sampler._rotate_householder_inplace(samples)

        self.assertEqual(actual.data_ptr(), storage)
        torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
