import unittest

import numpy as np
import torch

from src.torch_log_iv import torch_log_iv
from src.vmf_kappa import scipy_brentq_kappa, scipy_log_iv
from src.vmf_kappa_torch import (
    banerjee_kappa_torch,
    bessel_ratio_from_logs_torch,
    converged_newton_kappa_torch,
    direct_log_likelihood_kappa_torch,
    sra_kappa_torch,
)


class TorchLogIvTests(unittest.TestCase):
    def test_float64_matches_scipy_where_scipy_is_finite(self):
        orders = np.array([0.0, 0.5, 1.0, 10.0, 50.0, 100.0, 500.0])
        arguments = np.array([0.01, 0.1, 1.0, 10.0, 30.0, 100.0, 1000.0])
        order_grid, argument_grid = np.meshgrid(orders, arguments, indexing="ij")
        expected = scipy_log_iv(order_grid, argument_grid)
        actual = torch_log_iv(
            torch.from_numpy(order_grid), torch.from_numpy(argument_grid)
        ).numpy()
        finite = np.isfinite(expected)
        np.testing.assert_allclose(actual[finite], expected[finite], rtol=2e-13, atol=2e-11)
        self.assertTrue(np.all(np.isfinite(actual)))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_runs_on_mps(self):
        order = torch.tensor([0.0, 1.0, 10.0, 100.0], device="mps")
        argument = torch.tensor([0.1, 1.0, 10.0, 100.0], device="mps")
        result = torch_log_iv(order, argument).cpu().numpy()
        expected = scipy_log_iv(order.cpu().numpy(), argument.cpu().numpy())
        np.testing.assert_allclose(result, expected, rtol=2e-5, atol=2e-5)


class TorchKappaEstimatorTests(unittest.TestCase):
    def test_methods_match_cpu_reference_in_float64(self):
        r_bar = torch.tensor([0.15, 0.7, 0.95], dtype=torch.float64)
        dimensions = torch.tensor([3.0, 16.0, 128.0], dtype=torch.float64)
        reference = scipy_brentq_kappa(r_bar.numpy(), dimensions.numpy())

        banerjee = banerjee_kappa_torch(r_bar, dimensions)
        sra = sra_kappa_torch(r_bar, dimensions, torch_log_iv)
        converged = converged_newton_kappa_torch(r_bar, dimensions, torch_log_iv)
        direct = direct_log_likelihood_kappa_torch(
            r_bar, dimensions, torch_log_iv, iterations=72
        )

        self.assertTrue(torch.all(torch.isfinite(banerjee)))
        self.assertTrue(torch.all(converged.converged))
        np.testing.assert_allclose(sra.numpy(), reference, rtol=2e-8, atol=1e-10)
        np.testing.assert_allclose(
            converged.kappa.numpy(), reference, rtol=2e-10, atol=1e-11
        )
        np.testing.assert_allclose(direct.numpy(), reference, rtol=2e-7, atol=2e-8)

    def test_ratio_runs_in_float32(self):
        kappa = torch.tensor([1.0, 10.0, 100.0], dtype=torch.float32)
        dimensions = torch.tensor([3.0, 16.0, 128.0], dtype=torch.float32)
        ratio = bessel_ratio_from_logs_torch(kappa, dimensions, torch_log_iv)
        self.assertTrue(torch.all((ratio > 0.0) & (ratio < 1.0)))


if __name__ == "__main__":
    unittest.main()
