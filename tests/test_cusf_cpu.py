import unittest
from pathlib import Path

import numpy as np
import torch

from src.cusf_cpu import iv_log
from src.vmf_kappa import scipy_log_iv
from src.vmf_kappa_torch import converged_newton_kappa_torch, sra_kappa_torch


_CUSF_AVAILABLE = (Path(__file__).resolve().parents[2] / "cusf" / "cusf" / "src").exists()


@unittest.skipUnless(_CUSF_AVAILABLE, "sibling CUSF checkout is unavailable")
class CusfCpuExtensionTests(unittest.TestCase):
    def test_compiled_log_iv_matches_scipy(self):
        order = np.array([0.0, 0.5, 1.0, 10.0, 50.0, 100.0, 500.0])
        argument = np.array([0.01, 0.1, 1.0, 10.0, 30.0, 100.0, 1000.0])
        expected = scipy_log_iv(order, argument)
        for dtype, tolerance in [(torch.float64, 2e-12), (torch.float32, 2e-4)]:
            actual = iv_log(
                torch.tensor(order, dtype=dtype), torch.tensor(argument, dtype=dtype)
            ).numpy()
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)

    def test_compiled_cusf_drives_estimators(self):
        r_bar = torch.tensor([0.15, 0.7, 0.95], dtype=torch.float64)
        dimensions = torch.tensor([3.0, 16.0, 128.0], dtype=torch.float64)
        sra = sra_kappa_torch(r_bar, dimensions, iv_log)
        converged = converged_newton_kappa_torch(r_bar, dimensions, iv_log)
        self.assertTrue(torch.all(converged.converged))
        torch.testing.assert_close(sra, converged.kappa, rtol=2e-8, atol=1e-10)

    def test_rejects_unsupported_16_bit_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.assertRaisesRegex(RuntimeError, "float32 and float64"):
                iv_log(torch.ones(2, dtype=dtype), torch.ones(2, dtype=dtype))


if __name__ == "__main__":
    unittest.main()
