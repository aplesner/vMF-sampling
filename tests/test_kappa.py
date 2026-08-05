import unittest

import numpy as np

from src.vmf_kappa import (
    banerjee_kappa,
    bessel_ratio_from_logs,
    converged_newton_kappa,
    direct_log_likelihood_kappa,
    mean_resultant_length,
    scipy_brentq_kappa,
    scipy_log_iv,
    scipy_ratio,
    sra_kappa,
)


class KappaEstimatorTests(unittest.TestCase):
    def test_mean_resultant_length(self):
        samples = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]]])
        np.testing.assert_allclose(mean_resultant_length(samples), [np.sqrt(0.5), 1.0])

    def test_uniform_resultant_has_zero_estimate(self):
        r_bar = np.zeros(3)
        dimensions = np.array([2, 10, 500])
        np.testing.assert_array_equal(banerjee_kappa(r_bar, dimensions), 0.0)
        np.testing.assert_array_equal(sra_kappa(r_bar, dimensions), 0.0)
        result = converged_newton_kappa(r_bar, dimensions)
        np.testing.assert_array_equal(result.kappa, 0.0)
        self.assertTrue(np.all(result.converged))

    def test_log_bessel_ratio_matches_scaled_scipy_ratio(self):
        dimensions = np.array([3, 16, 128, 500])
        kappas = np.array([0.1, 5.0, 100.0, 1000.0])
        actual = bessel_ratio_from_logs(kappas, dimensions, scipy_log_iv)
        expected = np.array([scipy_ratio(k, int(p)) for k, p in zip(kappas, dimensions)])
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-15)

    def test_reproduces_selected_sra_table_2_errors(self):
        # Sra reports errors after evaluating R_bar=A_p(kappa_true), rather than
        # after drawing finite samples.  These cases cover different p/kappa regimes.
        dimensions = np.array([500, 500, 1000, 5000], dtype=int)
        true_kappa = np.array([100.0, 5000.0, 1000.0, 10000.0])
        r_bar = np.array(
            [scipy_ratio(k, int(p)) for k, p in zip(true_kappa, dimensions)]
        )
        banerjee_error = np.abs(banerjee_kappa(r_bar, dimensions) - true_kappa)
        sra_error = np.abs(sra_kappa(r_bar, dimensions) - true_kappa)
        np.testing.assert_allclose(
            banerjee_error,
            [6.84e-3, 4.52e-1, 1.71e-1, 2.96e-1],
            rtol=6e-3,
            atol=2e-5,
        )
        self.assertTrue(np.all(sra_error < 1e-7))

    def test_all_numerical_mle_methods_agree(self):
        dimensions = np.array([3, 16, 128])
        r_bar = np.array([0.15, 0.7, 0.95])
        reference = scipy_brentq_kappa(r_bar, dimensions)
        sra = sra_kappa(r_bar, dimensions)
        converged = converged_newton_kappa(r_bar, dimensions)
        direct = direct_log_likelihood_kappa(r_bar, dimensions)
        self.assertTrue(np.all(converged.converged))
        np.testing.assert_allclose(sra, reference, rtol=2e-8, atol=1e-10)
        np.testing.assert_allclose(converged.kappa, reference, rtol=2e-10, atol=1e-11)
        np.testing.assert_allclose(direct, reference, rtol=2e-6, atol=2e-7)

    def test_rejects_invalid_inputs(self):
        for r_bar, dimension in [(-0.1, 3), (1.0, 3), (0.5, 1), (0.5, 2.5)]:
            with self.assertRaises(ValueError):
                banerjee_kappa(r_bar, dimension)


if __name__ == "__main__":
    unittest.main()
