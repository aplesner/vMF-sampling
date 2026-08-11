import math
import unittest

import torch

from src.torch_log_iv import torch_log_iv
from src.torch_random import make_generator
from src.vmf_kappa_torch import bessel_ratio_from_logs_torch
from src.vmf_torch_batched import sample_vmf_rows
from src.vmf_torch_hh import TorchvMFHH


def bessel_ratio(kappa: float, dim: int) -> float:
    k = torch.tensor([kappa], dtype=torch.float64)
    return float(bessel_ratio_from_logs_torch(k, float(dim), torch_log_iv)[0])


def bessel_ratio_derivative(kappa: float, dim: int) -> float:
    a = bessel_ratio(kappa, dim)
    return 1.0 - a * a - (dim - 1.0) * a / kappa


def ks_statistic(x: torch.Tensor, y: torch.Tensor) -> float:
    xs = torch.sort(x).values
    ys = torch.sort(y).values
    grid = torch.sort(torch.cat([xs, ys])).values
    cdf_x = torch.searchsorted(xs, grid, right=True).double() / xs.numel()
    cdf_y = torch.searchsorted(ys, grid, right=True).double() / ys.numel()
    return float((cdf_x - cdf_y).abs().max())


class BatchedShapeAndApiTests(unittest.TestCase):
    def test_shape_dtype_and_unit_norm(self):
        gen = make_generator(torch.device("cpu"), 0)
        mu = torch.randn(257, 64, dtype=torch.float64)
        samples = sample_vmf_rows(mu, 10.0, generator=gen)
        self.assertEqual(samples.shape, (257, 64))
        self.assertEqual(samples.dtype, torch.float64)
        norms = samples.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-12, atol=1e-12)

    def test_output_dtype_follows_request(self):
        gen = make_generator(torch.device("cpu"), 1)
        mu = torch.randn(33, 16)
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            out = sample_vmf_rows(mu, 5.0, generator=gen, dtype=dtype)
            self.assertEqual(out.dtype, dtype)
            self.assertTrue(torch.all(torch.isfinite(out)))

    def test_scalar_and_row_kappa_broadcast(self):
        gen = make_generator(torch.device("cpu"), 2)
        mu = torch.randn(8, 4)
        self.assertEqual(sample_vmf_rows(mu, 3.0, generator=gen).shape, (8, 4))
        kappas = torch.linspace(0.0, 50.0, 8)
        self.assertEqual(sample_vmf_rows(mu, kappas, generator=gen).shape, (8, 4))

    def test_rejects_bad_inputs(self):
        mu = torch.randn(4, 8)
        with self.assertRaises(ValueError):
            sample_vmf_rows(mu, -1.0)
        with self.assertRaises(ValueError):
            sample_vmf_rows(torch.zeros(4, 8), 1.0)
        with self.assertRaises(ValueError):
            sample_vmf_rows(mu, torch.ones(5))

    def test_generator_reproducible_and_global_rng_untouched(self):
        mu = torch.randn(1000, 32)
        kappa = torch.linspace(0.0, 100.0, 1000)
        a = sample_vmf_rows(mu, kappa, generator=make_generator(torch.device("cpu"), 7))
        b = sample_vmf_rows(mu, kappa, generator=make_generator(torch.device("cpu"), 7))
        self.assertTrue(torch.equal(a, b))

        torch.manual_seed(0)
        expected = torch.rand(4)
        torch.manual_seed(0)
        sample_vmf_rows(mu, kappa, generator=make_generator(torch.device("cpu"), 3))
        self.assertTrue(torch.equal(torch.rand(4), expected))

    def test_north_pole_row_is_finite(self):
        gen = make_generator(torch.device("cpu"), 11)
        mu = torch.zeros(16, 5)
        mu[:, 0] = 1.0
        out = sample_vmf_rows(mu, 10.0, generator=gen)
        self.assertTrue(torch.all(torch.isfinite(out)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_runs_on_cuda(self):
        device = torch.device("cuda")
        mu = torch.randn(1024, 128, device=device)
        out = sample_vmf_rows(mu, 20.0, generator=make_generator(device, 5))
        self.assertEqual(out.device.type, "cuda")
        norms = out.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-5)


class BatchedDistributionTests(unittest.TestCase):
    def check_mean_cosine(self, dim: int, kappa: float, n: int, seed: int) -> None:
        gen = make_generator(torch.device("cpu"), seed)
        mu = torch.randn(n, dim, dtype=torch.float64)
        samples = sample_vmf_rows(mu, kappa, generator=gen)
        cosine = (samples * (mu / mu.norm(dim=-1, keepdim=True))).sum(dim=-1)
        expected = bessel_ratio(kappa, dim)
        sd = math.sqrt(bessel_ratio_derivative(kappa, dim))
        self.assertLess(
            abs(float(cosine.mean()) - expected),
            6.0 * sd / math.sqrt(n),
            msg=f"dim={dim} kappa={kappa}",
        )

    def test_mean_cosine_matches_bessel_ratio(self):
        for dim in (4, 16, 128):
            for kappa in (0.5, 5.0, 50.0, 500.0):
                self.check_mean_cosine(dim, kappa, n=8192, seed=hash((dim, 1)) % 2**31)

    def test_mean_cosine_dim2_and_dim3(self):
        for dim in (2, 3):
            for kappa in (0.5, 5.0, 50.0):
                self.check_mean_cosine(dim, kappa, n=16384, seed=hash((dim, 2)) % 2**31)

    def test_zero_kappa_is_uniform(self):
        dim, n = 32, 32768
        gen = make_generator(torch.device("cpu"), 13)
        mu = torch.randn(n, dim, dtype=torch.float64)
        samples = sample_vmf_rows(mu, 0.0, generator=gen)
        cosine = (samples * (mu / mu.norm(dim=-1, keepdim=True))).sum(dim=-1)
        self.assertLess(abs(float(cosine.mean())), 6.0 / math.sqrt(n * dim))

    def test_heterogeneous_kappa_in_one_call(self):
        dim, n_per = 64, 8192
        kappas = [0.0, 1.0, 20.0, 1000.0]
        gen = make_generator(torch.device("cpu"), 17)
        kappa_col = torch.tensor(kappas, dtype=torch.float64).repeat_interleave(n_per)
        mu = torch.randn(n_per * len(kappas), dim, dtype=torch.float64)
        samples = sample_vmf_rows(mu, kappa_col, generator=gen)
        cosine = (samples * (mu / mu.norm(dim=-1, keepdim=True))).sum(dim=-1)
        for i, kappa in enumerate(kappas):
            group = cosine[i * n_per : (i + 1) * n_per]
            if kappa == 0.0:
                self.assertLess(abs(float(group.mean())), 6.0 / math.sqrt(n_per * dim))
            else:
                expected = bessel_ratio(kappa, dim)
                sd = math.sqrt(bessel_ratio_derivative(kappa, dim))
                self.assertLess(
                    abs(float(group.mean()) - expected),
                    6.0 * sd / math.sqrt(n_per),
                    msg=f"kappa={kappa}",
                )

    def test_transverse_component_is_isotropic(self):
        dim, n = 16, 16384
        gen = make_generator(torch.device("cpu"), 19)
        mu = torch.zeros(n, dim, dtype=torch.float64)
        mu[:, 0] = 1.0
        samples = sample_vmf_rows(mu, 30.0, generator=gen)
        transverse = samples[:, 1:]
        self.assertLess(float(transverse.mean(dim=0).abs().max()), 6.0 / math.sqrt(n * dim))

    def test_matches_scalar_sampler_marginal(self):
        dim, kappa, n = 32, 8.0, 20000
        mu = torch.zeros(dim, dtype=torch.float64)
        mu[0] = 1.0
        scalar = TorchvMFHH(dim, mu=mu, kappa=kappa, seed=23, dtype=torch.float64)
        reference = scalar.sample(n)[:, 0]
        batched = sample_vmf_rows(
            mu.expand(n, dim).clone(),
            kappa,
            generator=make_generator(torch.device("cpu"), 29),
        )[:, 0]
        # Two-sample KS, alpha = 0.001 critical value.
        threshold = 1.949 * math.sqrt(2.0 / n)
        self.assertLess(ks_statistic(batched, reference), threshold)


if __name__ == "__main__":
    unittest.main()
