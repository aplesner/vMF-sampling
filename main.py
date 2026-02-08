from __future__ import annotations

from pathlib import Path

import numpy as np

from src.vmf import vMF

SAVE_SAMPLES = False


def main() -> None:
    dims = [3, 8, 32, 128, 512, 2048]
    kappa = 5.0
    num_samples = 1_000
    for dim in dims:
        mu = np.random.randn(dim)
        mu /= np.linalg.norm(mu)
        sampler = vMF(dim=dim, kappa=kappa, backend="numpy_hh")
        samples = sampler.sample(num_samples, mu=mu)
        print(f"samples shape: {samples.shape}")


if __name__ == "__main__":
    main()
