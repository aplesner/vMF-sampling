from __future__ import annotations

from pathlib import Path

import numpy as np

from src.vmf import vMF

SAVE_SAMPLES = False
OUTPUT_PATH = Path("data_samples/vmf_samples.npy")


def main() -> None:
    dim = 3
    kappa = 5.0
    num_samples = 1_000

    mu = np.zeros(dim, dtype=np.float64)
    mu[0] = 1.0

    sampler = vMF(dim=dim, kappa=kappa, backend="numpy")
    samples = sampler.sample(num_samples, mu=mu)
    print(f"samples shape: {samples.shape}")

    if SAVE_SAMPLES:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(OUTPUT_PATH, samples)
        print(f"saved samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
