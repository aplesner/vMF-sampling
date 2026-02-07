from __future__ import annotations

from pathlib import Path

import numpy as np

from src.vmf import vMF

SAVE_SAMPLES = False


def main() -> None:
    dim = 3
    kappa = 5.0
    num_samples = 1_000
    output_path = Path(f"data_samples/vmf_samples_{dim}d_{kappa}kappa_{num_samples}samples.npy")

    mu = np.random.randn(dim)
    mu /= np.linalg.norm(mu)

    sampler = vMF(dim=dim, kappa=kappa, backend="numpy_hh")
    samples = sampler.sample(num_samples, mu=mu)
    print(f"samples shape: {samples.shape}")

    if SAVE_SAMPLES:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, samples)
        print(f"saved samples to {output_path}")


if __name__ == "__main__":
    main()
