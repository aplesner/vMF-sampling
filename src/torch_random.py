"""Generator-aware random helpers for the PyTorch samplers.

PyTorch's distribution ``sample`` methods do not accept a ``Generator``.  The
small helpers here use the same algorithms while taking an explicit generator,
so sampler instances no longer share or reset PyTorch's process-wide RNG.
"""

from __future__ import annotations

import math

import torch


def make_generator(device: torch.device, seed: int | None) -> torch.Generator:
    """Create an independent generator on ``device``.

    ``Generator.seed`` obtains a non-deterministic seed when none was supplied;
    otherwise ``manual_seed`` gives reproducible sampler instances.
    """
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    return generator


def sample_symmetric_beta(
    concentration: float,
    size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw ``Beta(concentration, concentration)`` with an explicit RNG."""
    work_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    parameters = torch.full((size,), concentration, device=device, dtype=work_dtype)
    # This is the same gamma construction used by torch.distributions.Beta.
    left = torch._standard_gamma(parameters, generator=generator)
    right = torch._standard_gamma(parameters, generator=generator)
    return (left / (left + right)).to(dtype=dtype)


@torch.no_grad()
def sample_von_mises(
    loc: torch.Tensor,
    concentration: float,
    size: int,
    *,
    generator: torch.Generator,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Best--Fisher von Mises sampler with an explicit PyTorch generator.

    This follows ``torch.distributions.VonMises`` and computes internally in
    float64 to retain its behavior for small concentrations.
    """
    loc64 = loc.to(dtype=torch.float64)
    kappa = torch.as_tensor(concentration, device=loc.device, dtype=torch.float64)
    tau = 1.0 + torch.sqrt(1.0 + 4.0 * kappa**2)
    rho = (tau - torch.sqrt(2.0 * tau)) / (2.0 * kappa)
    proposal = (1.0 + rho**2) / (2.0 * rho)
    proposal = torch.where(kappa < 1e-5, 1.0 / kappa + kappa, proposal)

    result = torch.empty(size, device=loc.device, dtype=torch.float64)
    done = torch.zeros(size, device=loc.device, dtype=torch.bool)
    while not bool(done.all()):
        uniform = torch.rand(
            (3, size),
            device=loc.device,
            dtype=torch.float64,
            generator=generator,
        )
        u1, u2, u3 = uniform.unbind()
        z = torch.cos(math.pi * u1)
        f = (1.0 + proposal * z) / (proposal + z)
        c = kappa * (proposal - f)
        accept = (c * (2.0 - c) > u2) | (torch.log(c / u2) + 1.0 - c >= 0.0)
        newly_accepted = accept & ~done
        result[newly_accepted] = torch.sign(u3[newly_accepted] - 0.5) * torch.acos(
            f[newly_accepted]
        )
        done |= accept

    result = (result + math.pi + loc64) % (2.0 * math.pi) - math.pi
    return result.to(dtype=output_dtype)
