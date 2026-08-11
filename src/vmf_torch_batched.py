"""Device-native per-row (mu_i, kappa_i) batched vMF sampling for PyTorch.

Unlike the single-(mu, kappa) samplers in ``vmf_torch.py`` / ``vmf_torch_hh.py``,
this module draws one sample per row from vMF(mu_i, kappa_i) in a single call.
It exists for latent-noise injection and S-VAE-style workloads where every row
of a batch carries its own mean direction and concentration.

Algorithm: Wood (1994) / Ulrich rejection sampling for the polar coordinate,
tensorized over rows.  The symmetric-Beta proposal concentration depends only
on the dimension, so proposals remain row-independent; only the transform and
the acceptance test are kappa-dependent.  The acceptance loop uses
mask-and-refill over not-yet-accepted rows.  The envelope parameter is computed
in the rationalized form ``b = d / (2*kappa + sqrt(4*kappa^2 + d^2))``, which is
cancellation-free at both kappa -> 0 (b -> 1, acceptance becomes automatic,
recovering the uniform distribution) and kappa -> infinity (b -> 0).

Orientation uses one Householder reflection per row mapping e1 -> mu_i, valid
because vMF is rotationally symmetric about mu_i.
"""

from __future__ import annotations

import math

import torch

from .torch_random import make_generator, sample_symmetric_beta, sample_von_mises

_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_MAX_REJECTION_ROUNDS = 10_000


def _calc_dtype(dtype: torch.dtype) -> torch.dtype:
    # log/exp in the rejection test are numerically fragile in half precision;
    # compute in float32 minimum, mirroring the scalar samplers.
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _polar_dim2(
    mu: torch.Tensor, kappa: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """vMF on the circle reduces to von Mises; angles are absolute, no rotation."""
    mean_angle = torch.atan2(mu[:, 1], mu[:, 0])
    angles = sample_von_mises(
        mean_angle,
        kappa,
        mu.shape[0],
        generator=generator,
        output_dtype=mu.dtype,
    )
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)


def _polar_coordinate_dim3(
    kappa: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Closed-form inverse-CDF polar coordinate about the north pole, per row.

    Written with expm1/log1p so the kappa -> 0 limit is exact: the expression
    tends to ``2*u - 1`` (the uniform marginal) and only exactly-zero kappa
    needs the explicit fallback.
    """
    u = torch.rand(
        kappa.shape, device=kappa.device, dtype=kappa.dtype, generator=generator
    )
    u = torch.clamp(u, min=1e-12)
    inner = torch.log1p((1.0 - u) * torch.expm1(-2.0 * kappa))
    x = torch.where(kappa > 0.0, 1.0 + inner / kappa.clamp_min(1e-300), 2.0 * u - 1.0)
    return x.clamp(-1.0, 1.0)


def _polar_coordinate_rejection(
    kappa: torch.Tensor, dim: int, generator: torch.Generator
) -> torch.Tensor:
    """Wood/Ulrich rejection sampling of the polar coordinate, per row."""
    d = float(dim - 1)
    halfdim = 0.5 * d
    # Rationalized: equals (-2k + sqrt(4k^2 + d^2)) / d without cancellation.
    b = d / (2.0 * kappa + torch.sqrt(4.0 * kappa.square() + d * d))
    node = (1.0 - b) / (1.0 + b)
    correction = kappa * node + d * (
        math.log(4.0) + torch.log(b) - 2.0 * torch.log1p(b)
    )

    n = kappa.shape[0]
    x = torch.empty(n, device=kappa.device, dtype=kappa.dtype)
    done = torch.zeros(n, device=kappa.device, dtype=torch.bool)
    for _ in range(_MAX_REJECTION_ROUNDS):
        active = ~done
        n_active = int(active.sum())
        if n_active == 0:
            break
        sym_beta = sample_symmetric_beta(
            halfdim,
            n_active,
            device=kappa.device,
            dtype=kappa.dtype,
            generator=generator,
        )
        b_a = b[active]
        k_a = kappa[active]
        c_a = correction[active]
        proposal = (1.0 - (1.0 + b_a) * sym_beta) / (1.0 - (1.0 - b_a) * sym_beta)
        log_u = torch.log(
            torch.rand(n_active, device=kappa.device, dtype=kappa.dtype, generator=generator)
        )
        accept = (
            k_a * proposal
            + d * torch.log((1.0 + b_a - proposal + proposal * b_a) / (1.0 + b_a))
            - c_a
            > log_u
        )
        active_idx = active.nonzero(as_tuple=True)[0]
        x[active_idx[accept]] = proposal[accept]
        done[active_idx[accept]] = True
    else:
        raise RuntimeError(
            f"vMF rejection sampling did not finish in {_MAX_REJECTION_ROUNDS} rounds"
        )
    return x


def _rotate_rows(samples: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Apply one Householder reflection per row mapping e1 -> mu_i, in place.

    Rows with mu_i == e1 get u = 0 exactly (v == 0), making the update a no-op,
    so no special-casing of the north pole is needed.
    """
    v = -mu.clone()
    v[:, 0] += 1.0
    tiny = torch.finfo(samples.dtype).tiny
    u = v / v.norm(dim=-1, keepdim=True).clamp_min(tiny)
    projection = (samples * u).sum(dim=-1, keepdim=True)
    samples.addcmul_(projection, u, value=-2.0)
    return samples


@torch.no_grad()
def sample_vmf_rows(
    mu: torch.Tensor,
    kappa: torch.Tensor | float,
    *,
    generator: torch.Generator | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Draw one vMF(mu_i, kappa_i) sample per row.

    Parameters
    ----------
    mu:
        ``(B, p)`` (or ``(p,)``) mean directions. Rows are normalized
        internally; zero rows are rejected. Need not be unit norm on entry.
    kappa:
        ``(B,)`` concentrations, or a scalar broadcast to all rows.
        ``kappa >= 0``; zero rows draw from the uniform distribution.
    generator:
        Optional per-call ``torch.Generator`` on ``mu.device``. A fresh
        nondeterministic generator is created when omitted; the global RNG is
        never touched.
    dtype:
        Output dtype (default: ``mu.dtype``). Half-precision outputs are
        computed internally in float32.

    Returns
    -------
    ``(B, p)`` unit-norm samples in ``dtype`` on ``mu.device``.
    """
    if not isinstance(mu, torch.Tensor):
        raise TypeError("mu must be a torch tensor")
    if mu.ndim == 1:
        mu = mu.unsqueeze(0)
    if mu.ndim != 2:
        raise ValueError(f"mu must have shape (B, p), got {tuple(mu.shape)}")
    out_dtype = dtype if dtype is not None else mu.dtype
    if out_dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {_DTYPES}, got {out_dtype}")

    batch, dim = mu.shape
    if dim < 2:
        raise ValueError("dim must be greater than 1")
    device = mu.device
    work_dtype = _calc_dtype(out_dtype)
    if generator is None:
        generator = make_generator(device, None)

    kappa_t = torch.as_tensor(kappa, device=device, dtype=work_dtype).reshape(-1)
    if kappa_t.numel() == 1:
        kappa_t = kappa_t.expand(batch)
    elif kappa_t.numel() != batch:
        raise ValueError(
            f"kappa must be scalar or have {batch} rows, got {kappa_t.numel()}"
        )
    if bool((kappa_t < 0.0).any()):
        raise ValueError("kappa must be non-negative")

    mu_w = mu.to(dtype=work_dtype)
    row_norm = mu_w.norm(dim=-1, keepdim=True)
    if bool((row_norm == 0.0).any()):
        raise ValueError("mu rows must be nonzero")
    mu_w = mu_w / row_norm

    if dim == 2:
        return _polar_dim2(mu_w, kappa_t, generator).to(dtype=out_dtype)

    if dim == 3:
        x = _polar_coordinate_dim3(kappa_t, generator)
    else:
        x = _polar_coordinate_rejection(kappa_t, dim, generator)

    circ = torch.randn(
        (batch, dim - 1), device=device, dtype=work_dtype, generator=generator
    )
    circ.div_(circ.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(work_dtype).tiny))
    coord_rest = torch.sqrt(torch.clamp(1.0 - x.square(), min=0.0)).unsqueeze(-1) * circ

    samples = torch.cat([x.unsqueeze(-1), coord_rest], dim=-1)
    return _rotate_rows(samples, mu_w).to(dtype=out_dtype)
