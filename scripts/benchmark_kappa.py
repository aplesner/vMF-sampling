"""Benchmark von Mises--Fisher concentration estimators against Sra's method.

Three studies are kept separate, following the design in
``.pi/skills/vmf-kappa/SKILL.md``:

``population``
    Set ``r_bar`` from a high-precision population ratio ``A_p(kappa)``.  Every
    exact solver must recover the generating kappa, so this isolates numerical
    method error.  An ``mpmath`` reference replaces SciPy's scaled Bessel ratio,
    which underflows to ``0/0`` at large dimension.

``sampled``
    Draw independent vMF datasets with the repository's fast Householder
    samplers, reduce each once to ``r_bar``, and feed identical values to every
    method.  Numerical error against the high-precision MLE for the same sample
    is reported separately from statistical error against the generating kappa.

``timing``
    Time estimators on precomputed ``r_bar`` batches.  Sample generation and the
    reduction to ``r_bar`` are excluded.  Sampler cost is reported separately so
    the study's own generation budget is visible.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mpmath import besseli, findroot, mp, mpf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vmf_kappa import (  # noqa: E402
    banerjee_kappa,
    converged_newton_kappa,
    direct_log_likelihood_kappa,
    scipy_brentq_kappa,
    scipy_log_iv,
    sra_kappa,
)
from src.vmf_kappa_torch import (  # noqa: E402
    banerjee_kappa_torch,
    converged_newton_kappa_torch,
    sra_kappa_torch,
)
from src.vmf_numpy_hh import NumpyvMFHH  # noqa: E402
from src.vmf_scipy import ScipyvMF  # noqa: E402
from src.vmf_torch_hh import TorchvMFHH  # noqa: E402

DEFAULT_DIMENSIONS = (3, 16, 64, 256, 1024, 4096)
DEFAULT_KAPPAS = (1.0, 10.0, 50.0, 200.0, 1000.0)
MPMATH_DPS = 50


# --------------------------------------------------------------------------
# High-precision reference
# --------------------------------------------------------------------------


def population_r_bar(dimension: int, kappa: float) -> float:
    """Return ``A_p(kappa)`` evaluated at high precision."""

    with mp.workdps(MPMATH_DPS):
        order = mpf(dimension) / 2 - 1
        value = besseli(order + 1, mpf(kappa)) / besseli(order, mpf(kappa))
        return float(value)


def mpmath_mle_kappa(r_bar: float, dimension: int) -> float:
    """Solve ``A_p(kappa) = r_bar`` at high precision.

    Unlike the SciPy Brent reference this stays finite at every dimension in
    the benchmark grid, so it can serve as the MLE reference for sampled data.
    """

    if r_bar <= 0.0:
        return 0.0
    with mp.workdps(MPMATH_DPS):
        target = mpf(r_bar)
        order = mpf(dimension) / 2 - 1
        denominator = 1 - target**2
        lower = max(target * (mpf(dimension) - 2) / denominator, mpf("1e-12"))
        upper = target * mpf(dimension) / denominator

        def score(kappa: mpf) -> mpf:
            return besseli(order + 1, kappa) / besseli(order, kappa) - target

        root = findroot(score, (lower, upper), solver="anderson", tol=mpf("1e-40"))
        return float(root)


# --------------------------------------------------------------------------
# Estimator registry
# --------------------------------------------------------------------------

Estimator = Callable[[np.ndarray, int], np.ndarray]


def _torch_log_iv_factory(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if name == "cusf":
        from src.cusf_cpu import iv_log

        return iv_log
    if name == "torch":
        from src.torch_log_iv import torch_log_iv

        return torch_log_iv
    raise ValueError(f"unknown log-Bessel backend: {name}")


def _as_torch(r_bar: np.ndarray, dtype: torch.dtype, device: str) -> torch.Tensor:
    return torch.as_tensor(r_bar, dtype=dtype, device=device)


@dataclass(frozen=True)
class Method:
    """One estimator under test."""

    name: str
    kind: str  # "closed_form", "sra", "converged", "reference"
    call: Estimator
    vectorized: bool = True


def build_methods(
    log_iv_backend: str,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    include_slow: bool = True,
) -> list[Method]:
    """Return the estimators to compare, in increasing order of cost."""

    torch_log_iv = _torch_log_iv_factory(log_iv_backend)
    torch_label = "CUSF" if log_iv_backend == "cusf" else "TorchBessel"

    methods = [
        Method("Banerjee", "closed_form", lambda r, p: banerjee_kappa(r, p)),
        Method("Sra two-step (SciPy)", "sra", lambda r, p: sra_kappa(r, p, scipy_log_iv)),
        Method(
            "Converged Newton (SciPy)",
            "converged",
            lambda r, p: converged_newton_kappa(r, p, scipy_log_iv).kappa,
        ),
        Method(
            "Banerjee (Torch)",
            "closed_form",
            lambda r, p: banerjee_kappa_torch(_as_torch(r, dtype, device), p)
            .double()
            .cpu()
            .numpy(),
        ),
        Method(
            f"Sra two-step ({torch_label})",
            "sra",
            lambda r, p: sra_kappa_torch(_as_torch(r, dtype, device), p, torch_log_iv)
            .double()
            .cpu()
            .numpy(),
        ),
        Method(
            f"Converged Newton ({torch_label})",
            "converged",
            lambda r, p: converged_newton_kappa_torch(
                _as_torch(r, dtype, device), p, torch_log_iv
            )
            .kappa.double()
            .cpu()
            .numpy(),
        ),
    ]
    if include_slow:
        methods += [
            Method("SciPy Brent", "reference", scipy_brentq_kappa, vectorized=False),
            Method(
                "Direct likelihood",
                "reference",
                lambda r, p: direct_log_likelihood_kappa(r, p, scipy_log_iv),
                vectorized=False,
            ),
        ]
    return methods


def _safe_call(method: Method, r_bar: np.ndarray, dimension: int) -> np.ndarray:
    """Evaluate a method, mapping failures to NaN rather than hiding them."""

    try:
        with np.errstate(all="ignore"):
            return np.asarray(method.call(r_bar, dimension), dtype=np.float64)
    except Exception:  # noqa: BLE001 - unsupported regimes are recorded, not raised
        return np.full(np.shape(r_bar), np.nan)


# --------------------------------------------------------------------------
# Study 1: numerical approximation on population r_bar
# --------------------------------------------------------------------------


def run_population_study(
    dimensions: Sequence[int],
    kappas: Sequence[float],
    methods: Sequence[Method],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dimension in dimensions:
        for kappa in kappas:
            r_bar = population_r_bar(dimension, kappa)
            if not np.isfinite(r_bar) or r_bar <= 0.0 or r_bar >= 1.0:
                continue
            batch = np.array([r_bar])
            reference = mpmath_mle_kappa(r_bar, dimension)
            for method in methods:
                estimate = float(_safe_call(method, batch, dimension)[0])
                rows.append(
                    {
                        "dimension": dimension,
                        "kappa_true": kappa,
                        "population_r_bar": r_bar,
                        "mpmath_mle": reference,
                        "method": method.name,
                        "kind": method.kind,
                        "kappa_hat": estimate,
                        "abs_error": abs(estimate - kappa),
                        "rel_error": abs(estimate - kappa) / kappa,
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Study 2: estimation from sampled data
# --------------------------------------------------------------------------


def draw_r_bar(
    dimension: int,
    kappa: float,
    num_samples: int,
    replicates: int,
    seed: int,
    backend: str,
) -> tuple[np.ndarray, float]:
    """Return ``replicates`` mean resultant lengths and the sampling seconds.

    Each replicate uses an independent random mean direction, so the study
    exercises the sampler's rotation path rather than only the ``e_1`` case.
    """

    generator = np.random.default_rng(seed)
    r_bar = np.empty(replicates, dtype=np.float64)
    start = time.perf_counter()
    if backend == "torch_hh":
        sampler = TorchvMFHH(
            dimension, kappa=kappa, seed=seed, device="cpu", dtype=torch.float64
        )
        for index in range(replicates):
            mu = torch.as_tensor(generator.normal(size=dimension), dtype=torch.float64)
            mu /= torch.linalg.norm(mu)
            samples = sampler.sample(num_samples, mu=mu)
            r_bar[index] = float(torch.linalg.norm(samples.mean(dim=0)))
    else:
        sampler_class = {"numpy_hh": NumpyvMFHH, "scipy": ScipyvMF}[backend]
        sampler = sampler_class(dimension, kappa=kappa, seed=seed)
        for index in range(replicates):
            mu = generator.normal(size=dimension)
            mu /= np.linalg.norm(mu)
            samples = np.asarray(sampler.sample(num_samples, mu=mu))
            r_bar[index] = float(np.linalg.norm(samples.mean(axis=0)))
    return r_bar, time.perf_counter() - start


def run_sampled_study(
    dimensions: Sequence[int],
    kappas: Sequence[float],
    methods: Sequence[Method],
    *,
    num_samples: int,
    replicates: int,
    seed: int,
    backend: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    sampling_rows: list[dict[str, object]] = []
    for dimension in dimensions:
        for kappa in kappas:
            r_bar, seconds = draw_r_bar(
                dimension, kappa, num_samples, replicates, seed + dimension, backend
            )
            sampling_rows.append(
                {
                    "dimension": dimension,
                    "kappa_true": kappa,
                    "backend": backend,
                    "num_samples": num_samples,
                    "replicates": replicates,
                    "seconds": seconds,
                    "samples_per_second": replicates * num_samples / seconds,
                }
            )
            usable = np.isfinite(r_bar) & (r_bar > 0.0) & (r_bar < 1.0)
            if not np.any(usable):
                continue
            reference = np.array(
                [mpmath_mle_kappa(float(value), dimension) for value in r_bar[usable]]
            )
            for method in methods:
                estimate = _safe_call(method, r_bar[usable], dimension)
                estimate = np.broadcast_to(estimate, reference.shape)
                for replicate, (value, mle, observed) in enumerate(
                    zip(r_bar[usable], reference, estimate)
                ):
                    rows.append(
                        {
                            "dimension": dimension,
                            "kappa_true": kappa,
                            "num_samples": num_samples,
                            "replicate": replicate,
                            "r_bar": float(value),
                            "mpmath_mle": float(mle),
                            "method": method.name,
                            "kind": method.kind,
                            "kappa_hat": float(observed),
                            # Numerical: how far the solver is from the MLE for
                            # this exact sample.
                            "numerical_rel_error": abs(observed - mle) / mle,
                            # Statistical: how far the MLE problem itself is
                            # from the generating kappa.
                            "statistical_rel_error": abs(observed - kappa) / kappa,
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(sampling_rows)


# --------------------------------------------------------------------------
# Study 3: throughput on precomputed r_bar
# --------------------------------------------------------------------------


def _median_seconds(
    function: Callable[[], object],
    *,
    repeats: int,
    synchronize: bool,
) -> float:
    timings = []
    for _ in range(repeats):
        if synchronize:
            torch.cuda.synchronize()
        start = time.perf_counter()
        function()
        if synchronize:
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def run_timing_study(
    dimension: int,
    kappa: float,
    methods: Sequence[Method],
    *,
    batch_sizes: Sequence[int],
    repeats: int,
    device: str,
) -> pd.DataFrame:
    r_bar_center = population_r_bar(dimension, kappa)
    rows: list[dict[str, object]] = []
    synchronize = device.startswith("cuda")
    for batch_size in batch_sizes:
        # Jitter around the population value so batches are not degenerate.
        generator = np.random.default_rng(0)
        r_bar = np.clip(
            r_bar_center * (1.0 + 0.02 * generator.standard_normal(batch_size)),
            1e-9,
            1.0 - 1e-12,
        )
        for method in methods:
            if not method.vectorized and batch_size > 4096:
                rows.append(
                    {
                        "dimension": dimension,
                        "kappa_true": kappa,
                        "device": device,
                        "method": method.name,
                        "kind": method.kind,
                        "batch_size": batch_size,
                        "median_seconds": np.nan,
                        "estimates_per_second": np.nan,
                        "note": "skipped: scalar loop too slow at this batch size",
                    }
                )
                continue
            _safe_call(method, r_bar[: min(batch_size, 8)], dimension)  # warm up
            try:
                seconds = _median_seconds(
                    lambda m=method: m.call(r_bar, dimension),
                    repeats=repeats,
                    synchronize=synchronize,
                )
            except Exception as error:  # noqa: BLE001
                rows.append(
                    {
                        "dimension": dimension,
                        "kappa_true": kappa,
                        "device": device,
                        "method": method.name,
                        "kind": method.kind,
                        "batch_size": batch_size,
                        "median_seconds": np.nan,
                        "estimates_per_second": np.nan,
                        "note": f"failed: {type(error).__name__}",
                    }
                )
                continue
            rows.append(
                {
                    "dimension": dimension,
                    "kappa_true": kappa,
                    "device": device,
                    "method": method.name,
                    "kind": method.kind,
                    "batch_size": batch_size,
                    "median_seconds": seconds,
                    "estimates_per_second": batch_size / seconds,
                    "note": "",
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Study 4: how much data kappa estimation actually needs
# --------------------------------------------------------------------------


def run_reliability_study(
    dimensions: Sequence[int],
    kappas: Sequence[float],
    sample_counts: Sequence[int],
    *,
    log_iv_backend: str,
    seed: int,
    max_samples_per_cell: int = 1_600_000,
    max_replicates: int = 16,
    min_replicates: int = 4,
) -> pd.DataFrame:
    """Map the statistical reliability of the vMF concentration MLE.

    Every accurate solver returns the same value, so the error measured here is
    a property of the estimation problem rather than of the numerical method.
    This is the regime information needed before reading a kappa fitted to
    real high-dimensional embeddings, and it is only cheap to map because
    sample generation is fast.
    """

    torch_log_iv = _torch_log_iv_factory(log_iv_backend)
    rows: list[dict[str, object]] = []
    for dimension in dimensions:
        for kappa in kappas:
            for num_samples in sample_counts:
                replicates = int(
                    min(
                        max_replicates,
                        max(min_replicates, max_samples_per_cell // num_samples),
                    )
                )
                r_bar, seconds = draw_r_bar(
                    dimension,
                    kappa,
                    num_samples,
                    replicates,
                    seed + dimension + num_samples,
                    "torch_hh",
                )
                usable = np.isfinite(r_bar) & (r_bar > 0.0) & (r_bar < 1.0)
                if not np.any(usable):
                    continue
                values = torch.as_tensor(r_bar[usable], dtype=torch.float64)
                estimate = (
                    converged_newton_kappa_torch(values, dimension, torch_log_iv)
                    .kappa.numpy()
                )
                closed_form = banerjee_kappa(r_bar[usable], dimension)
                relative = np.abs(estimate - kappa) / kappa
                rows.append(
                    {
                        "dimension": dimension,
                        "kappa_true": kappa,
                        "num_samples": num_samples,
                        "replicates": int(usable.sum()),
                        "seconds": seconds,
                        "median_r_bar": float(np.median(r_bar[usable])),
                        "median_kappa_hat": float(np.median(estimate)),
                        "median_rel_error": float(np.median(relative)),
                        "p90_rel_error": float(np.quantile(relative, 0.9)),
                        # Signed, to expose the upward finite-sample bias.
                        "median_signed_rel_bias": float(
                            np.median((estimate - kappa) / kappa)
                        ),
                        "banerjee_median_rel_error": float(
                            np.median(np.abs(closed_form - kappa) / kappa)
                        ),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Sampler cost for the study's own data generation
# --------------------------------------------------------------------------


def run_sampler_cost_study(
    dimensions: Sequence[int],
    *,
    kappa: float,
    num_samples: int,
    replicates: int,
    seed: int,
    backends: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dimension in dimensions:
        for backend in backends:
            _, seconds = draw_r_bar(
                dimension, kappa, num_samples, replicates, seed, backend
            )
            rows.append(
                {
                    "dimension": dimension,
                    "kappa_true": kappa,
                    "backend": backend,
                    "num_samples": num_samples,
                    "replicates": replicates,
                    "seconds": seconds,
                    "samples_per_second": replicates * num_samples / seconds,
                }
            )
    frame = pd.DataFrame(rows)
    baseline = frame[frame["backend"] == "scipy"].set_index("dimension")["seconds"]
    if not baseline.empty:
        frame["speedup_over_scipy"] = (
            frame["dimension"].map(baseline) / frame["seconds"]
        )
    return frame


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def environment_record(log_iv_backend: str, device: str) -> dict[str, object]:
    record = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "torch_threads": torch.get_num_threads(),
        "log_iv_backend": log_iv_backend,
        "device": device,
        "mpmath_dps": MPMATH_DPS,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        record["cuda_device"] = torch.cuda.get_device_name(0)
        record["cuda_version"] = torch.version.cuda
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        nargs="+",
        default=["population", "sampled", "timing", "sampler-cost"],
        choices=["population", "sampled", "timing", "sampler-cost", "reliability"],
    )
    parser.add_argument(
        "--reliability-dimensions", type=int, nargs="+", default=[64, 256, 1024, 4096]
    )
    parser.add_argument(
        "--reliability-kappas", type=float, nargs="+", default=[10.0, 50.0, 200.0, 1000.0]
    )
    parser.add_argument(
        "--reliability-sample-counts",
        type=int,
        nargs="+",
        default=[100, 1_000, 10_000, 100_000],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("measurements/kappa"))
    parser.add_argument("--tag", default="", help="Suffix appended to output file names.")
    parser.add_argument("--dimensions", type=int, nargs="+", default=list(DEFAULT_DIMENSIONS))
    parser.add_argument("--kappas", type=float, nargs="+", default=list(DEFAULT_KAPPAS))
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sampler",
        default="torch_hh",
        choices=["torch_hh", "numpy_hh", "scipy"],
        help="Backend used to generate the sampled study's datasets.",
    )
    parser.add_argument(
        "--log-iv",
        default="cusf",
        choices=["cusf", "torch"],
        help="Log-Bessel backend for the device estimators.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", default="float64", choices=["float32", "float64"]
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 64, 4096, 65_536, 1_048_576]
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--timing-dimension", type=int, default=256)
    parser.add_argument("--timing-kappa", type=float, default=50.0)
    # The sampler-cost study includes the SciPy reference, whose dense QR map
    # costs O(n d^2).  It is sized independently so a full-size sampled study
    # does not force a multi-minute SciPy run at dimension 4096.
    parser.add_argument("--cost-num-samples", type=int, default=2_000)
    parser.add_argument("--cost-replicates", type=int, default=3)
    parser.add_argument(
        "--quick", action="store_true", help="Small grid for smoke testing."
    )
    args = parser.parse_args()

    if args.quick:
        args.dimensions = [3, 64, 1024]
        args.kappas = [10.0, 200.0]
        args.num_samples = 2_000
        args.replicates = 3
        args.batch_sizes = [1, 1024, 65_536]
        args.repeats = 3
        args.cost_num_samples = 2_000
        args.cost_replicates = 2

    dtype = {"float32": torch.float32, "float64": torch.float64}[args.dtype]
    methods = build_methods(args.log_iv, device=args.device, dtype=dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    (args.output_dir / f"environment{suffix}.json").write_text(
        json.dumps(environment_record(args.log_iv, args.device), indent=2) + "\n"
    )

    if "population" in args.study:
        frame = run_population_study(args.dimensions, args.kappas, methods)
        path = args.output_dir / f"population{suffix}.csv"
        frame.to_csv(path, index=False)
        print(f"wrote {path} ({len(frame)} rows)")

    if "sampled" in args.study:
        frame, sampling = run_sampled_study(
            args.dimensions,
            args.kappas,
            methods,
            num_samples=args.num_samples,
            replicates=args.replicates,
            seed=args.seed,
            backend=args.sampler,
        )
        path = args.output_dir / f"sampled{suffix}.csv"
        frame.to_csv(path, index=False)
        sampling.to_csv(args.output_dir / f"sampled_generation{suffix}.csv", index=False)
        print(f"wrote {path} ({len(frame)} rows)")

    if "timing" in args.study:
        frame = run_timing_study(
            args.timing_dimension,
            args.timing_kappa,
            methods,
            batch_sizes=args.batch_sizes,
            repeats=args.repeats,
            device=args.device,
        )
        path = args.output_dir / f"timing{suffix}.csv"
        frame.to_csv(path, index=False)
        print(f"wrote {path} ({len(frame)} rows)")

    if "reliability" in args.study:
        frame = run_reliability_study(
            args.reliability_dimensions,
            args.reliability_kappas,
            args.reliability_sample_counts,
            log_iv_backend=args.log_iv,
            seed=args.seed,
        )
        path = args.output_dir / f"reliability{suffix}.csv"
        frame.to_csv(path, index=False)
        print(f"wrote {path} ({len(frame)} rows)")

    if "sampler-cost" in args.study:
        frame = run_sampler_cost_study(
            args.dimensions,
            kappa=args.timing_kappa,
            num_samples=args.cost_num_samples,
            replicates=args.cost_replicates,
            seed=args.seed,
            backends=("scipy", "numpy_hh", "torch_hh"),
        )
        path = args.output_dir / f"sampler_cost{suffix}.csv"
        frame.to_csv(path, index=False)
        print(f"wrote {path} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
