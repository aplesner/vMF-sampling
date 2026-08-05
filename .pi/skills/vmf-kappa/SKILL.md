---
name: vmf-kappa
description: Implements, validates, and benchmarks von Mises-Fisher concentration-parameter estimators using CUSF log-Bessel functions, Sra's two-step Newton approximation, SciPy references, and direct likelihood optimization. Use for kappa-estimation math, CPU/CUDA benchmarking, sampled-data accuracy studies, or CUSF integration.
compatibility: Run from the vMF-sampling repository with uv; compiled CUSF CPU work requires Ninja and the sibling ../cusf checkout, while CUDA measurements require a built CUSF CUDA extension and a GPU node.
---

# vMF kappa-estimation workflow

## Mathematical contract

Read `papers/sra-paper.pdf` when changing the method definitions. Use `p` for
ambient dimension and

```text
Rbar = norm(mean(samples, axis=sample_axis))
nu = p / 2 - 1
A_p(kappa) = I_(nu+1)(kappa) / I_nu(kappa)
```

The joint vMF MLE estimates the unknown mean direction from the sample
resultant and solves `A_p(kappa) = Rbar` for the concentration.

Keep the attribution precise:

- **Banerjee et al. (2005), Sra Eq. 4** is the closed form
  `Rbar * (p - Rbar^2) / (1 - Rbar^2)`.
- **Sra (2012), Eq. 6** starts from Banerjee's value and performs exactly two
  Newton steps. Do not call the closed form alone “the Sra estimator.”
- The Newton derivative is
  `1 - A_p(kappa)^2 - ((p - 1) / kappa) * A_p(kappa)`.
- With log-Bessel functions, compute the ratio as
  `exp(log_I(nu + 1, kappa) - log_I(nu, kappa))`.

## Methods compared

Preserve these five distinct methods:

1. **Banerjee**: closed form, no Bessel calls.
2. **Sra two-step**: Banerjee initialization and exactly two Newton updates.
3. **CUSF converged Newton**: iterate the score equation to tolerance. Use
   Sra's lower/upper bounds and replace invalid or out-of-bracket Newton
   proposals with bisection.
4. **SciPy Brent reference**: CPU-only root solve using SciPy's scaled Bessel
   ratio. This is an independent reference, not the expected throughput
   winner. Record unsupported cases instead of hiding them.
5. **Direct likelihood**: independently maximize
   `nu*log(kappa) - log_I(nu, kappa) + kappa*Rbar`. This validates the score
   solution but is expected to be slower and can be less precise because the
   likelihood is flat near its maximum.

Source locations:

- NumPy/SciPy estimators: `src/vmf_kappa.py`
- Batched device estimators: `src/vmf_kappa_torch.py`
- Compiled CPU CUSF adapter: `src/cusf_cpu.py` and
  `src/cusf_cpu_extension.cpp`
- Pure PyTorch/MPS log-Bessel port: `src/torch_log_iv.py`

## Current precision and device scope

The current benchmark scope is CPU and CUDA with float32 and float64. Ignore
native float16, bfloat16, and MPS unless the user reopens those comparisons.

CUSF currently exposes only float and double. Native 16-bit arithmetic is also
poorly suited to subtracting two large, nearby log-Bessel values and then
forming the cancellation-sensitive Newton derivative. The MPS translation is
tested and may remain in the repository, but it is not compiled CUSF and must
not appear in headline comparisons.

## CUSF CPU build

The sibling `../cusf` Makefile requires Linux and CUDA even though CUSF contains
CPU implementations. `src/cusf_cpu.py` therefore uses PyTorch's extension
loader to compile CUSF's original `cusf/cusf/src/bessel/iv_log.cpp` directly.
The build is cached by PyTorch.

Smoke-test the extension:

```bash
MAX_JOBS=4 uv run python - <<'PY'
import torch
from src.cusf_cpu import iv_log
v = torch.tensor([0., 1., 10., 100.], dtype=torch.float64)
x = torch.tensor([0.1, 1., 10., 100.], dtype=torch.float64)
print(iv_log(v, x))
PY
```

The local Apple Clang build currently ignores CUSF's OpenMP pragma because
OpenMP is not installed. Record this in Mac CPU results. A Linux benchmark
build should enable and report OpenMP explicitly rather than silently comparing
different thread configurations.

For CUDA, use CUSF's compiled PyTorch extension on a GPU node. Do not add GPU
resources to the canonical CPU-only sampling Slurm files; make separate kappa
benchmark scripts and jobs.

## Benchmark design

Separate three questions:

### 1. Numerical approximation

Set `Rbar` from a high-precision population ratio for a known `(p, kappa)`.
Every exact solver should recover the generating kappa. This isolates numerical
method error from sample noise and reproduces the style of Sra's Table 2.

### 2. Estimation from samples

Generate independent vMF datasets, reduce each once to `Rbar`, and feed the
same values to every method and device. Report both:

- deviation from a high-accuracy MLE for the same sample;
- deviation from the generating kappa.

Do not interpret the second as purely numerical error. With unknown mean
direction, finite-sample noise inflates the norm of the sample mean. At fixed
`kappa` and sample count this can cause large upward bias as `p` grows.

### 3. Performance

Exclude sample generation and `Rbar` reduction from estimator timing. Measure:

- scalar latency for a single dataset;
- batched throughput for many independent datasets;
- log-Bessel evaluations or Newton iterations where useful;
- convergence/failure counts, not only successful timing rows.

GPU throughput is meaningful only for sufficiently large batches. Synchronize
CUDA before and after timed regions. Use identical precomputed inputs for CPU
and GPU. Record dtype, batch size, thread count, CPU/GPU model, CUSF commit, and
software versions.

## Existing preliminary references

The local float64 `kappa=50` smoke data are:

- `measurements/kappa/population_k50.csv`
- `measurements/kappa/reference_k50_n10000.csv`
- `measurements/kappa/reference_cpu_timings.csv`

The sampled file uses 20 seeds and 10,000 samples per dimension. It shows real
finite-sample upward bias at high dimension; it is not evidence that the
accurate solvers disagree. The timing file is from an Apple M3 Max CPU without
OpenMP and is not a final cluster benchmark.

Known numerical behavior at `kappa=50`:

- Sra's two updates are effectively converged on dimensions 3--4096 when using
  population `Rbar` values.
- SciPy's `ive` numerator and denominator both underflow at dimensions 1024 and
  4096, so its Brent reference is unavailable there.
- Direct likelihood optimization remains close to the MLE but loses a few more
  digits and is much slower than score-based Newton.

## Validation

Run:

```bash
MAX_JOBS=4 uv run python -m unittest discover -s tests -v
python3 -m py_compile \
  src/cusf_cpu.py src/torch_log_iv.py src/vmf_kappa.py src/vmf_kappa_torch.py
git diff --check
```

Relevant tests are `tests/test_cusf_cpu.py`, `tests/test_kappa.py`, and
`tests/test_kappa_torch.py`. They check selected Sra Table 2 values, agreement
with SciPy where SciPy is finite, compiled CUSF integration, and device code.
