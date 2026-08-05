# vMF-sampling project memory

## Purpose and scientific invariants

This repository compares implementations of von Mises–Fisher sampling. Preserve
the following distinctions when changing code, benchmarks, or documentation:

- `src/vmf_scipy.py` is the historical reference. It constructs a dense map with
  `scipy.linalg.qr`, applies it with `numpy.einsum`, and intentionally retains
  `numpy.random.RandomState`. Do not modernize this path; it is the baseline.
- `src/vmf_numpy.py` keeps the QR geometry but uses NumPy operations such as
  `numpy.linalg.qr` and batched `matmul`, plus a per-instance `PCG64DXSM`
  generator.
- The Householder implementations use one reflection mapping `e1` to `mu`.
  This is valid because vMF is rotationally symmetric around `mu`. Construction
  and storage are O(d), and batch application is O(nd). Keep the in-place paths
  allocation-conscious.
- The PyTorch implementations must remain device-native and support CPU/CUDA
  with float16, bfloat16, float32, and float64 where numerically possible. Keep
  sample batches row-major for `mm`, normalize/scale owned tensors in place,
  and use fused `addmm` for the Householder rank-one update. Use a per-device
  generator; sampler construction must never reset global RNG state.
- Headline Householder speedups are end-to-end throughput ratios against the
  actual SciPy reference, not against the NumPy or PyTorch QR experiments.

Run RNG tests after sampler changes:

```bash
uv run python -m unittest discover -s tests -v
```

## Benchmark record

The current complete float64 CPU runs each contain 39 CSV files and 273 timing
rows (13 dimensions, three backend groups, three seeds, two-second target):

- Primary/public results: `measurements/runs/20260723T094321Z-tikgpu10` on the
  newer AMD EPYC 7742. At dimension 4096, NumPy and PyTorch in-place reflection
  reach 126.97x and 99.01x over SciPy, respectively.
- Hardware-generalization appendix: `measurements/runs/20260723T160232Z-arton10`
  on the older Intel Xeon E5-2690 v2. NumPy reaches 86.45x. This CPU lacks AVX2,
  so PyTorch selects its substantially slower generic CPU kernels.

These measurements agree with independent same-hardware results from the
neighboring `../AgenticOptimisation` repository. Do not replace or merge raw
CSVs silently. Validate complete runs before rendering.

## Kappa-estimation benchmark

Use `/skill:vmf-kappa` for concentration-parameter estimation work. The reference
paper is `papers/sra-paper.pdf` (Suvrit Sra, 2012). Preserve its attribution:
the closed-form estimate

```text
kappa_0 = Rbar * (p - Rbar^2) / (1 - Rbar^2)
```

is Banerjee et al. (2005), while Sra's method initializes with that value and
performs exactly two Newton updates for `A_p(kappa) - Rbar = 0`. Here
`A_p(kappa) = I_(p/2)(kappa) / I_(p/2-1)(kappa)`. With CUSF, evaluate the ratio
as the exponential of the difference of two log-Bessel values.

The five current comparison methods are:

1. Banerjee's closed-form approximation;
2. Sra's prescribed two Newton steps;
3. safeguarded Newton iterated to tolerance, using Sra's analytical bracket;
4. a CPU-only SciPy Brent root solver as an independent reference;
5. direct bounded maximization of the scalar log likelihood as another check.

Implementations live in `src/vmf_kappa.py` and `src/vmf_kappa_torch.py`.
`src/cusf_cpu.py` lazily compiles CUSF's original CPU `iv_log.cpp` from the
sibling `../cusf` checkout through `src/cusf_cpu_extension.cpp`; Ninja is a
project dependency. It supports float32 and float64. The Apple build currently
has no OpenMP. CUSF's regular extension supplies the future CUDA path.
`src/torch_log_iv.py` is a tested pure-PyTorch/MPS translation, but the current
benchmark scope explicitly omits MPS and native float16/bfloat16. Do not present
the PyTorch translation as compiled CUSF.

Benchmark estimators from the same precomputed mean resultant lengths. Keep
sample generation and the reduction to `Rbar` outside estimator timings. Report
numerical error against a sample-MLE reference separately from statistical
error against the generating kappa. For unknown mean direction, `Rbar` is the
norm of the sample mean; at fixed sample count and fixed kappa it becomes noisy
relative to its signal as dimension grows, producing genuine finite-sample
upward bias shared by all accurate MLE solvers.

Preliminary local float64 reference data for `kappa=50` are under
`measurements/kappa/`:

- `population_k50.csv` isolates numerical approximation using high-precision
  population `Rbar` values;
- `reference_k50_n10000.csv` contains 20 sampled datasets per dimension with
  10,000 observations each;
- `reference_cpu_timings.csv` contains batched Apple M3 Max CPU smoke timings.

These are reference numbers, not final cluster results. Important observed
behavior: Sra's two steps are already effectively converged on the population
`kappa=50` grid; SciPy's scaled-Bessel ratio becomes `0/0` at dimensions 1024
and 4096; direct likelihood optimization is slower and slightly less precise
because the objective is flat near its maximum.

Run all estimator and sampler tests after changes:

```bash
MAX_JOBS=4 uv run python -m unittest discover -s tests -v
```

## Blog conventions

The reader-facing post has moved to
`../aplesner.github.io/blog/2026-07-23-faster-vmf-sampling/index.html`; do not
restore a duplicate under `docs/`. `scripts/make_blog.py` regenerates the source
figures and tables under `docs/`, while the website repository owns the
published HTML. Keep both synchronized when benchmark claims change. Keep the
prose light and let figures and tables carry the result.

Reader-facing content should:

- identify hardware by CPU model, not internal cluster hostname;
- omit Slurm job IDs, run IDs, and cluster-specific reproduction details;
- begin with a “What changed” section showing the relevant SciPy QR and
  `einsum` lines, then present the optimization in two stages: first keep QR and
  use NumPy's direct `matmul` path, then remove QR with Householder;
- explain the SciPy, NumPy QR, Householder, and PyTorch implementations before
  presenting throughput;
- state that the displayed benchmark is CPU-only float64 while PyTorch also
  enables direct GPU and lower-precision generation;
- keep the newer AMD measurements in the main result and present the slower,
  older Intel measurements only as a hardware-generalization appendix;
- avoid a “peak dimension” callout, since it is normally the largest tested
  dimension.

Regenerate with:

```bash
uv run python scripts/make_blog.py \
  --run measurements/runs/20260723T094321Z-tikgpu10 \
  --comparison-run measurements/runs/20260723T160232Z-arton10
```

Then copy only the changed publication figures into the website post, update
its HTML values, run `python3 -m py_compile scripts/make_blog.py`, validate the
website with `make validate`, and run `git diff --check` in both repositories.

## Cluster workflow

Use `/skill:vmf-cluster` for the established CPU sampling sweep,
`/skill:vmf-kappa` for kappa-estimator work, and `/skill:vmf-blog` for post or
figure updates. Large environments and benchmark outputs belong under
`/itet-stor/aplesner/net_scratch`, never the small cluster home directory. The
sampling Slurm files deliberately request no GPU; create separate kappa
CPU/CUDA jobs rather than adding GPU resources to the canonical sampling sweep.
