---
name: vmf-blog
description: Maintains and regenerates the vMF benchmark HTML blog, figures, and CSV tables from complete benchmark runs. Use when editing the post, changing benchmark presentation, updating implementation explanations, or comparing CPU measurements.
compatibility: Run from the vMF-sampling repository with uv and the fetched measurement directories available.
---

# vMF benchmark blog workflow

The reader-facing post has moved to
`../aplesner.github.io/blog/2026-07-23-faster-vmf-sampling/index.html`. Do not
recreate a duplicate post under `docs/`. This repository generates benchmark
figures and tables; the website repository owns the published HTML.

## Canonical data

Use these complete float64 runs unless the user explicitly selects newer data:

- Primary: `measurements/runs/20260723T094321Z-tikgpu10`
  (newer AMD EPYC 7742)
- Generalization appendix: `measurements/runs/20260723T160232Z-arton10`
  (older Intel Xeon E5-2690 v2, generic non-AVX2 PyTorch path)

Each complete run has 39 `benchmark_*.csv` files and 273 timing rows. Never mix
files from different run directories or render a partial run as final data.

## Render

```bash
uv run python scripts/make_blog.py \
  --run measurements/runs/20260723T094321Z-tikgpu10 \
  --comparison-run measurements/runs/20260723T160232Z-arton10
```

Outputs are source SVG/high-resolution PNG figures in `docs/assets/` and
machine-readable summaries in `docs/tables/`. Copy only the four figures used
by the article into the website post's `figures/` directory and update the
website HTML when values or prose change.

## Editorial contract

Keep prose concise and blog-like. Preserve these choices:

1. Open with a “What changed” section. Show the actual SciPy QR and `einsum`
   lines with separate visual highlights. Explain the optimization as two
   steps: keep QR but replace the general `einsum` contraction with NumPy's
   direct batched `matmul` path, then remove QR and use one Householder
   reflection. Do not claim every NumPy-QR gain is caused by `einsum` alone,
   because that control also changes the QR and RNG APIs.
2. Introduce the Householder argument before results: vMF symmetry means an
   orthogonal reflection mapping `e1` to `mu` is sufficient.
3. Explain implementations before throughput:
   - SciPy reference: `scipy.linalg.qr`, batch rotation with `numpy.einsum`,
     legacy `RandomState` retained intentionally.
   - NumPy QR: same geometry, but `numpy.linalg.qr`, batched `matmul`, and a
     per-instance `PCG64DXSM` generator.
   - Reflection: matrix-vector plus outer-product update, including an in-place
     path.
   - PyTorch: native tensor implementations for direct CPU/GPU generation and
     float16, bfloat16, float32, or float64 output.
4. The published plots are CPU-only float64 measurements so all libraries are
   comparable. Do not imply that these runs benchmark GPU performance.
5. Speedups use the actual SciPy implementation as denominator. Never relabel a
   NumPy/PyTorch QR denominator as SciPy.
6. Use CPU model names in reader-facing text and legends. Do not expose cluster
   hostnames, run IDs, Slurm job IDs, scratch paths, or internal reproduction
   placeholders in the HTML.
7. Keep the newer AMD EPYC results in the main sections. Present the slower
   Intel Xeon run only as a hardware-generalization appendix with its full
   measurement table and note that its generic PyTorch path lacks AVX2.
8. Do not call out “peak dimension”; it will generally be the highest tested
   dimension. A maximum measured speedup is acceptable without a peak-dimension
   column or card.

Current reference points at dimension 4096:

- AMD EPYC 7742: NumPy reflection 12.3k samples/s (126.97x over SciPy) and
  PyTorch reflection 9.57k samples/s (99.01x).
- Intel Xeon E5-2690 v2: NumPy reflection 7.86k samples/s and SciPy 91
  samples/s, an 86.45x speedup.

## Validate

```bash
python3 -m py_compile scripts/make_blog.py
uv run python -m unittest discover -s tests -v
git diff --check
```

Parse the website post HTML, verify every referenced asset exists, and search
it for accidental `tikgpu10`, `arton10`, run-ID, or job-ID leakage. Run
`make validate` in `../aplesner.github.io`. Inspect changed figures visually
whenever plotting code changes.
