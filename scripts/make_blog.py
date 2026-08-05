from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "numpy": "#4C78A8",
    "numpy_hh_inplace": "#F58518",
    "scipy": "#54A24B",
    "torch_cpu": "#B279A2",
    "torch_hh_inplace_cpu": "#E45756",
}
LABELS = {
    "numpy": "NumPy · QR",
    "numpy_hh": "NumPy · reflection",
    "numpy_hh_inplace": "NumPy · reflection, in-place",
    "scipy": "SciPy reference",
    "torch_cpu": "PyTorch · QR",
    "torch_hh_cpu": "PyTorch · reflection",
    "torch_hh_inplace_cpu": "PyTorch · reflection, in-place",
}
MAIN_BACKENDS = list(COLORS)


def _dtype_name(value: str) -> str:
    match = re.search(r"(float16|float32|float64|bfloat16)", value)
    return match.group(1) if match else value


def load_run(run_dir: Path) -> pd.DataFrame:
    paths = sorted(path for path in run_dir.glob("benchmark_*.csv") if path.name != "benchmark_all.csv")
    if not paths:
        raise FileNotFoundError(f"No benchmark CSVs found in {run_dir}")
    frames = [pd.read_csv(path) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    required = {"backend", "dtype", "dim", "seed", "time_s", "num_samples"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Benchmark data is missing columns: {sorted(missing)}")
    data["dtype_name"] = data["dtype"].map(_dtype_name)
    data["samples_per_s"] = data["num_samples"] / data["time_s"]
    return data


def validate_run(data: pd.DataFrame) -> None:
    keys = ["backend", "dtype", "dim", "seed"]
    if data.duplicated(keys).any():
        duplicates = data.loc[data.duplicated(keys, keep=False), keys]
        raise ValueError(f"Run contains duplicate timing rows:\n{duplicates.to_string(index=False)}")
    seed_counts = data.groupby(["backend", "dtype", "dim"])["seed"].nunique()
    incomplete = seed_counts[seed_counts != 3]
    if not incomplete.empty:
        raise ValueError(f"Expected three seeds per configuration:\n{incomplete.to_string()}")


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(["backend", "dtype_name", "dim"], as_index=False)
        .agg(
            throughput_mean=("samples_per_s", "mean"),
            throughput_std=("samples_per_s", "std"),
            seeds=("seed", "nunique"),
            num_samples=("num_samples", "median"),
        )
        .sort_values(["backend", "dtype_name", "dim"])
    )


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.18,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, assets: Path, name: str) -> None:
    fig.savefig(assets / f"{name}.svg", bbox_inches="tight")
    fig.savefig(assets / f"{name}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def plot_throughput(summary: pd.DataFrame, assets: Path) -> None:
    frame = summary[summary["dtype_name"] == "float64"]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for backend in MAIN_BACKENDS:
        part = frame[frame["backend"] == backend]
        if part.empty:
            continue
        x = part["dim"].to_numpy()
        y = part["throughput_mean"].to_numpy() / 1e6
        std = part["throughput_std"].fillna(0).to_numpy() / 1e6
        ax.plot(x, y, marker="o", ms=4.5, lw=2, color=COLORS[backend], label=LABELS[backend])
        # A noisy row can have std > mean; keep uncertainty bands positive
        # without letting a near-zero lower bound destroy the log-axis scale.
        lower = np.maximum(y - std, y * 0.1)
        ax.fill_between(x, lower, y + std, color=COLORS[backend], alpha=0.12)
    ax.set(xscale="log", yscale="log", xlabel="Dimension $d$", ylabel="Throughput (million samples/s)")
    ax.set_title("Sampling throughput across dimension")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, which="both", alpha=0.18)
    fig.tight_layout()
    _save(fig, assets, "throughput_float64")


def _speedup_frame(summary: pd.DataFrame, optimized: str, baseline: str) -> pd.DataFrame:
    frame = summary[summary["dtype_name"] == "float64"]
    opt = frame[frame["backend"] == optimized].set_index("dim")["throughput_mean"]
    base = frame[frame["backend"] == baseline].set_index("dim")["throughput_mean"]
    common = opt.index.intersection(base.index)
    return pd.DataFrame({"dim": common, "speedup": opt.loc[common] / base.loc[common]})


def plot_speedup(summary: pd.DataFrame, assets: Path) -> None:
    """Plot end-to-end throughput against the actual SciPy QR reference.

    The repository also has NumPy/PyTorch QR experiments, but those are not the
    original SciPy implementation and must not be used for the headline ratio.
    """
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    pairs = [
        ("numpy_hh_inplace", "NumPy · reflection, in-place", COLORS["numpy_hh_inplace"]),
        ("torch_hh_inplace_cpu", "PyTorch · reflection, in-place", COLORS["torch_hh_inplace_cpu"]),
    ]
    for optimized, label, color in pairs:
        frame = _speedup_frame(summary, optimized, "scipy")
        if frame.empty:
            continue
        ax.plot(frame["dim"], frame["speedup"], marker="o", ms=5, lw=2.2, color=color, label=label)
    ax.axhline(1, color="#555555", lw=1, ls="--")
    ax.set(xscale="log", yscale="log", xlabel="Dimension $d$", ylabel="Speedup over SciPy")
    ax.set_title("End-to-end speedup over SciPy's QR sampler")
    ax.legend()
    ax.grid(True, which="both", alpha=0.18)
    fig.tight_layout()
    _save(fig, assets, "speedup_over_scipy")


def plot_method_diagram(assets: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-0.15, 1.2)
        ax.set_ylim(-0.15, 1.15)
        ax.axis("off")
        circle = plt.Circle((0, 0), 1, fill=False, color="#D0D5DD", lw=1.5)
        ax.add_patch(circle)
        ax.arrow(0, 0, 1, 0, width=0.012, head_width=0.07, length_includes_head=True, color="#4C78A8")
        ax.arrow(0, 0, 0.62, 0.78, width=0.012, head_width=0.07, length_includes_head=True, color="#F58518")
        ax.text(1.03, -0.03, "$e_1$", color="#4C78A8", fontsize=12)
        ax.text(0.61, 0.86, "$\\mu$", color="#F58518", fontsize=12)
    axes[0].set_title("QR rotation")
    axes[0].text(0.5, -0.13, "$Qe_1=\\mu$\nstore $d^2$ values", ha="center", va="top")
    # In two dimensions the Householder hyperplane is the angle bisector.
    axes[1].plot([0, 0.9], [0, 0.45], color="#7A7A7A", ls="--", lw=1.5)
    axes[1].set_title("Householder reflection")
    axes[1].text(0.5, -0.13, "$H=I-2uu^T$\nstore $d$ values", ha="center", va="top")
    fig.suptitle("Both maps are valid: vMF is symmetric around its mean direction", y=1.02, fontsize=13)
    fig.tight_layout()
    _save(fig, assets, "rotation_methods")


def _fmt_rate(value: float) -> str:
    if pd.isna(value):
        return "—"
    if value >= 100e6:
        return f"{value / 1e6:.0f} M/s"
    if value >= 10e6:
        return f"{value / 1e6:.1f} M/s"
    if value >= 1e6:
        return f"{value / 1e6:.2f} M/s"
    if value >= 100e3:
        return f"{value / 1e3:.0f} k/s"
    if value >= 10e3:
        return f"{value / 1e3:.1f} k/s"
    if value >= 1e3:
        return f"{value / 1e3:.2f} k/s"
    return f"{value:.0f}/s"


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _cpu_model(run_dir: Path) -> str:
    records = sorted(run_dir.glob("environment_*.txt"))
    if not records:
        return "CPU model unavailable"
    match = re.search(
        r"^Model name:\s+(.+)$",
        records[0].read_text(errors="replace"),
        re.MULTILINE,
    )
    return match.group(1).strip() if match else "CPU model unavailable"


def _cpu_label(run_dir: Path) -> str:
    label = _cpu_model(run_dir)
    label = re.sub(r"\(R\)|\(TM\)", "", label)
    label = re.sub(r"\bCPU\b|\b64-Core Processor\b", "", label)
    label = re.sub(r"\s+@\s+[\d.]+GHz", "", label)
    return re.sub(r"\s+", " ", label).strip()


def make_hardware_comparison(
    primary: pd.DataFrame,
    comparison: pd.DataFrame,
    primary_run: Path,
    comparison_run: Path,
    assets: Path,
    tables_dir: Path,
) -> str:
    runs = [
        (_cpu_label(primary_run), primary, "#F58518"),
        (_cpu_label(comparison_run), comparison, "#4C78A8"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    rows: list[dict[str, str | int]] = []
    all_dims = sorted(set(primary["dim"]).intersection(comparison["dim"]))

    for label, summary, color in runs:
        hh = summary[
            (summary["dtype_name"] == "float64")
            & (summary["backend"] == "numpy_hh_inplace")
        ].set_index("dim")["throughput_mean"]
        speeds = _speedup_frame(summary, "numpy_hh_inplace", "scipy").set_index("dim")["speedup"]
        common = hh.index.intersection(speeds.index)
        axes[0].plot(common, hh.loc[common] / 1e6, marker="o", lw=2.2, color=color, label=label)
        axes[1].plot(common, speeds.loc[common], marker="o", lw=2.2, color=color, label=label)

    axes[0].set(xscale="log", yscale="log", xlabel="Dimension $d$", ylabel="Million samples/s", title="NumPy reflection throughput")
    axes[1].set(xscale="log", yscale="log", xlabel="Dimension $d$", ylabel="Speedup over SciPy", title="End-to-end speedup")
    axes[1].axhline(1, color="#555555", lw=1, ls="--")
    for ax in axes:
        ax.legend()
        ax.grid(True, which="both", alpha=0.18)
    fig.suptitle("CPU-only generalization: same code, 12 allocated CPUs", fontsize=14)
    fig.tight_layout()
    _save(fig, assets, "cpu_node_comparison")

    selected_dims = [dim for dim in (128, 512, 2048, 4096) if dim in all_dims]
    for dim in selected_dims:
        row: dict[str, str | int] = {"Dimension": dim}
        for label, summary, _ in runs:
            frame = summary[summary["dim"] == dim].set_index("backend")["throughput_mean"]
            row[f"{label} reflection"] = _fmt_rate(frame.get("numpy_hh_inplace", np.nan))
            ratio = frame.get("numpy_hh_inplace", np.nan) / frame.get("scipy", np.nan)
            row[f"{label} vs SciPy"] = f"{ratio:.2f}×"
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(tables_dir / "cpu_node_comparison.csv", index=False)
    return _markdown_table(table)


def make_tables(summary: pd.DataFrame, tables_dir: Path) -> tuple[str, str]:
    float64 = summary[summary["dtype_name"] == "float64"]
    available_dims = sorted(float64["dim"].unique())
    preferred = [3, 16, 128, 1024, 4096]
    selected_dims = [dim for dim in preferred if dim in available_dims]
    rows: list[dict[str, str | int]] = []
    for dim in selected_dims:
        row: dict[str, str | int] = {"Dimension": dim}
        for backend in MAIN_BACKENDS:
            values = float64[(float64["backend"] == backend) & (float64["dim"] == dim)]["throughput_mean"]
            row[LABELS[backend]] = _fmt_rate(values.iloc[0]) if len(values) else "—"
        rows.append(row)
    headline = pd.DataFrame(rows)
    headline.to_csv(tables_dir / "headline_throughput.csv", index=False)

    speed_rows = []
    for optimized, family in [
        ("numpy_hh_inplace", "NumPy reflection, in-place"),
        ("torch_hh_inplace_cpu", "PyTorch reflection, in-place"),
    ]:
        speeds = _speedup_frame(summary, optimized, "scipy")
        speeds = speeds[speeds["dim"] >= 4]
        if speeds.empty:
            continue
        maximum = speeds.loc[speeds["speedup"].idxmax()]
        speed_rows.append(
            {
                "Implementation": family,
                "Baseline": "SciPy QR",
                "Geometric mean": f"{np.exp(np.log(speeds['speedup']).mean()):.2f}×",
                "Maximum measured": f"{maximum['speedup']:.2f}×",
            }
        )
    speedups = pd.DataFrame(speed_rows)
    speedups.to_csv(tables_dir / "speedup_summary.csv", index=False)
    return _markdown_table(headline), _markdown_table(speedups)


def make_appendix_measurements(summary: pd.DataFrame, tables_dir: Path) -> str:
    float64 = summary[summary["dtype_name"] == "float64"]
    rows: list[dict[str, str | int]] = []
    for dim in sorted(float64["dim"].unique()):
        values = float64[float64["dim"] == dim].set_index("backend")["throughput_mean"]
        scipy_rate = values.get("scipy", np.nan)
        numpy_hh = values.get("numpy_hh_inplace", np.nan)
        torch_hh = values.get("torch_hh_inplace_cpu", np.nan)
        rows.append(
            {
                "Dimension": dim,
                "SciPy QR": _fmt_rate(scipy_rate),
                "NumPy QR": _fmt_rate(values.get("numpy", np.nan)),
                "NumPy reflection": _fmt_rate(numpy_hh),
                "NumPy vs SciPy": f"{numpy_hh / scipy_rate:.2f}×",
                "PyTorch QR": _fmt_rate(values.get("torch_cpu", np.nan)),
                "PyTorch reflection": _fmt_rate(torch_hh),
                "PyTorch vs SciPy": f"{torch_hh / scipy_rate:.2f}×",
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(tables_dir / "appendix_cpu_measurements.csv", index=False)
    return _markdown_table(table)


def _table_html(markdown: str) -> str:
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]
    if len(rows) < 2:
        return ""
    header = "".join(f"<th>{html.escape(cell)}</th>" for cell in rows[0])
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows[2:]
    )
    return f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


def write_blog(
    run_dir: Path,
    output: Path,
    headline: str,
    speedups: str,
    row_count: int,
    summary: pd.DataFrame,
    comparison_table: str = "",
    appendix_table: str = "",
    comparison_run: Path | None = None,
) -> None:
    primary_cpu = _cpu_label(run_dir)
    numpy_speedups = _speedup_frame(summary, "numpy_hh_inplace", "scipy")
    peak = numpy_speedups.loc[numpy_speedups["speedup"].idxmax()]
    headline_html = _table_html(headline)
    speedups_html = _table_html(speedups)
    comparison_html = ""
    if comparison_run is not None:
        secondary_cpu = _cpu_label(comparison_run)
        comparison_html = f"""
        <section class="appendix" id="appendix">
          <div class="section-heading"><span>A</span><h2>Appendix: generalization to an older CPU</h2></div>
          <p>The headline benchmark uses the newer {html.escape(primary_cpu)}. To test whether the conclusion generalizes beyond that processor, the same CPU-only float64 sweep was repeated on an older {html.escape(secondary_cpu)}. Both measurements use the same code, 12 allocated CPUs, three seeds, and no GPU allocation.</p>
          <p>The older processor does not support AVX2, so PyTorch selects its generic CPU kernel path and is substantially slower there. This appendix is a hardware-generalization check, not the primary performance result.</p>
          <figure class="wide"><img src="assets/cpu_node_comparison.svg" alt="CPU-only generalization from {html.escape(primary_cpu)} to {html.escape(secondary_cpu)}"></figure>
          {_table_html(comparison_table)}
          <h3>Full {html.escape(secondary_cpu)} measurements</h3>
          <p>Throughput is reported as samples per second. The two speedup columns use the SciPy QR implementation on the same CPU as their baseline.</p>
          {_table_html(appendix_table)}
        </section>"""
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="CPU benchmarks for faster von Mises–Fisher sampling with a Householder reflection.">
  <title>Faster von Mises–Fisher sampling with one reflection</title>
  <style>
    :root {{ --ink:#17202a; --muted:#5f6b76; --line:#dfe5ea; --paper:#fff; --soft:#f5f7f9; --blue:#4c78a8; --orange:#f58518; --red:#e45756; }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{ margin:0; color:var(--ink); background:var(--paper); font:17px/1.68 Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    header {{ padding:5.5rem max(1.5rem,calc((100vw - 1040px)/2)) 4.5rem; background:linear-gradient(135deg,#f7f9fb 0%,#fff8f1 100%); border-bottom:1px solid var(--line); }}
    .eyebrow {{ color:var(--orange); font-size:.78rem; font-weight:750; letter-spacing:.14em; text-transform:uppercase; }}
    h1 {{ max-width:900px; margin:.65rem 0 1rem; font-size:clamp(2.5rem,6vw,5.1rem); line-height:1.01; letter-spacing:-.055em; }}
    .dek {{ max-width:720px; color:var(--muted); font-size:1.2rem; }}
    .chips {{ display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.7rem; }}
    .chip {{ padding:.35rem .7rem; border:1px solid #d8dee4; border-radius:999px; background:#fff; color:#49545f; font-size:.82rem; }}
    main {{ max-width:1040px; margin:auto; padding:2.5rem 1.5rem 6rem; }}
    section {{ padding:3.2rem 0; border-bottom:1px solid var(--line); }}
    .section-heading {{ display:flex; align-items:baseline; gap:.8rem; margin-bottom:1rem; }}
    .section-heading span {{ color:var(--orange); font:700 .78rem/1 ui-monospace,SFMono-Regular,monospace; }}
    h2 {{ margin:0; font-size:clamp(1.7rem,3vw,2.5rem); line-height:1.15; letter-spacing:-.03em; }}
    p {{ max-width:780px; }}
    code {{ padding:.12rem .34rem; border-radius:4px; background:#f0f3f5; font:85% ui-monospace,SFMono-Regular,Menlo,monospace; }}
    pre {{ overflow:auto; padding:1.2rem 1.35rem; border-radius:9px; color:#ecf2f8; background:#17212b; line-height:1.55; }}
    pre code {{ padding:0; color:inherit; background:none; }}
    figure {{ margin:2rem 0; }}
    figure img {{ display:block; width:100%; height:auto; }}
    figcaption {{ margin-top:.65rem; color:var(--muted); font-size:.88rem; text-align:center; }}
    .equation {{ max-width:780px; margin:1.7rem 0; padding:1.1rem; border-left:4px solid var(--orange); background:var(--soft); font:1.15rem Georgia,serif; text-align:center; }}
    .stats {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1rem; max-width:680px; margin:2rem 0; }}
    .stat {{ padding:1.25rem; border:1px solid var(--line); border-radius:10px; background:var(--soft); }}
    .stat strong {{ display:block; color:var(--orange); font-size:1.8rem; line-height:1.1; }}
    .stat small {{ color:var(--muted); }}
    .table-wrap {{ overflow-x:auto; margin:1.6rem 0 2.2rem; border:1px solid var(--line); border-radius:9px; }}
    table {{ width:100%; border-collapse:collapse; font-size:.88rem; line-height:1.4; }}
    th,td {{ padding:.75rem .85rem; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
    th:first-child,td:first-child {{ text-align:left; }}
    th {{ color:#3b4650; background:var(--soft); font-size:.76rem; letter-spacing:.035em; text-transform:uppercase; }}
    tbody tr:last-child td {{ border-bottom:0; }}
    tbody tr:hover {{ background:#fffaf5; }}
    .code-card {{ margin:1.7rem 0; overflow:hidden; border:1px solid #2c3945; border-radius:11px; background:#17212b; box-shadow:0 12px 30px rgba(23,32,42,.1); }}
    .code-card header {{ display:flex; justify-content:space-between; align-items:center; padding:.7rem 1rem; border:0; background:#222e39; color:#cbd5df; font-size:.77rem; }}
    .code-card header span:last-child {{ color:#8997a5; font-family:ui-monospace,SFMono-Regular,monospace; }}
    .code-card pre {{ margin:0; border-radius:0; }}
    .code-hl {{ display:block; margin:0 -.7rem; padding:0 .7rem; border-left:3px solid; }}
    .code-hl.qr {{ border-color:var(--orange); background:rgba(245,133,24,.17); }}
    .code-hl.einsum {{ border-color:#60a5fa; background:rgba(96,165,250,.16); }}
    .code-legend {{ display:flex; flex-wrap:wrap; gap:1rem; margin:.9rem 0 2rem; color:var(--muted); font-size:.82rem; }}
    .code-legend i {{ display:inline-block; width:.75rem; height:.75rem; margin-right:.35rem; border-radius:2px; vertical-align:-.05rem; }}
    .change-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin:1.5rem 0; }}
    .change {{ padding:1.3rem; border:1px solid var(--line); border-radius:10px; background:var(--soft); }}
    .change .step {{ color:var(--orange); font:700 .72rem/1 ui-monospace,SFMono-Regular,monospace; letter-spacing:.08em; text-transform:uppercase; }}
    .change h3 {{ margin:.45rem 0; font-size:1.15rem; }}
    .change pre {{ margin:1rem 0 0; font-size:.78rem; }}
    .method-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin:1.5rem 0; }}
    .method {{ padding:1.25rem; border-radius:9px; border:1px solid var(--line); border-top:4px solid var(--blue); }}
    .method.updated {{ border-color:#f4b578; border-top-color:var(--orange); background:#fff9f3; }}
    .method.torch {{ border-top-color:var(--red); }}
    .method h3 {{ margin:.1rem 0 .5rem; font-size:1.08rem; }}
    .method p {{ margin:.45rem 0; color:#414c56; font-size:.94rem; }}
    .method .tag {{ color:var(--muted); font:700 .7rem/1 ui-monospace,SFMono-Regular,monospace; letter-spacing:.06em; text-transform:uppercase; }}
    .details {{ display:grid; grid-template-columns:1fr 1fr; gap:2rem; }}
    .details ul {{ margin:0; padding-left:1.2rem; }}
    .details li {{ margin:.4rem 0; }}
    .appendix {{ margin-top:2.5rem; padding:3rem 2rem; border:1px solid var(--line); border-radius:12px; background:#fafbfc; }}
    .appendix h3 {{ margin:2.5rem 0 .5rem; font-size:1.25rem; }}
    footer {{ padding:2rem 1.5rem; color:var(--muted); background:var(--soft); text-align:center; font-size:.85rem; }}
    a {{ color:var(--blue); }}
    @media (max-width:760px) {{ header {{ padding-top:3.5rem; }} .stats,.change-grid,.method-grid,.details {{ grid-template-columns:1fr; }} body {{ font-size:16px; }} .code-card pre {{ font-size:.72rem; }} }}
    @media print {{ header {{ padding:2rem 0; }} main {{ padding:1rem 0; }} section {{ break-inside:avoid; }} }}
  </style>
</head>
<body>
<header>
  <div class="eyebrow">Directional statistics · benchmark note</div>
  <h1>Faster von Mises–Fisher sampling with one reflection</h1>
  <p class="dek">Replacing a dense QR rotation with a single Householder reflection makes high-dimensional sampling dramatically cheaper.</p>
  <div class="chips"><span class="chip">float64</span><span class="chip">κ = 50</span><span class="chip">dimensions 2–4096</span><span class="chip">CPU only</span></div>
</header>
<main>
  <section id="highlights">
    <div class="section-heading"><span>01</span><h2>What changed</h2></div>
    <p>The original SciPy implementation does two expensive things in the rotation path: it builds a dense orthogonal matrix with QR, then uses the general-purpose <code>einsum</code> contraction machinery to apply that matrix to every sample.</p>
    <div class="code-card">
      <header><strong>Original SciPy rotation path</strong><span>src/vmf_scipy.py</span></header>
      <pre><code>embedded = np.concatenate([
    self.mu[None, :],
    np.zeros((self.dim - 1, self.dim), dtype=self.mu.dtype),
])
<span class="code-hl qr">self.rotmatrix, _ = la.qr(np.transpose(embedded))</span>

<span class="code-hl einsum">samples = np.einsum(&quot;ij,...j-&gt;...i&quot;, self.rotmatrix, samples) * self.rotsign</span></code></pre>
    </div>
    <div class="code-legend"><span><i style="background:#f58518"></i>dense QR construction</span><span><i style="background:#60a5fa"></i>general tensor contraction</span></div>
    <p><code>einsum</code> is flexible, but it is not the best primitive for this repeated dense matrix–vector operation. A direct matrix multiplication expresses the workload more clearly and can dispatch to optimized linear-algebra kernels without the general contraction path.</p>
    <div class="change-grid">
      <article class="change">
        <div class="step">Change 1 · keep QR</div>
        <h3>Use the vectorized NumPy path</h3>
        <p>The NumPy QR control keeps the same dense map, but uses <code>numpy.linalg.qr</code> and batched <code>matmul</code>. This alone shows that a substantial speedup is available before changing the mathematics.</p>
        <pre><code>rotmatrix, _ = np.linalg.qr(embedded.T)
samples = (rotmatrix @ samples.T).T * rotsign</code></pre>
      </article>
      <article class="change">
        <div class="step">Change 2 · remove QR</div>
        <h3>Apply one Householder reflection</h3>
        <p>Because vMF is rotationally symmetric, we can drop the dense rotation entirely and store only the reflection vector <em>u</em>.</p>
        <pre><code>u = (e1 - mu) / np.linalg.norm(e1 - mu)
samples -= 2 * np.outer(samples @ u, u)</code></pre>
      </article>
    </div>
  </section>

  <section id="idea">
    <div class="section-heading"><span>02</span><h2>Why one reflection is enough</h2></div>
    <p>Sampling from a von Mises–Fisher distribution has two parts: draw around the first coordinate axis, then map that axis onto the requested mean direction μ. Constructing and applying a dense QR rotation becomes the dominant cost in high dimension.</p>
    <figure><img src="assets/rotation_methods.svg" alt="QR rotation and Householder reflection"><figcaption>Any orthogonal map sending e₁ to μ is valid because vMF is rotationally symmetric around μ.</figcaption></figure>
    <p>A single reflection is enough:</p>
    <div class="equation">H = I − 2uu<sup>T</sup>, &nbsp; u = (e₁ − μ) / ‖e₁ − μ‖</div>
    <p>It replaces a dense <em>d × d</em> matrix with one length-<em>d</em> vector and can be applied to the complete sample batch in place.</p>
  </section>

  <section id="implementations">
    <div class="section-heading"><span>03</span><h2>Implementations compared</h2></div>
    <p>The benchmark separates two choices: the array library used to execute the sampler and the orthogonal map used to move samples from e₁ to μ. This gives us QR and reflection implementations in both NumPy and PyTorch, alongside the original SciPy behavior.</p>
    <div class="method-grid">
      <div class="method">
        <div class="tag">Reference baseline</div>
        <h3>SciPy QR</h3>
        <p>This path deliberately stays close to the original implementation. It constructs the dense rotation with <code>scipy.linalg.qr</code> and applies that matrix to a batch with <code>numpy.einsum</code>. It also retains the legacy <code>RandomState</code> stream so the reference is not silently modernized.</p>
      </div>
      <div class="method">
        <div class="tag">Same geometry, modern NumPy</div>
        <h3>NumPy QR</h3>
        <p>The sampling algorithm and dense QR map are unchanged, but the numerical operations use NumPy directly: <code>numpy.linalg.qr</code>, batched <code>matmul</code>, and a per-sampler <code>Generator</code>. In particular, expressing the batch rotation as matrix multiplication lets NumPy use its optimized vectorized linear-algebra path.</p>
      </div>
      <div class="method updated">
        <div class="tag">Proposed map</div>
        <h3>Householder reflection</h3>
        <p>The NumPy and PyTorch reflection variants remove QR entirely. They apply <em>s ← s − 2(s·u)u</em> with a matrix–vector product and outer product. The in-place variants reuse the sample output and avoid a second output-sized allocation.</p>
      </div>
      <div class="method torch">
        <div class="tag">Device-native sampling</div>
        <h3>PyTorch QR and reflection</h3>
        <p>The PyTorch versions mirror both NumPy maps using native tensor operations. They keep sample batches row-major for matrix multiplication, normalize and scale freshly generated tensors in place, and fuse the Householder rank-one update with <code>addmm</code>. Samples stay on the selected CPU or GPU without a NumPy round trip. The implementation supports <code>float16</code>, <code>bfloat16</code>, <code>float32</code>, and <code>float64</code>.</p>
      </div>
    </div>
    <p>This post uses CPU float64 measurements so all implementations can be compared on equal footing. The PyTorch paths are included because real workloads often need samples directly on an accelerator or in a lower precision.</p>
  </section>

  <section id="throughput">
    <div class="section-heading"><span>04</span><h2>Throughput</h2></div>
    <figure class="wide"><img src="assets/throughput_float64.svg" alt="Float64 sampling throughput by dimension"></figure>
    <p>Points are means over three seeded runs; bands show one standard deviation. Dimensions 2 and 3 use dedicated closed-form samplers.</p>
    {headline_html}
  </section>

  <section id="speedup">
    <div class="section-heading"><span>05</span><h2>Speedup over SciPy</h2></div>
    <div class="stats"><div class="stat"><strong>{peak['speedup']:.2f}×</strong><small>maximum measured NumPy speedup</small></div><div class="stat"><strong>{html.escape(primary_cpu)}</strong><small>primary CPU</small></div></div>
    <figure class="wide"><img src="assets/speedup_over_scipy.svg" alt="End-to-end speedup over SciPy's QR sampler"></figure>
    {speedups_html}
    <p>These are end-to-end throughput ratios against the actual SciPy reference—not against the repository's separate NumPy or PyTorch QR experiments. The precise ratio is hardware-dependent because dense linear algebra and random-number generation scale differently across CPUs.</p>
  </section>

  <section id="complexity">
    <div class="section-heading"><span>06</span><h2>Complexity and random state</h2></div>
    <div class="table-wrap"><table><thead><tr><th>Area</th><th>Reflection path</th><th>QR reference</th></tr></thead><tbody><tr><td>Map e₁ → μ</td><td>One Householder reflection</td><td>Dense QR rotation</td></tr><tr><td>Map construction</td><td>O(d)</td><td>O(d³)</td></tr><tr><td>Rotation storage</td><td>O(d)</td><td>O(d²)</td></tr><tr><td>Batch application</td><td>O(nd)</td><td>O(nd²)</td></tr><tr><td>NumPy RNG</td><td>Per-instance PCG64DXSM</td><td>Legacy RandomState in SciPy</td></tr><tr><td>PyTorch RNG</td><td>Per-device generator</td><td>Per-device generator</td></tr></tbody></table></div>
  </section>

  <section id="details">
    <div class="section-heading"><span>07</span><h2>Benchmark details</h2></div>
    <div class="details"><ul><li><strong>CPU</strong> {html.escape(primary_cpu)}</li><li><strong>Workload</strong> float64, κ = 50, dimensions 2–4096</li><li><strong>Measurements</strong> {row_count} raw timing rows</li></ul><ul><li><strong>Timing</strong> calibrated batches with a 2 s window</li><li><strong>Repetitions</strong> three seeded runs</li><li><strong>Allocation</strong> 12 CPUs, one array task at a time</li><li><strong>GPU allocation</strong> none</li></ul></div>
    <p>Machine-readable summaries are available under <a href="tables/">docs/tables/</a>.</p>
  </section>

  {comparison_html}
</main>
<footer>Generated from reproducible Slurm measurements</footer>
</body>
</html>
"""
    output.write_text(document)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the vMF benchmark blog post.")
    parser.add_argument("--run", required=True, type=Path, help="Fetched run directory.")
    parser.add_argument(
        "--comparison-run",
        type=Path,
        help="Optional second complete run used for a CPU-node comparison panel.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional standalone HTML preview. The published post lives in the website repository.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Render an incomplete run (useful only while debugging).",
    )
    args = parser.parse_args()

    csv_count = len(
        [path for path in args.run.glob("benchmark_*.csv") if path.name != "benchmark_all.csv"]
    )
    if csv_count != 39 and not args.allow_partial:
        raise ValueError(f"Expected 39 benchmark CSVs, found {csv_count}; use --allow-partial to override")

    data = load_run(args.run)
    if not args.allow_partial:
        validate_run(data)
    summary = summarize(data)
    docs = Path("docs")
    assets = docs / "assets"
    tables = docs / "tables"
    assets.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    _style()
    plot_method_diagram(assets)
    plot_throughput(summary, assets)
    plot_speedup(summary, assets)
    headline, speedups = make_tables(summary, tables)
    summary.to_csv(tables / "benchmark_summary.csv", index=False)

    comparison_table = ""
    appendix_table = ""
    if args.comparison_run is not None:
        comparison_csvs = [
            path
            for path in args.comparison_run.glob("benchmark_*.csv")
            if path.name != "benchmark_all.csv"
        ]
        if len(comparison_csvs) != 39 and not args.allow_partial:
            raise ValueError(
                f"Expected 39 comparison CSVs, found {len(comparison_csvs)}; "
                "use --allow-partial to override"
            )
        comparison_data = load_run(args.comparison_run)
        if not args.allow_partial:
            validate_run(comparison_data)
        comparison_summary = summarize(comparison_data)
        comparison_table = make_hardware_comparison(
            summary,
            comparison_summary,
            args.run,
            args.comparison_run,
            assets,
            tables,
        )
        appendix_table = make_appendix_measurements(comparison_summary, tables)

    if args.output is not None:
        write_blog(
            args.run,
            args.output,
            headline,
            speedups,
            len(data),
            summary,
            comparison_table,
            appendix_table,
            args.comparison_run,
        )
    preview = f", plus {args.output}" if args.output is not None else ""
    print(
        f"Wrote {len(list(assets.glob('*')))} figures and "
        f"{len(list(tables.glob('*.csv')))} CSV tables{preview}"
    )


if __name__ == "__main__":
    main()
