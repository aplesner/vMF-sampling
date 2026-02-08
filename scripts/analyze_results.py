from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import pandas as pd


def _load_csvs(pattern: str) -> pd.DataFrame:
    paths = sorted(Path().glob(pattern))
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def _load_dbs(pattern: str, table: str) -> pd.DataFrame:
    paths = sorted(Path().glob(pattern))
    if not paths:
        return pd.DataFrame()
    frames = []
    for path in paths:
        conn = sqlite3.connect(path)
        try:
            frames.append(pd.read_sql_query(f"SELECT * FROM {table}", conn))
        finally:
            conn.close()
    return pd.concat(frames, ignore_index=True)


def _summarize_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["backend", "dtype", "dim", "kappa"])
    time_col = "time_s" if "time_s" in df.columns else "time_s_mean"
    summary = grouped.agg(
        time_s_mean=(time_col, "mean"),
        time_s_std=(time_col, "std"),
        n=("seed", "nunique"),
    )
    return summary.reset_index()


def _summarize_twist(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["method", "dtype", "dim", "mode"])
    summary = grouped.agg(
        twist_deg_mean=("twist_deg", "mean"),
        twist_deg_std=("twist_deg", "std"),
        probe_std_deg_mean=("probe_std_deg", "mean"),
        probe_std_deg_std=("probe_std_deg", "std"),
        n=("seed", "nunique"),
    )
    return summary.reset_index()


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    if "time_s" in df.columns or "time_s_mean" in df.columns:
        return _summarize_benchmark(df)
    if "twist_deg" in df.columns:
        return _summarize_twist(df)
    raise ValueError("Unrecognized CSV format; expected benchmark or twist columns.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate incremental benchmark results.")
    parser.add_argument("--input", help="Glob pattern for input CSVs.")
    parser.add_argument("--db-glob", help="Glob pattern for input SQLite DBs.")
    parser.add_argument("--db-table", default="benchmark_results", help="SQLite table to read.")
    parser.add_argument("--output", required=True, type=Path, help="Summary CSV output path.")
    parser.add_argument("--interval", type=float, default=60.0, help="Seconds between scans.")
    parser.add_argument("--once", action="store_true", help="Run once and exit.")
    args = parser.parse_args()

    if args.input is None and args.db_glob is None:
        raise ValueError("Provide either --input for CSVs or --db-glob for SQLite DBs.")

    last_signature = None
    while True:
        if args.db_glob is not None:
            df = _load_dbs(args.db_glob, args.db_table)
        else:
            df = _load_csvs(args.input)
        if df.empty:
            if args.once:
                return
            time.sleep(args.interval)
            continue

        signature = (len(df), tuple(sorted(df.columns)))
        if signature != last_signature:
            summary = _summarize(df)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(args.output, index=False)
            last_signature = signature

        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
