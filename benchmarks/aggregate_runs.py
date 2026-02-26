#!/usr/bin/env python3
"""
aggregate_runs.py

Aggregate per-run metrics.csv files into a single master table.

Assumes raw runs live at:
  benchmarks/results/raw_runs/results_<YYYY-MM-DD_HH-MM-SS>/

Writes to:
  benchmarks/results/aggregated/runs_master.csv
Optionally:
  benchmarks/results/aggregated/runs_master.parquet (if pyarrow installed)

Usage:
  cd ~/ai-lab/benchmarks
  python aggregate_runs.py

Tip:
  This script is intentionally separate from run_suite. Keep run_suite focused on execution.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


RUN_DIR_RE = re.compile(r"^results_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})$")


@dataclass
class Paths:
    bench_dir: Path
    raw_runs_dir: Path
    aggregated_dir: Path
    out_csv: Path
    out_parquet: Path


def get_paths() -> Paths:
    bench_dir = Path(__file__).resolve().parent
    raw_runs_dir = bench_dir / "results" / "raw_runs"
    aggregated_dir = bench_dir / "results" / "aggregated"
    aggregated_dir.mkdir(parents=True, exist_ok=True)

    return Paths(
        bench_dir=bench_dir,
        raw_runs_dir=raw_runs_dir,
        aggregated_dir=aggregated_dir,
        out_csv=aggregated_dir / "runs_master.csv",
        out_parquet=aggregated_dir / "runs_master.parquet",
    )


def parse_run_timestamp(run_dir_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (run_id, date_str, iso_ts) if name matches, else (None, None, None).
    """
    m = RUN_DIR_RE.match(run_dir_name)
    if not m:
        return None, None, None
    date_part, time_part = m.group(1), m.group(2)
    run_id = f"{date_part}_{time_part}"
    try:
        # time_part uses '-' separators (HH-MM-SS)
        dt = datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H-%M-%S")
        iso_ts = dt.isoformat()
    except Exception:
        iso_ts = None
    return run_id, date_part, iso_ts


def safe_read_metrics_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        # Normalize column names a bit (optional)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None


def count_jsonl_rows(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        n = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    n += 1
        return n
    except Exception as e:
        print(f"[WARN] Failed to count jsonl rows in {path}: {e}")
        return None


def aggregate_one_run(run_dir: Path) -> Optional[pd.DataFrame]:
    """
    Returns a DataFrame for this run with run-level metadata columns attached.
    If metrics.csv exists, we return it (possibly multiple rows).
    If it doesn't, we skip the run.
    """
    run_name = run_dir.name
    run_id, date_str, iso_ts = parse_run_timestamp(run_name)

    metrics_path = run_dir / "metrics.csv"
    responses_path = run_dir / "responses.jsonl"

    df = safe_read_metrics_csv(metrics_path)
    if df is None:
        print(f"[WARN] Skipping {run_name}: missing or unreadable metrics.csv")
        return None

    # --- Safe run metadata (don't collide with newer schemas) ---
    # Prefer keeping the run_suite-provided run_id if it exists.
    if "run_id" not in df.columns:
        df.insert(0, "run_id", run_id if run_id else run_name)

    # Keep run_dirname-derived timestamp as separate columns (won't collide)
    if "run_name" not in df.columns:
        df.insert(0, "run_name", run_name)
    if "run_dir" not in df.columns:
        df.insert(0, "run_dir", str(run_dir))
    if "run_date" not in df.columns:
        df.insert(0, "run_date", date_str)
    if "run_timestamp_iso" not in df.columns:
        df.insert(0, "run_timestamp_iso", iso_ts)

    # File pointers + counts (safe to overwrite each run)
    df["metrics_path"] = str(metrics_path) if metrics_path.exists() else None
    df["responses_path"] = str(responses_path) if responses_path.exists() else None
    df["responses_rows"] = count_jsonl_rows(responses_path)

    # File existence + counts
    df["metrics_path"] = str(metrics_path) if metrics_path.exists() else None
    df["responses_path"] = str(responses_path) if responses_path.exists() else None
    df["responses_rows"] = count_jsonl_rows(responses_path)
    df["run_id_from_dir"] = run_id if run_id else run_name

    # Helpful: infer prompt/model columns if your older schema used different names
    # (No-op if already present)
    rename_map = {}
    for col in df.columns:
        if col.lower() == "model_name":
            rename_map[col] = "model"
        if col.lower() == "prompt_name":
            rename_map[col] = "prompt"
    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def main() -> int:
    paths = get_paths()

    if not paths.raw_runs_dir.exists():
        print(f"[ERROR] Raw runs directory not found: {paths.raw_runs_dir}")
        return 1

    run_dirs = sorted([p for p in paths.raw_runs_dir.iterdir() if p.is_dir() and p.name.startswith("results_")])
    if not run_dirs:
        print(f"[WARN] No run directories found under: {paths.raw_runs_dir}")
        return 0

    frames: List[pd.DataFrame] = []
    for rd in run_dirs:
        out = aggregate_one_run(rd)
        if out is not None:
            frames.append(out)

    if not frames:
        print("[WARN] No metrics aggregated (all runs missing/invalid metrics.csv).")
        return 0

    master = pd.concat(frames, ignore_index=True, sort=True)

    # Optional: stable ordering — put key cols first if they exist
    preferred = [
        "run_id",
        "run_timestamp_iso",
        "run_name",
        "run_dir",
        "model",
        "prompt",
        "rep",  # if you have a rep column
        "score",  # if you have a score column
        "responses_rows",
        "metrics_path",
        "responses_path",
    ]
    cols = list(master.columns)
    ordered = [c for c in preferred if c in cols] + [c for c in cols if c not in preferred]
    master = master[ordered]

    # Write outputs
    master.to_csv(paths.out_csv, index=False)
    print(f"[OK] Wrote {len(master):,} rows -> {paths.out_csv}")

    try:
        master.to_parquet(paths.out_parquet, index=False)
        print(f"[OK] Wrote parquet -> {paths.out_parquet}")
    except Exception as e:
        print(f"[INFO] Parquet not written (install pyarrow to enable): {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())