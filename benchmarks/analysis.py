from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
AGG_ROOT = REPO_ROOT / "benchmarks" / "results" / "aggregated"
AGG_PARQUET = AGG_ROOT / "runs_master.parquet"
AGG_CSV = AGG_ROOT / "runs_master.csv"


def normalize_prompt_id_series(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .str.replace(r"_+$", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )


def load_master() -> pd.DataFrame:
    if AGG_PARQUET.exists():
        df = pd.read_parquet(AGG_PARQUET)
    elif AGG_CSV.exists():
        df = pd.read_csv(AGG_CSV)
    else:
        raise FileNotFoundError(f"No runs_master found in {AGG_ROOT}")

    if "prompt_id" in df.columns:
        df["prompt_id"] = normalize_prompt_id_series(df["prompt_id"])

    for col in ["ok", "checks_ok", "overall_ok", "elapsed_s", "words_per_s", "checks_total"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "suite_name" in df.columns:
        df["suite_name"] = df["suite_name"].astype("string")
    if "experiment_name" in df.columns:
        df["experiment_name"] = df["experiment_name"].astype("string")
    if "model" in df.columns:
        df["model"] = df["model"].astype("string")
    if "prompt_id" in df.columns:
        df["prompt_id"] = df["prompt_id"].astype("string")

    return df


def make_experiment_metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    required = {"suite_name", "experiment_name", "prompt_id", "model"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required - set(df.columns)}")

    d = df.copy()

    if "checks_total" in d.columns:
        d["has_checks"] = d["checks_total"].fillna(0) > 0
    else:
        d["has_checks"] = False

    grouped = (
        d.groupby(["suite_name", "experiment_name", "prompt_id", "model"], as_index=False)
        .agg(
            rows=("model", "size"),
            pipeline_success_rate=("ok", "mean"),
            overall_pass_rate=("overall_ok", "mean"),
            avg_latency=("elapsed_s", "mean"),
            avg_wps=("words_per_s", "mean"),
        )
    )

    if "checks_ok" in d.columns:
        checks_only = d[d["has_checks"]].copy()
        if len(checks_only):
            checks_summary = (
                checks_only.groupby(
                    ["suite_name", "experiment_name", "prompt_id", "model"],
                    as_index=False,
                )
                .agg(checks_pass_rate=("checks_ok", "mean"))
            )
            grouped = grouped.merge(
                checks_summary,
                on=["suite_name", "experiment_name", "prompt_id", "model"],
                how="left",
            )
        else:
            grouped["checks_pass_rate"] = pd.NA
    else:
        grouped["checks_pass_rate"] = pd.NA

    for col in [
        "pipeline_success_rate",
        "checks_pass_rate",
        "overall_pass_rate",
        "avg_latency",
        "avg_wps",
    ]:
        if col in grouped.columns:
            grouped[col] = pd.to_numeric(grouped[col], errors="coerce")

    return grouped.sort_values(
        ["suite_name", "experiment_name", "prompt_id", "model"]
    ).reset_index(drop=True)


def make_experiment_delta_table(
    experiment_summary: pd.DataFrame,
    suite_name: str,
    baseline_experiment: str,
    comparison_experiment: str,
    metric_col: str,
) -> pd.DataFrame:
    d = experiment_summary.copy()
    d = d[d["suite_name"].astype("string") == str(suite_name)].copy()

    base = d[d["experiment_name"].astype("string") == str(baseline_experiment)].copy()
    comp = d[d["experiment_name"].astype("string") == str(comparison_experiment)].copy()

    if base.empty:
        raise ValueError(f"No rows found for baseline experiment '{baseline_experiment}' in suite '{suite_name}'")
    if comp.empty:
        raise ValueError(f"No rows found for comparison experiment '{comparison_experiment}' in suite '{suite_name}'")
    if metric_col not in d.columns:
        raise ValueError(f"Metric '{metric_col}' not found in experiment summary")

    base = base[["prompt_id", "model", metric_col]].rename(columns={metric_col: "baseline_pass_rate"})
    comp = comp[["prompt_id", "model", metric_col]].rename(columns={metric_col: "comparison_pass_rate"})

    merged = base.merge(comp, on=["prompt_id", "model"], how="outer")

    merged["baseline_pass_rate"] = pd.to_numeric(merged["baseline_pass_rate"], errors="coerce").fillna(0.0)
    merged["comparison_pass_rate"] = pd.to_numeric(merged["comparison_pass_rate"], errors="coerce").fillna(0.0)
    merged["delta_pass_rate"] = merged["comparison_pass_rate"] - merged["baseline_pass_rate"]
    merged["abs_delta"] = merged["delta_pass_rate"].abs()

    return merged.sort_values(
        ["abs_delta", "prompt_id", "model"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def label_cell_behavior(
    baseline: float,
    comparison: float,
    delta: float,
    eps: float = 1e-9,
) -> str:
    if abs(delta) <= eps:
        if baseline == 0.0 and comparison == 0.0:
            return "stable_always_fail"
        if baseline == 1.0 and comparison == 1.0:
            return "stable_always_pass"
        return "stable_invariant"
    if delta > 0:
        return "improves"
    return "degrades"


def make_narrative_payload(
    experiment_summary: pd.DataFrame,
    suite_name: str,
    baseline_experiment: str,
    comparison_experiments: list[str],
    metric_col: str,
) -> dict[str, Any]:
    suite_summary = experiment_summary[
        experiment_summary["suite_name"].astype("string") == str(suite_name)
    ].copy()

    if suite_summary.empty:
        raise ValueError(f"No rows found for suite '{suite_name}'")

    cells: dict[tuple[str, str], dict[str, Any]] = {}

    baseline_rows = suite_summary[
        suite_summary["experiment_name"].astype("string") == str(baseline_experiment)
    ].copy()

    if baseline_rows.empty:
        raise ValueError(f"No rows found for baseline experiment '{baseline_experiment}' in suite '{suite_name}'")

    for _, row in baseline_rows.iterrows():
        key = (str(row["prompt_id"]), str(row["model"]))
        cells[key] = {
            "prompt_id": str(row["prompt_id"]),
            "model": str(row["model"]),
            "baseline_pass_rate": float(row[metric_col]) if pd.notna(row[metric_col]) else 0.0,
            "comparison_pass_rates": {},
            "deltas": {},
            "labels": {},
        }

    comparisons_payload = []

    for comp_name in comparison_experiments:
        delta_df = make_experiment_delta_table(
            experiment_summary=experiment_summary,
            suite_name=suite_name,
            baseline_experiment=baseline_experiment,
            comparison_experiment=comp_name,
            metric_col=metric_col,
        )

        comparison_cells = []
        for _, row in delta_df.iterrows():
            prompt_id = str(row["prompt_id"])
            model = str(row["model"])
            key = (prompt_id, model)

            if key not in cells:
                cells[key] = {
                    "prompt_id": prompt_id,
                    "model": model,
                    "baseline_pass_rate": float(row["baseline_pass_rate"]),
                    "comparison_pass_rates": {},
                    "deltas": {},
                    "labels": {},
                }

            comp_rate = float(row["comparison_pass_rate"])
            delta = float(row["delta_pass_rate"])

            cells[key]["comparison_pass_rates"][comp_name] = comp_rate
            cells[key]["deltas"][f"{comp_name}_vs_{baseline_experiment}"] = delta
            cells[key]["labels"][f"{comp_name}_vs_{baseline_experiment}"] = label_cell_behavior(
                baseline=float(row["baseline_pass_rate"]),
                comparison=comp_rate,
                delta=delta,
            )

            comparison_cells.append(
                {
                    "prompt_id": prompt_id,
                    "model": model,
                    "baseline_pass_rate": float(row["baseline_pass_rate"]),
                    "comparison_pass_rate": comp_rate,
                    "delta_pass_rate": delta,
                    "label": label_cell_behavior(
                        baseline=float(row["baseline_pass_rate"]),
                        comparison=comp_rate,
                        delta=delta,
                    ),
                }
            )

        comparisons_payload.append(
            {
                "comparison_experiment": comp_name,
                "cells": comparison_cells,
            }
        )

    return {
        "suite_name": suite_name,
        "metric": metric_col,
        "baseline_experiment": baseline_experiment,
        "comparison_experiments": comparison_experiments,
        "n_cells": len(cells),
        "cells": list(cells.values()),
        "comparisons": comparisons_payload,
    }


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)