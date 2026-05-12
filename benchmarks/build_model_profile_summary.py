from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError


DEFAULT_RESULTS_ROOT = Path("benchmarks/results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build durable model profile summary artifacts from aggregated benchmark telemetry."
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Results root containing aggregated summary files.",
    )
    parser.add_argument(
        "--baseline-experiment",
        default="temp0",
        help="Baseline experiment name, e.g. temp0.",
    )
    parser.add_argument(
        "--comparison-experiment",
        default="temp07",
        help="Comparison experiment name, e.g. temp03 or temp07.",
    )
    parser.add_argument(
        "--suite-name",
        default="suite_macro_fred",
        help="Suite name to filter before building model profiles.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def load_global_runs_master_for_scope(
    *,
    suite_name: str,
    baseline_experiment: str,
    comparison_experiment: str,
) -> pd.DataFrame:
    repo_root = Path(__file__).resolve().parents[1]
    global_path = repo_root / "benchmarks" / "results" / "aggregated" / "runs_master.csv"

    df = load_csv(global_path)
    if df.empty:
        return pd.DataFrame()

    required_cols = {"suite_name", "experiment_name", "model"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Global runs_master.csv is missing required columns: {sorted(missing)}"
        )

    experiments = {baseline_experiment, comparison_experiment}

    scoped = df[
        (df["suite_name"].astype(str) == suite_name)
        & (df["experiment_name"].astype(str).isin(experiments))
    ].copy()

    print(f"Loaded global runs_master.csv: {global_path}")
    print(f"Filtered suite_name={suite_name}")
    print(f"Filtered experiments={sorted(experiments)}")
    print(f"Rows after filter: {len(scoped)}")

    return scoped

def classify_consistency(hash_stability: float) -> str:
    if pd.isna(hash_stability):
        return "unknown"
    if hash_stability >= 0.75:
        return "high"
    if hash_stability >= 0.40:
        return "medium"
    return "low"


def classify_adaptability(avg_abs_delta: float) -> str:
    if pd.isna(avg_abs_delta):
        return "unknown"
    if avg_abs_delta >= 0.15:
        return "high"
    if avg_abs_delta >= 0.05:
        return "medium"
    return "low"


def classify_temperature_sensitivity(avg_abs_delta: float) -> str:
    if pd.isna(avg_abs_delta):
        return "unknown"
    if avg_abs_delta >= 0.15:
        return "high"
    if avg_abs_delta >= 0.05:
        return "moderate"
    return "low"


def classify_overall_direction(avg_delta: float) -> str:
    if pd.isna(avg_delta):
        return "unknown"
    if avg_delta >= 0.05:
        return "mild improvement"
    if avg_delta <= -0.05:
        return "mild degradation"
    return "stable"


def classify_model_role(row: pd.Series) -> str:
    sensitivity = row.get("temperature_sensitivity", "unknown")
    direction = row.get("overall_direction", "unknown")
    adaptability = row.get("adaptability", "unknown")

    if sensitivity == "low" and direction == "stable":
        return "anchor"

    if adaptability == "high" and direction in {"stable", "mild improvement"}:
        return "explorer"

    if direction == "mild degradation":
        return "drifting_capacity"

    return "mixed"


def infer_repair_focus(row: pd.Series) -> list[str]:
    focus: list[str] = []

    failure_mode = str(row.get("dominant_failure_type_v2", "")).strip()
    semantic_pattern = str(row.get("dominant_semantic_pattern", "")).strip()
    consistency = str(row.get("consistency", "")).strip()
    adaptability = str(row.get("adaptability", "")).strip()
    model_role = str(row.get("model_role", "")).strip()

    if model_role == "explorer" or adaptability == "high":
        focus.append("allow_exploration_then_constrain")

    if failure_mode == "verbosity_drift":
        focus.append("compress_output")

    if consistency == "low":
        focus.append("enforce_structured_output")

    if semantic_pattern == "over_selection":
        focus.append("tighten_selection")

    if semantic_pattern == "mixed_selection_error":
        focus.append("refine_filtering")

    if not focus:
        focus.append("minimal_repair")

    # preserve order while deduping
    return list(dict.fromkeys(focus))


def dominant_value(series: pd.Series) -> str:
    s = series.dropna().astype(str)
    if s.empty:
        return "unknown"
    return s.value_counts().idxmax()


def aggregate_raw_runs_to_behavior_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    required = {"suite_name", "experiment_name", "model"}
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(f"Cannot aggregate raw runs; missing columns: {sorted(missing)}")

    df = raw_df.copy()

    # Prefer modern telemetry columns, but tolerate older column names.
    if "checks_ok" in df.columns:
        df["checks_pass_numeric"] = df["checks_ok"].fillna(False).astype(bool).astype(float)
    elif "overall_ok" in df.columns:
        df["checks_pass_numeric"] = df["overall_ok"].fillna(False).astype(bool).astype(float)
    else:
        df["checks_pass_numeric"] = 0.0

    if "overall_ok" in df.columns:
        df["overall_pass_numeric"] = df["overall_ok"].fillna(False).astype(bool).astype(float)
    elif "ok" in df.columns:
        df["overall_pass_numeric"] = df["ok"].fillna(False).astype(bool).astype(float)
    else:
        df["overall_pass_numeric"] = df["checks_pass_numeric"]

    if "response_hash" not in df.columns:
        df["response_hash"] = pd.NA

    if "sentence_count" not in df.columns:
        df["sentence_count"] = pd.NA

    if "avg_sentence_length_words" not in df.columns:
        df["avg_sentence_length_words"] = pd.NA

    if "words" not in df.columns:
        df["words"] = pd.NA

    if "failure_type_v2" not in df.columns:
        df["failure_type_v2"] = pd.NA

    if "semantic_pattern" not in df.columns:
        df["semantic_pattern"] = pd.NA

    grouped = (
        df.groupby(["suite_name", "experiment_name", "model"], dropna=False)
        .agg(
            rows=("model", "size"),
            overall_pass_rate=("overall_pass_numeric", "mean"),
            checks_pass_rate=("checks_pass_numeric", "mean"),
            dominant_failure_type_v2=("failure_type_v2", dominant_value),
            dominant_semantic_pattern=("semantic_pattern", dominant_value),
            avg_sentence_count=("sentence_count", "mean"),
            avg_sentence_length=("avg_sentence_length_words", "mean"),
            avg_words=("words", "mean"),
            unique_response_hashes=("response_hash", lambda s: s.dropna().nunique()),
        )
        .reset_index()
    )

    grouped["response_hash_stability_rate"] = (
        1.0 - (grouped["unique_response_hashes"] / grouped["rows"])
    ).clip(lower=0.0, upper=1.0)

    return grouped


def make_behavior_summary(behavior_df: pd.DataFrame, comparison_experiment: str) -> pd.DataFrame:
    if behavior_df.empty:
        return pd.DataFrame()

    df = behavior_df.copy()

    if "experiment_name" in df.columns:
        df = df[df["experiment_name"] == comparison_experiment].copy()

    keep_cols = [
        "suite_name",
        "experiment_name",
        "model",
        "rows",
        "overall_pass_rate",
        "checks_pass_rate",
        "dominant_failure_type_v2",
        "dominant_semantic_pattern",
        "avg_sentence_count",
        "avg_sentence_length",
        "avg_words",
        "unique_response_hashes",
        "response_hash_stability_rate",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].copy()


def make_delta_summary(
    behavior_df: pd.DataFrame,
    baseline_experiment: str,
    comparison_experiment: str,
) -> pd.DataFrame:
    required = {"model", "experiment_name", "checks_pass_rate"}
    if behavior_df.empty or not required.issubset(set(behavior_df.columns)):
        return pd.DataFrame()

    df = behavior_df.copy()

    base = df[df["experiment_name"] == baseline_experiment][
        ["model", "checks_pass_rate"]
    ].rename(columns={"checks_pass_rate": "baseline_checks_pass_rate"})

    comp = df[df["experiment_name"] == comparison_experiment][
        ["model", "checks_pass_rate"]
    ].rename(columns={"checks_pass_rate": "comparison_checks_pass_rate"})

    merged = comp.merge(base, on="model", how="inner")

    if merged.empty:
        return pd.DataFrame()

    merged["avg_delta"] = (
        merged["comparison_checks_pass_rate"] - merged["baseline_checks_pass_rate"]
    )
    merged["avg_abs_delta"] = merged["avg_delta"].abs()

    # Best/worst prompt are unknown in this fallback unless prompt-level delta file exists.
    merged["best_prompt"] = "unknown"
    merged["best_delta"] = 0.0
    merged["worst_prompt"] = "unknown"
    merged["worst_delta"] = 0.0

    merged["temperature_sensitivity"] = merged["avg_abs_delta"].apply(
        classify_temperature_sensitivity
    )
    merged["overall_direction"] = merged["avg_delta"].apply(classify_overall_direction)

    return merged[
        [
            "model",
            "avg_delta",
            "avg_abs_delta",
            "temperature_sensitivity",
            "overall_direction",
            "best_prompt",
            "best_delta",
            "worst_prompt",
            "worst_delta",
        ]
    ]


def try_load_behavior_summary(
    *,
    agg_dir: Path,
    suite_name: str,
    baseline_experiment: str,
    comparison_experiment: str,
) -> pd.DataFrame:
    candidates = [
        agg_dir / "model_behavior_summary.csv",
        agg_dir / "behavior_summary.csv",
        agg_dir / "runs_master.csv",
    ]

    for path in candidates:
        df = load_csv(path)
        if not df.empty:
            print(f"Loaded behavior source: {path}")
            return df

    return load_global_runs_master_for_scope(
        suite_name=suite_name,
        baseline_experiment=baseline_experiment,
        comparison_experiment=comparison_experiment,
    )


def build_model_profile(
    *,
    behavior_df: pd.DataFrame,
    baseline_experiment: str,
    comparison_experiment: str,
) -> pd.DataFrame:
    if {"prompt_id", "checks_ok", "response_hash"}.intersection(behavior_df.columns):
        behavior_df = aggregate_raw_runs_to_behavior_summary(behavior_df)

    behavior_summary = make_behavior_summary(behavior_df, comparison_experiment)
    delta_summary = make_delta_summary(
        behavior_df,
        baseline_experiment,
        comparison_experiment,
    )

    if behavior_summary.empty:
        return pd.DataFrame()

    if delta_summary.empty:
        profile = behavior_summary.copy()
        profile["avg_delta"] = 0.0
        profile["avg_abs_delta"] = 0.0
        profile["temperature_sensitivity"] = "unknown"
        profile["overall_direction"] = "unknown"
        profile["best_prompt"] = "unknown"
        profile["best_delta"] = 0.0
        profile["worst_prompt"] = "unknown"
        profile["worst_delta"] = 0.0
    else:
        profile = delta_summary.merge(
            behavior_summary,
            on="model",
            how="left",
        )

    if "response_hash_stability_rate" in profile.columns:
        profile["consistency"] = profile["response_hash_stability_rate"].apply(
            classify_consistency
        )
    else:
        profile["consistency"] = "unknown"

    profile["adaptability"] = profile["avg_abs_delta"].apply(classify_adaptability)
    profile["model_role"] = profile.apply(classify_model_role, axis=1)
    profile["repair_focus"] = profile.apply(infer_repair_focus, axis=1)

    # Store list fields as comma-separated strings for CSV friendliness.
    profile["repair_focus"] = profile["repair_focus"].apply(lambda xs: ", ".join(xs))

    return profile.sort_values("model").reset_index(drop=True)


def main() -> None:
    args = parse_args()

    results_root = Path(args.results_root).resolve()
    agg_dir = results_root / "aggregated"

    behavior_df = try_load_behavior_summary(
        agg_dir=agg_dir,
        suite_name=args.suite_name,
        baseline_experiment=args.baseline_experiment,
        comparison_experiment=args.comparison_experiment,
    )

    if behavior_df.empty:
        raise FileNotFoundError(
            f"No behavior summary source found under {agg_dir}. "
            "Expected one of: model_behavior_summary.csv, behavior_summary.csv, runs_master.csv"
        )

    profile = build_model_profile(
        behavior_df=behavior_df,
        baseline_experiment=args.baseline_experiment,
        comparison_experiment=args.comparison_experiment,
    )

    if profile.empty:
        raise ValueError("Model profile summary is empty after processing.")

    out_csv = agg_dir / "model_profile_summary.csv"
    out_json = agg_dir / "model_profile_summary.json"

    profile.to_csv(out_csv, index=False)
    out_json.write_text(
        json.dumps(profile.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    print(f"Saved model profile summary CSV: {out_csv}")
    print(f"Saved model profile summary JSON: {out_json}")

    print("\n=== MODEL PROFILE SUMMARY ===")
    display_cols = [
        "model",
        "model_role",
        "temperature_sensitivity",
        "overall_direction",
        "dominant_failure_type_v2",
        "dominant_semantic_pattern",
        "consistency",
        "adaptability",
        "repair_focus",
    ]
    print(profile[[c for c in display_cols if c in profile.columns]].to_string(index=False))


if __name__ == "__main__":
    main()