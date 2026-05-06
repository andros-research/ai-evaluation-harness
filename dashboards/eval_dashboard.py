import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import re
from typing import Any
from pandas.errors import EmptyDataError

st.set_page_config(layout="wide")
st.title("AI Evaluation Harness Dashboard")

# -------------------------
# Auto-load latest results (benchmarks/results_*)
# -------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_RUNS_ROOT = REPO_ROOT / "benchmarks" / "results" / "raw_runs"
AGG_ROOT = REPO_ROOT / "benchmarks" / "results" / "aggregated"
AGG_PARQUET = AGG_ROOT / "runs_master.parquet"
AGG_CSV = AGG_ROOT / "runs_master.csv"
FAILURE_SUMMARY_CSV = AGG_ROOT / "failure_taxonomy_summary.csv"

RUNS_ROOT = REPO_ROOT / "benchmarks" / "results" / "runs"
ARCHIVE_ROOT = REPO_ROOT / "benchmarks" / "results" / "archive"
LEGACY_NARRATIVES_ROOT = REPO_ROOT / "benchmarks" / "results" / "narratives"
LEGACY_AGG_ROOT = REPO_ROOT / "benchmarks" / "results" / "aggregated"
SEMANTIC_PATTERN_CSV = AGG_ROOT / "semantic_pattern_summary.csv"

VERSION_COLS = [
    "harness_version",
    "check_schema_version",
    "failure_taxonomy_version",
    "semantic_pattern_version",
    "telemetry_schema_version",
]

REQUIRED_TELEMETRY_COLS = [
    "suite_name",
    "experiment_name",
    "prompt_id",
    "model",
    "overall_ok",
    "checks_ok",
    "checks_total",
    "failure_type_v2",
    "semantic_pattern",
    "task_type",
    "domain",
    "difficulty",
    "response_hash",
]

# -------------------------
# Scope discovery helpers
# -------------------------
def find_scope_folders(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()], reverse=True)

def get_scope_options() -> dict[str, list[str]]:
    return {
        "runs": find_scope_folders(RUNS_ROOT),
        "archive": find_scope_folders(ARCHIVE_ROOT),
    }

def resolve_scope_paths(scope_type: str, scope_name: str) -> dict[str, Path]:
    if scope_type == "runs":
        base = RUNS_ROOT / scope_name
        return {
            "base": base,
            "narratives_root": base / "narratives",
            "aggregated_root": base / "aggregated",
        }

    if scope_type == "archive":
        base = ARCHIVE_ROOT / scope_name

        # archive snapshots may be narrative-only
        agg_candidate = base / "aggregated"
        return {
            "base": base,
            "narratives_root": base if base.exists() else LEGACY_NARRATIVES_ROOT,
            "aggregated_root": agg_candidate if agg_candidate.exists() else LEGACY_AGG_ROOT,
        }

    return {
        "base": LEGACY_NARRATIVES_ROOT.parent,
        "narratives_root": LEGACY_NARRATIVES_ROOT,
        "aggregated_root": LEGACY_AGG_ROOT,
    }
    
# -------------------------
# Helpers
# -------------------------
def find_result_folders(raw_runs_root: Path) -> list[Path]:
    if not raw_runs_root.exists():
        return []
    return sorted(
        [p for p in raw_runs_root.iterdir() if p.is_dir() and p.name.startswith("results_")],
        reverse=True,
    )

REQUIRED_COLS = {"model", "prompt_id", "elapsed_s", "words", "failure_type", "words_per_s", "ok"}

def normalize_master(master: pd.DataFrame) -> pd.DataFrame:
    m = master.copy()

    # Coerce numeric fields
    for c in ["elapsed_s", "words_per_s", "words", "exit_code"]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")

    # Strings
    for c in ["failure_type", "error", "stderr", "model", "prompt_id", "run_id"]:
        if c in m.columns:
            m[c] = m[c].astype("string")

    m["failure_type"] = m.get("failure_type", pd.Series([], dtype="string")).fillna("").str.strip()
    m["error"] = m.get("error", pd.Series([], dtype="string")).fillna("").str.strip()
    m["stderr"] = m.get("stderr", pd.Series([], dtype="string")).fillna("").str.strip()

    # ok normalization (handles 1/0, True/False, "1"/"0", "true"/"false", NA)
    if "ok" in m.columns:
        ok_raw = m["ok"]
        if str(ok_raw.dtype).startswith(("int", "float")):
            m["is_ok"] = ok_raw.fillna(0).astype(float).eq(1.0)
        else:
            s = ok_raw.astype("string").fillna("").str.lower().str.strip()
            m["is_ok"] = s.isin(["1", "true", "t", "yes", "y", "ok"])
    else:
        m["is_ok"] = pd.Series(False, index=m.index)

    # exit code / error signals
    m["has_nonzero_exit"] = m.get("exit_code", pd.Series(pd.NA, index=m.index)).fillna(0).astype(float).ne(0.0)
    m["has_error_text"] = (m["error"] != "") | (m["stderr"] != "")

    # failure_type indicates failure (treat anything non-empty and not "ok" as failure)
    m["failure_type_is_bad"] = (m["failure_type"] != "") & (m["failure_type"].str.lower() != "ok")

    # Final failure flag (robust)
    m["is_failure"] = (~m["is_ok"]) | m["has_nonzero_exit"] | m["failure_type_is_bad"] | m["has_error_text"]

    return m

def mean_boolish(df: pd.DataFrame, col: str):
    if col not in df.columns or len(df) == 0:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    if s.notna().sum() == 0:
        return None
    return s.mean()

def checked_subset(df: pd.DataFrame) -> pd.DataFrame:
    if "checks_total" not in df.columns:
        return df.iloc[0:0].copy()
    ct = pd.to_numeric(df["checks_total"], errors="coerce").fillna(0)
    return df[ct > 0].copy()

def make_worst_runs(m: pd.DataFrame) -> pd.DataFrame:
    if "run_id" not in m.columns:
        return pd.DataFrame()

    g = (
        m.groupby("run_id", as_index=False)
        .agg(
            rows=("run_id", "size"),
            fail_rate=("is_failure", "mean"),
            ok_rate=("is_ok", "mean"),
            nonzero_exit_rate=("has_nonzero_exit", "mean"),
            non_ok_failure_type_rate=("failure_type_is_bad", "mean"),
            median_latency=("elapsed_s", "median"),
            p95_latency=("elapsed_s", lambda x: x.quantile(0.95)),
            median_wps=("words_per_s", "median"),
            p05_wps=("words_per_s", lambda x: x.quantile(0.05)),
            n_models=("model", "nunique"),
            n_prompts=("prompt_id", "nunique"),
        )
    )

    # Sort: worst-first
    g = g.sort_values(
        by=["fail_rate", "nonzero_exit_rate", "p95_latency", "p05_wps", "rows"],
        ascending=[False, False, False, True, False],
    )

    return g

def make_hotspots(m: pd.DataFrame) -> pd.DataFrame:
    if not {"prompt_id", "model"}.issubset(set(m.columns)):
        return pd.DataFrame()

    h = (
        m.groupby(["prompt_id", "model"], as_index=False)
        .agg(
            rows=("model", "size"),
            fail_rate=("is_failure", "mean"),
            median_latency=("elapsed_s", "median"),
            median_wps=("words_per_s", "median"),
        )
        .sort_values(by=["fail_rate", "rows"], ascending=[False, False])
    )

    return h

def normalize_prompt_id_series(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .str.replace(r"_+$", "", regex=True)  # remove trailing underscores
        .str.replace(r"\s+", "_", regex=True)
    )
    
def format_heatmap_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.applymap(lambda x: f"{x:.0%}" if pd.notna(x) else "")

def format_delta_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.applymap(lambda x: f"{x:+.0%}" if pd.notna(x) else "")

def green_tag(x):
    return (
        f"<span style='color:#22c55e; background-color:#111827; "
        f"padding:2px 6px; border-radius:5px; font-family:monospace;'>"
        f"{x}</span>"
    )
    

def make_prompt_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    if not {"prompt_id", "model"}.issubset(df.columns):
        return pd.DataFrame()

    d = df.copy()

    # Safe numeric coercion
    for col in ["ok", "checks_ok", "overall_ok", "elapsed_s", "words_per_s", "checks_total"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    # Only compute checks pass rate on rows that actually had checks
    if "checks_total" in d.columns:
        d["has_checks"] = d["checks_total"].fillna(0) > 0
    else:
        d["has_checks"] = False
    
    # Prompt-id normalizer
    d["prompt_id"] = normalize_prompt_id_series(d["prompt_id"])

    grouped = (
        d.groupby(["prompt_id", "model"], as_index=False)
        .agg(
            rows=("model", "size"),
            pipeline_success_rate=("ok", "mean") if "ok" in d.columns else ("model", "size"),
            overall_pass_rate=("overall_ok", "mean") if "overall_ok" in d.columns else ("model", "size"),
            avg_latency=("elapsed_s", "mean") if "elapsed_s" in d.columns else ("model", "size"),
            avg_wps=("words_per_s", "mean") if "words_per_s" in d.columns else ("model", "size"),
        )
    )

    # checks pass rate calculated only on checked rows
    if "checks_ok" in d.columns and "has_checks" in d.columns:
        checks_only = d[d["has_checks"]].copy()
        if len(checks_only):
            checks_summary = (
                checks_only.groupby(["prompt_id", "model"], as_index=False)
                .agg(checks_pass_rate=("checks_ok", "mean"))
            )
            grouped = grouped.merge(checks_summary, on=["prompt_id", "model"], how="left")
        else:
            grouped["checks_pass_rate"] = pd.NA
    else:
        grouped["checks_pass_rate"] = pd.NA

    # Make sure pass-rate columns are numeric even after merges to make heatmap more robust
    for col in ["pipeline_success_rate", "checks_pass_rate", "overall_pass_rate", "avg_latency", "avg_wps"]:
        if col in grouped.columns:
            grouped[col] = pd.to_numeric(grouped[col], errors="coerce")

    grouped = grouped.sort_values(["prompt_id", "model"]).reset_index(drop=True)
    return grouped

def run_is_compatible(run_dir: Path) -> bool:
    metrics = run_dir / "metrics.csv"
    if not metrics.exists():
        return False
    try:
        # cheap read: only header
        df0 = pd.read_csv(metrics, nrows=1)
        return REQUIRED_COLS.issubset(set(df0.columns))
    except Exception:
        return False

def make_passrate_heatmap(pm_summary: pd.DataFrame, value_col: str = "overall_pass_rate") -> pd.DataFrame:
    if pm_summary.empty:
        return pd.DataFrame()
    if value_col not in pm_summary.columns:
        return pd.DataFrame()

    heat = pm_summary.pivot(index="prompt_id", columns="model", values=value_col)
    return heat.sort_index()

def make_experiment_metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    required = {"suite_name", "experiment_name", "prompt_id", "model"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    d = df.copy()

    for col in ["ok", "checks_ok", "overall_ok", "elapsed_s", "words_per_s", "checks_total"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    d["prompt_id"] = normalize_prompt_id_series(d["prompt_id"])

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
                    as_index=False
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

    for col in ["pipeline_success_rate", "checks_pass_rate", "overall_pass_rate", "avg_latency", "avg_wps"]:
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
    if experiment_summary.empty or metric_col not in experiment_summary.columns:
        return pd.DataFrame()

    cols = ["suite_name", "experiment_name", "prompt_id", "model", metric_col]
    d = experiment_summary[cols].copy()

    d = d[d["suite_name"].astype("string") == str(suite_name)].copy()

    base = d[d["experiment_name"].astype("string") == str(baseline_experiment)].copy()
    comp = d[d["experiment_name"].astype("string") == str(comparison_experiment)].copy()

    if base.empty or comp.empty:
        return pd.DataFrame()

    base = base.rename(columns={metric_col: "baseline_pass_rate"})
    comp = comp.rename(columns={metric_col: "comparison_pass_rate"})

    merged = base.merge(
        comp,
        on=["suite_name", "prompt_id", "model"],
        how="inner",
    )

    merged["baseline_pass_rate"] = pd.to_numeric(
        merged["baseline_pass_rate"], errors="coerce"
    )
    merged["comparison_pass_rate"] = pd.to_numeric(
        merged["comparison_pass_rate"], errors="coerce"
    )

    merged = merged.dropna(subset=["baseline_pass_rate", "comparison_pass_rate"]).copy()
    merged["delta_pass_rate"] = merged["comparison_pass_rate"] - merged["baseline_pass_rate"]
    merged["abs_delta"] = merged["delta_pass_rate"].abs()

    merged = merged[
        ["prompt_id", "model", "baseline_pass_rate", "comparison_pass_rate", "delta_pass_rate", "abs_delta"]
    ].sort_values(
        ["abs_delta", "prompt_id", "model"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    return merged

def make_delta_heatmap(delta_table: pd.DataFrame) -> pd.DataFrame:
    if delta_table.empty:
        return pd.DataFrame()
    return delta_table.pivot(index="prompt_id", columns="model", values="delta_pass_rate").sort_index()

def classify_delta(x: float) -> str:
    if pd.isna(x):
        return "unknown"
    if x >= 0.15:
        return "strong improvement"
    if x >= 0.05:
        return "mild improvement"
    if x <= -0.15:
        return "strong degradation"
    if x <= -0.05:
        return "mild degradation"
    return "stable"


def describe_sensitivity(avg_abs_delta: float) -> str:
    if pd.isna(avg_abs_delta):
        return "unknown"
    if avg_abs_delta >= 0.15:
        return "high"
    if avg_abs_delta >= 0.07:
        return "moderate"
    return "low"


def make_model_delta_summary(delta_table: pd.DataFrame) -> pd.DataFrame:
    if delta_table.empty:
        return pd.DataFrame()

    d = delta_table.copy()
    d["delta_pass_rate"] = pd.to_numeric(d["delta_pass_rate"], errors="coerce")

    rows = []

    for model, g in d.groupby("model"):
        avg_delta = g["delta_pass_rate"].mean()
        avg_abs_delta = g["delta_pass_rate"].abs().mean()

        best = g.sort_values("delta_pass_rate", ascending=False).head(1)
        worst = g.sort_values("delta_pass_rate", ascending=True).head(1)

        best_prompt = best["prompt_id"].iloc[0] if len(best) else "—"
        best_delta = best["delta_pass_rate"].iloc[0] if len(best) else pd.NA

        worst_prompt = worst["prompt_id"].iloc[0] if len(worst) else "—"
        worst_delta = worst["delta_pass_rate"].iloc[0] if len(worst) else pd.NA

        profile = classify_delta(avg_delta)
        sensitivity = describe_sensitivity(avg_abs_delta)

        rows.append({
            "model": model,
            "avg_delta": avg_delta,
            "avg_abs_delta": avg_abs_delta,
            "temperature_sensitivity": sensitivity,
            "overall_direction": profile,
            "best_prompt": best_prompt,
            "best_delta": best_delta,
            "worst_prompt": worst_prompt,
            "worst_delta": worst_delta,
        })

    return pd.DataFrame(rows)

def render_model_delta_sentence(row: pd.Series, baseline_experiment: str, comparison_experiment: str) -> str:
    model = row["model"]
    sensitivity = row["temperature_sensitivity"]
    direction = row["overall_direction"]

    avg_delta = row["avg_delta"]
    best_prompt = row["best_prompt"]
    best_delta = row["best_delta"]
    worst_prompt = row["worst_prompt"]
    worst_delta = row["worst_delta"]

    avg_txt = f"{avg_delta:+.0%}" if pd.notna(avg_delta) else "unknown"
    best_txt = f"{best_delta:+.0%}" if pd.notna(best_delta) else "unknown"
    worst_txt = f"{worst_delta:+.0%}" if pd.notna(worst_delta) else "unknown"

    return (
        f"{model} shows {sensitivity} temperature sensitivity from "
        f"{baseline_experiment} to {comparison_experiment}, with an average delta of {avg_txt}. "
        f"Overall direction is {direction}. "
        f"Strongest relative area: {best_prompt} ({best_txt}); "
        f"weakest relative area: {worst_prompt} ({worst_txt})."
    )


def make_model_profile_summary(
    delta_summary: pd.DataFrame,
    behavior_summary: pd.DataFrame,
    comparison_experiment: str,
) -> pd.DataFrame:
    if delta_summary.empty or behavior_summary.empty:
        return pd.DataFrame()

    b = behavior_summary.copy()
    b = b[b["experiment_name"].astype("string") == str(comparison_experiment)].copy()

    merged = delta_summary.merge(
        b,
        on="model",
        how="left",
        suffixes=("_delta", "_behavior"),
    )

    return merged


def format_failure_mode(x: str) -> str:
    if pd.isna(x) or str(x).strip() == "":
        return "unknown"
    if str(x).strip().lower() == "ok":
        return "no dominant failure mode"
    return str(x)
 
 
def article(word):
    return "an" if word[0].lower() in "aeiou" else "a"


def render_model_profile_sentence(row, baseline_experiment, comparison_experiment):
    failure_str = format_failure_mode(row["dominant_failure_type_v2"])
    if failure_str == "no dominant failure mode":
        failure_sentence = "It shows no dominant failure mode"
    else:
        failure_sentence = f"Its dominant failure mode is {failure_str}"
    role = row.get("model_role", "mixed")
    return (
        f"{row['model']} shows {row['temperature_sensitivity']} sensitivity from "
        f"{baseline_experiment} to {comparison_experiment}, with average delta "
        f"{row['avg_delta']:+.0%}. "
        f"{failure_sentence} and its dominant semantic pattern is "
        f"{row['dominant_semantic_pattern']}. "
        f"Response hash stability is {row['response_hash_stability_rate']:.0%}. "
        f"This model behaves as {article(role)} {role} under this comparison. "
        f"Consistency is {row['consistency']}; adaptability is {row['adaptability']}."
    )


def classify_consistency(hash_stability: float) -> str:
    x = pd.to_numeric(hash_stability, errors="coerce")
    if pd.isna(x):
        return "unknown"
    if x >= 0.75:
        return "high"
    if x >= 0.40:
        return "medium"
    return "low"


def classify_adaptability(avg_abs_delta: float) -> str:
    x = pd.to_numeric(avg_abs_delta, errors="coerce")
    if pd.isna(x):
        return "unknown"
    if x >= 0.15:
        return "high"
    if x >= 0.07:
        return "medium"
    return "low"


REPAIR_STRATEGIES = {
    "over_selection": ["tighten_selection"],
    "verbosity_drift": ["compress_output"],
    "mixed_selection_error": ["refine_filtering"],
    "symbolic_output": ["enforce_literal_output"],
    "narrative_drift": ["tighten_claim_scope"],
    "semantic_error": ["recheck_evidence_alignment"],
}
REPAIR_STRATEGY_DESCRIPTIONS = {
    "tighten_selection": "Reduce over-selection by requiring fewer, more directly supported claims.",
    "compress_output": "Shorten output and reduce narrative drift.",
    "refine_filtering": "Improve distinction between relevant and irrelevant evidence.",
    "enforce_structured_output": "Force stricter schema, formatting, and response shape.",
    "allow_exploration_then_constrain": "Use the model for broader idea generation, then apply stricter validation and compression.",
    "minimal_repair": "Apply only light cleanup; no major behavioral correction indicated.",
}
REPAIR_PROMPT_PATCHES = {
    "tighten_selection": "Select only the strongest directly supported claims. Exclude weakly related or redundant evidence.",
    "compress_output": "Rewrite the answer in fewer sentences while preserving only the core supported claims.",
    "refine_filtering": "Re-check each selected claim against the prompt objective and remove mismatched evidence.",
    "enforce_structured_output": "Return only the requested schema. Do not add commentary outside the schema.",
    "allow_exploration_then_constrain": "First identify plausible interpretations, then rank them by evidence quality and keep only the strongest.",
    "minimal_repair": "Make only minor clarity edits without changing the substance.",
}

def describe_repair_strategy(strategy: str) -> str:
    return REPAIR_STRATEGY_DESCRIPTIONS.get(
        strategy,
        "No description available for this repair strategy.",
    )
    

def get_repair_prompt_patch(strategy: str) -> str:
    return REPAIR_PROMPT_PATCHES.get(strategy, "")


def infer_repair_focus(row: pd.Series) -> list[str]:
    strategies = []

    semantic_pattern = str(row.get("dominant_semantic_pattern", "")).strip()
    failure_type = str(row.get("dominant_failure_type_v2", "")).strip()

    strategies.extend(REPAIR_STRATEGIES.get(semantic_pattern, []))
    strategies.extend(REPAIR_STRATEGIES.get(failure_type, []))

    if row.get("consistency") == "low":
        strategies.append("enforce_structured_output")

    if row.get("adaptability") == "high":
        strategies.append("allow_exploration_then_constrain")

    return sorted(set(strategies)) if strategies else ["minimal_repair"]


def classify_model_role(row: pd.Series) -> str:
    sensitivity = str(row.get("temperature_sensitivity", "")).lower()
    direction = str(row.get("overall_direction", "")).lower()
    avg_abs_delta = pd.to_numeric(row.get("avg_abs_delta"), errors="coerce")

    if sensitivity == "low" and direction == "stable":
        return "anchor"

    if "degradation" in direction:
        return "drifting_capacity"

    if "improvement" in direction and pd.notna(avg_abs_delta) and avg_abs_delta >= 0.08:
        return "explorer"

    return "mixed"


def telemetry_complete_subset(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    missing = [c for c in REQUIRED_TELEMETRY_COLS if c not in d.columns]
    if missing:
        return d.iloc[0:0].copy()

    mask = pd.Series(True, index=d.index)

    for col in REQUIRED_TELEMETRY_COLS:
        if col == "semantic_pattern":
            # Blank/NA is expected for non-selection tasks.
            continue

        mask &= d[col].notna()
        mask &= d[col].astype("string").str.strip().ne("")

    mask &= pd.to_numeric(d["checks_total"], errors="coerce").fillna(0).gt(0)

    return d[mask].copy()


def add_run_datetime(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    if "run_timestamp_iso" in d.columns:
        d["_run_dt"] = pd.to_datetime(d["run_timestamp_iso"], errors="coerce")
    elif "run_date" in d.columns:
        d["_run_dt"] = pd.to_datetime(d["run_date"], errors="coerce")
    elif "ts" in d.columns:
        d["_run_dt"] = pd.to_datetime(d["ts"], errors="coerce")
    else:
        d["_run_dt"] = pd.NaT

    if d["_run_dt"].isna().all() and "run_id" in d.columns:
        d["_run_dt"] = pd.to_datetime(d["run_id"], errors="coerce")

    return d


def latest_n_runs_per_experiment(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    if df.empty or "run_id" not in df.columns:
        return df.iloc[0:0].copy()

    d = add_run_datetime(df)

    run_level = (
        d.groupby(["suite_name", "experiment_name", "run_id"], as_index=False)
        .agg(run_dt=("_run_dt", "max"))
    )

    if run_level["run_dt"].isna().all():
        run_level = run_level.sort_values(
            ["suite_name", "experiment_name", "run_id"],
            ascending=[True, True, False],
        )
    else:
        run_level = run_level.sort_values(
            ["suite_name", "experiment_name", "run_dt", "run_id"],
            ascending=[True, True, False, False],
        )

    run_level["_rank"] = run_level.groupby(
        ["suite_name", "experiment_name"]
    ).cumcount() + 1

    keep = run_level[run_level["_rank"] <= n][
        ["suite_name", "experiment_name", "run_id"]
    ]

    return d.merge(keep, on=["suite_name", "experiment_name", "run_id"], how="inner")


def make_model_behavior_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    d = df.copy()

    # Ensure numeric
    for col in ["overall_ok", "checks_ok", "sentence_count", "avg_sentence_length_words", "words"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    grouped = []

    for (suite, experiment, model), g in d.groupby(["suite_name", "experiment_name", "model"]):
        rows = len(g)

        overall_pass = g["overall_ok"].mean() if "overall_ok" in g else None

        checks_only = g[g["checks_total"].fillna(0) > 0]
        checks_pass = checks_only["checks_ok"].mean() if len(checks_only) else None

        # Dominant failure type
        failure_mode = None
        if "failure_type_v2" in g:
            vc = g["failure_type_v2"].value_counts()
            if len(vc):
                failure_mode = vc.index[0]

        # Dominant semantic pattern
        semantic_mode = None
        if "semantic_pattern" in g:
            sp = g["semantic_pattern"].dropna()
            if len(sp):
                vc = sp.value_counts()
                if len(vc):
                    semantic_mode = vc.index[0]

        # Style
        avg_sent = g["sentence_count"].mean() if "sentence_count" in g else None
        avg_sent_len = g["avg_sentence_length_words"].mean() if "avg_sentence_length_words" in g else None
        avg_words = g["words"].mean() if "words" in g else None

        # Stability
        unique_hashes = g["response_hash"].nunique() if "response_hash" in g else None
        stability = None
        if unique_hashes is not None and rows > 0:
            stability = 1 - (unique_hashes / rows)

        grouped.append({
            "suite_name": suite,
            "experiment_name": experiment,
            "model": model,
            "rows": rows,
            "overall_pass_rate": overall_pass,
            "checks_pass_rate": checks_pass,
            "dominant_failure_type_v2": failure_mode,
            "dominant_semantic_pattern": semantic_mode,
            "avg_sentence_count": avg_sent,
            "avg_sentence_length": avg_sent_len,
            "avg_words": avg_words,
            "unique_response_hashes": unique_hashes,
            "response_hash_stability_rate": stability,
        })

    return pd.DataFrame(grouped)


def build_model_profile(row):
    return {
        "model": row["model"],
        "role": classify_model_role(row),
        "performance": {
            "avg_delta": row["avg_delta"],
            "temperature_sensitivity": row["temperature_sensitivity"],
            "overall_direction": row["overall_direction"],
        },
        "behavior": {
            "dominant_pattern": row["dominant_semantic_pattern"],
            "failure_mode": row["dominant_failure_type_v2"],
        },
        "reliability": {
            "consistency": compute_consistency(row),
            "adaptability": compute_adaptability(row),
            "hash_stability": row["response_hash_stability_rate"],
        },
        "strengths": row["best_prompt"],
        "weaknesses": row["worst_prompt"],
    }


def find_narrative_files(narratives_root: Path, suffix: str) -> list[Path]:
    if not narratives_root.exists():
        return []
    return sorted(narratives_root.glob(f"*{suffix}"), reverse=True)

def load_json_file(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_repair_matrix_summary(results_root: Path) -> pd.DataFrame:
    path = results_root / "aggregated" / "repair_matrix_summary.csv"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def load_repair_strategy_recommendation(results_root: Path) -> dict:
    path = results_root / "aggregated" / "repair_strategy_recommendation.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def narrative_label_from_path(path: Path) -> str:
    return path.name

def normalize_audit_text(s: str) -> str:
    s = re.sub(r"\[CLAIMS:\s*[^\]]+\]", "", s)
    s = re.sub(r"\s+", " ", s.strip().lower())
    return s


def default_audit_path_from_parsed(parsed_path: Path) -> Path:
    stem = parsed_path.stem.replace("__parsed_narrative", "")
    return parsed_path.parent / f"{stem}__audit.json"


def index_audit_items(audit_payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for item in audit_payload.get("audit_items", []):
        key = (
            str(item.get("section", "")).strip().lower(),
            normalize_audit_text(str(item.get("text", ""))),
        )
        out[key] = item
    return out

def fmt_pct(x: Any) -> str:
    try:
        return f"{float(x):.2%}"
    except Exception:
        return "—"

@st.cache_data(show_spinner=False)
def load_agg() -> pd.DataFrame:
    if AGG_PARQUET.exists():
        return pd.read_parquet(AGG_PARQUET)
    if AGG_CSV.exists():
        return pd.read_csv(AGG_CSV)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_failure_summary() -> pd.DataFrame:
    if FAILURE_SUMMARY_CSV.exists():
        return pd.read_csv(FAILURE_SUMMARY_CSV)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_audit_items(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except EmptyDataError:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_audit_summary(path: str) -> dict[str, Any]:
    p = Path(path)
    if p.exists():
        return load_json_file(p)
    return {}

@st.cache_data(show_spinner=False)
def load_claim_coverage(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except EmptyDataError:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_claim_coverage_summary(path: str) -> dict[str, Any]:
    p = Path(path)
    if p.exists():
        return load_json_file(p)
    return {}

@st.cache_data(show_spinner=False)
def load_semantic_patterns():
    if SEMANTIC_PATTERN_CSV.exists():
        return pd.read_csv(SEMANTIC_PATTERN_CSV)
    return pd.DataFrame()

# -------------------------
# Tabs
# -------------------------
tab_run, tab_agg, tab_audit, tab_trace = st.tabs(
    ["Single Run", "All Runs (runs_master)", "Audit Analytics", "Narrative Traceability"]
)

with tab_run:
    runs_all = find_result_folders(RAW_RUNS_ROOT)
    runs = [r for r in runs_all if run_is_compatible(r)]

    if not runs:
        st.error(
            "No compatible results folders found (need metrics.csv with columns: "
            + ", ".join(sorted(REQUIRED_COLS))
            + f") under: {RAW_RUNS_ROOT}"
        )
        st.stop()

    st.sidebar.caption(f"Showing {len(runs)} compatible runs (hiding {len(runs_all)-len(runs)} older/incomplete runs).")

    selected_run = st.sidebar.selectbox(
        "Select Run",
        options=[str(p) for p in runs],
        key="select_run",
    )

    metrics_path = Path(selected_run) / "metrics.csv"
    df = pd.read_csv(metrics_path)
    st.sidebar.write(f"Loaded: {metrics_path}")
    
    # Expose metadata in the Single Run tab
    suite_val = df["suite_name"].dropna().astype("string").unique().tolist() if "suite_name" in df.columns else []
    experiment_val = df["experiment_name"].dropna().astype("string").unique().tolist() if "experiment_name" in df.columns else []

    suite_display = suite_val[0] if len(suite_val) == 1 else ", ".join(suite_val) if suite_val else "—"
    experiment_display = experiment_val[0] if len(experiment_val) == 1 else ", ".join(experiment_val) if experiment_val else "—"

    st.caption(f"Suite: {suite_display} | Experiment: {experiment_display}")

    models = st.sidebar.multiselect(
        "Models",
        options=sorted(df["model"].dropna().unique()),
        default=sorted(df["model"].dropna().unique()),
        key="models_run",
    )

    prompts = st.sidebar.multiselect(
        "Prompts",
        options=sorted(df["prompt_id"].dropna().unique()),
        default=sorted(df["prompt_id"].dropna().unique()),
        key="prompts_run",
    )

    df_filtered = df[(df["model"].isin(models)) & (df["prompt_id"].isin(prompts))]

    st.subheader("Raw Metrics Snapshot")
    st.dataframe(df_filtered.head(50))

    pipeline_rate = mean_boolish(df_filtered, "ok")
    checks_df = checked_subset(df_filtered)
    checks_rate = mean_boolish(checks_df, "checks_ok")
    overall_rate = mean_boolish(df_filtered, "overall_ok")

    # Single Run KPI block
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", len(df_filtered))
    col2.metric("Pipeline Success Rate", f"{pipeline_rate:.2%}" if pipeline_rate is not None else "—")
    col3.metric("Checks Pass Rate", f"{checks_rate:.2%}" if checks_rate is not None else "—")
    col4.metric("Overall Pass Rate", f"{overall_rate:.2%}" if overall_rate is not None else "—")
    
    col5, col6 = st.columns(2)
    col5.metric("Avg Latency (s)", f"{df_filtered['elapsed_s'].mean():.2f}" if len(df_filtered) else "—")
    col6.metric("Avg Words", f"{df_filtered['words'].mean():.1f}" if len(df_filtered) else "—")
    
    t1, t2, t3 = st.columns(3)
    t1.metric("Avg sentence count", f"{df_filtered['sentence_count'].mean():.2f}" if "sentence_count" in df_filtered.columns and len(df_filtered) else "—")
    t2.metric("Avg sentence length", f"{df_filtered['avg_sentence_length_words'].mean():.2f}" if "avg_sentence_length_words" in df_filtered.columns and len(df_filtered) else "—")
    t3.metric("Constraint failure rate", f"{(df_filtered['failure_type'] == 'constraint_failure').mean():.2%}" if "failure_type" in df_filtered.columns and len(df_filtered) else "—")

    st.subheader("Prompt × Model Summary")
    pm_summary = make_prompt_model_summary(df_filtered)
    if not pm_summary.empty:
        show = pm_summary.copy()
        for col in ["pipeline_success_rate", "checks_pass_rate", "overall_pass_rate"]:
            if col in show.columns:
                show[col] = show[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
        for col in ["avg_latency", "avg_wps"]:
            if col in show.columns:
                show[col] = show[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        st.dataframe(show)
    else:
        st.info("Not enough data to build prompt × model summary.")
    
    st.subheader("Prompt × Model Pass-Rate Heatmap")
    heat_value_col = st.selectbox(
        "Heatmap metric (Single Run)",
        ["overall_pass_rate", "checks_pass_rate", "pipeline_success_rate"],
        index=0,
        key="heatmap_metric_run",
    )

    heat_run = make_passrate_heatmap(pm_summary, heat_value_col)
    if not heat_run.empty:
        figH1, axH1 = plt.subplots(figsize=(8, max(3, 0.6 * len(heat_run))))
        annot_run = format_heatmap_labels(heat_run)

        hm = sns.heatmap(
            heat_run,
            annot=annot_run,
            fmt="",
            cmap="YlGn",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="white",
            ax=axH1,
        )

        cbar = hm.collections[0].colorbar
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
        
        axH1.set_xlabel("Model")
        axH1.set_ylabel("Prompt")
        axH1.set_title(
            f"{heat_value_col} by prompt × model\n"
            f"[suite={suite_display} | experiment={experiment_display}]"
        )
        st.pyplot(figH1)
    else:
        st.info("Not enough data to build pass-rate heatmap.")

    st.subheader("Latency Distribution by Model")
    fig1, ax1 = plt.subplots()
    sns.boxplot(data=df_filtered, x="model", y="elapsed_s", ax=ax1)
    st.pyplot(fig1)

    st.subheader("Output Length (Words) Distribution")
    fig2, ax2 = plt.subplots()
    sns.violinplot(data=df_filtered, x="model", y="words", ax=ax2)
    st.pyplot(fig2)

    st.subheader("Failure Type Breakdown")
    st.bar_chart(df_filtered["failure_type"].value_counts())

    st.subheader("Model Efficiency (Words per Second)")
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=df_filtered, x="model", y="words_per_s", ax=ax3)
    st.pyplot(fig3)
 
 
with tab_agg:
    master = load_agg()

    scope_options = get_scope_options()

    selected_scope_type = st.sidebar.selectbox(
        "Repair Matrix: Scope type",
        options=["runs", "archive"],
        index=0,
        key="repair_matrix_scope_type",
    )

    selected_scope_names = scope_options.get(selected_scope_type, [])

    if selected_scope_names:
        selected_scope_name = st.sidebar.selectbox(
            "Repair Matrix: Scope",
            options=selected_scope_names,
            index=0,
            key="repair_matrix_scope_name",
        )

        scope_paths = resolve_scope_paths(selected_scope_type, selected_scope_name)
        repair_results_root = scope_paths["base"]
    else:
        repair_results_root = REPO_ROOT / "benchmarks" / "results"
    
    # Safety block for new telemetry fields that do not exist in earlier data
    for col in ["task_type", "domain", "difficulty"]:
        if col not in master.columns:
            master[col] = pd.NA
        master[col] = master[col].astype("string").fillna("")

    for col in ["sentence_count", "avg_sentence_length_words"]:
        if col not in master.columns:
            master[col] = pd.NA
        master[col] = pd.to_numeric(master[col], errors="coerce")

    if master.empty:
        st.error(f"No runs_master found. Expected: {AGG_PARQUET} or {AGG_CSV}")
        st.stop()
    
    for col in VERSION_COLS:
        if col not in master.columns:
            master[col] = pd.NA
        master[col] = master[col].astype("string").fillna("")
    
    # Prompt-id normalizer
    master["prompt_id"] = normalize_prompt_id_series(master["prompt_id"])
    
    # ---- Growth table (daily) ----
    growth_src = master.copy()

    # Ensure we have a run id for unique run counting
    if "run_id" not in growth_src.columns and "run_id_from_dir" in growth_src.columns:
        growth_src["run_id"] = growth_src["run_id_from_dir"]

    # Prefer run_timestamp_iso if present, else fall back to run_date/date
    if "run_timestamp_iso" in growth_src.columns and growth_src["run_timestamp_iso"].notna().any():
        growth_src["dt"] = pd.to_datetime(growth_src["run_timestamp_iso"], errors="coerce")
    elif "run_date" in growth_src.columns and growth_src["run_date"].notna().any():
        growth_src["dt"] = pd.to_datetime(growth_src["run_date"], errors="coerce")
    elif "date" in growth_src.columns and growth_src["date"].notna().any():
        growth_src["dt"] = pd.to_datetime(growth_src["date"], errors="coerce")
    else:
        growth_src["dt"] = pd.NaT

    growth_src = growth_src.dropna(subset=["dt"]).copy()
    growth_src["day"] = growth_src["dt"].dt.date.astype(str)

    daily = (
        growth_src.groupby("day", as_index=False)
        .agg(
            rows=("day", "size"),
            unique_runs=("run_id", "nunique") if "run_id" in growth_src.columns else ("day", "size"),
        )
        .sort_values("day")
    )

    daily["cum_rows"] = daily["rows"].cumsum()
    daily["cum_unique_runs"] = daily["unique_runs"].cumsum()
    

    # Normalize / safety: ensure expected cols exist (some early runs have blanks)
    for col in [
        "model", "prompt_id", "failure_type", "ok", "elapsed_s", "words", "words_per_s",
        "suite_name", "experiment_name"
    ]:
        if col not in master.columns:
            master[col] = pd.NA

    st.subheader("All Runs Snapshot (runs_master)")
    st.caption(f"Rows: {len(master):,} | Unique runs: {master['run_id'].nunique() if 'run_id' in master.columns else '—'}")
    st.dataframe(master.head(50))

    # Sidebar filters for aggregate view
    st.sidebar.markdown("### Aggregate Filters")

    agg_models = st.sidebar.multiselect(
        "Agg: Models",
        options=sorted(master["model"].dropna().unique()),
        default=sorted(master["model"].dropna().unique()),
        key="models_agg",
    )

    agg_prompts = st.sidebar.multiselect(
        "Agg: Prompts",
        options=sorted(master["prompt_id"].dropna().unique()),
        default=sorted(master["prompt_id"].dropna().unique()),
        key="prompts_agg",
    )
    
    # Aggregate sidebar filters for suite run name and experiment name
    agg_suites = st.sidebar.multiselect(
        "Agg: Suites",
        options=sorted(master["suite_name"].dropna().astype("string").unique()),
        default=sorted(master["suite_name"].dropna().astype("string").unique()),
        key="suites_agg",
    )

    agg_experiments = st.sidebar.multiselect(
        "Agg: Experiments",
        options=sorted(master["experiment_name"].dropna().astype("string").unique()),
        default=sorted(master["experiment_name"].dropna().astype("string").unique()),
        key="experiments_agg",
    )
    
    agg_task_types = st.sidebar.multiselect(
        "Agg: Task types",
        options=sorted(master["task_type"].dropna().astype("string").unique()),
        default=sorted(master["task_type"].dropna().astype("string").unique()),
        key="task_types_agg",
    )

    agg_domains = st.sidebar.multiselect(
        "Agg: Domains",
        options=sorted(master["domain"].dropna().astype("string").unique()),
        default=sorted(master["domain"].dropna().astype("string").unique()),
        key="domains_agg",
    )

    agg_difficulties = st.sidebar.multiselect(
        "Agg: Difficulties",
        options=sorted(master["difficulty"].dropna().astype("string").unique()),
        default=sorted(master["difficulty"].dropna().astype("string").unique()),
        key="difficulties_agg",
    )

    ok_only = st.sidebar.selectbox("Agg: Outcomes", ["All", "Only ok", "Only failures"], index=0)

    m = master.copy()
    m = m[
        m["model"].isin(agg_models)
        & m["prompt_id"].isin(agg_prompts)
        & m["suite_name"].astype("string").isin(agg_suites)
        & m["experiment_name"].astype("string").isin(agg_experiments)
        & m["task_type"].astype("string").isin(agg_task_types)
        & m["domain"].astype("string").isin(agg_domains)
        & m["difficulty"].astype("string").isin(agg_difficulties)
    ]
    experiment_summary_all = make_experiment_metric_summary(m)
    
    # Show selected suite/experiment context in the aggregate tab
    suite_label = ", ".join(agg_suites) if agg_suites else "—"
    experiment_label = ", ".join(agg_experiments) if agg_experiments else "—"
    st.caption(f"Filtered to suites: {suite_label}")
    st.caption(f"Filtered to experiments: {experiment_label}")

    # -------------------------
    # Comparison UI block
    # -------------------------   
    st.subheader("Experiment Comparison (within suite)")

    compare_src = master.copy()
    compare_src = compare_src[
        compare_src["model"].isin(agg_models)
        & compare_src["prompt_id"].isin(agg_prompts)
        & compare_src["suite_name"].astype("string").isin(agg_suites)
    ].copy()

    st.markdown("#### Comparison Data Scope")

    scope_col1, scope_col2, scope_col3 = st.columns(3)

    with scope_col1:
        use_telemetry_complete = st.checkbox(
            "Core telemetry-complete rows only",
            value=True,
            key="compare_telemetry_complete",
        )

    with scope_col2:
        same_schema_only = st.checkbox(
            "Same schema/version cohort only",
            value=True,
            key="compare_same_schema_only",
        )

    with scope_col3:
        latest_n = st.number_input(
            "Latest N runs per experiment",
            min_value=1,
            max_value=25,
            value=3,
            step=1,
            key="compare_latest_n",
        )

    if use_telemetry_complete:
        compare_src = telemetry_complete_subset(compare_src)

    if same_schema_only:
        schema_filter_cols = [c for c in VERSION_COLS if c in compare_src.columns]

        if schema_filter_cols and not compare_src.empty:
            st.caption("Schema/version cohort filters")

            selected_versions = {}
            cols = st.columns(len(schema_filter_cols))

            for i, col in enumerate(schema_filter_cols):
                vals = sorted(
                    [
                        v for v in compare_src[col].dropna().astype("string").unique().tolist()
                        if str(v).strip()
                    ]
                )

                default_vals = vals[-1:] if vals else []

                selected_versions[col] = cols[i].multiselect(
                    col,
                    options=vals,
                    default=default_vals,
                    key=f"schema_filter_{col}",
                )

            for col, selected in selected_versions.items():
                if selected:
                    compare_src = compare_src[
                        compare_src[col].astype("string").isin(selected)
                    ].copy()

    compare_mode = st.radio(
        "Comparison run scope",
        options=["All pooled matching rows", "Latest N runs per experiment"],
        index=1,
        horizontal=True,
        key="comparison_run_scope",
    )

    if compare_mode == "Latest N runs per experiment":
        compare_src = latest_n_runs_per_experiment(compare_src, int(latest_n))

    st.caption(
        f"Comparison rows after scope filters: {len(compare_src):,} | "
        f"Unique runs: {compare_src['run_id'].nunique() if 'run_id' in compare_src.columns else '—'}"
    )
    
    experiment_summary_compare = make_experiment_metric_summary(compare_src)

    available_compare_suites = sorted(
        compare_src["suite_name"].dropna().astype("string").unique()
    )

    if available_compare_suites:
        default_compare_suite = available_compare_suites[0]
        if len(agg_suites) == 1 and agg_suites[0] in available_compare_suites:
            default_compare_suite = agg_suites[0]

        comparison_suite = st.selectbox(
            "Comparison suite",
            options=available_compare_suites,
            index=available_compare_suites.index(default_compare_suite),
            key="comparison_suite",
        )

        suite_compare_src = compare_src[
            compare_src["suite_name"].astype("string") == str(comparison_suite)
        ].copy()

        available_experiments = sorted(
            suite_compare_src["experiment_name"].dropna().astype("string").unique()
        )

        if len(available_experiments) >= 2:
            default_baseline = "temp0" if "temp0" in available_experiments else available_experiments[0]
            remaining = [x for x in available_experiments if x != default_baseline]
            default_comparison = "temp03" if "temp03" in remaining else remaining[0]

            ccmp1, ccmp2, ccmp3 = st.columns(3)

            baseline_experiment = ccmp1.selectbox(
                "Baseline experiment",
                options=available_experiments,
                index=available_experiments.index(default_baseline),
                key="baseline_experiment",
            )

            comparison_options = [x for x in available_experiments if x != baseline_experiment]
            comparison_experiment = ccmp2.selectbox(
                "Comparison experiment",
                options=comparison_options,
                index=comparison_options.index(default_comparison) if default_comparison in comparison_options else 0,
                key="comparison_experiment",
            )

            comparison_metric = ccmp3.selectbox(
                "Comparison metric",
                options=["checks_pass_rate", "overall_pass_rate"],
                index=0,
                key="comparison_metric",
            )
            
            st.caption(
                f"Note: comparison mode = {compare_mode}. "
                "Use latest-run mode when comparing recent suite changes."
            )
            delta_table = make_experiment_delta_table(
                experiment_summary_compare,
                suite_name=comparison_suite,
                baseline_experiment=baseline_experiment,
                comparison_experiment=comparison_experiment,
                metric_col=comparison_metric,
            )

            if not delta_table.empty:
                st.caption(
                    f"Delta = {comparison_experiment} - {baseline_experiment} "
                    f"within suite={comparison_suite}"
                )

                show_delta = delta_table.drop(columns=["abs_delta"], errors="ignore").copy()

                for col in ["baseline_pass_rate", "comparison_pass_rate"]:
                    if col in show_delta.columns:
                        show_delta[col] = show_delta[col].apply(
                            lambda x: f"{x:.2%}" if pd.notna(x) else "—"
                        )

                if "delta_pass_rate" in show_delta.columns:
                    show_delta["delta_pass_rate"] = show_delta["delta_pass_rate"].apply(
                        lambda x: f"{x:+.2%}" if pd.notna(x) else "—"
                    )
                # st.dataframe(show_delta)

                st.dataframe(show_delta, use_container_width=True)

                delta_heat = make_delta_heatmap(delta_table)
                if not delta_heat.empty:
                    figD, axD = plt.subplots(figsize=(8, max(3, 0.6 * len(delta_heat))))
                    annot_delta = format_delta_labels(delta_heat)

                    hmD = sns.heatmap(
                        delta_heat,
                        annot=annot_delta,
                        fmt="",
                        cmap="RdYlGn",
                        center=0.0,
                        vmin=-1.0,
                        vmax=1.0,
                        linewidths=0.5,
                        linecolor="white",
                        ax=axD,
                    )

                    cbar = hmD.collections[0].colorbar
                    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
                    cbar.set_ticklabels(["-100%", "-50%", "0%", "+50%", "+100%"])

                    axD.set_xlabel("Model")
                    axD.set_ylabel("Prompt")
                    axD.set_title(
                        f"{comparison_metric} delta by prompt × model\n"
                        f"[suite={comparison_suite} | {comparison_experiment} - {baseline_experiment}]"
                    )
                    st.pyplot(figD)
                else:
                    st.info("Not enough data to build delta heatmap.")
            else:
                st.info("No comparable prompt × model rows found for this suite/experiment pair.")
        else:
            st.info("Need at least two experiments within the selected suite to compare.")
    else:
        st.info("No suites available for comparison.")
    
    # -------------------------
    # Model delta summary (determinisitc)
    # ------------------------- 
    st.subheader("Deterministic Model Delta Summaries")

    model_delta_summary = make_model_delta_summary(delta_table)

    if not model_delta_summary.empty:
        show_mds = model_delta_summary.copy()

        for col in ["avg_delta", "avg_abs_delta", "best_delta", "worst_delta"]:
            show_mds[col] = show_mds[col].apply(lambda x: f"{x:+.2%}" if pd.notna(x) else "—")

        st.dataframe(show_mds, use_container_width=True)

        st.markdown("#### Summary Sentences")
        for _, row in model_delta_summary.iterrows():
            st.write(
                "- " + render_model_delta_sentence(
                    row,
                    baseline_experiment=baseline_experiment,
                    comparison_experiment=comparison_experiment,
                )
            )
    else:
        st.info("Not enough data to generate model delta summaries.")
    
    # model profile = dynamic change + static behavior
    behavior_summary = make_model_behavior_summary(compare_src)
    model_profile = make_model_profile_summary(
        model_delta_summary,
        behavior_summary,
        comparison_experiment=comparison_experiment,
    )
    model_profile["model_role"] = model_profile.apply(classify_model_role, axis=1)
    model_profile["consistency"] = model_profile["response_hash_stability_rate"].apply(classify_consistency)
    model_profile["adaptability"] = model_profile["avg_abs_delta"].apply(classify_adaptability)
    model_profile["repair_focus"] = model_profile.apply(infer_repair_focus, axis=1)
    
    st.subheader("Merged Model Profiles")

    st.dataframe(model_profile, use_container_width=True)

    for _, row in model_profile.iterrows():
        st.write(
            "- " + render_model_profile_sentence(
                row,
                baseline_experiment,
                comparison_experiment,
            )
        )
    
    # -------------------------
    # Model Profile Cards
    # -------------------------
    st.subheader("Model Profile Cards")

    if not model_profile.empty:
        for _, row in model_profile.iterrows():
            role = row.get("model_role", "mixed")
            failure_str = format_failure_mode(row.get("dominant_failure_type_v2", "unknown"))

            with st.container(border=True):
                st.markdown(f"### {row['model']}")

                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Role", role)
                c2.metric("Sensitivity", row.get("temperature_sensitivity", "unknown"))
                c3.metric("Direction", row.get("overall_direction", "unknown"))
                c4.metric(
                    "Hash Stability",
                    f"{row['response_hash_stability_rate']:.0%}"
                    if pd.notna(row.get("response_hash_stability_rate"))
                    else "—",
                )

                c5, c6, c7, c8 = st.columns(4)

                c5.metric(
                    "Avg Delta",
                    f"{row['avg_delta']:+.0%}" if pd.notna(row.get("avg_delta")) else "—",
                )
                c6.metric(
                    "Pass Rate",
                    f"{row['overall_pass_rate']:.0%}" if pd.notna(row.get("overall_pass_rate")) else "—",
                )
                c7.metric("Consistency", row.get("consistency", "unknown"))
                c8.metric("Adaptability", row.get("adaptability", "unknown"))
                
                repair_focus = row.get("repair_focus", ["minimal_repair"])
                if not isinstance(repair_focus, list):
                    repair_focus = [str(repair_focus)]
                
                st.markdown(
                    f"**Behavior** - Failure mode: {green_tag(failure_str)} - "
                    f"Semantic pattern: {green_tag(row.get('dominant_semantic_pattern', 'unknown'))}",
                    unsafe_allow_html=True,
                )

                st.markdown("**Repair Focus**")

                for strategy in repair_focus:
                    st.markdown(
                        f"{green_tag(strategy)} — {describe_repair_strategy(strategy)}",
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f"**Strength / Weakness** - "
                    f"Strongest relative area: {green_tag(row.get('best_prompt', '—'))} "
                    f"({row.get('best_delta', 0):+.0%}) - "
                    f"Weakest relative area: {green_tag(row.get('worst_prompt', '—'))} "
                    f"({row.get('worst_delta', 0):+.0%})",
                    unsafe_allow_html=True,
                )
                
                with st.expander("Repair prompt patches"):
                    for strategy in repair_focus:
                        st.markdown(f"**{strategy}**")
                        st.code(get_repair_prompt_patch(strategy), language="text")
                
    else:
        st.info("Not enough data to build model profile cards.")
        
    st.caption(
        f"Profile comparison: {comparison_experiment} minus {baseline_experiment} "
        f"| comparison mode: {compare_mode} "
        f"| latest N={int(latest_n) if compare_mode == 'Latest N runs per experiment' else 'all'}"
    )
    
    # -------------------------
    # Repair Matrix summary
    # -------------------------     
    st.subheader("Repair Matrix Summary")
    
    repair_matrix_path = repair_results_root / "aggregated" / "repair_matrix_summary.csv"
    repair_rec_path = repair_results_root / "aggregated" / "repair_strategy_recommendation.json"

    show_repair_debug = st.checkbox(
        "Show repair matrix debug paths",
        value=False,
        key="show_repair_matrix_debug",
    )

    if show_repair_debug:
        st.caption(f"Repair results root: `{repair_results_root}`")
        st.caption(f"Looking for matrix: `{repair_matrix_path}` | exists={repair_matrix_path.exists()}")
        st.caption(f"Looking for recommendation: `{repair_rec_path}` | exists={repair_rec_path.exists()}")

    repair_matrix_df = load_repair_matrix_summary(repair_results_root)
    repair_rec = load_repair_strategy_recommendation(repair_results_root)

    if repair_matrix_df.empty:
        st.info("No repair matrix summary found. Run summarize_repair_matrix.py first.")
    else:
        display_cols = [
            "rank",
            "repair_label",
            "repair_strategies",
            "repair_success",
            "used_claim_ratio_delta",
            "flagged_after",
            "missing_claim_refs_after",
            "unknown_claim_ids_after",
            "repair_score",
        ]

        st.dataframe(
            repair_matrix_df[[c for c in display_cols if c in repair_matrix_df.columns]],
            use_container_width=True,
        )
    
    st.subheader("Recommended Repair Strategy")

    if not repair_rec:
        st.info("No repair strategy recommendation found.")
    else:
        with st.container(border=True):
            st.markdown(f"### {repair_rec.get('repair_label', 'unknown')}")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Repair Score", f"{repair_rec.get('repair_score', 0):.1f}")
            c2.metric("Repair Success", str(repair_rec.get("repair_success", "unknown")))
            c3.metric("Coverage Δ", f"{repair_rec.get('used_claim_ratio_delta', 0):+.2f}")
            c4.metric("Flagged", repair_rec.get("flagged_after", "—"))

            st.markdown("**Strategies**")
            strategies = repair_rec.get("repair_strategies", "")
            if strategies:
                for s in [x.strip() for x in strategies.split(",") if x.strip()]:
                    st.markdown(f"- {s}")
            else:
                st.markdown("- baseline / no explicit strategy")

            st.caption(
                "Recommendation selected from repair matrix by score, excluding baseline when a successful non-baseline strategy is available."
            )
    
    
    # -------------------------
    # Model Behavior table
    # -------------------------   
    st.subheader("Model Behavioral Summary")

    model_summary = make_model_behavior_summary(compare_src)

    if not model_summary.empty:
        show = model_summary.copy()

        for col in ["overall_pass_rate", "checks_pass_rate", "response_hash_stability_rate"]:
            if col in show.columns:
                show[col] = show[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "—")

        for col in ["avg_sentence_count", "avg_sentence_length", "avg_words"]:
            if col in show.columns:
                show[col] = show[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")

        st.dataframe(show, use_container_width=True)
    else:
        st.info("Not enough data to build model behavioral summary.")
        
    
    mN = normalize_master(m)  # if you want filtered run health/hotspots

    run_health = make_worst_runs(mN)     # filtered view
    hotspots = make_hotspots(mN)         # filtered view

    st.subheader("Run Health (Filtered)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(mN))
    c2.metric("Pipeline Success Rate", f"{mN['is_ok'].mean():.2%}" if len(mN) else "—")
    c3.metric("Nonzero exit_code", f"{mN['has_nonzero_exit'].mean():.2%}" if len(mN) else "—")
    c4.metric("Non-ok failure_type", f"{mN['failure_type_is_bad'].mean():.2%}" if len(mN) else "—")

    st.subheader("Prompt × Model Summary (All Runs)")
    pm_summary_all = make_prompt_model_summary(m)
    
    if not pm_summary_all.empty:
        show = pm_summary_all.copy()
        for col in ["pipeline_success_rate", "checks_pass_rate", "overall_pass_rate"]:
            if col in show.columns:
                show[col] = show[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
        for col in ["avg_latency", "avg_wps"]:
            if col in show.columns:
                show[col] = show[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        st.dataframe(show)
    else:
        st.info("Not enough data to build prompt × model summary.")
        
    st.subheader("Failure Taxonomy v0.2")

    failure_summary = load_failure_summary()

    if failure_summary.empty:
        st.info("No failure_taxonomy_summary.csv found yet. Run aggregate_runs.py first.")
    else:
        fsum = failure_summary.copy()
        if "suite_macro_fred" in fsum["suite_name"].unique():
            st.caption("Tip: taxonomy data is currently most meaningful for suite_macro_fred.")

        for col in ["failure_count", "total_runs", "bucket_rate"]:
            if col in fsum.columns:
                fsum[col] = pd.to_numeric(fsum[col], errors="coerce")

        for col in ["suite_name", "prompt_id", "model", "failure_type_v2"]:
            if col in fsum.columns:
                fsum[col] = fsum[col].astype("string").fillna("")

        # Apply same aggregate filters
        fsum = fsum[
            fsum["suite_name"].isin(agg_suites)
            & fsum["prompt_id"].isin(agg_prompts)
            & fsum["model"].isin(agg_models)
        ].copy()

        # Optional toggle: exclude ok rows
        show_ok_bucket = st.checkbox("Include ok bucket", value=True, key="show_ok_bucket")
        if not show_ok_bucket:
            fsum = fsum[fsum["failure_type_v2"] != "ok"].copy()

        if fsum.empty:
            st.info("No failure taxonomy rows in current filter.")
        else:
            show_fsum = fsum.copy()
            if "bucket_rate" in show_fsum.columns:
                show_fsum["bucket_rate"] = show_fsum["bucket_rate"].apply(
                    lambda x: f"{x:.2%}" if pd.notna(x) else "—"
                )

            st.dataframe(
                show_fsum.sort_values(
                    ["prompt_id", "model", "failure_type_v2"]
                ).reset_index(drop=True),
                use_container_width=True,
            )

            chart_df = (
                fsum.groupby(["failure_type_v2", "model"], as_index=False)["bucket_rate"]
                .mean()
            )

            if not chart_df.empty:
                pivot_chart = chart_df.pivot(
                    index="failure_type_v2",
                    columns="model",
                    values="bucket_rate",
                ).fillna(0.0)

                st.markdown("#### Mean Bucket Rate by Failure Type")
                st.bar_chart(pivot_chart)
                
    st.subheader("Semantic Pattern Distribution (v0.3)")

    semantic_df = load_semantic_patterns()

    if semantic_df.empty:
        st.info("No semantic_pattern_summary.csv found yet.")
    else:
        s = semantic_df.copy()

        for col in ["suite_name", "prompt_id", "model", "semantic_pattern"]:
            if col in s.columns:
                s[col] = s[col].astype("string").fillna("")

        for col in ["pattern_count", "total_runs", "pattern_rate"]:
            if col in s.columns:
                s[col] = pd.to_numeric(s[col], errors="coerce")

        # Apply same filters
        s = s[
            s["suite_name"].isin(agg_suites)
            & s["prompt_id"].isin(agg_prompts)
            & s["model"].isin(agg_models)
        ].copy()

        if s.empty:
            st.info("No semantic pattern rows in current filter.")
        else:
            # Table view
            show_s = s.copy()
            show_s["pattern_rate"] = show_s["pattern_rate"].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else "—"
            )

            st.dataframe(
                show_s.sort_values(
                    ["prompt_id", "model", "semantic_pattern"]
                ),
                use_container_width=True,
            )

            # 🔥 Chart: model × semantic pattern
            st.markdown("#### Pattern Distribution by Model")

            chart_df = (
                s.groupby(["model", "semantic_pattern"], as_index=False)["pattern_rate"]
                .mean()
            )

            pivot_chart = chart_df.pivot(
                index="semantic_pattern",
                columns="model",
                values="pattern_rate",
            ).fillna(0.0)

            st.bar_chart(pivot_chart)

    telem_src = m.copy()
    telem_src = telem_src[telem_src["task_type"].astype("string").str.strip() != ""].copy()
    telem_src = telem_src[telem_src["domain"].astype("string").str.strip() != ""].copy()

    if not telem_src.empty:
        telem_summary = (
            telem_src.groupby(["task_type", "domain"], as_index=False)
            .agg(
                rows=("task_type", "size"),
                avg_sentence_count=("sentence_count", "mean"),
                avg_sentence_length=("avg_sentence_length_words", "mean"),
                constraint_failure_rate=("failure_type", lambda s: (s == "constraint_failure").mean()),
                pipeline_success_rate=("ok", "mean"),
            )
            .sort_values(["task_type", "domain"])
        )
        st.subheader("Telemetry Summary by Task Type / Domain")
        st.dataframe(telem_summary, use_container_width=True)
        
    st.subheader("Constraint Failure Rate by Task Type")

    cf_src = m[m["task_type"].astype("string").str.strip() != ""].copy()

    cf = (
        cf_src.groupby("task_type", as_index=False)
        .agg(constraint_failure_rate=("failure_type", lambda s: (s == "constraint_failure").mean()))
        .sort_values("constraint_failure_rate", ascending=False)
    )

    if not cf.empty:
        st.bar_chart(cf.set_index("task_type"))
    else:
        st.info("No telemetry-tagged task types in current filter.")

    st.subheader("Prompt × Model Pass-Rate Heatmap (All Runs)")
    heat_value_col_all = st.selectbox(
        "Heatmap metric (All Runs)",
        ["overall_pass_rate", "checks_pass_rate", "pipeline_success_rate"],
        index=0,
        key="heatmap_metric_all",
    )

    heat_all = make_passrate_heatmap(pm_summary_all, heat_value_col_all)
    if not heat_all.empty:
        figH2, axH2 = plt.subplots(figsize=(8, max(3, 0.6 * len(heat_all))))
        annot_run = format_heatmap_labels(heat_all)

        hm = sns.heatmap(
            heat_all,
            annot=annot_run,
            fmt="",
            cmap="YlGn",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="white",
            ax=axH2,
        )

        cbar = hm.collections[0].colorbar
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
        
        axH2.set_xlabel("Model")
        axH2.set_ylabel("Prompt")
        axH2.set_title(
            f"{heat_value_col_all} by prompt × model\n"
            f"[suite={suite_label} | experiment={experiment_label}]"
        )
        st.pyplot(figH2)
    else:
        st.info("Not enough data to build pass-rate heatmap.")

    st.subheader("Worst Runs")
    st.dataframe(run_health.head(20))

    st.subheader("Top Failure Hotspots (prompt × model)")
    st.dataframe(hotspots.head(30))

    if ok_only == "Only ok":
        m = m[m["ok"] == 1]
    elif ok_only == "Only failures":
        m = m[m["ok"] == 0]

    # -------------------------
    # Lab growth chart
    # -------------------------  
    st.subheader("Lab Growth Over Time")
    metric_choice = st.selectbox(
        "Growth metric",
        ["Cumulative rows", "Cumulative unique runs", "Daily rows", "Daily unique runs"],
        index=0,
        key="growth_metric",
    )
    ycol = {
        "Cumulative rows": "cum_rows",
        "Cumulative unique runs": "cum_unique_runs",
        "Daily rows": "rows",
        "Daily unique runs": "unique_runs",
    }[metric_choice]

    figG, axG = plt.subplots()
    axG.plot(pd.to_datetime(daily["day"]), daily[ycol])
    axG.set_xlabel("Date")
    axG.set_ylabel(metric_choice)
    axG.set_title(metric_choice)
    axG.tick_params(axis="x", rotation=30)
    st.pyplot(figG)

    with st.expander("Growth table (daily)", expanded=False):
        st.dataframe(daily)

    # All Runs KPI block
    pipeline_rate = mean_boolish(m, "ok")
    checks_df = checked_subset(m)
    checks_rate = mean_boolish(checks_df, "checks_ok")
    overall_rate = mean_boolish(m, "overall_ok")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows (filtered)", len(m))
    col2.metric("Pipeline Success Rate", f"{pipeline_rate:.2%}" if pipeline_rate is not None else "—")
    col3.metric("Checks Pass Rate", f"{checks_rate:.2%}" if checks_rate is not None else "—")
    col4.metric("Overall Pass Rate", f"{overall_rate:.2%}" if overall_rate is not None else "—")
    
    col5, col6 = st.columns(2)
    col5.metric("Avg Latency (s)", f"{m['elapsed_s'].mean():.2f}" if len(m) else "—")
    col6.metric("Avg Words/sec", f"{m['words_per_s'].mean():.2f}" if len(m) else "—")

    st.subheader("Latency by Model (All Runs)")
    figA, axA = plt.subplots()
    sns.boxplot(data=m, x="model", y="elapsed_s", ax=axA)
    st.pyplot(figA)

    st.subheader("Words/sec by Model (All Runs)")
    figB, axB = plt.subplots()
    sns.boxplot(data=m, x="model", y="words_per_s", ax=axB)
    st.pyplot(figB)

    st.subheader("Failure Type Breakdown (All Runs)")
    fails = m[m["failure_type"] != "ok"]
    st.bar_chart(fails["failure_type"].value_counts())
    # st.bar_chart(m["failure_type"].value_counts())

with tab_audit:
    st.subheader("Audit Analytics")
    
    scope_options = get_scope_options()

    sc1, sc2 = st.columns([1, 2])

    with sc1:
        selected_scope_type = st.selectbox(
            "Artifact source",
            options=["runs", "archive"],
            key="scope_type",
        )

    available_scopes = scope_options.get(selected_scope_type, [])

    with sc2:
        if available_scopes:
            selected_scope_name = st.selectbox(
                "Scope folder",
                options=available_scopes,
                key="scope_name",
            )
        else:
            selected_scope_name = None
            st.info(f"No folders found under {selected_scope_type}.")
            
    if selected_scope_name is not None:
        scope_paths = resolve_scope_paths(selected_scope_type, selected_scope_name)
    else:
        scope_paths = resolve_scope_paths("legacy", "")
        
    st.caption(f"Using scope: {selected_scope_type} / {selected_scope_name}")

    audit_items = load_audit_items(str(scope_paths["aggregated_root"] / "audit_items.csv"))
    audit_summary = load_audit_summary(str(scope_paths["aggregated_root"] / "audit_summary.json"))
    claim_coverage = load_claim_coverage(str(scope_paths["aggregated_root"] / "claim_coverage.csv"))
    claim_coverage_summary = load_claim_coverage_summary(str(scope_paths["aggregated_root"] / "claim_coverage_summary.json"))

    if audit_items.empty:
        st.info("No audit_items.csv found yet. Run summarize_audits.py first.")
    else:
        # -------------------------
        # Normalize audit fields
        # -------------------------
        a = audit_items.copy()

        for col in [
            "claim_id_overlap_ratio",
            "claim_id_overlap_count",
            "n_claim_ids",
            "n_matched_claim_ids",
            "n_unknown_claim_ids",
            "n_extra_matched_claim_ids",
            "n_unused_cited_claim_ids",
            "fidelity_score_artifact",
        ]:
            if col in a.columns:
                a[col] = pd.to_numeric(a[col], errors="coerce")

        for col in [
            "audit_status",
            "issue_type",
            "suite_name",
            "model",
            "prompt_id",
            "section",
            "artifact_name",
            "bullet_text",
            "notes",
        ]:
            if col in a.columns:
                a[col] = a[col].astype("string").fillna("")
                
        # Derived diagnostics
        a["has_claim_refs"] = a["n_claim_ids"].fillna(0) > 0
        a["is_meta_section"] = a["section"].str.lower().isin(["cautions", "meta", "meta_caution"])

        a["zero_overlap_empirical"] = (
            (a["audit_status"] == "supported")
            & (a["claim_id_overlap_ratio"].fillna(0) == 0)
            & (a["has_claim_refs"])
            & (~a["is_meta_section"])
        )
        
        for col in [
            "has_mixed_directions",
            "has_mixed_models",
            "has_mixed_prompts",
            "has_mixed_strengths",
        ]:
            if col in a.columns:
                a[col] = a[col].astype(str).str.lower().isin(["true", "1", "yes"])
                

        # -------------------------
        # Sidebar-style filters (in-tab)
        # -------------------------
        st.markdown("### Filters")
        f1, f2, f3, f4 = st.columns(4)

        selected_statuses = f1.multiselect(
            "Audit status",
            options=sorted(a["audit_status"].dropna().unique().tolist()),
            default=sorted(a["audit_status"].dropna().unique().tolist()),
            key="audit_status_filter",
        )

        selected_suites = f2.multiselect(
            "Suites",
            options=sorted(a["suite_name"].dropna().unique().tolist()),
            default=sorted(a["suite_name"].dropna().unique().tolist()),
            key="audit_suite_filter",
        )

        selected_models = f3.multiselect(
            "Models",
            options=sorted(a["model"].dropna().unique().tolist()),
            default=sorted(a["model"].dropna().unique().tolist()),
            key="audit_model_filter",
        )

        selected_prompts = f4.multiselect(
            "Prompts",
            options=sorted(a["prompt_id"].dropna().unique().tolist()),
            default=sorted(a["prompt_id"].dropna().unique().tolist()),
            key="audit_prompt_filter",
        )
        
        selected_support_regimes = st.multiselect(
            "Support regimes",
            options=sorted(a["strict_ref_support_regime"].dropna().unique().tolist()),
            default=sorted(a["strict_ref_support_regime"].dropna().unique().tolist()),
            key="audit_support_regime_filter",
        )

        a_filt = a[
            a["audit_status"].isin(selected_statuses)
            & a["suite_name"].isin(selected_suites)
            & a["model"].isin(selected_models)
            & a["prompt_id"].isin(selected_prompts)
            & a["strict_ref_support_regime"].isin(selected_support_regimes)
        ].copy()

        # -------------------------
        # KPIs
        # -------------------------
        status_counts = a_filt["audit_status"].value_counts()

        total_bullets = len(a_filt)
        supported_count = int(status_counts.get("supported", 0))
        flagged_count = int(status_counts.get("flagged", 0))
        meta_count = int(status_counts.get("meta_caution", 0))

        mean_overlap_ratio = (
            a_filt["claim_id_overlap_ratio"].dropna().mean()
            if "claim_id_overlap_ratio" in a_filt.columns and len(a_filt)
            else None
        )
        bullets_with_extra = int(
            (a_filt["n_extra_matched_claim_ids"].fillna(0) > 0).sum()
        ) if "n_extra_matched_claim_ids" in a_filt.columns else 0
        bullets_with_unused = int(
            (a_filt["n_unused_cited_claim_ids"].fillna(0) > 0).sum()
        ) if "n_unused_cited_claim_ids" in a_filt.columns else 0

        # Support-regime KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total bullets", total_bullets)
        k2.metric("Supported", supported_count)
        k3.metric("Flagged", flagged_count)
        k4.metric("Meta cautions", meta_count)
        
        regime_counts = a_filt["strict_ref_support_regime"].value_counts()

        trace_supported_count = int(regime_counts.get("trace_supported", 0))
        partial_trace_count = int(regime_counts.get("partially_trace_supported", 0))
        heuristic_only_count = int(regime_counts.get("heuristic_only_supported", 0))
        no_cited_refs_count = int(regime_counts.get("no_cited_refs", 0))

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Trace-supported", trace_supported_count)
        r2.metric("Partially trace-supported", partial_trace_count)
        r3.metric("Heuristic-only supported", heuristic_only_count)
        r4.metric("No cited refs", no_cited_refs_count)

        k5, k6, k7 = st.columns(3)
        k5.metric("Mean overlap ratio", fmt_pct(mean_overlap_ratio) if mean_overlap_ratio is not None else "—")
        k6.metric("Bullets w/ extra matched claims", bullets_with_extra)
        k7.metric("Bullets w/ unused cited claims", bullets_with_unused)

        if audit_summary:
            st.caption(
                f"Support mode: {audit_summary.get('support_mode', '—')}"
            )

        # -------------------------
        # Status + issue breakdown
        # -------------------------
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### Status Breakdown")
            st.bar_chart(status_counts)

        with c2:
            st.markdown("### Issue Type Breakdown")
            flagged_only = a_filt[a_filt["audit_status"] == "flagged"].copy()
            if not flagged_only.empty and "issue_type" in flagged_only.columns:
                issue_counts = (
                    flagged_only["issue_type"]
                    .replace("", pd.NA)
                    .dropna()
                    .value_counts()
                )
                if len(issue_counts):
                    st.bar_chart(issue_counts)
                else:
                    st.info("No flagged issue types in current filter.")
            else:
                st.info("No flagged bullets in current filter.")

        # Support-regime bar chart
        st.markdown("### Support Regime Breakdown")
        if "strict_ref_support_regime" in a_filt.columns:
            regime_counts = a_filt["strict_ref_support_regime"].replace("", pd.NA).dropna().value_counts()
            if len(regime_counts):
                st.bar_chart(regime_counts)
            else:
                st.info("No support-regime data in current filter.")

        # -------------------------
        # Grouped tables
        # -------------------------
        st.markdown("### Flagged Counts by Suite / Model / Prompt")

        g1, g2, g3 = st.columns(3)

        with g1:
            flagged_by_suite = (
                a_filt[a_filt["audit_status"] == "flagged"]
                .groupby("suite_name", as_index=False)
                .size()
                .rename(columns={"size": "flagged_count"})
                .sort_values("flagged_count", ascending=False)
            )
            st.dataframe(flagged_by_suite, use_container_width=True)

        with g2:
            flagged_by_model = (
                a_filt[a_filt["audit_status"] == "flagged"]
                .groupby("model", as_index=False)
                .size()
                .rename(columns={"size": "flagged_count"})
                .sort_values("flagged_count", ascending=False)
            )
            st.dataframe(flagged_by_model, use_container_width=True)

        with g3:
            flagged_by_prompt = (
                a_filt[a_filt["audit_status"] == "flagged"]
                .groupby("prompt_id", as_index=False)
                .size()
                .rename(columns={"size": "flagged_count"})
                .sort_values("flagged_count", ascending=False)
            )
            st.dataframe(flagged_by_prompt, use_container_width=True)

        # -------------------------
        # Support-discipline diagnostics
        # -------------------------
        st.markdown("### Support Discipline Diagnostics")
        md1, md2, md3, md4 = st.columns(4)
        md1.metric("Mixed-direction bullets", int(a_filt["has_mixed_directions"].sum()) if "has_mixed_directions" in a_filt.columns else 0)
        md2.metric("Mixed-model bullets", int(a_filt["has_mixed_models"].sum()) if "has_mixed_models" in a_filt.columns else 0)
        md3.metric("Mixed-prompt bullets", int(a_filt["has_mixed_prompts"].sum()) if "has_mixed_prompts" in a_filt.columns else 0)
        md4.metric("Mixed-strength bullets", int(a_filt["has_mixed_strengths"].sum()) if "has_mixed_strengths" in a_filt.columns else 0)
        
        
        st.markdown("### Heuristic-Only Supported Bullets")
        hos = a_filt[a_filt["strict_ref_support_regime"] == "heuristic_only_supported"].copy()

        hos_cols = [
            "suite_name",
            "model",
            "prompt_id",
            "section",
            "bullet_text",
            "claim_ids",
            "matched_claim_ids",
            "unused_cited_claim_ids",
            "extra_matched_claim_ids",
            "strict_ref_overlap_ratio",
            "claim_id_overlap_ratio",
        ]
        hos_cols = [c for c in hos_cols if c in hos.columns]

        if len(hos):
            st.dataframe(hos[hos_cols], use_container_width=True)
        else:
            st.info("No heuristic-only supported bullets in current filter.")
        
        
        st.markdown("#### Zero-Overlap Empirical Bullets")
        zoe = a_filt[a_filt["zero_overlap_empirical"]].copy()

        zoe_cols = [
            "suite_name",
            "model",
            "prompt_id",
            "section",
            "bullet_text",
            "claim_ids",
            "matched_claim_ids",
            "unused_cited_claim_ids",
            "extra_matched_claim_ids",
            "claim_id_overlap_ratio",
        ]
        zoe_cols = [c for c in zoe_cols if c in zoe.columns]

        st.dataframe(zoe[zoe_cols], use_container_width=True)
        
        
        st.markdown("#### Mixed-Direction Bullets")
        mdb = a_filt[a_filt["has_mixed_directions"]].copy()

        mdb_cols = [
            "suite_name",
            "model",
            "prompt_id",
            "section",
            "bullet_text",
            "claim_ids",
            "direction_set",
            "linked_models",
            "linked_prompt_ids",
            "linked_claim_strengths",
        ]
        mdb_cols = [c for c in mdb_cols if c in mdb.columns]

        if len(mdb):
            st.dataframe(mdb[mdb_cols], use_container_width=True)
        else:
            st.info("No mixed-direction bullets in current filter.")


        diag1, diag2 = st.columns(2)

        with diag1:
            st.markdown("#### Top Extra-Matched Bullets")
            extra_cols = [
                "suite_name",
                "model",
                "prompt_id",
                "section",
                "bullet_text",
                "claim_id_overlap_ratio",
                "n_claim_ids",
                "n_matched_claim_ids",
                "n_extra_matched_claim_ids",
                "extra_matched_claim_ids",
            ]
            extra_cols = [c for c in extra_cols if c in a_filt.columns]

            top_extra = (
                a_filt.sort_values(
                    ["n_extra_matched_claim_ids", "claim_id_overlap_ratio"],
                    ascending=[False, True],
                )[extra_cols]
                .head(20)
            )
            st.dataframe(top_extra, use_container_width=True)

        with diag2:
            st.markdown("#### Zero-Overlap Bullets")
            zero_cols = [
                "suite_name",
                "model",
                "prompt_id",
                "section",
                "bullet_text",
                "claim_ids",
                "matched_claim_ids",
                "unused_cited_claim_ids",
                "extra_matched_claim_ids",
                "claim_id_overlap_ratio",
            ]
            zero_cols = [c for c in zero_cols if c in a_filt.columns]

            zero_overlap = a_filt[a_filt["claim_id_overlap_ratio"].fillna(0) == 0].copy()
            st.dataframe(
                zero_overlap[zero_cols].head(20),
                use_container_width=True,
            )

        # -------------------------
        # Artifact-level summary
        # -------------------------
        st.markdown("### Artifact-Level Summary")

        artifact_cols = [
            "artifact_name",
            "suite_name",
            "fidelity_score_artifact",
            "audit_status",
            "n_extra_matched_claim_ids",
            "n_unused_cited_claim_ids",
        ]
        artifact_cols = [c for c in artifact_cols if c in a_filt.columns]

        artifact_summary = (
            a_filt.groupby(["artifact_name", "suite_name"], as_index=False)
            .agg(
                bullets=("artifact_name", "size"),
                fidelity_score=("fidelity_score_artifact", "max"),
                supported=("audit_status", lambda s: (s == "supported").sum()),
                flagged=("audit_status", lambda s: (s == "flagged").sum()),
                meta_caution=("audit_status", lambda s: (s == "meta_caution").sum()),
                trace_supported=("strict_ref_support_regime", lambda s: (s == "trace_supported").sum()),
                partially_trace_supported=("strict_ref_support_regime", lambda s: (s == "partially_trace_supported").sum()),
                heuristic_only_supported=("strict_ref_support_regime", lambda s: (s == "heuristic_only_supported").sum()),
                no_cited_refs=("strict_ref_support_regime", lambda s: (s == "no_cited_refs").sum()),
                mean_overlap_ratio=("claim_id_overlap_ratio", "mean"),
                mean_strict_ref_overlap_ratio=("strict_ref_overlap_ratio", "mean"),
                total_extra_matched=("n_extra_matched_claim_ids", "sum"),
            )
            .sort_values(
                ["heuristic_only_supported", "total_extra_matched", "mean_strict_ref_overlap_ratio"],
                ascending=[False, False, True],
            )
        )
        st.dataframe(artifact_summary, use_container_width=True)

        # -------------------------
        # Full audit row table
        # -------------------------
        st.markdown("### Full Audit Item Table")
        st.dataframe(a_filt, use_container_width=True)
        
        
        # -------------------------
        # Claim Coverage
        # -------------------------
        st.markdown("---")
        st.subheader("Claim Coverage")

        if claim_coverage.empty:
            st.info("No claim_coverage.csv found yet. Run summarize_audits.py first.")
        else:
            cc = claim_coverage.copy()

            for col in ["times_cited", "n_sections_used"]:
                if col in cc.columns:
                    cc[col] = pd.to_numeric(cc[col], errors="coerce")

            for col in [
                "artifact_name",
                "suite_name",
                "model",
                "prompt_id",
                "claim_type",
                "claim_strength",
                "label",
                "claim_id",
                "sections_used",
            ]:
                if col in cc.columns:
                    cc[col] = cc[col].astype("string").fillna("")

            if "used_in_narrative" in cc.columns:
                cc["used_in_narrative"] = cc["used_in_narrative"].astype(str).str.lower().isin(
                    ["true", "1", "yes"]
                )

            # Filter to current suite/model/prompt selections when possible
            cc_filt = cc[
                cc["suite_name"].isin(selected_suites)
                & cc["model"].isin(selected_models)
                & cc["prompt_id"].isin(selected_prompts)
            ].copy()

            # -------------------------
            # Claim Coverage KPIs
            # -------------------------
            selected_claims_count = len(cc_filt)
            used_claims_count = int(cc_filt["used_in_narrative"].sum()) if "used_in_narrative" in cc_filt.columns else 0
            unused_claims_count = selected_claims_count - used_claims_count
            used_claim_ratio = (
                used_claims_count / selected_claims_count if selected_claims_count > 0 else 0.0
            )

            ck1, ck2, ck3, ck4 = st.columns(4)
            ck1.metric("Selected claims", selected_claims_count)
            ck2.metric("Used claims", used_claims_count)
            ck3.metric("Unused claims", unused_claims_count)
            ck4.metric("Used-claim ratio", fmt_pct(used_claim_ratio))

            # -------------------------
            # Artifact-level coverage summary
            # -------------------------
            st.markdown("### Claim Coverage by Artifact")

            artifact_coverage = (
                cc_filt.groupby(["artifact_name", "suite_name"], as_index=False)
                .agg(
                    selected_claim_count=("claim_id", "size"),
                    used_claim_count=("used_in_narrative", "sum"),
                )
            )

            if not artifact_coverage.empty:
                artifact_coverage["unused_claim_count"] = (
                    artifact_coverage["selected_claim_count"] - artifact_coverage["used_claim_count"]
                )
                artifact_coverage["used_claim_ratio"] = (
                    artifact_coverage["used_claim_count"] / artifact_coverage["selected_claim_count"]
                )
                artifact_coverage = artifact_coverage.sort_values(
                    ["used_claim_ratio", "unused_claim_count"],
                    ascending=[True, False],
                )
                st.dataframe(artifact_coverage, use_container_width=True)
            else:
                st.info("No artifact-level claim coverage available in current filter.")
            
            # Used-claim-ratio bar chart by artifact
            st.markdown("### Used-Claim Ratio by Artifact")

            if not artifact_coverage.empty:
                chart_df = artifact_coverage.set_index("artifact_name")[["used_claim_ratio"]]
                st.bar_chart(chart_df)
                

            # -------------------------
            # Unused claims
            # -------------------------
            st.markdown("### Unused Selected Claims")

            unused_claims = cc_filt[~cc_filt["used_in_narrative"]].copy()
            unused_cols = [
                "suite_name",
                "artifact_name",
                "claim_id",
                "model",
                "prompt_id",
                "claim_type",
                "claim_strength",
                "label",
            ]
            unused_cols = [c for c in unused_cols if c in unused_claims.columns]

            if len(unused_claims):
                strength_order = {"strong": 3, "medium": 2, "weak": 1}
                unused_claims["_strength_rank"] = (
                    unused_claims["claim_strength"].astype("string").str.lower().map(strength_order).fillna(0)
                )

                st.dataframe(
                    unused_claims[unused_cols + ["_strength_rank"]]
                    .sort_values(
                        ["suite_name", "_strength_rank", "model", "prompt_id"],
                        ascending=[True, False, True, True],
                    )
                    .drop(columns=["_strength_rank"]),
                    use_container_width=True,
                )
            else:
                st.info("No unused selected claims in current filter.")

            # -------------------------
            # Reused claims
            # -------------------------
            st.markdown("### Reused Claims")

            reused_claims = cc_filt[cc_filt["times_cited"].fillna(0) > 1].copy()
            reused_claims["reuse_intensity"] = (
                reused_claims["times_cited"].fillna(0)
                / reused_claims["n_sections_used"].replace(0, pd.NA)
            )
            reused_cols = [
                "suite_name",
                "artifact_name",
                "claim_id",
                "model",
                "prompt_id",
                "times_cited",
                "n_sections_used",
                "reuse_intensity",
                "sections_used",
                "claim_type",
                "claim_strength",
            ]
            reused_cols = [c for c in reused_cols if c in reused_claims.columns]

            if len(reused_claims):
                st.dataframe(
                    reused_claims[reused_cols].sort_values(
                        ["reuse_intensity", "times_cited", "suite_name", "model"],
                        ascending=[False, False, True, True],
                    ),
                    use_container_width=True,
                )
            else:
                st.info("No multiply cited claims in current filter.")

with tab_trace:
    st.subheader("Narrative Traceability")
    
    scope_options = get_scope_options()
    if "scope_type" not in st.session_state:
        st.session_state["scope_type"] = "runs"
    if "scope_name" not in st.session_state:
        run_scopes = scope_options.get("runs", [])
        st.session_state["scope_name"] = run_scopes[0] if run_scopes else None
    
    selected_scope_type = st.session_state.get("scope_type", "runs")
    selected_scope_name = st.session_state.get("scope_name")

    if selected_scope_name is not None:
        scope_paths = resolve_scope_paths(selected_scope_type, selected_scope_name)
    else:
        scope_paths = resolve_scope_paths("legacy", "")

    parsed_files = find_narrative_files(scope_paths["narratives_root"], "__parsed_narrative.json")
    
    st.caption(f"Using scope: {selected_scope_type} / {selected_scope_name}")

    if not parsed_files:
        st.info("No parsed narrative artifacts found yet.")
    else:
        selected_parsed_path = st.selectbox(
            "Select parsed narrative artifact",
            options=[str(p) for p in parsed_files],
            format_func=lambda x: Path(x).name,
            key="parsed_narrative_select",
        )

        parsed_payload = load_json_file(Path(selected_parsed_path))

        st.caption(
            f"Suite: {parsed_payload.get('suite_name', '—')} | "
            f"Metric: {parsed_payload.get('metric', '—')} | "
            f"Baseline: {parsed_payload.get('baseline_experiment', '—')}"
        )

        summary = parsed_payload.get("summary", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Narrative items", summary.get("n_items", "—"))
        c2.metric("With claim refs", summary.get("n_items_with_claim_refs", "—"))
        c3.metric("Missing claim refs", summary.get("n_items_missing_claim_refs", "—"))
        c4.metric("Unknown claim IDs", summary.get("n_unknown_claim_ids", "—"))

        items = parsed_payload.get("items", [])

        if not items:
            st.info("No parsed narrative items found.")
        else:
            item_labels = [
                f"{i['item_id']}. [{i['section']}] {i['clean_text'][:100]}"
                for i in items
            ]

            selected_idx = st.selectbox(
                "Select narrative bullet",
                options=list(range(len(items))),
                format_func=lambda i: item_labels[i],
                key="trace_item_select",
            )

            item = items[selected_idx]

            left, right = st.columns([1.1, 1.4])

            with left:
                st.markdown("### Narrative Bullet")
                st.write(f"**Section:** {item.get('section', '—')}")
                st.write(f"**Clean text:** {item.get('clean_text', '')}")
                st.write(f"**Raw text:** {item.get('raw_text', '')}")
                st.write(f"**Claim IDs:**")
                st.write("**Direction set:**", " | ".join(item.get("direction_set", [])) if item.get("direction_set") else "—")
                st.write("**Mixed directions:**", item.get("has_mixed_directions", False))
                st.write("**Linked models:**", " | ".join(item.get("linked_models", [])) if item.get("linked_models") else "—")
                st.write("**Mixed models:**", item.get("has_mixed_models", False))
                st.write("**Linked prompts:**", " | ".join(item.get("linked_prompt_ids", [])) if item.get("linked_prompt_ids") else "—")
                st.write("**Mixed prompts:**", item.get("has_mixed_prompts", False))
                st.code("\n".join(item.get("claim_ids", [])) if item.get("claim_ids") else "—")

                if item.get("unknown_claim_ids"):
                    st.warning(f"Unknown claim IDs: {item['unknown_claim_ids']}")

            with right:
                st.markdown("### Linked Claims")
                linked_claims = item.get("linked_claims", [])

                if not linked_claims:
                    st.info("No linked claims for this bullet.")
                else:
                    for claim in linked_claims:
                        with st.expander(claim.get("claim_id", "claim"), expanded=False):
                            cc1, cc2 = st.columns(2)
                            cc1.write(f"**prompt_id:** {claim.get('prompt_id', '—')}")
                            cc2.write(f"**model:** {claim.get('model', '—')}")

                            cc3, cc4 = st.columns(2)
                            cc3.write(f"**baseline_experiment:** {claim.get('baseline_experiment', '—')}")
                            cc4.write(f"**comparison_experiment:** {claim.get('comparison_experiment', '—')}")

                            cc5, cc6, cc7 = st.columns(3)
                            cc5.write(f"**baseline_value:** {claim.get('baseline_value', '—')}")
                            cc6.write(f"**comparison_value:** {claim.get('comparison_value', '—')}")
                            cc7.write(f"**delta_value:** {claim.get('delta_value', '—')}")

                            cc8, cc9, cc10 = st.columns(3)
                            cc8.write(f"**label:** {claim.get('label', '—')}")
                            cc9.write(f"**claim_strength:** {claim.get('claim_strength', '—')}")
                            cc10.write(f"**claim_type:** {claim.get('claim_type', '—')}")