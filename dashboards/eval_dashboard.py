import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import re
from typing import Any

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

RUNS_ROOT = REPO_ROOT / "benchmarks" / "results" / "runs"
ARCHIVE_ROOT = REPO_ROOT / "benchmarks" / "results" / "archive"
LEGACY_NARRATIVES_ROOT = REPO_ROOT / "benchmarks" / "results" / "narratives"
LEGACY_AGG_ROOT = REPO_ROOT / "benchmarks" / "results" / "aggregated"


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
        how="outer",
    )

    merged["baseline_pass_rate"] = pd.to_numeric(
        merged["baseline_pass_rate"], errors="coerce"
    ).fillna(0.0)
    merged["comparison_pass_rate"] = pd.to_numeric(
        merged["comparison_pass_rate"], errors="coerce"
    ).fillna(0.0)
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
def load_audit_items(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
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
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_claim_coverage_summary(path: str) -> dict[str, Any]:
    p = Path(path)
    if p.exists():
        return load_json_file(p)
    return {}

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

    # Comparison should operate off master, constrained by current model/prompt filters,
    # but not accidentally constrained by Agg: Experiments multiselect.
    compare_src = master.copy()
    compare_src = compare_src[
        compare_src["model"].isin(agg_models)
        & compare_src["prompt_id"].isin(agg_prompts)
        & compare_src["suite_name"].astype("string").isin(agg_suites)
    ].copy()
    
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