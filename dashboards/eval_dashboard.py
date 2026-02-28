import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(layout="wide")
st.title("LLM Evaluation Dashboard (Phase 1)")

# -------------------------
# Auto-load latest results (benchmarks/results_*)
# -------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]          # /home/joe/ai-lab
# RESULTS_ROOT = REPO_ROOT / "benchmarks"                 # /home/joe/ai-lab/benchmarks
RAW_RUNS_ROOT = REPO_ROOT / "benchmarks" / "results" / "raw_runs"
AGG_ROOT = REPO_ROOT / "benchmarks" / "results" / "aggregated"
AGG_PARQUET = AGG_ROOT / "runs_master.parquet"
AGG_CSV = AGG_ROOT / "runs_master.csv"

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

runs = find_result_folders(RAW_RUNS_ROOT)

# Patch: filter runs to “compatible runs only”
REQUIRED_COLS = {"model", "prompt_id", "elapsed_s", "words", "failure_type", "words_per_s", "ok"}

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

@st.cache_data(show_spinner=False)
def load_agg() -> pd.DataFrame:
    if AGG_PARQUET.exists():
        return pd.read_parquet(AGG_PARQUET)
    if AGG_CSV.exists():
        return pd.read_csv(AGG_CSV)
    return pd.DataFrame()


# -------------------------
# Tabs
# -------------------------
tab_run, tab_agg = st.tabs(["Single Run", "All Runs (runs_master)"])

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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", len(df_filtered))
    col2.metric("Success Rate", f"{df_filtered['ok'].mean():.2%}" if len(df_filtered) else "—")
    col3.metric("Avg Latency (s)", f"{df_filtered['elapsed_s'].mean():.2f}" if len(df_filtered) else "—")
    col4.metric("Avg Words", f"{df_filtered['words'].mean():.1f}" if len(df_filtered) else "—")

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

    if master.empty:
        st.error(f"No runs_master found. Expected: {AGG_PARQUET} or {AGG_CSV}")
        st.stop()
    
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
    for col in ["model", "prompt_id", "failure_type", "ok", "elapsed_s", "words", "words_per_s"]:
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

    ok_only = st.sidebar.selectbox("Agg: Outcomes", ["All", "Only ok", "Only failures"], index=0)

    m = master.copy()
    m = m[m["model"].isin(agg_models) & m["prompt_id"].isin(agg_prompts)]
    
    masterN = normalize_master(master) # not currently used
    mN = normalize_master(m)  # if you want filtered run health/hotspots

    run_health = make_worst_runs(mN)     # filtered view
    hotspots = make_hotspots(mN)         # filtered view

    st.subheader("Run Health (Filtered)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(mN))
    c2.metric("OK rate", f"{mN['is_ok'].mean():.2%}" if len(mN) else "—")
    c3.metric("Nonzero exit_code", f"{mN['has_nonzero_exit'].mean():.2%}" if len(mN) else "—")
    c4.metric("Non-ok failure_type", f"{mN['failure_type_is_bad'].mean():.2%}" if len(mN) else "—")

    st.subheader("Worst Runs")
    st.dataframe(run_health.head(20))

    st.subheader("Top Failure Hotspots (prompt × model)")
    st.dataframe(hotspots.head(30))

    if ok_only == "Only ok":
        m = m[m["ok"] == 1]
    elif ok_only == "Only failures":
        m = m[m["ok"] == 0]
    
    # -----------------------------------------------
    # ------ Run health card + table (aggregate tab)
    # -----------------------------------------------
    st.subheader("Run Health (Filtered)")
    total_rows = len(m)

    ok_rate = m["ok"].mean() if total_rows and "ok" in m.columns else None
    err_rate = (m["error"].fillna(0) != 0).mean() if total_rows and "error" in m.columns else None
    exit_rate = (m["exit_code"].fillna(0) != 0).mean() if total_rows and "exit_code" in m.columns else None
    fail_rate = (m["failure_type"].fillna("ok") != "ok").mean() if total_rows and "failure_type" in m.columns else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{total_rows:,}")
    c2.metric("OK rate", f"{ok_rate:.2%}" if ok_rate is not None else "—")
    c3.metric("Nonzero exit_code", f"{exit_rate:.2%}" if exit_rate is not None else "—")
    c4.metric("Non-ok failure_type", f"{fail_rate:.2%}" if fail_rate is not None else "—")
    
    
    st.subheader("Top Failure Hotspots")
    hot = m.copy()
    hot["failure_type"] = hot["failure_type"].fillna("ok")
    hot["is_fail"] = (hot["ok"].fillna(0) == 0)

    group_cols = [c for c in ["prompt_id", "model"] if c in hot.columns]
    if group_cols:
        summary = (
            hot.groupby(group_cols, as_index=False)
            .agg(
                rows=("is_fail", "size"),
                fail_rate=("is_fail", "mean"),
                avg_latency=("elapsed_s", "mean") if "elapsed_s" in hot.columns else ("is_fail", "size"),
                avg_wps=("words_per_s", "mean") if "words_per_s" in hot.columns else ("is_fail", "size"),
            )
            .sort_values(["fail_rate", "rows"], ascending=[False, False])
            .head(20)
        )
        st.dataframe(summary)
    else:
        st.info("Not enough columns to compute hotspots (need at least model and/or prompt_id).")
    
    
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

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows (filtered)", len(m))
    col2.metric("Success Rate", f"{m['ok'].mean():.2%}" if len(m) else "—")
    col3.metric("Avg Latency (s)", f"{m['elapsed_s'].mean():.2f}" if len(m) else "—")
    col4.metric("Avg Words/sec", f"{m['words_per_s'].mean():.2f}" if len(m) else "—")

    st.subheader("Latency by Model (All Runs)")
    figA, axA = plt.subplots()
    sns.boxplot(data=m, x="model", y="elapsed_s", ax=axA)
    st.pyplot(figA)

    st.subheader("Words/sec by Model (All Runs)")
    figB, axB = plt.subplots()
    sns.boxplot(data=m, x="model", y="words_per_s", ax=axB)
    st.pyplot(figB)

    st.subheader("Failure Type Breakdown (All Runs)")
    st.bar_chart(m["failure_type"].value_counts())
