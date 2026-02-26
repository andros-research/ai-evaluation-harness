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
        
    # ---- Growth over time (lab growth) ----
    g = master.copy()

    # Prefer run_timestamp_iso (run-level) if present; else fall back to ts; else parse run_id.
    if "run_timestamp_iso" in g.columns:
        g["t"] = pd.to_datetime(g["run_timestamp_iso"], errors="coerce")
    elif "ts" in g.columns:
        g["t"] = pd.to_datetime(g["ts"], errors="coerce")
    else:
        # run_id like 2026-02-25_20-46-49
        g["t"] = pd.to_datetime(g["run_id"].astype(str).str.replace("_", " "), errors="coerce")

    g = g.dropna(subset=["t"]).sort_values("t")
    g["day"] = g["t"].dt.date

    # Daily counts
    daily = (
        g.groupby("day")
        .agg(
            rows=("day", "size"),
            unique_runs=("run_id", "nunique"),
            unique_models=("model", "nunique"),
            unique_prompts=("prompt_id", "nunique"),
        )
        .reset_index()
    )

    # Cumulative
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

    if ok_only == "Only ok":
        m = m[m["ok"] == 1]
    elif ok_only == "Only failures":
        m = m[m["ok"] == 0]

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
    axG.set_ylabel(ycol)
    axG.set_title(metric_choice)
    st.pyplot(figG)

    with st.expander("Growth table (daily)", expanded=False):
        st.dataframe(daily)