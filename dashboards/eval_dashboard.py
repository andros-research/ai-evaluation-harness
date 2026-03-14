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
        axH1.set_title(f"{heat_value_col} by prompt × model")
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
        axH2.set_title(f"{heat_value_col_all} by prompt × model")
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
