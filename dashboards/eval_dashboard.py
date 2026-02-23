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
RESULTS_ROOT = REPO_ROOT / "benchmarks"                 # /home/joe/ai-lab/benchmarks

def find_result_folders(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    return sorted(
        [p for p in results_root.iterdir()
         if p.is_dir() and p.name.startswith("results_")],
        reverse=True,  # newest first
    )

runs = find_result_folders(RESULTS_ROOT)

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

runs_all = find_result_folders(RESULTS_ROOT)
runs = [r for r in runs_all if run_is_compatible(r)]

if not runs:
    st.error(
        "No compatible results folders found (need metrics.csv with columns: "
        + ", ".join(sorted(REQUIRED_COLS))
        + f") under: {RESULTS_ROOT}"
    )
    st.stop()
    
st.sidebar.caption(f"Showing {len(runs)} compatible runs (hiding {len(runs_all)-len(runs)} older/incomplete runs).")

selected_run = st.sidebar.selectbox(
    "Select Run",
    options=[str(p) for p in runs],
    key="select_run",
)

metrics_path = Path(selected_run) / "metrics.csv"
if not metrics_path.exists():
    st.error(f"metrics.csv not found: {metrics_path}")
    st.stop()

df = pd.read_csv(metrics_path)
st.sidebar.write(f"Loaded: {metrics_path}")

# -------------------------
# Filters
# -------------------------
models = st.sidebar.multiselect(
    "Models",
    options=sorted(df["model"].unique()),
    default=sorted(df["model"].unique()),
    key="models",
)

prompts = st.sidebar.multiselect(
    "Prompts",
    options=sorted(df["prompt_id"].unique()),
    default=sorted(df["prompt_id"].unique()),
    key="prompts",
)

df_filtered = df[(df["model"].isin(models)) & (df["prompt_id"].isin(prompts))]

st.subheader("Raw Metrics Snapshot")
st.dataframe(df_filtered.head(50))

# -------------------------
# KPI Row
# -------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Runs", len(df_filtered))
col2.metric("Success Rate", f"{df_filtered['ok'].mean():.2%}" if len(df_filtered) else "—")
col3.metric("Avg Latency (s)", f"{df_filtered['elapsed_s'].mean():.2f}" if len(df_filtered) else "—")
col4.metric("Avg Words", f"{df_filtered['words'].mean():.1f}" if len(df_filtered) else "—")

# -------------------------
# Latency Distribution
# -------------------------
st.subheader("Latency Distribution by Model")
fig1, ax1 = plt.subplots()
sns.boxplot(data=df_filtered, x="model", y="elapsed_s", ax=ax1)
st.pyplot(fig1)

# -------------------------
# Output Length Distribution
# -------------------------
st.subheader("Output Length (Words) Distribution")
fig2, ax2 = plt.subplots()
sns.violinplot(data=df_filtered, x="model", y="words", ax=ax2)
st.pyplot(fig2)

# -------------------------
# Failure Mode Analysis
# -------------------------
st.subheader("Failure Type Breakdown")
failure_counts = df_filtered["failure_type"].value_counts()
st.bar_chart(failure_counts)

# -------------------------
# Words per Second
# -------------------------
st.subheader("Model Efficiency (Words per Second)")
fig3, ax3 = plt.subplots()
sns.boxplot(data=df_filtered, x="model", y="words_per_s", ax=ax3)
st.pyplot(fig3)