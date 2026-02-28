import pandas as pd
from pathlib import Path

path = Path("results/aggregated/runs_master.parquet")

if path.exists():
    df = pd.read_parquet(path)
else:
    df = pd.read_csv("results/aggregated/runs_master.csv")

print("=== FAILURE TYPE DEBUG ===")
print("Unique failure_type values:")
print(df["failure_type"].unique())

print("\nCounts:")
print(df["failure_type"].value_counts(dropna=False))

print("\nSample rows where ok == 0:")
print(df[df["ok"] == 0][["run_id", "model", "prompt_id", "failure_type"]].head(10))