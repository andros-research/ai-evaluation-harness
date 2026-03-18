from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
runs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("results_")])
latest = runs[-1]
df = pd.read_csv(latest / "metrics.csv")

df["elapsed_s"] = pd.to_numeric(df["elapsed_s"], errors="coerce")

pivot = df.pivot_table(index="prompt_id", columns="model", values="elapsed_s", aggfunc="mean")
pivot.plot(kind="bar")
plt.ylabel("elapsed_s")
plt.title(f"Elapsed seconds by prompt/model ({latest.name})")
plt.tight_layout()
out = latest / "elapsed_by_model.png"
plt.savefig(out, dpi=200)
print(f"Saved: {out}")
