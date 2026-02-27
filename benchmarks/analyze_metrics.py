from pathlib import Path
import pandas as pd

def load_latest_metrics(root: Path) -> pd.DataFrame:
    runs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("results_")])
    if not runs:
        raise FileNotFoundError("No results_* directories found.")
    latest = runs[-1]
    metrics_path = latest / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.csv in {latest}")
    df = pd.read_csv(metrics_path)
    df["run_dir"] = latest.name
    return df

def main():
    root = Path(__file__).resolve().parent
    df = load_latest_metrics(root)

    # Basic cleanup / types
    for col in ["elapsed_s", "chars", "words", "lines"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print("\n=== Latest run ===")
    print(df["run_dir"].iloc[0])
    print("\n=== Rows ===")
    print(len(df))

    print("\n=== By prompt_id x model (elapsed_s, words) ===")
    summary = (
        df.groupby(["prompt_id", "model"], as_index=False)
          .agg(elapsed_s=("elapsed_s", "mean"),
               words=("words", "mean"),
               chars=("chars", "mean"),
               lines=("lines", "mean"))
          .sort_values(["prompt_id", "elapsed_s"])
    )
    print(summary.to_string(index=False))

    print("\n=== Fastest per prompt_id (by elapsed_s) ===")
    fastest = summary.sort_values(["prompt_id", "elapsed_s"]).groupby("prompt_id").head(1)
    print(fastest[["prompt_id","model","elapsed_s","words"]].to_string(index=False))

if __name__ == "__main__":
    main()

