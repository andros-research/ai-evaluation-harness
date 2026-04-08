from __future__ import annotations

import argparse
from pathlib import Path

from analysis import (
    load_master,
    make_experiment_metric_summary,
    make_analysis_payload,
    save_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build narrative analysis payload from runs_master.")
    parser.add_argument("--suite", required=True, help="Suite name, e.g. suite_instruction")
    parser.add_argument("--baseline", required=True, help="Baseline experiment, e.g. temp0")
    parser.add_argument(
        "--comparisons",
        nargs="+",
        required=True,
        help="Comparison experiments, e.g. temp03 temp07",
    )
    parser.add_argument(
        "--metric",
        default="checks_pass_rate",
        choices=["checks_pass_rate", "overall_pass_rate", "pipeline_success_rate"],
        help="Metric to compare",
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Run-scoped results root, e.g. benchmarks/results/runs/v1_4_4_scaled",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save analysis artifacts. Defaults to <results-root>/narratives",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    master = load_master()
    experiment_summary = make_experiment_metric_summary(master)

    payload = make_analysis_payload(
        experiment_summary=experiment_summary,
        suite_name=args.suite,
        baseline_experiment=args.baseline,
        comparison_experiments=args.comparisons,
        metric_col=args.metric,
    )

    results_root = Path(args.results_root).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (results_root / "narratives")
    out_dir.mkdir(parents=True, exist_ok=True)

    comp_slug = "_".join(args.comparisons)
    stem = f"{args.suite}__{args.baseline}__vs__{comp_slug}__{args.metric}"

    out_path = out_dir / f"{stem}__analysis.json"
    save_json(payload, out_path)

    print(f"Saved analysis payload to: {out_path}")


if __name__ == "__main__":
    main()