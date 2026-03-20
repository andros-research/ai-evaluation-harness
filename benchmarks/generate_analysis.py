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
NARRATIVE_ROOT = REPO_ROOT / "benchmarks" / "results" / "narratives"


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
        "--output-dir",
        default=str(NARRATIVE_ROOT),
        help="Directory to save analysis artifacts",
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

    out_dir = Path(args.output_dir)
    comp_slug = "_".join(args.comparisons)
    stem = f"{args.suite}__{args.baseline}__vs__{comp_slug}__{args.metric}"

    save_json(payload, out_dir / f"{stem}__analysis.json")

    print(f"Saved analysis payload to: {out_dir / f'{stem}__analysis.json'}")


if __name__ == "__main__":
    main()