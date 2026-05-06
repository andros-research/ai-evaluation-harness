from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RESULTS_ROOT = Path("benchmarks/results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize labeled repair evaluation records into a repair matrix."
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Results root containing aggregated/repair_eval__*.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_best_repair_strategy(df: pd.DataFrame) -> dict:
    candidates = df.copy()

    # Prefer actual strategies over the no-strategy baseline when possible
    non_baseline = candidates[candidates["repair_label"] != "baseline_repair"].copy()
    if not non_baseline.empty:
        candidates = non_baseline

    successful = candidates[candidates["repair_success"] == True].copy()
    if not successful.empty:
        candidates = successful

    ranked = candidates.sort_values(
        ["repair_score", "flagged_after", "missing_claim_refs_after", "unknown_claim_ids_after"],
        ascending=[False, True, True, True],
    )

    return ranked.iloc[0].to_dict()


def score_repair_eval(record: dict[str, Any]) -> float:
    summary = record.get("comparison_summary", {})

    repair_success = 1.0 if summary.get("repair_success") else 0.0

    coverage_delta = (
        summary.get("claim_coverage", {})
        .get("delta", {})
        .get("used_claim_ratio_delta", 0.0)
    ) or 0.0

    after_audit = summary.get("audit", {}).get("after", {})

    flagged = after_audit.get("flagged", 0) or 0
    missing = after_audit.get("missing_claim_refs", 0) or 0
    unknown = after_audit.get("unknown_claim_ids", 0) or 0

    return (
        100.0 * repair_success
        + 25.0 * float(coverage_delta)
        - 10.0 * float(flagged)
        - 15.0 * float(missing)
        - 20.0 * float(unknown)
    )


def flatten_repair_eval(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    summary = record.get("comparison_summary", {})
    coverage = summary.get("claim_coverage", {})
    audit = summary.get("audit", {})

    cov_before = coverage.get("before", {})
    cov_after = coverage.get("after", {})
    cov_delta = coverage.get("delta", {})

    audit_before = audit.get("before", {})
    audit_after = audit.get("after", {})
    audit_delta = audit.get("delta", {})

    return {
        "source_file": str(path),
        "repair_label": record.get("repair_label", "unlabeled_repair"),
        "repair_strategies": ", ".join(record.get("repair_strategies", [])),
        "repair_success": bool(summary.get("repair_success")),
        "original_artifact": record.get("original_artifact"),
        "repaired_artifact": record.get("repaired_artifact"),
        "model": record.get("model"),
        "temperature": record.get("temperature"),
        "num_predict": record.get("num_predict"),

        "used_claim_ratio_before": cov_before.get("used_claim_ratio", 0.0),
        "used_claim_ratio_after": cov_after.get("used_claim_ratio", 0.0),
        "used_claim_ratio_delta": cov_delta.get("used_claim_ratio_delta", 0.0),
        "used_claim_count_delta": cov_delta.get("used_claim_count_delta", 0),
        "unused_claim_count_delta": cov_delta.get("unused_claim_count_delta", 0),

        "total_bullets_before": audit_before.get("total_bullets", 0),
        "total_bullets_after": audit_after.get("total_bullets", 0),
        "bullet_count_delta": audit_delta.get("bullet_count_delta", 0),

        "supported_after": audit_after.get("supported", 0),
        "flagged_after": audit_after.get("flagged", 0),
        "meta_caution_after": audit_after.get("meta_caution", 0),
        "unknown_claim_ids_after": audit_after.get("unknown_claim_ids", 0),
        "missing_claim_refs_after": audit_after.get("missing_claim_refs", 0),

        "flagged_delta": audit_delta.get("flagged_delta", 0),
        "unknown_claim_ids_delta": audit_delta.get("unknown_claim_ids_delta", 0),
        "missing_claim_refs_delta": audit_delta.get("missing_claim_refs_delta", 0),

        "repair_score": score_repair_eval(record),
    }


def main() -> None:
    args = parse_args()

    results_root = Path(args.results_root).resolve()
    agg_dir = results_root / "aggregated"

    files = sorted(agg_dir.glob("repair_eval__*.json"))

    if not files:
        raise FileNotFoundError(
            f"No repair eval records found under: {agg_dir}/repair_eval__*.json"
        )

    rows = []
    for path in files:
        record = load_json(path)
        rows.append(flatten_repair_eval(path, record))

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["repair_score", "repair_success", "used_claim_ratio_delta"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    df["rank"] = range(1, len(df) + 1)
    best_strategy = select_best_repair_strategy(df)

    out_csv = agg_dir / "repair_matrix_summary.csv"
    out_json = agg_dir / "repair_matrix_summary.json"

    df.to_csv(out_csv, index=False)
    out_json.write_text(
        json.dumps(df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    
    best_out = agg_dir / "repair_strategy_recommendation.json"
    best_out.write_text(
        json.dumps(best_strategy, indent=2),
        encoding="utf-8",
    )

    print(f"Loaded repair eval records: {len(files)}")
    print(f"Saved repair matrix CSV: {out_csv}")
    print(f"Saved repair matrix JSON: {out_json}")
    print(f"Saved repair strategy recommendations JSON: {best_out}")

    print("\n=== REPAIR MATRIX SUMMARY ===")
    cols = [
        "rank",
        "repair_label",
        "repair_strategies",
        "repair_success",
        "used_claim_ratio_delta",
        "flagged_after",
        "missing_claim_refs_after",
        "unknown_claim_ids_after",
        "repair_score",
    ]
    print(df[cols].to_string(index=False))
    
    print("\n=== RECOMMENDED REPAIR STRATEGY ===")
    print(f"repair_label: {best_strategy['repair_label']}")
    print(f"repair_strategies: {best_strategy['repair_strategies']}")
    print(f"repair_score: {best_strategy['repair_score']}")
    print(f"repair_success: {best_strategy['repair_success']}")


if __name__ == "__main__":
    main()