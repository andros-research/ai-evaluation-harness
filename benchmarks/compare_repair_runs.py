from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


AGGREGATED_DIR = Path("benchmarks/results/aggregated")
AUDIT_ITEMS_CSV = AGGREGATED_DIR / "audit_items.csv"
CLAIM_COVERAGE_CSV = AGGREGATED_DIR / "claim_coverage.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare one original artifact against one repaired artifact."
    )
    parser.add_argument(
        "--original-artifact",
        required=True,
        help="Artifact name for the original narrative.",
    )
    parser.add_argument(
        "--repaired-artifact",
        required=True,
        help="Artifact name for the repaired narrative.",
    )
    parser.add_argument(
        "--audit-items-csv",
        default=str(AUDIT_ITEMS_CSV),
        help="Path to aggregated audit_items.csv",
    )
    parser.add_argument(
        "--claim-coverage-csv",
        default=str(CLAIM_COVERAGE_CSV),
        help="Path to aggregated claim_coverage.csv",
    )
    parser.add_argument(
        "--output-csv",
        default=str(AGGREGATED_DIR / "repair_comparison.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-json",
        default=str(AGGREGATED_DIR / "repair_comparison_summary.json"),
        help="Output JSON path",
    )
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def safe_int(value: Any) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except Exception:
        return 0


def safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def normalize_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def compute_claim_coverage_metrics(
    claim_coverage: pd.DataFrame,
    artifact_name: str,
) -> dict[str, Any]:
    cc = claim_coverage[claim_coverage["artifact_name"] == artifact_name].copy()

    if cc.empty:
        return {
            "selected_claim_count": 0,
            "used_claim_count": 0,
            "unused_claim_count": 0,
            "used_claim_ratio": 0.0,
        }

    used_flags = normalize_bool_series(cc["used_in_narrative"])
    selected_claim_count = len(cc)
    used_claim_count = int(used_flags.sum())
    unused_claim_count = selected_claim_count - used_claim_count
    used_claim_ratio = (
        round(used_claim_count / selected_claim_count, 4)
        if selected_claim_count > 0 else 0.0
    )

    return {
        "selected_claim_count": selected_claim_count,
        "used_claim_count": used_claim_count,
        "unused_claim_count": unused_claim_count,
        "used_claim_ratio": used_claim_ratio,
    }


def compute_audit_metrics(
    audit_items: pd.DataFrame,
    artifact_name: str,
) -> dict[str, Any]:
    a = audit_items[audit_items["artifact_name"] == artifact_name].copy()

    if a.empty:
        return {
            "total_bullets": 0,
            "supported": 0,
            "flagged": 0,
            "meta_caution": 0,
            "unknown_claim_ids_total": 0,
            "bullets_with_missing_claim_refs": 0,
        }

    audit_status = a["audit_status"].astype(str).fillna("")
    n_unknown_claim_ids = pd.to_numeric(a.get("n_unknown_claim_ids", 0), errors="coerce").fillna(0)
    missing_claim_refs = a.get("missing_claim_refs", pd.Series([False] * len(a), index=a.index))
    missing_claim_refs = missing_claim_refs.astype(str).str.lower().isin(["true", "1", "yes"])

    return {
        "total_bullets": len(a),
        "supported": int((audit_status == "supported").sum()),
        "flagged": int((audit_status == "flagged").sum()),
        "meta_caution": int((audit_status == "meta_caution").sum()),
        "unknown_claim_ids_total": int(n_unknown_claim_ids.sum()),
        "bullets_with_missing_claim_refs": int(missing_claim_refs.sum()),
    }


def build_comparison_row(
    original_artifact: str,
    repaired_artifact: str,
    audit_items: pd.DataFrame,
    claim_coverage: pd.DataFrame,
) -> dict[str, Any]:
    before_cov = compute_claim_coverage_metrics(claim_coverage, original_artifact)
    after_cov = compute_claim_coverage_metrics(claim_coverage, repaired_artifact)

    before_audit = compute_audit_metrics(audit_items, original_artifact)
    after_audit = compute_audit_metrics(audit_items, repaired_artifact)

    row = {
        "original_artifact": original_artifact,
        "repaired_artifact": repaired_artifact,

        "selected_claim_count_before": before_cov["selected_claim_count"],
        "used_claim_count_before": before_cov["used_claim_count"],
        "unused_claim_count_before": before_cov["unused_claim_count"],
        "used_claim_ratio_before": before_cov["used_claim_ratio"],

        "selected_claim_count_after": after_cov["selected_claim_count"],
        "used_claim_count_after": after_cov["used_claim_count"],
        "unused_claim_count_after": after_cov["unused_claim_count"],
        "used_claim_ratio_after": after_cov["used_claim_ratio"],

        "used_claim_count_delta": after_cov["used_claim_count"] - before_cov["used_claim_count"],
        "unused_claim_count_delta": after_cov["unused_claim_count"] - before_cov["unused_claim_count"],
        "used_claim_ratio_delta": round(
            after_cov["used_claim_ratio"] - before_cov["used_claim_ratio"], 4
        ),

        "total_bullets_before": before_audit["total_bullets"],
        "supported_before": before_audit["supported"],
        "flagged_before": before_audit["flagged"],
        "meta_caution_before": before_audit["meta_caution"],
        "unknown_claim_ids_before": before_audit["unknown_claim_ids_total"],
        "missing_claim_refs_before": before_audit["bullets_with_missing_claim_refs"],

        "total_bullets_after": after_audit["total_bullets"],
        "supported_after": after_audit["supported"],
        "flagged_after": after_audit["flagged"],
        "meta_caution_after": after_audit["meta_caution"],
        "unknown_claim_ids_after": after_audit["unknown_claim_ids_total"],
        "missing_claim_refs_after": after_audit["bullets_with_missing_claim_refs"],

        "bullet_count_delta": after_audit["total_bullets"] - before_audit["total_bullets"],
        "flagged_delta": after_audit["flagged"] - before_audit["flagged"],
        "unknown_claim_ids_delta": (
            after_audit["unknown_claim_ids_total"] - before_audit["unknown_claim_ids_total"]
        ),
        "missing_claim_refs_delta": (
            after_audit["bullets_with_missing_claim_refs"] - before_audit["bullets_with_missing_claim_refs"]
        ),
    }

    repair_success = (
        row["used_claim_ratio_after"] > row["used_claim_ratio_before"]
        and row["flagged_after"] <= row["flagged_before"]
        and row["unknown_claim_ids_after"] == 0
    )

    row["repair_success"] = repair_success
    return row


def save_csv_row(row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def build_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "original_artifact": row["original_artifact"],
        "repaired_artifact": row["repaired_artifact"],
        "repair_success": row["repair_success"],
        "claim_coverage": {
            "before": {
                "selected_claim_count": row["selected_claim_count_before"],
                "used_claim_count": row["used_claim_count_before"],
                "unused_claim_count": row["unused_claim_count_before"],
                "used_claim_ratio": row["used_claim_ratio_before"],
            },
            "after": {
                "selected_claim_count": row["selected_claim_count_after"],
                "used_claim_count": row["used_claim_count_after"],
                "unused_claim_count": row["unused_claim_count_after"],
                "used_claim_ratio": row["used_claim_ratio_after"],
            },
            "delta": {
                "used_claim_count_delta": row["used_claim_count_delta"],
                "unused_claim_count_delta": row["unused_claim_count_delta"],
                "used_claim_ratio_delta": row["used_claim_ratio_delta"],
            },
        },
        "audit": {
            "before": {
                "total_bullets": row["total_bullets_before"],
                "supported": row["supported_before"],
                "flagged": row["flagged_before"],
                "meta_caution": row["meta_caution_before"],
                "unknown_claim_ids": row["unknown_claim_ids_before"],
                "missing_claim_refs": row["missing_claim_refs_before"],
            },
            "after": {
                "total_bullets": row["total_bullets_after"],
                "supported": row["supported_after"],
                "flagged": row["flagged_after"],
                "meta_caution": row["meta_caution_after"],
                "unknown_claim_ids": row["unknown_claim_ids_after"],
                "missing_claim_refs": row["missing_claim_refs_after"],
            },
            "delta": {
                "bullet_count_delta": row["bullet_count_delta"],
                "flagged_delta": row["flagged_delta"],
                "unknown_claim_ids_delta": row["unknown_claim_ids_delta"],
                "missing_claim_refs_delta": row["missing_claim_refs_delta"],
            },
        },
    }


def main() -> None:
    args = parse_args()

    audit_items_path = Path(args.audit_items_csv).resolve()
    claim_coverage_path = Path(args.claim_coverage_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_json = Path(args.output_json).resolve()

    audit_items = load_csv(audit_items_path)
    claim_coverage = load_csv(claim_coverage_path)

    row = build_comparison_row(
        original_artifact=args.original_artifact,
        repaired_artifact=args.repaired_artifact,
        audit_items=audit_items,
        claim_coverage=claim_coverage,
    )
    summary = build_summary(row)

    save_csv_row(row, output_csv)
    save_json(summary, output_json)

    print(f"Saved repair comparison CSV: {output_csv}")
    print(f"Saved repair comparison summary JSON: {output_json}")
    print(
        "RepairSuccess={success} | UsedRatioBefore={before:.4f} | UsedRatioAfter={after:.4f} | Delta={delta:.4f}".format(
            success=row["repair_success"],
            before=row["used_claim_ratio_before"],
            after=row["used_claim_ratio_after"],
            delta=row["used_claim_ratio_delta"],
        )
    )


if __name__ == "__main__":
    main()