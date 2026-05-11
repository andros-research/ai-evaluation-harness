from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError


DEFAULT_RESULTS_ROOT = Path("benchmarks/results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-model repair policy recommendations from model profiles and repair matrix recommendation."
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Results root containing aggregated/model_profile_summary.csv and aggregated/repair_strategy_recommendation.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def parse_strategy_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    try:
        if pd.isna(value):
            return []
    except TypeError:
        pass

    return [x.strip() for x in str(value).split(",") if x.strip()]


def infer_fallback_policy(row: pd.Series) -> tuple[str, list[str], str, str]:
    """
    Deterministic fallback policy when profile focus does not overlap
    with matrix-tested recommendation.

    Returns:
        selected_label, selected_strategies, reason, confidence
    """
    failure_mode = str(row.get("dominant_failure_type_v2", "")).strip()
    semantic_pattern = str(row.get("dominant_semantic_pattern", "")).strip()
    adaptability = str(row.get("adaptability", "")).strip()
    role = str(row.get("model_role", "")).strip()

    if failure_mode == "verbosity_drift":
        return (
            "compression_plus_structure",
            ["compress_output", "enforce_structured_output"],
            "Fallback selected because profile indicates verbosity drift.",
            "medium",
        )

    if semantic_pattern == "over_selection":
        return (
            "structure_then_selection",
            ["enforce_structured_output", "tighten_selection"],
            "Fallback selected because profile indicates over-selection, though selection strategies should be validated carefully.",
            "medium",
        )

    if adaptability == "high" or role == "explorer":
        return (
            "explore_then_constrain",
            ["allow_exploration_then_constrain", "compress_output", "enforce_structured_output"],
            "Fallback selected because profile indicates high adaptability or explorer behavior.",
            "medium",
        )

    return (
        "minimal_repair",
        ["enforce_structured_output"],
        "Fallback selected because no stronger profile-specific repair signal was available.",
        "low",
    )


def build_policy_record(
    row: pd.Series,
    matrix_rec: dict[str, Any],
) -> dict[str, Any]:
    profile_focus = parse_strategy_list(row.get("repair_focus", ""))

    matrix_label = matrix_rec.get("repair_label", "none")
    matrix_strategies = parse_strategy_list(matrix_rec.get("repair_strategies", ""))

    overlap = sorted(set(profile_focus).intersection(matrix_strategies))

    if matrix_rec and overlap:
        selected_label = matrix_label
        selected_strategies = matrix_strategies
        selection_reason = (
            "Matrix-tested recommendation overlaps with profile repair focus."
        )
        policy_confidence = "high"
    elif matrix_rec and not overlap:
        selected_label, selected_strategies, selection_reason, policy_confidence = infer_fallback_policy(row)

        # Keep matrix recommendation visible, but do not force it when there is no overlap.
        selection_reason = (
            selection_reason
            + " Current matrix-tested recommendation did not overlap with profile repair focus."
        )
    else:
        selected_label, selected_strategies, selection_reason, policy_confidence = infer_fallback_policy(row)
        selection_reason = (
            selection_reason
            + " No matrix-tested recommendation artifact was available."
        )

    return {
        "model": row.get("model"),
        "suite_name": row.get("suite_name"),
        "experiment_name": row.get("experiment_name"),

        "model_role": row.get("model_role", "unknown"),
        "temperature_sensitivity": row.get("temperature_sensitivity", "unknown"),
        "overall_direction": row.get("overall_direction", "unknown"),
        "dominant_failure_type_v2": row.get("dominant_failure_type_v2", "unknown"),
        "dominant_semantic_pattern": row.get("dominant_semantic_pattern", "unknown"),
        "consistency": row.get("consistency", "unknown"),
        "adaptability": row.get("adaptability", "unknown"),

        "profile_repair_focus": profile_focus,

        "matrix_recommended_label": matrix_label,
        "matrix_recommended_strategies": matrix_strategies,
        "matrix_repair_score": matrix_rec.get("repair_score"),
        "matrix_repair_success": matrix_rec.get("repair_success"),

        "overlap": overlap,
        "has_overlap": bool(overlap),

        "selected_repair_label": selected_label,
        "selected_repair_strategies": selected_strategies,
        "selection_reason": selection_reason,
        "policy_confidence": policy_confidence,
    }


def main() -> None:
    args = parse_args()

    results_root = Path(args.results_root).resolve()
    agg_dir = results_root / "aggregated"

    profile_path = agg_dir / "model_profile_summary.csv"
    matrix_rec_path = agg_dir / "repair_strategy_recommendation.json"

    model_profile = load_csv(profile_path)
    matrix_rec = load_json(matrix_rec_path)

    if model_profile.empty:
        raise FileNotFoundError(
            f"No model profile summary found at: {profile_path}\n"
            "Expected a model_profile_summary.csv artifact before building repair policy recommendations."
        )

    if not matrix_rec:
        print(f"Warning: no matrix recommendation found at {matrix_rec_path}")
        print("Proceeding with profile-only fallback policy recommendations.")

    records = [
        build_policy_record(row, matrix_rec)
        for _, row in model_profile.iterrows()
    ]

    out_json = agg_dir / "repair_policy_recommendations.json"
    out_csv = agg_dir / "repair_policy_recommendations.csv"

    out_json.write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)

    print(f"Loaded model profile rows: {len(model_profile)}")
    print(f"Loaded matrix recommendation: {bool(matrix_rec)}")
    print(f"Saved repair policy recommendations JSON: {out_json}")
    print(f"Saved repair policy recommendations CSV: {out_csv}")

    print("\n=== REPAIR POLICY RECOMMENDATIONS ===")
    display_cols = [
        "model",
        "model_role",
        "dominant_failure_type_v2",
        "dominant_semantic_pattern",
        "matrix_recommended_label",
        "overlap",
        "selected_repair_label",
        "selected_repair_strategies",
        "policy_confidence",
    ]
    print(df[[c for c in display_cols if c in df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()