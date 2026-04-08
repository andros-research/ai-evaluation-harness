from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def claim_strength(abs_delta: float, min_abs_delta: float = 0.10) -> str | None:
    if abs_delta < min_abs_delta:
        return None
    if abs_delta >= 0.30:
        return "strong"
    if abs_delta >= 0.15:
        return "medium"
    return "weak"


def validate_claim_label(
    label: str,
    baseline: float | None,
    comparison: float | None,
    delta: float | None,
    eps: float = 1e-9,
) -> dict[str, bool]:
    values_present = (
        baseline is not None and comparison is not None and delta is not None
    )

    direction_matches_delta = False
    if values_present:
        if label == "improves":
            direction_matches_delta = delta > eps
        elif label == "degrades":
            direction_matches_delta = delta < -eps
        elif label == "stable_always_pass":
            direction_matches_delta = abs(delta) <= eps
        elif label == "stable_always_fail":
            direction_matches_delta = abs(delta) <= eps
        elif label == "stable_invariant":
            direction_matches_delta = abs(delta) <= eps
        elif label == "missing_comparison":
            direction_matches_delta = False

    stable_pass_matches_values = (
        label != "stable_always_pass"
        or (values_present and baseline == 1.0 and comparison == 1.0)
    )
    stable_fail_matches_values = (
        label != "stable_always_fail"
        or (values_present and baseline == 0.0 and comparison == 0.0)
    )
    invariant_matches_values = (
        label != "stable_invariant"
        or (values_present and abs(delta) <= eps)
    )

    return {
        "values_present": values_present,
        "direction_matches_delta": direction_matches_delta,
        "stable_pass_matches_values": stable_pass_matches_values,
        "stable_fail_matches_values": stable_fail_matches_values,
        "invariant_matches_values": invariant_matches_values,
    }


def all_validation_passed(validation: dict[str, bool]) -> bool:
    return all(validation.values())


def infer_claim_type(label: str) -> str:
    if label in {"improves", "degrades"}:
        return "delta_change"
    if label == "stable_always_pass":
        return "stable_ceiling"
    if label == "stable_always_fail":
        return "stable_floor"
    if label == "stable_invariant":
        return "stable_invariant"
    return "unsupported"


def is_claim_eligible(
    label: str,
    baseline: float | None,
    comparison: float | None,
    delta: float | None,
    min_abs_delta: float,
) -> tuple[bool, str | None]:
    if baseline is None or comparison is None or delta is None:
        return False, None

    if label in {"stable_always_pass", "stable_always_fail"}:
        return True, "strong"

    if label == "stable_invariant":
        return True, "weak"

    if label in {"improves", "degrades"}:
        strength = claim_strength(abs(delta), min_abs_delta=min_abs_delta)
        return (strength is not None), strength

    return False, None


def build_claim_id(
    suite_name: str,
    prompt_id: str,
    model: str,
    comparison_experiment: str,
    baseline_experiment: str,
) -> str:
    safe_model = model.replace(":", "_")
    return (
        f"{suite_name}__{prompt_id}__{safe_model}__"
        f"{comparison_experiment}_vs_{baseline_experiment}"
    )


def make_claim_from_cell(
    *,
    suite_name: str,
    metric: str,
    baseline_experiment: str,
    comparison_experiment: str,
    cell: dict[str, Any],
    min_abs_delta: float,
) -> dict[str, Any]:
    prompt_id = str(cell["prompt_id"])
    model = str(cell["model"])
    baseline = cell.get("baseline_pass_rate")
    comparison = cell.get("comparison_pass_rate")
    delta = cell.get("delta_pass_rate")
    label = str(cell.get("label"))

    validation = validate_claim_label(
        label=label,
        baseline=baseline,
        comparison=comparison,
        delta=delta,
    )
    validation_passed = all_validation_passed(validation)

    claim_type = infer_claim_type(label)
    eligible, strength = is_claim_eligible(
        label=label,
        baseline=baseline,
        comparison=comparison,
        delta=delta,
        min_abs_delta=min_abs_delta,
    )

    direction = None
    if delta is not None:
        if delta > 0:
            direction = "up"
        elif delta < 0:
            direction = "down"
        else:
            direction = "flat"

    claim_status = "validated" if (validation_passed and eligible) else "rejected"

    return {
        "claim_id": build_claim_id(
            suite_name=suite_name,
            prompt_id=prompt_id,
            model=model,
            comparison_experiment=comparison_experiment,
            baseline_experiment=baseline_experiment,
        ),
        "claim_type": claim_type,
        "suite_name": suite_name,
        "metric": metric,
        "prompt_id": prompt_id,
        "model": model,
        "baseline_experiment": baseline_experiment,
        "comparison_experiment": comparison_experiment,
        "baseline_value": baseline,
        "comparison_value": comparison,
        "delta_value": delta,
        "direction": direction,
        "label": label,
        "claim_eligible": eligible,
        "claim_strength": strength,
        "validation": validation,
        "validation_passed": validation_passed,
        "claim_status": claim_status,
    }


def extract_claims(
    analysis: dict[str, Any],
    min_abs_delta: float = 0.10,
) -> dict[str, Any]:
    suite_name = str(analysis["suite_name"])
    metric = str(analysis["metric"])
    baseline_experiment = str(analysis["baseline_experiment"])

    claims: list[dict[str, Any]] = []

    for comp in analysis.get("comparisons", []):
        comparison_experiment = str(comp["comparison_experiment"])
        for cell in comp.get("cells", []):
            claims.append(
                make_claim_from_cell(
                    suite_name=suite_name,
                    metric=metric,
                    baseline_experiment=baseline_experiment,
                    comparison_experiment=comparison_experiment,
                    cell=cell,
                    min_abs_delta=min_abs_delta,
                )
            )

    validated_claims = [
        c for c in claims
        if c["claim_status"] == "validated"
    ]
    rejected_claims = [
        c for c in claims
        if c["claim_status"] != "validated"
    ]

    def sort_key(c: dict[str, Any]) -> tuple[float, str, str]:
        delta = c.get("delta_value")
        abs_delta = abs(delta) if delta is not None else -1.0
        return (abs_delta, c["prompt_id"], c["model"])

    validated_claims = sorted(validated_claims, key=sort_key, reverse=True)
    rejected_claims = sorted(rejected_claims, key=sort_key, reverse=True)

    return {
        "suite_name": suite_name,
        "metric": metric,
        "baseline_experiment": baseline_experiment,
        "comparison_experiments": analysis.get("comparison_experiments", []),
        "min_abs_delta": min_abs_delta,
        "n_claims_total": len(claims),
        "n_validated_claims": len(validated_claims),
        "n_rejected_claims": len(rejected_claims),
        "validated_claims": validated_claims,
        "rejected_claims": rejected_claims,
    }


def default_output_path(analysis_json: Path) -> Path:
    stem = analysis_json.stem.replace("__analysis", "")
    return analysis_json.parent / f"{stem}__claims.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate validated claims from an analysis payload."
    )
    parser.add_argument("--analysis-json", required=True)
    parser.add_argument("--results-root", default="benchmarks/results")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--min-abs-delta", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_root = Path(args.results_root).resolve()
    analysis_path = Path(args.analysis_json).resolve()

    if not analysis_path.exists():
        raise FileNotFoundError(f"analysis json not found: {analysis_path}")
    
    analysis = load_json(analysis_path)

    claims_payload = extract_claims(
        analysis=analysis,
        min_abs_delta=args.min_abs_delta,
    )
    
    claims_payload["results_root"] = str(results_root)
    claims_payload["source_analysis_json"] = str(analysis_path)
    
    output_path = (
        Path(args.output_json).resolve()
        if args.output_json
        else default_output_path(analysis_path)
    )

    save_json(claims_payload, output_path)
    print(f"Saved claims JSON: {output_path}")


if __name__ == "__main__":
    main()