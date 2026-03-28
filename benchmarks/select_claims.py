from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ----------------------------
# Core selection logic
# ----------------------------

def sort_by_abs_delta_desc(claims: List[dict[str, Any]]) -> List[dict[str, Any]]:
    def key(c: dict[str, Any]):
        d = c.get("delta_value")
        return abs(d) if d is not None else -1.0

    return sorted(claims, key=key, reverse=True)


def bucket_claims(validated_claims: List[dict[str, Any]]) -> dict[str, List[dict[str, Any]]]:
    buckets = {
        "strong_delta": [],
        "medium_delta": [],
        "weak_delta": [],
        "stable_floor": [],
        "stable_ceiling": [],
        "stable_invariant": [],
    }

    for c in validated_claims:
        ctype = c["claim_type"]
        strength = c.get("claim_strength")

        if ctype == "delta_change":
            if strength == "strong":
                buckets["strong_delta"].append(c)
            elif strength == "medium":
                buckets["medium_delta"].append(c)
            else:
                buckets["weak_delta"].append(c)

        elif ctype == "stable_floor":
            buckets["stable_floor"].append(c)

        elif ctype == "stable_ceiling":
            buckets["stable_ceiling"].append(c)

        elif ctype == "stable_invariant":
            buckets["stable_invariant"].append(c)

    return buckets


def select_claims_for_narrative(
    claims_payload: dict[str, Any],
    max_total_claims: int = 12,
    max_medium_delta: int = 4,
    max_stable: int = 4,
) -> dict[str, Any]:

    validated = claims_payload.get("validated_claims", [])

    buckets = bucket_claims(validated)

    # Sort delta buckets by importance
    strong_delta = sort_by_abs_delta_desc(buckets["strong_delta"])
    medium_delta = sort_by_abs_delta_desc(buckets["medium_delta"])
    weak_delta = sort_by_abs_delta_desc(buckets["weak_delta"])

    stable_floor = buckets["stable_floor"]
    stable_ceiling = buckets["stable_ceiling"]
    stable_invariant = buckets["stable_invariant"]

    selected: List[dict[str, Any]] = []

    # Always include all strong deltas
    selected.extend(strong_delta)

    # Add medium deltas (capped)
    selected.extend(medium_delta[:max_medium_delta])

    # Add stable claims (balanced mix) keeping count of n_invariant
    n_invariant = 1 if stable_invariant else 0
    remaining_stable = max_stable - n_invariant

    n_floor = remaining_stable // 2
    n_ceiling = remaining_stable - n_floor

    stable_combined = (
        stable_floor[:n_floor]
        + stable_ceiling[:n_ceiling]
    )

    if n_invariant:
        stable_combined += stable_invariant[:1]

    # If still room, include invariant
    if len(stable_combined) < max_stable:
        remaining = max_stable - len(stable_combined)
        stable_combined.extend(stable_invariant[:remaining])

    selected.extend(stable_combined)

    # Fill remaining space with weak deltas if needed
    if len(selected) < max_total_claims:
        remaining = max_total_claims - len(selected)
        selected.extend(weak_delta[:remaining])

    # Final trim (safety)
    selected = selected[:max_total_claims]

    return {
        "suite_name": claims_payload["suite_name"],
        "metric": claims_payload["metric"],
        "baseline_experiment": claims_payload["baseline_experiment"],
        "comparison_experiments": claims_payload["comparison_experiments"],
        "selection_policy": {
            "max_total_claims": max_total_claims,
            "max_medium_delta": max_medium_delta,
            "max_stable": max_stable,
        },
        "selection_summary": {
            "n_validated_claims_available": len(validated),
            "n_selected_claims": len(selected),
            "n_strong_delta": len(strong_delta),
            "n_medium_delta": len(medium_delta),
            "n_stable_total": (
                len(stable_floor) + len(stable_ceiling) + len(stable_invariant)
            ),
        },
        "selected_claims": selected,
    }


# ----------------------------
# CLI
# ----------------------------

def default_output_path(claims_json: Path) -> Path:
    stem = claims_json.stem.replace("__claims", "")
    return claims_json.parent / f"{stem}__selected_claims.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select high-value claims for narrative generation."
    )
    parser.add_argument("--claims-json", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--max-total", type=int, default=12)
    parser.add_argument("--max-medium", type=int, default=4)
    parser.add_argument("--max-stable", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    claims_path = Path(args.claims_json).resolve()
    claims = load_json(claims_path)

    selected_payload = select_claims_for_narrative(
        claims_payload=claims,
        max_total_claims=args.max_total,
        max_medium_delta=args.max_medium,
        max_stable=args.max_stable,
    )

    output_path = (
        Path(args.output_json).resolve()
        if args.output_json
        else default_output_path(claims_path)
    )

    save_json(selected_payload, output_path)

    print(f"Saved selected claims JSON: {output_path}")


if __name__ == "__main__":
    main()