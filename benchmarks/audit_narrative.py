from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def load_narrative_text(narrative_payload: dict[str, Any]) -> str:
    text = narrative_payload.get("narrative", "")
    if not isinstance(text, str):
        raise ValueError("Narrative payload missing string field 'narrative'")
    return text


def split_narrative_sections(narrative_text: str) -> list[dict[str, str]]:
    """
    Parse a narrative in the format:

    Observations:
    - ...
    - ...

    Tradeoffs:
    - ...

    Invariances:
    - ...

    Cautions:
    - ...
    """
    lines = narrative_text.splitlines()
    items: list[dict[str, str]] = []

    current_section = "unknown"

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.endswith(":") and not line.startswith("-"):
            current_section = line[:-1].strip().lower()
            continue

        if line.startswith("- "):
            items.append(
                {
                    "section": current_section,
                    "text": line[2:].strip(),
                }
            )

    return items


def extract_scope_flags(text: str) -> dict[str, bool]:
    t = normalize_text(text)

    return {
        "mentions_all_models": "all models" in t,
        "mentions_all_temperature_settings": (
            "all temperature settings" in t
            or "both temperature settings" in t
            or "across all temperature settings" in t
        ),
        "mentions_consistently": "consistently" in t,
        "mentions_improves": (
            "improve" in t or "improves" in t or "improvement" in t
        ),
        "mentions_degrades": (
            "degrade" in t or "degrades" in t or "degradation" in t
        ),
    }


def find_matching_claims(text: str, selected_claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    t = normalize_text(text)
    matches = []

    for claim in selected_claims:
        prompt_id_raw = str(claim.get("prompt_id", "")).lower()
        prompt_id_spaced = prompt_id_raw.replace("_", " ")

        model = str(claim.get("model", "")).lower()
        comparison_experiment = str(claim.get("comparison_experiment", "")).lower()

        prompt_match = (
            (prompt_id_raw and prompt_id_raw in t)
            or (prompt_id_spaced and prompt_id_spaced in t)
        )
        model_match = model in t if model else False
        comp_match = comparison_experiment in t if comparison_experiment else False

        mentions_prompt = prompt_match
        mentions_model = model_match

        if mentions_prompt and mentions_model:
            matches.append(claim)
        elif mentions_prompt and not mentions_model:
            matches.append(claim)
        elif mentions_model and not mentions_prompt:
            matches.append(claim)

    return matches


def unique_values(claims: list[dict[str, Any]], key: str) -> set[str]:
    vals = set()
    for claim in claims:
        value = claim.get(key)
        if value is not None:
            vals.add(str(value))
    return vals


def infer_issue_type(
    item_text: str,
    matched_claims: list[dict[str, Any]],
) -> tuple[bool, str, str]:
    """
    Returns:
        supported, issue_type, notes
    """
    if not matched_claims:
        return False, "unclear_match", "No matching claims found for this narrative item."

    flags = extract_scope_flags(item_text)

    models = unique_values(matched_claims, "model")
    prompts = unique_values(matched_claims, "prompt_id")
    comparisons = unique_values(matched_claims, "comparison_experiment")
    labels = unique_values(matched_claims, "label")

    # Rule 1: "all models" but matched claims only cover one model
    if flags["mentions_all_models"] and len(models) < 2:
        return (
            False,
            "scope_overreach",
            f"Sentence claims scope across all models, but matched claims cover models={sorted(models)} only.",
        )

    # Rule 2: "all/both temperature settings" but only one comparison experiment matched
    if flags["mentions_all_temperature_settings"] and len(comparisons) < 2:
        return (
            False,
            "scope_overreach",
            (
                "Sentence claims coverage across all/both temperature settings, "
                f"but matched claims cover comparison_experiments={sorted(comparisons)} only."
            ),
        )

    # Rule 3: "consistently" but only one matched claim
    if flags["mentions_consistently"] and len(matched_claims) < 2:
        return (
            False,
            "unsupported_generalization",
            "Sentence uses 'consistently' but only one supporting claim was matched.",
        )

    # Rule 4: sentence says improves, but matched claims are mixed or all non-improvement
    if flags["mentions_improves"]:
        if not any(lbl == "improves" for lbl in labels):
            return (
                False,
                "contradiction",
                f"Sentence implies improvement, but matched claim labels are {sorted(labels)}.",
            )

    # Rule 5: sentence says degrades, but matched claims are mixed or all non-degradation
    if flags["mentions_degrades"]:
        if not any(lbl == "degrades" for lbl in labels):
            return (
                False,
                "contradiction",
                f"Sentence implies degradation, but matched claim labels are {sorted(labels)}.",
            )

    # Rule 6: if matched claims span multiple prompts/models in a way that may be too broad, allow for now
    # v0 chooses not to over-flag broad but plausible summaries if they are directionally consistent.
    
    # Rule 6: contradictory scope (single model + all models)
    if len(models) == 1 and flags["mentions_all_models"]:
        return (
            False,
            "scope_overreach",
            f"Sentence refers to a specific model scope {sorted(models)} but also claims 'all models'."
        )

    return True, "supported", "Narrative item is supported by matched claims under v0 audit rules."


def audit_narrative(
    selected_claims_payload: dict[str, Any],
    narrative_payload: dict[str, Any],
) -> dict[str, Any]:
    narrative_text = load_narrative_text(narrative_payload)
    items = split_narrative_sections(narrative_text)

    selected_claims = selected_claims_payload.get("selected_claims", [])
    if not isinstance(selected_claims, list):
        raise ValueError("Selected claims payload missing list field 'selected_claims'")

    audit_items: list[dict[str, Any]] = []

    for item in items:
        text = item["text"]
        
        if item["section"] == "cautions":
            audit_items.append(
                {
                    "section": item["section"],
                    "text": text,
                    "supported": None,
                    "issue_type": "meta_caution",
                    "notes": "Meta-caution; excluded from claim-support scoring.",
                    "matched_claim_ids": [],
                }
            )
            continue
        
        matched_claims = find_matching_claims(text, selected_claims)
        supported, issue_type, notes = infer_issue_type(text, matched_claims)

        audit_items.append(
            {
                "section": item["section"],
                "text": text,
                "supported": supported,
                "issue_type": issue_type,
                "notes": notes,
                "matched_claim_ids": [c.get("claim_id") for c in matched_claims],
            }
        )

    n_items = len(audit_items)
    n_meta_cautions = sum(1 for x in audit_items if x["supported"] is None)
    n_scored_items = sum(1 for x in audit_items if x["supported"] is not None)
    n_supported = sum(1 for x in audit_items if x["supported"] is True)
    n_flagged = sum(1 for x in audit_items if x["supported"] is False)
    fidelity_score = (n_supported / n_scored_items) if n_scored_items else 0.0

    return {
        "source_selected_claims_json": selected_claims_payload.get("_source_path"),
        "source_narrative_json": narrative_payload.get("_source_path"),
        "suite_name": selected_claims_payload.get("suite_name"),
        "metric": selected_claims_payload.get("metric"),
        "baseline_experiment": selected_claims_payload.get("baseline_experiment"),
        "comparison_experiments": selected_claims_payload.get("comparison_experiments", []),
        "fidelity_score": fidelity_score,
        "summary": {
            "n_items": n_items,
            "n_scored_items": n_scored_items,
            "n_meta_cautions": n_meta_cautions,
            "n_supported": n_supported,
            "n_flagged": n_flagged,
        },
        "audit_items": audit_items,
    }


def default_output_path(narrative_json: Path) -> Path:
    stem = narrative_json.stem.replace("__narrative_from_claims", "")
    return narrative_json.parent / f"{stem}__audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a generated narrative against selected claims."
    )
    parser.add_argument("--selected-claims-json", required=True)
    parser.add_argument("--narrative-json", required=True)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selected_claims_path = Path(args.selected_claims_json).resolve()
    narrative_path = Path(args.narrative_json).resolve()

    selected_claims_payload = load_json(selected_claims_path)
    selected_claims_payload["_source_path"] = str(selected_claims_path)

    narrative_payload = load_json(narrative_path)
    narrative_payload["_source_path"] = str(narrative_path)

    audit_payload = audit_narrative(
        selected_claims_payload=selected_claims_payload,
        narrative_payload=narrative_payload,
    )

    output_path = (
        Path(args.output_json).resolve()
        if args.output_json
        else default_output_path(narrative_path)
    )

    save_json(audit_payload, output_path)
    print(f"Saved audit JSON: {output_path}")


if __name__ == "__main__":
    main()