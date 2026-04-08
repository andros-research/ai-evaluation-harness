from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


CLAIMS_BLOCK_RE = re.compile(r"\[CLAIMS:\s*([^\]]+)\]")
BULLET_RE = re.compile(r"^- (.+)$")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def split_narrative_sections(narrative_text: str) -> list[dict[str, str]]:
    lines = narrative_text.splitlines()
    items: list[dict[str, str]] = []
    current_section = "unknown"

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.endswith(":") and not line.startswith("- "):
            current_section = line[:-1].strip().lower()
            continue

        m = BULLET_RE.match(line)
        if m:
            items.append(
                {
                    "section": current_section,
                    "raw_text": m.group(1).strip(),
                }
            )

    return items


def normalize_claim_id(s: str) -> str:
    s = re.sub(r"\s+", "", s)
    s = s.strip().strip("'\"").rstrip("].,;:").lstrip("[")
    return s

def extract_claim_ids(text: str) -> list[str]:
    blocks = CLAIMS_BLOCK_RE.findall(text)
    claim_ids: list[str] = []

    for block in blocks:
        parts = [normalize_claim_id(p.strip()) for p in block.split(",")]
        claim_ids.extend([p for p in parts if p])

    # preserve order, dedupe
    seen = set()
    out = []
    for cid in claim_ids:
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def strip_claim_blocks(text: str) -> str:
    cleaned = CLAIMS_BLOCK_RE.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def index_selected_claims(selected_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    claims = selected_payload.get("selected_claims")
    if claims is None:
        claims = selected_payload.get("validated_claims", [])
    return {
        str(c["claim_id"]): c
        for c in claims
        if "claim_id" in c
    }


def extract_claim_direction(claim: dict[str, Any]) -> str:
    """
    Coarse semantic direction for a claim.
    """
    label = str(claim.get("label", "")).strip().lower()
    claim_type = str(claim.get("claim_type", "")).strip().lower()
    delta_value = claim.get("delta_value")

    if label in {"improves", "increase", "better"}:
        return "positive"
    if label in {"degrades", "decrease", "worse"}:
        return "negative"
    if label.startswith("stable_") or claim_type in {"stable_floor", "stable_ceiling"}:
        return "stable"

    try:
        dv = float(delta_value)
        if dv > 0:
            return "positive"
        if dv < 0:
            return "negative"
        if dv == 0:
            return "stable"
    except (TypeError, ValueError):
        pass

    return "unknown"


def compute_linked_claim_diagnostics(linked_claims: list[dict[str, Any]]) -> dict[str, Any]:
    directions = [extract_claim_direction(c) for c in linked_claims]
    directions_non_unknown = sorted({d for d in directions if d != "unknown"})

    models = sorted(
        {str(c.get("model", "")).strip() for c in linked_claims if str(c.get("model", "")).strip()}
    )
    prompts = sorted(
        {str(c.get("prompt_id", "")).strip() for c in linked_claims if str(c.get("prompt_id", "")).strip()}
    )
    strengths = sorted(
        {str(c.get("claim_strength", "")).strip() for c in linked_claims if str(c.get("claim_strength", "")).strip()}
    )

    has_mixed_directions = len(directions_non_unknown) > 1
    has_mixed_models = len(models) > 1
    has_mixed_prompts = len(prompts) > 1
    has_mixed_strengths = len(strengths) > 1

    return {
        "claim_directions": directions,
        "direction_set": directions_non_unknown,
        "n_distinct_directions": len(directions_non_unknown),
        "has_mixed_directions": has_mixed_directions,
        "linked_models": models,
        "linked_prompt_ids": prompts,
        "linked_claim_strengths": strengths,
        "has_mixed_models": has_mixed_models,
        "has_mixed_prompts": has_mixed_prompts,
        "has_mixed_strengths": has_mixed_strengths,
    }


def parse_narrative(
    selected_claims_payload: dict[str, Any],
    narrative_payload: dict[str, Any],
) -> dict[str, Any]:
    narrative_text = narrative_payload.get("narrative", "")
    if not isinstance(narrative_text, str):
        raise ValueError("Narrative payload missing string field 'narrative'")

    items = split_narrative_sections(narrative_text)
    claim_index = index_selected_claims(selected_claims_payload)

    parsed_items: list[dict[str, Any]] = []
    n_with_claims = 0
    n_missing_claim_refs = 0
    n_unknown_claim_ids = 0

    for i, item in enumerate(items, start=1):
        raw_text = item["raw_text"]
        claim_ids = extract_claim_ids(raw_text)
        clean_text = strip_claim_blocks(raw_text)

        linked_claims = []
        unknown_claim_ids = []

        for cid in claim_ids:
            claim = claim_index.get(cid)
            if claim is None:
                unknown_claim_ids.append(cid)
            else:
                linked_claims.append(claim)
        
        linked_claim_diag = compute_linked_claim_diagnostics(linked_claims)

        if claim_ids:
            n_with_claims += 1
        elif item["section"] != "cautions":
            n_missing_claim_refs += 1

        n_unknown_claim_ids += len(unknown_claim_ids)

        parsed_items.append(
            {
                "item_id": i,
                "section": item["section"],
                "raw_text": raw_text,
                "clean_text": clean_text,
                "claim_ids": claim_ids,
                "n_claim_ids": len(claim_ids),
                "unknown_claim_ids": unknown_claim_ids,
                "n_unknown_claim_ids": len(unknown_claim_ids),
                "linked_claims": linked_claims,
                "n_linked_claims": len(linked_claims),

                # mixed-claim diagnostics
                "claim_directions": linked_claim_diag["claim_directions"],
                "direction_set": linked_claim_diag["direction_set"],
                "n_distinct_directions": linked_claim_diag["n_distinct_directions"],
                "has_mixed_directions": linked_claim_diag["has_mixed_directions"],
                "linked_models": linked_claim_diag["linked_models"],
                "linked_prompt_ids": linked_claim_diag["linked_prompt_ids"],
                "linked_claim_strengths": linked_claim_diag["linked_claim_strengths"],
                "has_mixed_models": linked_claim_diag["has_mixed_models"],
                "has_mixed_prompts": linked_claim_diag["has_mixed_prompts"],
                "has_mixed_strengths": linked_claim_diag["has_mixed_strengths"],
            }
        )

    return {
        "source_selected_claims_json": selected_claims_payload.get("_source_path"),
        "source_narrative_json": narrative_payload.get("_source_path"),
        "suite_name": selected_claims_payload.get("suite_name"),
        "metric": selected_claims_payload.get("metric"),
        "baseline_experiment": selected_claims_payload.get("baseline_experiment"),
        "comparison_experiments": selected_claims_payload.get("comparison_experiments", []),
        "repair_mode": narrative_payload.get("repair_mode"),
        "repair_iteration": narrative_payload.get("repair_iteration"),
        "parent_narrative_json": narrative_payload.get("parent_narrative_json"),
        "target_claim_ids": narrative_payload.get("target_claim_ids", []),
        "summary": {
            "n_items": len(parsed_items),
            "n_items_with_claim_refs": n_with_claims,
            "n_items_missing_claim_refs": n_missing_claim_refs,
            "n_unknown_claim_ids": n_unknown_claim_ids,
            "n_mixed_direction_items": sum(
                1 for item in parsed_items if item.get("has_mixed_directions") is True
            ),
            "n_mixed_model_items": sum(
                1 for item in parsed_items if item.get("has_mixed_models") is True
            ),
            "n_mixed_prompt_items": sum(
                1 for item in parsed_items if item.get("has_mixed_prompts") is True
            ),
            "n_mixed_strength_items": sum(
                1 for item in parsed_items if item.get("has_mixed_strengths") is True
            ),
        },
        "items": parsed_items,
    }


def default_output_path(narrative_json: Path) -> Path:
    stem = narrative_json.stem.replace("__narrative_from_claims", "")
    return narrative_json.parent / f"{stem}__parsed_narrative.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse claim IDs from a narrative and link them to selected claims."
    )
    parser.add_argument("--selected-claims-json", required=True)
    parser.add_argument("--results-root", default="benchmarks/results")
    parser.add_argument("--narrative-json", required=True)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selected_claims_path = Path(args.selected_claims_json).resolve()
    narrative_path = Path(args.narrative_json).resolve()
    
    if not selected_claims_path.exists():
        raise FileNotFoundError(f"selected claims json not found: {selected_claims_path}")

    if not narrative_path.exists():
        raise FileNotFoundError(f"narrative json not found: {narrative_path}")
    
    results_root = Path(args.results_root).resolve()

    selected_claims_payload = load_json(selected_claims_path)
    selected_claims_payload["_source_path"] = str(selected_claims_path)

    narrative_payload = load_json(narrative_path)
    narrative_payload["_source_path"] = str(narrative_path)

    parsed_payload = parse_narrative(
        selected_claims_payload=selected_claims_payload,
        narrative_payload=narrative_payload,
    )
    parsed_payload["results_root"] = str(results_root)
    parsed_payload["source_selected_claims_json"] = str(selected_claims_path)
    parsed_payload["source_narrative_json"] = str(narrative_path)

    output_path = (
        Path(args.output_json).resolve()
        if args.output_json
        else default_output_path(narrative_path)
    )

    save_json(parsed_payload, output_path)
    print(f"Saved parsed narrative JSON: {output_path}")


if __name__ == "__main__":
    main()