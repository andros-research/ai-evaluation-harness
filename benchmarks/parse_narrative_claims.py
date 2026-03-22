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


def extract_claim_ids(text: str) -> list[str]:
    blocks = CLAIMS_BLOCK_RE.findall(text)
    claim_ids: list[str] = []

    for block in blocks:
        parts = [p.strip() for p in block.split(",")]
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
    claims = selected_payload.get("selected_claims", [])
    return {
        str(c["claim_id"]): c
        for c in claims
        if "claim_id" in c
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

        if claim_ids:
            n_with_claims += 1
        else:
            n_missing_claim_refs += 1

        n_unknown_claim_ids += len(unknown_claim_ids)

        parsed_items.append(
            {
                "item_id": i,
                "section": item["section"],
                "raw_text": raw_text,
                "clean_text": clean_text,
                "claim_ids": claim_ids,
                "unknown_claim_ids": unknown_claim_ids,
                "linked_claims": linked_claims,
            }
        )

    return {
        "source_selected_claims_json": selected_claims_payload.get("_source_path"),
        "source_narrative_json": narrative_payload.get("_source_path"),
        "suite_name": selected_claims_payload.get("suite_name"),
        "metric": selected_claims_payload.get("metric"),
        "baseline_experiment": selected_claims_payload.get("baseline_experiment"),
        "comparison_experiments": selected_claims_payload.get("comparison_experiments", []),
        "summary": {
            "n_items": len(parsed_items),
            "n_items_with_claim_refs": n_with_claims,
            "n_items_missing_claim_refs": n_missing_claim_refs,
            "n_unknown_claim_ids": n_unknown_claim_ids,
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

    parsed_payload = parse_narrative(
        selected_claims_payload=selected_claims_payload,
        narrative_payload=narrative_payload,
    )

    output_path = (
        Path(args.output_json).resolve()
        if args.output_json
        else default_output_path(narrative_path)
    )

    save_json(parsed_payload, output_path)
    print(f"Saved parsed narrative JSON: {output_path}")


if __name__ == "__main__":
    main()