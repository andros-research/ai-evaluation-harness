#!/usr/bin/env python3
"""
Generate a simple claim-cited FRED narrative from selected FRED claims.

v1.6.2 scaffold:
- reads selected FRED claim artifacts
- emits a constrained markdown narrative
- every empirical bullet cites exactly one source claim ID
- writes durable narrative and metadata artifacts

This first version is deterministic/template-based. Later versions may add
LLM-generated narrative text while preserving the same citation contract.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import re


NARRATIVE_SCHEMA_VERSION = "fred_narrative_v0_1"
GENERATION_METHOD = "deterministic_claim_bullets"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "fred_claims"
    / "selected_fred_claims.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "results" / "fred_narratives"


REQUIRED_SELECTED_FIELDS = [
    "claim_id",
    "claim_text",
    "source_series",
    "source_observation_date",
    "claim_type",
    "metric_name",
    "comparison_window",
    "direction",
    "supporting_values",
    "selection_rank",
    "selection_method",
    "selection_schema_version",
]


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def load_selected_claims(path: Path) -> list[dict]:
    """Load selected FRED claims from JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Selected claims file does not exist: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, list):
        raise ValueError(f"Expected list of selected claim records in {path}")

    return payload


def validate_selected_claims(selected_claims: list[dict]) -> None:
    """Validate selected claims before narrative generation."""
    errors: list[str] = []

    for idx, claim in enumerate(selected_claims):
        claim_id = claim.get("claim_id", f"<row {idx}>")

        for field in REQUIRED_SELECTED_FIELDS:
            if field not in claim:
                errors.append(f"{claim_id}: missing required field {field!r}")
                continue

            value = claim[field]

            if value is None:
                errors.append(f"{claim_id}: required field {field!r} is None")
                continue

            if isinstance(value, str) and not value.strip():
                errors.append(f"{claim_id}: required field {field!r} is empty")

        if not isinstance(claim.get("supporting_values"), dict):
            errors.append(f"{claim_id}: supporting_values must be a dict")

        if not isinstance(claim.get("selection_rank"), int):
            errors.append(f"{claim_id}: selection_rank must be an integer")

    if errors:
        joined = "\n".join(f"- {err}" for err in errors)
        raise ValueError(f"Selected claim validation failed:\n{joined}")


def render_claim_bullet(claim: dict) -> str:
    """Render one selected claim as a cited narrative bullet."""
    claim_text = claim["claim_text"].rstrip(".")
    claim_id = claim["claim_id"]
    return f"- {claim_text}. [CLAIMS: {claim_id}]"


def build_narrative_markdown(
    selected_claims: list[dict],
    generated_at: str,
) -> str:
    """Build a deterministic claim-cited markdown narrative."""
    if not selected_claims:
        return (
            "# FRED Macro Narrative\n\n"
            f"Generated at: {generated_at}\n\n"
            "No selected FRED claims were available for narrative generation.\n"
        )

    comparison_windows = sorted(
        {claim.get("comparison_window") for claim in selected_claims if claim.get("comparison_window")}
    )
    observation_dates = sorted(
        {claim.get("source_observation_date") for claim in selected_claims if claim.get("source_observation_date")}
    )

    window_text = ", ".join(comparison_windows)
    date_text = ", ".join(observation_dates)

    bullets = "\n".join(render_claim_bullet(claim) for claim in selected_claims)

    return (
        "# FRED Macro Narrative\n\n"
        f"Generated at: {generated_at}\n\n"
        "## Context\n\n"
        f"This deterministic narrative uses selected FRED-native claims for comparison window(s): {window_text}.\n"
        f"The current source observation date is: {date_text}.\n\n"
        "## Claim-Cited Summary\n\n"
        f"{bullets}\n"
    )


def extract_claim_ids(selected_claims: list[dict]) -> list[str]:
    """Return claim IDs in selected order."""
    return [claim["claim_id"] for claim in selected_claims]


def extract_cited_claim_ids(narrative_text: str) -> list[str]:
    """Extract claim IDs cited in [CLAIMS: ...] blocks."""
    cited: list[str] = []

    matches = re.findall(r"\[CLAIMS:\s*([^\]]+)\]", narrative_text)

    for match in matches:
        parts = [part.strip() for part in match.split(",")]
        cited.extend(part for part in parts if part)

    return cited


def validate_narrative_citations(
    *,
    narrative_text: str,
    selected_claims: list[dict],
) -> None:
    """Validate narrative citation coverage against selected claims."""
    selected_claim_ids = extract_claim_ids(selected_claims)
    selected_claim_id_set = set(selected_claim_ids)

    cited_claim_ids = extract_cited_claim_ids(narrative_text)
    cited_claim_id_set = set(cited_claim_ids)

    errors: list[str] = []

    missing_from_narrative = [
        claim_id for claim_id in selected_claim_ids if claim_id not in cited_claim_id_set
    ]
    if missing_from_narrative:
        errors.append(
            "Selected claim IDs missing from narrative citations: "
            + ", ".join(missing_from_narrative)
        )

    unknown_citations = [
        claim_id for claim_id in cited_claim_ids if claim_id not in selected_claim_id_set
    ]
    if unknown_citations:
        errors.append(
            "Narrative cites unknown claim IDs: "
            + ", ".join(unknown_citations)
        )

    if len(cited_claim_ids) != len(set(cited_claim_ids)):
        errors.append("Narrative contains duplicate claim citations.")

    if errors:
        joined = "\n".join(f"- {err}" for err in errors)
        raise ValueError(f"FRED narrative citation validation failed:\n{joined}")


def write_json(path: Path, payload: object) -> None:
    """Write JSON with stable formatting."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_narrative_artifacts(
    *,
    input_path: Path,
    output_dir: Path,
) -> None:
    """Generate and write FRED narrative artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_at = utc_now_iso()
    selected_claims = load_selected_claims(input_path)
    validate_selected_claims(selected_claims)

    narrative_md = build_narrative_markdown(
        selected_claims=selected_claims,
        generated_at=generated_at,
    )
    
    validate_narrative_citations(
        narrative_text=narrative_md,
        selected_claims=selected_claims,
    )

    narrative_path = output_dir / "fred_narrative.md"
    metadata_path = output_dir / "fred_narrative_metadata.json"

    narrative_path.write_text(narrative_md, encoding="utf-8")

    metadata = {
        "narrative_schema_version": NARRATIVE_SCHEMA_VERSION,
        "generation_method": GENERATION_METHOD,
        "input_file": str(input_path),
        "generated_at": generated_at,
        "n_selected_claims": int(len(selected_claims)),
        "used_claim_ids": extract_claim_ids(selected_claims),
        "cited_claim_ids": extract_cited_claim_ids(narrative_md),
        "citation_validation": {
            "all_selected_claims_cited": True,
            "all_citations_known": True,
            "duplicate_citations": False,
        },
        "output_files": {
            "narrative_md": str(narrative_path),
            "metadata_json": str(metadata_path),
        },
    }
    write_json(metadata_path, metadata)

    print("Wrote FRED narrative artifacts:")
    print(f"  {narrative_path}")
    print(f"  {metadata_path}")
    print(f"n_selected_claims={len(selected_claims)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a claim-cited FRED macro narrative."
    )
    parser.add_argument(
        "--input-claims",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input selected_fred_claims.json file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where FRED narrative artifacts will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_narrative_artifacts(
        input_path=args.input_claims,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()