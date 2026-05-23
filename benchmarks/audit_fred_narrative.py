#!/usr/bin/env python3
"""
Audit a FRED narrative against selected FRED claims.

v1.6.3 scaffold:
- reads a claim-cited FRED narrative markdown artifact
- reads selected FRED claims
- extracts [CLAIMS: ...] citation blocks
- verifies that cited claim IDs exist in the selected claim set
- optionally requires every selected claim to be cited
- writes a durable audit artifact

This is intentionally narrow. Later versions may validate numeric values,
directions, claim text overlap, unsupported macro interpretation, and repair
recommendations.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


AUDIT_SCHEMA_VERSION = "fred_narrative_audit_v0_1"
AUDIT_METHOD = "citation_coverage_audit"

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_NARRATIVE_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "fred_narratives"
    / "fred_narrative.md"
)

DEFAULT_SELECTED_CLAIMS_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "fred_claims"
    / "selected_fred_claims.json"
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "fred_audits"
)


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> object:
    """Load a JSON artifact."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_text(path: Path) -> str:
    """Load a text artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Text file does not exist: {path}")
    return path.read_text(encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    """Write JSON with stable formatting."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def extract_bullets(narrative_text: str) -> list[str]:
    """Extract markdown bullet lines from narrative text."""
    bullets: list[str] = []
    for line in narrative_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            bullets.append(stripped)
    return bullets


def extract_cited_claim_ids(text: str) -> list[str]:
    """Extract claim IDs cited in [CLAIMS: ...] blocks."""
    cited: list[str] = []
    matches = re.findall(r"\[CLAIMS:\s*([^\]]+)\]", text)

    for match in matches:
        parts = [part.strip() for part in match.split(",")]
        cited.extend(part for part in parts if part)

    return cited


def build_claim_lookup(selected_claims: list[dict]) -> dict[str, dict]:
    """Build lookup from claim_id to selected claim record."""
    lookup: dict[str, dict] = {}

    for claim in selected_claims:
        claim_id = claim.get("claim_id")
        if not claim_id:
            raise ValueError("Selected claim missing claim_id.")
        if claim_id in lookup:
            raise ValueError(f"Duplicate selected claim_id: {claim_id}")
        lookup[claim_id] = claim

    return lookup


def audit_bullet(
    *,
    bullet_text: str,
    selected_claim_ids: set[str],
) -> dict:
    """Audit one markdown bullet for claim citation validity."""
    cited_claim_ids = extract_cited_claim_ids(bullet_text)
    unknown_claim_ids = [
        claim_id for claim_id in cited_claim_ids if claim_id not in selected_claim_ids
    ]

    has_claim_citation = bool(cited_claim_ids)
    citation_status = "supported" if has_claim_citation and not unknown_claim_ids else "failed"

    if not has_claim_citation:
        issue_type = "missing_claim_citation"
    elif unknown_claim_ids:
        issue_type = "unknown_claim_id"
    else:
        issue_type = None

    return {
        "bullet_text": bullet_text,
        "cited_claim_ids": cited_claim_ids,
        "has_claim_citation": has_claim_citation,
        "unknown_claim_ids": unknown_claim_ids,
        "citation_status": citation_status,
        "issue_type": issue_type,
    }


def audit_narrative(
    *,
    narrative_text: str,
    selected_claims: list[dict],
    strict_selected_claim_coverage: bool = True,
) -> dict:
    """Audit narrative citation coverage against selected claims."""
    claim_lookup = build_claim_lookup(selected_claims)
    selected_claim_ids = set(claim_lookup)

    bullets = extract_bullets(narrative_text)
    bullet_audits = [
        audit_bullet(
            bullet_text=bullet,
            selected_claim_ids=selected_claim_ids,
        )
        for bullet in bullets
    ]

    cited_claim_ids = extract_cited_claim_ids(narrative_text)
    cited_claim_id_set = set(cited_claim_ids)

    unknown_citations = [
        claim_id for claim_id in cited_claim_ids if claim_id not in selected_claim_ids
    ]

    selected_claims_missing_from_narrative = [
        claim_id for claim_id in claim_lookup if claim_id not in cited_claim_id_set
    ]

    duplicate_citations = sorted(
        {
            claim_id
            for claim_id in cited_claim_ids
            if cited_claim_ids.count(claim_id) > 1
        }
    )

    bullets_missing_citations = [
        item for item in bullet_audits if not item["has_claim_citation"]
    ]

    bullets_with_unknown_citations = [
        item for item in bullet_audits if item["unknown_claim_ids"]
    ]

    errors: list[str] = []

    if unknown_citations:
        errors.append("unknown_claim_ids")

    if bullets_missing_citations:
        errors.append("bullets_missing_claim_citations")

    if duplicate_citations:
        errors.append("duplicate_claim_citations")

    if strict_selected_claim_coverage and selected_claims_missing_from_narrative:
        errors.append("selected_claims_missing_from_narrative")

    audit_pass = not errors

    return {
        "audit_pass": audit_pass,
        "errors": errors,
        "strict_selected_claim_coverage": strict_selected_claim_coverage,
        "n_selected_claims": len(selected_claims),
        "n_bullets": len(bullets),
        "n_citations": len(cited_claim_ids),
        "n_unique_citations": len(cited_claim_id_set),
        "selected_claim_ids": sorted(selected_claim_ids),
        "cited_claim_ids": cited_claim_ids,
        "unknown_citations": unknown_citations,
        "duplicate_citations": duplicate_citations,
        "selected_claims_missing_from_narrative": selected_claims_missing_from_narrative,
        "n_bullets_missing_citations": len(bullets_missing_citations),
        "n_bullets_with_unknown_citations": len(bullets_with_unknown_citations),
        "bullet_audits": bullet_audits,
    }


def write_audit_artifact(
    *,
    narrative_path: Path,
    selected_claims_path: Path,
    output_dir: Path,
    strict_selected_claim_coverage: bool,
) -> None:
    """Run audit and write audit artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)

    audited_at = utc_now_iso()
    narrative_text = load_text(narrative_path)
    selected_claims_payload = load_json(selected_claims_path)

    if not isinstance(selected_claims_payload, list):
        raise ValueError("Selected claims payload must be a list.")

    audit_result = audit_narrative(
        narrative_text=narrative_text,
        selected_claims=selected_claims_payload,
        strict_selected_claim_coverage=strict_selected_claim_coverage,
    )

    audit_path = output_dir / "fred_narrative_audit.json"

    payload = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit_method": AUDIT_METHOD,
        "audited_at": audited_at,
        "narrative_file": str(narrative_path),
        "selected_claims_file": str(selected_claims_path),
        **audit_result,
        "output_files": {
            "audit_json": str(audit_path),
        },
    }

    write_json(audit_path, payload)

    print("Wrote FRED narrative audit artifact:")
    print(f"  {audit_path}")
    print(f"audit_pass={audit_result['audit_pass']}")
    print(f"n_bullets={audit_result['n_bullets']}")
    print(f"n_citations={audit_result['n_citations']}")
    print(f"n_selected_claims={audit_result['n_selected_claims']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a FRED narrative against selected FRED claims."
    )
    parser.add_argument(
        "--narrative",
        type=Path,
        default=DEFAULT_NARRATIVE_PATH,
        help="Input fred_narrative.md file.",
    )
    parser.add_argument(
        "--selected-claims",
        type=Path,
        default=DEFAULT_SELECTED_CLAIMS_PATH,
        help="Input selected_fred_claims.json file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where FRED audit artifacts will be written.",
    )
    parser.add_argument(
        "--no-strict-selected-claim-coverage",
        action="store_true",
        help="Do not require every selected claim to be cited in the narrative.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_audit_artifact(
        narrative_path=args.narrative,
        selected_claims_path=args.selected_claims,
        output_dir=args.output_dir,
        strict_selected_claim_coverage=not args.no_strict_selected_claim_coverage,
    )


if __name__ == "__main__":
    main()