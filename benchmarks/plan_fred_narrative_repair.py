#!/usr/bin/env python3
"""
Plan repairs for a FRED narrative based on an audit artifact.

v1.6.4 scaffold:
- reads fred_narrative_audit.json
- reads fred_narrative.md
- reads selected_fred_claims.json
- determines whether repair is needed
- maps audit failure categories to repair actions
- writes a durable repair plan artifact

This version does not rewrite the narrative. It only produces an explicit
repair plan that future versions can apply.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


REPAIR_PLAN_SCHEMA_VERSION = "fred_repair_plan_v0_1"
REPAIR_METHOD = "citation_audit_repair_plan"

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_AUDIT_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "fred_audits"
    / "fred_narrative_audit.json"
)

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
    / "fred_repairs"
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


def repair_action_for_issue(issue_type: str | None) -> str | None:
    """Map bullet-level audit issue types to repair actions."""
    if issue_type is None:
        return None

    mapping = {
        "missing_claim_citation": "add_valid_claim_citation_or_remove_bullet",
        "unknown_claim_id": "replace_unknown_claim_id_or_remove_citation",
    }
    return mapping.get(issue_type, "manual_review_required")


def build_bullet_repair_actions(audit_payload: dict) -> list[dict]:
    """Build repair actions from bullet-level audit records."""
    actions: list[dict] = []

    for idx, bullet_audit in enumerate(audit_payload.get("bullet_audits", []), start=1):
        issue_type = bullet_audit.get("issue_type")
        repair_action = repair_action_for_issue(issue_type)

        if repair_action is None:
            continue

        actions.append(
            {
                "action_id": f"repair_bullet_{idx}",
                "action_scope": "bullet",
                "bullet_index": idx,
                "issue_type": issue_type,
                "repair_action": repair_action,
                "bullet_text": bullet_audit.get("bullet_text"),
                "cited_claim_ids": bullet_audit.get("cited_claim_ids", []),
                "unknown_claim_ids": bullet_audit.get("unknown_claim_ids", []),
                "status": "planned",
            }
        )

    return actions


def build_missing_selected_claim_actions(audit_payload: dict) -> list[dict]:
    """Build repair actions for selected claims missing from the narrative."""
    actions: list[dict] = []

    for idx, claim_id in enumerate(
        audit_payload.get("selected_claims_missing_from_narrative", []),
        start=1,
    ):
        actions.append(
            {
                "action_id": f"add_missing_selected_claim_{idx}",
                "action_scope": "narrative",
                "issue_type": "selected_claim_missing_from_narrative",
                "repair_action": "add_claim_bullet_or_relax_strict_coverage",
                "claim_id": claim_id,
                "status": "planned",
            }
        )

    return actions


def build_duplicate_citation_actions(audit_payload: dict) -> list[dict]:
    """Build repair actions for duplicate claim citations."""
    actions: list[dict] = []

    for idx, claim_id in enumerate(audit_payload.get("duplicate_citations", []), start=1):
        actions.append(
            {
                "action_id": f"dedupe_claim_citation_{idx}",
                "action_scope": "narrative",
                "issue_type": "duplicate_claim_citation",
                "repair_action": "deduplicate_claim_citation",
                "claim_id": claim_id,
                "status": "planned",
            }
        )

    return actions


def build_repair_plan(
    *,
    audit_payload: dict,
    narrative_text: str,
    selected_claims: list[dict],
    planned_at: str,
) -> dict:
    """Build a repair plan from the audit payload."""
    claim_lookup = build_claim_lookup(selected_claims)

    audit_pass = bool(audit_payload.get("audit_pass"))
    audit_errors = audit_payload.get("errors", [])

    repair_actions: list[dict] = []
    repair_actions.extend(build_bullet_repair_actions(audit_payload))
    repair_actions.extend(build_missing_selected_claim_actions(audit_payload))
    repair_actions.extend(build_duplicate_citation_actions(audit_payload))

    repair_needed = not audit_pass

    return {
        "repair_plan_schema_version": REPAIR_PLAN_SCHEMA_VERSION,
        "repair_method": REPAIR_METHOD,
        "planned_at": planned_at,
        "repair_needed": repair_needed,
        "audit_pass": audit_pass,
        "audit_errors": audit_errors,
        "n_selected_claims": len(selected_claims),
        "n_known_claim_ids": len(claim_lookup),
        "n_repair_actions": len(repair_actions),
        "repair_actions": repair_actions,
        "inputs_summary": {
            "narrative_chars": len(narrative_text),
            "audit_schema_version": audit_payload.get("audit_schema_version"),
            "audit_method": audit_payload.get("audit_method"),
            "audit_file_passed": audit_pass,
        },
    }


def write_repair_plan_artifact(
    *,
    audit_path: Path,
    narrative_path: Path,
    selected_claims_path: Path,
    output_dir: Path,
) -> None:
    """Build and write a FRED narrative repair plan."""
    output_dir.mkdir(parents=True, exist_ok=True)

    planned_at = utc_now_iso()

    audit_payload = load_json(audit_path)
    narrative_text = load_text(narrative_path)
    selected_claims_payload = load_json(selected_claims_path)

    if not isinstance(audit_payload, dict):
        raise ValueError("Audit payload must be a JSON object.")
    if not isinstance(selected_claims_payload, list):
        raise ValueError("Selected claims payload must be a list.")

    repair_plan = build_repair_plan(
        audit_payload=audit_payload,
        narrative_text=narrative_text,
        selected_claims=selected_claims_payload,
        planned_at=planned_at,
    )

    repair_plan_path = output_dir / "fred_repair_plan.json"

    payload = {
        **repair_plan,
        "input_files": {
            "audit_json": str(audit_path),
            "narrative_md": str(narrative_path),
            "selected_claims_json": str(selected_claims_path),
        },
        "output_files": {
            "repair_plan_json": str(repair_plan_path),
        },
    }

    write_json(repair_plan_path, payload)

    print("Wrote FRED narrative repair plan artifact:")
    print(f"  {repair_plan_path}")
    print(f"repair_needed={repair_plan['repair_needed']}")
    print(f"audit_pass={repair_plan['audit_pass']}")
    print(f"n_repair_actions={repair_plan['n_repair_actions']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan repairs for a FRED narrative based on audit results."
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=DEFAULT_AUDIT_PATH,
        help="Input fred_narrative_audit.json file.",
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
        help="Directory where FRED repair plan artifacts will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_repair_plan_artifact(
        audit_path=args.audit,
        narrative_path=args.narrative,
        selected_claims_path=args.selected_claims,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()