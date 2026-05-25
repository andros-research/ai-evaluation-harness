#!/usr/bin/env python3
"""
Build a FRED source-to-narrative traceability summary.

v1.6.5 scaffold:
- reads FRED claims, selected claims, narrative audit, and repair plan artifacts
- joins claim-level metadata to selected-claim ranks and narrative bullet audit status
- writes durable traceability artifacts for downstream dashboard/demo use

This layer makes the v1.6 artifact spine inspectable in one place.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


TRACEABILITY_SCHEMA_VERSION = "fred_traceability_summary_v0_1"
TRACEABILITY_METHOD = "claim_selection_narrative_audit_join"

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CLAIMS_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "fred_claims"
    / "fred_claims.json"
)

DEFAULT_SELECTED_CLAIMS_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "fred_claims"
    / "selected_fred_claims.json"
)

DEFAULT_AUDIT_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "fred_audits"
    / "fred_narrative_audit.json"
)

DEFAULT_REPAIR_PLAN_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "fred_repairs"
    / "fred_repair_plan.json"
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "fred_traceability"
)


TRACEABILITY_COLUMNS = [
    "claim_id",
    "source_type",
    "source_series",
    "source_series_name",
    "source_observation_date",
    "claim_type",
    "metric_name",
    "comparison_window",
    "current_value",
    "prior_value",
    "delta_value",
    "direction",
    "claim_text",
    "eligible_for_narrative",
    "was_selected",
    "selection_rank",
    "selection_method",
    "was_cited",
    "citation_count",
    "narrative_bullet_index",
    "narrative_bullet_text",
    "audit_citation_status",
    "audit_issue_type",
    "repair_needed",
    "repair_action_count",
    "repair_action_ids",
]


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> object:
    """Load a JSON artifact."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    """Write JSON with stable formatting."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_claim_lookup(claims: list[dict]) -> dict[str, dict]:
    """Build a claim_id lookup and reject duplicate IDs."""
    lookup: dict[str, dict] = {}

    for claim in claims:
        claim_id = claim.get("claim_id")
        if not claim_id:
            raise ValueError("Claim missing claim_id.")
        if claim_id in lookup:
            raise ValueError(f"Duplicate claim_id: {claim_id}")
        lookup[claim_id] = claim

    return lookup


def build_selected_lookup(selected_claims: list[dict]) -> dict[str, dict]:
    """Build selected claim lookup by claim_id."""
    return build_claim_lookup(selected_claims)


def build_bullet_lookup_from_audit(audit_payload: dict) -> dict[str, list[dict]]:
    """Map cited claim IDs to bullet audit records."""
    lookup: dict[str, list[dict]] = {}

    for idx, bullet in enumerate(audit_payload.get("bullet_audits", []), start=1):
        for claim_id in bullet.get("cited_claim_ids", []):
            lookup.setdefault(claim_id, []).append(
                {
                    "bullet_index": idx,
                    "bullet_text": bullet.get("bullet_text"),
                    "citation_status": bullet.get("citation_status"),
                    "issue_type": bullet.get("issue_type"),
                }
            )

    return lookup


def build_repair_lookup(repair_plan: dict) -> dict[str, list[dict]]:
    """Map claim IDs to repair actions where available."""
    lookup: dict[str, list[dict]] = {}

    for action in repair_plan.get("repair_actions", []):
        claim_id = action.get("claim_id")
        if claim_id:
            lookup.setdefault(claim_id, []).append(action)

        for cited_claim_id in action.get("cited_claim_ids", []):
            lookup.setdefault(cited_claim_id, []).append(action)

        for unknown_claim_id in action.get("unknown_claim_ids", []):
            lookup.setdefault(unknown_claim_id, []).append(action)

    return lookup


def build_traceability_rows(
    *,
    claims: list[dict],
    selected_claims: list[dict],
    audit_payload: dict,
    repair_plan: dict,
) -> list[dict]:
    """Build claim-level traceability rows."""
    selected_lookup = build_selected_lookup(selected_claims)
    bullet_lookup = build_bullet_lookup_from_audit(audit_payload)
    repair_lookup = build_repair_lookup(repair_plan)

    repair_needed = bool(repair_plan.get("repair_needed"))

    rows: list[dict] = []

    for claim in claims:
        claim_id = claim["claim_id"]

        selected = selected_lookup.get(claim_id)
        bullet_records = bullet_lookup.get(claim_id, [])
        repair_actions = repair_lookup.get(claim_id, [])

        first_bullet = bullet_records[0] if bullet_records else {}

        rows.append(
            {
                "claim_id": claim_id,
                "source_type": claim.get("source_type"),
                "source_series": claim.get("source_series"),
                "source_series_name": claim.get("source_series_name"),
                "source_observation_date": claim.get("source_observation_date"),
                "claim_type": claim.get("claim_type"),
                "metric_name": claim.get("metric_name"),
                "comparison_window": claim.get("comparison_window"),
                "current_value": claim.get("current_value"),
                "prior_value": claim.get("prior_value"),
                "delta_value": claim.get("delta_value"),
                "direction": claim.get("direction"),
                "claim_text": claim.get("claim_text"),
                "eligible_for_narrative": claim.get("eligible_for_narrative"),
                "was_selected": selected is not None,
                "selection_rank": selected.get("selection_rank") if selected else None,
                "selection_method": selected.get("selection_method") if selected else None,
                "was_cited": bool(bullet_records),
                "citation_count": len(bullet_records),
                "narrative_bullet_index": first_bullet.get("bullet_index"),
                "narrative_bullet_text": first_bullet.get("bullet_text"),
                "audit_citation_status": first_bullet.get("citation_status"),
                "audit_issue_type": first_bullet.get("issue_type"),
                "repair_needed": repair_needed,
                "repair_action_count": len(repair_actions),
                "repair_action_ids": [
                    action.get("action_id") for action in repair_actions
                ],
            }
        )

    return rows


def validate_traceability_rows(rows: list[dict]) -> None:
    """Validate traceability rows before writing artifacts."""
    errors: list[str] = []

    for idx, row in enumerate(rows):
        claim_id = row.get("claim_id", f"<row {idx}>")

        for field in ["claim_id", "source_series", "claim_text", "was_selected", "was_cited"]:
            if field not in row:
                errors.append(f"{claim_id}: missing required field {field!r}")
                continue

            value = row[field]
            if value is None:
                errors.append(f"{claim_id}: required field {field!r} is None")
            if isinstance(value, str) and not value.strip():
                errors.append(f"{claim_id}: required field {field!r} is empty")

        if not isinstance(row.get("was_selected"), bool):
            errors.append(f"{claim_id}: was_selected must be boolean")

        if not isinstance(row.get("was_cited"), bool):
            errors.append(f"{claim_id}: was_cited must be boolean")

    if errors:
        joined = "\n".join(f"- {err}" for err in errors)
        raise ValueError(f"FRED traceability validation failed:\n{joined}")


def write_traceability_artifacts(
    *,
    claims_path: Path,
    selected_claims_path: Path,
    audit_path: Path,
    repair_plan_path: Path,
    output_dir: Path,
) -> None:
    """Build and write traceability summary artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_at = utc_now_iso()

    claims_payload = load_json(claims_path)
    selected_claims_payload = load_json(selected_claims_path)
    audit_payload = load_json(audit_path)
    repair_plan_payload = load_json(repair_plan_path)

    if not isinstance(claims_payload, list):
        raise ValueError("Claims payload must be a list.")
    if not isinstance(selected_claims_payload, list):
        raise ValueError("Selected claims payload must be a list.")
    if not isinstance(audit_payload, dict):
        raise ValueError("Audit payload must be an object.")
    if not isinstance(repair_plan_payload, dict):
        raise ValueError("Repair plan payload must be an object.")

    rows = build_traceability_rows(
        claims=claims_payload,
        selected_claims=selected_claims_payload,
        audit_payload=audit_payload,
        repair_plan=repair_plan_payload,
    )
    validate_traceability_rows(rows)

    traceability_csv = output_dir / "fred_traceability_summary.csv"
    traceability_json = output_dir / "fred_traceability_summary.json"
    metadata_json = output_dir / "fred_traceability_summary_metadata.json"

    df = pd.DataFrame(rows, columns=TRACEABILITY_COLUMNS)

    csv_df = df.copy()
    if not csv_df.empty:
        csv_df["repair_action_ids"] = csv_df["repair_action_ids"].apply(
            lambda value: json.dumps(value, sort_keys=True)
        )

    csv_df.to_csv(traceability_csv, index=False)
    write_json(traceability_json, rows)

    metadata = {
        "traceability_schema_version": TRACEABILITY_SCHEMA_VERSION,
        "traceability_method": TRACEABILITY_METHOD,
        "generated_at": generated_at,
        "n_claims": int(len(claims_payload)),
        "n_selected_claims": int(len(selected_claims_payload)),
        "n_traceability_rows": int(len(rows)),
        "n_cited_claims": int(sum(1 for row in rows if row["was_cited"])),
        "n_uncited_claims": int(sum(1 for row in rows if not row["was_cited"])),
        "audit_pass": bool(audit_payload.get("audit_pass")),
        "repair_needed": bool(repair_plan_payload.get("repair_needed")),
        "input_files": {
            "claims_json": str(claims_path),
            "selected_claims_json": str(selected_claims_path),
            "audit_json": str(audit_path),
            "repair_plan_json": str(repair_plan_path),
        },
        "output_files": {
            "traceability_csv": str(traceability_csv),
            "traceability_json": str(traceability_json),
            "metadata_json": str(metadata_json),
        },
    }
    write_json(metadata_json, metadata)

    print("Wrote FRED traceability summary artifacts:")
    print(f"  {traceability_csv}")
    print(f"  {traceability_json}")
    print(f"  {metadata_json}")
    print(f"n_traceability_rows={len(rows)}")
    print(f"n_cited_claims={metadata['n_cited_claims']}")
    print(f"audit_pass={metadata['audit_pass']}")
    print(f"repair_needed={metadata['repair_needed']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FRED source-to-narrative traceability summary artifacts."
    )
    parser.add_argument(
        "--claims",
        type=Path,
        default=DEFAULT_CLAIMS_PATH,
        help="Input fred_claims.json file.",
    )
    parser.add_argument(
        "--selected-claims",
        type=Path,
        default=DEFAULT_SELECTED_CLAIMS_PATH,
        help="Input selected_fred_claims.json file.",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=DEFAULT_AUDIT_PATH,
        help="Input fred_narrative_audit.json file.",
    )
    parser.add_argument(
        "--repair-plan",
        type=Path,
        default=DEFAULT_REPAIR_PLAN_PATH,
        help="Input fred_repair_plan.json file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where traceability summary artifacts will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_traceability_artifacts(
        claims_path=args.claims,
        selected_claims_path=args.selected_claims,
        audit_path=args.audit,
        repair_plan_path=args.repair_plan,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()