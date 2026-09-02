#!/usr/bin/env python3
"""
Relocate existing FRED claim citations to claim-complete uncited bullets.

This deterministic repair strategy targets planned bullet-level repairs with:

    issue_type = missing_claim_citation
    repair_action = add_valid_claim_citation_or_remove_bullet

Unlike structural normalization, this strategy is used when an uncited
Markdown bullet contains a complete empirical fingerprint for exactly one
selected claim.

Claim identity is established conservatively from:

    prior_value
    current_value
    delta_value magnitude

using exact numeric-token matching rather than the historical auditor's
substring-based numeric presence check.

The executor then relocates an already-existing citation for that claim
from a non-audited narrative line to the audited bullet.

It does not:
- invent new claim citations,
- rewrite substantive prose,
- repair direction vocabulary,
- move citations between audited bullets,
- guess when claim identity or citation location is ambiguous.

The citation multiset must be identical before and after execution.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audit_fred_narrative import (
    build_claim_lookup,
    extract_cited_claim_ids,
    normalize_number_text,
)


REPAIR_EXECUTION_SCHEMA_VERSION = (
    "fred_repair_execution_v0_1"
)

REPAIR_METHOD = (
    "deterministic_evidence_relocation"
)

REPAIR_STRATEGY = (
    "relocate_existing_claim_citations"
)

TARGET_ISSUE_TYPE = (
    "missing_claim_citation"
)

TARGET_REPAIR_ACTION = (
    "add_valid_claim_citation_or_remove_bullet"
)

BULLET_PATTERN = re.compile(
    r"^[-*]\s+"
)

CITATION_BLOCK_PATTERN = re.compile(
    r"\[CLAIMS:\s*([^\]]+)\]"
)


def utc_now_iso() -> str:
    """Return current UTC time in ISO-8601 format."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def load_json(path: Path) -> Any:
    """Load JSON from disk."""
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_text(path: Path) -> str:
    """Load UTF-8 text from disk."""
    return path.read_text(
        encoding="utf-8"
    )


def write_json(
    path: Path,
    payload: Any,
) -> None:
    """Write formatted JSON."""
    path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def selected_claim_records(
    payload: Any,
) -> list[dict[str, Any]]:
    """Extract selected-claim records from supported artifact shapes."""
    if isinstance(
        payload,
        list,
    ):
        claims = payload
    elif isinstance(
        payload,
        dict,
    ):
        claims = (
            payload.get("selected_claims")
            or payload.get("claims")
            or []
        )
    else:
        raise ValueError(
            "Selected claims payload must be "
            "a list or object."
        )

    if not isinstance(
        claims,
        list,
    ):
        raise ValueError(
            "Selected claims collection must "
            "be a list."
        )

    for claim in claims:
        if not isinstance(
            claim,
            dict,
        ):
            raise ValueError(
                "Each selected claim must be "
                "an object."
            )

    return claims


def markdown_bullet_line_indexes(
    lines: list[str],
) -> list[int]:
    """
    Return line indexes using the auditor's simple
    Markdown bullet shape: '- ' or '* '.
    """
    indexes: list[int] = []

    for idx, line in enumerate(
        lines
    ):
        if BULLET_PATTERN.match(
            line.strip()
        ):
            indexes.append(
                idx
            )

    return indexes


def strict_numeric_value_present(
    text: str,
    value: object,
) -> bool:
    """
    Match a normalized number only when it is not
    embedded inside a longer numeric token.
    """
    normalized = (
        normalize_number_text(
            value
        )
    )

    if not normalized:
        return False

    pattern = (
        r"(?<![\d.])"
        + re.escape(normalized)
        + r"(?!\d|\.\d)"
    )

    return (
        re.search(
            pattern,
            text,
        )
        is not None
    )


def strict_delta_value_present(
    text: str,
    value: object,
) -> bool:
    """
    Match delta magnitude while leaving direction
    validation to the independent auditor.
    """
    if value is None:
        return False

    if isinstance(
        value,
        (int, float),
    ):
        value = abs(
            value
        )

    return strict_numeric_value_present(
        text,
        value,
    )


def claim_identity_matches(
    *,
    bullet_text: str,
    claim: dict[str, Any],
) -> bool:
    """
    Return whether prior/current/delta uniquely identify
    this claim in the bullet.

    Direction is intentionally excluded from identity.
    """
    return all(
        (
            strict_numeric_value_present(
                bullet_text,
                claim.get(
                    "current_value"
                ),
            ),
            strict_numeric_value_present(
                bullet_text,
                claim.get(
                    "prior_value"
                ),
            ),
            strict_delta_value_present(
                bullet_text,
                claim.get(
                    "delta_value"
                ),
            ),
        )
    )


def candidate_repair_actions(
    repair_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    """Select planner actions targeted by this strategy."""
    return [
        action
        for action in repair_plan.get(
            "repair_actions",
            [],
        )
        if (
            action.get("action_scope")
            == "bullet"
            and action.get("issue_type")
            == TARGET_ISSUE_TYPE
            and action.get("repair_action")
            == TARGET_REPAIR_ACTION
            and action.get("status")
            == "planned"
        )
    ]


def citation_locations(
    *,
    lines: list[str],
    claim_id: str,
) -> list[dict[str, Any]]:
    """
    Find citation blocks containing one target claim ID.
    """
    locations: list[
        dict[str, Any]
    ] = []

    for line_index, line in enumerate(
        lines
    ):
        for match in (
            CITATION_BLOCK_PATTERN.finditer(
                line
            )
        ):
            claim_ids = [
                part.strip()
                for part
                in match.group(1).split(",")
                if part.strip()
            ]

            if claim_id not in claim_ids:
                continue

            locations.append(
                {
                    "line_index":
                        line_index,
                    "citation_block":
                        match.group(0),
                    "claim_ids":
                        claim_ids,
                }
            )

    return locations


def apply_repairs(
    *,
    narrative_text: str,
    audit_payload: dict[str, Any],
    repair_plan: dict[str, Any],
    selected_claims: list[dict[str, Any]],
) -> tuple[
    str,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Relocate existing citations to uniquely matched
    claim-complete uncited bullets.

    Ambiguous or unsupported cases abstain rather than guess.
    """
    lines = narrative_text.splitlines()

    bullet_line_indexes = (
        markdown_bullet_line_indexes(
            lines
        )
    )

    bullet_line_index_set = set(
        bullet_line_indexes
    )

    bullet_audits = (
        audit_payload.get(
            "bullet_audits",
            [],
        )
    )

    if len(
        bullet_line_indexes
    ) != len(
        bullet_audits
    ):
        raise ValueError(
            "Narrative bullet count does not "
            "match audit bullet count: "
            f"{len(bullet_line_indexes)} vs "
            f"{len(bullet_audits)}"
        )

    claim_lookup = (
        build_claim_lookup(
            selected_claims
        )
    )

    actions = (
        candidate_repair_actions(
            repair_plan
        )
    )

    prepared_actions: list[
        dict[str, Any]
    ] = []

    abstained_actions: list[
        dict[str, Any]
    ] = []

    seen_bullet_indexes: set[int] = (
        set()
    )

    for action in actions:
        bullet_index = action.get(
            "bullet_index"
        )

        if not isinstance(
            bullet_index,
            int,
        ):
            raise ValueError(
                "Repair action missing integer "
                "bullet_index: "
                f"{action.get('action_id')}"
            )

        if bullet_index < 1:
            raise ValueError(
                "Repair bullet_index must be "
                f"1-based: {bullet_index}"
            )

        if bullet_index in (
            seen_bullet_indexes
        ):
            raise ValueError(
                "Duplicate repair action for "
                f"bullet_index={bullet_index}"
            )

        seen_bullet_indexes.add(
            bullet_index
        )

        audit_position = (
            bullet_index - 1
        )

        if audit_position >= len(
            bullet_audits
        ):
            raise ValueError(
                "Repair bullet_index exceeds "
                "audit bullet count: "
                f"{bullet_index}"
            )

        bullet_audit = (
            bullet_audits[
                audit_position
            ]
        )

        if (
            bullet_audit.get(
                "issue_type"
            )
            != TARGET_ISSUE_TYPE
        ):
            raise ValueError(
                "Repair plan and audit disagree "
                "about issue type for "
                f"bullet_index={bullet_index}"
            )

        line_index = (
            bullet_line_indexes[
                audit_position
            ]
        )

        original_line = (
            lines[line_index]
        )

        audited_text = (
            bullet_audit.get(
                "bullet_text"
            )
        )

        planned_text = (
            action.get(
                "bullet_text"
            )
        )

        if (
            isinstance(
                audited_text,
                str,
            )
            and original_line.strip()
            != audited_text.strip()
        ):
            raise ValueError(
                "Narrative and audit bullet text "
                "do not match for "
                f"bullet_index={bullet_index}"
            )

        if (
            isinstance(
                planned_text,
                str,
            )
            and original_line.strip()
            != planned_text.strip()
        ):
            raise ValueError(
                "Narrative and repair-plan "
                "bullet text do not match for "
                f"bullet_index={bullet_index}"
            )

        if extract_cited_claim_ids(
            original_line
        ):
            abstained_actions.append(
                {
                    "action_id":
                        action.get(
                            "action_id"
                        ),
                    "bullet_index":
                        bullet_index,
                    "reason":
                        "destination_already_cited",
                }
            )
            continue

        matching_claim_ids = [
            claim_id
            for claim_id, claim
            in claim_lookup.items()
            if claim_identity_matches(
                bullet_text=original_line,
                claim=claim,
            )
        ]

        if len(
            matching_claim_ids
        ) != 1:
            abstained_actions.append(
                {
                    "action_id":
                        action.get(
                            "action_id"
                        ),
                    "bullet_index":
                        bullet_index,
                    "reason":
                        (
                            "no_unique_claim_identity"
                        ),
                    "matching_claim_ids":
                        matching_claim_ids,
                }
            )
            continue

        prepared_actions.append(
            {
                "action":
                    action,
                "bullet_index":
                    bullet_index,
                "destination_line_index":
                    line_index,
                "claim_id":
                    matching_claim_ids[0],
            }
        )

    claim_id_counts = Counter(
        item["claim_id"]
        for item in prepared_actions
    )

    duplicate_destination_claim_ids = {
        claim_id
        for claim_id, count
        in claim_id_counts.items()
        if count > 1
    }

    remaining_actions: list[
        dict[str, Any]
    ] = []

    for item in prepared_actions:
        claim_id = item[
            "claim_id"
        ]

        if claim_id in (
            duplicate_destination_claim_ids
        ):
            action = item[
                "action"
            ]

            abstained_actions.append(
                {
                    "action_id":
                        action.get(
                            "action_id"
                        ),
                    "bullet_index":
                        item[
                            "bullet_index"
                        ],
                    "reason":
                        (
                            "claim_matches_multiple_"
                            "destination_bullets"
                        ),
                    "claim_id":
                        claim_id,
                }
            )
            continue

        remaining_actions.append(
            item
        )

    prepared_actions = (
        remaining_actions
    )

    applied_actions: list[
        dict[str, Any]
    ] = []

    for item in prepared_actions:
        action = item[
            "action"
        ]

        claim_id = item[
            "claim_id"
        ]

        destination_line_index = item[
            "destination_line_index"
        ]

        locations = (
            citation_locations(
                lines=lines,
                claim_id=claim_id,
            )
        )

        if len(
            locations
        ) != 1:
            abstained_actions.append(
                {
                    "action_id":
                        action.get(
                            "action_id"
                        ),
                    "bullet_index":
                        item[
                            "bullet_index"
                        ],
                    "reason":
                        (
                            "citation_source_not_unique"
                        ),
                    "claim_id":
                        claim_id,
                    "n_citation_sources":
                        len(
                            locations
                        ),
                }
            )
            continue

        location = locations[0]

        source_line_index = (
            location[
                "line_index"
            ]
        )

        citation_block = (
            location[
                "citation_block"
            ]
        )

        cited_ids_in_block = (
            location[
                "claim_ids"
            ]
        )

        if len(
            cited_ids_in_block
        ) != 1:
            abstained_actions.append(
                {
                    "action_id":
                        action.get(
                            "action_id"
                        ),
                    "bullet_index":
                        item[
                            "bullet_index"
                        ],
                    "reason":
                        (
                            "multi_claim_citation_block"
                        ),
                    "claim_id":
                        claim_id,
                    "citation_claim_ids":
                        cited_ids_in_block,
                }
            )
            continue

        if source_line_index in (
            bullet_line_index_set
        ):
            abstained_actions.append(
                {
                    "action_id":
                        action.get(
                            "action_id"
                        ),
                    "bullet_index":
                        item[
                            "bullet_index"
                        ],
                    "reason":
                        (
                            "citation_source_is_"
                            "audited_bullet"
                        ),
                    "claim_id":
                        claim_id,
                    "source_line_number":
                        source_line_index + 1,
                }
            )
            continue

        if (
            source_line_index
            == destination_line_index
        ):
            abstained_actions.append(
                {
                    "action_id":
                        action.get(
                            "action_id"
                        ),
                    "bullet_index":
                        item[
                            "bullet_index"
                        ],
                    "reason":
                        (
                            "citation_already_on_"
                            "destination_line"
                        ),
                    "claim_id":
                        claim_id,
                }
            )
            continue

        source_before = (
            lines[
                source_line_index
            ]
        )

        destination_before = (
            lines[
                destination_line_index
            ]
        )

        if source_before.count(
            citation_block
        ) != 1:
            raise ValueError(
                "Expected exactly one citation "
                "block occurrence on source line "
                f"for claim_id={claim_id}"
            )

        source_after = (
            source_before.replace(
                citation_block,
                "",
                1,
            )
            .rstrip()
        )

        destination_after = (
            destination_before.rstrip()
            + " "
            + citation_block
        )

        lines[
            source_line_index
        ] = source_after

        lines[
            destination_line_index
        ] = destination_after

        applied_actions.append(
            {
                "action_id":
                    action.get(
                        "action_id"
                    ),
                "bullet_index":
                    item[
                        "bullet_index"
                    ],
                "issue_type":
                    TARGET_ISSUE_TYPE,
                "planned_repair_action":
                    TARGET_REPAIR_ACTION,
                "executed_strategy":
                    REPAIR_STRATEGY,
                "claim_id":
                    claim_id,
                "source_line_number":
                    source_line_index + 1,
                "destination_line_number":
                    destination_line_index + 1,
                "source_before":
                    source_before,
                "source_after":
                    source_after,
                "destination_before":
                    destination_before,
                "destination_after":
                    destination_after,
                "status":
                    "applied",
            }
        )

    repaired_text = "\n".join(
        lines
    )

    if narrative_text.endswith(
        "\n"
    ):
        repaired_text += "\n"

    citations_before = Counter(
        extract_cited_claim_ids(
            narrative_text
        )
    )

    citations_after = Counter(
        extract_cited_claim_ids(
            repaired_text
        )
    )

    if (
        citations_before
        != citations_after
    ):
        raise ValueError(
            "Citation conservation invariant "
            "failed: citation multiset changed."
        )

    return (
        repaired_text,
        applied_actions,
        abstained_actions,
    )


def write_repair_execution(
    *,
    narrative_path: Path,
    audit_path: Path,
    repair_plan_path: Path,
    selected_claims_path: Path,
    output_dir: Path,
) -> None:
    """
    Apply citation-relocation repairs and write
    durable output artifacts.
    """
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    narrative_text = load_text(
        narrative_path
    )

    audit_payload = load_json(
        audit_path
    )

    repair_plan = load_json(
        repair_plan_path
    )

    selected_claims_payload = (
        load_json(
            selected_claims_path
        )
    )

    if not isinstance(
        audit_payload,
        dict,
    ):
        raise ValueError(
            "Audit payload must be an object."
        )

    if not isinstance(
        repair_plan,
        dict,
    ):
        raise ValueError(
            "Repair plan must be an object."
        )

    selected_claims = (
        selected_claim_records(
            selected_claims_payload
        )
    )

    repaired_text, applied_actions, abstained_actions = (
        apply_repairs(
            narrative_text=narrative_text,
            audit_payload=audit_payload,
            repair_plan=repair_plan,
            selected_claims=selected_claims,
        )
    )

    repaired_narrative_path = (
        output_dir
        / "fred_narrative_repaired.md"
    )

    execution_path = (
        output_dir
        / "fred_repair_execution.json"
    )

    repaired_narrative_path.write_text(
        repaired_text,
        encoding="utf-8",
    )

    planned_actions = (
        repair_plan.get(
            "repair_actions",
            [],
        )
    )

    candidate_actions = (
        candidate_repair_actions(
            repair_plan
        )
    )

    execution = {
        "repair_execution_schema_version":
            REPAIR_EXECUTION_SCHEMA_VERSION,
        "repair_method":
            REPAIR_METHOD,
        "repair_strategy":
            REPAIR_STRATEGY,
        "executed_at":
            utc_now_iso(),
        "target_issue_type":
            TARGET_ISSUE_TYPE,
        "target_repair_action":
            TARGET_REPAIR_ACTION,
        "repair_applied":
            bool(
                applied_actions
            ),
        "n_planned_repair_actions":
            len(
                planned_actions
            ),
        "n_candidate_repair_actions":
            len(
                candidate_actions
            ),
        "n_actions_applied":
            len(
                applied_actions
            ),
        "n_actions_abstained":
            len(
                abstained_actions
            ),
        "applied_actions":
            applied_actions,
        "abstained_actions":
            abstained_actions,
        "citation_conservation": {
            "n_citations_before":
                len(
                    extract_cited_claim_ids(
                        narrative_text
                    )
                ),
            "n_citations_after":
                len(
                    extract_cited_claim_ids(
                        repaired_text
                    )
                ),
            "n_unique_citations_before":
                len(
                    set(
                        extract_cited_claim_ids(
                            narrative_text
                        )
                    )
                ),
            "n_unique_citations_after":
                len(
                    set(
                        extract_cited_claim_ids(
                            repaired_text
                        )
                    )
                ),
            "multiset_preserved":
                True,
        },
        "inputs": {
            "narrative_file":
                str(
                    narrative_path
                ),
            "audit_file":
                str(
                    audit_path
                ),
            "repair_plan_file":
                str(
                    repair_plan_path
                ),
            "selected_claims_file":
                str(
                    selected_claims_path
                ),
            "audit_pass_before":
                audit_payload.get(
                    "audit_pass"
                ),
            "audit_errors_before":
                audit_payload.get(
                    "errors",
                    [],
                ),
            "repair_needed":
                repair_plan.get(
                    "repair_needed"
                ),
        },
        "outputs": {
            "repaired_narrative":
                str(
                    repaired_narrative_path
                ),
            "repair_execution_json":
                str(
                    execution_path
                ),
        },
    }

    write_json(
        execution_path,
        execution,
    )

    print(
        "Wrote repaired FRED narrative:"
    )
    print(
        f"  {repaired_narrative_path}"
    )

    print(
        "Wrote FRED repair execution artifact:"
    )
    print(
        f"  {execution_path}"
    )

    print(
        "repair_strategy="
        f"{REPAIR_STRATEGY}"
    )

    print(
        "n_candidate_repair_actions="
        f"{len(candidate_actions)}"
    )

    print(
        "n_actions_applied="
        f"{len(applied_actions)}"
    )

    print(
        "n_actions_abstained="
        f"{len(abstained_actions)}"
    )

    print(
        "repair_applied="
        f"{bool(applied_actions)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Relocate existing FRED claim "
            "citations to uniquely matched "
            "uncited bullets."
        )
    )

    parser.add_argument(
        "--narrative",
        type=Path,
        required=True,
        help=(
            "Input fred_narrative.md file."
        ),
    )

    parser.add_argument(
        "--audit",
        type=Path,
        required=True,
        help=(
            "Input fred_narrative_audit.json file."
        ),
    )

    parser.add_argument(
        "--repair-plan",
        type=Path,
        required=True,
        help=(
            "Input fred_repair_plan.json file."
        ),
    )

    parser.add_argument(
        "--selected-claims",
        type=Path,
        required=True,
        help=(
            "Input selected_fred_claims.json file."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory for repaired narrative "
            "and repair execution artifact."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    write_repair_execution(
        narrative_path=args.narrative,
        audit_path=args.audit,
        repair_plan_path=args.repair_plan,
        selected_claims_path=args.selected_claims,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
