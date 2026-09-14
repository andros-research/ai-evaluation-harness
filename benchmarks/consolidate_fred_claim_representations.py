#!/usr/bin/env python3
"""
Consolidate duplicate FRED claim representations.

This deterministic repair strategy targets planned bullet-level repairs with:

    issue_type = missing_claim_citation
    repair_action = add_valid_claim_citation_or_remove_bullet

It handles two narrowly supported cases.

Subtype A:
    A cited audited bullet is deficient only because prior/current values
    are missing, while an uncited bullet uniquely matches the same claim
    and would pass the full claim-content audit once cited.

    Action:
        remove the deficient cited representation,
        move its existing citation to the complete representation.

Subtype B:
    A cited audited bullet already passes, while an uncited bullet contains
    only a redundant Prior value / Current value / Delta magnitude detail
    representation of the same claim.

    Action:
        keep the valid cited representation,
        remove the redundant uncited detail representation.

The executor does not invent citations, duplicate citations, rewrite
substantive prose, or guess when claim identity/source pairing is ambiguous.

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
    audit_claim_content_against_bullet,
    extract_cited_claim_ids,
)

from relocate_fred_claim_citations import (
    candidate_repair_actions,
    claim_identity_matches,
    markdown_bullet_line_indexes,
    selected_claim_records,
)


REPAIR_EXECUTION_SCHEMA_VERSION = (
    "fred_repair_execution_v0_1"
)

REPAIR_METHOD = (
    "deterministic_claim_representation_consolidation"
)

REPAIR_STRATEGY = (
    "consolidate_duplicate_claim_representations"
)

TARGET_ISSUE_TYPE = (
    "missing_claim_citation"
)

TARGET_REPAIR_ACTION = (
    "add_valid_claim_citation_or_remove_bullet"
)

DEFICIENT_SOURCE_CONTENT_ISSUES = {
    "missing_current_value",
    "missing_prior_value",
}

CITATION_BLOCK_PATTERN = re.compile(
    r"\[CLAIMS:\s*([^\]]+)\]"
)

REDUNDANT_DETAIL_PATTERN = re.compile(
    r"""
    ^[-*]\s+
    Prior\s+value:\s*[^,]+,\s*
    Current\s+value:\s*[^,]+,\s*
    Delta\s+magnitude:\s*.+$
    """,
    re.IGNORECASE | re.VERBOSE,
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


def unique_citation_block(
    line: str,
    claim_id: str,
) -> str | None:
    """
    Return the unique single-claim citation block for claim_id,
    otherwise None.
    """
    matches = list(
        CITATION_BLOCK_PATTERN.finditer(
            line
        )
    )

    valid_blocks: list[str] = []

    for match in matches:
        ids = [
            item.strip()
            for item in match.group(1).split(",")
            if item.strip()
        ]

        if ids == [claim_id]:
            valid_blocks.append(
                match.group(0)
            )

    if len(valid_blocks) != 1:
        return None

    return valid_blocks[0]


def destination_content_passes(
    *,
    bullet_text: str,
    claim: dict[str, Any],
) -> bool:
    """
    Test whether the uncited destination would satisfy
    the historical content auditor for the matched claim.
    """
    result = (
        audit_claim_content_against_bullet(
            bullet_text=bullet_text,
            claim=claim,
        )
    )

    return bool(
        result.get(
            "content_audit_pass"
        )
    )


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
    Consolidate narrowly supported duplicate claim representations.
    """
    lines = narrative_text.splitlines()

    bullet_line_indexes = (
        markdown_bullet_line_indexes(
            lines
        )
    )

    bullet_audits = audit_payload.get(
        "bullet_audits",
        [],
    )

    if len(
        bullet_line_indexes
    ) != len(
        bullet_audits
    ):
        raise ValueError(
            "Narrative bullet count does not match "
            "audit bullet count: "
            f"{len(bullet_line_indexes)} vs "
            f"{len(bullet_audits)}"
        )

    actions = candidate_repair_actions(
        repair_plan
    )

    prepared: list[
        dict[str, Any]
    ] = []

    abstained: list[
        dict[str, Any]
    ] = []

    seen_destinations: set[int] = set()

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
                "bullet_index."
            )

        if bullet_index in seen_destinations:
            raise ValueError(
                "Duplicate repair action for "
                f"bullet_index={bullet_index}"
            )

        seen_destinations.add(
            bullet_index
        )

        if (
            bullet_index < 1
            or bullet_index > len(
                bullet_audits
            )
        ):
            raise ValueError(
                "Repair bullet_index outside "
                "audit range."
            )

        destination_audit = (
            bullet_audits[
                bullet_index - 1
            ]
        )

        if (
            destination_audit.get(
                "issue_type"
            )
            != TARGET_ISSUE_TYPE
        ):
            raise ValueError(
                "Repair plan and audit disagree "
                "about destination issue type."
            )

        destination_line_index = (
            bullet_line_indexes[
                bullet_index - 1
            ]
        )

        destination_text = (
            lines[
                destination_line_index
            ]
        )

        audited_text = (
            destination_audit.get(
                "bullet_text"
            )
        )

        if (
            isinstance(
                audited_text,
                str,
            )
            and destination_text.strip()
            != audited_text.strip()
        ):
            raise ValueError(
                "Narrative and audit destination "
                "text do not match."
            )

        if extract_cited_claim_ids(
            destination_text
        ):
            abstained.append(
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

        matching_claims = [
            claim
            for claim in selected_claims
            if claim_identity_matches(
                bullet_text=destination_text,
                claim=claim,
            )
        ]

        if len(
            matching_claims
        ) != 1:
            abstained.append(
                {
                    "action_id":
                        action.get(
                            "action_id"
                        ),
                    "bullet_index":
                        bullet_index,
                    "reason":
                        "no_unique_claim_identity",
                    "matching_claim_ids":
                        [
                            claim.get(
                                "claim_id"
                            )
                            for claim
                            in matching_claims
                        ],
                }
            )
            continue

        claim = matching_claims[0]
        claim_id = claim.get(
            "claim_id"
        )

        if not isinstance(
            claim_id,
            str,
        ):
            raise ValueError(
                "Matched claim missing claim_id."
            )

        source_candidates = []

        for source_index, source_audit in enumerate(
            bullet_audits,
            start=1,
        ):
            if (
                source_audit.get(
                    "cited_claim_ids"
                )
                == [claim_id]
            ):
                source_candidates.append(
                    (
                        source_index,
                        source_audit,
                    )
                )

        if len(
            source_candidates
        ) != 1:
            abstained.append(
                {
                    "action_id":
                        action.get(
                            "action_id"
                        ),
                    "bullet_index":
                        bullet_index,
                    "reason":
                        "cited_source_not_unique",
                    "claim_id":
                        claim_id,
                    "n_sources":
                        len(
                            source_candidates
                        ),
                }
            )
            continue

        (
            source_bullet_index,
            source_audit,
        ) = source_candidates[0]

        source_line_index = (
            bullet_line_indexes[
                source_bullet_index - 1
            ]
        )

        source_text = lines[
            source_line_index
        ]

        citation_block = (
            unique_citation_block(
                source_text,
                claim_id,
            )
        )

        if citation_block is None:
            abstained.append(
                {
                    "action_id":
                        action.get(
                            "action_id"
                        ),
                    "bullet_index":
                        bullet_index,
                    "reason":
                        "source_citation_block_not_unique",
                    "claim_id":
                        claim_id,
                }
            )
            continue

        source_issue = (
            source_audit.get(
                "issue_type"
            )
        )

        source_content_issues = set(
            source_audit.get(
                "content_issues",
                [],
            )
        )

        subtype: str | None = None

        if (
            source_issue
            == "claim_content_mismatch"
            and source_content_issues
            == DEFICIENT_SOURCE_CONTENT_ISSUES
            and destination_content_passes(
                bullet_text=destination_text,
                claim=claim,
            )
        ):
            subtype = (
                "replace_deficient_cited_source"
            )

        elif (
            source_issue is None
            and REDUNDANT_DETAIL_PATTERN.match(
                destination_text.strip()
            )
        ):
            subtype = (
                "remove_redundant_uncited_detail"
            )

        if subtype is None:
            abstained.append(
                {
                    "action_id":
                        action.get(
                            "action_id"
                        ),
                    "bullet_index":
                        bullet_index,
                    "reason":
                        "unsupported_consolidation_state",
                    "claim_id":
                        claim_id,
                    "source_issue_type":
                        source_issue,
                    "source_content_issues":
                        sorted(
                            source_content_issues
                        ),
                    "destination_content_passes":
                        destination_content_passes(
                            bullet_text=
                                destination_text,
                            claim=claim,
                        ),
                    "destination_matches_"
                    "redundant_detail_grammar":
                        bool(
                            REDUNDANT_DETAIL_PATTERN.match(
                                destination_text.strip()
                            )
                        ),
                }
            )
            continue

        prepared.append(
            {
                "action":
                    action,
                "subtype":
                    subtype,
                "claim_id":
                    claim_id,
                "source_bullet_index":
                    source_bullet_index,
                "source_line_index":
                    source_line_index,
                "destination_bullet_index":
                    bullet_index,
                "destination_line_index":
                    destination_line_index,
                "citation_block":
                    citation_block,
            }
        )

    claim_counts = Counter(
        item["claim_id"]
        for item in prepared
    )

    duplicate_claim_ids = {
        claim_id
        for claim_id, count
        in claim_counts.items()
        if count > 1
    }

    filtered: list[
        dict[str, Any]
    ] = []

    for item in prepared:
        if item[
            "claim_id"
        ] in duplicate_claim_ids:
            abstained.append(
                {
                    "action_id":
                        item["action"].get(
                            "action_id"
                        ),
                    "bullet_index":
                        item[
                            "destination_bullet_index"
                        ],
                    "reason":
                        "claim_matches_multiple_"
                        "destination_bullets",
                    "claim_id":
                        item["claim_id"],
                }
            )
            continue

        filtered.append(
            item
        )

    prepared = filtered

    applied: list[
        dict[str, Any]
    ] = []

    # Work from original line indexes, marking removals
    # as None so indexes do not shift mid-execution.
    mutable_lines: list[
        str | None
    ] = list(
        lines
    )

    for item in prepared:
        action = item[
            "action"
        ]

        subtype = item[
            "subtype"
        ]

        claim_id = item[
            "claim_id"
        ]

        source_line_index = item[
            "source_line_index"
        ]

        destination_line_index = item[
            "destination_line_index"
        ]

        citation_block = item[
            "citation_block"
        ]

        source_before = (
            mutable_lines[
                source_line_index
            ]
        )

        destination_before = (
            mutable_lines[
                destination_line_index
            ]
        )

        if (
            not isinstance(
                source_before,
                str,
            )
            or not isinstance(
                destination_before,
                str,
            )
        ):
            raise ValueError(
                "Prepared consolidation refers "
                "to already-removed line."
            )

        if subtype == (
            "replace_deficient_cited_source"
        ):
            destination_after = (
                destination_before.rstrip()
                + " "
                + citation_block
            )

            source_after = None

            mutable_lines[
                source_line_index
            ] = None

            mutable_lines[
                destination_line_index
            ] = destination_after

        elif subtype == (
            "remove_redundant_uncited_detail"
        ):
            source_after = (
                source_before
            )

            destination_after = None

            mutable_lines[
                destination_line_index
            ] = None

        else:
            raise ValueError(
                f"Unknown subtype: {subtype}"
            )

        applied.append(
            {
                "action_id":
                    action.get(
                        "action_id"
                    ),
                "bullet_index":
                    item[
                        "destination_bullet_index"
                    ],
                "issue_type":
                    TARGET_ISSUE_TYPE,
                "planned_repair_action":
                    TARGET_REPAIR_ACTION,
                "executed_strategy":
                    REPAIR_STRATEGY,
                "consolidation_subtype":
                    subtype,
                "claim_id":
                    claim_id,
                "source_bullet_index":
                    item[
                        "source_bullet_index"
                    ],
                "destination_bullet_index":
                    item[
                        "destination_bullet_index"
                    ],
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

    repaired_lines = [
        line
        for line in mutable_lines
        if line is not None
    ]

    repaired_text = "\n".join(
        repaired_lines
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

    if citations_before != citations_after:
        raise ValueError(
            "Citation conservation invariant "
            "failed: citation multiset changed."
        )

    return (
        repaired_text,
        applied,
        abstained,
    )


def write_repair_execution(
    *,
    narrative_path: Path,
    audit_path: Path,
    repair_plan_path: Path,
    selected_claims_path: Path,
    output_dir: Path,
) -> None:
    """Execute consolidation and write durable artifacts."""
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

    selected_claims = (
        selected_claim_records(
            load_json(
                selected_claims_path
            )
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

    (
        repaired_text,
        applied_actions,
        abstained_actions,
    ) = apply_repairs(
        narrative_text=narrative_text,
        audit_payload=audit_payload,
        repair_plan=repair_plan,
        selected_claims=selected_claims,
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
        "Wrote consolidated FRED narrative:"
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
            "Consolidate duplicate FRED "
            "claim representations."
        )
    )

    parser.add_argument(
        "--narrative",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--audit",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--repair-plan",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--selected-claims",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
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