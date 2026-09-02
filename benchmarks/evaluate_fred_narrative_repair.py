#!/usr/bin/env python3
"""
Evaluate one executed FRED narrative repair.

This evaluator compares:

    original audit
    + repair execution
    + repaired audit

and writes a durable before/after repair-result artifact.

The evaluator deliberately separates:

    targeted repair success

from:

    full audit success

A repair can correctly resolve its assigned failure while unrelated
audit failures remain.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPAIR_RESULT_SCHEMA_VERSION = (
    "fred_repair_result_v0_2"
)

REPAIR_EVALUATION_METHOD = (
    "deterministic_audit_before_after_with_unmasking"
)


def utc_now_iso() -> str:
    """Return current UTC time in ISO-8601 format."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load a JSON object from disk."""
    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return payload


def write_json(
    path: Path,
    payload: dict[str, Any],
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


def audit_errors(
    audit_payload: dict[str, Any],
) -> set[str]:
    """Return normalized top-level audit errors."""
    values = audit_payload.get(
        "errors",
        [],
    )

    if not isinstance(
        values,
        list,
    ):
        raise ValueError(
            "Audit errors must be a list."
        )

    if not all(
        isinstance(value, str)
        for value in values
    ):
        raise ValueError(
            "Audit errors must contain "
            "only strings."
        )

    return set(values)


def audit_snapshot(
    audit_payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract the audit measurements useful for repair comparison.
    """
    return {
        "audit_pass":
            audit_payload.get(
                "audit_pass"
            ),

        "errors":
            audit_payload.get(
                "errors",
                [],
            ),

        "n_bullets":
            audit_payload.get(
                "n_bullets"
            ),

        "n_citations":
            audit_payload.get(
                "n_citations"
            ),

        "n_unique_citations":
            audit_payload.get(
                "n_unique_citations"
            ),

        "n_bullets_missing_citations":
            audit_payload.get(
                "n_bullets_missing_citations"
            ),

        "n_bullets_with_content_mismatches":
            audit_payload.get(
                "n_bullets_with_content_mismatches"
            ),

        "content_issue_counts":
            audit_payload.get(
                "content_issue_counts",
                {},
            ),
    }


def classify_unmasked_errors(
    *,
    before_audit: dict[str, Any],
    repair_execution: dict[str, Any],
    after_audit: dict[str, Any],
    newly_observed_errors: set[str],
) -> tuple[
    set[str],
    list[dict[str, Any]],
]:
    """
    Conservatively classify newly observed audit errors
    that were exposed, rather than caused, by repair.

    Current supported proof:

    - citation-relocation strategy
    - bullet was previously blocked by a missing citation
    - that exact bullet was repaired
    - substantive bullet text is unchanged
    - exactly the relocated claim citation was appended
    - the repaired bullet becomes a content mismatch

    Anything not positively established remains an
    introduced error.
    """
    if (
        "claim_content_mismatches"
        not in newly_observed_errors
    ):
        return set(), []

    if (
        repair_execution.get(
            "repair_strategy"
        )
        != "relocate_existing_claim_citations"
    ):
        return set(), []

    before_bullets = (
        before_audit.get(
            "bullet_audits",
            [],
        )
    )

    after_bullets = (
        after_audit.get(
            "bullet_audits",
            [],
        )
    )

    if (
        not isinstance(
            before_bullets,
            list,
        )
        or not isinstance(
            after_bullets,
            list,
        )
        or len(before_bullets)
        != len(after_bullets)
    ):
        return set(), []

    applied_actions = (
        repair_execution.get(
            "applied_actions",
            [],
        )
    )

    if not isinstance(
        applied_actions,
        list,
    ):
        return set(), []

    actions_by_bullet: dict[
        int,
        dict[str, Any],
    ] = {}

    for action in applied_actions:
        if not isinstance(
            action,
            dict,
        ):
            continue

        bullet_index = action.get(
            "bullet_index"
        )

        if not isinstance(
            bullet_index,
            int,
        ):
            continue

        if bullet_index in (
            actions_by_bullet
        ):
            return set(), []

        actions_by_bullet[
            bullet_index
        ] = action

    mismatch_indexes = [
        index
        for index, item
        in enumerate(
            after_bullets,
            start=1,
        )
        if (
            isinstance(
                item,
                dict,
            )
            and item.get(
                "issue_type"
            )
            == "claim_content_mismatch"
        )
    ]

    if not mismatch_indexes:
        return set(), []

    details: list[
        dict[str, Any]
    ] = []

    for bullet_index in (
        mismatch_indexes
    ):
        before_item = (
            before_bullets[
                bullet_index - 1
            ]
        )

        after_item = (
            after_bullets[
                bullet_index - 1
            ]
        )

        action = actions_by_bullet.get(
            bullet_index
        )

        if (
            not isinstance(
                before_item,
                dict,
            )
            or not isinstance(
                after_item,
                dict,
            )
            or action is None
        ):
            return set(), []

        if (
            before_item.get(
                "issue_type"
            )
            != "missing_claim_citation"
        ):
            return set(), []

        if action.get(
            "status"
        ) != "applied":
            return set(), []

        if action.get(
            "executed_strategy"
        ) != (
            "relocate_existing_claim_citations"
        ):
            return set(), []

        claim_id = action.get(
            "claim_id"
        )

        if not isinstance(
            claim_id,
            str,
        ) or not claim_id:
            return set(), []

        before_text = (
            before_item.get(
                "bullet_text"
            )
        )

        after_text = (
            after_item.get(
                "bullet_text"
            )
        )

        destination_before = (
            action.get(
                "destination_before"
            )
        )

        destination_after = (
            action.get(
                "destination_after"
            )
        )

        if not all(
            isinstance(
                value,
                str,
            )
            for value in (
                before_text,
                after_text,
                destination_before,
                destination_after,
            )
        ):
            return set(), []

        if (
            before_text.strip()
            != destination_before.strip()
        ):
            return set(), []

        if (
            after_text.strip()
            != destination_after.strip()
        ):
            return set(), []

        expected_after = (
            destination_before.rstrip()
            + " "
            + f"[CLAIMS: {claim_id}]"
        )

        if (
            destination_after.strip()
            != expected_after.strip()
        ):
            return set(), []

        if before_item.get(
            "cited_claim_ids"
        ) not in (
            [],
            None,
        ):
            return set(), []

        if after_item.get(
            "cited_claim_ids"
        ) != [
            claim_id
        ]:
            return set(), []

        content_issues = (
            after_item.get(
                "content_issues",
                [],
            )
        )

        if (
            not isinstance(
                content_issues,
                list,
            )
            or not content_issues
        ):
            return set(), []

        details.append(
            {
                "error":
                    "claim_content_mismatches",
                "bullet_index":
                    bullet_index,
                "claim_id":
                    claim_id,
                "before_issue_type":
                    before_item.get(
                        "issue_type"
                    ),
                "after_issue_type":
                    after_item.get(
                        "issue_type"
                    ),
                "content_issues":
                    content_issues,
                "reason":
                    (
                        "citation_relocation_exposed_"
                        "content_audit_without_"
                        "substantive_text_change"
                    ),
            }
        )

    return (
        {
            "claim_content_mismatches"
        },
        details,
    )


def evaluate_repair(
    *,
    before_audit: dict[str, Any],
    repair_execution: dict[str, Any],
    after_audit: dict[str, Any],
    target_error: str,
    evaluated_at: str,
) -> dict[str, Any]:
    """
    Build one deterministic repair evaluation result.
    """
    before_errors = audit_errors(
        before_audit
    )

    after_errors = audit_errors(
        after_audit
    )

    resolved_errors = sorted(
        before_errors - after_errors
    )

    residual_errors = sorted(
        after_errors
    )

    newly_observed_errors = (
        after_errors - before_errors
    )

    (
        unmasked_errors,
        unmasked_error_details,
    ) = classify_unmasked_errors(
        before_audit=before_audit,
        repair_execution=repair_execution,
        after_audit=after_audit,
        newly_observed_errors=newly_observed_errors,
    )

    introduced_errors = sorted(
        newly_observed_errors
        - unmasked_errors
    )

    target_error_present_before = (
        target_error in before_errors
    )

    target_error_resolved = (
        target_error_present_before
        and target_error not in after_errors
    )

    repair_applied = (
        repair_execution.get(
            "repair_applied"
        )
        is True
    )

    targeted_repair_success = (
        repair_applied
        and target_error_resolved
        and not introduced_errors
    )

    audit_pass_before = (
        before_audit.get(
            "audit_pass"
        )
    )

    audit_pass_after = (
        after_audit.get(
            "audit_pass"
        )
    )

    if not isinstance(
        audit_pass_before,
        bool,
    ):
        raise ValueError(
            "Before audit missing boolean "
            "audit_pass."
        )

    if not isinstance(
        audit_pass_after,
        bool,
    ):
        raise ValueError(
            "After audit missing boolean "
            "audit_pass."
        )

    n_errors_before = len(
        before_errors
    )

    n_errors_after = len(
        after_errors
    )

    return {
        "repair_result_schema_version":
            REPAIR_RESULT_SCHEMA_VERSION,

        "repair_evaluation_method":
            REPAIR_EVALUATION_METHOD,

        "evaluated_at":
            evaluated_at,

        "repair_strategy":
            repair_execution.get(
                "repair_strategy"
            ),

        "target_error":
            target_error,

        "repair_applied":
            repair_applied,

        "n_actions_applied":
            repair_execution.get(
                "n_actions_applied"
            ),

        "target_error_present_before":
            target_error_present_before,

        "target_error_resolved":
            target_error_resolved,

        "targeted_repair_success":
            targeted_repair_success,

        "full_audit_success":
            audit_pass_after,

        "new_errors_introduced":
            bool(introduced_errors),

        "resolved_errors":
            resolved_errors,

        "residual_errors":
            residual_errors,

        "newly_observed_errors":
            sorted(
                newly_observed_errors
            ),

        "unmasked_errors":
            sorted(
                unmasked_errors
            ),

        "unmasked_error_details":
            unmasked_error_details,

        "introduced_errors":
            introduced_errors,

        "audit_pass_before":
            audit_pass_before,

        "audit_pass_after":
            audit_pass_after,

        "audit_pass_changed":
            (
                audit_pass_before
                != audit_pass_after
            ),

        "n_top_level_errors_before":
            n_errors_before,

        "n_top_level_errors_after":
            n_errors_after,

        "top_level_error_count_delta":
            (
                n_errors_after
                - n_errors_before
            ),

        "top_level_error_count_improved":
            (
                n_errors_after
                < n_errors_before
            ),

        "audit_before":
            audit_snapshot(
                before_audit
            ),

        "audit_after":
            audit_snapshot(
                after_audit
            ),

        "inputs": {
            "before_audit_file":
                before_audit.get(
                    "output_files",
                    {},
                ).get(
                    "audit_json"
                ),

            "repair_execution_file":
                repair_execution.get(
                    "outputs",
                    {},
                ).get(
                    "repair_execution_json"
                ),

            "after_audit_file":
                after_audit.get(
                    "output_files",
                    {},
                ).get(
                    "audit_json"
                ),
        },
    }


def write_repair_result(
    *,
    before_audit_path: Path,
    repair_execution_path: Path,
    after_audit_path: Path,
    target_error: str,
    output_dir: Path,
) -> None:
    """Evaluate repair and write result artifact."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    before_audit = load_json(
        before_audit_path
    )

    repair_execution = load_json(
        repair_execution_path
    )

    after_audit = load_json(
        after_audit_path
    )

    result = evaluate_repair(
        before_audit=before_audit,
        repair_execution=repair_execution,
        after_audit=after_audit,
        target_error=target_error,
        evaluated_at=utc_now_iso(),
    )

    result_path = (
        output_dir
        / "fred_repair_result.json"
    )

    result["inputs"] = {
        "before_audit_file":
            str(before_audit_path),

        "repair_execution_file":
            str(repair_execution_path),

        "after_audit_file":
            str(after_audit_path),
    }

    result["outputs"] = {
        "repair_result_json":
            str(result_path),
    }

    write_json(
        result_path,
        result,
    )

    print(
        "Wrote FRED repair result artifact:"
    )
    print(
        f"  {result_path}"
    )

    print(
        "repair_strategy="
        f"{result['repair_strategy']}"
    )

    print(
        "target_error="
        f"{result['target_error']}"
    )

    print(
        "repair_applied="
        f"{result['repair_applied']}"
    )

    print(
        "target_error_resolved="
        f"{result['target_error_resolved']}"
    )

    print(
        "targeted_repair_success="
        f"{result['targeted_repair_success']}"
    )

    print(
        "full_audit_success="
        f"{result['full_audit_success']}"
    )

    print(
        "new_errors_introduced="
        f"{result['new_errors_introduced']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a deterministic FRED "
            "narrative repair."
        )
    )

    parser.add_argument(
        "--before-audit",
        type=Path,
        required=True,
        help=(
            "Original fred_narrative_audit.json."
        ),
    )

    parser.add_argument(
        "--repair-execution",
        type=Path,
        required=True,
        help=(
            "fred_repair_execution.json."
        ),
    )

    parser.add_argument(
        "--after-audit",
        type=Path,
        required=True,
        help=(
            "Repaired fred_narrative_audit.json."
        ),
    )

    parser.add_argument(
        "--target-error",
        required=True,
        help=(
            "Top-level audit error targeted "
            "by the repair."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory for fred_repair_result.json."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    write_repair_result(
        before_audit_path=args.before_audit,
        repair_execution_path=args.repair_execution,
        after_audit_path=args.after_audit,
        target_error=args.target_error,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
