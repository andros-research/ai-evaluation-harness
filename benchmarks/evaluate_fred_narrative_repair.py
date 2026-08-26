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
    "fred_repair_result_v0_1"
)

REPAIR_EVALUATION_METHOD = (
    "deterministic_audit_before_after"
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

    introduced_errors = sorted(
        after_errors - before_errors
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
