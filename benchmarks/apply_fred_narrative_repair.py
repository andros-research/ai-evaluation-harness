#!/usr/bin/env python3
"""
Apply deterministic repairs to a FRED narrative.

v1.8.3 begins with one intentionally narrow executable strategy:

    normalize_uncited_detail_bullets

For planned bullet-level repairs with:

    issue_type = missing_claim_citation
    repair_action = add_valid_claim_citation_or_remove_bullet

the executor removes Markdown bullet shape while preserving the
underlying text.

This converts structural detail lines from audited narrative bullets
into plain detail lines without inventing citations, deleting content,
or performing semantic rewriting.

The repaired narrative and a durable repair-execution artifact are
written separately from the original run artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPAIR_EXECUTION_SCHEMA_VERSION = (
    "fred_repair_execution_v0_1"
)

REPAIR_METHOD = (
    "deterministic_structural_repair"
)

REPAIR_STRATEGY = (
    "normalize_uncited_detail_bullets"
)

TARGET_ISSUE_TYPE = (
    "missing_claim_citation"
)

TARGET_REPAIR_ACTION = (
    "add_valid_claim_citation_or_remove_bullet"
)

DETAIL_BULLET_PREFIXES = (
    "Direction:",
    "Prior value:",
    "Current value:",
    "Delta magnitude:",
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


def markdown_bullet_line_indexes(
    lines: list[str],
) -> list[int]:
    """Return line indexes for Markdown '- ' bullets."""
    return [
        idx
        for idx, line in enumerate(lines)
        if line.lstrip().startswith("- ")
    ]


def is_supported_detail_bullet(
    bullet_text: object,
) -> bool:
    """
    Return whether a bullet matches the narrow detail-line
    structure supported by this repair strategy.
    """
    if not isinstance(
        bullet_text,
        str,
    ):
        return False

    stripped = bullet_text.lstrip()

    if not stripped.startswith("- "):
        return False

    content = stripped[2:].strip()

    return content.startswith(
        DETAIL_BULLET_PREFIXES
    )


def candidate_repair_actions(
    repair_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Select only planned missing-citation detail bullets
    supported by this executor.
    """
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
            and is_supported_detail_bullet(
                action.get("bullet_text")
            )
        )
    ]


def normalize_bullet_line(
    line: str,
) -> str:
    """
    Remove one Markdown '- ' bullet marker while preserving text.
    """
    stripped = line.lstrip()

    if not stripped.startswith("- "):
        raise ValueError(
            "Expected Markdown bullet line."
        )

    indent_length = (
        len(line)
        - len(stripped)
    )

    return (
        line[:indent_length]
        + stripped[2:]
    )


def apply_repairs(
    *,
    narrative_text: str,
    audit_payload: dict[str, Any],
    repair_plan: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """
    Apply supported repair-plan actions to the narrative.

    Strong validation is intentional:
    - audit bullet count must match narrative bullet count
    - repair-plan bullet indexes must be valid
    - corresponding audit record must show the target issue
    - repair-plan text must match the audited/narrative bullet

    Any mismatch fails rather than guessing.
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

    if len(bullet_line_indexes) != len(
        bullet_audits
    ):
        raise ValueError(
            "Narrative bullet count does not "
            "match audit bullet count: "
            f"{len(bullet_line_indexes)} vs "
            f"{len(bullet_audits)}"
        )

    actions = candidate_repair_actions(
        repair_plan
    )

    applied_actions: list[
        dict[str, Any]
    ] = []

    seen_bullet_indexes: set[int] = set()

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

        repaired_line = (
            normalize_bullet_line(
                original_line
            )
        )

        lines[line_index] = (
            repaired_line
        )

        applied_actions.append(
            {
                "action_id":
                    action.get(
                        "action_id"
                    ),
                "bullet_index":
                    bullet_index,
                "line_number":
                    line_index + 1,
                "issue_type":
                    TARGET_ISSUE_TYPE,
                "planned_repair_action":
                    TARGET_REPAIR_ACTION,
                "executed_strategy":
                    REPAIR_STRATEGY,
                "before":
                    original_line,
                "after":
                    repaired_line,
                "status":
                    "applied",
            }
        )

    repaired_text = "\n".join(
        lines
    )

    if narrative_text.endswith("\n"):
        repaired_text += "\n"

    return (
        repaired_text,
        applied_actions,
    )


def write_repair_execution(
    *,
    narrative_path: Path,
    audit_path: Path,
    repair_plan_path: Path,
    output_dir: Path,
) -> None:
    """
    Apply supported repairs and write durable output artifacts.
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

    repaired_text, applied_actions = (
        apply_repairs(
            narrative_text=narrative_text,
            audit_payload=audit_payload,
            repair_plan=repair_plan,
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
            bool(applied_actions),

        "n_planned_repair_actions":
            len(planned_actions),

        "n_candidate_repair_actions":
            len(candidate_actions),

        "n_actions_applied":
            len(applied_actions),

        "applied_actions":
            applied_actions,

        "inputs": {
            "narrative_file":
                str(narrative_path),

            "audit_file":
                str(audit_path),

            "repair_plan_file":
                str(repair_plan_path),

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
        "repair_applied="
        f"{bool(applied_actions)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply deterministic repairs to "
            "a FRED narrative."
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
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
