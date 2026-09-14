#!/usr/bin/env python3
"""
Aggregate a completed FRED ACT population into analysis-ready artifacts.

Inputs:
    completed repair-population output root

Outputs:
    one row per original observation
    overall population summary
    repair-strategy summary
    model/prompt/temperature cell summary

This script adds no repair logic and does not reinterpret executor
eligibility. It summarizes durable population and workflow artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AGGREGATE_SCHEMA_VERSION = (
    "fred_repair_population_aggregate_v0_1"
)

AGGREGATE_METHOD = (
    "population_run_and_workflow_artifact_reduction"
)


ROW_FIELDS = (
    "batch_name",
    "cell_name",
    "repetition",
    "source_run_dir",

    "narrative_model",
    "narrative_prompt_variant",
    "narrative_temperature",

    "readiness_status",
    "population_status",
    "workflow_outcome",

    "audit_pass_before",
    "repair_needed",
    "audit_errors_before",

    "selection_status",
    "selection_reason",
    "strategy_selected",
    "eligible_strategies",
    "ambiguous",

    "repair_applied",
    "n_actions_applied",
    "target_error",
    "target_error_resolved",
    "targeted_repair_success",

    "audit_pass_after",
    "full_audit_success",
    "new_errors_introduced",

    "resolved_errors",
    "residual_errors",
    "newly_observed_errors",
    "unmasked_errors",
    "introduced_errors",

    "strict_selected_claim_coverage",

    "upstream_validation_error",
    "error_type",
    "error_message",
)


CELL_FIELDS = (
    "cell_name",
    "narrative_model",
    "narrative_prompt_variant",
    "narrative_temperature",

    "n_runs",
    "n_upstream_validation_failed",
    "n_workflow_ready",

    "n_audit_pass_before",
    "n_audit_fail_before",

    "n_no_repair_needed",
    "n_no_supported_deterministic_repair",
    "n_ambiguous_repair_selection",
    "n_repair_executed",

    "n_normalize",
    "n_relocate",
    "n_consolidate",

    "n_targeted_repair_success",
    "n_full_audit_success",
    "n_new_errors_introduced",

    "n_final_audit_accepted",

    "repair_selection_rate_among_audit_failures",
    "targeted_success_rate_among_repairs",
    "full_success_rate_among_repairs",
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
) -> Any:
    """Load JSON from disk."""
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_json(
    path: Path,
    payload: Any,
) -> None:
    """Write formatted JSON."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def safe_rate(
    numerator: int,
    denominator: int,
) -> float | None:
    """Return a ratio or None for a zero denominator."""
    if denominator == 0:
        return None

    return (
        numerator
        / denominator
    )


def json_cell(
    value: Any,
) -> Any:
    """
    Convert structured values to JSON strings for CSV output.
    """
    if isinstance(
        value,
        (list, dict),
    ):
        return json.dumps(
            value,
            separators=(
                ",",
                ":",
            ),
            sort_keys=True,
        )

    return value


def load_run_metadata(
    source_run_dir: Path,
) -> dict[str, Any]:
    """
    Load durable evidence-loop metadata used to identify the experiment cell.
    """
    path = (
        source_run_dir
        / "fred_runs"
        / "latest_fred_evidence_loop_run.json"
    )

    if not path.exists():
        return {}

    payload = load_json(
        path
    )

    if not isinstance(
        payload,
        dict,
    ):
        return {}

    return payload


def load_before_artifacts(
    source_run_dir: Path,
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Load original audit and repair-plan artifacts when present."""
    audit_path = (
        source_run_dir
        / "fred_audits"
        / "fred_narrative_audit.json"
    )

    repair_path = (
        source_run_dir
        / "fred_repairs"
        / "fred_repair_plan.json"
    )

    audit = None
    repair = None

    if audit_path.exists():
        payload = load_json(
            audit_path
        )

        if isinstance(
            payload,
            dict,
        ):
            audit = payload

    if repair_path.exists():
        payload = load_json(
            repair_path
        )

        if isinstance(
            payload,
            dict,
        ):
            repair = payload

    return (
        audit,
        repair,
    )


def load_workflow(
    population_record: dict[str, Any],
) -> dict[str, Any] | None:
    """Load the single-run ACT workflow artifact when one exists."""
    value = population_record.get(
        "workflow_json"
    )

    if not isinstance(
        value,
        str,
    ):
        return None

    path = Path(
        value
    )

    if not path.exists():
        raise FileNotFoundError(
            "Population record references missing "
            f"workflow artifact: {path}"
        )

    payload = load_json(
        path
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "Workflow artifact must be an object: "
            f"{path}"
        )

    return payload


def build_row(
    population_record: dict[str, Any],
) -> dict[str, Any]:
    """Build one analysis-ready row from durable artifacts."""
    source = population_record.get(
        "source",
        {}
    )

    if not isinstance(
        source,
        dict,
    ):
        raise ValueError(
            "Population record source must be an object."
        )

    source_run_value = source.get(
        "run_dir"
    )

    if not isinstance(
        source_run_value,
        str,
    ):
        raise ValueError(
            "Population record missing source run_dir."
        )

    source_run_dir = Path(
        source_run_value
    )

    metadata = load_run_metadata(
        source_run_dir
    )

    (
        before_audit,
        repair_plan,
    ) = load_before_artifacts(
        source_run_dir
    )

    workflow = load_workflow(
        population_record
    )

    audit_pass_before = (
        before_audit.get(
            "audit_pass"
        )
        if before_audit
        is not None
        else None
    )

    audit_errors_before = (
        before_audit.get(
            "errors",
            [],
        )
        if before_audit
        is not None
        else []
    )

    repair_needed = (
        repair_plan.get(
            "repair_needed"
        )
        if repair_plan
        is not None
        else None
    )

    row = {
        "batch_name":
            source.get(
                "batch_name"
            ),

        "cell_name":
            source.get(
                "cell_name"
            ),

        "repetition":
            source.get(
                "repetition"
            ),

        "source_run_dir":
            source_run_value,

        "narrative_model":
            metadata.get(
                "narrative_model"
            ),

        "narrative_prompt_variant":
            metadata.get(
                "narrative_prompt_variant"
            ),

        "narrative_temperature":
            metadata.get(
                "narrative_temperature"
            ),

        "readiness_status":
            population_record.get(
                "readiness_status"
            ),

        "population_status":
            population_record.get(
                "population_status"
            ),

        "workflow_outcome":
            population_record.get(
                "workflow_outcome"
            ),

        "audit_pass_before":
            audit_pass_before,

        "repair_needed":
            repair_needed,

        "audit_errors_before":
            audit_errors_before,

        "selection_status":
            (
                workflow.get(
                    "selection_status"
                )
                if workflow
                is not None
                else None
            ),

        "selection_reason":
            (
                workflow.get(
                    "selection_reason"
                )
                if workflow
                is not None
                else None
            ),

        "strategy_selected":
            population_record.get(
                "strategy_selected"
            ),

        "eligible_strategies":
            (
                workflow.get(
                    "eligible_strategies",
                    [],
                )
                if workflow
                is not None
                else []
            ),

        "ambiguous":
            (
                workflow.get(
                    "ambiguous"
                )
                if workflow
                is not None
                else None
            ),

        "repair_applied":
            population_record.get(
                "repair_applied"
            ),

        "n_actions_applied":
            (
                workflow.get(
                    "n_actions_applied"
                )
                if workflow
                is not None
                else None
            ),

        "target_error":
            (
                workflow.get(
                    "target_error"
                )
                if workflow
                is not None
                else None
            ),

        "target_error_resolved":
            (
                workflow.get(
                    "target_error_resolved"
                )
                if workflow
                is not None
                else None
            ),

        "targeted_repair_success":
            population_record.get(
                "targeted_repair_success"
            ),

        "audit_pass_after":
            population_record.get(
                "audit_pass_after"
            ),

        "full_audit_success":
            population_record.get(
                "full_audit_success"
            ),

        "new_errors_introduced":
            population_record.get(
                "new_errors_introduced"
            ),

        "resolved_errors":
            (
                workflow.get(
                    "resolved_errors",
                    [],
                )
                if workflow
                is not None
                else []
            ),

        "residual_errors":
            (
                workflow.get(
                    "residual_errors",
                    [],
                )
                if workflow
                is not None
                else []
            ),

        "newly_observed_errors":
            (
                workflow.get(
                    "newly_observed_errors",
                    [],
                )
                if workflow
                is not None
                else []
            ),

        "unmasked_errors":
            (
                workflow.get(
                    "unmasked_errors",
                    [],
                )
                if workflow
                is not None
                else []
            ),

        "introduced_errors":
            (
                workflow.get(
                    "introduced_errors",
                    [],
                )
                if workflow
                is not None
                else []
            ),

        "strict_selected_claim_coverage":
            (
                workflow.get(
                    "strict_selected_claim_coverage"
                )
                if workflow
                is not None
                else (
                    before_audit.get(
                        "strict_selected_claim_coverage"
                    )
                    if before_audit
                    is not None
                    else None
                )
            ),

        "upstream_validation_error":
            population_record.get(
                "validation_error"
            ),

        "error_type":
            population_record.get(
                "error_type"
            ),

        "error_message":
            population_record.get(
                "error_message"
            ),
    }

    return row


def summarize_rows(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build overall aggregate counts and rates."""
    n_runs = len(
        rows
    )

    n_upstream = sum(
        row[
            "readiness_status"
        ]
        == "upstream_validation_failed"
        for row in rows
    )

    n_workflow_ready = sum(
        row[
            "readiness_status"
        ]
        == "workflow_ready"
        for row in rows
    )

    n_audit_pass_before = sum(
        row[
            "audit_pass_before"
        ]
        is True
        for row in rows
    )

    n_audit_fail_before = sum(
        row[
            "audit_pass_before"
        ]
        is False
        for row in rows
    )

    n_repair_executed = sum(
        row[
            "workflow_outcome"
        ]
        == "repair_executed"
        for row in rows
    )

    n_unsupported = sum(
        row[
            "workflow_outcome"
        ]
        == "no_supported_deterministic_repair"
        for row in rows
    )

    n_ambiguous = sum(
        row[
            "workflow_outcome"
        ]
        == "ambiguous_repair_selection"
        for row in rows
    )

    n_targeted_success = sum(
        row[
            "targeted_repair_success"
        ]
        is True
        for row in rows
    )

    n_full_success = sum(
        row[
            "full_audit_success"
        ]
        is True
        for row in rows
    )

    n_new_errors = sum(
        row[
            "new_errors_introduced"
        ]
        is True
        for row in rows
    )

    n_final_audit_accepted = (
        n_audit_pass_before
        + n_full_success
    )

    return {
        "n_runs":
            n_runs,

        "readiness_counts":
            dict(
                sorted(
                    Counter(
                        row[
                            "readiness_status"
                        ]
                        for row in rows
                    ).items()
                )
            ),

        "population_status_counts":
            dict(
                sorted(
                    Counter(
                        row[
                            "population_status"
                        ]
                        for row in rows
                    ).items()
                )
            ),

        "workflow_outcome_counts":
            dict(
                sorted(
                    Counter(
                        row[
                            "workflow_outcome"
                        ]
                        for row in rows
                    ).items()
                )
            ),

        "strategy_selected_counts":
            dict(
                sorted(
                    Counter(
                        row[
                            "strategy_selected"
                        ]
                        for row in rows
                        if row[
                            "strategy_selected"
                        ]
                        is not None
                    ).items()
                )
            ),

        "n_upstream_validation_failed":
            n_upstream,

        "n_workflow_ready":
            n_workflow_ready,

        "n_audit_pass_before":
            n_audit_pass_before,

        "n_audit_fail_before":
            n_audit_fail_before,

        "n_repair_executed":
            n_repair_executed,

        "n_no_supported_deterministic_repair":
            n_unsupported,

        "n_ambiguous_repair_selection":
            n_ambiguous,

        "n_targeted_repair_success":
            n_targeted_success,

        "n_full_audit_success":
            n_full_success,

        "n_new_errors_introduced":
            n_new_errors,

        "n_final_audit_accepted":
            n_final_audit_accepted,

        "repair_selection_rate_among_audit_failures":
            safe_rate(
                n_repair_executed,
                n_audit_fail_before,
            ),

        "targeted_success_rate_among_repairs":
            safe_rate(
                n_targeted_success,
                n_repair_executed,
            ),

        "full_success_rate_among_repairs":
            safe_rate(
                n_full_success,
                n_repair_executed,
            ),

        "final_acceptance_rate_among_workflow_ready":
            safe_rate(
                n_final_audit_accepted,
                n_workflow_ready,
            ),

        "final_acceptance_rate_among_all_attempts":
            safe_rate(
                n_final_audit_accepted,
                n_runs,
            ),
    }


def summarize_strategies(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one summary record per selected repair strategy."""
    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(
        list
    )

    for row in rows:
        strategy = row[
            "strategy_selected"
        ]

        if isinstance(
            strategy,
            str,
        ):
            grouped[
                strategy
            ].append(
                row
            )

    result = []

    for strategy in sorted(
        grouped
    ):
        items = grouped[
            strategy
        ]

        n_selected = len(
            items
        )

        n_targeted_success = sum(
            row[
                "targeted_repair_success"
            ]
            is True
            for row in items
        )

        n_full_success = sum(
            row[
                "full_audit_success"
            ]
            is True
            for row in items
        )

        n_new_errors = sum(
            row[
                "new_errors_introduced"
            ]
            is True
            for row in items
        )

        result.append(
            {
                "strategy":
                    strategy,

                "n_selected":
                    n_selected,

                "n_targeted_repair_success":
                    n_targeted_success,

                "n_full_audit_success":
                    n_full_success,

                "n_new_errors_introduced":
                    n_new_errors,

                "targeted_success_rate":
                    safe_rate(
                        n_targeted_success,
                        n_selected,
                    ),

                "full_success_rate":
                    safe_rate(
                        n_full_success,
                        n_selected,
                    ),
            }
        )

    return result


def summarize_cells(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one summary record per experiment cell."""
    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(
        list
    )

    for row in rows:
        cell_name = row[
            "cell_name"
        ]

        if not isinstance(
            cell_name,
            str,
        ):
            raise ValueError(
                "Aggregate row missing cell_name."
            )

        grouped[
            cell_name
        ].append(
            row
        )

    result = []

    for cell_name in sorted(
        grouped
    ):
        items = grouped[
            cell_name
        ]

        first = items[0]

        n_runs = len(
            items
        )

        n_upstream = sum(
            row[
                "readiness_status"
            ]
            == "upstream_validation_failed"
            for row in items
        )

        n_workflow_ready = sum(
            row[
                "readiness_status"
            ]
            == "workflow_ready"
            for row in items
        )

        n_audit_pass = sum(
            row[
                "audit_pass_before"
            ]
            is True
            for row in items
        )

        n_audit_fail = sum(
            row[
                "audit_pass_before"
            ]
            is False
            for row in items
        )

        n_no_repair = sum(
            row[
                "workflow_outcome"
            ]
            == "no_repair_needed"
            for row in items
        )

        n_unsupported = sum(
            row[
                "workflow_outcome"
            ]
            == "no_supported_deterministic_repair"
            for row in items
        )

        n_ambiguous = sum(
            row[
                "workflow_outcome"
            ]
            == "ambiguous_repair_selection"
            for row in items
        )

        n_repair = sum(
            row[
                "workflow_outcome"
            ]
            == "repair_executed"
            for row in items
        )

        n_normalize = sum(
            row[
                "strategy_selected"
            ]
            == "normalize_uncited_detail_bullets"
            for row in items
        )

        n_relocate = sum(
            row[
                "strategy_selected"
            ]
            == "relocate_existing_claim_citations"
            for row in items
        )

        n_consolidate = sum(
            row[
                "strategy_selected"
            ]
            == "consolidate_duplicate_claim_representations"
            for row in items
        )

        n_targeted_success = sum(
            row[
                "targeted_repair_success"
            ]
            is True
            for row in items
        )

        n_full_success = sum(
            row[
                "full_audit_success"
            ]
            is True
            for row in items
        )

        n_new_errors = sum(
            row[
                "new_errors_introduced"
            ]
            is True
            for row in items
        )

        n_final_accepted = (
            n_audit_pass
            + n_full_success
        )

        result.append(
            {
                "cell_name":
                    cell_name,

                "narrative_model":
                    first[
                        "narrative_model"
                    ],

                "narrative_prompt_variant":
                    first[
                        "narrative_prompt_variant"
                    ],

                "narrative_temperature":
                    first[
                        "narrative_temperature"
                    ],

                "n_runs":
                    n_runs,

                "n_upstream_validation_failed":
                    n_upstream,

                "n_workflow_ready":
                    n_workflow_ready,

                "n_audit_pass_before":
                    n_audit_pass,

                "n_audit_fail_before":
                    n_audit_fail,

                "n_no_repair_needed":
                    n_no_repair,

                "n_no_supported_deterministic_repair":
                    n_unsupported,

                "n_ambiguous_repair_selection":
                    n_ambiguous,

                "n_repair_executed":
                    n_repair,

                "n_normalize":
                    n_normalize,

                "n_relocate":
                    n_relocate,

                "n_consolidate":
                    n_consolidate,

                "n_targeted_repair_success":
                    n_targeted_success,

                "n_full_audit_success":
                    n_full_success,

                "n_new_errors_introduced":
                    n_new_errors,

                "n_final_audit_accepted":
                    n_final_accepted,

                "repair_selection_rate_among_audit_failures":
                    safe_rate(
                        n_repair,
                        n_audit_fail,
                    ),

                "targeted_success_rate_among_repairs":
                    safe_rate(
                        n_targeted_success,
                        n_repair,
                    ),

                "full_success_rate_among_repairs":
                    safe_rate(
                        n_full_success,
                        n_repair,
                    ),
            }
        )

    return result


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write one JSON object per line."""
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                )
                + "\n"
            )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: tuple[str, ...],
) -> None:
    """Write CSV with structured cells encoded as compact JSON."""
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key:
                        json_cell(
                            row.get(
                                key
                            )
                        )
                    for key in fieldnames
                }
            )


def aggregate_population(
    *,
    population_dir: Path,
    output_dir: Path,
    source_code_commit: str,
) -> None:
    """Build durable aggregate artifacts."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    population_manifest_path = (
        population_dir
        / "fred_repair_population.json"
    )

    population_manifest = load_json(
        population_manifest_path
    )

    if not isinstance(
        population_manifest,
        dict,
    ):
        raise ValueError(
            "Population manifest must be an object."
        )

    population_paths = sorted(
        population_dir.rglob(
            "population_run.json"
        )
    )

    rows = []

    for path in population_paths:
        record = load_json(
            path
        )

        if not isinstance(
            record,
            dict,
        ):
            raise ValueError(
                "Population-run artifact must "
                f"be an object: {path}"
            )

        rows.append(
            build_row(
                record
            )
        )

    expected_processed = (
        population_manifest.get(
            "n_processed_runs"
        )
    )

    if (
        isinstance(
            expected_processed,
            int,
        )
        and len(rows)
        != expected_processed
    ):
        raise ValueError(
            "Aggregate row count does not match "
            "population manifest processed count: "
            f"{len(rows)} vs {expected_processed}"
        )

    rows = sorted(
        rows,
        key=lambda row: (
            str(
                row[
                    "batch_name"
                ]
            ),
            str(
                row[
                    "cell_name"
                ]
            ),
            str(
                row[
                    "repetition"
                ]
            ),
        ),
    )

    overall = summarize_rows(
        rows
    )

    strategy_summary = (
        summarize_strategies(
            rows
        )
    )

    cell_summary = (
        summarize_cells(
            rows
        )
    )

    jsonl_path = (
        output_dir
        / "fred_repair_population_rows.jsonl"
    )

    csv_path = (
        output_dir
        / "fred_repair_population_rows.csv"
    )

    cell_csv_path = (
        output_dir
        / "fred_repair_population_cells.csv"
    )

    aggregate_path = (
        output_dir
        / "fred_repair_population_aggregate.json"
    )

    write_jsonl(
        jsonl_path,
        rows,
    )

    write_csv(
        csv_path,
        rows,
        ROW_FIELDS,
    )

    write_csv(
        cell_csv_path,
        cell_summary,
        CELL_FIELDS,
    )

    aggregate = {
        "aggregate_schema_version":
            AGGREGATE_SCHEMA_VERSION,

        "aggregate_method":
            AGGREGATE_METHOD,

        "aggregated_at":
            utc_now_iso(),

        "source_code_commit":
            source_code_commit,

        "source_population": {
            "population_dir":
                str(
                    population_dir
                ),

            "population_manifest":
                str(
                    population_manifest_path
                ),

            "population_schema_version":
                population_manifest.get(
                    "population_schema_version"
                ),

            "family_name":
                population_manifest.get(
                    "family_name"
                ),

            "n_batch_dirs":
                population_manifest.get(
                    "n_batch_dirs"
                ),

            "n_discovered_runs":
                population_manifest.get(
                    "n_discovered_runs"
                ),

            "n_processed_runs":
                population_manifest.get(
                    "n_processed_runs"
                ),
        },

        "overall":
            overall,

        "strategy_summary":
            strategy_summary,

        "cell_summary":
            cell_summary,

        "outputs": {
            "aggregate_json":
                str(
                    aggregate_path
                ),

            "rows_jsonl":
                str(
                    jsonl_path
                ),

            "rows_csv":
                str(
                    csv_path
                ),

            "cells_csv":
                str(
                    cell_csv_path
                ),
        },
    }

    write_json(
        aggregate_path,
        aggregate,
    )

    print(
        "Wrote FRED repair population aggregate:"
    )
    print(
        f"  {aggregate_path}"
    )

    print()
    print(
        "=== AGGREGATE SUMMARY ==="
    )

    for key, value in (
        overall.items()
    ):
        print(
            f"{key}:",
            value,
        )

    print()
    print(
        "strategy_summary:"
    )

    for item in strategy_summary:
        print(
            " ",
            item,
        )

    print()
    print(
        "n_cells:",
        len(
            cell_summary
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate a completed FRED "
            "ACT population."
        )
    )

    parser.add_argument(
        "--population-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--source-code-commit",
        required=True,
        help=(
            "Git commit used for the "
            "authoritative population run."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    aggregate_population(
        population_dir=
            args.population_dir,
        output_dir=
            args.output_dir,
        source_code_commit=
            args.source_code_commit,
    )


if __name__ == "__main__":
    main()
