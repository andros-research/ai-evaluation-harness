#!/usr/bin/env python3
"""
Validate the authoritative v1.8.3 FRED ACT population aggregate.

This validator operates on the durable aggregate package rather than
re-reading the original experiment tree.

It checks:

- provenance and source-code identity,
- one-to-one run identity,
- balanced experiment design,
- frozen population denominators,
- upstream-failure semantics,
- workflow-ready semantics,
- no-repair semantics,
- audit-failure accounting,
- repair-selection semantics,
- strategy counts,
- abstention semantics,
- repair safety,
- full-audit success semantics,
- final acceptance accounting,
- infrastructure/data anomaly absence,
- selector ambiguity absence,
- aggregate-summary consistency,
- logical CSV record counts.

Every invariant is written to a durable validation artifact.

A failed invariant does not prevent the validation artifact from being
written. The command exits nonzero after writing the artifact if any
check fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_fred_repair_workflow as workflow


VALIDATION_SCHEMA_VERSION = (
    "fred_repair_population_validation_v0_1"
)

VALIDATION_METHOD = (
    "authoritative_aggregate_invariant_validation"
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


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    """Load JSONL rows."""
    rows: list[
        dict[str, Any]
    ] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue

            payload = json.loads(
                line
            )

            if not isinstance(
                payload,
                dict,
            ):
                raise ValueError(
                    "JSONL row must be an object: "
                    f"{path}:{line_number}"
                )

            rows.append(
                payload
            )

    return rows


def logical_csv_record_count(
    path: Path,
) -> int:
    """Count logical CSV records, excluding the header."""
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return sum(
            1
            for _ in csv.DictReader(
                handle
            )
        )


def normalize_counter(
    value: Counter[Any],
) -> dict[str, int]:
    """Return a JSON-safe sorted Counter representation."""
    return {
        str(key):
            count
        for key, count
        in sorted(
            value.items(),
            key=lambda item:
                str(
                    item[0]
                ),
        )
    }


def add_check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    passed: bool,
    observed: Any,
    expected: Any,
    details: Any = None,
) -> None:
    """Append one durable invariant result."""
    item: dict[
        str,
        Any,
    ] = {
        "name":
            name,

        "passed":
            bool(
                passed
            ),

        "observed":
            observed,

        "expected":
            expected,
    }

    if details is not None:
        item[
            "details"
        ] = details

    checks.append(
        item
    )


def validate_population(
    *,
    aggregate_dir: Path,
    output_dir: Path,
    expected_source_code_commit: str,
) -> bool:
    """
    Validate an authoritative FRED ACT aggregate from its own contracts.

    Scientific outcome counts are derived from the aggregate rows rather
    than hard-coded here.

    The only externally supplied expectation is the source-code commit,
    which acts as a provenance assertion rather than an expected result.
    """
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregate_path = (
        aggregate_dir
        / "fred_repair_population_aggregate.json"
    )

    rows_jsonl_path = (
        aggregate_dir
        / "fred_repair_population_rows.jsonl"
    )

    rows_csv_path = (
        aggregate_dir
        / "fred_repair_population_rows.csv"
    )

    cells_csv_path = (
        aggregate_dir
        / "fred_repair_population_cells.csv"
    )

    validation_path = (
        output_dir
        / "fred_repair_population_validation.json"
    )

    required_files = (
        aggregate_path,
        rows_jsonl_path,
        rows_csv_path,
        cells_csv_path,
    )

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required aggregate artifact(s) missing: "
            + ", ".join(
                missing_files
            )
        )

    aggregate = load_json(
        aggregate_path
    )

    if not isinstance(
        aggregate,
        dict,
    ):
        raise ValueError(
            "Aggregate artifact must be an object."
        )

    rows = load_jsonl(
        rows_jsonl_path
    )

    overall = aggregate.get(
        "overall",
        {},
    )

    if not isinstance(
        overall,
        dict,
    ):
        raise ValueError(
            "Aggregate overall field must be an object."
        )

    source_population = aggregate.get(
        "source_population",
        {},
    )

    if not isinstance(
        source_population,
        dict,
    ):
        raise ValueError(
            "Aggregate source_population must be an object."
        )

    strategy_summary = aggregate.get(
        "strategy_summary",
        [],
    )

    if not isinstance(
        strategy_summary,
        list,
    ):
        raise ValueError(
            "Aggregate strategy_summary must be a list."
        )

    cell_summary = aggregate.get(
        "cell_summary",
        [],
    )

    if not isinstance(
        cell_summary,
        list,
    ):
        raise ValueError(
            "Aggregate cell_summary must be a list."
        )

    checks: list[
        dict[str, Any]
    ] = []


    def safe_rate(
        numerator: int,
        denominator: int,
    ) -> float | None:
        if denominator == 0:
            return None

        return (
            numerator
            / denominator
        )


    def rates_match(
        observed: Any,
        expected: Any,
        *,
        tolerance: float = 1e-12,
    ) -> bool:
        if (
            observed is None
            and expected is None
        ):
            return True

        if (
            isinstance(
                observed,
                (int, float),
            )
            and isinstance(
                expected,
                (int, float),
            )
        ):
            return (
                abs(
                    float(observed)
                    - float(expected)
                )
                <= tolerance
            )

        return observed == expected


    # ----------------------------------------------------------
    # Derived row populations
    # ----------------------------------------------------------

    readiness_counts = Counter(
        row.get(
            "readiness_status"
        )
        for row in rows
    )

    population_status_counts = Counter(
        row.get(
            "population_status"
        )
        for row in rows
    )

    workflow_outcome_counts = Counter(
        row.get(
            "workflow_outcome"
        )
        for row in rows
    )

    strategy_counts = Counter(
        row.get(
            "strategy_selected"
        )
        for row in rows
        if row.get(
            "strategy_selected"
        )
        is not None
    )

    upstream = [
        row
        for row in rows
        if (
            row.get(
                "readiness_status"
            )
            == "upstream_validation_failed"
        )
    ]

    ready = [
        row
        for row in rows
        if (
            row.get(
                "readiness_status"
            )
            == "workflow_ready"
        )
    ]

    initial_pass = [
        row
        for row in ready
        if (
            row.get(
                "audit_pass_before"
            )
            is True
        )
    ]

    initial_fail = [
        row
        for row in ready
        if (
            row.get(
                "audit_pass_before"
            )
            is False
        )
    ]

    repairs = [
        row
        for row in initial_fail
        if (
            row.get(
                "workflow_outcome"
            )
            == "repair_executed"
        )
    ]

    unsupported = [
        row
        for row in initial_fail
        if (
            row.get(
                "workflow_outcome"
            )
            == "no_supported_deterministic_repair"
        )
    ]

    ambiguous_rows = [
        row
        for row in initial_fail
        if (
            row.get(
                "workflow_outcome"
            )
            == "ambiguous_repair_selection"
        )
    ]

    full_success = [
        row
        for row in repairs
        if (
            row.get(
                "full_audit_success"
            )
            is True
        )
    ]

    targeted_success = [
        row
        for row in repairs
        if (
            row.get(
                "targeted_repair_success"
            )
            is True
        )
    ]

    new_error_rows = [
        row
        for row in repairs
        if (
            row.get(
                "new_errors_introduced"
            )
            is True
        )
    ]


    # ----------------------------------------------------------
    # 1. Provenance / identity
    # ----------------------------------------------------------

    source_commit = aggregate.get(
        "source_code_commit"
    )

    add_check(
        checks,
        name=
            "source_code_commit",
        passed=(
            source_commit
            == expected_source_code_commit
        ),
        observed=
            source_commit,
        expected=
            expected_source_code_commit,
    )

    expected_processed = (
        source_population.get(
            "n_processed_runs"
        )
    )

    add_check(
        checks,
        name=
            "row_count_matches_source_population",
        passed=(
            isinstance(
                expected_processed,
                int,
            )
            and len(rows)
            == expected_processed
        ),
        observed=
            len(rows),
        expected=
            expected_processed,
    )

    source_paths = [
        row.get(
            "source_run_dir"
        )
        for row in rows
    ]

    logical_ids = [
        (
            row.get(
                "batch_name"
            ),
            row.get(
                "cell_name"
            ),
            row.get(
                "repetition"
            ),
        )
        for row in rows
    ]

    add_check(
        checks,
        name=
            "unique_source_paths",
        passed=(
            len(
                set(
                    source_paths
                )
            )
            == len(rows)
        ),
        observed=
            len(
                set(
                    source_paths
                )
            ),
        expected=
            len(rows),
    )

    add_check(
        checks,
        name=
            "unique_logical_run_ids",
        passed=(
            len(
                set(
                    logical_ids
                )
            )
            == len(rows)
        ),
        observed=
            len(
                set(
                    logical_ids
                )
            ),
        expected=
            len(rows),
    )


    # ----------------------------------------------------------
    # 2. Required analysis metadata
    # ----------------------------------------------------------

    metadata_fields = (
        "cell_name",
        "narrative_model",
        "narrative_prompt_variant",
        "narrative_temperature",
    )

    metadata_missing = {
        key:
            sum(
                row.get(
                    key
                )
                is None
                for row in rows
            )
        for key in metadata_fields
    }

    add_check(
        checks,
        name=
            "analysis_metadata_complete",
        passed=(
            all(
                count == 0
                for count
                in metadata_missing.values()
            )
        ),
        observed=
            metadata_missing,
        expected={
            key:
                0
            for key in metadata_fields
        },
    )


    # ----------------------------------------------------------
    # 3. Cell summary / experiment-design consistency
    # ----------------------------------------------------------

    rows_by_cell: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for row in rows:
        cell_name = row.get(
            "cell_name"
        )

        if isinstance(
            cell_name,
            str,
        ):
            rows_by_cell.setdefault(
                cell_name,
                [],
            ).append(
                row
            )

    cells_by_name = {
        item.get(
            "cell_name"
        ):
            item
        for item in cell_summary
        if isinstance(
            item,
            dict,
        )
        and isinstance(
            item.get(
                "cell_name"
            ),
            str,
        )
    }

    cell_names_match = (
        set(
            rows_by_cell
        )
        == set(
            cells_by_name
        )
    )

    cell_contract_violations = []

    for cell_name, cell_rows in (
        rows_by_cell.items()
    ):
        summary = cells_by_name.get(
            cell_name
        )

        if not isinstance(
            summary,
            dict,
        ):
            cell_contract_violations.append(
                {
                    "cell_name":
                        cell_name,
                    "reason":
                        "missing_cell_summary",
                }
            )
            continue

        models = {
            row.get(
                "narrative_model"
            )
            for row in cell_rows
        }

        prompts = {
            row.get(
                "narrative_prompt_variant"
            )
            for row in cell_rows
        }

        temperatures = {
            row.get(
                "narrative_temperature"
            )
            for row in cell_rows
        }

        n_upstream = sum(
            row.get(
                "readiness_status"
            )
            == "upstream_validation_failed"
            for row in cell_rows
        )

        n_ready = sum(
            row.get(
                "readiness_status"
            )
            == "workflow_ready"
            for row in cell_rows
        )

        n_pass = sum(
            row.get(
                "audit_pass_before"
            )
            is True
            for row in cell_rows
        )

        n_fail = sum(
            row.get(
                "audit_pass_before"
            )
            is False
            for row in cell_rows
        )

        n_no_repair = sum(
            row.get(
                "workflow_outcome"
            )
            == "no_repair_needed"
            for row in cell_rows
        )

        n_unsupported = sum(
            row.get(
                "workflow_outcome"
            )
            == "no_supported_deterministic_repair"
            for row in cell_rows
        )

        n_ambiguous = sum(
            row.get(
                "workflow_outcome"
            )
            == "ambiguous_repair_selection"
            for row in cell_rows
        )

        n_repair = sum(
            row.get(
                "workflow_outcome"
            )
            == "repair_executed"
            for row in cell_rows
        )

        n_normalize = sum(
            row.get(
                "strategy_selected"
            )
            == "normalize_uncited_detail_bullets"
            for row in cell_rows
        )

        n_relocate = sum(
            row.get(
                "strategy_selected"
            )
            == "relocate_existing_claim_citations"
            for row in cell_rows
        )

        n_consolidate = sum(
            row.get(
                "strategy_selected"
            )
            == "consolidate_duplicate_claim_representations"
            for row in cell_rows
        )

        n_targeted = sum(
            row.get(
                "targeted_repair_success"
            )
            is True
            for row in cell_rows
        )

        n_full = sum(
            row.get(
                "full_audit_success"
            )
            is True
            for row in cell_rows
        )

        n_new_errors = sum(
            row.get(
                "new_errors_introduced"
            )
            is True
            for row in cell_rows
        )

        expected_values = {
            "n_runs":
                len(
                    cell_rows
                ),

            "n_upstream_validation_failed":
                n_upstream,

            "n_workflow_ready":
                n_ready,

            "n_audit_pass_before":
                n_pass,

            "n_audit_fail_before":
                n_fail,

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
                n_targeted,

            "n_full_audit_success":
                n_full,

            "n_new_errors_introduced":
                n_new_errors,

            "n_final_audit_accepted":
                (
                    n_pass
                    + n_full
                ),
        }

        bad_fields = {
            key: {
                "observed":
                    summary.get(
                        key
                    ),
                "expected":
                    value,
            }
            for key, value
            in expected_values.items()
            if summary.get(
                key
            )
            != value
        }

        expected_rates = {
            "repair_selection_rate_among_audit_failures":
                safe_rate(
                    n_repair,
                    n_fail,
                ),

            "targeted_success_rate_among_repairs":
                safe_rate(
                    n_targeted,
                    n_repair,
                ),

            "full_success_rate_among_repairs":
                safe_rate(
                    n_full,
                    n_repair,
                ),
        }

        bad_rates = {
            key: {
                "observed":
                    summary.get(
                        key
                    ),
                "expected":
                    value,
            }
            for key, value
            in expected_rates.items()
            if not rates_match(
                summary.get(
                    key
                ),
                value,
            )
        }

        metadata_ok = (
            len(
                models
            )
            == 1
            and len(
                prompts
            )
            == 1
            and len(
                temperatures
            )
            == 1
            and summary.get(
                "narrative_model"
            )
            in models
            and summary.get(
                "narrative_prompt_variant"
            )
            in prompts
            and summary.get(
                "narrative_temperature"
            )
            in temperatures
        )

        if (
            bad_fields
            or bad_rates
            or not metadata_ok
        ):
            cell_contract_violations.append(
                {
                    "cell_name":
                        cell_name,
                    "metadata_ok":
                        metadata_ok,
                    "bad_fields":
                        bad_fields,
                    "bad_rates":
                        bad_rates,
                }
            )

    add_check(
        checks,
        name=
            "cell_summary_consistency",
        passed=(
            cell_names_match
            and not cell_contract_violations
        ),
        observed={
            "n_row_cells":
                len(
                    rows_by_cell
                ),
            "n_summary_cells":
                len(
                    cells_by_name
                ),
            "cell_names_match":
                cell_names_match,
            "n_contract_violations":
                len(
                    cell_contract_violations
                ),
        },
        expected={
            "n_row_cells":
                len(
                    rows_by_cell
                ),
            "n_summary_cells":
                len(
                    rows_by_cell
                ),
            "cell_names_match":
                True,
            "n_contract_violations":
                0,
        },
        details=
            cell_contract_violations,
    )


    # ----------------------------------------------------------
    # 4. Upstream-failure contract
    # ----------------------------------------------------------

    upstream_violations = [
        row.get(
            "source_run_dir"
        )
        for row in upstream
        if not (
            row.get(
                "workflow_outcome"
            )
            == "upstream_validation_failed"
            and row.get(
                "audit_pass_before"
            )
            is None
            and row.get(
                "repair_needed"
            )
            is None
            and row.get(
                "strategy_selected"
            )
            is None
            and row.get(
                "repair_applied"
            )
            is None
            and row.get(
                "audit_pass_after"
            )
            is None
            and isinstance(
                row.get(
                    "upstream_validation_error"
                ),
                str,
            )
            and bool(
                row.get(
                    "upstream_validation_error"
                )
            )
        )
    ]

    add_check(
        checks,
        name=
            "upstream_failure_contract",
        passed=(
            not upstream_violations
        ),
        observed={
            "n_rows":
                len(
                    upstream
                ),
            "n_contract_violations":
                len(
                    upstream_violations
                ),
        },
        expected={
            "n_rows":
                len(
                    upstream
                ),
            "n_contract_violations":
                0,
        },
        details=
            upstream_violations[
                :20
            ],
    )


    # ----------------------------------------------------------
    # 5. Workflow-ready contract
    # ----------------------------------------------------------

    ready_violations = [
        row.get(
            "source_run_dir"
        )
        for row in ready
        if not (
            isinstance(
                row.get(
                    "audit_pass_before"
                ),
                bool,
            )
            and isinstance(
                row.get(
                    "strict_selected_claim_coverage"
                ),
                bool,
            )
        )
    ]

    add_check(
        checks,
        name=
            "workflow_ready_contract",
        passed=(
            not ready_violations
        ),
        observed={
            "n_rows":
                len(
                    ready
                ),
            "n_contract_violations":
                len(
                    ready_violations
                ),
        },
        expected={
            "n_rows":
                len(
                    ready
                ),
            "n_contract_violations":
                0,
        },
        details=
            ready_violations[
                :20
            ],
    )


    # ----------------------------------------------------------
    # 6. No-repair contract
    # ----------------------------------------------------------

    initial_pass_violations = [
        row.get(
            "source_run_dir"
        )
        for row in initial_pass
        if not (
            row.get(
                "repair_needed"
            )
            is False
            and row.get(
                "workflow_outcome"
            )
            == "no_repair_needed"
            and row.get(
                "strategy_selected"
            )
            is None
            and row.get(
                "repair_applied"
            )
            is False
            and row.get(
                "audit_pass_after"
            )
            is None
        )
    ]

    add_check(
        checks,
        name=
            "no_repair_needed_contract",
        passed=(
            not initial_pass_violations
        ),
        observed={
            "n_rows":
                len(
                    initial_pass
                ),
            "n_contract_violations":
                len(
                    initial_pass_violations
                ),
        },
        expected={
            "n_rows":
                len(
                    initial_pass
                ),
            "n_contract_violations":
                0,
        },
        details=
            initial_pass_violations[
                :20
            ],
    )


    # ----------------------------------------------------------
    # 7. Audit-failure routing contract
    # ----------------------------------------------------------

    allowed_failure_outcomes = {
        "repair_executed",
        "no_supported_deterministic_repair",
        "ambiguous_repair_selection",
    }

    failure_routing_violations = [
        row.get(
            "source_run_dir"
        )
        for row in initial_fail
        if not (
            row.get(
                "repair_needed"
            )
            is True
            and row.get(
                "workflow_outcome"
            )
            in allowed_failure_outcomes
        )
    ]

    routed_failure_count = (
        len(
            repairs
        )
        + len(
            unsupported
        )
        + len(
            ambiguous_rows
        )
    )

    add_check(
        checks,
        name=
            "audit_failure_routing_contract",
        passed=(
            not failure_routing_violations
            and routed_failure_count
            == len(
                initial_fail
            )
        ),
        observed={
            "n_initial_failures":
                len(
                    initial_fail
                ),
            "n_routed":
                routed_failure_count,
            "n_contract_violations":
                len(
                    failure_routing_violations
                ),
        },
        expected={
            "n_initial_failures":
                len(
                    initial_fail
                ),
            "n_routed":
                len(
                    initial_fail
                ),
            "n_contract_violations":
                0,
        },
        details=
            failure_routing_violations[
                :20
            ],
    )


    # ----------------------------------------------------------
    # 8. Repair-selection / capability-registry contract
    # ----------------------------------------------------------

    allowed_strategies = set(
        workflow.STRATEGY_REGISTRY
    )

    repair_violations = []

    for row in repairs:
        strategy = row.get(
            "strategy_selected"
        )

        capability = (
            workflow.STRATEGY_REGISTRY.get(
                strategy
            )
            if isinstance(
                strategy,
                str,
            )
            else None
        )

        expected_target_error = (
            capability.get(
                "target_error"
            )
            if isinstance(
                capability,
                dict,
            )
            else None
        )

        valid = (
            row.get(
                "selection_status"
            )
            == "selected"
            and strategy
            in allowed_strategies
            and row.get(
                "eligible_strategies"
            )
            == [
                strategy
            ]
            and row.get(
                "ambiguous"
            )
            is False
            and row.get(
                "repair_applied"
            )
            is True
            and row.get(
                "target_error"
            )
            == expected_target_error
            and row.get(
                "target_error_resolved"
            )
            is True
            and row.get(
                "targeted_repair_success"
            )
            is True
            and isinstance(
                row.get(
                    "audit_pass_after"
                ),
                bool,
            )
        )

        if not valid:
            repair_violations.append(
                {
                    "source_run_dir":
                        row.get(
                            "source_run_dir"
                        ),
                    "strategy":
                        strategy,
                    "expected_target_error":
                        expected_target_error,
                    "observed_target_error":
                        row.get(
                            "target_error"
                        ),
                }
            )

    add_check(
        checks,
        name=
            "repair_capability_contract",
        passed=(
            not repair_violations
        ),
        observed={
            "n_repairs":
                len(
                    repairs
                ),
            "strategy_counts":
                normalize_counter(
                    strategy_counts
                ),
            "registered_strategies":
                sorted(
                    allowed_strategies
                ),
            "n_contract_violations":
                len(
                    repair_violations
                ),
        },
        expected={
            "n_repairs":
                len(
                    repairs
                ),
            "strategy_counts":
                normalize_counter(
                    strategy_counts
                ),
            "registered_strategies":
                sorted(
                    allowed_strategies
                ),
            "n_contract_violations":
                0,
        },
        details=
            repair_violations[
                :20
            ],
    )


    # ----------------------------------------------------------
    # 9. Unsupported-abstention contract
    # ----------------------------------------------------------

    unsupported_violations = [
        row.get(
            "source_run_dir"
        )
        for row in unsupported
        if not (
            row.get(
                "selection_status"
            )
            == "abstained"
            and row.get(
                "strategy_selected"
            )
            is None
            and row.get(
                "eligible_strategies"
            )
            == []
            and row.get(
                "ambiguous"
            )
            is False
            and row.get(
                "repair_applied"
            )
            is False
            and row.get(
                "audit_pass_after"
            )
            is None
        )
    ]

    add_check(
        checks,
        name=
            "unsupported_abstention_contract",
        passed=(
            not unsupported_violations
        ),
        observed={
            "n_rows":
                len(
                    unsupported
                ),
            "n_contract_violations":
                len(
                    unsupported_violations
                ),
        },
        expected={
            "n_rows":
                len(
                    unsupported
                ),
            "n_contract_violations":
                0,
        },
        details=
            unsupported_violations[
                :20
            ],
    )


    # ----------------------------------------------------------
    # 10. Ambiguous-abstention contract
    # ----------------------------------------------------------

    ambiguous_violations = [
        row.get(
            "source_run_dir"
        )
        for row in ambiguous_rows
        if not (
            row.get(
                "selection_status"
            )
            == "abstained"
            and row.get(
                "strategy_selected"
            )
            is None
            and isinstance(
                row.get(
                    "eligible_strategies"
                ),
                list,
            )
            and len(
                row.get(
                    "eligible_strategies"
                )
            )
            >= 2
            and row.get(
                "ambiguous"
            )
            is True
            and row.get(
                "repair_applied"
            )
            is False
            and row.get(
                "audit_pass_after"
            )
            is None
        )
    ]

    add_check(
        checks,
        name=
            "ambiguous_abstention_contract",
        passed=(
            not ambiguous_violations
        ),
        observed={
            "n_rows":
                len(
                    ambiguous_rows
                ),
            "n_contract_violations":
                len(
                    ambiguous_violations
                ),
        },
        expected={
            "n_rows":
                len(
                    ambiguous_rows
                ),
            "n_contract_violations":
                0,
        },
        details=
            ambiguous_violations[
                :20
            ],
    )


    # ----------------------------------------------------------
    # 11. Repair safety
    # ----------------------------------------------------------

    safety_violations = [
        row.get(
            "source_run_dir"
        )
        for row in repairs
        if not (
            row.get(
                "new_errors_introduced"
            )
            is False
            and row.get(
                "introduced_errors"
            )
            == []
        )
    ]

    add_check(
        checks,
        name=
            "repair_safety",
        passed=(
            not safety_violations
        ),
        observed={
            "n_repairs":
                len(
                    repairs
                ),
            "n_contract_violations":
                len(
                    safety_violations
                ),
        },
        expected={
            "n_repairs":
                len(
                    repairs
                ),
            "n_contract_violations":
                0,
        },
        details=
            safety_violations[
                :20
            ],
    )


    # ----------------------------------------------------------
    # 12. Full-audit success contract
    # ----------------------------------------------------------

    full_success_violations = [
        row.get(
            "source_run_dir"
        )
        for row in full_success
        if not (
            row.get(
                "audit_pass_after"
            )
            is True
            and row.get(
                "residual_errors"
            )
            == []
        )
    ]

    add_check(
        checks,
        name=
            "full_audit_success_contract",
        passed=(
            not full_success_violations
        ),
        observed={
            "n_full_success":
                len(
                    full_success
                ),
            "n_contract_violations":
                len(
                    full_success_violations
                ),
        },
        expected={
            "n_full_success":
                len(
                    full_success
                ),
            "n_contract_violations":
                0,
        },
        details=
            full_success_violations[
                :20
            ],
    )


    # ----------------------------------------------------------
    # 13. Final acceptance accounting
    # ----------------------------------------------------------

    final_acceptance = (
        len(
            initial_pass
        )
        + len(
            full_success
        )
    )

    add_check(
        checks,
        name=
            "final_acceptance_accounting",
        passed=(
            overall.get(
                "n_final_audit_accepted"
            )
            == final_acceptance
        ),
        observed=
            overall.get(
                "n_final_audit_accepted"
            ),
        expected=
            final_acceptance,
        details={
            "initially_accepted":
                len(
                    initial_pass
                ),
            "promoted_by_act":
                len(
                    full_success
                ),
        },
    )


    # ----------------------------------------------------------
    # 14. Infrastructure / data anomaly contract
    # ----------------------------------------------------------

    infrastructure_violations = [
        row.get(
            "source_run_dir"
        )
        for row in rows
        if not (
            row.get(
                "population_status"
            )
            == "completed"
            and row.get(
                "error_type"
            )
            is None
            and row.get(
                "error_message"
            )
            is None
        )
    ]

    add_check(
        checks,
        name=
            "infrastructure_and_data_anomalies",
        passed=(
            not infrastructure_violations
        ),
        observed={
            "n_rows":
                len(
                    rows
                ),
            "n_contract_violations":
                len(
                    infrastructure_violations
                ),
        },
        expected={
            "n_rows":
                len(
                    rows
                ),
            "n_contract_violations":
                0,
        },
        details=
            infrastructure_violations[
                :20
            ],
    )


    # ----------------------------------------------------------
    # 15. Overall aggregate consistency
    # ----------------------------------------------------------

    calculated_overall = {
        "n_runs":
            len(
                rows
            ),

        "readiness_counts":
            normalize_counter(
                readiness_counts
            ),

        "population_status_counts":
            normalize_counter(
                population_status_counts
            ),

        "workflow_outcome_counts":
            normalize_counter(
                workflow_outcome_counts
            ),

        "strategy_selected_counts":
            normalize_counter(
                strategy_counts
            ),

        "n_upstream_validation_failed":
            len(
                upstream
            ),

        "n_workflow_ready":
            len(
                ready
            ),

        "n_audit_pass_before":
            len(
                initial_pass
            ),

        "n_audit_fail_before":
            len(
                initial_fail
            ),

        "n_repair_executed":
            len(
                repairs
            ),

        "n_no_supported_deterministic_repair":
            len(
                unsupported
            ),

        "n_ambiguous_repair_selection":
            len(
                ambiguous_rows
            ),

        "n_targeted_repair_success":
            len(
                targeted_success
            ),

        "n_full_audit_success":
            len(
                full_success
            ),

        "n_new_errors_introduced":
            len(
                new_error_rows
            ),

        "n_final_audit_accepted":
            final_acceptance,
    }

    overall_bad_fields = {
        key: {
            "observed":
                overall.get(
                    key
                ),
            "expected":
                value,
        }
        for key, value
        in calculated_overall.items()
        if overall.get(
            key
        )
        != value
    }

    expected_overall_rates = {
        "repair_selection_rate_among_audit_failures":
            safe_rate(
                len(
                    repairs
                ),
                len(
                    initial_fail
                ),
            ),

        "targeted_success_rate_among_repairs":
            safe_rate(
                len(
                    targeted_success
                ),
                len(
                    repairs
                ),
            ),

        "full_success_rate_among_repairs":
            safe_rate(
                len(
                    full_success
                ),
                len(
                    repairs
                ),
            ),

        "final_acceptance_rate_among_workflow_ready":
            safe_rate(
                final_acceptance,
                len(
                    ready
                ),
            ),

        "final_acceptance_rate_among_all_attempts":
            safe_rate(
                final_acceptance,
                len(
                    rows
                ),
            ),
    }

    overall_bad_rates = {
        key: {
            "observed":
                overall.get(
                    key
                ),
            "expected":
                value,
        }
        for key, value
        in expected_overall_rates.items()
        if not rates_match(
            overall.get(
                key
            ),
            value,
        )
    }

    add_check(
        checks,
        name=
            "overall_aggregate_consistency",
        passed=(
            not overall_bad_fields
            and not overall_bad_rates
        ),
        observed={
            "bad_fields":
                overall_bad_fields,
            "bad_rates":
                overall_bad_rates,
        },
        expected={
            "bad_fields":
                {},
            "bad_rates":
                {},
        },
    )


    # ----------------------------------------------------------
    # 16. Strategy-summary consistency
    # ----------------------------------------------------------

    summaries_by_strategy = {
        item.get(
            "strategy"
        ):
            item
        for item in strategy_summary
        if isinstance(
            item,
            dict,
        )
        and isinstance(
            item.get(
                "strategy"
            ),
            str,
        )
    }

    strategy_names_match = (
        set(
            summaries_by_strategy
        )
        == set(
            strategy_counts
        )
    )

    strategy_summary_violations = []

    for strategy, n_selected in (
        strategy_counts.items()
    ):
        summary = (
            summaries_by_strategy.get(
                strategy
            )
        )

        if not isinstance(
            summary,
            dict,
        ):
            strategy_summary_violations.append(
                {
                    "strategy":
                        strategy,
                    "reason":
                        "missing_strategy_summary",
                }
            )
            continue

        strategy_rows = [
            row
            for row in repairs
            if (
                row.get(
                    "strategy_selected"
                )
                == strategy
            )
        ]

        n_targeted = sum(
            row.get(
                "targeted_repair_success"
            )
            is True
            for row in strategy_rows
        )

        n_full = sum(
            row.get(
                "full_audit_success"
            )
            is True
            for row in strategy_rows
        )

        n_new = sum(
            row.get(
                "new_errors_introduced"
            )
            is True
            for row in strategy_rows
        )

        expected_fields = {
            "n_selected":
                n_selected,

            "n_targeted_repair_success":
                n_targeted,

            "n_full_audit_success":
                n_full,

            "n_new_errors_introduced":
                n_new,
        }

        bad_fields = {
            key: {
                "observed":
                    summary.get(
                        key
                    ),
                "expected":
                    value,
            }
            for key, value
            in expected_fields.items()
            if summary.get(
                key
            )
            != value
        }

        expected_rates = {
            "targeted_success_rate":
                safe_rate(
                    n_targeted,
                    n_selected,
                ),

            "full_success_rate":
                safe_rate(
                    n_full,
                    n_selected,
                ),
        }

        bad_rates = {
            key: {
                "observed":
                    summary.get(
                        key
                    ),
                "expected":
                    value,
            }
            for key, value
            in expected_rates.items()
            if not rates_match(
                summary.get(
                    key
                ),
                value,
            )
        }

        if (
            bad_fields
            or bad_rates
        ):
            strategy_summary_violations.append(
                {
                    "strategy":
                        strategy,
                    "bad_fields":
                        bad_fields,
                    "bad_rates":
                        bad_rates,
                }
            )

    add_check(
        checks,
        name=
            "strategy_summary_consistency",
        passed=(
            strategy_names_match
            and not strategy_summary_violations
        ),
        observed={
            "strategy_names_match":
                strategy_names_match,
            "n_contract_violations":
                len(
                    strategy_summary_violations
                ),
        },
        expected={
            "strategy_names_match":
                True,
            "n_contract_violations":
                0,
        },
        details=
            strategy_summary_violations,
    )


    # ----------------------------------------------------------
    # 17. Logical CSV record counts
    # ----------------------------------------------------------

    logical_row_csv_records = (
        logical_csv_record_count(
            rows_csv_path
        )
    )

    logical_cell_csv_records = (
        logical_csv_record_count(
            cells_csv_path
        )
    )

    add_check(
        checks,
        name=
            "logical_csv_record_counts",
        passed=(
            logical_row_csv_records
            == len(
                rows
            )
            and logical_cell_csv_records
            == len(
                cell_summary
            )
        ),
        observed={
            "row_csv_records":
                logical_row_csv_records,
            "cell_csv_records":
                logical_cell_csv_records,
        },
        expected={
            "row_csv_records":
                len(
                    rows
                ),
            "cell_csv_records":
                len(
                    cell_summary
                ),
        },
    )


    # ----------------------------------------------------------
    # Validation result
    # ----------------------------------------------------------

    n_checks = len(
        checks
    )

    n_passed = sum(
        check[
            "passed"
        ]
        for check in checks
    )

    n_failed = (
        n_checks
        - n_passed
    )

    overall_pass = (
        n_failed
        == 0
    )

    validation = {
        "validation_schema_version":
            VALIDATION_SCHEMA_VERSION,

        "validation_method":
            VALIDATION_METHOD,

        "validated_at":
            utc_now_iso(),

        "aggregate_dir":
            str(
                aggregate_dir
            ),

        "source_aggregate":
            str(
                aggregate_path
            ),

        "expected_source_code_commit":
            expected_source_code_commit,

        "observed_source_code_commit":
            source_commit,

        "overall_pass":
            overall_pass,

        "n_checks":
            n_checks,

        "n_passed":
            n_passed,

        "n_failed":
            n_failed,

        "derived_population": {
            "n_runs":
                len(
                    rows
                ),

            "n_cells":
                len(
                    rows_by_cell
                ),

            "n_upstream_validation_failed":
                len(
                    upstream
                ),

            "n_workflow_ready":
                len(
                    ready
                ),

            "n_initially_accepted":
                len(
                    initial_pass
                ),

            "n_initial_audit_failures":
                len(
                    initial_fail
                ),

            "n_repairs_executed":
                len(
                    repairs
                ),

            "n_unsupported":
                len(
                    unsupported
                ),

            "n_ambiguous":
                len(
                    ambiguous_rows
                ),

            "n_targeted_success":
                len(
                    targeted_success
                ),

            "n_full_success":
                len(
                    full_success
                ),

            "n_final_accepted":
                final_acceptance,
        },

        "checks":
            checks,

        "outputs": {
            "validation_json":
                str(
                    validation_path
                ),
        },
    }

    write_json(
        validation_path,
        validation,
    )

    print(
        "Wrote FRED repair population validation:"
    )
    print(
        f"  {validation_path}"
    )

    print()
    print(
        "=== GENERIC FRED ACT POPULATION VALIDATION ==="
    )

    for check in checks:
        status = (
            "PASS"
            if check[
                "passed"
            ]
            else "FAIL"
        )

        print(
            f"{status:4} "
            f"{check['name']}"
        )

    print()
    print(
        "checks:",
        n_checks,
    )
    print(
        "passed:",
        n_passed,
    )
    print(
        "failed:",
        n_failed,
    )
    print(
        "overall_pass:",
        overall_pass,
    )

    return overall_pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the authoritative "
            "v1.8.3 FRED ACT aggregate."
        )
    )

    parser.add_argument(
        "--aggregate-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--expected-source-code-commit",
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    passed = validate_population(
        aggregate_dir=
            args.aggregate_dir,
        output_dir=
            args.output_dir,
        expected_source_code_commit=
            args.expected_source_code_commit,
    )

    if not passed:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
