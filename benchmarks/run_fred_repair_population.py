#!/usr/bin/env python3
"""
Run deterministic FRED ACT workflows across a frozen experiment population.

The population runner:

    discovers the unsuffixed original family root plus __batch_* roots
        ->
    discovers every repetition run
        ->
    classifies each run's artifact readiness
        ->
    runs the single-run ACT workflow when workflow-ready
        OR
    records an explicit upstream validation failure
        OR
    records an unrecognized/inconsistent input state
        ->
    persists one population-run artifact per source run
        ->
    writes a population manifest and summary

The runner does not reinterpret repair semantics. It delegates all
single-run ACT behavior to run_fred_repair_workflow.py.

A population run continues after individual workflow/input failures.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_fred_repair_workflow as workflow


POPULATION_SCHEMA_VERSION = (
    "fred_repair_population_v0_1"
)

POPULATION_RUN_SCHEMA_VERSION = (
    "fred_repair_population_run_v0_1"
)

POPULATION_METHOD = (
    "discover_classify_execute_preserve"
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


def batch_sort_key(
    path: Path,
    family_name: str,
) -> tuple[int, int]:
    """
    Sort the original unsuffixed family root first,
    followed by numbered batch roots.
    """
    if path.name == family_name:
        return (
            0,
            1,
        )

    match = re.fullmatch(
        re.escape(
            family_name
        )
        + r"__batch_(\d+)",
        path.name,
    )

    if match is None:
        return (
            2,
            0,
        )

    return (
        1,
        int(
            match.group(1)
        ),
    )


def discover_batch_dirs(
    *,
    input_base: Path,
    family_name: str,
) -> list[Path]:
    """Discover the original family root and numbered batch roots."""
    batch_dirs: list[
        Path
    ] = []

    original = (
        input_base
        / family_name
    )

    if original.is_dir():
        batch_dirs.append(
            original
        )

    batch_dirs.extend(
        path
        for path in input_base.glob(
            family_name
            + "__batch_*"
        )
        if path.is_dir()
    )

    batch_dirs = sorted(
        set(
            batch_dirs
        ),
        key=lambda path:
            batch_sort_key(
                path,
                family_name,
            ),
    )

    return batch_dirs


def discover_run_dirs(
    batch_dirs: list[Path],
) -> list[
    tuple[Path, Path]
]:
    """
    Return (batch_dir, run_dir) tuples for every repetition.
    """
    discovered: list[
        tuple[Path, Path]
    ] = []

    for batch_dir in batch_dirs:
        runs_dir = (
            batch_dir
            / "runs"
        )

        if not runs_dir.is_dir():
            continue

        for run_dir in sorted(
            runs_dir.glob(
                "*/repetition_*"
            )
        ):
            if run_dir.is_dir():
                discovered.append(
                    (
                        batch_dir,
                        run_dir,
                    )
                )

    return discovered


def source_artifact_paths(
    run_dir: Path,
) -> dict[str, Path]:
    """Return source artifact paths used for readiness classification."""
    return {
        "narrative":
            run_dir
            / "fred_narratives"
            / "fred_narrative.md",

        "audit":
            run_dir
            / "fred_audits"
            / "fred_narrative_audit.json",

        "repair_plan":
            run_dir
            / "fred_repairs"
            / "fred_repair_plan.json",

        "selected_claims":
            run_dir
            / "fred_claims"
            / "selected_fred_claims.json",

        "failed_narrative":
            run_dir
            / "fred_narratives"
            / "fred_narrative_failed_validation.md",

        "failed_narrative_metadata":
            run_dir
            / "fred_narratives"
            / "fred_narrative_failed_validation_metadata.json",

        "latest_evidence_loop_run":
            run_dir
            / "fred_runs"
            / "latest_fred_evidence_loop_run.json",
    }


def classify_run(
    run_dir: Path,
) -> dict[str, Any]:
    """Classify one source run from its durable artifact state."""
    paths = source_artifact_paths(
        run_dir
    )

    core_names = (
        "narrative",
        "audit",
        "repair_plan",
        "selected_claims",
    )

    core_exists = {
        name:
            paths[
                name
            ].exists()
        for name in core_names
    }

    if all(
        core_exists.values()
    ):
        return {
            "readiness_status":
                "workflow_ready",

            "readiness_reason":
                "all_required_act_inputs_present",

            "artifact_presence":
                {
                    name:
                        path.exists()
                    for name, path
                    in paths.items()
                },
        }

    failed_validation_pair = (
        paths[
            "failed_narrative"
        ].exists()
        and paths[
            "failed_narrative_metadata"
        ].exists()
    )

    core_act_outputs_absent = (
        not paths[
            "narrative"
        ].exists()
        and not paths[
            "audit"
        ].exists()
        and not paths[
            "repair_plan"
        ].exists()
    )

    if (
        paths[
            "selected_claims"
        ].exists()
        and failed_validation_pair
        and core_act_outputs_absent
    ):
        metadata = load_json(
            paths[
                "failed_narrative_metadata"
            ]
        )

        validation_error = None
        generation_method = None

        if isinstance(
            metadata,
            dict,
        ):
            validation_error = (
                metadata.get(
                    "validation_error"
                )
            )

            generation_method = (
                metadata.get(
                    "generation_method"
                )
            )

        return {
            "readiness_status":
                "upstream_validation_failed",

            "readiness_reason":
                "narrative_failed_pre_audit_validation",

            "generation_method":
                generation_method,

            "validation_error":
                validation_error,

            "artifact_presence":
                {
                    name:
                        path.exists()
                    for name, path
                    in paths.items()
                },
        }

    return {
        "readiness_status":
            "population_input_error",

        "readiness_reason":
            "unrecognized_or_incomplete_artifact_state",

        "artifact_presence":
            {
                name:
                    path.exists()
                for name, path
                in paths.items()
            },
    }


def output_run_dir(
    *,
    population_output_dir: Path,
    batch_dir: Path,
    run_dir: Path,
) -> Path:
    """Mirror source batch/cell/repetition identity in output."""
    cell_name = (
        run_dir.parent.name
    )

    repetition_name = (
        run_dir.name
    )

    return (
        population_output_dir
        / batch_dir.name
        / "runs"
        / cell_name
        / repetition_name
    )


def population_run_path(
    output_dir: Path,
) -> Path:
    return (
        output_dir
        / "population_run.json"
    )


def process_run(
    *,
    batch_dir: Path,
    run_dir: Path,
    population_output_dir: Path,
) -> dict[str, Any]:
    """Classify and, when eligible, execute one source run."""
    started_at = (
        utc_now_iso()
    )

    classification = (
        classify_run(
            run_dir
        )
    )

    out = output_run_dir(
        population_output_dir=
            population_output_dir,
        batch_dir=
            batch_dir,
        run_dir=
            run_dir,
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    readiness_status = (
        classification[
            "readiness_status"
        ]
    )

    record: dict[
        str,
        Any,
    ] = {
        "population_run_schema_version":
            POPULATION_RUN_SCHEMA_VERSION,

        "population_status":
            None,

        "readiness_status":
            readiness_status,

        "readiness_reason":
            classification[
                "readiness_reason"
            ],

        "source": {
            "batch_name":
                batch_dir.name,

            "cell_name":
                run_dir.parent.name,

            "repetition":
                run_dir.name,

            "run_dir":
                str(
                    run_dir
                ),
        },

        "artifact_presence":
            classification.get(
                "artifact_presence",
                {},
            ),

        "started_at":
            started_at,

        "completed_at":
            None,

        "workflow_outcome":
            None,

        "workflow_json":
            None,

        "error_type":
            None,

        "error_message":
            None,
    }

    if (
        readiness_status
        == "upstream_validation_failed"
    ):
        record[
            "population_status"
        ] = (
            "completed"
        )

        record[
            "workflow_outcome"
        ] = (
            "upstream_validation_failed"
        )

        record[
            "generation_method"
        ] = classification.get(
            "generation_method"
        )

        record[
            "validation_error"
        ] = classification.get(
            "validation_error"
        )

        record[
            "completed_at"
        ] = utc_now_iso()

        write_json(
            population_run_path(
                out
            ),
            record,
        )

        return record

    if (
        readiness_status
        != "workflow_ready"
    ):
        record[
            "population_status"
        ] = (
            "input_error"
        )

        record[
            "error_type"
        ] = (
            "population_input_error"
        )

        record[
            "error_message"
        ] = (
            classification[
                "readiness_reason"
            ]
        )

        record[
            "completed_at"
        ] = utc_now_iso()

        write_json(
            population_run_path(
                out
            ),
            record,
        )

        return record

    paths = source_artifact_paths(
        run_dir
    )

    try:
        captured_stdout = (
            io.StringIO()
        )

        with contextlib.redirect_stdout(
            captured_stdout
        ):
            workflow.run_workflow(
                narrative_path=
                    paths[
                        "narrative"
                    ],

                audit_path=
                    paths[
                        "audit"
                    ],

                repair_plan_path=
                    paths[
                        "repair_plan"
                    ],

                selected_claims_path=
                    paths[
                        "selected_claims"
                    ],

                output_dir=
                    out
                    / "workflow",
            )

        workflow_json = (
            out
            / "workflow"
            / "fred_repair_workflow.json"
        )

        workflow_payload = (
            load_json(
                workflow_json
            )
        )

        if not isinstance(
            workflow_payload,
            dict,
        ):
            raise ValueError(
                "Workflow artifact must "
                "be an object."
            )

        record[
            "population_status"
        ] = (
            "completed"
        )

        record[
            "workflow_outcome"
        ] = workflow_payload.get(
            "workflow_outcome"
        )

        record[
            "workflow_json"
        ] = str(
            workflow_json
        )

        record[
            "strategy_selected"
        ] = workflow_payload.get(
            "strategy_selected"
        )

        record[
            "repair_applied"
        ] = workflow_payload.get(
            "repair_applied"
        )

        record[
            "audit_pass_before"
        ] = workflow_payload.get(
            "audit_pass_before"
        )

        record[
            "audit_pass_after"
        ] = workflow_payload.get(
            "audit_pass_after"
        )

        record[
            "targeted_repair_success"
        ] = workflow_payload.get(
            "targeted_repair_success"
        )

        record[
            "full_audit_success"
        ] = workflow_payload.get(
            "full_audit_success"
        )

        record[
            "new_errors_introduced"
        ] = workflow_payload.get(
            "new_errors_introduced"
        )

    except Exception as exc:
        record[
            "population_status"
        ] = (
            "workflow_failed"
        )

        record[
            "error_type"
        ] = type(
            exc
        ).__name__

        record[
            "error_message"
        ] = str(
            exc
        )

        record[
            "traceback"
        ] = traceback.format_exc()

    record[
        "completed_at"
    ] = utc_now_iso()

    write_json(
        population_run_path(
            out
        ),
        record,
    )

    return record


def summarize_records(
    records: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    """Build compact population summary counts."""
    readiness = Counter(
        record.get(
            "readiness_status"
        )
        for record in records
    )

    population_status = Counter(
        record.get(
            "population_status"
        )
        for record in records
    )

    workflow_outcomes = Counter(
        record.get(
            "workflow_outcome"
        )
        for record in records
        if record.get(
            "workflow_outcome"
        )
        is not None
    )

    strategies = Counter(
        record.get(
            "strategy_selected"
        )
        for record in records
        if record.get(
            "strategy_selected"
        )
        is not None
    )

    return {
        "n_runs":
            len(
                records
            ),

        "readiness_counts":
            dict(
                sorted(
                    readiness.items()
                )
            ),

        "population_status_counts":
            dict(
                sorted(
                    population_status.items()
                )
            ),

        "workflow_outcome_counts":
            dict(
                sorted(
                    workflow_outcomes.items()
                )
            ),

        "strategy_selected_counts":
            dict(
                sorted(
                    strategies.items()
                )
            ),

        "n_repair_applied":
            sum(
                record.get(
                    "repair_applied"
                )
                is True
                for record in records
            ),

        "n_targeted_repair_success":
            sum(
                record.get(
                    "targeted_repair_success"
                )
                is True
                for record in records
            ),

        "n_full_audit_success":
            sum(
                record.get(
                    "full_audit_success"
                )
                is True
                for record in records
            ),

        "n_new_errors_introduced":
            sum(
                record.get(
                    "new_errors_introduced"
                )
                is True
                for record in records
            ),
    }


def run_population(
    *,
    input_base: Path,
    family_name: str,
    output_dir: Path,
    limit: int | None,
    resume: bool,
) -> None:
    """Run ACT across the discovered frozen family."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    started_at = (
        utc_now_iso()
    )

    batch_dirs = (
        discover_batch_dirs(
            input_base=
                input_base,
            family_name=
                family_name,
        )
    )

    discovered = (
        discover_run_dirs(
            batch_dirs
        )
    )

    if limit is not None:
        discovered = (
            discovered[
                :limit
            ]
        )

    records: list[
        dict[str, Any]
    ] = []

    n_total = len(
        discovered
    )

    for index, (
        batch_dir,
        run_dir,
    ) in enumerate(
        discovered,
        start=1,
    ):
        out = output_run_dir(
            population_output_dir=
                output_dir,
            batch_dir=
                batch_dir,
            run_dir=
                run_dir,
        )

        existing_path = (
            population_run_path(
                out
            )
        )

        if (
            resume
            and existing_path.exists()
        ):
            existing = load_json(
                existing_path
            )

            if not isinstance(
                existing,
                dict,
            ):
                raise ValueError(
                    "Existing population-run "
                    "artifact must be an object: "
                    f"{existing_path}"
                )

            records.append(
                existing
            )

            print(
                f"[{index}/{n_total}] "
                "resume "
                f"{batch_dir.name}/"
                f"{run_dir.parent.name}/"
                f"{run_dir.name}"
            )

            continue

        record = process_run(
            batch_dir=
                batch_dir,
            run_dir=
                run_dir,
            population_output_dir=
                output_dir,
        )

        records.append(
            record
        )

        print(
            f"[{index}/{n_total}] "
            f"{record['readiness_status']} "
            "→ "
            f"{record['population_status']} "
            f"{batch_dir.name}/"
            f"{run_dir.parent.name}/"
            f"{run_dir.name}"
        )

    manifest_path = (
        output_dir
        / "fred_repair_population.json"
    )

    manifest = {
        "population_schema_version":
            POPULATION_SCHEMA_VERSION,

        "population_method":
            POPULATION_METHOD,

        "started_at":
            started_at,

        "completed_at":
            utc_now_iso(),

        "input_base":
            str(
                input_base
            ),

        "family_name":
            family_name,

        "n_batch_dirs":
            len(
                batch_dirs
            ),

        "n_discovered_runs":
            len(
                discover_run_dirs(
                    batch_dirs
                )
            ),

        "n_processed_runs":
            len(
                records
            ),

        "limit":
            limit,

        "resume":
            resume,

        "batch_dirs": [
            str(
                path
            )
            for path in batch_dirs
        ],

        "summary":
            summarize_records(
                records
            ),

        "outputs": {
            "population_manifest":
                str(
                    manifest_path
                ),
        },
    }

    write_json(
        manifest_path,
        manifest,
    )

    print()
    print(
        "Wrote FRED repair population artifact:"
    )
    print(
        f"  {manifest_path}"
    )

    print()
    print(
        "=== POPULATION SUMMARY ==="
    )

    for key, value in (
        manifest[
            "summary"
        ].items()
    ):
        print(
            f"{key}:",
            value,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic FRED "
            "repair workflows across "
            "a frozen experiment family."
        )
    )

    parser.add_argument(
        "--input-base",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--family-name",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N "
            "discovered runs."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse existing per-run "
            "population artifacts."
        ),
    )

    args = (
        parser.parse_args()
    )

    if (
        args.limit is not None
        and args.limit < 1
    ):
        parser.error(
            "--limit must be >= 1."
        )

    return args


def main() -> None:
    args = parse_args()

    run_population(
        input_base=
            args.input_base,
        family_name=
            args.family_name,
        output_dir=
            args.output_dir,
        limit=
            args.limit,
        resume=
            args.resume,
    )


if __name__ == "__main__":
    main()
