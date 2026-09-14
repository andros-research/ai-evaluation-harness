#!/usr/bin/env python3
"""
Run the deterministic FRED narrative repair workflow.

Workflow:

    existing narrative + audit + repair plan + selected claims
        ->
    deterministic repair selection
        ->
    selected repair execution OR clean abstention
        ->
    independent re-audit
        ->
    repair outcome evaluation
        ->
    durable workflow artifact

The workflow contains no repair-specific decision logic beyond a small
capability registry. Repair selection remains the responsibility of
select_fred_repair_strategy.py.

A completed workflow does not imply that a repair was performed or that
the narrative passed audit. Clean abstention is a valid workflow outcome.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import apply_fred_narrative_repair as normalize
import audit_fred_narrative as auditor
import consolidate_fred_claim_representations as consolidate
import evaluate_fred_narrative_repair as evaluator
import relocate_fred_claim_citations as relocate
import select_fred_repair_strategy as selector


WORKFLOW_SCHEMA_VERSION = (
    "fred_repair_workflow_v0_1"
)

WORKFLOW_METHOD = (
    "deterministic_select_execute_reaudit_evaluate"
)


STRATEGY_REGISTRY = {
    normalize.REPAIR_STRATEGY: {
        "module":
            normalize,
        "requires_selected_claims":
            False,
        "target_error":
            "bullets_missing_claim_citations",
    },

    relocate.REPAIR_STRATEGY: {
        "module":
            relocate,
        "requires_selected_claims":
            True,
        "target_error":
            "bullets_missing_claim_citations",
    },

    consolidate.REPAIR_STRATEGY: {
        "module":
            consolidate,
        "requires_selected_claims":
            True,
        "target_error":
            "bullets_missing_claim_citations",
    },
}


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
    path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def workflow_paths(
    output_dir: Path,
) -> dict[str, Path]:
    """Return standard workflow artifact paths."""
    return {
        "selection_dir":
            output_dir
            / "selection",

        "selection_json":
            output_dir
            / "selection"
            / "fred_repair_selection.json",

        "repair_dir":
            output_dir
            / "repair",

        "repaired_narrative":
            output_dir
            / "repair"
            / "fred_narrative_repaired.md",

        "repair_execution_json":
            output_dir
            / "repair"
            / "fred_repair_execution.json",

        "audit_after_dir":
            output_dir
            / "audit_after",

        "audit_after_json":
            output_dir
            / "audit_after"
            / "fred_narrative_audit.json",

        "evaluation_dir":
            output_dir
            / "evaluation",

        "repair_result_json":
            output_dir
            / "evaluation"
            / "fred_repair_result.json",

        "workflow_json":
            output_dir
            / "fred_repair_workflow.json",
    }


def execute_selected_strategy(
    *,
    strategy: str,
    narrative_path: Path,
    audit_path: Path,
    repair_plan_path: Path,
    selected_claims_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Execute the selected strategy using capability metadata.

    The workflow does not branch on individual strategy names.
    """
    capability = (
        STRATEGY_REGISTRY.get(
            strategy
        )
    )

    if capability is None:
        raise ValueError(
            "Selected repair strategy is not "
            "registered with the workflow: "
            f"{strategy}"
        )

    module = capability[
        "module"
    ]

    kwargs: dict[str, Any] = {
        "narrative_path":
            narrative_path,
        "audit_path":
            audit_path,
        "repair_plan_path":
            repair_plan_path,
        "output_dir":
            output_dir,
    }

    if capability[
        "requires_selected_claims"
    ]:
        kwargs[
            "selected_claims_path"
        ] = selected_claims_path

    module.write_repair_execution(
        **kwargs
    )

    return {
        "target_error":
            capability[
                "target_error"
            ],
        "requires_selected_claims":
            capability[
                "requires_selected_claims"
            ],
    }


def run_workflow(
    *,
    narrative_path: Path,
    audit_path: Path,
    repair_plan_path: Path,
    selected_claims_path: Path,
    output_dir: Path,
) -> None:
    """Run the complete deterministic repair workflow."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = workflow_paths(
        output_dir
    )

    started_at = utc_now_iso()

    before_audit = load_json(
        audit_path
    )

    if not isinstance(
        before_audit,
        dict,
    ):
        raise ValueError(
            "Before-audit payload must be an object."
        )

    strict_coverage = (
        before_audit.get(
            "strict_selected_claim_coverage"
        )
    )

    if not isinstance(
        strict_coverage,
        bool,
    ):
        raise ValueError(
            "Before-audit artifact must record "
            "boolean "
            "strict_selected_claim_coverage."
        )

    selector.write_selection(
        narrative_path=narrative_path,
        audit_path=audit_path,
        repair_plan_path=repair_plan_path,
        selected_claims_path=
            selected_claims_path,
        output_dir=
            paths[
                "selection_dir"
            ],
    )

    selection = load_json(
        paths[
            "selection_json"
        ]
    )

    if not isinstance(
        selection,
        dict,
    ):
        raise ValueError(
            "Repair selection artifact must "
            "be an object."
        )

    selection_status = (
        selection.get(
            "selection_status"
        )
    )

    strategy_selected = (
        selection.get(
            "strategy_selected"
        )
    )

    selection_reason = (
        selection.get(
            "selection_reason"
        )
    )

    ambiguous = bool(
        selection.get(
            "ambiguous"
        )
    )

    audit_pass_before = (
        before_audit.get(
            "audit_pass"
        )
    )

    if selection_status != "selected":
        if audit_pass_before is True:
            workflow_outcome = (
                "no_repair_needed"
            )

        elif ambiguous:
            workflow_outcome = (
                "ambiguous_repair_selection"
            )

        else:
            workflow_outcome = (
                "no_supported_deterministic_repair"
            )

        workflow = {
            "workflow_schema_version":
                WORKFLOW_SCHEMA_VERSION,

            "workflow_method":
                WORKFLOW_METHOD,

            "workflow_status":
                "completed",

            "workflow_outcome":
                workflow_outcome,

            "started_at":
                started_at,

            "completed_at":
                utc_now_iso(),

            "selection_status":
                selection_status,

            "strategy_selected":
                strategy_selected,

            "selection_reason":
                selection_reason,

            "eligible_strategies":
                selection.get(
                    "eligible_strategies",
                    [],
                ),

            "ambiguous":
                ambiguous,

            "repair_applied":
                False,

            "audit_pass_before":
                audit_pass_before,

            "audit_pass_after":
                None,

            "target_error":
                None,

            "targeted_repair_success":
                None,

            "full_audit_success":
                None,

            "strict_selected_claim_coverage":
                strict_coverage,

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
            },

            "artifacts": {
                "selection_json":
                    str(
                        paths[
                            "selection_json"
                        ]
                    ),

                "repair_execution_json":
                    None,

                "repaired_narrative":
                    None,

                "after_audit_json":
                    None,

                "repair_result_json":
                    None,
            },
        }

        write_json(
            paths[
                "workflow_json"
            ],
            workflow,
        )

        print(
            "Wrote FRED repair workflow artifact:"
        )
        print(
            f"  {paths['workflow_json']}"
        )
        print(
            "workflow_status=completed"
        )
        print(
            "workflow_outcome="
            f"{workflow_outcome}"
        )
        print(
            "repair_applied=False"
        )

        return

    if not isinstance(
        strategy_selected,
        str,
    ):
        raise ValueError(
            "Selected workflow must record "
            "a strategy_selected value."
        )

    capability = (
        execute_selected_strategy(
            strategy=strategy_selected,
            narrative_path=narrative_path,
            audit_path=audit_path,
            repair_plan_path=
                repair_plan_path,
            selected_claims_path=
                selected_claims_path,
            output_dir=
                paths[
                    "repair_dir"
                ],
        )
    )

    repair_execution = load_json(
        paths[
            "repair_execution_json"
        ]
    )

    if not isinstance(
        repair_execution,
        dict,
    ):
        raise ValueError(
            "Repair execution artifact must "
            "be an object."
        )

    if (
        repair_execution.get(
            "repair_applied"
        )
        is not True
    ):
        raise ValueError(
            "Selector chose an eligible repair "
            "strategy, but execution did not "
            "apply a repair."
        )

    auditor.write_audit_artifact(
        narrative_path=
            paths[
                "repaired_narrative"
            ],
        selected_claims_path=
            selected_claims_path,
        output_dir=
            paths[
                "audit_after_dir"
            ],
        strict_selected_claim_coverage=
            strict_coverage,
    )

    evaluator.write_repair_result(
        before_audit_path=
            audit_path,
        repair_execution_path=
            paths[
                "repair_execution_json"
            ],
        after_audit_path=
            paths[
                "audit_after_json"
            ],
        target_error=
            capability[
                "target_error"
            ],
        output_dir=
            paths[
                "evaluation_dir"
            ],
    )

    after_audit = load_json(
        paths[
            "audit_after_json"
        ]
    )

    repair_result = load_json(
        paths[
            "repair_result_json"
        ]
    )

    if not isinstance(
        after_audit,
        dict,
    ):
        raise ValueError(
            "After-audit artifact must "
            "be an object."
        )

    if not isinstance(
        repair_result,
        dict,
    ):
        raise ValueError(
            "Repair result artifact must "
            "be an object."
        )

    workflow = {
        "workflow_schema_version":
            WORKFLOW_SCHEMA_VERSION,

        "workflow_method":
            WORKFLOW_METHOD,

        "workflow_status":
            "completed",

        "workflow_outcome":
            "repair_executed",

        "started_at":
            started_at,

        "completed_at":
            utc_now_iso(),

        "selection_status":
            selection_status,

        "strategy_selected":
            strategy_selected,

        "selection_reason":
            selection_reason,

        "eligible_strategies":
            selection.get(
                "eligible_strategies",
                [],
            ),

        "ambiguous":
            ambiguous,

        "repair_applied":
            repair_execution.get(
                "repair_applied"
            ),

        "n_actions_applied":
            repair_execution.get(
                "n_actions_applied"
            ),

        "audit_pass_before":
            audit_pass_before,

        "audit_pass_after":
            after_audit.get(
                "audit_pass"
            ),

        "target_error":
            capability[
                "target_error"
            ],

        "target_error_resolved":
            repair_result.get(
                "target_error_resolved"
            ),

        "targeted_repair_success":
            repair_result.get(
                "targeted_repair_success"
            ),

        "full_audit_success":
            repair_result.get(
                "full_audit_success"
            ),

        "new_errors_introduced":
            repair_result.get(
                "new_errors_introduced"
            ),

        "resolved_errors":
            repair_result.get(
                "resolved_errors",
                [],
            ),

        "residual_errors":
            repair_result.get(
                "residual_errors",
                [],
            ),

        "newly_observed_errors":
            repair_result.get(
                "newly_observed_errors",
                [],
            ),

        "unmasked_errors":
            repair_result.get(
                "unmasked_errors",
                [],
            ),

        "introduced_errors":
            repair_result.get(
                "introduced_errors",
                [],
            ),

        "strict_selected_claim_coverage":
            strict_coverage,

        "capability": {
            "repair_strategy":
                strategy_selected,

            "target_error":
                capability[
                    "target_error"
                ],

            "requires_selected_claims":
                capability[
                    "requires_selected_claims"
                ],
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
        },

        "artifacts": {
            "selection_json":
                str(
                    paths[
                        "selection_json"
                    ]
                ),

            "repair_execution_json":
                str(
                    paths[
                        "repair_execution_json"
                    ]
                ),

            "repaired_narrative":
                str(
                    paths[
                        "repaired_narrative"
                    ]
                ),

            "after_audit_json":
                str(
                    paths[
                        "audit_after_json"
                    ]
                ),

            "repair_result_json":
                str(
                    paths[
                        "repair_result_json"
                    ]
                ),
        },
    }

    write_json(
        paths[
            "workflow_json"
        ],
        workflow,
    )

    print(
        "Wrote FRED repair workflow artifact:"
    )
    print(
        f"  {paths['workflow_json']}"
    )

    print(
        "workflow_status=completed"
    )

    print(
        "workflow_outcome=repair_executed"
    )

    print(
        "strategy_selected="
        f"{strategy_selected}"
    )

    print(
        "repair_applied="
        f"{workflow['repair_applied']}"
    )

    print(
        "targeted_repair_success="
        f"{workflow['targeted_repair_success']}"
    )

    print(
        "full_audit_success="
        f"{workflow['full_audit_success']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic FRED "
            "repair workflow."
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

    run_workflow(
        narrative_path=args.narrative,
        audit_path=args.audit,
        repair_plan_path=args.repair_plan,
        selected_claims_path=
            args.selected_claims,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
