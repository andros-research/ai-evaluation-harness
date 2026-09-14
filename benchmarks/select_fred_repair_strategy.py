#!/usr/bin/env python3
"""
Select a deterministic FRED narrative repair strategy.

The selector does not execute or persist a repair.

Instead, it probes the existing deterministic repair executors in memory
and records which strategies are eligible for the supplied narrative,
audit, repair plan, and selected claims.

Selection policy:

    exactly one eligible strategy
        -> select it

    zero eligible strategies
        -> abstain

    multiple eligible strategies
        -> abstain as ambiguous

No precedence ordering between repair strategies is encoded.

The selector adapts historical executor interfaces without requiring the
executors themselves to expose identical APIs.
"""

from __future__ import annotations

import argparse
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import apply_fred_narrative_repair as normalize
import consolidate_fred_claim_representations as consolidate
import relocate_fred_claim_citations as relocate


SELECTION_SCHEMA_VERSION = (
    "fred_repair_selection_v0_1"
)

SELECTION_METHOD = (
    "deterministic_executor_eligibility"
)

STRATEGY_MODULES = (
    normalize,
    relocate,
    consolidate,
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


def probe_strategy(
    *,
    module: Any,
    narrative_text: str,
    audit_payload: dict[str, Any],
    repair_plan: dict[str, Any],
    selected_claims: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Probe one repair strategy without writing repair artifacts.

    Candidate-plan filtering occurs before apply_repairs() so a strategy
    with no supported planned actions is immediately classified as
    ineligible.

    Historical two-value and newer three-value apply_repairs() return
    shapes are normalized here rather than changing proven executors.
    """
    candidate_actions = (
        module.candidate_repair_actions(
            repair_plan
        )
    )

    strategy = module.REPAIR_STRATEGY

    if not candidate_actions:
        return {
            "repair_strategy":
                strategy,
            "eligible":
                False,
            "n_candidate_repair_actions":
                0,
            "n_actions_applied":
                0,
            "n_actions_abstained":
                0,
            "probe_reason":
                "no_supported_candidate_actions",
            "abstention_reasons":
                [],
        }

    kwargs: dict[str, Any] = {
        "narrative_text":
            narrative_text,
        "audit_payload":
            audit_payload,
        "repair_plan":
            repair_plan,
    }

    signature = inspect.signature(
        module.apply_repairs
    )

    if (
        "selected_claims"
        in signature.parameters
    ):
        kwargs[
            "selected_claims"
        ] = selected_claims

    result = module.apply_repairs(
        **kwargs
    )

    if len(result) == 2:
        (
            _repaired_text,
            applied_actions,
        ) = result

        abstained_actions: list[
            dict[str, Any]
        ] = []

        executor_return_shape = (
            "repaired_text_applied_actions"
        )

    elif len(result) == 3:
        (
            _repaired_text,
            applied_actions,
            abstained_actions,
        ) = result

        executor_return_shape = (
            "repaired_text_applied_actions_"
            "abstained_actions"
        )

    else:
        raise ValueError(
            "Unsupported apply_repairs return "
            f"shape for {strategy}: "
            f"{len(result)} values"
        )

    eligible = bool(
        applied_actions
    )

    abstention_reasons = sorted(
        {
            item.get("reason")
            for item in abstained_actions
            if item.get("reason")
        }
    )

    return {
        "repair_strategy":
            strategy,
        "eligible":
            eligible,
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
        "probe_reason":
            (
                "repair_actions_applicable"
                if eligible
                else
                "executor_abstained"
            ),
        "abstention_reasons":
            abstention_reasons,
        "executor_return_shape":
            executor_return_shape,
    }


def select_strategy(
    *,
    narrative_text: str,
    audit_payload: dict[str, Any],
    repair_plan: dict[str, Any],
    selected_claims: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Probe all registered strategies and select only when exactly one
    strategy is eligible.
    """
    probes = [
        probe_strategy(
            module=module,
            narrative_text=narrative_text,
            audit_payload=audit_payload,
            repair_plan=repair_plan,
            selected_claims=selected_claims,
        )
        for module in STRATEGY_MODULES
    ]

    eligible_strategies = [
        probe["repair_strategy"]
        for probe in probes
        if probe["eligible"]
    ]

    if len(
        eligible_strategies
    ) == 1:
        selection_status = (
            "selected"
        )

        strategy_selected = (
            eligible_strategies[0]
        )

        selection_reason = (
            "exactly_one_eligible_strategy"
        )

        ambiguous = False

    elif not eligible_strategies:
        selection_status = (
            "abstained"
        )

        strategy_selected = None

        selection_reason = (
            "no_supported_deterministic_repair"
        )

        ambiguous = False

    else:
        selection_status = (
            "abstained"
        )

        strategy_selected = None

        selection_reason = (
            "multiple_eligible_strategies"
        )

        ambiguous = True

    return {
        "selection_status":
            selection_status,
        "strategy_selected":
            strategy_selected,
        "selection_reason":
            selection_reason,
        "eligible_strategies":
            eligible_strategies,
        "ambiguous":
            ambiguous,
        "strategy_probes":
            probes,
    }


def write_selection(
    *,
    narrative_path: Path,
    audit_path: Path,
    repair_plan_path: Path,
    selected_claims_path: Path,
    output_dir: Path,
) -> None:
    """Select a repair strategy and write a durable selection artifact."""
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

    selected_claims_payload = load_json(
        selected_claims_path
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
        relocate.selected_claim_records(
            selected_claims_payload
        )
    )

    selection = select_strategy(
        narrative_text=narrative_text,
        audit_payload=audit_payload,
        repair_plan=repair_plan,
        selected_claims=selected_claims,
    )

    output_path = (
        output_dir
        / "fred_repair_selection.json"
    )

    artifact = {
        "repair_selection_schema_version":
            SELECTION_SCHEMA_VERSION,
        "selection_method":
            SELECTION_METHOD,
        "selected_at":
            utc_now_iso(),
        **selection,
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
            "repair_selection_json":
                str(
                    output_path
                ),
        },
    }

    write_json(
        output_path,
        artifact,
    )

    print(
        "Wrote FRED repair selection artifact:"
    )
    print(
        f"  {output_path}"
    )

    print(
        "selection_status="
        f"{artifact['selection_status']}"
    )

    print(
        "strategy_selected="
        f"{artifact['strategy_selected']}"
    )

    print(
        "selection_reason="
        f"{artifact['selection_reason']}"
    )

    print(
        "eligible_strategies="
        f"{artifact['eligible_strategies']}"
    )

    print(
        "ambiguous="
        f"{artifact['ambiguous']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a deterministic FRED "
            "narrative repair strategy."
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

    write_selection(
        narrative_path=args.narrative,
        audit_path=args.audit,
        repair_plan_path=args.repair_plan,
        selected_claims_path=args.selected_claims,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
