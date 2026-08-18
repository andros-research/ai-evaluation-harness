#!/usr/bin/env python3
"""
Build empirical model profiles from completed compatible FRED
model-comparison batches.

v0.1 intentionally reports measurements only:
- population counts
- process completion
- audit pass
- acceptance
- repair frequency
- elapsed-time summaries
- model × prompt × temperature condition metrics

No behavioral labels or interpretive classifications are assigned.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from benchmarks.fred_model_comparison_utils import (  # noqa: E402
    collect_completed_compatible_population,
    find_model_comparison_dirs,
)


PROFILE_SCHEMA_VERSION = "fred_model_profiles_v0_1"

DEFAULT_RESULTS_ROOT = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "model_comparisons"
)

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "model_profiles"
    / "fred_model_profiles.json"
)

PROMPT_ORDER = {
    "weak": 0,
    "intermediate": 1,
    "hardened": 2,
}


def true_count(
    rows: list[dict[str, Any]],
    field: str,
) -> int:
    """Count rows whose field is explicitly True."""
    return sum(
        row.get(field) is True
        for row in rows
    )


def evaluated_count(
    rows: list[dict[str, Any]],
    field: str,
) -> int:
    """Count rows where a nullable stage outcome was evaluated."""
    return sum(
        row.get(field) is not None
        for row in rows
    )


def safe_rate(
    numerator: int,
    denominator: int,
) -> float | None:
    """Return a rate, preserving undefined zero-denominator cases."""
    if denominator == 0:
        return None

    return numerator / denominator


def elapsed_values(
    rows: list[dict[str, Any]],
) -> list[float]:
    """Return valid numeric elapsed-second observations."""
    values = []

    for row in rows:
        value = row.get(
            "elapsed_seconds"
        )

        if (
            isinstance(
                value,
                (int, float),
            )
            and not isinstance(
                value,
                bool,
            )
        ):
            values.append(
                float(value)
            )

    return values


def summarize_rows(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Calculate denominator-aware outcome metrics.

    Denominators:
    - process_completion_rate: all attempted rows
    - audit_pass_rate: rows where audit_pass is not null
    - acceptance_rate: all attempted rows
    - repair_rate: rows where repair_needed is not null
    """
    n_attempted = len(rows)

    n_process_ok = true_count(
        rows,
        "process_ok",
    )

    n_audit_evaluated = (
        evaluated_count(
            rows,
            "audit_pass",
        )
    )

    n_audit_pass = true_count(
        rows,
        "audit_pass",
    )

    n_repair_evaluated = (
        evaluated_count(
            rows,
            "repair_needed",
        )
    )

    n_repair_needed = (
        true_count(
            rows,
            "repair_needed",
        )
    )

    n_accepted = true_count(
        rows,
        "accepted_output",
    )

    elapsed = elapsed_values(
        rows
    )

    return {
        "population": {
            "n_attempted": n_attempted,
            "n_process_ok": (
                n_process_ok
            ),
            "n_audit_evaluated": (
                n_audit_evaluated
            ),
            "n_audit_pass": (
                n_audit_pass
            ),
            "n_repair_evaluated": (
                n_repair_evaluated
            ),
            "n_repair_needed": (
                n_repair_needed
            ),
            "n_accepted": n_accepted,
        },
        "rates": {
            "process_completion_rate": (
                safe_rate(
                    n_process_ok,
                    n_attempted,
                )
            ),
            "audit_pass_rate": (
                safe_rate(
                    n_audit_pass,
                    n_audit_evaluated,
                )
            ),
            "acceptance_rate": (
                safe_rate(
                    n_accepted,
                    n_attempted,
                )
            ),
            "repair_rate": (
                safe_rate(
                    n_repair_needed,
                    n_repair_evaluated,
                )
            ),
        },
        "elapsed_seconds": {
            "n_observed": len(
                elapsed
            ),
            "mean": (
                statistics.mean(
                    elapsed
                )
                if elapsed
                else None
            ),
            "median": (
                statistics.median(
                    elapsed
                )
                if elapsed
                else None
            ),
        },
    }


def build_model_profile(
    *,
    model: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build measured overall and condition-level metrics for one model."""
    model_rows = [
        row
        for row in rows
        if row.get("model") == model
    ]

    prompt_variants = sorted(
        {
            str(
                row.get(
                    "prompt_variant"
                )
            )
            for row in model_rows
        },
        key=lambda value: (
            PROMPT_ORDER.get(
                value,
                999,
            ),
            value,
        ),
    )

    conditions = {}

    for prompt_variant in prompt_variants:
        prompt_rows = [
            row
            for row in model_rows
            if (
                row.get(
                    "prompt_variant"
                )
                == prompt_variant
            )
        ]

        temperatures = sorted(
            {
                float(
                    row.get(
                        "temperature"
                    )
                )
                for row in prompt_rows
                if (
                    row.get(
                        "temperature"
                    )
                    is not None
                )
            }
        )

        prompt_conditions = {}

        for temperature in temperatures:
            condition_rows = [
                row
                for row in prompt_rows
                if (
                    row.get(
                        "temperature"
                    )
                    is not None
                    and float(
                        row.get(
                            "temperature"
                        )
                    )
                    == temperature
                )
            ]

            prompt_conditions[
                str(temperature)
            ] = summarize_rows(
                condition_rows
            )

        conditions[
            prompt_variant
        ] = prompt_conditions

    overall = summarize_rows(
        model_rows
    )

    return {
        "population": overall[
            "population"
        ],
        "overall": overall[
            "rates"
        ],
        "elapsed_seconds": overall[
            "elapsed_seconds"
        ],
        "conditions": conditions,
    }


def resolve_selected_comparison(
    *,
    value: str,
    results_root: Path,
) -> Path:
    """
    Resolve a selected comparison supplied either as a path
    or as a directory name under the results root.
    """
    candidate = Path(
        value
    )

    if candidate.exists():
        return candidate.resolve()

    candidate = (
        results_root
        / value
    )

    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(
        "Selected comparison not found: "
        f"{value}"
    )


def build_profiles(
    *,
    selected_comparison: Path,
    results_root: Path,
) -> dict[str, Any]:
    """Build the complete empirical profile artifact."""
    comparison_dirs = (
        find_model_comparison_dirs(
            results_root
        )
    )

    population = (
        collect_completed_compatible_population(
            selected_comparison=(
                selected_comparison
            ),
            comparison_dirs=(
                comparison_dirs
            ),
        )
    )

    rows = population[
        "rows"
    ]

    included_batches = population[
        "included_batches"
    ]

    excluded_batches = population[
        "excluded_batches"
    ]

    reference_id = population[
        "reference_comparison_id"
    ]

    included_ids = {
        batch[
            "comparison_id"
        ]
        for batch in included_batches
    }

    if reference_id not in included_ids:
        raise ValueError(
            "Selected comparison must itself "
            "be a completed compatible batch "
            "with normalized rows."
        )

    if not rows:
        raise ValueError(
            "No completed compatible "
            "population rows found."
        )

    models = sorted(
        {
            str(
                row.get("model")
            )
            for row in rows
        }
    )

    model_profiles = {
        model: build_model_profile(
            model=model,
            rows=rows,
        )
        for model in models
    }

    overall = summarize_rows(
        rows
    )

    return {
        "schema_version": (
            PROFILE_SCHEMA_VERSION
        ),
        "population": {
            "reference_comparison_id": (
                reference_id
            ),
            "comparison_family_id": (
                population[
                    "comparison_family_id"
                ]
            ),
            "context_sha256": (
                population[
                    "context_sha256"
                ]
            ),
            "comparison_window": (
                population[
                    "comparison_window"
                ]
            ),
            "n_batches": len(
                included_batches
            ),
            "n_attempted": len(
                rows
            ),
            "included_batches": (
                included_batches
            ),
            "excluded_batches": (
                excluded_batches
            ),
        },
        "overall": overall,
        "models": model_profiles,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build empirical model profiles "
            "from completed compatible FRED "
            "model-comparison batches."
        )
    )

    parser.add_argument(
        "--selected-comparison",
        required=True,
        help=(
            "Reference comparison directory "
            "name or path."
        ),
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=(
            "Root containing model-comparison "
            "experiment directories."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Destination JSON profile artifact."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    results_root = (
        args.results_root.resolve()
    )

    selected_comparison = (
        resolve_selected_comparison(
            value=(
                args.selected_comparison
            ),
            results_root=(
                results_root
            ),
        )
    )

    artifact = build_profiles(
        selected_comparison=(
            selected_comparison
        ),
        results_root=results_root,
    )

    output = args.output.resolve()

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            artifact,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "schema:",
        artifact[
            "schema_version"
        ],
    )

    print(
        "family:",
        artifact[
            "population"
        ][
            "comparison_family_id"
        ],
    )

    print(
        "batches:",
        artifact[
            "population"
        ][
            "n_batches"
        ],
    )

    print(
        "attempted:",
        artifact[
            "population"
        ][
            "n_attempted"
        ],
    )

    print(
        "models:",
        ", ".join(
            artifact[
                "models"
            ].keys()
        ),
    )

    print(
        "output:",
        output,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
