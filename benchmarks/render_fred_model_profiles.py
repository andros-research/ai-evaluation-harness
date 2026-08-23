#!/usr/bin/env python3
"""
Render human-readable empirical FRED model profiles.

The canonical empirical measurements live in fred_model_profiles.json.
This renderer does not recompute experiment metrics and does not assign
behavioral personality or role labels.

Its purpose is to translate the measured profile artifact into concise,
evidence-backed Markdown reports.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "model_profiles"
    / "fred_model_profiles.json"
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "model_profiles"
)

PROMPT_ORDER = (
    "weak",
    "intermediate",
    "hardened",
)

TEMPERATURE_ORDER = (
    "0.0",
    "0.7",
)

PROMPT_TRANSITION_ORDER = (
    "weak_to_intermediate",
    "intermediate_to_hardened",
)


def load_json_object(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return payload


def fmt_pct(
    value: float | None,
) -> str:
    if value is None:
        return "—"

    return f"{value * 100.0:.1f}%"


def fmt_pp(
    value: float | None,
) -> str:
    if value is None:
        return "—"

    return f"{value:+.1f} pp"


def fmt_spread_pp(
    value: float | None,
) -> str:
    """Format a non-directional percentage-point spread."""
    if value is None:
        return "—"

    return f"{value:.1f} pp"


def fmt_seconds(
    value: float | None,
) -> str:
    if value is None:
        return "—"

    return f"{value:.2f}s"


def fmt_rate_range(
    *,
    minimum: float | None,
    maximum: float | None,
) -> str:
    if (
        minimum is None
        or maximum is None
    ):
        return "—"

    return (
        f"{minimum * 100.0:.1f}%–"
        f"{maximum * 100.0:.1f}%"
    )


def markdown_table(
    headers: list[str],
    rows: list[list[str]],
) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| "
        + " | ".join(
            "---"
            for _ in headers
        )
        + " |",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(row)
            + " |"
        )

    return lines


def model_slug(
    model: str,
) -> str:
    return re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        model,
    ).strip("_").lower()


def format_success_distribution(
    stability: dict[str, Any],
    metric: str,
) -> str:
    repetitions = stability.get(
        "repetitions_per_batch"
    )

    distribution = stability[
        metric
    ][
        "success_count_distribution"
    ]

    if not distribution:
        return "—"

    pieces = []

    for successes, n_batches in sorted(
        distribution.items(),
        key=lambda item: int(
            item[0]
        ),
    ):
        pieces.append(
            f"{successes}/{repetitions}: "
            f"{n_batches}"
        )

    return ", ".join(pieces)


def render_population_header(
    data: dict[str, Any],
) -> list[str]:
    population = data[
        "population"
    ]

    included = population.get(
        "included_batches",
        [],
    )

    batch_numbers = [
        int(
            item["batch_number"]
        )
        for item in included
    ]

    batch_range = "—"

    if batch_numbers:
        batch_range = (
            f"{min(batch_numbers)}–"
            f"{max(batch_numbers)}"
        )

    lines = [
        "## Experiment population",
        "",
        f"- Schema: `{data['schema_version']}`",
        (
            "- Comparison family: "
            f"`{population['comparison_family_id']}`"
        ),
        (
            "- Reference comparison: "
            f"`{population['reference_comparison_id']}`"
        ),
        (
            "- Comparison window: "
            f"`{population['comparison_window']}`"
        ),
        (
            "- Completed compatible batches: "
            f"{population['n_batches']} "
            f"(batch range {batch_range})"
        ),
        (
            "- Attempted runs: "
            f"{population['n_attempted']}"
        ),
        (
            "- Excluded compatible batches: "
            f"{len(population.get('excluded_batches', []))}"
        ),
        "",
        (
            "All statements below are descriptive of this frozen "
            "controlled experiment population. They are not general "
            "claims about model behavior outside this task."
        ),
        "",
    ]

    return lines


def render_model_snapshot(
    model: str,
    profile: dict[str, Any],
) -> list[str]:
    lines = [
        "### Profile snapshot",
        "",
    ]

    for prompt in PROMPT_ORDER:
        t0 = profile[
            "conditions"
        ][prompt]["0.0"]

        t07 = profile[
            "conditions"
        ][prompt]["0.7"]

        lines.append(
            "- "
            f"**{prompt.title()} prompt:** "
            "process completion "
            f"{fmt_pct(t0['rates']['process_completion_rate'])} "
            "at t=0.0 and "
            f"{fmt_pct(t07['rates']['process_completion_rate'])} "
            "at t=0.7; acceptance "
            f"{fmt_pct(t0['rates']['acceptance_rate'])} "
            "and "
            f"{fmt_pct(t07['rates']['acceptance_rate'])}, "
            "respectively."
        )

    intermediate_effect = profile[
        "temperature_effects"
    ][
        "intermediate"
    ][
        "deltas_pp"
    ]

    lines.extend(
        [
            "",
            (
                "- **Intermediate temperature effect "
                "(t=0.7 minus t=0.0):** "
                "process "
                f"{fmt_pp(intermediate_effect['process_completion_delta_pp'])}; "
                "audit "
                f"{fmt_pp(intermediate_effect['audit_pass_delta_pp'])}; "
                "acceptance "
                f"{fmt_pp(intermediate_effect['acceptance_delta_pp'])}; "
                "repair "
                f"{fmt_pp(intermediate_effect['repair_delta_pp'])}."
            ),
        ]
    )

    stability = profile[
        "conditions"
    ][
        "intermediate"
    ][
        "0.7"
    ][
        "batch_stability"
    ]

    acceptance_stats = stability[
        "acceptance"
    ][
        "rate_distribution"
    ]

    lines.append(
        "- **Intermediate t=0.7 batch repeatability:** "
        "mean acceptance "
        f"{fmt_pct(acceptance_stats['mean'])}; "
        "population SD "
        f"{fmt_spread_pp(acceptance_stats['population_sd'] * 100.0)}; "
        "range "
        f"{fmt_rate_range(minimum=acceptance_stats['min'], maximum=acceptance_stats['max'])}; "
        "accepted-run distribution "
        f"`{format_success_distribution(stability, 'acceptance')}`."
    )

    lines.append("")

    return lines


def render_condition_matrix(
    profile: dict[str, Any],
) -> list[str]:
    rows = []

    for prompt in PROMPT_ORDER:
        for temperature in TEMPERATURE_ORDER:
            condition = profile[
                "conditions"
            ][prompt][temperature]

            pop = condition[
                "population"
            ]

            rates = condition[
                "rates"
            ]

            elapsed = condition[
                "elapsed_seconds"
            ]

            rows.append(
                [
                    prompt,
                    temperature,
                    str(
                        pop[
                            "n_attempted"
                        ]
                    ),
                    fmt_pct(
                        rates[
                            "process_completion_rate"
                        ]
                    ),
                    fmt_pct(
                        rates[
                            "audit_pass_rate"
                        ]
                    ),
                    fmt_pct(
                        rates[
                            "acceptance_rate"
                        ]
                    ),
                    fmt_pct(
                        rates[
                            "repair_rate"
                        ]
                    ),
                    fmt_seconds(
                        elapsed[
                            "mean"
                        ]
                    ),
                ]
            )

    return [
        "### Condition matrix",
        "",
        *markdown_table(
            [
                "Prompt",
                "Temp",
                "N",
                "Process",
                "Audit",
                "Acceptance",
                "Repair",
                "Mean elapsed",
            ],
            rows,
        ),
        "",
    ]


def render_outcome_stages(
    profile: dict[str, Any],
) -> list[str]:
    rows = []

    for prompt in PROMPT_ORDER:
        for temperature in TEMPERATURE_ORDER:
            condition = profile[
                "conditions"
            ][prompt][temperature]

            counts = condition[
                "outcome_stages"
            ][
                "counts"
            ]

            rows.append(
                [
                    prompt,
                    temperature,
                    str(
                        counts[
                            "accepted"
                        ]
                    ),
                    str(
                        counts[
                            "generation_contract_failure"
                        ]
                    ),
                    str(
                        counts[
                            "audit_failure"
                        ]
                    ),
                    str(
                        counts[
                            "other_process_failure"
                        ]
                    ),
                    str(
                        counts[
                            "completed_unaccepted_other"
                        ]
                    ),
                ]
            )

    return [
        "### Outcome-stage counts",
        "",
        *markdown_table(
            [
                "Prompt",
                "Temp",
                "Accepted",
                "Generation failure",
                "Audit failure",
                "Other process",
                "Other completed",
            ],
            rows,
        ),
        "",
    ]


def render_temperature_effects(
    profile: dict[str, Any],
) -> list[str]:
    rows = []

    effects = profile[
        "temperature_effects"
    ]

    for prompt in PROMPT_ORDER:
        deltas = effects[
            prompt
        ][
            "deltas_pp"
        ]

        rows.append(
            [
                prompt,
                fmt_pp(
                    deltas[
                        "process_completion_delta_pp"
                    ]
                ),
                fmt_pp(
                    deltas[
                        "audit_pass_delta_pp"
                    ]
                ),
                fmt_pp(
                    deltas[
                        "acceptance_delta_pp"
                    ]
                ),
                fmt_pp(
                    deltas[
                        "repair_delta_pp"
                    ]
                ),
            ]
        )

    return [
        "### Temperature effects",
        "",
        "Measured as t=0.7 minus t=0.0 at fixed prompt.",
        "",
        *markdown_table(
            [
                "Prompt",
                "Process Δ",
                "Audit Δ",
                "Acceptance Δ",
                "Repair Δ",
            ],
            rows,
        ),
        "",
    ]


def render_prompt_effects(
    profile: dict[str, Any],
) -> list[str]:
    rows = []

    effects = profile[
        "prompt_effects"
    ]

    for temperature in TEMPERATURE_ORDER:
        for transition in PROMPT_TRANSITION_ORDER:
            item = effects[
                temperature
            ][
                transition
            ]

            deltas = item[
                "deltas_pp"
            ]

            label = (
                item[
                    "baseline_prompt"
                ]
                + " → "
                + item[
                    "comparison_prompt"
                ]
            )

            rows.append(
                [
                    temperature,
                    label,
                    fmt_pp(
                        deltas[
                            "process_completion_delta_pp"
                        ]
                    ),
                    fmt_pp(
                        deltas[
                            "audit_pass_delta_pp"
                        ]
                    ),
                    fmt_pp(
                        deltas[
                            "acceptance_delta_pp"
                        ]
                    ),
                    fmt_pp(
                        deltas[
                            "repair_delta_pp"
                        ]
                    ),
                ]
            )

    return [
        "### Prompt-transition effects",
        "",
        (
            "Measured between adjacent prompt-specificity regimes "
            "at fixed temperature."
        ),
        "",
        *markdown_table(
            [
                "Temp",
                "Transition",
                "Process Δ",
                "Audit Δ",
                "Acceptance Δ",
                "Repair Δ",
            ],
            rows,
        ),
        "",
    ]


def render_batch_repeatability(
    profile: dict[str, Any],
) -> list[str]:
    rows = []

    for prompt in PROMPT_ORDER:
        for temperature in TEMPERATURE_ORDER:
            stability = profile[
                "conditions"
            ][prompt][temperature][
                "batch_stability"
            ]

            acceptance_stats = stability[
                "acceptance"
            ][
                "rate_distribution"
            ]

            rows.append(
                [
                    prompt,
                    temperature,
                    format_success_distribution(
                        stability,
                        "process",
                    ),
                    format_success_distribution(
                        stability,
                        "acceptance",
                    ),
                    fmt_spread_pp(
                        acceptance_stats[
                            "population_sd"
                        ]
                        * 100.0
                    ),
                    fmt_rate_range(
                        minimum=acceptance_stats[
                            "min"
                        ],
                        maximum=acceptance_stats[
                            "max"
                        ],
                    ),
                ]
            )

    return [
        "### Batch repeatability",
        "",
        (
            "Success-count distributions show the number of "
            "successful runs out of five within each of the "
            "33 repeated batches."
        ),
        "",
        *markdown_table(
            [
                "Prompt",
                "Temp",
                "Process counts",
                "Acceptance counts",
                "Acceptance SD",
                "Acceptance range",
            ],
            rows,
        ),
        "",
    ]


def render_audit_failures(
    profile: dict[str, Any],
) -> list[str]:
    lines = [
        "### Audit failure details",
        "",
    ]

    found = False

    for prompt in PROMPT_ORDER:
        for temperature in TEMPERATURE_ORDER:
            audit = profile[
                "conditions"
            ][prompt][temperature][
                "audit_error_incidence"
            ]

            n_failures = audit[
                "n_audit_failures"
            ]

            if n_failures == 0:
                continue

            found = True

            lines.append(
                f"**{prompt}, t={temperature}: "
                f"{n_failures} audit failures**"
            )

            lines.append("")

            for error, stats in audit[
                "errors"
            ].items():
                lines.append(
                    "- "
                    f"`{error}`: "
                    f"{stats['count']}/{n_failures} "
                    f"({fmt_pct(stats['incidence_rate'])})"
                )

            lines.append("")

    if not found:
        lines.append(
            "No audit failures were observed."
        )
        lines.append("")

    lines.append(
        (
            "Audit-error incidence is multi-label; percentages "
            "may sum above 100%."
        )
    )
    lines.append("")

    return lines


def render_model_profile(
    *,
    model: str,
    profile: dict[str, Any],
    include_title: bool = True,
) -> list[str]:
    lines = []

    if include_title:
        lines.extend(
            [
                f"# Empirical FRED Model Profile: {model}",
                "",
                (
                    "This profile is a deterministic rendering of "
                    "`fred_model_profiles.json`. No behavioral "
                    "personality or role label is assigned."
                ),
                "",
            ]
        )

    population = profile[
        "population"
    ]

    overall = profile[
        "overall"
    ]

    lines.extend(
        [
            (
                f"Attempted runs: **{population['n_attempted']}**  "
            ),
            (
                f"Process-complete: **{population['n_process_ok']}** "
                f"({fmt_pct(overall['process_completion_rate'])})  "
            ),
            (
                f"Accepted: **{population['n_accepted']}** "
                f"({fmt_pct(overall['acceptance_rate'])})"
            ),
            "",
        ]
    )

    lines.extend(
        render_model_snapshot(
            model,
            profile,
        )
    )

    lines.extend(
        render_condition_matrix(
            profile
        )
    )

    lines.extend(
        render_outcome_stages(
            profile
        )
    )

    lines.extend(
        render_temperature_effects(
            profile
        )
    )

    lines.extend(
        render_prompt_effects(
            profile
        )
    )

    lines.extend(
        render_batch_repeatability(
            profile
        )
    )

    lines.extend(
        render_audit_failures(
            profile
        )
    )

    return lines


def render_combined_report(
    data: dict[str, Any],
) -> str:
    lines = [
        "# FRED Empirical Model Profiles",
        "",
        (
            "Human-readable rendering of the frozen v1.8.2 "
            "model-profile artifact."
        ),
        "",
    ]

    lines.extend(
        render_population_header(
            data
        )
    )

    summary_rows = []

    for model, profile in data[
        "models"
    ].items():
        population = profile[
            "population"
        ]

        overall = profile[
            "overall"
        ]

        summary_rows.append(
            [
                model,
                str(
                    population[
                        "n_attempted"
                    ]
                ),
                fmt_pct(
                    overall[
                        "process_completion_rate"
                    ]
                ),
                fmt_pct(
                    overall[
                        "audit_pass_rate"
                    ]
                ),
                fmt_pct(
                    overall[
                        "acceptance_rate"
                    ]
                ),
                fmt_pct(
                    overall[
                        "repair_rate"
                    ]
                ),
            ]
        )

    lines.extend(
        [
            "## Model-level summary",
            "",
            *markdown_table(
                [
                    "Model",
                    "N",
                    "Process",
                    "Audit",
                    "Acceptance",
                    "Repair",
                ],
                summary_rows,
            ),
            "",
        ]
    )

    for model, profile in data[
        "models"
    ].items():
        lines.extend(
            [
                "---",
                "",
                f"## {model}",
                "",
            ]
        )

        lines.extend(
            render_model_profile(
                model=model,
                profile=profile,
                include_title=False,
            )
        )

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render human-readable Markdown "
            "from empirical FRED model profiles."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Input fred_model_profiles.json artifact."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Destination directory for Markdown reports."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()

    data = load_json_object(
        input_path
    )

    schema_version = str(
        data.get(
            "schema_version",
            "",
        )
    )

    if not schema_version.startswith(
        "fred_model_profiles_"
    ):
        raise ValueError(
            "Unexpected profile schema: "
            f"{schema_version}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_path = (
        output_dir
        / "fred_model_profiles.md"
    )

    combined_path.write_text(
        render_combined_report(
            data
        ),
        encoding="utf-8",
    )

    written = [
        combined_path
    ]

    for model, profile in data[
        "models"
    ].items():
        path = (
            output_dir
            / (
                "fred_model_profile_"
                f"{model_slug(model)}.md"
            )
        )

        lines = render_model_profile(
            model=model,
            profile=profile,
        )

        lines.extend(
            [
                "---",
                "",
                (
                    "Source population: "
                    f"`{data['population']['comparison_family_id']}`, "
                    f"{data['population']['n_batches']} batches, "
                    f"{data['population']['n_attempted']} total runs."
                ),
                "",
            ]
        )

        path.write_text(
            "\n".join(
                lines
            ).rstrip()
            + "\n",
            encoding="utf-8",
        )

        written.append(
            path
        )

    print(
        "schema:",
        schema_version,
    )

    print(
        "models:",
        ", ".join(
            data[
                "models"
            ].keys()
        ),
    )

    print(
        "reports:",
        len(
            written
        ),
    )

    for path in written:
        print(
            "  ",
            path,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
