#!/usr/bin/env python3
"""
Shared helpers for FRED model-comparison experiment identity.

These functions define experiment-family, batch, expected-run, and
experimental-design semantics shared by downstream analytical tooling.

Keep this module free of Streamlit and pandas dependencies so it can be
used by both dashboard and command-line analysis code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def infer_comparison_family_id(
    *,
    manifest: dict[str, Any],
    comparison_dir: Path,
) -> str:
    """Return the comparison family ID, including legacy batch support."""
    family_id = manifest.get(
        "comparison_family_id"
    )

    if family_id:
        return str(family_id)

    comparison_id = str(
        manifest.get(
            "comparison_id",
            comparison_dir.name,
        )
    )

    if "__batch_" in comparison_id:
        return comparison_id.split(
            "__batch_",
            1,
        )[0]

    return comparison_id


def infer_batch_number(
    *,
    manifest: dict[str, Any],
    comparison_dir: Path,
) -> int:
    """Return the batch number, including legacy batch support."""
    value = manifest.get("batch_number")

    if value is not None:
        try:
            return int(value)
        except Exception:
            pass

    name = comparison_dir.name

    if "__batch_" in name:
        try:
            return int(
                name.rsplit(
                    "__batch_",
                    1,
                )[1]
            )
        except Exception:
            pass

    return 1


def expected_runs_from_config(
    config: dict[str, Any],
) -> int:
    """Return the total configured repetitions across run definitions."""
    total = 0

    for run in config.get(
        "runs",
        [],
    ):
        try:
            total += int(
                run.get(
                    "repetitions",
                    1,
                )
            )
        except Exception:
            continue

    return total


def comparison_design_signature(
    config: dict[str, Any],
) -> tuple:
    """
    Return the experimental design components that must match before
    batches can be pooled.
    """
    design_rows = []

    for run in config.get(
        "runs",
        [],
    ):
        mode = run.get("mode")

        temperature = (
            run.get("temperature")
            if mode == "llm"
            else None
        )

        if temperature is not None:
            try:
                temperature = float(
                    temperature
                )
            except Exception:
                temperature = str(
                    temperature
                )

        design_rows.append(
            (
                str(
                    run.get(
                        "label",
                        "",
                    )
                ),
                str(
                    mode
                    or ""
                ),
                run.get("model"),
                run.get(
                    "prompt_variant"
                ),
                temperature,
                int(
                    run.get(
                        "repetitions",
                        1,
                    )
                ),
            )
        )

    return tuple(
        sorted(
            design_rows,
            key=lambda row: row[0],
        )
    )
