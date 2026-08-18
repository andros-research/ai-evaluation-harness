#!/usr/bin/env python3
"""
Shared helpers for FRED model-comparison experiment identity.

These functions define experiment-family, batch, expected-run, and
experimental-design semantics shared by downstream analytical tooling.

Keep this module free of Streamlit and pandas dependencies so it can be
used by both dashboard and command-line analysis code.
"""

from __future__ import annotations
import json
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


def find_model_comparison_dirs(
    root: Path,
) -> list[Path]:
    """Return model-comparison experiment directories, newest first."""
    if not root.exists():
        return []

    return sorted(
        [
            path
            for path in root.iterdir()
            if path.is_dir()
            and (
                path
                / "comparison_manifest.json"
            ).exists()
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def read_json_object(
    path: Path,
) -> dict[str, Any]:
    """Read and validate a JSON object."""
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


def read_jsonl_rows(
    path: Path,
) -> list[dict[str, Any]]:
    """Read and validate JSONL object rows."""
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            if not isinstance(row, dict):
                raise ValueError(
                    "Expected JSON object at "
                    f"{path}:{line_number}"
                )

            rows.append(row)

    return rows


def collect_completed_compatible_population(
    *,
    selected_comparison: Path,
    comparison_dirs: list[Path],
) -> dict[str, Any]:
    """
    Collect normalized rows from completed compatible batches.

    Compatibility requires the same:
    - comparison family
    - context hash
    - comparison window
    - experimental design

    Unlike the dashboard's live population collector, this function:
    - accepts completed batches only
    - requires normalized comparison_rows.jsonl
    - requires the normalized row count to equal expected runs
    """
    selected_dir = Path(
        selected_comparison
    )

    selected_manifest = read_json_object(
        selected_dir
        / "comparison_manifest.json"
    )

    selected_config = read_json_object(
        selected_dir
        / "comparison_config.json"
    )

    selected_family = (
        infer_comparison_family_id(
            manifest=selected_manifest,
            comparison_dir=selected_dir,
        )
    )

    selected_context = (
        selected_manifest.get(
            "context_sha256"
        )
    )

    selected_window = (
        selected_manifest.get(
            "comparison_window"
        )
    )

    selected_design = (
        comparison_design_signature(
            selected_config
        )
    )

    population_rows = []
    included_batches = []
    excluded_batches = []

    for comparison_dir in comparison_dirs:
        comparison_dir = Path(
            comparison_dir
        )

        manifest = read_json_object(
            comparison_dir
            / "comparison_manifest.json"
        )

        family_id = (
            infer_comparison_family_id(
                manifest=manifest,
                comparison_dir=comparison_dir,
            )
        )

        # Other experiment families are outside
        # this population rather than exclusions.
        if family_id != selected_family:
            continue

        reasons = []

        if (
            manifest.get(
                "context_sha256"
            )
            != selected_context
        ):
            reasons.append(
                "context_sha256"
            )

        if (
            manifest.get(
                "comparison_window"
            )
            != selected_window
        ):
            reasons.append(
                "comparison_window"
            )

        config_path = (
            comparison_dir
            / "comparison_config.json"
        )

        config = None

        if not config_path.exists():
            reasons.append(
                "missing_comparison_config"
            )
        else:
            config = read_json_object(
                config_path
            )

            if (
                comparison_design_signature(
                    config
                )
                != selected_design
            ):
                reasons.append(
                    "experimental_design"
                )

        if (
            manifest.get("status")
            != "completed"
        ):
            reasons.append(
                "status_not_completed"
            )

        rows_path = (
            comparison_dir
            / "summary"
            / "comparison_rows.jsonl"
        )

        batch_rows = None

        if not rows_path.exists():
            reasons.append(
                "missing_normalized_rows"
            )
        else:
            batch_rows = read_jsonl_rows(
                rows_path
            )

        expected_runs = (
            manifest.get(
                "summary",
                {},
            ).get(
                "n_expected_runs"
            )
        )

        if (
            expected_runs is None
            and config is not None
        ):
            expected_runs = (
                expected_runs_from_config(
                    config
                )
            )

        if expected_runs is not None:
            expected_runs = int(
                expected_runs
            )

        if (
            batch_rows is not None
            and expected_runs is not None
            and len(batch_rows)
            != expected_runs
        ):
            reasons.append(
                "normalized_row_count"
            )

        comparison_id = str(
            manifest.get(
                "comparison_id",
                comparison_dir.name,
            )
        )

        batch_number = (
            infer_batch_number(
                manifest=manifest,
                comparison_dir=comparison_dir,
            )
        )

        batch_info = {
            "comparison_id": comparison_id,
            "batch_number": batch_number,
            "status": manifest.get(
                "status"
            ),
            "expected_runs": expected_runs,
            "normalized_rows": (
                len(batch_rows)
                if batch_rows is not None
                else None
            ),
        }

        if reasons:
            excluded_batches.append(
                {
                    **batch_info,
                    "reasons": reasons,
                }
            )
            continue

        if batch_rows is None:
            continue

        # Backfill provenance for legacy batches
        # that predate family/batch metadata.
        for row in batch_rows:
            normalized_row = dict(row)

            normalized_row[
                "comparison_id"
            ] = comparison_id

            normalized_row[
                "comparison_family_id"
            ] = family_id

            normalized_row[
                "batch_number"
            ] = batch_number

            population_rows.append(
                normalized_row
            )

        included_batches.append(
            batch_info
        )

    included_batches.sort(
        key=lambda item: item[
            "batch_number"
        ]
    )

    excluded_batches.sort(
        key=lambda item: item[
            "batch_number"
        ]
    )

    return {
        "reference_comparison_id": str(
            selected_manifest.get(
                "comparison_id",
                selected_dir.name,
            )
        ),
        "comparison_family_id": (
            selected_family
        ),
        "context_sha256": (
            selected_context
        ),
        "comparison_window": (
            selected_window
        ),
        "rows": population_rows,
        "included_batches": (
            included_batches
        ),
        "excluded_batches": (
            excluded_batches
        ),
    }