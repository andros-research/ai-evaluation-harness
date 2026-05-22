#!/usr/bin/env python3
"""
Select FRED-native claims for narrative use.

v1.6.0 scaffold:
- reads FRED claim artifacts produced by build_fred_claims.py
- selects claims eligible for narrative generation
- writes durable selected-claim artifacts
- preserves the source-grounded evidence contract for downstream narrative generation
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SCHEMA_VERSION = "fred_selected_claims_v0_1"
SELECTION_METHOD = "eligible_for_narrative_filter"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = REPO_ROOT / "benchmarks" / "results" / "fred_claims" / "fred_claims.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "results" / "fred_claims"

SELECTED_CLAIM_COLUMNS = [
    "selection_rank",
    "claim_id",
    "claim_text",
    "source_type",
    "source_series",
    "source_series_name",
    "source_observation_date",
    "source_release_date",
    "prompt_id",
    "model",
    "experiment_name",
    "claim_type",
    "metric_name",
    "comparison_window",
    "current_value",
    "prior_value",
    "delta_value",
    "direction",
    "supporting_values",
    "claim_strength",
    "eligible_for_narrative",
    "generation_method",
    "claim_schema_version",
    "selection_method",
    "selection_schema_version",
    "selected_at",
]

REQUIRED_SELECTED_FIELDS = [
    "selection_rank",
    "claim_id",
    "claim_text",
    "source_type",
    "source_series",
    "source_observation_date",
    "claim_type",
    "metric_name",
    "comparison_window",
    "supporting_values",
    "claim_strength",
    "eligible_for_narrative",
    "generation_method",
    "claim_schema_version",
    "selection_method",
    "selection_schema_version",
    "selected_at",
]


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def load_claims(path: Path) -> list[dict]:
    """Load FRED claim records from JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Input claims file does not exist: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, list):
        raise ValueError(f"Expected list of claim records in {path}")

    return payload


def select_claims(claim_records: list[dict], selected_at: str) -> list[dict]:
    """Select narrative-eligible FRED claims while preserving input order."""
    selected: list[dict] = []

    for claim in claim_records:
        if claim.get("eligible_for_narrative") is not True:
            continue

        record = dict(claim)
        record["selection_rank"] = len(selected) + 1
        record["claim_schema_version"] = record.pop("schema_version", None)
        record["selection_method"] = SELECTION_METHOD
        record["selection_schema_version"] = SCHEMA_VERSION
        record["selected_at"] = selected_at
        selected.append(record)

    return selected


def validate_selected_claims(selected_claims: list[dict]) -> None:
    """Validate selected claim records before writing artifacts."""
    errors: list[str] = []

    for idx, claim in enumerate(selected_claims):
        claim_id = claim.get("claim_id", f"<row {idx}>")

        for field in REQUIRED_SELECTED_FIELDS:
            if field not in claim:
                errors.append(f"{claim_id}: missing required field {field!r}")
                continue

            value = claim[field]

            if value is None:
                errors.append(f"{claim_id}: required field {field!r} is None")
                continue

            if isinstance(value, str) and not value.strip():
                errors.append(f"{claim_id}: required field {field!r} is empty")

        if not isinstance(claim.get("selection_rank"), int):
            errors.append(f"{claim_id}: selection_rank must be an integer")

        if claim.get("selection_method") != SELECTION_METHOD:
            errors.append(
                f"{claim_id}: selection_method={claim.get('selection_method')!r} "
                f"expected {SELECTION_METHOD!r}"
            )

        if claim.get("selection_schema_version") != SCHEMA_VERSION:
            errors.append(
                f"{claim_id}: selection_schema_version={claim.get('selection_schema_version')!r} "
                f"expected {SCHEMA_VERSION!r}"
            )

        if not isinstance(claim.get("supporting_values"), dict):
            errors.append(f"{claim_id}: supporting_values must be a dict before CSV serialization")

    if errors:
        joined = "\n".join(f"- {err}" for err in errors)
        raise ValueError(f"Selected FRED claim validation failed:\n{joined}")


def write_json(path: Path, payload: object) -> None:
    """Write JSON with stable formatting."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_selected_artifacts(
    *,
    input_path: Path,
    output_dir: Path,
) -> None:
    """Select and write FRED selected-claim artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_at = utc_now_iso()
    claim_records = load_claims(input_path)
    selected_claims = select_claims(claim_records, selected_at=selected_at)

    validate_selected_claims(selected_claims)

    selected_df = pd.DataFrame(selected_claims, columns=SELECTED_CLAIM_COLUMNS)

    selected_csv = output_dir / "selected_fred_claims.csv"
    selected_json = output_dir / "selected_fred_claims.json"
    metadata_json = output_dir / "selected_fred_claims_metadata.json"

    csv_df = selected_df.copy()
    if not csv_df.empty:
        csv_df["supporting_values"] = csv_df["supporting_values"].apply(
            lambda value: json.dumps(value, sort_keys=True)
        )

    csv_df.to_csv(selected_csv, index=False)
    write_json(selected_json, selected_claims)

    metadata = {
        "selection_schema_version": SCHEMA_VERSION,
        "selection_method": SELECTION_METHOD,
        "input_file": str(input_path),
        "selected_at": selected_at,
        "n_input_claims": int(len(claim_records)),
        "n_selected_claims": int(len(selected_claims)),
        "selected_claim_ids": [claim["claim_id"] for claim in selected_claims],
        "series_included": sorted(
            selected_df["source_series"].dropna().unique().tolist()
        )
        if not selected_df.empty
        else [],
        "claim_types_included": sorted(
            selected_df["claim_type"].dropna().unique().tolist()
        )
        if not selected_df.empty
        else [],
        "comparison_windows_included": sorted(
            selected_df["comparison_window"].dropna().unique().tolist()
        )
        if not selected_df.empty
        else [],
        "output_files": {
            "selected_csv": str(selected_csv),
            "selected_json": str(selected_json),
            "metadata_json": str(metadata_json),
        },
    }
    write_json(metadata_json, metadata)

    print("Wrote selected FRED claim artifacts:")
    print(f"  {selected_csv}")
    print(f"  {selected_json}")
    print(f"  {metadata_json}")
    print(f"n_input_claims={len(claim_records)}")
    print(f"n_selected_claims={len(selected_claims)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select FRED-native claims for narrative use."
    )
    parser.add_argument(
        "--input-claims",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input fred_claims.json file produced by build_fred_claims.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where selected FRED claim artifacts will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_selected_artifacts(
        input_path=args.input_claims,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()