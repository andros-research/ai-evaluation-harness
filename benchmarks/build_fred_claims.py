#!/usr/bin/env python3
"""
Build FRED-native claim artifacts.

v1.6.0 scaffold:
- creates durable FRED claim output artifacts
- writes empty-but-valid CSV/JSON/metadata files
- establishes the artifact contract before real FRED parsing is added

Future versions will consume structured FRED context and emit deterministic
macro claims tied directly to source observations.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SCHEMA_VERSION = "fred_claim_schema_v0_1"
GENERATION_METHOD = "deterministic_fred_rule"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "results" / "fred_claims"

CLAIM_COLUMNS = [
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
    "schema_version",
    "created_at",
]


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def build_empty_claims_frame() -> pd.DataFrame:
    """Return an empty claims DataFrame with the v0.1 schema columns."""
    return pd.DataFrame(columns=CLAIM_COLUMNS)


def write_json(path: Path, payload: object) -> None:
    """Write JSON with stable formatting."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_artifacts(output_dir: Path) -> None:
    """Write empty-but-valid FRED claim artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    created_at = utc_now_iso()
    claims_df = build_empty_claims_frame()

    claims_csv = output_dir / "fred_claims.csv"
    claims_json = output_dir / "fred_claims.json"
    metadata_json = output_dir / "fred_claims_metadata.json"

    claims_df.to_csv(claims_csv, index=False)

    # Empty list for now; future versions will write records.
    write_json(claims_json, [])

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generation_method": GENERATION_METHOD,
        "input_file": None,
        "created_at": created_at,
        "n_claims": int(len(claims_df)),
        "series_included": [],
        "claim_types_included": [],
        "output_files": {
            "claims_csv": str(claims_csv),
            "claims_json": str(claims_json),
            "metadata_json": str(metadata_json),
        },
    }
    write_json(metadata_json, metadata)

    print("Wrote FRED claim artifacts:")
    print(f"  {claims_csv}")
    print(f"  {claims_json}")
    print(f"  {metadata_json}")
    print(f"n_claims={len(claims_df)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FRED-native claim artifacts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where FRED claim artifacts will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_artifacts(args.output_dir)


if __name__ == "__main__":
    main()