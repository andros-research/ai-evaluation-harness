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

REQUIRED_CLAIM_FIELDS = [
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
    "schema_version",
    "created_at",
]

METRIC_CONFIG = {
    "CPI_YOY": {
        "metric_name": "cpi_yoy",
        "metric_label": "CPI year-over-year inflation",
        "source_series": "CPIAUCSL",
        "source_series_name": "Consumer Price Index for All Urban Consumers: All Items",
        "units": "percentage points",
    },
    "UNRATE": {
        "metric_name": "unrate",
        "metric_label": "Unemployment rate",
        "source_series": "UNRATE",
        "source_series_name": "Unemployment Rate",
        "units": "percentage points",
    },
    "FEDFUNDS": {
        "metric_name": "fedfunds",
        "metric_label": "Effective federal funds rate",
        "source_series": "FEDFUNDS",
        "source_series_name": "Effective Federal Funds Rate",
        "units": "percentage points",
    },
    "GS10": {
        "metric_name": "gs10",
        "metric_label": "10-year Treasury yield",
        "source_series": "GS10",
        "source_series_name": "10-Year Treasury Constant Maturity Rate",
        "units": "percentage points",
    },
}


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def build_empty_claims_frame() -> pd.DataFrame:
    """Return an empty claims DataFrame with the v0.1 schema columns."""
    return pd.DataFrame(columns=CLAIM_COLUMNS)


def validate_claim_records(claim_records: list[dict]) -> None:
    """Validate required fields before writing claim artifacts."""
    errors: list[str] = []

    for idx, claim in enumerate(claim_records):
        claim_id = claim.get("claim_id", f"<row {idx}>")

        for field in REQUIRED_CLAIM_FIELDS:
            if field not in claim:
                errors.append(f"{claim_id}: missing required field {field!r}")
                continue

            value = claim[field]

            if value is None:
                errors.append(f"{claim_id}: required field {field!r} is None")
                continue

            if isinstance(value, str) and not value.strip():
                errors.append(f"{claim_id}: required field {field!r} is empty")

        if claim.get("schema_version") != SCHEMA_VERSION:
            errors.append(
                f"{claim_id}: schema_version={claim.get('schema_version')!r} "
                f"expected {SCHEMA_VERSION!r}"
            )

        if claim.get("generation_method") != GENERATION_METHOD:
            errors.append(
                f"{claim_id}: generation_method={claim.get('generation_method')!r} "
                f"expected {GENERATION_METHOD!r}"
            )

        if not isinstance(claim.get("supporting_values"), dict):
            errors.append(
                f"{claim_id}: supporting_values must be a dict before JSON serialization"
            )

    if errors:
        joined = "\n".join(f"- {err}" for err in errors)
        raise ValueError(f"FRED claim validation failed:\n{joined}")


def round_metric_value(value: float | None, digits: int = 6) -> float | None:
    """Round numeric metric values for stable claim artifacts."""
    if value is None:
        return None
    return round(float(value), digits)


def classify_direction(delta_value: float, tolerance: float = 1e-9) -> str:
    """Classify numeric direction from a delta."""
    if delta_value > tolerance:
        return "up"
    if delta_value < -tolerance:
        return "down"
    return "unchanged"


def make_claim_id(
    metric_name: str,
    source_series: str,
    claim_type: str,
    source_observation_date: str,
) -> str:
    """Build a stable, human-readable FRED claim ID."""
    return (
        f"fred__{metric_name}__{source_series}__"
        f"{claim_type}__{source_observation_date}"
    )


def build_prior_comparison_claim(
    *,
    metric_name: str,
    metric_label: str,
    source_series: str,
    source_series_name: str,
    current_date: str,
    current_value: float,
    prior_date: str,
    prior_value: float,
    units: str,
    created_at: str,
    comparison_window: str = "prior_observation",
    source_release_date: str | None = None,
    prompt_id: str | None = None,
) -> dict:
    """Build a deterministic prior-comparison FRED claim record."""
    claim_type = "prior_comparison"
    current_value = round_metric_value(current_value)
    prior_value = round_metric_value(prior_value)
    delta_value = round_metric_value(current_value - prior_value)
    direction = classify_direction(delta_value)

    if direction == "up":
        direction_word = "increased"
    elif direction == "down":
        direction_word = "decreased"
    else:
        direction_word = "was unchanged"

    abs_delta = abs(delta_value)

    if direction == "unchanged":
        claim_text = (
            f"{metric_label} was unchanged versus the prior observation "
            f"at {current_value:g} {units}."
        )
    else:
        claim_text = (
            f"{metric_label} {direction_word} by {abs_delta:g} {units} "
            f"versus the prior observation, from {prior_value:g} to "
            f"{current_value:g}."
        )

    supporting_values = {
        "current_date": current_date,
        "current_value": current_value,
        "prior_date": prior_date,
        "prior_value": prior_value,
        "delta_value": delta_value,
        "units": units,
    }

    return {
        "claim_id": make_claim_id(
            metric_name=metric_name,
            source_series=source_series,
            claim_type=claim_type,
            source_observation_date=current_date,
        ),
        "claim_text": claim_text,
        "source_type": "fred",
        "source_series": source_series,
        "source_series_name": source_series_name,
        "source_observation_date": current_date,
        "source_release_date": source_release_date,
        "prompt_id": prompt_id,
        "model": None,
        "experiment_name": None,
        "claim_type": claim_type,
        "metric_name": metric_name,
        "comparison_window": comparison_window,
        "current_value": current_value,
        "prior_value": prior_value,
        "delta_value": delta_value,
        "direction": direction,
        "supporting_values": supporting_values,
        "claim_strength": "deterministic",
        "eligible_for_narrative": True,
        "generation_method": GENERATION_METHOD,
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
    }


def build_sample_claims(created_at: str) -> list[dict]:
    """Build a tiny sample claim set for validating the v0.1 schema."""
    return [
        build_prior_comparison_claim(
            metric_name="cpi_yoy",
            metric_label="CPI year-over-year inflation",
            source_series="CPIAUCSL",
            source_series_name="Consumer Price Index for All Urban Consumers: All Items in U.S. City Average",
            current_date="2026-04-01",
            current_value=3.1,
            prior_date="2026-03-01",
            prior_value=3.0,
            units="percentage points",
            created_at=created_at,
            source_release_date=None,
            prompt_id="macro_fred_cpi_snapshot",
        )
    ]
    

def load_fred_context(path: Path) -> dict:
    """Load structured FRED macro context JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def build_claims_from_context(
    *,
    context_path: Path,
    comparison_window: str,
    created_at: str,
    prompt_id: str | None = "macro_compare_12m",
) -> list[dict]:
    """Build deterministic FRED claims from structured macro context JSON."""
    context = load_fred_context(context_path)

    if comparison_window not in {"6m", "12m", "24m"}:
        raise ValueError(
            f"Unsupported comparison_window={comparison_window!r}. "
            "Expected one of: 6m, 12m, 24m."
        )

    latest_date = context["latest_date"]
    prior_date = context[f"prior_date_{comparison_window}"]

    latest_values = context["latest_values"]
    prior_values = context[f"prior_values_{comparison_window}"]

    claim_records = []

    for metric_key, config in METRIC_CONFIG.items():
        if metric_key not in latest_values or metric_key not in prior_values:
            continue

        claim_records.append(
            build_prior_comparison_claim(
                metric_name=config["metric_name"],
                metric_label=config["metric_label"],
                source_series=config["source_series"],
                source_series_name=config["source_series_name"],
                current_date=latest_date,
                current_value=latest_values[metric_key],
                prior_date=prior_date,
                prior_value=prior_values[metric_key],
                units=config["units"],
                created_at=created_at,
                comparison_window=comparison_window,
                source_release_date=None,
                prompt_id=prompt_id,
            )
        )

    return claim_records


def write_json(path: Path, payload: object) -> None:
    """Write JSON with stable formatting."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_artifacts(
    output_dir: Path,
    include_sample: bool = False,
    input_context: Path | None = None,
    comparison_window: str = "12m",
) -> None:
    """Write FRED claim artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    created_at = utc_now_iso()
    if input_context is not None:
        claim_records = build_claims_from_context(
            context_path=input_context,
            comparison_window=comparison_window,
            created_at=created_at,
            prompt_id=f"macro_compare_{comparison_window}",
        )
    elif include_sample:
        claim_records = build_sample_claims(created_at)
    else:
        claim_records = []

    validate_claim_records(claim_records)

    claims_df = pd.DataFrame(claim_records, columns=CLAIM_COLUMNS)

    claims_csv = output_dir / "fred_claims.csv"
    claims_json = output_dir / "fred_claims.json"
    metadata_json = output_dir / "fred_claims_metadata.json"

    csv_df = claims_df.copy()
    if not csv_df.empty:
        csv_df["supporting_values"] = csv_df["supporting_values"].apply(
            lambda value: json.dumps(value, sort_keys=True)
        )

    csv_df.to_csv(claims_csv, index=False)

    write_json(claims_json, claim_records)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generation_method": GENERATION_METHOD,
        "input_file": str(input_context) if input_context is not None else None,
        "comparison_window": comparison_window if input_context is not None else None,
        "created_at": created_at,
        "n_claims": int(len(claims_df)),
        "series_included": sorted(
            claims_df["source_series"].dropna().unique().tolist()
        )
        if not claims_df.empty
        else [],
        "claim_types_included": sorted(
            claims_df["claim_type"].dropna().unique().tolist()
        )
        if not claims_df.empty
        else [],
        "include_sample": include_sample,
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
    parser.add_argument(
        "--input-context",
        type=Path,
        default=None,
        help="Structured FRED macro context JSON produced by build_fred_prompt_context.py.",
    )
    parser.add_argument(
        "--comparison-window",
        default="12m",
        choices=["6m", "12m", "24m"],
        help="Comparison window to use from the FRED context.",
    )
    parser.add_argument(
        "--include-sample",
        action="store_true",
        help="Emit a tiny deterministic sample claim for schema validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_artifacts(
        args.output_dir,
        include_sample=args.include_sample,
        input_context=args.input_context,
        comparison_window=args.comparison_window,
    )
    

if __name__ == "__main__":
    main()