#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from macro_evidence_contract_v0_2 import (
    validate_macro_evidence,
)


ROOT = Path(__file__).resolve().parents[1]

SNAPSHOT_ROOT = (
    ROOT
    / "benchmarks"
    / "fixtures"
    / "macro_snapshot_2026_10_v0_2"
)

CLAIM_DIR = (
    SNAPSHOT_ROOT
    / "fred_claims"
)

EVIDENCE_PATH = (
    SNAPSHOT_ROOT
    / "macro_evidence.json"
)

CLAIMS_PATH = (
    CLAIM_DIR
    / "selected_fred_claims.json"
)

MANIFEST_PATH = (
    SNAPSHOT_ROOT
    / "snapshot_manifest.json"
)

PROVENANCE_PATH = (
    SNAPSHOT_ROOT
    / "source_provenance.json"
)


def sha256_path(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def write_json(
    path: Path,
    payload,
):
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main():
    if SNAPSHOT_ROOT.exists():
        raise FileExistsError(
            f"Refusing to replace frozen snapshot: "
            f"{SNAPSHOT_ROOT}"
        )

    CLAIM_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    frozen_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    # --------------------------------------------------------
    # Source-grounded macro evidence.
    # --------------------------------------------------------

    evidence = {
        "schema_version":
            "macro_evidence_v0_2",

        "observation_date":
            "2026-10-01",

        "records": [
            {
                "evidence_id":
                    "oct2026__headline_cpi_mom",

                "evidence_kind":
                    "period_change",

                "metric_name":
                    "cpi_headline_mom",

                "display_name":
                    "Headline CPI m/m",

                "source_type":
                    "bls",

                "source_series":
                    "CPI-U",

                "source_observation_date":
                    "2026-10-01",

                "comparison_window":
                    "1m",

                "availability_status":
                    "available",

                "current_value":
                    0.4,

                "prior_value":
                    None,

                "delta_value":
                    None,

                "direction":
                    "up",

                "units":
                    "percent",

                "source_note":
                    (
                        "Seasonally adjusted change "
                        "from September 2026 to "
                        "October 2026."
                    ),

                "supporting_values": {
                    "period":
                        "2026-10",

                    "change_percent":
                        0.4,

                    "seasonal_adjustment":
                        "seasonally_adjusted",
                },
            },

            {
                "evidence_id":
                    "oct2026__core_cpi_mom",

                "evidence_kind":
                    "period_change",

                "metric_name":
                    "cpi_core_mom",

                "display_name":
                    "Core CPI m/m",

                "source_type":
                    "bls",

                "source_series":
                    "CPI-U less food and energy",

                "source_observation_date":
                    "2026-10-01",

                "comparison_window":
                    "1m",

                "availability_status":
                    "available",

                "current_value":
                    0.3,

                "prior_value":
                    None,

                "delta_value":
                    None,

                "direction":
                    "up",

                "units":
                    "percent",

                "source_note":
                    (
                        "Seasonally adjusted change "
                        "from September 2026 to "
                        "October 2026."
                    ),

                "supporting_values": {
                    "period":
                        "2026-10",

                    "change_percent":
                        0.3,

                    "seasonal_adjustment":
                        "seasonally_adjusted",
                },
            },

            {
                "evidence_id":
                    "oct2026__cpi_yoy_unavailable",

                "evidence_kind":
                    "availability_note",

                "metric_name":
                    "cpi_yoy",

                "display_name":
                    "Headline CPI YoY",

                "source_type":
                    "bls",

                "source_series":
                    "CPI-U",

                "source_observation_date":
                    "2026-10-01",

                "comparison_window":
                    "12m",

                "availability_status":
                    "unavailable_source_gap",

                "current_value":
                    None,

                "prior_value":
                    None,

                "delta_value":
                    None,

                "direction":
                    None,

                "units":
                    "percent",

                "source_note":
                    (
                        "BLS did not collect survey "
                        "data for October 2025 during "
                        "the lapse in appropriations. "
                        "The required prior observation "
                        "is unavailable, so the October "
                        "2026 12-month all-items CPI "
                        "change cannot be published."
                    ),

                "supporting_values": {
                    "missing_period":
                        "2025-10",

                    "required_for":
                        "2026-10 12m comparison",
                },
            },

            {
                "evidence_id":
                    "oct2026__unrate_1m",

                "evidence_kind":
                    "prior_comparison",

                "metric_name":
                    "unrate",

                "display_name":
                    "Unemployment",

                "source_type":
                    "fred",

                "source_series":
                    "UNRATE",

                "source_observation_date":
                    "2026-10-01",

                "comparison_window":
                    "1m",

                "availability_status":
                    "available",

                "current_value":
                    4.1,

                "prior_value":
                    4.2,

                "delta_value":
                    -0.1,

                "direction":
                    "down",

                "units":
                    "percentage points",

                "source_note":
                    None,

                "supporting_values": {
                    "current_date":
                        "2026-10-01",

                    "current_value":
                        4.1,

                    "prior_date":
                        "2026-09-01",

                    "prior_value":
                        4.2,

                    "delta_value":
                        -0.1,

                    "units":
                        "percentage points",
                },
            },

            {
                "evidence_id":
                    "oct2026__fedfunds_1m",

                "evidence_kind":
                    "prior_comparison",

                "metric_name":
                    "fedfunds",

                "display_name":
                    "Fed funds",

                "source_type":
                    "fred",

                "source_series":
                    "FEDFUNDS",

                "source_observation_date":
                    "2026-10-01",

                "comparison_window":
                    "1m",

                "availability_status":
                    "available",

                "current_value":
                    3.88,

                "prior_value":
                    3.75,

                "delta_value":
                    0.13,

                "direction":
                    "up",

                "units":
                    "percentage points",

                "source_note":
                    "Monthly average of daily figures.",

                "supporting_values": {
                    "current_date":
                        "2026-10-01",

                    "current_value":
                        3.88,

                    "prior_date":
                        "2026-09-01",

                    "prior_value":
                        3.75,

                    "delta_value":
                        0.13,

                    "units":
                        "percentage points",
                },
            },

            {
                "evidence_id":
                    "oct2026__gs10_1m",

                "evidence_kind":
                    "prior_comparison",

                "metric_name":
                    "gs10",

                "display_name":
                    "10Y Treasury",

                "source_type":
                    "fred",

                "source_series":
                    "GS10",

                "source_observation_date":
                    "2026-10-01",

                "comparison_window":
                    "1m",

                "availability_status":
                    "available",

                "current_value":
                    5.13,

                "prior_value":
                    4.99,

                "delta_value":
                    0.14,

                "direction":
                    "up",

                "units":
                    "percentage points",

                "source_note":
                    "Monthly average of business days.",

                "supporting_values": {
                    "current_date":
                        "2026-10-01",

                    "current_value":
                        5.13,

                    "prior_date":
                        "2026-09-01",

                    "prior_value":
                        4.99,

                    "delta_value":
                        0.14,

                    "units":
                        "percentage points",
                },
            },
        ],
    }

    validate_macro_evidence(
        evidence
    )

    write_json(
        EVIDENCE_PATH,
        evidence,
    )

    # --------------------------------------------------------
    # Narrative-eligible deterministic claims.
    #
    # The unavailable CPI YoY record intentionally does NOT
    # become a claim. It will enter the mixed prompt as an
    # explicit evidence limitation.
    # --------------------------------------------------------

    claim_specs = [
        {
            "evidence_id":
                "oct2026__headline_cpi_mom",

            "claim_id":
                "macro__cpi_headline_mom__BLS_CPI_U__period_change__2026-10-01",

            "claim_text":
                (
                    "Headline CPI increased "
                    "0.4 percent in October "
                    "on a seasonally adjusted basis"
                ),

            "claim_type":
                "period_change",
        },

        {
            "evidence_id":
                "oct2026__core_cpi_mom",

            "claim_id":
                "macro__cpi_core_mom__BLS_CPI_U_CORE__period_change__2026-10-01",

            "claim_text":
                (
                    "Core CPI increased "
                    "0.3 percent in October "
                    "on a seasonally adjusted basis"
                ),

            "claim_type":
                "period_change",
        },

        {
            "evidence_id":
                "oct2026__unrate_1m",

            "claim_id":
                "macro__unrate__UNRATE__prior_comparison__2026-10-01",

            "claim_text":
                (
                    "Unemployment rate decreased "
                    "by 0.1 percentage point versus "
                    "September, from 4.2 to 4.1"
                ),

            "claim_type":
                "prior_comparison",
        },

        {
            "evidence_id":
                "oct2026__fedfunds_1m",

            "claim_id":
                "macro__fedfunds__FEDFUNDS__prior_comparison__2026-10-01",

            "claim_text":
                (
                    "Effective federal funds rate "
                    "increased by 0.13 percentage "
                    "point versus September, from "
                    "3.75 to 3.88"
                ),

            "claim_type":
                "prior_comparison",
        },

        {
            "evidence_id":
                "oct2026__gs10_1m",

            "claim_id":
                "macro__gs10__GS10__prior_comparison__2026-10-01",

            "claim_text":
                (
                    "10-year Treasury yield increased "
                    "by 0.14 percentage point versus "
                    "September, from 4.99 to 5.13"
                ),

            "claim_type":
                "prior_comparison",
        },
    ]

    evidence_by_id = {
        record["evidence_id"]:
            record
        for record
        in evidence["records"]
    }

    selected_claims = []

    for rank, spec in enumerate(
        claim_specs,
        start=1,
    ):
        source = evidence_by_id[
            spec["evidence_id"]
        ]

        selected_claims.append({
            "claim_id":
                spec["claim_id"],

            "claim_schema_version":
                "macro_claim_schema_v0_2",

            "claim_strength":
                "deterministic",

            "claim_text":
                spec["claim_text"] + ".",

            "claim_type":
                spec["claim_type"],

            "comparison_window":
                source[
                    "comparison_window"
                ],

            "created_at":
                frozen_at,

            "current_value":
                source.get(
                    "current_value"
                ),

            "prior_value":
                source.get(
                    "prior_value"
                ),

            "delta_value":
                source.get(
                    "delta_value"
                ),

            "direction":
                source.get(
                    "direction"
                ),

            "eligible_for_narrative":
                True,

            "experiment_name":
                "v1_9_3_oct2026_multi_model_v0_1",

            "generation_method":
                "deterministic_macro_evidence_v0_2",

            "metric_name":
                source[
                    "metric_name"
                ],

            "model":
                None,

            "prompt_id":
                "macro_mixed_v0_1",

            "selected_at":
                frozen_at,

            "selection_method":
                "eligible_for_narrative_filter",

            "selection_rank":
                rank,

            "selection_schema_version":
                "fred_selected_claims_v0_1",

            "source_observation_date":
                source[
                    "source_observation_date"
                ],

            "source_release_date":
                None,

            "source_series":
                source[
                    "source_series"
                ],

            "source_series_name":
                source[
                    "display_name"
                ],

            "source_type":
                source[
                    "source_type"
                ],

            "supporting_values":
                source[
                    "supporting_values"
                ],
        })

    write_json(
        CLAIMS_PATH,
        selected_claims,
    )

    provenance = {
        "schema_version":
            "macro_source_provenance_v0_1",

        "frozen_at":
            frozen_at,

        "observation_date":
            "2026-10-01",

        "sources": [
            {
                "source":
                    "BLS CPI October 2026",

                "release_date":
                    "2026-11-10",

                "supports": [
                    "headline CPI +0.4% m/m SA",
                    "core CPI +0.3% m/m SA",
                    "October 2026 CPI YoY availability gap",
                ],

                "url":
                    "https://www.bls.gov/news.release/cpi.htm",
            },
            {
                "source":
                    "FRED UNRATE",

                "supports": [
                    "September 2026 4.2%",
                    "October 2026 4.1%",
                ],

                "url":
                    "https://fred.stlouisfed.org/series/UNRATE",
            },
            {
                "source":
                    "FRED FEDFUNDS",

                "supports": [
                    "September 2026 3.75%",
                    "October 2026 3.88%",
                ],

                "url":
                    "https://fred.stlouisfed.org/series/FEDFUNDS",
            },
            {
                "source":
                    "FRED GS10",

                "supports": [
                    "September 2026 4.99%",
                    "October 2026 5.13%",
                ],

                "url":
                    "https://fred.stlouisfed.org/series/GS10",
            },
        ],

        "policy":
            (
                "Values manually frozen from official "
                "BLS/FRED publications after source "
                "verification. No unavailable CPI YoY "
                "value was reconstructed."
            ),
    }

    write_json(
        PROVENANCE_PATH,
        provenance,
    )

    manifest = {
        "schema_version":
            "macro_snapshot_manifest_v0_2",

        "observation_month":
            "2026-10-01",

        "comparison_windows": [
            "1m",
            "12m availability-only",
        ],

        "frozen_at_utc":
            frozen_at,

        "vintage_policy":
            (
                "Current published values at freeze; "
                "explicit source-gap state preserved."
            ),

        "evidence_record_count":
            len(
                evidence["records"]
            ),

        "selected_claim_count":
            len(
                selected_claims
            ),

        "availability_note_count":
            sum(
                record[
                    "evidence_kind"
                ]
                == "availability_note"
                for record
                in evidence["records"]
            ),

        "files_sha256": {
            "macro_evidence.json":
                sha256_path(
                    EVIDENCE_PATH
                ),

            "fred_claims/selected_fred_claims.json":
                sha256_path(
                    CLAIMS_PATH
                ),

            "source_provenance.json":
                sha256_path(
                    PROVENANCE_PATH
                ),
        },
    }

    write_json(
        MANIFEST_PATH,
        manifest,
    )

    print("=" * 80)
    print("OCTOBER 2026 MACRO EVIDENCE FROZEN")
    print("=" * 80)
    print(
        "Evidence records:",
        len(
            evidence["records"]
        ),
    )
    print(
        "Narrative claims:",
        len(
            selected_claims
        ),
    )
    print(
        "Availability notes:",
        manifest[
            "availability_note_count"
        ],
    )
    print()
    print(
        "Evidence SHA:",
        manifest[
            "files_sha256"
        ][
            "macro_evidence.json"
        ],
    )
    print(
        "Claims SHA:",
        manifest[
            "files_sha256"
        ][
            "fred_claims/selected_fred_claims.json"
        ],
    )
    print(
        "Snapshot:",
        SNAPSHOT_ROOT,
    )


if __name__ == "__main__":
    main()
