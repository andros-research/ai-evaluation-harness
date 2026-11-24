#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from market_state_contract_v0_1 import (
    validate_market_state,
)


ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT
    / "benchmarks"
    / "fixtures"
    / "market_state_universe_v0_1"
    / "series_registry.json"
)

OUT = (
    ROOT
    / "benchmarks"
    / "fixtures"
    / "market_state_universe_v0_2"
)

REGISTRY = OUT / "series_registry.json"
MANIFEST = OUT / "series_registry_manifest.json"


def sha(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def write_json(path: Path, payload):
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


TREASURY_CONFIG = {
    "ust_2y": {
        "source_type":
            "us_treasury",
        "source_series":
            "daily_treasury_yield_curve:2YR",
        "source_dataset":
            "daily_treasury_yield_curve",
        "source_field":
            "2YR",
        "source_definition":
            "U.S. Treasury 2-year par yield curve rate.",
        "release_name":
            "Daily Treasury Par Yield Curve Rates",
    },

    "ust_5y": {
        "source_type":
            "us_treasury",
        "source_series":
            "daily_treasury_yield_curve:5YR",
        "source_dataset":
            "daily_treasury_yield_curve",
        "source_field":
            "5YR",
        "source_definition":
            "U.S. Treasury 5-year par yield curve rate.",
        "release_name":
            "Daily Treasury Par Yield Curve Rates",
    },

    "ust_10y": {
        "source_type":
            "us_treasury",
        "source_series":
            "daily_treasury_yield_curve:10YR",
        "source_dataset":
            "daily_treasury_yield_curve",
        "source_field":
            "10YR",
        "source_definition":
            "U.S. Treasury 10-year par yield curve rate.",
        "release_name":
            "Daily Treasury Par Yield Curve Rates",
    },

    "ust_30y": {
        "source_type":
            "us_treasury",
        "source_series":
            "daily_treasury_yield_curve:30YR",
        "source_dataset":
            "daily_treasury_yield_curve",
        "source_field":
            "30YR",
        "source_definition":
            "U.S. Treasury 30-year par yield curve rate.",
        "release_name":
            "Daily Treasury Par Yield Curve Rates",
    },

    "tips_5y_real": {
        "source_type":
            "us_treasury",
        "source_series":
            "daily_treasury_real_yield_curve:5YR",
        "source_dataset":
            "daily_treasury_real_yield_curve",
        "source_field":
            "5YR",
        "source_definition":
            "U.S. Treasury 5-year par real yield curve rate.",
        "release_name":
            "Daily Treasury Par Real Yield Curve Rates",
    },

    "tips_10y_real": {
        "source_type":
            "us_treasury",
        "source_series":
            "daily_treasury_real_yield_curve:10YR",
        "source_dataset":
            "daily_treasury_real_yield_curve",
        "source_field":
            "10YR",
        "source_definition":
            "U.S. Treasury 10-year par real yield curve rate.",
        "release_name":
            "Daily Treasury Par Real Yield Curve Rates",
    },

    "breakeven_5y": {
        "source_type":
            "derived_us_treasury",
        "source_series":
            "TREASURY_5Y_NOMINAL_MINUS_REAL",
        "source_definition":
            (
                "Derived 5-year breakeven compensation: "
                "Treasury 5-year nominal par yield minus "
                "Treasury 5-year par real yield on the "
                "same observation date."
            ),
        "release_name":
            "Derived from Treasury.gov primary-source curves",
        "derivation":
            {
                "nominal":
                    "ust_5y",
                "real":
                    "tips_5y_real",
            },
    },

    "breakeven_10y": {
        "source_type":
            "derived_us_treasury",
        "source_series":
            "TREASURY_10Y_NOMINAL_MINUS_REAL",
        "source_definition":
            (
                "Derived 10-year breakeven compensation: "
                "Treasury 10-year nominal par yield minus "
                "Treasury 10-year par real yield on the "
                "same observation date."
            ),
        "release_name":
            "Derived from Treasury.gov primary-source curves",
        "derivation":
            {
                "nominal":
                    "ust_10y",
                "real":
                    "tips_10y_real",
            },
    },
}


def main():
    if OUT.exists():
        raise FileExistsError(
            f"Refusing to replace frozen universe: {OUT}"
        )

    source = json.loads(
        SOURCE.read_text()
    )

    rows = []

    for row in source["series"]:
        row = dict(row)

        config = TREASURY_CONFIG.get(
            row["series_id"]
        )

        if config is not None:
            row.update(
                config
            )

            row["source_provider"] = (
                "U.S. Department of the Treasury"
            )

            row["revision_policy"] = (
                "source_defined"
            )

        rows.append(
            row
        )

    payload = {
        "schema_version":
            "market_state_series_registry_v0_2",

        "registry_id":
            "cross_asset_core_v0_2",

        "purpose":
            (
                "Cross-asset core universe with "
                "Treasury.gov primary-source nominal "
                "and real curves and internally "
                "derived synchronized breakevens."
            ),

        "series":
            rows,

        "families":
            sorted({
                row["family"]
                for row in rows
            }),

        "frozen_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "supersedes_registry":
            source["registry_id"],
    }

    validation_state = {
        "schema_version":
            "macro_market_state_v0_1",

        "state_id":
            "registry_v0_2_validation",

        "as_of_timestamp":
            "2099-01-01T00:00:00+00:00",

        "created_at":
            "2099-01-01T00:00:01+00:00",

        "series":
            rows,

        "observations":
            [],
    }

    validate_market_state(
        validation_state
    )

    OUT.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_json(
        REGISTRY,
        payload,
    )

    write_json(
        MANIFEST,
        {
            "schema_version":
                "market_state_series_registry_freeze_v0_2",

            "registry_id":
                payload["registry_id"],

            "status":
                "frozen",

            "series_count":
                len(rows),

            "registry_file":
                REGISTRY.name,

            "registry_sha256":
                sha(REGISTRY),

            "primary_rate_source":
                "U.S. Department of the Treasury",

            "breakeven_method":
                (
                    "same-date nominal minus "
                    "real Treasury curve"
                ),

            "supersedes_registry":
                source["registry_id"],
        },
    )

    print("=" * 80)
    print("MARKET-STATE UNIVERSE v0.2 FROZEN")
    print("=" * 80)

    print(
        "Registry:",
        payload["registry_id"],
    )

    print(
        "Series:",
        len(rows),
    )

    print(
        "SHA:",
        sha(REGISTRY),
    )


if __name__ == "__main__":
    main()
