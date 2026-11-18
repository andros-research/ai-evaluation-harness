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

OUT_DIR = (
    ROOT
    / "benchmarks"
    / "fixtures"
    / "market_state_universe_v0_1"
)

UNIVERSE_PATH = (
    OUT_DIR
    / "series_registry.json"
)

MANIFEST_PATH = (
    OUT_DIR
    / "series_registry_manifest.json"
)


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


def sha(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def series(
    *,
    series_id,
    metric_name,
    display_name,
    family,
    source_series,
    units,
    value_type,
    source_definition,
    source_provider,
    release_name,
    quote_convention=None,
    history_note=None,
):
    return {
        "series_id":
            series_id,

        "metric_name":
            metric_name,

        "display_name":
            display_name,

        "family":
            family,

        # Ingestion source.
        "source_type":
            "fred",

        "source_series":
            source_series,

        "frequency":
            "daily",

        "units":
            units,

        "value_type":
            value_type,

        "aggregation_category":
            "point",

        # Conservative initial choice.
        # We will preserve retrieved vintages rather
        # than assuming market series never revise.
        "revision_policy":
            "source_defined",

        "source_definition":
            source_definition,

        "source_provider":
            source_provider,

        "release_name":
            release_name,

        "quote_convention":
            quote_convention,

        "history_note":
            history_note,

        "availability_policy":
            (
                "Observation may enter a market state "
                "only after its recorded available_timestamp. "
                "FRED update/release timing or an explicitly "
                "conservative availability timestamp must "
                "be preserved."
            ),
    }


SERIES = [
    # --------------------------------------------------
    # Nominal Treasury curve
    # --------------------------------------------------
    series(
        series_id="ust_2y",
        metric_name="ust_2y",
        display_name="2Y Treasury",
        family="rates",
        source_series="DGS2",
        units="percent",
        value_type="yield",
        source_definition=(
            "2-Year Treasury constant-maturity "
            "market yield."
        ),
        source_provider=(
            "Board of Governors of the "
            "Federal Reserve System"
        ),
        release_name="H.15 Selected Interest Rates",
    ),

    series(
        series_id="ust_5y",
        metric_name="ust_5y",
        display_name="5Y Treasury",
        family="rates",
        source_series="DGS5",
        units="percent",
        value_type="yield",
        source_definition=(
            "5-Year Treasury constant-maturity "
            "market yield."
        ),
        source_provider=(
            "Board of Governors of the "
            "Federal Reserve System"
        ),
        release_name="H.15 Selected Interest Rates",
    ),

    series(
        series_id="ust_10y",
        metric_name="ust_10y",
        display_name="10Y Treasury",
        family="rates",
        source_series="DGS10",
        units="percent",
        value_type="yield",
        source_definition=(
            "10-Year Treasury constant-maturity "
            "market yield."
        ),
        source_provider=(
            "Board of Governors of the "
            "Federal Reserve System"
        ),
        release_name="H.15 Selected Interest Rates",
    ),

    series(
        series_id="ust_30y",
        metric_name="ust_30y",
        display_name="30Y Treasury",
        family="rates",
        source_series="DGS30",
        units="percent",
        value_type="yield",
        source_definition=(
            "30-Year Treasury constant-maturity "
            "market yield."
        ),
        source_provider=(
            "Board of Governors of the "
            "Federal Reserve System"
        ),
        release_name="H.15 Selected Interest Rates",
    ),

    # --------------------------------------------------
    # Real yields / inflation pricing
    # --------------------------------------------------
    series(
        series_id="tips_5y_real",
        metric_name="tips_5y_real",
        display_name="5Y real yield",
        family="inflation_pricing",
        source_series="DFII5",
        units="percent",
        value_type="real_yield",
        source_definition=(
            "5-Year inflation-indexed Treasury "
            "constant-maturity yield."
        ),
        source_provider=(
            "Board of Governors of the "
            "Federal Reserve System"
        ),
        release_name="H.15 Selected Interest Rates",
    ),

    series(
        series_id="tips_10y_real",
        metric_name="tips_10y_real",
        display_name="10Y real yield",
        family="inflation_pricing",
        source_series="DFII10",
        units="percent",
        value_type="real_yield",
        source_definition=(
            "10-Year inflation-indexed Treasury "
            "constant-maturity yield."
        ),
        source_provider=(
            "Board of Governors of the "
            "Federal Reserve System"
        ),
        release_name="H.15 Selected Interest Rates",
    ),

    series(
        series_id="breakeven_5y",
        metric_name="breakeven_5y",
        display_name="5Y breakeven",
        family="inflation_pricing",
        source_series="T5YIE",
        units="percent",
        value_type="breakeven_rate",
        source_definition=(
            "5-Year breakeven inflation rate "
            "derived from nominal and "
            "inflation-indexed Treasury yields."
        ),
        source_provider=(
            "Federal Reserve Bank of St. Louis"
        ),
        release_name="Interest Rate Spreads",
    ),

    series(
        series_id="breakeven_10y",
        metric_name="breakeven_10y",
        display_name="10Y breakeven",
        family="inflation_pricing",
        source_series="T10YIE",
        units="percent",
        value_type="breakeven_rate",
        source_definition=(
            "10-Year breakeven inflation rate "
            "derived from nominal and "
            "inflation-indexed Treasury yields."
        ),
        source_provider=(
            "Federal Reserve Bank of St. Louis"
        ),
        release_name="Interest Rate Spreads",
    ),

    # --------------------------------------------------
    # Credit
    # --------------------------------------------------
    series(
        series_id="hy_oas",
        metric_name="hy_oas",
        display_name="US HY OAS",
        family="credit",
        source_series="BAMLH0A0HYM2",
        units="percent",
        value_type="option_adjusted_spread",
        source_definition=(
            "ICE BofA US High Yield Index "
            "Option-Adjusted Spread."
        ),
        source_provider="ICE Data Indices, LLC",
        release_name="ICE BofA Indices",
        history_note=(
            "FRED notes that beginning in April 2026 "
            "the series only exposes three years of "
            "observations."
        ),
    ),

    # --------------------------------------------------
    # FX
    # --------------------------------------------------
    series(
        series_id="eurusd",
        metric_name="eurusd",
        display_name="EUR/USD",
        family="fx",
        source_series="DEXUSEU",
        units="usd_per_eur",
        value_type="fx_rate",
        source_definition=(
            "U.S. Dollars to Euro Spot Exchange Rate; "
            "New York noon buying rate."
        ),
        source_provider=(
            "Board of Governors of the "
            "Federal Reserve System"
        ),
        release_name="H.10 Foreign Exchange Rates",
        quote_convention=(
            "U.S. dollars per one euro; "
            "higher value = stronger EUR / weaker USD."
        ),
    ),

    series(
        series_id="usdjpy",
        metric_name="usdjpy",
        display_name="USD/JPY",
        family="fx",
        source_series="DEXJPUS",
        units="jpy_per_usd",
        value_type="fx_rate",
        source_definition=(
            "Japanese Yen to U.S. Dollar Spot "
            "Exchange Rate; New York noon buying rate."
        ),
        source_provider=(
            "Board of Governors of the "
            "Federal Reserve System"
        ),
        release_name="H.10 Foreign Exchange Rates",
        quote_convention=(
            "Japanese yen per one U.S. dollar; "
            "higher value = stronger USD / weaker JPY."
        ),
    ),

    # --------------------------------------------------
    # Commodity
    # --------------------------------------------------
    series(
        series_id="wti",
        metric_name="wti",
        display_name="WTI crude",
        family="commodities",
        source_series="DCOILWTICO",
        units="usd_per_barrel",
        value_type="commodity_price",
        source_definition=(
            "West Texas Intermediate crude oil "
            "spot price at Cushing, Oklahoma."
        ),
        source_provider=(
            "U.S. Energy Information Administration"
        ),
        release_name="Spot Prices",
    ),

    # --------------------------------------------------
    # Equity
    # --------------------------------------------------
    series(
        series_id="sp500",
        metric_name="sp500",
        display_name="S&P 500",
        family="equities",
        source_series="SP500",
        units="index",
        value_type="index_level",
        source_definition=(
            "S&P 500 daily index value at market close."
        ),
        source_provider="S&P Dow Jones Indices LLC",
        release_name="Standard & Poors",
        quote_convention="Daily closing index level.",
    ),
]


def main():
    if OUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to replace frozen universe: "
            f"{OUT_DIR}"
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    frozen_at = datetime.now(
        timezone.utc
    ).isoformat()

    payload = {
        "schema_version":
            "market_state_series_registry_v0_1",

        "registry_id":
            "fred_cross_asset_core_v0_1",

        "purpose":
            (
                "First deliberately compact cross-asset "
                "FRED universe for the macro research engine."
            ),

        "series":
            SERIES,

        "families":
            sorted({
                item["family"]
                for item in SERIES
            }),

        "frozen_at":
            frozen_at,
    }

    # Reuse the generic market-state validator by
    # embedding the registry into an observation-free
    # synthetic state. This verifies every series field
    # without yet ingesting any market observations.
    validation_state = {
        "schema_version":
            "macro_market_state_v0_1",

        "state_id":
            "registry_validation_probe",

        "as_of_timestamp":
            "2099-01-01T00:00:00+00:00",

        "created_at":
            "2099-01-01T00:00:01+00:00",

        "series":
            SERIES,

        "observations":
            [],
    }

    validate_market_state(
        validation_state
    )

    write_json(
        UNIVERSE_PATH,
        payload,
    )

    manifest = {
        "schema_version":
            "market_state_series_registry_freeze_v0_1",

        "registry_id":
            payload["registry_id"],

        "status":
            "frozen",

        "series_count":
            len(SERIES),

        "family_count":
            len(
                payload[
                    "families"
                ]
            ),

        "families":
            payload["families"],

        "registry_file":
            UNIVERSE_PATH.name,

        "registry_sha256":
            sha(
                UNIVERSE_PATH
            ),

        "observations_frozen":
            0,

        "note":
            (
                "Series definitions only. "
                "No real observations are included yet."
            ),
    }

    write_json(
        MANIFEST_PATH,
        manifest,
    )

    print("=" * 80)
    print(
        "MARKET-STATE UNIVERSE FROZEN"
    )
    print("=" * 80)

    print(
        "Registry:",
        payload[
            "registry_id"
        ],
    )

    print(
        "Series:",
        len(SERIES),
    )

    print(
        "Families:",
        ", ".join(
            payload[
                "families"
            ]
        ),
    )

    print(
        "SHA:",
        sha(
            UNIVERSE_PATH
        ),
    )


if __name__ == "__main__":
    main()
