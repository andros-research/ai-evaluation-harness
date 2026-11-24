#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess

from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


try:
    from benchmarks.market_state_contract_v0_1 import (
        validate_market_state,
    )
    from benchmarks.build_fred_market_state_v0_1 import (
        fetch_fred_csv,
        parse_fred_csv,
        build_observations,
        write_json,
    )
except ModuleNotFoundError:
    from market_state_contract_v0_1 import (
        validate_market_state,
    )
    from build_fred_market_state_v0_1 import (
        fetch_fred_csv,
        parse_fred_csv,
        build_observations,
        write_json,
    )


ROOT = Path(__file__).resolve().parents[1]

REGISTRY_PATH = (
    ROOT
    / "benchmarks"
    / "fixtures"
    / "market_state_universe_v0_2"
    / "series_registry.json"
)

OUTPUT_ROOT = (
    ROOT
    / "benchmarks"
    / "results"
    / "market_states"
    / "cross_asset_core_v0_2"
)

TREASURY_BASE = (
    "https://home.treasury.gov/"
    "resource-center/data-chart-center/"
    "interest-rates/"
    "daily-treasury-rates.csv"
)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(
        data
    ).hexdigest()


def sha_path(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def normalize_header(value: str) -> str:
    return re.sub(
        r"[^A-Z0-9]",
        "",
        value.upper(),
    )


def treasury_url(
    *,
    year: int,
    dataset: str,
):
    return (
        f"{TREASURY_BASE}/{year}/all"
        f"?_format=csv"
        f"&field_tdr_date_value={year}"
        f"&page="
        f"&type={dataset}"
    )


def curl_bytes(
    url: str,
    *,
    timeout: int,
):
    command = [
        "curl",
        "-fsSL",

        "--connect-timeout",
        "15",

        "--max-time",
        str(timeout),

        "--retry",
        "3",

        "--retry-delay",
        "1",

        "--retry-all-errors",

        url,
    ]

    result = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if not result.stdout:
        raise RuntimeError(
            f"Empty response: {url}"
        )

    return result.stdout


def parse_treasury_csv(
    raw: bytes,
    *,
    wanted_fields: set[str],
    source_url: str,
    snapshot_sha: str,
    retrieved_at: str,
):
    text = raw.decode(
        "utf-8-sig"
    )

    reader = csv.DictReader(
        io.StringIO(text)
    )

    if not reader.fieldnames:
        raise RuntimeError(
            "Treasury CSV has no header."
        )

    normalized = {
        normalize_header(field):
            field
        for field in reader.fieldnames
    }

    date_field = normalized.get(
        "DATE"
    )

    if date_field is None:
        raise RuntimeError(
            "Treasury CSV DATE field not found."
        )

    resolved = {}

    for wanted in wanted_fields:
        field = normalized.get(
            wanted
        )

        if field is None:
            raise RuntimeError(
                f"Treasury field "
                f"{wanted!r} not found. "
                f"Available normalized fields: "
                f"{sorted(normalized)}"
            )

        resolved[wanted] = field

    out = defaultdict(
        list
    )

    for row in reader:
        raw_date = (
            row.get(
                date_field,
                ""
            )
            or ""
        ).strip()

        if not raw_date:
            continue

        obs_date = None

        for fmt in (
            "%m/%d/%Y",
            "%Y-%m-%d",
        ):
            try:
                obs_date = datetime.strptime(
                    raw_date,
                    fmt,
                ).date()
                break
            except ValueError:
                pass

        if obs_date is None:
            continue

        for wanted, field in (
            resolved.items()
        ):
            raw_value = (
                row.get(
                    field,
                    ""
                )
                or ""
            ).strip()

            if (
                not raw_value
                or raw_value.upper()
                in {
                    "N/A",
                    "NA",
                    ".",
                }
            ):
                continue

            try:
                value = float(
                    raw_value
                )
            except ValueError:
                continue

            out[wanted].append({
                "date":
                    obs_date,

                "value":
                    value,

                "source_url":
                    source_url,

                "source_snapshot_sha256":
                    snapshot_sha,

                "retrieved_at":
                    retrieved_at,
            })

    return out


def make_treasury_observation(
    *,
    meta: dict,
    row: dict,
):
    obs_date = (
        row["date"].isoformat()
    )

    return {
        "observation_id":
            (
                f"{meta['series_id']}"
                f"__{obs_date}"
            ),

        "series_id":
            meta["series_id"],

        "status":
            "observed",

        "value":
            float(
                row["value"]
            ),

        "observation_date":
            obs_date,

        "period_start":
            obs_date,

        "period_end":
            obs_date,

        "release_timestamp":
            None,

        "available_timestamp":
            row[
                "retrieved_at"
            ],

        "retrieved_at":
            row[
                "retrieved_at"
            ],

        "vintage": {
            "revision_policy":
                meta[
                    "revision_policy"
                ],

            "vintage_date":
                row[
                    "retrieved_at"
                ][:10],
        },

        "unavailable_reason":
            None,

        "availability_method":
            "retrieval_time_conservative",

        "source_url":
            row[
                "source_url"
            ],

        "source_snapshot_sha256":
            row[
                "source_snapshot_sha256"
            ],
    }


def derive_breakeven(
    *,
    meta: dict,
    nominal: list[dict],
    real: list[dict],
):
    nominal_by_date = {
        row[
            "observation_date"
        ]:
            row
        for row in nominal
    }

    real_by_date = {
        row[
            "observation_date"
        ]:
            row
        for row in real
    }

    dates = sorted(
        set(nominal_by_date)
        & set(real_by_date)
    )

    rows = []

    for obs_date in dates:
        n = nominal_by_date[
            obs_date
        ]

        r = real_by_date[
            obs_date
        ]

        available = max(
            n[
                "available_timestamp"
            ],
            r[
                "available_timestamp"
            ],
        )

        retrieved = max(
            n["retrieved_at"],
            r["retrieved_at"],
        )

        value = (
            float(n["value"])
            - float(r["value"])
        )

        component_hash = (
            sha_bytes(
                (
                    n[
                        "source_snapshot_sha256"
                    ]
                    + "|"
                    + r[
                        "source_snapshot_sha256"
                    ]
                    + "|"
                    + obs_date
                ).encode()
            )
        )

        rows.append({
            "observation_id":
                (
                    f"{meta['series_id']}"
                    f"__{obs_date}"
                ),

            "series_id":
                meta[
                    "series_id"
                ],

            "status":
                "observed",

            "value":
                round(
                    value,
                    10,
                ),

            "observation_date":
                obs_date,

            "period_start":
                obs_date,

            "period_end":
                obs_date,

            "release_timestamp":
                None,

            "available_timestamp":
                available,

            "retrieved_at":
                retrieved,

            "vintage": {
                "revision_policy":
                    meta[
                        "revision_policy"
                    ],

                "vintage_date":
                    retrieved[:10],
            },

            "unavailable_reason":
                None,

            "availability_method":
                (
                    "derived_from_same_date_"
                    "treasury_primary_sources"
                ),

            "derived_from": [
                n[
                    "observation_id"
                ],
                r[
                    "observation_id"
                ],
            ],

            "source_snapshot_sha256":
                component_hash,
        })

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lookback-days",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
    )

    parser.add_argument(
        "--timezone",
        default="America/New_York",
    )

    args = parser.parse_args()

    registry = json.loads(
        REGISTRY_PATH.read_text()
    )

    series = registry[
        "series"
    ]

    metadata = {
        row[
            "series_id"
        ]:
            row
        for row in series
    }

    tz = ZoneInfo(
        args.timezone
    )

    started_at = datetime.now(
        tz
    )

    maximum_date = (
        started_at.date()
    )

    cutoff_date = (
        maximum_date
        - timedelta(
            days=args.lookback_days
        )
    )

    years = range(
        cutoff_date.year,
        maximum_date.year + 1,
    )

    print("=" * 80)
    print(
        "BUILD CROSS-ASSET MARKET STATE v0.2"
    )
    print("=" * 80)

    print(
        "Registry:",
        registry[
            "registry_id"
        ],
    )

    print(
        "Lookback:",
        args.lookback_days,
        "calendar days",
    )

    # --------------------------------------------------
    # Treasury primary-source curves
    # --------------------------------------------------

    treasury_series_rows = defaultdict(
        list
    )

    source_manifest = {}

    for dataset in (
        "daily_treasury_yield_curve",
        "daily_treasury_real_yield_curve",
    ):
        relevant = [
            row
            for row in series
            if (
                row.get(
                    "source_type"
                )
                == "us_treasury"
                and row.get(
                    "source_dataset"
                )
                == dataset
            )
        ]

        wanted = {
            row[
                "source_field"
            ]
            for row in relevant
        }

        for year in years:
            url = treasury_url(
                year=year,
                dataset=dataset,
            )

            print()
            print(
                "Treasury:",
                dataset,
                year,
            )

            raw = curl_bytes(
                url,
                timeout=args.timeout,
            )

            retrieved_at = (
                datetime.now(
                    tz
                ).isoformat()
            )

            snapshot_sha = (
                sha_bytes(raw)
            )

            parsed = (
                parse_treasury_csv(
                    raw,
                    wanted_fields=
                        wanted,
                    source_url=
                        url,
                    snapshot_sha=
                        snapshot_sha,
                    retrieved_at=
                        retrieved_at,
                )
            )

            source_manifest[
                f"treasury__{dataset}__{year}"
            ] = {
                "source_type":
                    "us_treasury",

                "dataset":
                    dataset,

                "year":
                    year,

                "url":
                    url,

                "retrieved_at":
                    retrieved_at,

                "source_snapshot_sha256":
                    snapshot_sha,
            }

            for meta in relevant:
                field = meta[
                    "source_field"
                ]

                treasury_series_rows[
                    meta[
                        "series_id"
                    ]
                ].extend(
                    parsed[field]
                )

    series_observations = {}

    for meta in series:
        if (
            meta.get(
                "source_type"
            )
            != "us_treasury"
        ):
            continue

        rows = [
            row
            for row in treasury_series_rows[
                meta[
                    "series_id"
                ]
            ]
            if (
                cutoff_date
                <= row["date"]
                <= maximum_date
            )
        ]

        rows.sort(
            key=lambda row:
                row["date"]
        )

        observations = [
            make_treasury_observation(
                meta=meta,
                row=row,
            )
            for row in rows
        ]

        if not observations:
            raise RuntimeError(
                f"No Treasury observations "
                f"for {meta['series_id']}"
            )

        series_observations[
            meta[
                "series_id"
            ]
        ] = observations

        latest = observations[-1]

        print(
            f"{meta['series_id']:18} "
            f"{latest['observation_date']} "
            f"= {latest['value']}"
        )

    # --------------------------------------------------
    # Derived synchronized breakevens
    # --------------------------------------------------

    for series_id in (
        "breakeven_5y",
        "breakeven_10y",
    ):
        meta = metadata[
            series_id
        ]

        derivation = meta[
            "derivation"
        ]

        observations = (
            derive_breakeven(
                meta=meta,
                nominal=
                    series_observations[
                        derivation[
                            "nominal"
                        ]
                    ],
                real=
                    series_observations[
                        derivation[
                            "real"
                        ]
                    ],
            )
        )

        if not observations:
            raise RuntimeError(
                f"No derived observations "
                f"for {series_id}"
            )

        series_observations[
            series_id
        ] = observations

        latest = observations[-1]

        print(
            f"{series_id:18} "
            f"{latest['observation_date']} "
            f"= {latest['value']}"
        )

    # --------------------------------------------------
    # Remaining FRED series
    # --------------------------------------------------

    for meta in series:
        if (
            meta.get(
                "source_type"
            )
            != "fred"
        ):
            continue

        source_series = (
            meta[
                "source_series"
            ]
        )

        print()
        print(
            "FRED:",
            meta[
                "series_id"
            ],
            f"({source_series})",
        )

        fetched = fetch_fred_csv(
            source_series,
            observation_start=
                cutoff_date.isoformat(),
            observation_end=
                maximum_date.isoformat(),
            timeout=args.timeout,
        )

        retrieved_at = (
            datetime.now(
                tz
            )
        )

        rows = parse_fred_csv(
            fetched["raw"],
            source_series=
                source_series,
        )

        observations = (
            build_observations(
                series_meta=meta,
                rows=rows,
                cutoff_date=
                    cutoff_date,
                maximum_date=
                    maximum_date,
                retrieved_at=
                    retrieved_at,
                fetch_url=
                    fetched["url"],
                fetch_sha256=
                    fetched[
                        "sha256"
                    ],
            )
        )

        series_observations[
            meta[
                "series_id"
            ]
        ] = observations

        source_manifest[
            f"fred__{meta['series_id']}"
        ] = {
            "source_type":
                "fred",

            "source_series":
                source_series,

            "url":
                fetched[
                    "url"
                ],

            "retrieved_at":
                retrieved_at.isoformat(),

            "source_snapshot_sha256":
                fetched[
                    "sha256"
                ],
        }

        latest = observations[-1]

        print(
            "   latest:",
            latest[
                "observation_date"
            ],
            "=",
            latest[
                "value"
            ],
        )

    # --------------------------------------------------
    # Assemble state
    # --------------------------------------------------

    all_observations = []

    latest_summary = {}

    for meta in series:
        series_id = (
            meta[
                "series_id"
            ]
        )

        observations = (
            series_observations[
                series_id
            ]
        )

        all_observations.extend(
            observations
        )

        latest = observations[-1]

        latest_summary[
            series_id
        ] = {
            "latest_observation_date":
                latest[
                    "observation_date"
                ],

            "latest_value":
                latest[
                    "value"
                ],

            "observations":
                len(
                    observations
                ),

            "source_type":
                meta[
                    "source_type"
                ],
        }

    as_of = datetime.now(
        tz
    )

    created_at = datetime.now(
        tz
    )

    state_id = (
        "cross_asset_core_v0_2__"
        + as_of.strftime(
            "%Y%m%dT%H%M%S"
        )
    )

    state = {
        "schema_version":
            "macro_market_state_v0_1",

        "state_id":
            state_id,

        "as_of_timestamp":
            as_of.isoformat(),

        "created_at":
            created_at.isoformat(),

        "series":
            series,

        "observations":
            all_observations,
    }

    validate_market_state(
        state
    )

    output_dir = (
        OUTPUT_ROOT
        / state_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    state_path = (
        output_dir
        / "market_state.json"
    )

    source_path = (
        output_dir
        / "source_fetch_manifest.json"
    )

    manifest_path = (
        output_dir
        / "market_state_manifest.json"
    )

    write_json(
        state_path,
        state,
    )

    write_json(
        source_path,
        source_manifest,
    )

    manifest = {
        "schema_version":
            "macro_market_state_build_v0_2",

        "state_id":
            state_id,

        "registry_id":
            registry[
                "registry_id"
            ],

        "registry_sha256":
            sha_path(
                REGISTRY_PATH
            ),

        "lookback_days":
            args.lookback_days,

        "as_of_timestamp":
            state[
                "as_of_timestamp"
            ],

        "series_count":
            len(series),

        "observation_count":
            len(
                all_observations
            ),

        "latest_by_series":
            latest_summary,

        "rate_source":
            (
                "U.S. Department "
                "of the Treasury"
            ),

        "breakeven_method":
            (
                "same-date Treasury nominal "
                "minus Treasury real yield"
            ),

        "remaining_market_source":
            "FRED",

        "market_state_file":
            state_path.name,

        "market_state_sha256":
            sha_path(
                state_path
            ),

        "source_fetch_manifest":
            source_path.name,

        "source_fetch_manifest_sha256":
            sha_path(
                source_path
            ),
    }

    write_json(
        manifest_path,
        manifest,
    )

    write_json(
        OUTPUT_ROOT
        / "latest_run.json",
        {
            "state_id":
                state_id,

            "market_state_path":
                str(
                    state_path
                ),

            "market_state_sha256":
                manifest[
                    "market_state_sha256"
                ],
        },
    )

    print()
    print("=" * 80)
    print(
        "MARKET STATE v0.2 VALID AND WRITTEN"
    )
    print("=" * 80)

    print(
        "State:",
        state_id,
    )

    print(
        "Series:",
        len(series),
    )

    print(
        "Observations:",
        len(
            all_observations
        ),
    )

    print(
        "SHA:",
        manifest[
            "market_state_sha256"
        ],
    )

    print(
        "Output:",
        output_dir,
    )


if __name__ == "__main__":
    main()
