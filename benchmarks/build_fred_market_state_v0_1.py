#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import time
import urllib.parse

from datetime import (
    datetime,
    timedelta,
)
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    from benchmarks.market_state_contract_v0_1 import (
        validate_market_state,
    )
except ModuleNotFoundError:
    from market_state_contract_v0_1 import (
        validate_market_state,
    )


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REGISTRY = (
    ROOT
    / "benchmarks"
    / "fixtures"
    / "market_state_universe_v0_1"
    / "series_registry.json"
)

DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "benchmarks"
    / "results"
    / "market_states"
    / "fred_cross_asset_core_v0_1"
)

FRED_CSV_URL = (
    "https://fred.stlouisfed.org/"
    "graph/fredgraph.csv"
)

MISSING_VALUES = {
    "",
    ".",
    "NA",
    "NaN",
    "nan",
}


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


def sha256_bytes(
    payload: bytes,
) -> str:
    return hashlib.sha256(
        payload
    ).hexdigest()


def sha256_path(
    path: Path,
) -> str:
    return sha256_bytes(
        path.read_bytes()
    )


def fetch_fred_csv(
    source_series: str,
    *,
    observation_start: str,
    observation_end: str,
    timeout: int,
    attempts: int = 3,
):
    query = urllib.parse.urlencode({
        "id":
            source_series,

        "cosd":
            observation_start,

        "coed":
            observation_end,
    })

    url = (
        FRED_CSV_URL
        + "?"
        + query
    )

    command = [
        "curl",
        "-fsSL",

        "--connect-timeout",
        "15",

        "--max-time",
        str(timeout),

        "--retry",
        str(attempts),

        "--retry-delay",
        "1",

        "--retry-all-errors",

        url,
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    except subprocess.CalledProcessError as exc:
        stderr = (
            exc.stderr.decode(
                "utf-8",
                errors="replace",
            )
            if exc.stderr
            else ""
        )

        raise RuntimeError(
            f"FRED curl fetch failed for "
            f"{source_series}: "
            f"{stderr.strip()}"
        ) from exc

    raw = result.stdout

    if not raw:
        raise RuntimeError(
            f"FRED returned empty payload "
            f"for {source_series}"
        )

    return {
        "url":
            url,

        "raw":
            raw,

        "sha256":
            sha256_bytes(
                raw
            ),
    }


def parse_fred_csv(
    raw: bytes,
    *,
    source_series: str,
):
    text = raw.decode(
        "utf-8-sig"
    )

    reader = csv.DictReader(
        io.StringIO(
            text
        )
    )

    fields = (
        reader.fieldnames
        or []
    )

    if not fields:
        raise RuntimeError(
            f"{source_series}: "
            "CSV has no header"
        )

    if (
        source_series
        in fields
    ):
        value_field = (
            source_series
        )
    else:
        # Defensive fallback in case
        # FRED changes the label.
        value_field = fields[-1]

    if (
        "observation_date"
        in fields
    ):
        date_field = (
            "observation_date"
        )

    elif "DATE" in fields:
        date_field = "DATE"

    else:
        date_field = fields[0]

    rows = []

    for row in reader:
        raw_date = (
            row.get(
                date_field,
                ""
            )
            or ""
        ).strip()

        raw_value = (
            row.get(
                value_field,
                ""
            )
            or ""
        ).strip()

        if (
            not raw_date
            or raw_value
            in MISSING_VALUES
        ):
            continue

        try:
            obs_date = (
                datetime.strptime(
                    raw_date,
                    "%Y-%m-%d",
                ).date()
            )

            value = float(
                raw_value.replace(
                    ",",
                    "",
                )
            )

        except ValueError:
            continue

        rows.append({
            "date":
                obs_date,

            "value":
                value,
        })

    if not rows:
        raise RuntimeError(
            f"{source_series}: "
            "no numeric observations"
        )

    rows.sort(
        key=lambda item:
            item["date"]
    )

    return rows


def build_observations(
    *,
    series_meta: dict,
    rows: list[dict],
    cutoff_date,
    maximum_date,
    retrieved_at: datetime,
    fetch_url: str,
    fetch_sha256: str,
):
    eligible = [
        row
        for row in rows
        if (
            cutoff_date
            <= row["date"]
            <= maximum_date
        )
    ]

    if not eligible:
        raise RuntimeError(
            f"{series_meta['series_id']}: "
            "no observations inside "
            "requested lookback"
        )

    retrieved_iso = (
        retrieved_at.isoformat()
    )

    observations = []

    for row in eligible:
        obs_date = (
            row["date"].isoformat()
        )

        observations.append({
            "observation_id":
                (
                    f"{series_meta['series_id']}"
                    f"__{obs_date}"
                ),

            "series_id":
                series_meta[
                    "series_id"
                ],

            "status":
                "observed",

            "value":
                row["value"],

            # Economic date represented
            # by the FRED observation.
            "observation_date":
                obs_date,

            "period_start":
                obs_date,

            "period_end":
                obs_date,

            # We have not yet implemented
            # release-specific timestamp
            # metadata for these feeds.
            "release_timestamp":
                None,

            # Conservative rule:
            # the data is definitely known
            # to our system by retrieval.
            "available_timestamp":
                retrieved_iso,

            "retrieved_at":
                retrieved_iso,

            "vintage": {
                "revision_policy":
                    series_meta[
                        "revision_policy"
                    ],

                "vintage_date":
                    retrieved_at.date(
                    ).isoformat(),
            },

            "unavailable_reason":
                None,

            # Extra provenance fields are
            # allowed by the v0.1 contract.
            "availability_method":
                (
                    "retrieval_time_"
                    "conservative"
                ),

            "source_url":
                fetch_url,

            "source_snapshot_sha256":
                fetch_sha256,
        })

    return observations


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--lookback-days",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--timezone",
        default="America/New_York",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
    )

    args = parser.parse_args()

    if args.lookback_days < 30:
        raise ValueError(
            "lookback-days must "
            "be at least 30"
        )

    registry_path = (
        args.registry.resolve()
    )

    registry = json.loads(
        registry_path.read_text(
            encoding="utf-8"
        )
    )

    series_registry = (
        registry["series"]
    )

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

    all_observations = []

    fetch_manifest = {}

    latest_summary = {}

    print("=" * 80)
    print(
        "BUILD FRED CROSS-ASSET MARKET STATE"
    )
    print("=" * 80)

    print(
        "Registry:",
        registry[
            "registry_id"
        ],
    )

    print(
        "Series:",
        len(
            series_registry
        ),
    )

    print(
        "Lookback:",
        args.lookback_days,
        "calendar days",
    )

    print()

    for index, meta in enumerate(
        series_registry,
        start=1,
    ):
        series_id = (
            meta[
                "series_id"
            ]
        )

        source_series = (
            meta[
                "source_series"
            ]
        )

        print(
            f"[{index:02d}/"
            f"{len(series_registry):02d}] "
            f"{series_id} "
            f"({source_series})"
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
                    fetched["sha256"],
            )
        )

        all_observations.extend(
            observations
        )

        latest = observations[-1]

        latest_summary[
            series_id
        ] = {
            "source_series":
                source_series,

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
        }

        fetch_manifest[
            series_id
        ] = {
            "source_series":
                source_series,

            "url":
                fetched[
                    "url"
                ],

            "source_snapshot_sha256":
                fetched[
                    "sha256"
                ],

            "retrieved_at":
                retrieved_at.isoformat(),

            "parsed_numeric_rows":
                len(rows),

            "retained_rows":
                len(
                    observations
                ),
        }

        print(
            "   latest:",
            latest[
                "observation_date"
            ],
            "=",
            latest[
                "value"
            ],
            "| retained:",
            len(
                observations
            ),
        )

    # State becomes "as of" only after
    # every source retrieval completed.
    as_of = datetime.now(
        tz
    )

    created_at = datetime.now(
        tz
    )

    state_id = (
        "fred_cross_asset_core_v0_1__"
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
            series_registry,

        "observations":
            all_observations,
    }

    validate_market_state(
        state
    )

    output_dir = (
        args.output_root.resolve()
        / state_id
    )

    if output_dir.exists():
        raise FileExistsError(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    state_path = (
        output_dir
        / "market_state.json"
    )

    fetch_path = (
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
        fetch_path,
        fetch_manifest,
    )

    manifest = {
        "schema_version":
            "macro_market_state_build_v0_1",

        "state_id":
            state_id,

        "registry_id":
            registry[
                "registry_id"
            ],

        "registry_sha256":
            sha256_path(
                registry_path
            ),

        "lookback_days":
            args.lookback_days,

        "timezone":
            args.timezone,

        "cutoff_date":
            cutoff_date.isoformat(),

        "as_of_timestamp":
            state[
                "as_of_timestamp"
            ],

        "created_at":
            state[
                "created_at"
            ],

        "series_count":
            len(
                series_registry
            ),

        "observation_count":
            len(
                all_observations
            ),

        "latest_by_series":
            latest_summary,

        "availability_policy":
            (
                "retrieval_time_conservative; "
                "no inferred historical release "
                "timestamps"
            ),

        "market_state_file":
            state_path.name,

        "market_state_sha256":
            sha256_path(
                state_path
            ),

        "source_fetch_manifest":
            fetch_path.name,

        "source_fetch_manifest_sha256":
            sha256_path(
                fetch_path
            ),
    }

    write_json(
        manifest_path,
        manifest,
    )

    latest_pointer = (
        args.output_root.resolve()
        / "latest_run.json"
    )

    write_json(
        latest_pointer,
        {
            "state_id":
                state_id,

            "output_dir":
                str(
                    output_dir
                ),

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
        "MARKET STATE VALID AND WRITTEN"
    )
    print("=" * 80)

    print(
        "State:",
        state_id,
    )

    print(
        "As-of:",
        state[
            "as_of_timestamp"
        ],
    )

    print(
        "Series:",
        len(
            series_registry
        ),
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
