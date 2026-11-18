#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import statistics

from collections import defaultdict
from datetime import datetime
from pathlib import Path


try:
    from benchmarks.market_feature_contract_v0_1 import (
        validate_market_features,
    )
except ModuleNotFoundError:
    from market_feature_contract_v0_1 import (
        validate_market_features,
    )


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_STATE_POINTER = (
    ROOT
    / "benchmarks"
    / "results"
    / "market_states"
    / "fred_cross_asset_core_v0_1"
    / "latest_run.json"
)

DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "benchmarks"
    / "results"
    / "market_features"
    / "fred_cross_asset_core_v0_1"
)

HORIZONS = {
    "obs_1": 1,
    "obs_5": 5,
    "obs_20": 20,
    "obs_63": 63,
    "obs_252": 252,
}

BP_VALUE_TYPES = {
    "yield",
    "real_yield",
    "breakeven_rate",
    "option_adjusted_spread",
}

PCT_VALUE_TYPES = {
    "fx_rate",
    "commodity_price",
    "index_level",
}


def sha256_path(path: Path):
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
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def transform_for_series(
    meta: dict,
):
    value_type = meta[
        "value_type"
    ]

    if value_type in BP_VALUE_TYPES:
        return "bp_change"

    if value_type in PCT_VALUE_TYPES:
        return "pct_return"

    raise ValueError(
        f"Unsupported value_type: "
        f"{value_type}"
    )


def transformed_change(
    *,
    transform: str,
    latest: float,
    prior: float,
):
    raw_delta = (
        latest - prior
    )

    if transform == "bp_change":
        return (
            raw_delta,
            raw_delta * 100.0,
            "basis_points",
        )

    if transform == "pct_return":
        if prior == 0:
            raise ZeroDivisionError(
                "Cannot calculate return "
                "from zero prior value."
            )

        return (
            raw_delta,
            (
                latest / prior
                - 1.0
            )
            * 100.0,
            "percent",
        )

    raise ValueError(
        transform
    )


def horizon_feature(
    *,
    observations: list,
    offset: int,
    transform: str,
):
    latest = observations[-1]

    if len(observations) <= offset:
        return {
            "available":
                False,

            "observation_offset":
                offset,

            "prior_observation_date":
                None,

            "prior_value":
                None,

            "latest_value":
                latest["value"],

            "raw_delta":
                None,

            "feature_value":
                None,

            "feature_units":
                (
                    "basis_points"
                    if transform
                    == "bp_change"
                    else "percent"
                ),

            "reason":
                "insufficient_history",
        }

    prior = observations[
        -(offset + 1)
    ]

    raw_delta, value, units = (
        transformed_change(
            transform=transform,
            latest=float(
                latest["value"]
            ),
            prior=float(
                prior["value"]
            ),
        )
    )

    return {
        "available":
            True,

        "observation_offset":
            offset,

        "prior_observation_date":
            prior[
                "observation_date"
            ],

        "prior_value":
            float(
                prior["value"]
            ),

        "latest_value":
            float(
                latest["value"]
            ),

        "raw_delta":
            raw_delta,

        "feature_value":
            value,

        "feature_units":
            units,

        "reason":
            None,
    }


def zscore(
    values: list[float],
):
    if len(values) < 20:
        return None

    window = values[-20:]

    mean = statistics.fmean(
        window
    )

    std = statistics.pstdev(
        window
    )

    if std == 0:
        return 0.0

    return (
        values[-1] - mean
    ) / std


def percentile_252(
    values: list[float],
):
    if len(values) < 252:
        return None

    window = values[-252:]

    latest = window[-1]

    count = sum(
        1
        for value in window
        if value <= latest
    )

    return (
        count
        / len(window)
        * 100.0
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--state",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    args = parser.parse_args()

    if args.state is None:
        pointer = json.loads(
            DEFAULT_STATE_POINTER.read_text()
        )

        state_path = Path(
            pointer[
                "market_state_path"
            ]
        )

    else:
        state_path = (
            args.state.resolve()
        )

    state = json.loads(
        state_path.read_text()
    )

    state_sha = sha256_path(
        state_path
    )

    as_of = datetime.fromisoformat(
        state[
            "as_of_timestamp"
        ]
    )

    series_meta = {
        row["series_id"]:
            row
        for row in state["series"]
    }

    grouped = defaultdict(
        list
    )

    for obs in state[
        "observations"
    ]:
        if (
            obs["status"]
            == "observed"
        ):
            grouped[
                obs["series_id"]
            ].append(
                obs
            )

    rows = []

    for series_id, meta in (
        series_meta.items()
    ):
        observations = sorted(
            grouped[
                series_id
            ],
            key=lambda row:
                row[
                    "observation_date"
                ],
        )

        if not observations:
            raise RuntimeError(
                f"No observations for "
                f"{series_id}"
            )

        latest = observations[-1]

        latest_date = (
            datetime.fromisoformat(
                latest[
                    "observation_date"
                ]
            ).date()
        )

        transform = (
            transform_for_series(
                meta
            )
        )

        horizons = {
            name:
                horizon_feature(
                    observations=
                        observations,
                    offset=offset,
                    transform=
                        transform,
                )
            for name, offset
            in HORIZONS.items()
        }

        values = [
            float(
                row["value"]
            )
            for row in observations
        ]

        rows.append({
            "series_id":
                series_id,

            "metric_name":
                meta[
                    "metric_name"
                ],

            "display_name":
                meta[
                    "display_name"
                ],

            "family":
                meta[
                    "family"
                ],

            "source_series":
                meta[
                    "source_series"
                ],

            "value_type":
                meta[
                    "value_type"
                ],

            "units":
                meta[
                    "units"
                ],

            "transform":
                transform,

            "history_count":
                len(
                    observations
                ),

            "latest_value":
                float(
                    latest[
                        "value"
                    ]
                ),

            "latest_observation_date":
                latest[
                    "observation_date"
                ],

            "age_days":
                (
                    as_of.date()
                    - latest_date
                ).days,

            "horizons":
                horizons,

            "statistics": {
                "zscore_20":
                    zscore(
                        values
                    ),

                "percentile_252":
                    percentile_252(
                        values
                    ),

                "zscore_method":
                    (
                        "population z-score "
                        "over latest 20 "
                        "observations"
                    ),

                "percentile_method":
                    (
                        "empirical percentile "
                        "rank over latest 252 "
                        "observations"
                    ),
            },
        })

    payload = {
        "schema_version":
            "macro_market_features_v0_1",

        "source_state_id":
            state[
                "state_id"
            ],

        "source_state_sha256":
            state_sha,

        "as_of_timestamp":
            state[
                "as_of_timestamp"
            ],

        "feature_policy": {
            "horizons":
                HORIZONS,

            "rate_transform":
                "basis-point change",

            "price_transform":
                "percentage return",

            "insufficient_history":
                (
                    "null; no substitute "
                    "horizon"
                ),
        },

        "series_features":
            rows,
    }

    validate_market_features(
        payload
    )

    output_dir = (
        args.output_root.resolve()
        / state[
            "state_id"
        ]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_path = (
        output_dir
        / "market_features.json"
    )

    write_json(
        feature_path,
        payload,
    )

    manifest = {
        "schema_version":
            "macro_market_features_build_v0_1",

        "source_state_id":
            state[
                "state_id"
            ],

        "source_state_sha256":
            state_sha,

        "series_count":
            len(rows),

        "feature_file":
            feature_path.name,

        "feature_sha256":
            sha256_path(
                feature_path
            ),
    }

    write_json(
        output_dir
        / "market_features_manifest.json",
        manifest,
    )

    write_json(
        args.output_root.resolve()
        / "latest_run.json",
        {
            "source_state_id":
                state[
                    "state_id"
                ],

            "market_features_path":
                str(
                    feature_path
                ),

            "market_features_sha256":
                manifest[
                    "feature_sha256"
                ],
        },
    )

    print("=" * 80)
    print(
        "MARKET FEATURES BUILT"
    )
    print("=" * 80)

    print(
        "State:",
        state[
            "state_id"
        ],
    )

    print(
        "Series:",
        len(rows),
    )

    print(
        "SHA:",
        manifest[
            "feature_sha256"
        ],
    )

    print(
        "Output:",
        feature_path,
    )


if __name__ == "__main__":
    main()
