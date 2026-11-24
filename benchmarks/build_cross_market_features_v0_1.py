#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import statistics

from collections import defaultdict
from pathlib import Path


try:
    from benchmarks.cross_market_feature_contract_v0_1 import (
        validate_cross_market_features,
    )
except ModuleNotFoundError:
    from cross_market_feature_contract_v0_1 import (
        validate_cross_market_features,
    )


ROOT = Path(__file__).resolve().parents[1]

STATE_POINTER = (
    ROOT
    / "benchmarks"
    / "results"
    / "market_states"
    / "fred_cross_asset_core_v0_1"
    / "latest_run.json"
)

FEATURE_POINTER = (
    ROOT
    / "benchmarks"
    / "results"
    / "market_features"
    / "fred_cross_asset_core_v0_1"
    / "latest_run.json"
)

DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "benchmarks"
    / "results"
    / "cross_market_features"
    / "fred_cross_asset_core_v0_1"
)

HORIZONS = {
    "obs_1": 1,
    "obs_5": 5,
    "obs_20": 20,
}


SPREAD_SPECS = [
    {
        "feature_id":
            "nominal_2s10s",

        "display_name":
            "2s10s nominal Treasury curve",

        "family":
            "curve_geometry",

        # 10Y - 2Y
        "coefficients": {
            "ust_10y": 1.0,
            "ust_2y": -1.0,
        },
    },

    {
        "feature_id":
            "nominal_5s30s",

        "display_name":
            "5s30s nominal Treasury curve",

        "family":
            "curve_geometry",

        # 30Y - 5Y
        "coefficients": {
            "ust_30y": 1.0,
            "ust_5y": -1.0,
        },
    },

    {
        "feature_id":
            "real_5s10s",

        "display_name":
            "5s10s real-yield curve",

        "family":
            "real_curve_geometry",

        # 10Y real - 5Y real
        "coefficients": {
            "tips_10y_real": 1.0,
            "tips_5y_real": -1.0,
        },
    },

    {
        "feature_id":
            "breakeven_5s10s",

        "display_name":
            "5s10s breakeven curve",

        "family":
            "inflation_curve_geometry",

        # 10Y BE - 5Y BE
        "coefficients": {
            "breakeven_10y": 1.0,
            "breakeven_5y": -1.0,
        },
    },
]


DECOMPOSITION_SPECS = [
    {
        "feature_id":
            "nominal_real_be_5y",

        "display_name":
            "5Y nominal / real / breakeven decomposition",

        "tenor":
            "5y",

        "components": {
            "nominal":
                "ust_5y",

            "real":
                "tips_5y_real",

            "breakeven":
                "breakeven_5y",
        },
    },

    {
        "feature_id":
            "nominal_real_be_10y",

        "display_name":
            "10Y nominal / real / breakeven decomposition",

        "tenor":
            "10y",

        "components": {
            "nominal":
                "ust_10y",

            "real":
                "tips_10y_real",

            "breakeven":
                "breakeven_10y",
        },
    },
]


def sha256_path(
    path: Path,
):
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


def common_dates(
    value_maps: dict,
    series_ids: list[str],
):
    sets = [
        set(
            value_maps[
                series_id
            ]
        )
        for series_id in series_ids
    ]

    if not sets:
        return []

    return sorted(
        set.intersection(
            *sets
        )
    )


def spread_value_bp(
    *,
    date: str,
    value_maps: dict,
    coefficients: dict,
):
    value_percent = sum(
        coefficient
        * value_maps[
            series_id
        ][date]
        for (
            series_id,
            coefficient,
        )
        in coefficients.items()
    )

    return (
        value_percent
        * 100.0
    )


def zscore_20(
    values: list[float],
):
    if len(values) < 20:
        return None

    window = values[-20:]

    std = statistics.pstdev(
        window
    )

    if std == 0:
        return 0.0

    return (
        values[-1]
        - statistics.fmean(
            window
        )
    ) / std


def percentile_252(
    values: list[float],
):
    if len(values) < 252:
        return None

    window = values[-252:]

    latest = window[-1]

    return (
        sum(
            1
            for value in window
            if value <= latest
        )
        / len(window)
        * 100.0
    )


def build_spread_feature(
    *,
    spec: dict,
    value_maps: dict,
):
    coefficients = (
        spec[
            "coefficients"
        ]
    )

    series_ids = list(
        coefficients
    )

    dates = common_dates(
        value_maps,
        series_ids,
    )

    if not dates:
        raise RuntimeError(
            f"No common observations for "
            f"{spec['feature_id']}"
        )

    levels = [
        spread_value_bp(
            date=date,
            value_maps=
                value_maps,
            coefficients=
                coefficients,
        )
        for date in dates
    ]

    latest_date = dates[-1]

    latest_value = (
        levels[-1]
    )

    horizons = {}

    for name, offset in (
        HORIZONS.items()
    ):
        if len(dates) <= offset:
            horizons[name] = {
                "available":
                    False,

                "observation_offset":
                    offset,

                "prior_common_date":
                    None,

                "latest_common_date":
                    latest_date,

                "prior_value_bp":
                    None,

                "latest_value_bp":
                    latest_value,

                "change_bp":
                    None,
            }

            continue

        prior_date = dates[
            -(offset + 1)
        ]

        prior_value = (
            levels[
                -(offset + 1)
            ]
        )

        horizons[name] = {
            "available":
                True,

            "observation_offset":
                offset,

            "prior_common_date":
                prior_date,

            "latest_common_date":
                latest_date,

            "prior_value_bp":
                prior_value,

            "latest_value_bp":
                latest_value,

            "change_bp":
                (
                    latest_value
                    - prior_value
                ),
        }

    return {
        "feature_id":
            spec[
                "feature_id"
            ],

        "display_name":
            spec[
                "display_name"
            ],

        "family":
            spec[
                "family"
            ],

        "component_series":
            series_ids,

        "coefficients":
            coefficients,

        "latest_common_date":
            latest_date,

        "common_history_count":
            len(dates),

        "latest_value_bp":
            latest_value,

        "horizons":
            horizons,

        "statistics": {
            "zscore_20":
                zscore_20(
                    levels
                ),

            "percentile_252":
                percentile_252(
                    levels
                ),
        },
    }


def decomposition_values(
    *,
    date: str,
    value_maps: dict,
    components: dict,
):
    nominal = value_maps[
        components[
            "nominal"
        ]
    ][date]

    real = value_maps[
        components[
            "real"
        ]
    ][date]

    breakeven = value_maps[
        components[
            "breakeven"
        ]
    ][date]

    gap_bp = (
        nominal
        - real
        - breakeven
    ) * 100.0

    return {
        "nominal_percent":
            nominal,

        "real_percent":
            real,

        "breakeven_percent":
            breakeven,

        "identity_gap_bp":
            gap_bp,
    }


def build_decomposition_feature(
    *,
    spec: dict,
    value_maps: dict,
):
    components = (
        spec[
            "components"
        ]
    )

    series_ids = list(
        components.values()
    )

    dates = common_dates(
        value_maps,
        series_ids,
    )

    if not dates:
        raise RuntimeError(
            f"No common observations for "
            f"{spec['feature_id']}"
        )

    latest_date = (
        dates[-1]
    )

    latest = (
        decomposition_values(
            date=latest_date,
            value_maps=
                value_maps,
            components=
                components,
        )
    )

    horizons = {}

    for name, offset in (
        HORIZONS.items()
    ):
        if len(dates) <= offset:
            horizons[name] = {
                "available":
                    False,

                "observation_offset":
                    offset,

                "prior_common_date":
                    None,

                "latest_common_date":
                    latest_date,

                "nominal_change_bp":
                    None,

                "real_change_bp":
                    None,

                "breakeven_change_bp":
                    None,

                "change_identity_gap_bp":
                    None,

                "prior_identity_gap_bp":
                    None,

                "latest_identity_gap_bp":
                    latest[
                        "identity_gap_bp"
                    ],
            }

            continue

        prior_date = dates[
            -(offset + 1)
        ]

        prior = (
            decomposition_values(
                date=prior_date,
                value_maps=
                    value_maps,
                components=
                    components,
            )
        )

        nominal_change = (
            latest[
                "nominal_percent"
            ]
            - prior[
                "nominal_percent"
            ]
        ) * 100.0

        real_change = (
            latest[
                "real_percent"
            ]
            - prior[
                "real_percent"
            ]
        ) * 100.0

        breakeven_change = (
            latest[
                "breakeven_percent"
            ]
            - prior[
                "breakeven_percent"
            ]
        ) * 100.0

        horizons[name] = {
            "available":
                True,

            "observation_offset":
                offset,

            "prior_common_date":
                prior_date,

            "latest_common_date":
                latest_date,

            "nominal_change_bp":
                nominal_change,

            "real_change_bp":
                real_change,

            "breakeven_change_bp":
                breakeven_change,

            "change_identity_gap_bp":
                (
                    nominal_change
                    - real_change
                    - breakeven_change
                ),

            "prior_identity_gap_bp":
                prior[
                    "identity_gap_bp"
                ],

            "latest_identity_gap_bp":
                latest[
                    "identity_gap_bp"
                ],
        }

    return {
        "feature_id":
            spec[
                "feature_id"
            ],

        "display_name":
            spec[
                "display_name"
            ],

        "family":
            "yield_decomposition",

        "tenor":
            spec[
                "tenor"
            ],

        "component_series":
            components,

        "latest_common_date":
            latest_date,

        "common_history_count":
            len(dates),

        "latest":
            latest,

        "horizons":
            horizons,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--state",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--series-features",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    args = parser.parse_args()

    output_root = (
        args.output_root.resolve()
    )

    if args.state is None:
        state_pointer = json.loads(
            STATE_POINTER.read_text()
        )

        state_path = Path(
            state_pointer[
                "market_state_path"
            ]
        )

    else:
        state_path = (
            args.state.resolve()
        )

    if args.series_features is None:
        feature_pointer = json.loads(
            FEATURE_POINTER.read_text()
        )

        series_feature_path = Path(
            feature_pointer[
                "market_features_path"
            ]
        )

    else:
        series_feature_path = (
            args.series_features.resolve()
        )

    state = json.loads(
        state_path.read_text()
    )

    series_features = json.loads(
        series_feature_path.read_text()
    )

    if (
        series_features[
            "source_state_id"
        ]
        != state[
            "state_id"
        ]
    ):
        raise RuntimeError(
            "Series-feature artifact "
            "does not match market state."
        )

    value_maps = defaultdict(
        dict
    )

    for obs in state[
        "observations"
    ]:
        if (
            obs[
                "status"
            ]
            != "observed"
        ):
            continue

        value_maps[
            obs[
                "series_id"
            ]
        ][
            obs[
                "observation_date"
            ]
        ] = float(
            obs[
                "value"
            ]
        )

    spread_features = [
        build_spread_feature(
            spec=spec,
            value_maps=value_maps,
        )
        for spec in SPREAD_SPECS
    ]

    decomposition_features = [
        build_decomposition_feature(
            spec=spec,
            value_maps=value_maps,
        )
        for spec
        in DECOMPOSITION_SPECS
    ]

    payload = {
        "schema_version":
            "macro_cross_market_features_v0_1",

        "source_state_id":
            state[
                "state_id"
            ],

        "source_state_sha256":
            sha256_path(
                state_path
            ),

        "source_series_features_sha256":
            sha256_path(
                series_feature_path
            ),

        "as_of_timestamp":
            state[
                "as_of_timestamp"
            ],

        "alignment_policy": {
            "method":
                "intersection_of_observation_dates",

            "rule":
                (
                    "Every cross-series level and "
                    "change uses dates present in all "
                    "component series."
                ),

            "horizons":
                HORIZONS,
        },

        "spread_features":
            spread_features,

        "decomposition_features":
            decomposition_features,
    }

    validate_cross_market_features(
        payload
    )

    output_dir = (
        output_root
        / state[
            "state_id"
        ]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "cross_market_features.json"
    )

    write_json(
        output_path,
        payload,
    )

    manifest = {
        "schema_version":
            "macro_cross_market_features_build_v0_1",

        "source_state_id":
            state[
                "state_id"
            ],

        "feature_file":
            output_path.name,

        "feature_sha256":
            sha256_path(
                output_path
            ),

        "spread_feature_count":
            len(
                spread_features
            ),

        "decomposition_feature_count":
            len(
                decomposition_features
            ),
    }

    write_json(
        output_dir
        / "cross_market_features_manifest.json",
        manifest,
    )

    write_json(
        output_root
        / "latest_run.json",
        {
            "source_state_id":
                state[
                    "state_id"
                ],

            "cross_market_features_path":
                str(
                    output_path
                ),

            "cross_market_features_sha256":
                manifest[
                    "feature_sha256"
                ],
        },
    )

    print("=" * 80)
    print(
        "CROSS-MARKET FEATURES BUILT"
    )
    print("=" * 80)

    print(
        "State:",
        state[
            "state_id"
        ],
    )

    print(
        "Spreads:",
        len(
            spread_features
        ),
    )

    print(
        "Decompositions:",
        len(
            decomposition_features
        ),
    )

    print(
        "SHA:",
        manifest[
            "feature_sha256"
        ],
    )

    print(
        "Output:",
        output_path,
    )


if __name__ == "__main__":
    main()
