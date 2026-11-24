#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SERIES_POINTER = (
    ROOT
    / "benchmarks"
    / "results"
    / "market_features"
    / "fred_cross_asset_core_v0_1"
    / "latest_run.json"
)

DEFAULT_CROSS_POINTER = (
    ROOT
    / "benchmarks"
    / "results"
    / "cross_market_features"
    / "fred_cross_asset_core_v0_1"
    / "latest_run.json"
)

DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "benchmarks"
    / "results"
    / "macro_state_packets"
    / "fred_cross_asset_core_v0_1"
)


def sha256_path(path: Path) -> str:
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


def value_or_none(
    feature: dict,
    field: str,
):
    if not feature.get(
        "available"
    ):
        return None

    return feature.get(
        field
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--series-features",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--cross-market-features",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    args = parser.parse_args()

    if args.series_features is None:
        series_pointer = json.loads(
            DEFAULT_SERIES_POINTER.read_text()
        )

        series_path = Path(
            series_pointer[
                "market_features_path"
            ]
        )
    else:
        series_path = (
            args.series_features.resolve()
        )

    if args.cross_market_features is None:
        cross_pointer = json.loads(
            DEFAULT_CROSS_POINTER.read_text()
        )

        cross_path = Path(
            cross_pointer[
                "cross_market_features_path"
            ]
        )
    else:
        cross_path = (
            args.cross_market_features.resolve()
        )

    output_root = (
        args.output_root.resolve()
    )

    series_payload = json.loads(
        series_path.read_text()
    )

    cross_payload = json.loads(
        cross_path.read_text()
    )

    state_id = series_payload[
        "source_state_id"
    ]

    if (
        cross_payload[
            "source_state_id"
        ]
        != state_id
    ):
        raise RuntimeError(
            "Series and cross-market "
            "features use different states."
        )

    # --------------------------------------------------
    # Per-series state
    # --------------------------------------------------

    series_state = []

    for row in series_payload[
        "series_features"
    ]:
        horizons = row[
            "horizons"
        ]

        statistics = row[
            "statistics"
        ]

        series_state.append({
            "series_id":
                row[
                    "series_id"
                ],

            "display_name":
                row[
                    "display_name"
                ],

            "family":
                row[
                    "family"
                ],

            "latest_value":
                row[
                    "latest_value"
                ],

            "units":
                row[
                    "units"
                ],

            "latest_observation_date":
                row[
                    "latest_observation_date"
                ],

            "age_days":
                row[
                    "age_days"
                ],

            "transform":
                row[
                    "transform"
                ],

            "change_1obs":
                value_or_none(
                    horizons[
                        "obs_1"
                    ],
                    "feature_value",
                ),

            "change_5obs":
                value_or_none(
                    horizons[
                        "obs_5"
                    ],
                    "feature_value",
                ),

            "change_20obs":
                value_or_none(
                    horizons[
                        "obs_20"
                    ],
                    "feature_value",
                ),

            "change_63obs":
                value_or_none(
                    horizons[
                        "obs_63"
                    ],
                    "feature_value",
                ),

            "change_252obs":
                value_or_none(
                    horizons[
                        "obs_252"
                    ],
                    "feature_value",
                ),

            "zscore_20":
                statistics[
                    "zscore_20"
                ],

            "percentile_252":
                statistics[
                    "percentile_252"
                ],
        })

    # --------------------------------------------------
    # Curve geometry
    # --------------------------------------------------

    curves = []

    for row in cross_payload[
        "spread_features"
    ]:
        h = row[
            "horizons"
        ]

        curves.append({
            "feature_id":
                row[
                    "feature_id"
                ],

            "display_name":
                row[
                    "display_name"
                ],

            "common_date":
                row[
                    "latest_common_date"
                ],

            "level_bp":
                row[
                    "latest_value_bp"
                ],

            "change_1obs_bp":
                value_or_none(
                    h[
                        "obs_1"
                    ],
                    "change_bp",
                ),

            "change_5obs_bp":
                value_or_none(
                    h[
                        "obs_5"
                    ],
                    "change_bp",
                ),

            "change_20obs_bp":
                value_or_none(
                    h[
                        "obs_20"
                    ],
                    "change_bp",
                ),

            "zscore_20":
                row[
                    "statistics"
                ][
                    "zscore_20"
                ],

            "percentile_252":
                row[
                    "statistics"
                ][
                    "percentile_252"
                ],
        })

    # --------------------------------------------------
    # Nominal / real / breakeven decompositions
    # --------------------------------------------------

    decompositions = []

    for row in cross_payload[
        "decomposition_features"
    ]:
        h = row[
            "horizons"
        ]

        decomposition = {
            "feature_id":
                row[
                    "feature_id"
                ],

            "display_name":
                row[
                    "display_name"
                ],

            "common_date":
                row[
                    "latest_common_date"
                ],

            "latest":
                row[
                    "latest"
                ],

            "horizons":
                {},
        }

        for horizon in (
            "obs_1",
            "obs_5",
            "obs_20",
        ):
            item = h[
                horizon
            ]

            if not item[
                "available"
            ]:
                decomposition[
                    "horizons"
                ][horizon] = None

                continue

            decomposition[
                "horizons"
            ][horizon] = {
                "prior_common_date":
                    item[
                        "prior_common_date"
                    ],

                "latest_common_date":
                    item[
                        "latest_common_date"
                    ],

                "nominal_change_bp":
                    item[
                        "nominal_change_bp"
                    ],

                "real_change_bp":
                    item[
                        "real_change_bp"
                    ],

                "breakeven_change_bp":
                    item[
                        "breakeven_change_bp"
                    ],

                "identity_gap_bp":
                    item[
                        "change_identity_gap_bp"
                    ],
            }

        decompositions.append(
            decomposition
        )

    packet = {
        "schema_version":
            "macro_state_packet_v0_1",

        "source_state_id":
            state_id,

        "as_of_timestamp":
            series_payload[
                "as_of_timestamp"
            ],

        "source_artifacts": {
            "series_features_sha256":
                sha256_path(
                    series_path
                ),

            "cross_market_features_sha256":
                sha256_path(
                    cross_path
                ),
        },

        "representation_policy": {
            "interpretation":
                "none",

            "series_horizons":
                (
                    "observation-count horizons; "
                    "dates retained in source artifacts"
                ),

            "cross_series_alignment":
                (
                    "intersection of common "
                    "observation dates"
                ),

            "null_policy":
                (
                    "insufficient history remains "
                    "null; no substituted horizon"
                ),
        },

        "series_state":
            series_state,

        "curve_geometry":
            curves,

        "yield_decompositions":
            decompositions,
    }

    output_dir = (
        output_root
        / state_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_dir
        / "macro_state_packet.json"
    )

    write_json(
        json_path,
        packet,
    )

    # --------------------------------------------------
    # Deterministic model-visible text representation
    # --------------------------------------------------

    lines = []

    lines.append(
        "# Cross-Asset Macro State"
    )

    lines.append("")

    lines.append(
        f"As of: "
        f"{packet['as_of_timestamp']}"
    )

    lines.append("")

    lines.append(
        "All values below are deterministic "
        "source-derived features. No narrative "
        "interpretation is embedded in this packet."
    )

    lines.append("")

    lines.append(
        "## Series State"
    )

    lines.append("")

    for row in series_state:
        transform_units = (
            "bp"
            if row[
                "transform"
            ]
            == "bp_change"
            else "%"
        )

        percentile = (
            "NA"
            if row[
                "percentile_252"
            ]
            is None
            else (
                f"{row['percentile_252']:.1f}"
            )
        )

        zscore = (
            "NA"
            if row[
                "zscore_20"
            ]
            is None
            else (
                f"{row['zscore_20']:+.2f}"
            )
        )

        def change(name):
            value = row[name]

            if value is None:
                return "NA"

            return (
                f"{value:+.2f}"
                f"{transform_units}"
            )

        lines.append(
            "- "
            f"{row['display_name']}: "
            f"latest={row['latest_value']} "
            f"{row['units']}; "
            f"date={row['latest_observation_date']}; "
            f"age={row['age_days']}d; "
            f"1obs={change('change_1obs')}; "
            f"5obs={change('change_5obs')}; "
            f"20obs={change('change_20obs')}; "
            f"63obs={change('change_63obs')}; "
            f"252obs={change('change_252obs')}; "
            f"z20={zscore}; "
            f"pct252={percentile}"
        )

    lines.append("")
    lines.append(
        "## Synchronized Curve Geometry"
    )
    lines.append("")

    for row in curves:
        percentile = (
            "NA"
            if row[
                "percentile_252"
            ]
            is None
            else (
                f"{row['percentile_252']:.1f}"
            )
        )

        zscore = (
            "NA"
            if row[
                "zscore_20"
            ]
            is None
            else (
                f"{row['zscore_20']:+.2f}"
            )
        )

        lines.append(
            "- "
            f"{row['display_name']}: "
            f"date={row['common_date']}; "
            f"level={row['level_bp']:+.1f}bp; "
            f"1obs={row['change_1obs_bp']:+.1f}bp; "
            f"5obs={row['change_5obs_bp']:+.1f}bp; "
            f"20obs={row['change_20obs_bp']:+.1f}bp; "
            f"z20={zscore}; "
            f"pct252={percentile}"
        )

    lines.append("")
    lines.append(
        "## Synchronized Yield Decomposition"
    )
    lines.append("")

    for row in decompositions:
        latest = row[
            "latest"
        ]

        lines.append(
            "- "
            f"{row['display_name']}: "
            f"date={row['common_date']}; "
            f"nominal="
            f"{latest['nominal_percent']:.2f}%; "
            f"real="
            f"{latest['real_percent']:.2f}%; "
            f"breakeven="
            f"{latest['breakeven_percent']:.2f}%; "
            f"identity_gap="
            f"{latest['identity_gap_bp']:+.2f}bp"
        )

        for horizon in (
            "obs_1",
            "obs_5",
            "obs_20",
        ):
            item = row[
                "horizons"
            ][horizon]

            if item is None:
                continue

            lines.append(
                "  - "
                f"{horizon}: "
                f"{item['prior_common_date']} -> "
                f"{item['latest_common_date']}; "
                f"nominal="
                f"{item['nominal_change_bp']:+.1f}bp; "
                f"real="
                f"{item['real_change_bp']:+.1f}bp; "
                f"breakeven="
                f"{item['breakeven_change_bp']:+.1f}bp; "
                f"identity_gap="
                f"{item['identity_gap_bp']:+.2f}bp"
            )

    text_path = (
        output_dir
        / "macro_state_packet.md"
    )

    text_path.write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version":
            "macro_state_packet_build_v0_1",

        "source_state_id":
            state_id,

        "json_file":
            json_path.name,

        "json_sha256":
            sha256_path(
                json_path
            ),

        "text_file":
            text_path.name,

        "text_sha256":
            sha256_path(
                text_path
            ),

        "series_count":
            len(
                series_state
            ),

        "curve_feature_count":
            len(
                curves
            ),

        "decomposition_count":
            len(
                decompositions
            ),
    }

    write_json(
        output_dir
        / "macro_state_packet_manifest.json",
        manifest,
    )

    write_json(
        output_root
        / "latest_run.json",
        {
            "source_state_id":
                state_id,

            "packet_json":
                str(
                    json_path
                ),

            "packet_markdown":
                str(
                    text_path
                ),

            "packet_sha256":
                manifest[
                    "text_sha256"
                ],
        },
    )

    print("=" * 80)
    print(
        "MACRO STATE PACKET BUILT"
    )
    print("=" * 80)

    print(
        "State:",
        state_id,
    )

    print(
        "Series:",
        len(
            series_state
        ),
    )

    print(
        "Curves:",
        len(
            curves
        ),
    )

    print(
        "Decompositions:",
        len(
            decompositions
        ),
    )

    print(
        "Text SHA:",
        manifest[
            "text_sha256"
        ],
    )

    print(
        "Model-visible packet:",
        text_path,
    )


if __name__ == "__main__":
    main()
