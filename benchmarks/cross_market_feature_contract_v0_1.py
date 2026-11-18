#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path


SCHEMA_VERSION = (
    "macro_cross_market_features_v0_1"
)

EXPECTED_HORIZONS = {
    "obs_1",
    "obs_5",
    "obs_20",
}


def _number(value):
    return (
        isinstance(
            value,
            (int, float),
        )
        and not isinstance(
            value,
            bool,
        )
    )


def validate_cross_market_features(
    payload: dict,
):
    errors = []

    if (
        payload.get(
            "schema_version"
        )
        != SCHEMA_VERSION
    ):
        errors.append(
            "unexpected schema_version"
        )

    for field in (
        "source_state_id",
        "source_state_sha256",
        "source_series_features_sha256",
        "as_of_timestamp",
    ):
        if (
            not isinstance(
                payload.get(field),
                str,
            )
            or not payload.get(field)
        ):
            errors.append(
                f"invalid {field}"
            )

    spreads = payload.get(
        "spread_features"
    )

    if not isinstance(
        spreads,
        list,
    ):
        errors.append(
            "spread_features must be list"
        )
        spreads = []

    seen = set()

    for i, row in enumerate(
        spreads
    ):
        prefix = (
            f"spread_features[{i}]"
        )

        feature_id = row.get(
            "feature_id"
        )

        if (
            not isinstance(
                feature_id,
                str,
            )
            or not feature_id
        ):
            errors.append(
                f"{prefix}: invalid feature_id"
            )

        elif feature_id in seen:
            errors.append(
                f"{prefix}: duplicate feature_id"
            )

        else:
            seen.add(
                feature_id
            )

        components = row.get(
            "component_series"
        )

        if (
            not isinstance(
                components,
                list,
            )
            or len(components) < 2
        ):
            errors.append(
                f"{prefix}: invalid components"
            )

        if not _number(
            row.get(
                "latest_value_bp"
            )
        ):
            errors.append(
                f"{prefix}: invalid latest_value_bp"
            )

        if (
            not isinstance(
                row.get(
                    "latest_common_date"
                ),
                str,
            )
        ):
            errors.append(
                f"{prefix}: invalid latest_common_date"
            )

        if (
            not isinstance(
                row.get(
                    "common_history_count"
                ),
                int,
            )
            or row[
                "common_history_count"
            ] < 1
        ):
            errors.append(
                f"{prefix}: invalid history count"
            )

        horizons = row.get(
            "horizons"
        )

        if (
            not isinstance(
                horizons,
                dict,
            )
            or set(horizons)
            != EXPECTED_HORIZONS
        ):
            errors.append(
                f"{prefix}: invalid horizons"
            )

        else:
            for name, feature in (
                horizons.items()
            ):
                available = feature.get(
                    "available"
                )

                if not isinstance(
                    available,
                    bool,
                ):
                    errors.append(
                        f"{prefix}.{name}: "
                        "invalid available"
                    )
                    continue

                if available:
                    for field in (
                        "prior_value_bp",
                        "latest_value_bp",
                        "change_bp",
                    ):
                        if not _number(
                            feature.get(
                                field
                            )
                        ):
                            errors.append(
                                f"{prefix}.{name}: "
                                f"invalid {field}"
                            )

                    for field in (
                        "prior_common_date",
                        "latest_common_date",
                    ):
                        if not isinstance(
                            feature.get(field),
                            str,
                        ):
                            errors.append(
                                f"{prefix}.{name}: "
                                f"invalid {field}"
                            )

        stats = row.get(
            "statistics"
        )

        if not isinstance(
            stats,
            dict,
        ):
            errors.append(
                f"{prefix}: statistics "
                "must be object"
            )

    decompositions = payload.get(
        "decomposition_features"
    )

    if not isinstance(
        decompositions,
        list,
    ):
        errors.append(
            "decomposition_features must be list"
        )
        decompositions = []

    for i, row in enumerate(
        decompositions
    ):
        prefix = (
            f"decomposition_features[{i}]"
        )

        feature_id = row.get(
            "feature_id"
        )

        if (
            not isinstance(
                feature_id,
                str,
            )
            or not feature_id
        ):
            errors.append(
                f"{prefix}: invalid feature_id"
            )

        elif feature_id in seen:
            errors.append(
                f"{prefix}: duplicate feature_id"
            )

        else:
            seen.add(
                feature_id
            )

        components = row.get(
            "component_series"
        )

        if (
            not isinstance(
                components,
                dict,
            )
            or set(components)
            != {
                "nominal",
                "real",
                "breakeven",
            }
        ):
            errors.append(
                f"{prefix}: invalid components"
            )

        latest = row.get(
            "latest"
        )

        if not isinstance(
            latest,
            dict,
        ):
            errors.append(
                f"{prefix}: latest must be object"
            )

        else:
            for field in (
                "nominal_percent",
                "real_percent",
                "breakeven_percent",
                "identity_gap_bp",
            ):
                if not _number(
                    latest.get(field)
                ):
                    errors.append(
                        f"{prefix}: invalid "
                        f"latest {field}"
                    )

        horizons = row.get(
            "horizons"
        )

        if (
            not isinstance(
                horizons,
                dict,
            )
            or set(horizons)
            != EXPECTED_HORIZONS
        ):
            errors.append(
                f"{prefix}: invalid horizons"
            )

        else:
            for name, feature in (
                horizons.items()
            ):
                available = feature.get(
                    "available"
                )

                if not isinstance(
                    available,
                    bool,
                ):
                    errors.append(
                        f"{prefix}.{name}: "
                        "invalid available"
                    )
                    continue

                if available:
                    for field in (
                        "nominal_change_bp",
                        "real_change_bp",
                        "breakeven_change_bp",
                        "change_identity_gap_bp",
                        "prior_identity_gap_bp",
                        "latest_identity_gap_bp",
                    ):
                        if not _number(
                            feature.get(
                                field
                            )
                        ):
                            errors.append(
                                f"{prefix}.{name}: "
                                f"invalid {field}"
                            )

    if errors:
        raise ValueError(
            "Cross-market feature "
            "validation failed:\n"
            + "\n".join(
                f"- {error}"
                for error in errors
            )
        )


def load_cross_market_features(
    path: Path,
):
    payload = json.loads(
        path.read_text()
    )

    validate_cross_market_features(
        payload
    )

    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "path",
        type=Path,
    )

    args = parser.parse_args()

    payload = (
        load_cross_market_features(
            args.path
        )
    )

    print(
        "CROSS-MARKET FEATURES VALID"
    )

    print(
        "state:",
        payload[
            "source_state_id"
        ],
    )

    print(
        "spreads:",
        len(
            payload[
                "spread_features"
            ]
        ),
    )

    print(
        "decompositions:",
        len(
            payload[
                "decomposition_features"
            ]
        ),
    )
