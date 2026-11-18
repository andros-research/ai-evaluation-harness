#!/usr/bin/env python3

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


SCHEMA_VERSION = "macro_market_features_v0_1"

TRANSFORMS = {
    "bp_change",
    "pct_return",
}

EXPECTED_HORIZONS = {
    "obs_1",
    "obs_5",
    "obs_20",
    "obs_63",
    "obs_252",
}


def _number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    )


def validate_market_features(
    payload: dict,
):
    errors = []

    if (
        payload.get("schema_version")
        != SCHEMA_VERSION
    ):
        errors.append(
            "unexpected schema_version"
        )

    for field in (
        "source_state_id",
        "source_state_sha256",
        "as_of_timestamp",
    ):
        value = payload.get(field)

        if (
            not isinstance(value, str)
            or not value
        ):
            errors.append(
                f"invalid {field}"
            )

    try:
        as_of = datetime.fromisoformat(
            payload[
                "as_of_timestamp"
            ]
        )
    except Exception:
        errors.append(
            "invalid as_of_timestamp"
        )
        as_of = None

    rows = payload.get(
        "series_features"
    )

    if not isinstance(
        rows,
        list,
    ):
        errors.append(
            "series_features must be list"
        )
        rows = []

    seen = set()

    for i, row in enumerate(rows):
        prefix = (
            f"series_features[{i}]"
        )

        series_id = row.get(
            "series_id"
        )

        if (
            not isinstance(
                series_id,
                str,
            )
            or not series_id
        ):
            errors.append(
                f"{prefix}: invalid series_id"
            )

        elif series_id in seen:
            errors.append(
                f"{prefix}: duplicate series_id"
            )

        else:
            seen.add(
                series_id
            )

        if (
            row.get("transform")
            not in TRANSFORMS
        ):
            errors.append(
                f"{prefix}: invalid transform"
            )

        if not _number(
            row.get("latest_value")
        ):
            errors.append(
                f"{prefix}: invalid latest_value"
            )

        if (
            not isinstance(
                row.get("age_days"),
                int,
            )
            or row["age_days"] < 0
        ):
            errors.append(
                f"{prefix}: invalid age_days"
            )

        try:
            latest_date = (
                datetime.fromisoformat(
                    row[
                        "latest_observation_date"
                    ]
                ).date()
            )

            if (
                as_of is not None
                and latest_date
                > as_of.date()
            ):
                errors.append(
                    f"{prefix}: latest date "
                    "after state as-of"
                )

        except Exception:
            errors.append(
                f"{prefix}: invalid "
                "latest_observation_date"
            )

        horizons = row.get(
            "horizons"
        )

        if not isinstance(
            horizons,
            dict,
        ):
            errors.append(
                f"{prefix}: horizons "
                "must be object"
            )
            continue

        if (
            set(horizons)
            != EXPECTED_HORIZONS
        ):
            errors.append(
                f"{prefix}: unexpected "
                "horizon keys"
            )

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
                    "available must be bool"
                )
                continue

            if available:
                for field in (
                    "prior_value",
                    "latest_value",
                    "raw_delta",
                    "feature_value",
                ):
                    if not _number(
                        feature.get(field)
                    ):
                        errors.append(
                            f"{prefix}.{name}: "
                            f"invalid {field}"
                        )

                if (
                    not isinstance(
                        feature.get(
                            "prior_observation_date"
                        ),
                        str,
                    )
                ):
                    errors.append(
                        f"{prefix}.{name}: "
                        "missing prior date"
                    )

            else:
                if (
                    feature.get(
                        "feature_value"
                    )
                    is not None
                ):
                    errors.append(
                        f"{prefix}.{name}: "
                        "unavailable feature "
                        "must have null value"
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

    if errors:
        raise ValueError(
            "Market-feature validation failed:\n"
            + "\n".join(
                f"- {x}"
                for x in errors
            )
        )


def load_market_features(
    path: Path,
):
    payload = json.loads(
        path.read_text()
    )

    validate_market_features(
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

    payload = load_market_features(
        args.path
    )

    print(
        "MARKET FEATURES VALID"
    )

    print(
        "state:",
        payload[
            "source_state_id"
        ],
    )

    print(
        "series:",
        len(
            payload[
                "series_features"
            ]
        ),
    )
