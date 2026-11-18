#!/usr/bin/env python3

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path


SCHEMA_VERSION = (
    "macro_market_state_v0_1"
)

FREQUENCIES = {
    "daily",
    "weekly",
    "monthly",
    "quarterly",
    "event",
}

AGGREGATION_CATEGORIES = {
    "point",
    "average",
    "sum",
    "change",
}

OBSERVATION_STATUSES = {
    "observed",
    "unavailable",
}

REVISION_POLICIES = {
    "not_revised",
    "latest_vintage",
    "first_release",
    "source_defined",
}


def _is_number(value) -> bool:
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


def _parse_date(
    value: str,
) -> date:
    return date.fromisoformat(
        value
    )


def _parse_timestamp(
    value: str,
) -> datetime:
    parsed = datetime.fromisoformat(
        value.replace(
            "Z",
            "+00:00",
        )
    )

    if parsed.tzinfo is None:
        raise ValueError(
            "Timestamp must include "
            "timezone/offset."
        )

    return parsed


def validate_market_state(
    payload: dict,
) -> None:
    errors = []

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "Market state must be a dict."
        )

    if (
        payload.get(
            "schema_version"
        )
        != SCHEMA_VERSION
    ):
        errors.append(
            "unexpected schema_version"
        )

    state_id = payload.get(
        "state_id"
    )

    if (
        not isinstance(
            state_id,
            str,
        )
        or not state_id.strip()
    ):
        errors.append(
            "state_id must be "
            "a non-empty string"
        )

    try:
        as_of = _parse_timestamp(
            payload.get(
                "as_of_timestamp"
            )
        )
    except Exception:
        errors.append(
            "invalid as_of_timestamp"
        )
        as_of = None

    try:
        created_at = _parse_timestamp(
            payload.get(
                "created_at"
            )
        )
    except Exception:
        errors.append(
            "invalid created_at"
        )
        created_at = None

    if (
        as_of is not None
        and created_at is not None
        and created_at < as_of
    ):
        errors.append(
            "created_at cannot precede "
            "state as_of_timestamp"
        )

    # --------------------------------------------------
    # Series registry
    # --------------------------------------------------

    series = payload.get(
        "series"
    )

    if not isinstance(
        series,
        list,
    ):
        errors.append(
            "series must be a list"
        )
        series = []

    series_by_id = {}

    for index, item in enumerate(
        series
    ):
        prefix = (
            f"series[{index}]"
        )

        if not isinstance(
            item,
            dict,
        ):
            errors.append(
                f"{prefix}: must be dict"
            )
            continue

        series_id = item.get(
            "series_id"
        )

        if (
            not isinstance(
                series_id,
                str,
            )
            or not series_id.strip()
        ):
            errors.append(
                f"{prefix}: invalid series_id"
            )
            continue

        if series_id in series_by_id:
            errors.append(
                f"{prefix}: duplicate "
                f"series_id {series_id!r}"
            )
            continue

        series_by_id[
            series_id
        ] = item

        for field in (
            "metric_name",
            "display_name",
            "family",
            "source_type",
            "source_series",
            "units",
            "value_type",
            "source_definition",
        ):
            value = item.get(
                field
            )

            if (
                not isinstance(
                    value,
                    str,
                )
                or not value.strip()
            ):
                errors.append(
                    f"{prefix}: invalid "
                    f"{field}"
                )

        if (
            item.get(
                "frequency"
            )
            not in FREQUENCIES
        ):
            errors.append(
                f"{prefix}: invalid frequency"
            )

        if (
            item.get(
                "aggregation_category"
            )
            not in AGGREGATION_CATEGORIES
        ):
            errors.append(
                f"{prefix}: invalid "
                "aggregation_category"
            )

        if (
            item.get(
                "revision_policy"
            )
            not in REVISION_POLICIES
        ):
            errors.append(
                f"{prefix}: invalid "
                "revision_policy"
            )

    # --------------------------------------------------
    # Observations
    # --------------------------------------------------

    observations = payload.get(
        "observations"
    )

    if not isinstance(
        observations,
        list,
    ):
        errors.append(
            "observations must be a list"
        )
        observations = []

    seen_observation_ids = set()

    for index, obs in enumerate(
        observations
    ):
        prefix = (
            f"observations[{index}]"
        )

        if not isinstance(
            obs,
            dict,
        ):
            errors.append(
                f"{prefix}: must be dict"
            )
            continue

        obs_id = obs.get(
            "observation_id"
        )

        if (
            not isinstance(
                obs_id,
                str,
            )
            or not obs_id.strip()
        ):
            errors.append(
                f"{prefix}: invalid "
                "observation_id"
            )

        elif (
            obs_id
            in seen_observation_ids
        ):
            errors.append(
                f"{prefix}: duplicate "
                f"observation_id {obs_id!r}"
            )

        else:
            seen_observation_ids.add(
                obs_id
            )

        series_id = obs.get(
            "series_id"
        )

        if series_id not in series_by_id:
            errors.append(
                f"{prefix}: unknown "
                f"series_id {series_id!r}"
            )
            continue

        series_meta = (
            series_by_id[
                series_id
            ]
        )

        status = obs.get(
            "status"
        )

        if (
            status
            not in OBSERVATION_STATUSES
        ):
            errors.append(
                f"{prefix}: invalid status"
            )

        value = obs.get(
            "value"
        )

        if status == "observed":
            if not _is_number(
                value
            ):
                errors.append(
                    f"{prefix}: observed "
                    "value must be numeric"
                )

        elif status == "unavailable":
            if value is not None:
                errors.append(
                    f"{prefix}: unavailable "
                    "value must be null"
                )

            reason = obs.get(
                "unavailable_reason"
            )

            if (
                not isinstance(
                    reason,
                    str,
                )
                or not reason.strip()
            ):
                errors.append(
                    f"{prefix}: unavailable "
                    "record requires reason"
                )

        try:
            observation_date = (
                _parse_date(
                    obs.get(
                        "observation_date"
                    )
                )
            )
        except Exception:
            errors.append(
                f"{prefix}: invalid "
                "observation_date"
            )
            observation_date = None

        try:
            period_start = (
                _parse_date(
                    obs.get(
                        "period_start"
                    )
                )
            )
        except Exception:
            errors.append(
                f"{prefix}: invalid "
                "period_start"
            )
            period_start = None

        try:
            period_end = (
                _parse_date(
                    obs.get(
                        "period_end"
                    )
                )
            )
        except Exception:
            errors.append(
                f"{prefix}: invalid "
                "period_end"
            )
            period_end = None

        if (
            period_start is not None
            and period_end is not None
            and period_start > period_end
        ):
            errors.append(
                f"{prefix}: period_start "
                "after period_end"
            )

        if (
            observation_date is not None
            and period_start is not None
            and period_end is not None
            and not (
                period_start
                <= observation_date
                <= period_end
            )
        ):
            errors.append(
                f"{prefix}: observation_date "
                "outside observation period"
            )

        # For point observations the period
        # must collapse to one economic date.
        if (
            series_meta.get(
                "aggregation_category"
            )
            == "point"
            and observation_date
            is not None
            and period_start
            is not None
            and period_end
            is not None
            and not (
                period_start
                == observation_date
                == period_end
            )
        ):
            errors.append(
                f"{prefix}: point observation "
                "must have a one-day period"
            )

        release_raw = obs.get(
            "release_timestamp"
        )

        release = None

        if release_raw is not None:
            try:
                release = (
                    _parse_timestamp(
                        release_raw
                    )
                )
            except Exception:
                errors.append(
                    f"{prefix}: invalid "
                    "release_timestamp"
                )

        try:
            available = (
                _parse_timestamp(
                    obs.get(
                        "available_timestamp"
                    )
                )
            )
        except Exception:
            errors.append(
                f"{prefix}: invalid "
                "available_timestamp"
            )
            available = None

        try:
            retrieved = (
                _parse_timestamp(
                    obs.get(
                        "retrieved_at"
                    )
                )
            )
        except Exception:
            errors.append(
                f"{prefix}: invalid "
                "retrieved_at"
            )
            retrieved = None

        if (
            release is not None
            and available is not None
            and available < release
        ):
            errors.append(
                f"{prefix}: available before "
                "official release"
            )

        if (
            available is not None
            and as_of is not None
            and available > as_of
        ):
            errors.append(
                f"{prefix}: LOOK-AHEAD — "
                "observation unavailable "
                "at state as-of time"
            )

        if (
            retrieved is not None
            and available is not None
            and retrieved < available
        ):
            errors.append(
                f"{prefix}: retrieved before "
                "data became available"
            )

        if (
            retrieved is not None
            and created_at is not None
            and retrieved > created_at
        ):
            errors.append(
                f"{prefix}: retrieved after "
                "state creation time"
            )

        vintage = obs.get(
            "vintage"
        )

        if not isinstance(
            vintage,
            dict,
        ):
            errors.append(
                f"{prefix}: vintage "
                "must be an object"
            )

        else:
            if (
                vintage.get(
                    "revision_policy"
                )
                != series_meta.get(
                    "revision_policy"
                )
            ):
                errors.append(
                    f"{prefix}: observation "
                    "revision policy does not "
                    "match series metadata"
                )

    if errors:
        raise ValueError(
            "Market-state validation failed:\n"
            + "\n".join(
                f"- {error}"
                for error in errors
            )
        )


def load_market_state(
    path: Path,
) -> dict:
    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    validate_market_state(
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

    state = load_market_state(
        args.path
    )

    print(
        "MARKET STATE VALID"
    )

    print(
        "state:",
        state[
            "state_id"
        ],
    )

    print(
        "as-of:",
        state[
            "as_of_timestamp"
        ],
    )

    print(
        "series:",
        len(
            state["series"]
        ),
    )

    print(
        "observations:",
        len(
            state[
                "observations"
            ]
        ),
    )
