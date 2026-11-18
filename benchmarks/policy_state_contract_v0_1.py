#!/usr/bin/env python3

from __future__ import annotations

import json
from datetime import date
from pathlib import Path


SCHEMA_VERSION = "macro_policy_state_v0_1"

ACTIONS = {
    "hike",
    "cut",
    "hold",
}


def _number(value) -> bool:
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


def _date(value: str) -> date:
    return date.fromisoformat(
        value
    )


def validate_policy_state(
    payload: dict,
) -> None:
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

    period = payload.get(
        "observation_period"
    )

    as_of = payload.get(
        "as_of_date"
    )

    if (
        not isinstance(
            period,
            str,
        )
        or len(period) != 7
    ):
        errors.append(
            "observation_period must be YYYY-MM"
        )

    try:
        as_of_date = _date(
            as_of
        )
    except Exception:
        errors.append(
            "invalid as_of_date"
        )
        as_of_date = None

    if (
        as_of_date is not None
        and period
        and not as_of.startswith(
            period
        )
    ):
        errors.append(
            "as_of_date must fall inside "
            "observation_period"
        )

    rate = payload.get(
        "policy_rate"
    )

    if not isinstance(
        rate,
        dict,
    ):
        errors.append(
            "policy_rate must be an object"
        )
        rate = {}

    if (
        rate.get("measure_id")
        != "effr"
    ):
        errors.append(
            "policy_rate.measure_id "
            "must be 'effr'"
        )

    latest = rate.get(
        "latest"
    )

    prior = rate.get(
        "prior_regime"
    )

    transition = rate.get(
        "last_transition"
    )

    for name, obj in (
        ("latest", latest),
        ("prior_regime", prior),
        ("last_transition", transition),
    ):
        if not isinstance(
            obj,
            dict,
        ):
            errors.append(
                f"policy_rate.{name} "
                "must be an object"
            )

    latest = (
        latest
        if isinstance(
            latest,
            dict,
        )
        else {}
    )

    prior = (
        prior
        if isinstance(
            prior,
            dict,
        )
        else {}
    )

    transition = (
        transition
        if isinstance(
            transition,
            dict,
        )
        else {}
    )

    if not _number(
        latest.get("value")
    ):
        errors.append(
            "latest EFFR must be numeric"
        )

    if not _number(
        prior.get("value")
    ):
        errors.append(
            "prior-regime EFFR must be numeric"
        )

    if not _number(
        transition.get(
            "delta_bp"
        )
    ):
        errors.append(
            "last_transition.delta_bp "
            "must be numeric"
        )

    if (
        _number(
            latest.get(
                "value"
            )
        )
        and _number(
            prior.get(
                "value"
            )
        )
        and _number(
            transition.get(
                "delta_bp"
            )
        )
    ):
        implied_bp = round(
            (
                latest["value"]
                - prior["value"]
            )
            * 100,
            6,
        )

        supplied_bp = round(
            float(
                transition[
                    "delta_bp"
                ]
            ),
            6,
        )

        if (
            implied_bp
            != supplied_bp
        ):
            errors.append(
                "EFFR regime-change arithmetic "
                "does not reconcile"
            )

    target = payload.get(
        "target_range"
    )

    if not isinstance(
        target,
        dict,
    ):
        errors.append(
            "target_range must be an object"
        )
        target = {}

    lower = target.get(
        "lower"
    )

    upper = target.get(
        "upper"
    )

    if (
        not _number(lower)
        or not _number(upper)
    ):
        errors.append(
            "target range bounds "
            "must be numeric"
        )

    elif lower >= upper:
        errors.append(
            "target range lower "
            "must be below upper"
        )

    action = payload.get(
        "latest_fomc_action"
    )

    if not isinstance(
        action,
        dict,
    ):
        errors.append(
            "latest_fomc_action "
            "must be an object"
        )
        action = {}

    action_name = action.get(
        "action"
    )

    if action_name not in ACTIONS:
        errors.append(
            "invalid FOMC action"
        )

    move_bp = action.get(
        "delta_bp"
    )

    if not _number(
        move_bp
    ):
        errors.append(
            "FOMC delta_bp "
            "must be numeric"
        )

    elif (
        action_name == "hold"
        and move_bp != 0
    ):
        errors.append(
            "hold must have "
            "delta_bp = 0"
        )

    if (
        _number(lower)
        and _number(upper)
    ):
        if (
            action.get(
                "target_lower"
            )
            != lower
            or action.get(
                "target_upper"
            )
            != upper
        ):
            errors.append(
                "latest FOMC action target "
                "range does not match "
                "current target range"
            )

    # Every economic event in the object must
    # already have occurred by the state as-of date.
    if as_of_date is not None:
        for label, value in (
            (
                "latest EFFR",
                latest.get("date"),
            ),
            (
                "prior regime EFFR",
                prior.get("date"),
            ),
            (
                "policy transition",
                transition.get(
                    "effective_date"
                ),
            ),
            (
                "FOMC action",
                action.get(
                    "meeting_date"
                ),
            ),
        ):
            try:
                event_date = _date(
                    value
                )
            except Exception:
                errors.append(
                    f"invalid {label} date"
                )
                continue

            if (
                event_date
                > as_of_date
            ):
                errors.append(
                    f"{label} occurs after "
                    "as_of_date"
                )

    provenance = payload.get(
        "provenance"
    )

    if (
        not isinstance(
            provenance,
            list,
        )
        or not provenance
    ):
        errors.append(
            "provenance must be "
            "a non-empty list"
        )

    if errors:
        raise ValueError(
            "Policy-state validation failed:\n"
            + "\n".join(
                f"- {error}"
                for error
                in errors
            )
        )


def load_policy_state(
    path: Path,
) -> dict:
    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    validate_policy_state(
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

    payload = load_policy_state(
        args.path
    )

    print(
        "POLICY STATE VALID"
    )

    print(
        "period:",
        payload[
            "observation_period"
        ],
    )

    print(
        "as-of:",
        payload[
            "as_of_date"
        ],
    )

    print(
        "EFFR:",
        payload[
            "policy_rate"
        ][
            "latest"
        ][
            "value"
        ],
    )

    print(
        "FOMC action:",
        payload[
            "latest_fomc_action"
        ][
            "action"
        ],
    )
