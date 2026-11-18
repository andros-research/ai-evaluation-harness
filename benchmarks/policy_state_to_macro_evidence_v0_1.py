#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from macro_evidence_contract_v0_2 import (
    validate_macro_evidence,
)

from policy_state_contract_v0_1 import (
    load_policy_state,
)


def build_policy_evidence_record(
    state: dict,
) -> dict:
    rate = state[
        "policy_rate"
    ]

    latest = rate[
        "latest"
    ]

    prior = rate[
        "prior_regime"
    ]

    transition = rate[
        "last_transition"
    ]

    target = state[
        "target_range"
    ]

    fomc = state[
        "latest_fomc_action"
    ]

    delta_pp = (
        float(latest["value"])
        - float(prior["value"])
    )

    if delta_pp > 0:
        direction = "up"
    elif delta_pp < 0:
        direction = "down"
    else:
        direction = "flat"

    period = state[
        "observation_period"
    ]

    return {
        "evidence_id":
            (
                "policy__effr__"
                + period
            ),

        "evidence_kind":
            "policy_state",

        "metric_name":
            "effr_policy_state",

        "display_name":
            "Federal Reserve policy state",

        "source_type":
            "composite_policy_state",

        "source_series":
            "DFF+FOMC",

        "source_observation_date":
            state["as_of_date"],

        "comparison_window":
            "policy_regime",

        "availability_status":
            "available",

        "current_value":
            float(
                latest["value"]
            ),

        "prior_value":
            float(
                prior["value"]
            ),

        "delta_value":
            round(
                delta_pp,
                10,
            ),

        "direction":
            direction,

        "units":
            "percent",

        "source_note":
            (
                "Event-aware policy state. "
                "The primary comparison is the "
                "latest EFFR versus the prior "
                "policy regime point, not a "
                "monthly-average FEDFUNDS change."
            ),

        "supporting_values": {
            "observation_period":
                period,

            "as_of_date":
                state[
                    "as_of_date"
                ],

            "latest_effr_date":
                latest["date"],

            "latest_effr":
                latest["value"],

            "prior_regime_date":
                prior["date"],

            "prior_regime_effr":
                prior["value"],

            "transition_effective_date":
                transition[
                    "effective_date"
                ],

            "transition_delta_bp":
                transition[
                    "delta_bp"
                ],

            "target_lower":
                target["lower"],

            "target_upper":
                target["upper"],

            "fomc_meeting_date":
                fomc[
                    "meeting_date"
                ],

            "fomc_action":
                fomc["action"],

            "fomc_delta_bp":
                fomc[
                    "delta_bp"
                ],
        },
    }


def build_policy_macro_evidence(
    state: dict,
) -> dict:
    payload = {
        "schema_version":
            "macro_evidence_v0_2",

        "observation_date":
            state["as_of_date"],

        "records": [
            build_policy_evidence_record(
                state
            )
        ],
    }

    validate_macro_evidence(
        payload
    )

    return payload


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "policy_state",
        type=Path,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    state = load_policy_state(
        args.policy_state
    )

    payload = (
        build_policy_macro_evidence(
            state
        )
    )

    text = (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )

    if args.output:
        args.output.write_text(
            text,
            encoding="utf-8",
        )

        print(
            "Output:",
            args.output,
        )
    else:
        print(text)


if __name__ == "__main__":
    main()
