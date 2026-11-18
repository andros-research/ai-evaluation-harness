#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from policy_state_contract_v0_1 import (
    validate_policy_state,
)


ROOT = Path(__file__).resolve().parents[1]

OUT = (
    ROOT
    / "benchmarks"
    / "fixtures"
    / "macro_policy_state_2026_10_v0_1"
)

STATE = (
    OUT
    / "policy_state.json"
)

MANIFEST = (
    OUT
    / "policy_state_manifest.json"
)


def sha(path: Path) -> str:
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
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main():
    if OUT.exists():
        raise FileExistsError(
            f"Refusing to replace "
            f"frozen fixture: {OUT}"
        )

    OUT.mkdir(
        parents=True,
        exist_ok=False,
    )

    frozen_at = datetime.now(
        timezone.utc
    ).isoformat()

    payload = {
        "schema_version":
            "macro_policy_state_v0_1",

        # Economic period being described.
        "observation_period":
            "2026-10",

        # Point-in-time boundary for the
        # state object. This is deliberately
        # distinct from the month label.
        "as_of_date":
            "2026-10-30",

        "policy_rate": {
            "measure_id":
                "effr",

            "display_name":
                "Effective Federal Funds Rate",

            "frequency":
                "daily",

            "units":
                "percent",

            "latest": {
                "date":
                    "2026-10-30",

                "value":
                    3.88,
            },

            "prior_regime": {
                "date":
                    "2026-09-16",

                "value":
                    3.63,
            },

            "last_transition": {
                "effective_date":
                    "2026-09-17",

                "direction":
                    "up",

                "delta_bp":
                    25,
            },

            "source_series":
                "DFF",

            "aggregation_policy":
                "point_level_no_monthly_average",
        },

        "target_range": {
            "lower":
                3.75,

            "upper":
                4.00,

            "units":
                "percent",

            "effective_date":
                "2026-09-17",

            "confirmed_unchanged_date":
                "2026-10-28",
        },

        "latest_fomc_action": {
            "meeting_date":
                "2026-10-28",

            "action":
                "hold",

            "delta_bp":
                0,

            "target_lower":
                3.75,

            "target_upper":
                4.00,

            "target_range_unchanged":
                True,
        },

        "representation_policy": {
            "primary_policy_signal":
                (
                    "Latest point EFFR plus "
                    "discrete policy transition "
                    "and FOMC action."
                ),

            "monthly_average_role":
                (
                    "Permitted as a derived "
                    "statistic but not as the "
                    "primary representation of "
                    "policy-state change."
                ),
        },

        "provenance": [
            {
                "source_type":
                    "fred",

                "series":
                    "DFF",

                "supports": [
                    (
                        "EFFR 3.63 percent "
                        "on 2026-09-16"
                    ),
                    (
                        "EFFR 3.88 percent "
                        "from 2026-09-17 "
                        "through October"
                    ),
                ],

                "url":
                    (
                        "https://fred.stlouisfed.org/"
                        "series/DFF"
                    ),
            },

            {
                "source_type":
                    "federal_reserve",

                "document":
                    (
                        "September 16, 2026 "
                        "FOMC statement and "
                        "implementation note"
                    ),

                "supports": [
                    (
                        "25 bp hike to "
                        "3.75-4.00 percent"
                    ),
                    (
                        "effective "
                        "September 17, 2026"
                    ),
                ],

                "url":
                    (
                        "https://www.federalreserve.gov/"
                        "newsevents/pressreleases/"
                        "monetary20260916a.htm"
                    ),
            },

            {
                "source_type":
                    "federal_reserve",

                "document":
                    (
                        "October 28, 2026 "
                        "FOMC statement"
                    ),

                "supports": [
                    (
                        "Target range held "
                        "at 3.75-4.00 percent"
                    ),
                ],

                "url":
                    (
                        "https://www.federalreserve.gov/"
                        "newsevents/pressreleases/"
                        "monetary20261028a.htm"
                    ),
            },
        ],
    }

    validate_policy_state(
        payload
    )

    write_json(
        STATE,
        payload,
    )

    manifest = {
        "schema_version":
            "macro_policy_state_freeze_v0_1",

        "status":
            "frozen",

        "observation_period":
            "2026-10",

        "as_of_date":
            "2026-10-30",

        "frozen_at":
            frozen_at,

        "policy_state_file":
            STATE.name,

        "policy_state_sha256":
            sha(
                STATE
            ),

        "historical_experiment_mutated":
            False,

        "note":
            (
                "First event-aware policy-state "
                "fixture. The frozen October "
                "narrative experiment remains "
                "unchanged."
            ),
    }

    write_json(
        MANIFEST,
        manifest,
    )

    print("=" * 80)
    print(
        "OCTOBER POLICY STATE FROZEN"
    )
    print("=" * 80)

    print(
        "Period:",
        payload[
            "observation_period"
        ],
    )

    print(
        "As-of:",
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
        "Prior regime:",
        payload[
            "policy_rate"
        ][
            "prior_regime"
        ][
            "value"
        ],
    )

    print(
        "Transition:",
        (
            f"+{payload['policy_rate']['last_transition']['delta_bp']} bp"
        ),
    )

    print(
        "FOMC:",
        payload[
            "latest_fomc_action"
        ][
            "action"
        ],
    )

    print(
        "SHA:",
        sha(
            STATE
        ),
    )


if __name__ == "__main__":
    main()
