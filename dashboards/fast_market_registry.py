from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

NOV17_EXPERIMENT_ID = (
    "v1_9_3_cross_asset_state_20261117_v0_1"
)

NOV23_EXPERIMENT_ID = (
    "v1_9_3_cross_asset_state_20261123_v0_2"
)

DEFAULT_FAST_MARKET_EXPERIMENT_ID = (
    NOV23_EXPERIMENT_ID
)


def _experiment(
    *,
    experiment_id: str,
    label: str,
    observation_date: str,
):
    fixture_root = (
        ROOT
        / "benchmarks"
        / "fixtures"
        / "macro_narrative_experiments"
        / experiment_id
    )

    review_root = (
        ROOT
        / "benchmarks"
        / "results"
        / "macro_narrative_reviews"
        / experiment_id
        / "blind_review_v0_1"
    )

    return {
        "experiment_id":
            experiment_id,

        "label":
            label,

        "observation_date":
            observation_date,

        "fixture_root":
            fixture_root,

        "state_json":
            (
                fixture_root
                / "inputs"
                / "macro_state_packet.json"
            ),

        "state_md":
            (
                fixture_root
                / "inputs"
                / "macro_state_packet.md"
            ),

        "review_packet":
            (
                review_root
                / "blind_review_packet.md"
            ),

        "review_manifest":
            (
                review_root
                / "review_manifest.json"
            ),
    }


FAST_MARKET_EXPERIMENTS = {
    NOV23_EXPERIMENT_ID:
        _experiment(
            experiment_id=
                NOV23_EXPERIMENT_ID,

            label=
                "November 23, 2026",

            observation_date=
                "2026-11-23",
        ),

    NOV17_EXPERIMENT_ID:
        _experiment(
            experiment_id=
                NOV17_EXPERIMENT_ID,

            label=
                "November 17, 2026",

            observation_date=
                "2026-11-17",
        ),
}


def get_fast_market_experiment(
    experiment_id: str,
):
    return FAST_MARKET_EXPERIMENTS[
        experiment_id
    ]


def list_fast_market_experiments():
    return sorted(
        FAST_MARKET_EXPERIMENTS,
        key=lambda experiment_id:
            FAST_MARKET_EXPERIMENTS[
                experiment_id
            ][
                "observation_date"
            ],
        reverse=True,
    )


def previous_fast_market_experiment(
    experiment_id: str,
):
    ordered = (
        list_fast_market_experiments()
    )

    index = ordered.index(
        experiment_id
    )

    if (
        index + 1
        >= len(ordered)
    ):
        return None

    return ordered[
        index + 1
    ]
