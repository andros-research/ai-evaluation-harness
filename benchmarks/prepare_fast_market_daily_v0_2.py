#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

STATE_ROOT = (
    ROOT
    / "benchmarks"
    / "results"
    / "market_states"
    / "cross_asset_core_v0_2"
)

FEATURE_ROOT = (
    ROOT
    / "benchmarks"
    / "results"
    / "market_features"
    / "cross_asset_core_v0_2"
)

CROSS_ROOT = (
    ROOT
    / "benchmarks"
    / "results"
    / "cross_market_features"
    / "cross_asset_core_v0_2"
)

PACKET_ROOT = (
    ROOT
    / "benchmarks"
    / "results"
    / "macro_state_packets"
    / "cross_asset_core_v0_2"
)


def run(*args):
    command = [
        sys.executable,
        *map(str, args),
    ]

    print()
    print("=" * 80)
    print(
        "RUN:",
        " ".join(command),
    )
    print("=" * 80)

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def read_pointer(
    root: Path,
    key: str,
) -> Path:
    pointer = json.loads(
        (
            root
            / "latest_run.json"
        ).read_text()
    )

    return Path(
        pointer[key]
    ).resolve()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare one Treasury-primary "
            "Fast Markets daily observation."
        )
    )

    parser.add_argument(
        "--observation-date",
        required=True,
        help="YYYY-MM-DD experiment label",
    )

    parser.add_argument(
        "--lookback-days",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
    )

    args = parser.parse_args()

    # ----------------------------------
    # 1. Mixed-source market state:
    # Treasury primary + remaining FRED.
    # ----------------------------------

    run(
        "benchmarks/build_market_state_v0_2.py",
        "--lookback-days",
        args.lookback_days,
        "--timeout",
        args.timeout,
    )

    state_path = read_pointer(
        STATE_ROOT,
        "market_state_path",
    )

    # ----------------------------------
    # 2. Per-series deterministic features.
    # ----------------------------------

    run(
        "benchmarks/build_market_features_v0_1.py",
        "--state",
        state_path,
        "--output-root",
        FEATURE_ROOT,
    )

    feature_path = read_pointer(
        FEATURE_ROOT,
        "market_features_path",
    )

    # ----------------------------------
    # 3. Synchronized cross-series geometry.
    # ----------------------------------

    run(
        "benchmarks/build_cross_market_features_v0_1.py",
        "--state",
        state_path,
        "--series-features",
        feature_path,
        "--output-root",
        CROSS_ROOT,
    )

    cross_path = read_pointer(
        CROSS_ROOT,
        "cross_market_features_path",
    )

    # ----------------------------------
    # 4. Compact model-visible state packet.
    # ----------------------------------

    run(
        "benchmarks/build_macro_state_packet_v0_2.py",
        "--series-features",
        feature_path,
        "--cross-market-features",
        cross_path,
        "--output-root",
        PACKET_ROOT,
    )

    # ----------------------------------
    # 5. Freeze daily experiment.
    # ----------------------------------

    run(
        "benchmarks/freeze_cross_asset_weekly_experiment_v0_2.py",
        "--observation-date",
        args.observation_date,
    )

    compact_date = (
        args.observation_date
        .replace("-", "")
    )

    experiment_id = (
        "v1_9_3_cross_asset_state_"
        f"{compact_date}_v0_2"
    )

    llama_run_id = (
        experiment_id
        + "_llama3_8b_bf16_001"
    )

    print()
    print("=" * 80)
    print(
        "FAST MARKETS DAILY v0.2 PREP COMPLETE"
    )
    print("=" * 80)

    print(
        "Observation:",
        args.observation_date,
    )

    print(
        "State:",
        state_path,
    )

    print(
        "Experiment:",
        experiment_id,
    )

    print()
    print(
        "Next GPT-OSS command:"
    )

    print(
        "python benchmarks/"
        "run_frozen_macro_gpt_oss_arms.py "
        "--experiment-id "
        f"{experiment_id} "
        "--system-date "
        f"{args.observation_date}"
    )

    print()
    print(
        "Next Llama preparation:"
    )

    print(
        "python benchmarks/"
        "prepare_frozen_macro_llama_arm.py "
        "--experiment-id "
        f"{experiment_id} "
        "--run-id "
        f"{llama_run_id}"
    )


if __name__ == "__main__":
    main()
