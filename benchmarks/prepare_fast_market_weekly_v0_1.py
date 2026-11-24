#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    command = [
        sys.executable,
        *args,
    ]

    print()
    print("=" * 80)
    print(
        "RUN:",
        " ".join(
            str(x)
            for x in command
        ),
    )
    print("=" * 80)

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--observation-date",
        default=date.today().isoformat(),
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

    run(
        "benchmarks/"
        "build_fred_market_state_v0_1.py",

        "--lookback-days",
        str(
            args.lookback_days
        ),

        "--timeout",
        str(
            args.timeout
        ),
    )

    run(
        "benchmarks/"
        "build_market_features_v0_1.py"
    )

    run(
        "benchmarks/"
        "build_cross_market_features_v0_1.py"
    )

    run(
        "benchmarks/"
        "build_macro_state_packet_v0_1.py"
    )

    run(
        "benchmarks/"
        "freeze_cross_asset_weekly_experiment_v0_1.py",

        "--observation-date",
        args.observation_date,
    )

    compact = (
        args.observation_date
        .replace(
            "-",
            "",
        )
    )

    experiment_id = (
        "v1_9_3_cross_asset_state_"
        + compact
        + "_v0_1"
    )

    llama_run_id = (
        "v1_9_3_cross_asset_state_"
        + compact
        + "_llama3_8b_bf16_001"
    )

    print()
    print("=" * 80)
    print(
        "FAST MARKETS WEEKLY PREP COMPLETE"
    )
    print("=" * 80)

    print(
        "Experiment:",
        experiment_id,
    )

    print()
    print(
        "Next: GPT-OSS generation"
    )

    print(
        "  python "
        "benchmarks/"
        "run_frozen_macro_gpt_oss_arms.py "
        f"--experiment-id {experiment_id} "
        f"--system-date {args.observation_date}"
    )

    print()
    print(
        "Next: prepare Llama arm"
    )

    print(
        "  python "
        "benchmarks/"
        "prepare_frozen_macro_llama_arm.py "
        f"--experiment-id {experiment_id} "
        f"--run-id {llama_run_id}"
    )


if __name__ == "__main__":
    main()
