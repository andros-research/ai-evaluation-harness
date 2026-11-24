#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PACKET_POINTER = (
    ROOT
    / "benchmarks"
    / "results"
    / "macro_state_packets"
    / "cross_asset_core_v0_2"
    / "latest_run.json"
)

REFERENCE_EXPERIMENT = (
    ROOT
    / "benchmarks"
    / "fixtures"
    / "macro_narrative_experiments"
    / "v1_9_3_cross_asset_state_20261117_v0_1"
)

EXPERIMENT_ROOT = (
    ROOT
    / "benchmarks"
    / "fixtures"
    / "macro_narrative_experiments"
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--observation-date",
        required=True,
        help="YYYY-MM-DD weekly observation label",
    )

    args = parser.parse_args()

    obs_date = date.fromisoformat(
        args.observation_date
    )

    experiment_id = (
        "v1_9_3_cross_asset_state_"
        + obs_date.strftime("%Y%m%d")
        + "_v0_2"
    )

    out = (
        EXPERIMENT_ROOT
        / experiment_id
    )

    inputs = out / "inputs"

    if out.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {out}"
        )

    pointer = json.loads(
        PACKET_POINTER.read_text()
    )

    packet_md = Path(
        pointer["packet_markdown"]
    )

    packet_json = Path(
        pointer["packet_json"]
    )

    packet_payload = json.loads(
        packet_json.read_text()
    )

    # --------------------------------------------------
    # Preserve the exact week-1 task contract.
    # Only the supplied state changes.
    # --------------------------------------------------

    reference_prompt = (
        REFERENCE_EXPERIMENT
        / "inputs"
        / "narrative_prompt.txt"
    ).read_text(
        encoding="utf-8"
    )

    marker = (
        "SUPPLIED STATE\n"
        "=============="
    )

    if marker not in reference_prompt:
        raise RuntimeError(
            "Reference prompt marker not found."
        )

    instruction = (
        reference_prompt
        .split(marker, 1)[0]
        .rstrip()
        + "\n\n"
        + marker
        + "\n\n"
    )

    packet_text = packet_md.read_text(
        encoding="utf-8"
    )

    prompt = (
        instruction
        + packet_text
    )

    inputs.mkdir(
        parents=True,
        exist_ok=False,
    )

    frozen_md = (
        inputs
        / "macro_state_packet.md"
    )

    frozen_json = (
        inputs
        / "macro_state_packet.json"
    )

    prompt_path = (
        inputs
        / "narrative_prompt.txt"
    )

    shutil.copyfile(
        packet_md,
        frozen_md,
    )

    shutil.copyfile(
        packet_json,
        frozen_json,
    )

    prompt_path.write_text(
        prompt,
        encoding="utf-8",
    )

    manifest = {
        "schema_version":
            "macro_state_analysis_experiment_v0_1",

        "experiment_id":
            experiment_id,

        "purpose":
            (
                "Weekly multi-model analysis of a "
                "deterministic high-dimensional "
                "cross-asset macro state."
            ),

        "observation_date":
            args.observation_date,

        "input": {
            "source_claims_path":
                str(
                    packet_json.relative_to(
                        ROOT
                    )
                ),

            "frozen_claims_file":
                "inputs/macro_state_packet.json",

            "claims_sha256":
                sha256_path(
                    frozen_json
                ),

            "claim_count":
                0,

            "input_kind":
                "macro_state_packet",

            "source_state_id":
                packet_payload[
                    "source_state_id"
                ],

            "frozen_packet_markdown":
                "inputs/macro_state_packet.md",

            "packet_markdown_sha256":
                sha256_path(
                    frozen_md
                ),

            "packet_json_sha256":
                sha256_path(
                    frozen_json
                ),
        },

        "prompt": {
            "variant":
                "cross_asset_selection_v0_1",

            "file":
                "inputs/narrative_prompt.txt",

            "sha256":
                sha256_path(
                    prompt_path
                ),

            "characters":
                len(prompt),

            "prompt_contract_source":
                (
                    "v1_9_3_cross_asset_state_"
                    "20261117_v0_1"
                ),
        },

        "generation_arms": [
            {
                "arm_id":
                    "llama3_8b_bf16",

                "model":
                    (
                        "meta-llama/"
                        "Meta-Llama-3-8B-Instruct"
                    ),

                "reasoning_effort":
                    None,

                "dtype":
                    "bfloat16",

                "do_sample":
                    False,

                "max_new_tokens":
                    768,
            },

            {
                "arm_id":
                    "gpt_oss_20b_low",

                "model":
                    "openai/gpt-oss-20b",

                "reasoning_effort":
                    "low",

                "dtype":
                    "mxfp4",

                "do_sample":
                    False,

                "max_new_tokens":
                    4096,
            },

            {
                "arm_id":
                    "gpt_oss_20b_medium",

                "model":
                    "openai/gpt-oss-20b",

                "reasoning_effort":
                    "medium",

                "dtype":
                    "mxfp4",

                "do_sample":
                    False,

                "max_new_tokens":
                    4096,
            },

            {
                "arm_id":
                    "gpt_oss_20b_high",

                "model":
                    "openai/gpt-oss-20b",

                "reasoning_effort":
                    "high",

                "dtype":
                    "mxfp4",

                "do_sample":
                    False,

                "max_new_tokens":
                    4096,
            },
        ],

        "review_policy": {
            "blind_model_identity":
                True,

            "analysis_channel_hidden":
                True,

            "telemetry_hidden_until_review":
                True,

            "human_interpretation_added_to_prompt":
                False,

            "week_1_mapping_remains_sealed":
                True,
        },
    }

    write_json(
        out
        / "experiment_manifest.json",
        manifest,
    )

    print("=" * 80)
    print(
        "WEEKLY CROSS-ASSET EXPERIMENT FROZEN"
    )
    print("=" * 80)

    print(
        "Experiment:",
        experiment_id,
    )

    print(
        "Observation:",
        args.observation_date,
    )

    print(
        "State:",
        packet_payload[
            "source_state_id"
        ],
    )

    print(
        "Prompt SHA:",
        sha256_path(
            prompt_path
        ),
    )

    print(
        "Prompt chars:",
        len(prompt),
    )

    print(
        "Reference prompt contract:",
        "2026-11-17",
    )


if __name__ == "__main__":
    main()
