#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_ROOT = (
    ROOT
    / "benchmarks"
    / "fixtures"
    / "macro_narrative_experiments"
)

RESULT_ROOT = (
    ROOT
    / "benchmarks"
    / "results"
    / "native_fred_generation"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment-id",
        required=True,
    )

    parser.add_argument(
        "--run-id",
        required=True,
    )

    args = parser.parse_args()

    experiment_dir = (
        EXPERIMENT_ROOT
        / args.experiment_id
    )

    manifest_path = (
        experiment_dir
        / "experiment_manifest.json"
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    prompt_path = (
        experiment_dir
        / manifest["prompt"]["file"]
    )

    prompt_bytes = prompt_path.read_bytes()

    actual_prompt_hash = sha256_bytes(
        prompt_bytes
    )

    expected_prompt_hash = (
        manifest["prompt"]["sha256"]
    )

    if actual_prompt_hash != expected_prompt_hash:
        raise RuntimeError(
            "Frozen prompt hash mismatch."
        )

    arm = next(
        x
        for x in manifest["generation_arms"]
        if x["arm_id"] == "llama3_8b_bf16"
    )

    run_dir = (
        RESULT_ROOT
        / args.run_id
    )

    if run_dir.exists():
        raise FileExistsError(
            f"Run already exists: {run_dir}"
        )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # Preserve the exact frozen prompt bytes.
    (run_dir / "prompt.txt").write_bytes(
        prompt_bytes
    )

    plan = {
        "schema_version":
            "native_fred_generation_plan_v0_1",

        "run_id":
            args.run_id,

        "experiment": {
            "experiment_id":
                args.experiment_id,

            "manifest_path":
                str(
                    manifest_path
                    .relative_to(ROOT)
                ),

            "manifest_sha256":
                sha256_bytes(
                    manifest_path.read_bytes()
                ),

            "arm_id":
                arm["arm_id"],
        },

        "source": {
            "claims_path":
                str(
                    (
                        experiment_dir
                        / manifest["input"][
                            "frozen_claims_file"
                        ]
                    ).relative_to(ROOT)
                ),

            "claims_sha256":
                manifest["input"][
                    "claims_sha256"
                ],

            "claim_count":
                manifest["input"][
                    "claim_count"
                ],
        },

        "prompt": {
            "variant":
                manifest["prompt"][
                    "variant"
                ],

            "sha256":
                expected_prompt_hash,

            "characters":
                manifest["prompt"][
                    "characters"
                ],

            "file":
                "prompt.txt",
        },

        "inference": {
            "backend":
                "transformers_pytorch",

            "model_id":
                arm["model"],

            "dtype":
                "bfloat16",

            "chat_template":
                "model_native",

            "do_sample":
                False,

            "temperature":
                0.0,

            "max_new_tokens":
                arm["max_new_tokens"],
        },

        "planned_outputs": {
            "narrative":
                "fred_narrative.md",

            "generation_metadata":
                "native_generation_metadata.json",

            "generation_telemetry":
                "native_generation_telemetry.json",
        },
    }

    (run_dir / "run_plan.json").write_text(
        json.dumps(
            plan,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("FROZEN LLAMA ARM PREPARED")
    print("=" * 72)
    print("Experiment:", args.experiment_id)
    print("Run:", args.run_id)
    print("Arm:", arm["arm_id"])
    print("Model:", arm["model"])
    print("Prompt SHA-256:", expected_prompt_hash)
    print("Output:", run_dir)


if __name__ == "__main__":
    main()
