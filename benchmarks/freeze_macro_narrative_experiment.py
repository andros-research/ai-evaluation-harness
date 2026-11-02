#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.generate_fred_narrative_from_claims import (
    build_llm_prompt,
    load_selected_claims,
    validate_selected_claims,
)

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ROOT = (
    ROOT
    / "benchmarks"
    / "fixtures"
    / "macro_narrative_experiments"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--claims-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--experiment-id",
        required=True,
    )

    parser.add_argument(
        "--prompt-variant",
        default="semantic_analytical",
    )

    args = parser.parse_args()

    if (
        not args.experiment_id
        or Path(args.experiment_id).name != args.experiment_id
        or args.experiment_id in (".", "..")
    ):
        raise ValueError("Invalid experiment ID.")

    claims_path = args.claims_path.resolve()

    selected_claims = load_selected_claims(
        claims_path
    )

    validate_selected_claims(
        selected_claims
    )

    prompt = build_llm_prompt(
        selected_claims,
        args.prompt_variant,
    )

    experiment_dir = (
        DEFAULT_ROOT
        / args.experiment_id
    )

    if experiment_dir.exists():
        raise FileExistsError(
            f"Experiment already exists: "
            f"{experiment_dir}"
        )

    inputs_dir = (
        experiment_dir
        / "inputs"
    )

    inputs_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # Preserve the original frozen claim bytes exactly.
    claims_bytes = claims_path.read_bytes()

    frozen_claims_path = (
        inputs_dir
        / "selected_fred_claims.json"
    )

    frozen_claims_path.write_bytes(
        claims_bytes
    )

    prompt_path = (
        inputs_dir
        / "narrative_prompt.txt"
    )

    prompt_path.write_text(
        prompt,
        encoding="utf-8",
    )

    prompt_builder_path = (
        ROOT
        / "benchmarks"
        / "generate_fred_narrative_from_claims.py"
    )

    source_snapshot_manifest = (
        claims_path
        .parent
        .parent
        / "snapshot_manifest.json"
    )

    if not source_snapshot_manifest.exists():
        raise FileNotFoundError(
            "Could not resolve source snapshot manifest: "
            f"{source_snapshot_manifest}"
        )

    manifest = {
        "schema_version":
            "macro_narrative_experiment_v0_1",

        "experiment_id":
            args.experiment_id,

        "purpose":
            "First v1.9.3 paired multi-model "
            "macro narrative experiment",

        "input": {
            "source_claims_path":
                str(
                    claims_path.relative_to(ROOT)
                ),

            "frozen_claims_file":
                "inputs/selected_fred_claims.json",

            "claims_sha256":
                sha256_bytes(claims_bytes),

            "claim_count":
                len(selected_claims),

            "source_snapshot_manifest":
                str(
                    source_snapshot_manifest
                    .relative_to(ROOT)
                ),

            "source_snapshot_manifest_sha256":
                sha256_path(
                    source_snapshot_manifest
                ),
        },

        "prompt": {
            "variant":
                args.prompt_variant,

            "file":
                "inputs/narrative_prompt.txt",

            "sha256":
                sha256_bytes(
                    prompt.encode("utf-8")
                ),

            "characters":
                len(prompt),

            "builder":
                "benchmarks/"
                "generate_fred_narrative_from_claims.py",

            "builder_sha256":
                sha256_path(
                    prompt_builder_path
                ),
        },

        "generation_arms": [
            {
                "arm_id":
                    "llama3_8b_bf16",

                "model":
                    "meta-llama/"
                    "Meta-Llama-3-8B-Instruct",

                "reasoning_effort":
                    None,

                "dtype":
                    "bfloat16",

                "do_sample":
                    False,

                "max_new_tokens":
                    512,
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
            "generation_count_per_arm":
                1,

            "human_review_before_telemetry":
                True,

            "human_review_blinded_to_arm":
                True,

            "semantic_classifier":
                "compact_balanced_v0_2_1",

            "semantic_classifier_reasoning_effort":
                "low",

            "classifier_applied_after_human_review":
                True,
        },
    }

    manifest_path = (
        experiment_dir
        / "experiment_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("MACRO NARRATIVE EXPERIMENT FROZEN")
    print("=" * 72)

    print("Experiment:", args.experiment_id)
    print(
        "Claims:",
        len(selected_claims),
    )
    print(
        "Claims SHA-256:",
        manifest["input"]["claims_sha256"],
    )
    print(
        "Prompt variant:",
        args.prompt_variant,
    )
    print(
        "Prompt characters:",
        len(prompt),
    )
    print(
        "Prompt SHA-256:",
        manifest["prompt"]["sha256"],
    )
    print(
        "Builder SHA-256:",
        manifest["prompt"]["builder_sha256"],
    )
    print(
        "Arms:",
        len(manifest["generation_arms"]),
    )
    print(
        "Output:",
        experiment_dir,
    )


if __name__ == "__main__":
    main()
