#!/usr/bin/env python3

from __future__ import annotations

import argparse

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

_PARSER = argparse.ArgumentParser(
    description=(
        "Build a sealed blind review packet "
        "for a frozen cross-asset experiment."
    )
)

_PARSER.add_argument(
    "--experiment-id",
    required=True,
)

_PARSER.add_argument(
    "--llama-run-id",
    required=True,
)

_ARGS = _PARSER.parse_args()


EXPERIMENT_ID = _ARGS.experiment_id

LLAMA_RUN_ID = _ARGS.llama_run_id

EXPERIMENT_DIR = (
    ROOT
    / "benchmarks"
    / "fixtures"
    / "macro_narrative_experiments"
    / EXPERIMENT_ID
)

GPT_ROOT = (
    ROOT
    / "benchmarks"
    / "results"
    / "macro_narrative_experiments"
    / EXPERIMENT_ID
)

LLAMA_ROOT = (
    ROOT
    / "benchmarks"
    / "results"
    / "native_fred_generation"
    / LLAMA_RUN_ID
)

OUT = (
    ROOT
    / "benchmarks"
    / "results"
    / "macro_narrative_reviews"
    / EXPERIMENT_ID
    / "blind_review_v0_1"
)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(
        data
    ).hexdigest()


def sha_path(path: Path) -> str:
    return sha_bytes(
        path.read_bytes()
    )


def write_json(
    path: Path,
    payload,
):
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
    if OUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {OUT}"
        )

    manifest_path = (
        EXPERIMENT_DIR
        / "experiment_manifest.json"
    )

    manifest = json.loads(
        manifest_path.read_text()
    )

    prompt_path = (
        EXPERIMENT_DIR
        / manifest["prompt"]["file"]
    )

    packet_path = (
        EXPERIMENT_DIR
        / "inputs"
        / "macro_state_packet.md"
    )

    prompt_text = (
        prompt_path.read_text(
            encoding="utf-8"
        )
    )

    prompt_sha = (
        sha_path(
            prompt_path
        )
    )

    expected_prompt_sha = (
        manifest[
            "prompt"
        ][
            "sha256"
        ]
    )

    if (
        prompt_sha
        != expected_prompt_sha
    ):
        raise RuntimeError(
            "Frozen prompt hash mismatch."
        )

    marker = (
        "SUPPLIED STATE\n"
        "=============="
    )

    if marker not in prompt_text:
        raise RuntimeError(
            "Could not isolate task contract."
        )

    task_contract = (
        prompt_text
        .split(
            marker,
            1,
        )[0]
        .strip()
    )

    sources = [
        {
            "source_id":
                "gpt_oss_20b_low",

            "path":
                (
                    GPT_ROOT
                    / "gpt_oss_20b_low"
                    / "generation_001"
                    / "fred_narrative.md"
                ),
        },

        {
            "source_id":
                "gpt_oss_20b_medium",

            "path":
                (
                    GPT_ROOT
                    / "gpt_oss_20b_medium"
                    / "generation_001"
                    / "fred_narrative.md"
                ),
        },

        {
            "source_id":
                "llama3_8b_bf16",

            "path":
                (
                    LLAMA_ROOT
                    / "fred_narrative.md"
                ),
        },
    ]

    for source in sources:
        if not source[
            "path"
        ].exists():
            raise FileNotFoundError(
                source["path"]
            )

        if (
            source["path"].stat().st_size
            == 0
        ):
            raise RuntimeError(
                f"Empty narrative: "
                f"{source['source_id']}"
            )

        source[
            "narrative_sha256"
        ] = sha_path(
            source["path"]
        )

        source[
            "text"
        ] = source[
            "path"
        ].read_text(
            encoding="utf-8"
        ).strip()

    # Observation-specific secret permutation.
    #
    # The assignment cannot be reconstructed from
    # public prompt hashes or source IDs. The resulting
    # identity mapping is persisted only in the sealed
    # mapping artifact.
    import secrets

    secrets.SystemRandom().shuffle(
        sources
    )

    labels = [
        "NARRATIVE_A",
        "NARRATIVE_B",
        "NARRATIVE_C",
    ]

    OUT.mkdir(
        parents=True,
        exist_ok=False,
    )

    mapping = {
        "schema_version":
            "cross_asset_blind_mapping_v0_1",

        "experiment_id":
            EXPERIMENT_ID,

        "mapping": {},

        "noncompleted_arms": [
            {
                "source_id":
                    "gpt_oss_20b_high",

                "reached_token_cap":
                    True,

                "generated_token_count":
                    4096,

                "analysis_token_count":
                    4093,

                "final_token_count":
                    0,
            }
        ],
    }

    sections = []

    sections.append(
        "# Blind Cross-Asset State Review"
    )

    sections.append("")

    sections.append(
        "Model identities are sealed. "
        "Review only the three completed "
        "final answers below."
    )

    sections.append("")

    sections.append(
        "One fourth arm was a predeclared "
        "noncompletion and is not included "
        "as a review narrative."
    )

    sections.append("")

    sections.append(
        "## Review criteria"
    )

    sections.append("")

    sections.append(
        "For each narrative assess:"
    )

    sections.append("")

    sections.append(
        "- selection: did it identify the "
        "most economically important configurations?"
    )

    sections.append(
        "- evidence discipline: are observations "
        "distinguished from interpretation?"
    )

    sections.append(
        "- cross-series reasoning: does it use "
        "synchronized evidence correctly?"
    )

    sections.append(
        "- calibration: are interpretations "
        "appropriately qualified?"
    )

    sections.append(
        "- omission: did it miss an important "
        "configuration visible in the state?"
    )

    sections.append(
        "- usefulness: does it generate worthwhile "
        "quantitative questions?"
    )

    sections.append("")

    sections.append(
        "Then assign an overall ranking "
        "1 = best, 3 = weakest."
    )

    sections.append("")

    sections.append(
        "## Frozen task contract"
    )

    sections.append("")

    sections.append(
        task_contract
    )

    sections.append("")

    sections.append(
        "## Frozen source state"
    )

    sections.append("")

    sections.append(
        packet_path.read_text(
            encoding="utf-8"
        ).strip()
    )

    sections.append("")

    for label, source in zip(
        labels,
        sources,
    ):
        mapping[
            "mapping"
        ][label] = {
            "source_id":
                source[
                    "source_id"
                ],

            "narrative_sha256":
                source[
                    "narrative_sha256"
                ],
        }

        sections.append(
            "---"
        )

        sections.append("")

        sections.append(
            f"# {label}"
        )

        sections.append("")

        sections.append(
            source[
                "text"
            ]
        )

        sections.append("")

        sections.append(
            "### Human review"
        )

        sections.append("")

        sections.append(
            "- Overall rank: "
        )

        sections.append(
            "- Selection quality: "
        )

        sections.append(
            "- Evidence discipline: "
        )

        sections.append(
            "- Economic reasoning: "
        )

        sections.append(
            "- Important omissions: "
        )

        sections.append(
            "- Best candidate quantitative test: "
        )

        sections.append(
            "- Notes: "
        )

        sections.append("")

    packet_text = (
        "\n".join(
            sections
        )
        + "\n"
    )

    packet_out = (
        OUT
        / "blind_review_packet.md"
    )

    packet_out.write_text(
        packet_text,
        encoding="utf-8",
    )

    mapping_path = (
        OUT
        / "BLIND_MAPPING_DO_NOT_OPEN.json"
    )

    write_json(
        mapping_path,
        mapping,
    )

    review_manifest = {
        "schema_version":
            "cross_asset_blind_review_manifest_v0_1",

        "experiment_id":
            EXPERIMENT_ID,

        "prompt_sha256":
            prompt_sha,

        "source_state_packet_sha256":
            sha_path(
                packet_path
            ),

        "completed_narrative_count":
            3,

        "noncompletion_count":
            1,

        "labels":
            labels,

        "blind_review_packet":
            packet_out.name,

        "blind_review_packet_sha256":
            sha_path(
                packet_out
            ),

        "sealed_mapping":
            mapping_path.name,

        "sealed_mapping_sha256":
            sha_path(
                mapping_path
            ),

        "analysis_channels_included":
            False,

        "telemetry_included":
            False,

        "model_identity_visible":
            False,
    }

    write_json(
        OUT
        / "review_manifest.json",
        review_manifest,
    )

    print("=" * 80)
    print(
        "BLIND CROSS-ASSET REVIEW PACKET BUILT"
    )
    print("=" * 80)

    print(
        "Experiment:",
        EXPERIMENT_ID,
    )

    print(
        "Completed narratives:",
        3,
    )

    print(
        "Noncompletions:",
        1,
    )

    print(
        "Packet SHA:",
        review_manifest[
            "blind_review_packet_sha256"
        ],
    )

    print(
        "Mapping:",
        (
            "SEALED — DO NOT OPEN "
            "UNTIL REVIEW FREEZE"
        ),
    )

    print(
        "Packet:",
        packet_out,
    )


if __name__ == "__main__":
    main()
