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

OSS_RESULT_ROOT = (
    ROOT
    / "benchmarks"
    / "results"
    / "macro_narrative_experiments"
)

LLAMA_RESULT_ROOT = (
    ROOT
    / "benchmarks"
    / "results"
    / "native_fred_generation"
)

REVIEW_ROOT = (
    ROOT
    / "benchmarks"
    / "results"
    / "macro_narrative_reviews"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8")
    )




def resolve_llama_run(
    experiment_id: str,
):
    """
    Resolve the unique native Llama generation whose
    run_plan.json points back to this experiment.
    """

    matches = []

    if not LLAMA_RESULT_ROOT.exists():
        raise FileNotFoundError(
            f"Llama result root not found: "
            f"{LLAMA_RESULT_ROOT}"
        )

    for run_dir in sorted(
        LLAMA_RESULT_ROOT.iterdir()
    ):
        if not run_dir.is_dir():
            continue

        plan_path = (
            run_dir
            / "run_plan.json"
        )

        metadata_path = (
            run_dir
            / "native_generation_metadata.json"
        )

        narrative_path = (
            run_dir
            / "fred_narrative.md"
        )

        if not plan_path.exists():
            continue

        try:
            plan = read_json(
                plan_path
            )
        except Exception:
            continue

        experiment = (
            plan.get(
                "experiment",
                {}
            )
            or {}
        )

        if (
            experiment.get(
                "experiment_id"
            )
            != experiment_id
        ):
            continue

        if (
            experiment.get(
                "arm_id"
            )
            != "llama3_8b_bf16"
        ):
            continue

        if not metadata_path.exists():
            continue

        if not narrative_path.exists():
            continue

        matches.append(
            run_dir
        )

    if not matches:
        raise RuntimeError(
            "No completed native Llama run "
            f"found for experiment "
            f"{experiment_id!r}."
        )

    if len(matches) > 1:
        raise RuntimeError(
            "Multiple native Llama runs found "
            f"for experiment {experiment_id!r}: "
            + ", ".join(
                path.name
                for path in matches
            )
        )

    run_dir = matches[0]

    return (
        run_dir.name,
        run_dir,
    )

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment-id",
        required=True,
    )

    args = parser.parse_args()

    exp_dir = (
        EXPERIMENT_ROOT
        / args.experiment_id
    )

    manifest = read_json(
        exp_dir
        / "experiment_manifest.json"
    )

    raw_prompt_sha = (
        manifest["prompt"]["sha256"]
    )

    sources = []

    # ----------------------------------------------------------
    # Llama
    # ----------------------------------------------------------

    (
        llama_run_id,
        llama_dir,
    ) = resolve_llama_run(
        args.experiment_id
    )

    llama_meta = read_json(
        llama_dir
        / "native_generation_metadata.json"
    )

    if (
        llama_meta["prompt_sha256"]
        != raw_prompt_sha
    ):
        raise RuntimeError(
            "Llama prompt SHA mismatch."
        )

    llama_text = (
        llama_dir
        / "fred_narrative.md"
    ).read_text(
        encoding="utf-8"
    )

    sources.append({
        "source_id": "llama3_8b_bf16",
        "status": "completed",
        "narrative": llama_text,
        "narrative_sha256":
            sha256_bytes(
                llama_text.encode("utf-8")
            ),
    })

    # ----------------------------------------------------------
    # GPT-OSS low / medium / high
    # ----------------------------------------------------------

    for effort in (
        "low",
        "medium",
        "high",
    ):
        arm_id = (
            f"gpt_oss_20b_{effort}"
        )

        arm_dir = (
            OSS_RESULT_ROOT
            / args.experiment_id
            / arm_id
            / "generation_001"
        )

        meta = read_json(
            arm_dir
            / "generation_metadata.json"
        )

        if (
            meta["raw_prompt_sha256"]
            != raw_prompt_sha
        ):
            raise RuntimeError(
                f"{arm_id}: prompt SHA mismatch."
            )

        narrative = (
            arm_dir
            / "fred_narrative.md"
        ).read_text(
            encoding="utf-8"
        )

        status = (
            "completed"
            if narrative.strip()
            else "noncompletion"
        )

        sources.append({
            "source_id": arm_id,
            "status": status,
            "narrative": narrative,
            "narrative_sha256":
                meta["narrative_sha256"],
            "reached_token_cap":
                meta["reached_token_cap"],
            "analysis_token_count":
                meta["analysis_token_count"],
            "final_token_count":
                meta["final_token_count"],
        })

    completed = [
        x for x in sources
        if x["status"] == "completed"
    ]

    failed = [
        x for x in sources
        if x["status"] != "completed"
    ]

    if len(completed) != 3:
        raise RuntimeError(
            f"Expected 3 completed narratives, got {len(completed)}"
        )

    # ----------------------------------------------------------
    # Deterministic blind ordering.
    #
    # The assignment is derived from experiment ID + narrative SHA,
    # but the mapping is written separately and never printed.
    # ----------------------------------------------------------

    ranked = sorted(
        completed,
        key=lambda x: sha256_bytes(
            (
                args.experiment_id
                + "|"
                + x["narrative_sha256"]
            ).encode("utf-8")
        ),
    )

    blind_labels = [
        "NARRATIVE_A",
        "NARRATIVE_B",
        "NARRATIVE_C",
    ]

    review_dir = (
        REVIEW_ROOT
        / args.experiment_id
        / "blind_review_v0_1"
    )

    if review_dir.exists():
        raise FileExistsError(
            f"Review already exists: {review_dir}"
        )

    review_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    mapping = {}

    packet_parts = [
        "# Blind Macro Narrative Review",
        "",
        f"Experiment: `{args.experiment_id}`",
        "",
        "Review the narratives below without reference to model identity, "
        "reasoning effort, hidden analysis, or token telemetry.",
        "",
        "The same frozen September 2026 FRED evidence and narrative "
        "instruction were used for every completed generation.",
        "",
    ]

    for label, source in zip(
        blind_labels,
        ranked,
    ):
        mapping[label] = {
            "source_id":
                source["source_id"],
            "narrative_sha256":
                source["narrative_sha256"],
        }

        packet_parts.extend([
            "---",
            "",
            f"## {label}",
            "",
            source["narrative"].strip(),
            "",
        ])

    packet_parts.extend([
        "---",
        "",
        "## Operational note",
        "",
        "One additional generation arm did not produce a final narrative "
        "within its frozen generation budget and is therefore not included "
        "as a reviewable narrative.",
        "",
    ])

    packet = "\n".join(
        packet_parts
    )

    packet_path = (
        review_dir
        / "blind_review_packet.md"
    )

    packet_path.write_text(
        packet,
        encoding="utf-8",
    )

    mapping_path = (
        review_dir
        / "BLIND_MAPPING_DO_NOT_OPEN.json"
    )

    mapping_path.write_text(
        json.dumps(
            {
                "schema_version":
                    "macro_blind_mapping_v0_1",

                "experiment_id":
                    args.experiment_id,

                "mapping":
                    mapping,

                "noncompleted_arms": [
                    {
                        "source_id":
                            x["source_id"],

                        "reached_token_cap":
                            x.get(
                                "reached_token_cap"
                            ),

                        "analysis_token_count":
                            x.get(
                                "analysis_token_count"
                            ),

                        "final_token_count":
                            x.get(
                                "final_token_count"
                            ),
                    }
                    for x in failed
                ],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    review_manifest = {
        "schema_version":
            "macro_blind_review_packet_v0_1",

        "experiment_id":
            args.experiment_id,

        "raw_prompt_sha256":
            raw_prompt_sha,

        "completed_narrative_count":
            len(completed),

        "noncompletion_count":
            len(failed),

        "packet_file":
            "blind_review_packet.md",

        "packet_sha256":
            sha256_bytes(
                packet.encode("utf-8")
            ),

        "mapping_file":
            "BLIND_MAPPING_DO_NOT_OPEN.json",

        "review_policy": {
            "model_identity_hidden":
                True,

            "reasoning_effort_hidden":
                True,

            "analysis_hidden":
                True,

            "telemetry_hidden":
                True,
        },
    }

    (
        review_dir
        / "review_manifest.json"
    ).write_text(
        json.dumps(
            review_manifest,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("BLIND REVIEW PACKET CREATED")
    print("=" * 72)
    print("Experiment:", args.experiment_id)
    print("Completed narratives:", len(completed))
    print("Noncompletions:", len(failed))
    print("Packet:", packet_path)
    print(
        "Packet SHA-256:",
        review_manifest[
            "packet_sha256"
        ],
    )
    print()
    print(
        "Model mapping written but NOT displayed:"
    )
    print(mapping_path)


if __name__ == "__main__":
    main()
