#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_ROOT = (
    ROOT
    / "benchmarks/fixtures/macro_narrative_experiments"
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_json(path: Path, obj):
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def normalize_annotation(a: dict):
    # Old September schema already uses these names.
    if "claim_kind" in a:
        return {
            "claim_kind":
                a["claim_kind"],
            "support_status":
                a["support_status"],
            "assertion_strength":
                a["assertion_strength"],
            "proposition":
                a.get("proposition"),
            "rationale":
                a.get("rationale"),
        }

    # New August human-review schema.
    return {
        "claim_kind":
            a["kind"],

        "support_status":
            a["support"],

        "assertion_strength":
            a["assertion_strength"],

        "proposition":
            a.get(
                "commitment"
            ),

        "rationale":
            a.get(
                "rationale"
            ),
    }


def main():
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

    human_dir = (
        exp_dir
        / "human_review_v0_1"
    )

    old_ref = (
        human_dir
        / "human_macro_semantic_reference_v0_1.json"
    )

    old_manifest = (
        human_dir
        / "human_reference_freeze_manifest_v0_1.json"
    )

    # September already has the canonical old schema.
    if (
        old_ref.exists()
        and old_manifest.exists()
    ):
        print(
            "Compatibility reference already exists:"
        )
        print(old_ref)
        return

    source_ref = (
        human_dir
        / "human_review_reference.json"
    )

    source_manifest = (
        human_dir
        / "freeze_manifest.json"
    )

    experiment_manifest = (
        exp_dir
        / "experiment_manifest.json"
    )

    for path in (
        source_ref,
        source_manifest,
        experiment_manifest,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    ref = read_json(
        source_ref
    )

    freeze = read_json(
        source_manifest
    )

    experiment = read_json(
        experiment_manifest
    )

    source_ref_sha = sha(
        source_ref.read_bytes()
    )

    if (
        freeze["reference_sha256"]
        != source_ref_sha
    ):
        raise RuntimeError(
            "New human reference does not "
            "match its freeze manifest."
        )

    if (
        ref["experiment_id"]
        != args.experiment_id
    ):
        raise RuntimeError(
            "Human reference experiment mismatch."
        )

    narratives = []

    annotation_count = 0
    sentence_count = 0

    for blind_label in (
        "NARRATIVE_A",
        "NARRATIVE_B",
        "NARRATIVE_C",
    ):
        narrative = (
            ref["narratives"][
                blind_label
            ]
        )

        sentences = []

        for sentence in narrative[
            "sentences"
        ]:
            annotations = [
                normalize_annotation(
                    a
                )
                for a
                in sentence[
                    "annotations"
                ]
            ]

            sentence_count += 1
            annotation_count += len(
                annotations
            )

            sentences.append({
                "sentence_id":
                    sentence[
                        "sentence_id"
                    ],

                "claim_span":
                    sentence[
                        "text"
                    ],

                "annotations":
                    annotations,
            })

        narratives.append({
            "blind_label":
                blind_label,

            "commentary_sentences":
                sentences,
        })

    claims_path = (
        exp_dir
        / experiment[
            "input"
        ][
            "frozen_claims_file"
        ]
    )

    narrative_prompt_path = (
        exp_dir
        / experiment[
            "prompt"
        ][
            "file"
        ]
    )

    packet_path = (
        ROOT
        / ref[
            "source_blind_packet"
        ]
    )

    if (
        sha(
            packet_path.read_bytes()
        )
        != ref[
            "source_blind_packet_sha256"
        ]
    ):
        raise RuntimeError(
            "Blind review packet changed."
        )

    compat = {
        "schema_version":
            "human_macro_semantic_reference_v0_1",

        "experiment_id":
            args.experiment_id,

        "reference_scope":
            "commentary_sentences",

        "frozen_source": {
            "blind_review_packet_path":
                ref[
                    "source_blind_packet"
                ],

            "blind_review_packet_sha256":
                ref[
                    "source_blind_packet_sha256"
                ],

            "selected_fred_claims_sha256":
                sha(
                    claims_path.read_bytes()
                ),

            "narrative_prompt_sha256":
                sha(
                    narrative_prompt_path.read_bytes()
                ),
        },

        "review_provenance": {
            "method":
                "collaborative_human_assisted_review",

            "status":
                "frozen",

            "limitations": [
                ref.get(
                    "blinding_note",
                    ""
                ),
                (
                    "Compatibility representation "
                    "derived from the newer frozen "
                    "human-review schema."
                ),
            ],

            "human_preference":
                ref[
                    "overall_ranking"
                ],

            "model_identity_mapping_accessed":
                False,

            "hidden_model_analysis_or_token_telemetry_accessed":
                False,

            "source_reference_file":
                str(
                    source_ref.relative_to(
                        ROOT
                    )
                ),

            "source_reference_sha256":
                source_ref_sha,
        },

        "ontology": {
            "source_guide":
                "docs/"
                "v1_9_semantic_annotation_guide_"
                "compact_v0_2_1.md",

            "config_id":
                "compact_balanced_v0_2_1",

            "source":
                "human_review",
        },

        "operational_notes": {
            "generation_arms_planned":
                4,

            "completed_narratives":
                3,

            "noncompletion_arms": [
                ref[
                    "operational_noncompletion"
                ][
                    "source_id"
                ]
            ],

            "reason":
                "Frozen generation token cap.",
        },

        "narratives":
            narratives,

        "totals": {
            "narratives":
                len(narratives),

            "commentary_sentences":
                sentence_count,

            "semantic_annotations":
                annotation_count,

            "commentary_sentences_by_narrative":
                {
                    narrative[
                        "blind_label"
                    ]:
                        len(
                            narrative[
                                "commentary_sentences"
                            ]
                        )
                    for narrative
                    in narratives
                },
        },
    }

    write_json(
        old_ref,
        compat,
    )

    compat_sha = sha(
        old_ref.read_bytes()
    )

    readable = (
        human_dir
        / "human_review_reference.md"
    )

    old_freeze = {
        "schema_version":
            "human_macro_semantic_reference_freeze_v0_1",

        "status":
            "source_verified_frozen",

        "experiment_id":
            args.experiment_id,

        "source_packet_path":
            ref[
                "source_blind_packet"
            ],

        "source_packet_sha256":
            ref[
                "source_blind_packet_sha256"
            ],

        "selected_fred_claims_sha256":
            compat[
                "frozen_source"
            ][
                "selected_fred_claims_sha256"
            ],

        "narrative_prompt_sha256":
            compat[
                "frozen_source"
            ][
                "narrative_prompt_sha256"
            ],

        "reference_file":
            str(
                old_ref.relative_to(
                    ROOT
                )
            ),

        "reference_sha256":
            compat_sha,

        "readable_review_file":
            str(
                readable.relative_to(
                    ROOT
                )
            ),

        "readable_review_sha256":
            (
                sha(
                    readable.read_bytes()
                )
                if readable.exists()
                else None
            ),

        "narratives":
            len(narratives),

        "commentary_sentences":
            sentence_count,

        "semantic_annotations":
            annotation_count,

        "blind_mapping_accessed":
            False,

        "model_analysis_or_telemetry_accessed":
            False,

        "review_method":
            "collaborative_human_assisted_review",

        "limitations": [
            ref.get(
                "blinding_note",
                ""
            ),
            (
                "Compatibility layer generated "
                "after the canonical human review "
                "was independently frozen."
            ),
        ],

        "source_new_reference_file":
            str(
                source_ref.relative_to(
                    ROOT
                )
            ),

        "source_new_reference_sha256":
            source_ref_sha,
    }

    write_json(
        old_manifest,
        old_freeze,
    )

    print("=" * 88)
    print("MACRO SEMANTIC REFERENCE COMPATIBILITY FREEZE")
    print("=" * 88)
    print(
        "Experiment:",
        args.experiment_id,
    )
    print(
        "Sentences:",
        sentence_count,
    )
    print(
        "Annotations:",
        annotation_count,
    )
    print(
        "Reference SHA:",
        compat_sha,
    )
    print(
        "Reference:",
        old_ref,
    )
    print(
        "Freeze manifest:",
        old_manifest,
    )


if __name__ == "__main__":
    main()
