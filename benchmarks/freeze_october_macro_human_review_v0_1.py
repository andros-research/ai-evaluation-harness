#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT = (
    "v1_9_3_oct2026_multi_model_v0_1"
)

EXP_DIR = (
    ROOT
    / "benchmarks"
    / "fixtures"
    / "macro_narrative_experiments"
    / EXPERIMENT
)

PACKET = (
    ROOT
    / "benchmarks"
    / "results"
    / "macro_narrative_reviews"
    / EXPERIMENT
    / "blind_review_v0_1"
    / "blind_review_packet.md"
)

OUT_DIR = (
    EXP_DIR
    / "human_review_v0_1"
)

REFERENCE = (
    OUT_DIR
    / "human_macro_semantic_reference_v0_1.json"
)

READABLE = (
    OUT_DIR
    / "human_macro_semantic_reference_v0_1.md"
)

FREEZE_MANIFEST = (
    OUT_DIR
    / "human_reference_freeze_manifest_v0_1.json"
)

CLAIMS = (
    EXP_DIR
    / "inputs"
    / "selected_fred_claims.json"
)

PROMPT = (
    EXP_DIR
    / "inputs"
    / "narrative_prompt.txt"
)

EXPECTED_PACKET_SHA = (
    "1652cc34db00f464d7e7f0d948112510"
    "70b16b85e5c41d5f1b979431e1b2d064"
)


def sha(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def write_json(
    path: Path,
    value,
):
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def ann(
    kind,
    support,
    strength,
    proposition,
    rationale,
):
    return {
        "claim_kind":
            kind,

        "support_status":
            support,

        "assertion_strength":
            strength,

        "proposition":
            proposition,

        "rationale":
            rationale,
    }


NARRATIVES = [
    {
        "blind_label":
            "NARRATIVE_A",

        "human_rank":
            1,

        "assessment":
            (
                "Preferred narrative. Coherent and useful, "
                "but contains unsupported persistence, "
                "labor-market, and policy/scope extensions."
            ),

        "commentary_sentences": [
            {
                "sentence_id":
                    "A_S01",

                "claim_span":
                    (
                        "Headline and core inflation both rose "
                        "modestly in October, indicating persistent "
                        "price pressures."
                    ),

                "annotations": [
                    ann(
                        "empirical_comparison",
                        "supported",
                        "asserted",
                        (
                            "Headline and core CPI both "
                            "increased in October."
                        ),
                        (
                            "The supplied monthly CPI records "
                            "show +0.4% headline and +0.3% core."
                        ),
                    ),
                    ann(
                        "magnitude_judgment",
                        "insufficient_evidence",
                        "asserted",
                        (
                            "The CPI increases were modest."
                        ),
                        (
                            "The evidence supplies the monthly "
                            "changes but no threshold defining "
                            "a modest move."
                        ),
                    ),
                    ann(
                        "scope_generalization",
                        "insufficient_evidence",
                        "qualified_inference",
                        (
                            "The CPI observations characterize "
                            "broader price pressures."
                        ),
                        (
                            "Two CPI monthly changes do not by "
                            "themselves establish the broader "
                            "condition of price pressures."
                        ),
                    ),
                    ann(
                        "temporal_pattern",
                        "insufficient_evidence",
                        "qualified_inference",
                        (
                            "Price pressures are persistent."
                        ),
                        (
                            "Persistent asserts continuity across "
                            "time that one monthly observation "
                            "does not establish."
                        ),
                    ),
                ],
            },
            {
                "sentence_id":
                    "A_S02",

                "claim_span":
                    (
                        "The unemployment rate fell slightly, "
                        "suggesting a tightening labor market."
                    ),

                "annotations": [
                    ann(
                        "empirical_comparison",
                        "supported",
                        "asserted",
                        "The unemployment rate decreased.",
                        (
                            "The supplied comparison is "
                            "4.2% to 4.1%."
                        ),
                    ),
                    ann(
                        "magnitude_judgment",
                        "insufficient_evidence",
                        "asserted",
                        (
                            "The unemployment decline was slight."
                        ),
                        (
                            "The 0.1 percentage-point move is "
                            "observed, but no magnitude threshold "
                            "is supplied."
                        ),
                    ),
                    ann(
                        "scope_generalization",
                        "insufficient_evidence",
                        "qualified_inference",
                        (
                            "The unemployment decline indicates "
                            "a tightening labor market."
                        ),
                        (
                            "One unemployment-rate observation "
                            "does not establish broader labor-"
                            "market tightening."
                        ),
                    ),
                ],
            },
            {
                "sentence_id":
                    "A_S03",

                "claim_span":
                    (
                        "Meanwhile, the federal funds rate and "
                        "10-year Treasury yield both advanced, "
                        "reflecting a shift toward higher short-term "
                        "and long-term borrowing costs."
                    ),

                "annotations": [
                    ann(
                        "empirical_comparison",
                        "supported",
                        "asserted",
                        (
                            "The monthly-average effective "
                            "federal funds rate increased."
                        ),
                        (
                            "The frozen comparison is "
                            "3.75% to 3.88%."
                        ),
                    ),
                    ann(
                        "empirical_comparison",
                        "supported",
                        "asserted",
                        (
                            "The monthly-average 10-year "
                            "Treasury yield increased."
                        ),
                        (
                            "The frozen comparison is "
                            "4.99% to 5.13%."
                        ),
                    ),
                    ann(
                        "scope_generalization",
                        "insufficient_evidence",
                        "qualified_inference",
                        (
                            "The two rate observations represent "
                            "a broader shift toward higher short- "
                            "and long-term borrowing costs."
                        ),
                        (
                            "EFFR and the 10-year Treasury yield "
                            "are specific rates; broader borrowing "
                            "costs are not directly supplied."
                        ),
                    ),
                ],
            },
            {
                "sentence_id":
                    "A_S04",

                "claim_span":
                    (
                        "These observations point to a combination "
                        "of inflationary and monetary tightening "
                        "dynamics."
                    ),

                "annotations": [
                    ann(
                        "scope_generalization",
                        "insufficient_evidence",
                        "qualified_inference",
                        (
                            "The supplied observations establish "
                            "broader inflationary dynamics."
                        ),
                        (
                            "The evidence is too narrow to establish "
                            "a broader inflation regime."
                        ),
                    ),
                    ann(
                        "policy_interpretation",
                        "insufficient_evidence",
                        "qualified_inference",
                        (
                            "The supplied observations indicate "
                            "monetary tightening dynamics."
                        ),
                        (
                            "The monthly-average EFFR increase does "
                            "not establish an October policy tightening "
                            "action, and the Treasury yield is not "
                            "itself the policy rate."
                        ),
                    ),
                ],
            },
        ],
    },

    {
        "blind_label":
            "NARRATIVE_B",

        "human_rank":
            3,

        "assessment":
            (
                "Weakest narrative. Introduces unsupported "
                "inflation moderation, an unemployment trend, "
                "and a tightening-policy interpretation."
            ),

        "commentary_sentences": [
            {
                "sentence_id":
                    "B_S01",

                "claim_span":
                    (
                        "The October inflation data shows a slight "
                        "moderation in price pressures, with headline "
                        "CPI increasing 0.4 percent and core CPI "
                        "increasing 0.3 percent."
                    ),

                "annotations": [
                    ann(
                        "empirical_comparison",
                        "supported",
                        "asserted",
                        (
                            "Headline CPI increased 0.4% and "
                            "core CPI increased 0.3% in October."
                        ),
                        (
                            "Both monthly changes are directly "
                            "supplied."
                        ),
                    ),
                    ann(
                        "temporal_pattern",
                        "insufficient_evidence",
                        "asserted",
                        (
                            "Price pressures moderated."
                        ),
                        (
                            "Moderation requires comparison with an "
                            "earlier inflation pace; no such prior "
                            "monthly CPI change is supplied."
                        ),
                    ),
                    ann(
                        "magnitude_judgment",
                        "insufficient_evidence",
                        "asserted",
                        (
                            "The moderation was slight."
                        ),
                        (
                            "Neither moderation nor a magnitude "
                            "threshold for slight is established."
                        ),
                    ),
                    ann(
                        "scope_generalization",
                        "insufficient_evidence",
                        "asserted",
                        (
                            "The CPI observations characterize "
                            "broader price pressures."
                        ),
                        (
                            "The statement generalizes from two CPI "
                            "monthly changes to broader conditions."
                        ),
                    ),
                ],
            },
            {
                "sentence_id":
                    "B_S02",

                "claim_span":
                    (
                        "The unemployment rate continued its downward "
                        "trend, decreasing by 0.1 percentage point to "
                        "4.1 percent."
                    ),

                "annotations": [
                    ann(
                        "empirical_comparison",
                        "supported",
                        "asserted",
                        (
                            "The unemployment rate decreased "
                            "by 0.1 percentage point to 4.1%."
                        ),
                        (
                            "The frozen comparison is "
                            "4.2% to 4.1%."
                        ),
                    ),
                    ann(
                        "temporal_pattern",
                        "insufficient_evidence",
                        "asserted",
                        (
                            "Unemployment continued a downward trend."
                        ),
                        (
                            "A single September-to-October comparison "
                            "does not establish a continuing trend."
                        ),
                    ),
                ],
            },
            {
                "sentence_id":
                    "B_S03",

                "claim_span":
                    (
                        "The effective federal funds rate and 10-year "
                        "Treasury yield also increased, indicating a "
                        "tightening monetary policy stance."
                    ),

                "annotations": [
                    ann(
                        "empirical_comparison",
                        "supported",
                        "asserted",
                        (
                            "The monthly-average effective federal "
                            "funds rate increased."
                        ),
                        (
                            "The frozen comparison is "
                            "3.75% to 3.88%."
                        ),
                    ),
                    ann(
                        "empirical_comparison",
                        "supported",
                        "asserted",
                        (
                            "The monthly-average 10-year "
                            "Treasury yield increased."
                        ),
                        (
                            "The frozen comparison is "
                            "4.99% to 5.13%."
                        ),
                    ),
                    ann(
                        "policy_interpretation",
                        "insufficient_evidence",
                        "qualified_inference",
                        (
                            "The two increases indicate a tighter "
                            "monetary-policy stance."
                        ),
                        (
                            "The evidence does not establish an "
                            "October policy tightening action; the "
                            "10-year Treasury yield is market-determined "
                            "and the EFFR comparison is a monthly average."
                        ),
                    ),
                ],
            },
        ],
    },

    {
        "blind_label":
            "NARRATIVE_C",

        "human_rank":
            2,

        "assessment":
            (
                "Second-ranked. Less ambitious and less useful "
                "than A, but still introduces unsupported "
                "persistence, labor tightening, and broad rate "
                "generalization."
            ),

        "commentary_sentences": [
            {
                "sentence_id":
                    "C_S01",

                "claim_span":
                    (
                        "Headline and core CPI both increased by "
                        "0.4 and 0.3 percent respectively in October."
                    ),

                "annotations": [
                    ann(
                        "empirical_comparison",
                        "supported",
                        "asserted",
                        (
                            "Headline and core CPI increased "
                            "0.4% and 0.3% respectively."
                        ),
                        (
                            "Both monthly changes are directly "
                            "supplied."
                        ),
                    ),
                ],
            },
            {
                "sentence_id":
                    "C_S02",

                "claim_span":
                    (
                        "The unemployment rate fell from "
                        "4.2 to 4.1 percent."
                    ),

                "annotations": [
                    ann(
                        "empirical_comparison",
                        "supported",
                        "asserted",
                        "The unemployment rate decreased.",
                        (
                            "The frozen comparison is "
                            "4.2% to 4.1%."
                        ),
                    ),
                ],
            },
            {
                "sentence_id":
                    "C_S03",

                "claim_span":
                    (
                        "These movements suggest persistent "
                        "inflationary pressures and a tightening "
                        "labor market."
                    ),

                "annotations": [
                    ann(
                        "scope_generalization",
                        "insufficient_evidence",
                        "qualified_inference",
                        (
                            "The CPI observations indicate broader "
                            "inflationary pressure."
                        ),
                        (
                            "Two monthly CPI changes alone do not "
                            "establish the broader inflation condition."
                        ),
                    ),
                    ann(
                        "temporal_pattern",
                        "insufficient_evidence",
                        "qualified_inference",
                        (
                            "Inflationary pressures are persistent."
                        ),
                        (
                            "Persistence requires evidence over "
                            "multiple periods."
                        ),
                    ),
                    ann(
                        "scope_generalization",
                        "insufficient_evidence",
                        "qualified_inference",
                        (
                            "The unemployment decline indicates "
                            "a tightening labor market."
                        ),
                        (
                            "A single unemployment-rate move does "
                            "not establish broader labor-market "
                            "tightening."
                        ),
                    ),
                ],
            },
            {
                "sentence_id":
                    "C_S04",

                "claim_span":
                    (
                        "The rise in the federal funds rate and "
                        "10-year Treasury yield indicates a shift "
                        "toward higher interest rates."
                    ),

                "annotations": [
                    ann(
                        "empirical_comparison",
                        "supported",
                        "asserted",
                        (
                            "The monthly-average effective "
                            "federal funds rate increased."
                        ),
                        (
                            "The frozen comparison is "
                            "3.75% to 3.88%."
                        ),
                    ),
                    ann(
                        "empirical_comparison",
                        "supported",
                        "asserted",
                        (
                            "The monthly-average 10-year "
                            "Treasury yield increased."
                        ),
                        (
                            "The frozen comparison is "
                            "4.99% to 5.13%."
                        ),
                    ),
                    ann(
                        "scope_generalization",
                        "insufficient_evidence",
                        "qualified_inference",
                        (
                            "The two observed rate increases "
                            "represent a broader shift toward "
                            "higher interest rates."
                        ),
                        (
                            "Two specific rates do not establish "
                            "the direction of interest rates "
                            "generally."
                        ),
                    ),
                ],
            },
        ],
    },
]


def main():
    if not PACKET.exists():
        raise FileNotFoundError(
            PACKET
        )

    actual_packet_sha = sha(
        PACKET
    )

    if (
        actual_packet_sha
        != EXPECTED_PACKET_SHA
    ):
        raise RuntimeError(
            "Blind packet SHA mismatch:\n"
            f"actual={actual_packet_sha}\n"
            f"expected={EXPECTED_PACKET_SHA}"
        )

    packet_text = PACKET.read_text(
        encoding="utf-8"
    )

    def normalize_review_text(
        value: str,
    ) -> str:
        # Generated narratives can contain Unicode
        # hyphen variants depending on model/tokenizer
        # rendering. Treat those as typographically
        # equivalent for source verification only.
        for char in (
            "\u2010",  # hyphen
            "\u2011",  # non-breaking hyphen
            "\u2012",  # figure dash
            "\u2013",  # en dash
            "\u2212",  # minus sign
        ):
            value = value.replace(
                char,
                "-",
            )

        return value

    normalized_packet_text = (
        normalize_review_text(
            packet_text
        )
    )

    # Verify every reviewed sentence against the
    # frozen blind packet. Unicode hyphen variants
    # are treated as typographically equivalent.
    for narrative in NARRATIVES:
        for sentence in narrative[
            "commentary_sentences"
        ]:
            if (
                normalize_review_text(
                    sentence["claim_span"]
                )
                not in normalized_packet_text
            ):
                raise RuntimeError(
                    "Reviewed sentence not found "
                    "in blind packet after Unicode "
                    "hyphen normalization: "
                    + sentence["sentence_id"]
                )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    now = datetime.now(
        timezone.utc
    ).isoformat()

    annotation_count = sum(
        len(
            sentence["annotations"]
        )
        for narrative in NARRATIVES
        for sentence in narrative[
            "commentary_sentences"
        ]
    )

    sentence_count = sum(
        len(
            narrative[
                "commentary_sentences"
            ]
        )
        for narrative in NARRATIVES
    )

    reference = {
        "schema_version":
            "macro_human_semantic_reference_v0_1",

        "experiment_id":
            EXPERIMENT,

        "reference_scope":
            (
                "Human semantic review of all "
                "commentary sentences in the three "
                "completed blind narratives."
            ),

        "frozen_source": {
            "blind_review_packet_path":
                str(
                    PACKET.relative_to(
                        ROOT
                    )
                ),

            "blind_review_packet_sha256":
                actual_packet_sha,

            "selected_fred_claims_sha256":
                sha(
                    CLAIMS
                ),

            "narrative_prompt_sha256":
                sha(
                    PROMPT
                ),
        },

        "review_provenance": {
            "method":
                (
                    "collaborative_human_assisted_review_"
                    "with_blind_model_identity"
                ),

            "status":
                "frozen",

            "limitations": [
                (
                    "Model identity mapping was not "
                    "accessed before freeze."
                ),
                (
                    "Hidden analysis and token telemetry "
                    "were not inspected before freeze."
                ),
                (
                    "Summary bullets were assessed "
                    "qualitatively but were not included "
                    "as classifier targets."
                ),
                (
                    "The packet header incorrectly refers "
                    "to September 2026 evidence; the "
                    "narratives and claims themselves are "
                    "October 2026."
                ),
                (
                    "The monthly-average EFFR comparison "
                    "is numerically valid but can be "
                    "economically misleading as a policy-"
                    "state representation."
                ),
            ],

            "human_preference": [
                "NARRATIVE_A",
                "NARRATIVE_C",
                "NARRATIVE_B",
            ],

            "model_identity_mapping_accessed":
                False,

            "hidden_model_analysis_or_token_telemetry_accessed":
                False,
        },

        "ontology": {
            "source_guide":
                (
                    "docs/"
                    "v1_9_semantic_annotation_guide_"
                    "compact_v0_2_1.md"
                ),

            "config_id":
                "compact_balanced_v0_2_1",

            "source":
                "frozen_v0_2_1_ontology",
        },

        "operational_notes": {
            "generation_arms_planned":
                4,

            "completed_narratives":
                3,

            "noncompletion_arms":
                1,

            "reason":
                (
                    "One generation exhausted the "
                    "4096-token generation budget "
                    "without producing a final channel."
                ),
        },

        "narratives":
            NARRATIVES,

        "totals": {
            "narratives":
                3,

            "commentary_sentences":
                sentence_count,

            "semantic_annotations":
                annotation_count,

            "commentary_sentences_by_narrative":
                {
                    n["blind_label"]:
                        len(
                            n[
                                "commentary_sentences"
                            ]
                        )
                    for n in NARRATIVES
                },
        },

        "frozen_at":
            now,
    }

    write_json(
        REFERENCE,
        reference,
    )

    lines = [
        "# Human Macro Narrative Review — October 2026",
        "",
        f"Experiment: `{EXPERIMENT}`",
        "",
        "**Blind ranking:** A > C > B",
        "",
        (
            f"Reviewed {sentence_count} commentary "
            f"sentences with {annotation_count} "
            "semantic annotations."
        ),
        "",
        (
            "**Important note:** the blind packet header "
            "incorrectly says September 2026. The frozen "
            "claims and narratives are October 2026; the "
            "header error did not expose model identity."
        ),
        "",
    ]

    for narrative in NARRATIVES:
        lines.extend([
            f"## {narrative['blind_label']}",
            "",
            narrative["assessment"],
            "",
        ])

        for sentence in narrative[
            "commentary_sentences"
        ]:
            lines.extend([
                f"### {sentence['sentence_id']}",
                "",
                f"> {sentence['claim_span']}",
                "",
            ])

            for a in sentence[
                "annotations"
            ]:
                lines.append(
                    "- "
                    f"`{a['claim_kind']}` / "
                    f"`{a['support_status']}` / "
                    f"`{a['assertion_strength']}` — "
                    f"{a['proposition']} "
                    f"{a['rationale']}"
                )

            lines.append("")

    READABLE.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version":
            "macro_human_reference_freeze_v0_1",

        "status":
            "frozen",

        "experiment_id":
            EXPERIMENT,

        "source_packet_path":
            str(
                PACKET.relative_to(
                    ROOT
                )
            ),

        "source_packet_sha256":
            actual_packet_sha,

        "selected_fred_claims_sha256":
            sha(
                CLAIMS
            ),

        "narrative_prompt_sha256":
            sha(
                PROMPT
            ),

        "reference_file":
            REFERENCE.name,

        "reference_sha256":
            sha(
                REFERENCE
            ),

        "readable_review_file":
            READABLE.name,

        "readable_review_sha256":
            sha(
                READABLE
            ),

        "narratives":
            3,

        "commentary_sentences":
            sentence_count,

        "semantic_annotations":
            annotation_count,

        "blind_mapping_accessed":
            False,

        "model_analysis_or_telemetry_accessed":
            False,

        "review_method":
            (
                "collaborative_human_assisted_review_"
                "with_blind_model_identity"
            ),

        "limitations":
            reference[
                "review_provenance"
            ][
                "limitations"
            ],
    }

    write_json(
        FREEZE_MANIFEST,
        manifest,
    )

    print("=" * 80)
    print("OCTOBER HUMAN SEMANTIC REFERENCE FROZEN")
    print("=" * 80)
    print(
        "Ranking: A > C > B"
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
        "Packet SHA:",
        actual_packet_sha,
    )
    print(
        "Reference SHA:",
        sha(
            REFERENCE
        ),
    )
    print(
        "Mapping accessed: NO"
    )
    print(
        "Telemetry inspected: NO"
    )


if __name__ == "__main__":
    main()
