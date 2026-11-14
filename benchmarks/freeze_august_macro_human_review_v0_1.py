#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_ID = (
    "v1_9_3_aug2026_backfill_multi_model_v0_1"
)

PACKET_DIR = (
    ROOT
    / "benchmarks/results/macro_narrative_reviews"
    / EXPERIMENT_ID
    / "blind_review_v0_1"
)

PACKET_PATH = (
    PACKET_DIR
    / "blind_review_packet.md"
)

MAPPING_PATH = (
    PACKET_DIR
    / "BLIND_MAPPING_DO_NOT_OPEN.json"
)

EXPECTED_PACKET_SHA = (
    "6d1a2163ff7394ef5dcfbaf18e983608"
    "f6c537ed9cbdd8d312821b28a1a1e9ed"
)

OUT_DIR = (
    ROOT
    / "benchmarks/fixtures/macro_narrative_experiments"
    / EXPERIMENT_ID
    / "human_review_v0_1"
)

REFERENCE_PATH = (
    OUT_DIR
    / "human_review_reference.json"
)

REFERENCE_MD = (
    OUT_DIR
    / "human_review_reference.md"
)

MANIFEST_PATH = (
    OUT_DIR
    / "freeze_manifest.json"
)

SHA_PATH = (
    OUT_DIR
    / "SHA256SUMS"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def ann(
    kind,
    support,
    strength,
    commitment,
    rationale,
):
    return {
        "kind": kind,
        "support": support,
        "assertion_strength": strength,
        "commitment": commitment,
        "rationale": rationale,
    }


def main():
    if not PACKET_PATH.exists():
        raise FileNotFoundError(
            PACKET_PATH
        )

    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            "Blind mapping file is missing. "
            "It is checked for existence only "
            "and is NOT opened."
        )

    actual_packet_sha = (
        sha256_path(PACKET_PATH)
    )

    if (
        actual_packet_sha
        != EXPECTED_PACKET_SHA
    ):
        raise RuntimeError(
            "Blind packet SHA mismatch:\n"
            f" expected {EXPECTED_PACKET_SHA}\n"
            f" actual   {actual_packet_sha}"
        )

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Frozen human-review directory "
            f"already exists: {OUT_DIR}"
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    review = {
        "schema_version":
            "macro_human_review_reference_v0_1",

        "experiment_id":
            EXPERIMENT_ID,

        "observation_date":
            "2026-08-01",

        "execution_mode":
            "retrospective_backfill",

        "source_blind_packet":
            str(
                PACKET_PATH.relative_to(
                    ROOT
                )
            ),

        "source_blind_packet_sha256":
            actual_packet_sha,

        "mapping_accessed_before_freeze":
            False,

        "blinding_status":
            "partially_unblinded_by_style",

        "blinding_note": (
            "The model mapping file remained unopened. "
            "However, recurring stylistic fingerprints "
            "made Narrative A and Narrative B recognizable "
            "to the reviewer during review. The ranking "
            "should therefore be treated as partially, "
            "not fully, blinded."
        ),

        "packet_note_error": {
            "present": True,
            "text": (
                "The packet incorrectly states that the "
                "same frozen September 2026 FRED evidence "
                "was used. The actual frozen evidence is "
                "August 2026, with claim IDs dated "
                "2026-08-01."
            ),
            "effect_on_review":
                "none_material",
        },

        "overall_ranking": [
            "NARRATIVE_C",
            "NARRATIVE_A",
            "NARRATIVE_B",
        ],

        "overall_assessment": {
            "NARRATIVE_A": (
                "Strongest cross-series reasoning, but "
                "several interpretive moves are relatively "
                "assertive and exceed the supplied evidence."
            ),

            "NARRATIVE_B": (
                "Generally reasonable evidence discipline "
                "through hedging, but comparatively generic "
                "and repeats unsupported temporal language "
                "around inflation."
            ),

            "NARRATIVE_C": (
                "Best balance of synthesis, concision, and "
                "readability, although accommodative-policy "
                "and still-strong-labor-market claims exceed "
                "the supplied evidence."
            ),
        },

        "narratives": {
            "NARRATIVE_A": {
                "sentences": [
                    {
                        "sentence_id": "A_S01",
                        "text": (
                            "The CPI year-over-year rate has risen, "
                            "indicating a modest uptick in headline inflation."
                        ),
                        "annotations": [
                            ann(
                                "empirical_comparison",
                                "supported",
                                "asserted",
                                "CPI year-over-year rate has risen",
                                "Directly supported by the supplied CPI comparison.",
                            ),
                            ann(
                                "magnitude_judgment",
                                "insufficient_evidence",
                                "asserted",
                                "modest uptick",
                                "The supplied endpoints establish direction and size, "
                                "but not the evaluative characterization 'modest'.",
                            ),
                        ],
                    },
                    {
                        "sentence_id": "A_S02",
                        "text": (
                            "Meanwhile, the unemployment rate has slipped, "
                            "suggesting continued labor market resilience."
                        ),
                        "annotations": [
                            ann(
                                "empirical_comparison",
                                "supported",
                                "asserted",
                                "unemployment rate has slipped",
                                "Directly supported by the supplied unemployment comparison.",
                            ),
                            ann(
                                "scope_generalization",
                                "insufficient_evidence",
                                "qualified_inference",
                                "labor market resilience",
                                "One unemployment-rate comparison does not establish "
                                "the condition of the broader labor market.",
                            ),
                            ann(
                                "temporal_pattern",
                                "insufficient_evidence",
                                "qualified_inference",
                                "continued resilience",
                                "The supplied evidence does not establish persistence.",
                            ),
                        ],
                    },
                    {
                        "sentence_id": "A_S03",
                        "text": (
                            "The federal funds rate has fallen, reflecting "
                            "accommodative policy, yet the 10-year Treasury "
                            "yield has risen, implying that bond investors are "
                            "demanding higher returns, possibly due to expectations "
                            "of future inflation or fiscal dynamics."
                        ),
                        "annotations": [
                            ann(
                                "empirical_comparison",
                                "supported",
                                "asserted",
                                "federal funds rate has fallen",
                                "Directly supported by the supplied fed-funds comparison.",
                            ),
                            ann(
                                "policy_interpretation",
                                "insufficient_evidence",
                                "asserted",
                                "reflecting accommodative policy",
                                "A lower effective federal funds rate does not by itself "
                                "establish the policy interpretation.",
                            ),
                            ann(
                                "empirical_comparison",
                                "supported",
                                "asserted",
                                "10-year Treasury yield has risen",
                                "Directly supported by the supplied 10-year comparison.",
                            ),
                            ann(
                                "market_interpretation",
                                "insufficient_evidence",
                                "qualified_inference",
                                "bond investors are demanding higher returns",
                                "The yield move alone does not establish investor motivation.",
                            ),
                            ann(
                                "causal_explanation",
                                "insufficient_evidence",
                                "qualified_possibility",
                                "expectations of future inflation",
                                "Possible explanation, but not established by the four claims.",
                            ),
                            ann(
                                "causal_explanation",
                                "insufficient_evidence",
                                "qualified_possibility",
                                "fiscal dynamics",
                                "Possible explanation, but no fiscal evidence was supplied.",
                            ),
                        ],
                    },
                    {
                        "sentence_id": "A_S04",
                        "text": (
                            "These mixed signals point to a complex interplay "
                            "between monetary policy, inflation expectations, "
                            "and labor market conditions."
                        ),
                        "annotations": [
                            ann(
                                "scope_generalization",
                                "insufficient_evidence",
                                "qualified_inference",
                                "mixed signals / broader macro configuration",
                                "The supplied observations can motivate synthesis, "
                                "but do not establish the broader state.",
                            ),
                            ann(
                                "causal_explanation",
                                "insufficient_evidence",
                                "qualified_inference",
                                "complex interplay between policy, inflation expectations, "
                                "and labor conditions",
                                "The relationships among these factors are not established "
                                "by the selected comparisons.",
                            ),
                        ],
                    },
                ],
            },

            "NARRATIVE_B": {
                "sentences": [
                    {
                        "sentence_id": "B_S01",
                        "text": (
                            "The recent data suggests a mixed bag of "
                            "economic indicators."
                        ),
                        "annotations": [
                            ann(
                                "empirical_comparison",
                                "supported",
                                "qualified_inference",
                                "mixed bag of economic indicators",
                                "A reasonable synthesis of differently directed supplied "
                                "observations, though analytically generic.",
                            ),
                        ],
                    },
                    {
                        "sentence_id": "B_S02",
                        "text": (
                            "While inflation continues to rise, the unemployment "
                            "rate has decreased, indicating a potentially improving "
                            "labor market."
                        ),
                        "annotations": [
                            ann(
                                "empirical_comparison",
                                "supported",
                                "asserted",
                                "inflation increased",
                                "The CPI year-over-year rate increased across the supplied endpoints.",
                            ),
                            ann(
                                "temporal_pattern",
                                "insufficient_evidence",
                                "asserted",
                                "inflation continues to rise",
                                "Two endpoints do not establish a continuing trajectory.",
                            ),
                            ann(
                                "empirical_comparison",
                                "supported",
                                "asserted",
                                "unemployment rate decreased",
                                "Directly supported by the supplied comparison.",
                            ),
                            ann(
                                "scope_generalization",
                                "insufficient_evidence",
                                "qualified_inference",
                                "potentially improving labor market",
                                "One unemployment-rate comparison does not establish "
                                "broader labor-market improvement.",
                            ),
                        ],
                    },
                    {
                        "sentence_id": "B_S03",
                        "text": (
                            "The effective federal funds rate has also decreased, "
                            "which could be a sign of monetary policy easing."
                        ),
                        "annotations": [
                            ann(
                                "empirical_comparison",
                                "supported",
                                "asserted",
                                "effective federal funds rate decreased",
                                "Directly supported by the supplied comparison.",
                            ),
                            ann(
                                "policy_interpretation",
                                "insufficient_evidence",
                                "qualified_possibility",
                                "monetary policy easing",
                                "Reasonably hedged, but policy intent is not established.",
                            ),
                        ],
                    },
                    {
                        "sentence_id": "B_S04",
                        "text": (
                            "However, the 10-year Treasury yield has increased, "
                            "which may indicate growing concerns about inflation "
                            "and the economy's growth prospects."
                        ),
                        "annotations": [
                            ann(
                                "empirical_comparison",
                                "supported",
                                "asserted",
                                "10-year Treasury yield increased",
                                "Directly supported by the supplied comparison.",
                            ),
                            ann(
                                "market_interpretation",
                                "insufficient_evidence",
                                "qualified_possibility",
                                "concerns about inflation and growth prospects",
                                "The yield move does not establish the market's motivation.",
                            ),
                            ann(
                                "temporal_pattern",
                                "insufficient_evidence",
                                "qualified_possibility",
                                "growing concerns",
                                "The evidence does not establish that such concerns "
                                "have been increasing over time.",
                            ),
                        ],
                    },
                ],
            },

            "NARRATIVE_C": {
                "sentences": [
                    {
                        "sentence_id": "C_S01",
                        "text": (
                            "The CPI rise signals that inflationary pressures are "
                            "still present, while the drop in the unemployment rate "
                            "suggests labor market resilience."
                        ),
                        "annotations": [
                            ann(
                                "empirical_comparison",
                                "supported",
                                "asserted",
                                "CPI rise",
                                "Directly supported by the CPI comparison.",
                            ),
                            ann(
                                "scope_generalization",
                                "insufficient_evidence",
                                "qualified_inference",
                                "inflationary pressures are present",
                                "Broader inflationary pressure is an interpretation "
                                "beyond the supplied CPI comparison.",
                            ),
                            ann(
                                "temporal_pattern",
                                "insufficient_evidence",
                                "qualified_inference",
                                "inflationary pressures are still present",
                                "The word 'still' asserts persistence not established "
                                "by the supplied evidence.",
                            ),
                            ann(
                                "empirical_comparison",
                                "supported",
                                "asserted",
                                "unemployment rate dropped",
                                "Directly supported by the unemployment comparison.",
                            ),
                            ann(
                                "scope_generalization",
                                "insufficient_evidence",
                                "qualified_inference",
                                "labor market resilience",
                                "The unemployment comparison alone does not establish "
                                "the broader labor-market condition.",
                            ),
                        ],
                    },
                    {
                        "sentence_id": "C_S02",
                        "text": (
                            "The concurrent decline in the federal funds rate "
                            "indicates accommodative monetary policy, yet the higher "
                            "10-year yield may reflect expectations of future tightening "
                            "or fiscal concerns."
                        ),
                        "annotations": [
                            ann(
                                "empirical_comparison",
                                "supported",
                                "asserted",
                                "federal funds rate declined",
                                "Directly supported by the supplied comparison.",
                            ),
                            ann(
                                "policy_interpretation",
                                "insufficient_evidence",
                                "asserted",
                                "accommodative monetary policy",
                                "The policy characterization is stronger than the evidence supports.",
                            ),
                            ann(
                                "empirical_comparison",
                                "supported",
                                "asserted",
                                "10-year yield is higher",
                                "Directly supported by the supplied comparison.",
                            ),
                            ann(
                                "market_interpretation",
                                "insufficient_evidence",
                                "qualified_possibility",
                                "expectations of future tightening",
                                "A plausible market interpretation, but not established.",
                            ),
                            ann(
                                "market_interpretation",
                                "insufficient_evidence",
                                "qualified_possibility",
                                "fiscal concerns",
                                "A plausible interpretation, but no fiscal evidence was supplied.",
                            ),
                        ],
                    },
                    {
                        "sentence_id": "C_S03",
                        "text": (
                            "These mixed signals could lead to a cautious stance "
                            "by investors, balancing concerns over inflation against "
                            "the backdrop of a still-strong labor market."
                        ),
                        "annotations": [
                            ann(
                                "market_interpretation",
                                "insufficient_evidence",
                                "qualified_possibility",
                                "cautious stance by investors",
                                "Investor positioning or stance is not established.",
                            ),
                            ann(
                                "market_interpretation",
                                "insufficient_evidence",
                                "qualified_possibility",
                                "investor concerns over inflation",
                                "The supplied claims do not establish investor concerns.",
                            ),
                            ann(
                                "scope_generalization",
                                "insufficient_evidence",
                                "qualified_inference",
                                "still-strong labor market",
                                "The unemployment-rate decline does not establish the "
                                "broader labor market as still strong.",
                            ),
                        ],
                    },
                ],
            },
        },

        "operational_noncompletion": {
            "source_id":
                "gpt_oss_20b_high",
            "status":
                "noncompletion",
            "generated_token_count":
                4096,
            "analysis_token_count":
                4093,
            "final_token_count":
                0,
            "reached_token_cap":
                True,
            "included_in_ranking":
                False,
        },

        "review_frozen_at":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    REFERENCE_PATH.write_text(
        json.dumps(
            review,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    md = f"""# August 2026 Macro Human Review Reference

Experiment: `{EXPERIMENT_ID}`

## Review status

- Mapping file accessed before freeze: **NO**
- Blinding status: **partially unblinded by style**
- Source blind packet SHA-256: `{actual_packet_sha}`
- Packet wording issue: stale reference to September evidence; actual evidence month is August 2026.

## Overall ranking

1. **NARRATIVE_C**
2. **NARRATIVE_A**
3. **NARRATIVE_B**

## Overall assessment

**Narrative C** — {review["overall_assessment"]["NARRATIVE_C"]}

**Narrative A** — {review["overall_assessment"]["NARRATIVE_A"]}

**Narrative B** — {review["overall_assessment"]["NARRATIVE_B"]}

## Operational noncompletion

GPT-OSS high reached the frozen 4,096-token generation budget with 4,093 analysis tokens and zero final tokens. It was excluded from the narrative ranking.

## Methodological note

The model mapping file remained unopened during annotation. However, recurring stylistic fingerprints made two narratives recognizable to the reviewer, so this August review is preserved as partially rather than fully blinded.
"""

    REFERENCE_MD.write_text(
        md,
        encoding="utf-8",
    )

    reference_sha = (
        sha256_path(
            REFERENCE_PATH
        )
    )

    md_sha = sha256_path(
        REFERENCE_MD
    )

    manifest = {
        "schema_version":
            "macro_human_review_freeze_manifest_v0_1",

        "experiment_id":
            EXPERIMENT_ID,

        "reference_file":
            str(
                REFERENCE_PATH.relative_to(
                    ROOT
                )
            ),

        "reference_sha256":
            reference_sha,

        "reference_markdown_file":
            str(
                REFERENCE_MD.relative_to(
                    ROOT
                )
            ),

        "reference_markdown_sha256":
            md_sha,

        "source_blind_packet_sha256":
            actual_packet_sha,

        "mapping_file_exists":
            True,

        "mapping_file_opened":
            False,

        "overall_ranking":
            review[
                "overall_ranking"
            ],

        "blinding_status":
            review[
                "blinding_status"
            ],

        "frozen_at":
            review[
                "review_frozen_at"
            ],
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_sha = (
        sha256_path(
            MANIFEST_PATH
        )
    )

    SHA_PATH.write_text(
        "\n".join([
            f"{reference_sha}  {REFERENCE_PATH.name}",
            f"{md_sha}  {REFERENCE_MD.name}",
            f"{manifest_sha}  {MANIFEST_PATH.name}",
        ])
        + "\n",
        encoding="utf-8",
    )

    annotation_count = sum(
        len(sentence["annotations"])
        for narrative
        in review["narratives"].values()
        for sentence
        in narrative["sentences"]
    )

    sentence_count = sum(
        len(narrative["sentences"])
        for narrative
        in review["narratives"].values()
    )

    print("=" * 88)
    print("AUGUST HUMAN REVIEW FROZEN")
    print("=" * 88)
    print("Experiment:", EXPERIMENT_ID)
    print("Narratives:", len(review["narratives"]))
    print("Commentary sentences:", sentence_count)
    print("Annotations:", annotation_count)
    print("Ranking:", " > ".join(review["overall_ranking"]))
    print("Blinding:", review["blinding_status"])
    print("Mapping opened: NO")
    print("Packet SHA:", actual_packet_sha)
    print("Reference SHA:", reference_sha)
    print("Manifest SHA:", manifest_sha)
    print("Output:", OUT_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
