#!/usr/bin/env python3
"""Prepare a probability-blind semantic review packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RUN_ROOT = (
    ROOT / "benchmarks/results/native_fred_generation"
)

REVIEW_ROOT = (
    ROOT / "benchmarks/results/semantic_boundary_reviews"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-id",
        default="llama3_bf16_sep2026_12m_002",
    )

    args = parser.parse_args()
    run_id = args.run_id

    if (
        not run_id
        or run_id in (".", "..")
        or Path(run_id).name != run_id
    ):
        raise ValueError("Invalid run ID.")

    run = RUN_ROOT / run_id
    output_dir = REVIEW_ROOT / run_id

    if output_dir.exists():
        raise FileExistsError(
            f"Review already exists: {output_dir}"
        )

    plan = json.loads(
        (run / "run_plan.json").read_text()
    )

    validation = json.loads(
        (
            run / "native_generation_validation.json"
        ).read_text()
    )

    if (
        plan["run_id"] != run_id
        or validation["run_id"] != run_id
        or validation["status"] != "PASS"
    ):
        raise RuntimeError(
            "Run identity or validation failed."
        )

    narrative_bytes = (
        run / "fred_narrative.md"
    ).read_bytes()

    narrative = narrative_bytes.decode("utf-8")

    metadata = json.loads(
        (
            run / "native_generation_metadata.json"
        ).read_text()
    )

    narrative_hash = sha256(narrative_bytes)

    if narrative_hash != metadata["narrative_sha256"]:
        raise RuntimeError(
            "Narrative hash mismatch."
        )

    source_path = Path(
        plan["source"]["claims_path"]
    )

    if sha256(source_path.read_bytes()) != (
        plan["source"]["claims_sha256"]
    ):
        raise RuntimeError(
            "Evidence hash mismatch."
        )

    claims = json.loads(
        source_path.read_text()
    )

    marker = "## Commentary\n"

    if narrative.count(marker) != 1:
        raise RuntimeError(
            "Expected exactly one Commentary section."
        )

    body_start = (
        narrative.index(marker) + len(marker)
    )

    raw_body = narrative[body_start:]
    leading = len(raw_body) - len(raw_body.lstrip())

    commentary = raw_body.strip()
    narrative_body_start = body_start + leading

    # Preliminary sentence segmentation.
    # Human review will judge semantic boundaries.
    sentences = [
        part.strip()
        for part in re.split(
            r"(?<=[.!?])\s+",
            commentary,
        )
        if part.strip()
    ]

    if not sentences:
        raise RuntimeError(
            "No commentary sentences found."
        )

    if " ".join(sentences) != " ".join(
        commentary.split()
    ):
        raise RuntimeError(
            "Sentence segmentation mismatch."
        )

    review_rows = []
    cursor = 0

    for index, sentence in enumerate(
        sentences, start=1
    ):
        start = commentary.find(sentence, cursor)

        if start < 0:
            raise RuntimeError(
                "Sentence offset alignment failed."
            )

        end = start + len(sentence)
        cursor = end

        review_rows.append({
            "sentence_id":
                f"{run_id}__commentary_{index:02d}",
            "text": sentence,
            "commentary_char_start": start,
            "commentary_char_end": end,
            "narrative_char_start":
                narrative_body_start + start,
            "narrative_char_end":
                narrative_body_start + end,
            "review_status": "unreviewed",
            "semantic_annotations": [],
            "boundary_decision": None,
            "transitions": [],
            "review_note": "",
        })

    evidence = [
        {
            key: claim.get(key)
            for key in (
                "claim_id",
                "claim_text",
                "source_series",
                "source_observation_date",
                "comparison_window",
                "current_value",
                "prior_value",
                "delta_value",
            )
        }
        for claim in claims
    ]

    packet = {
        "schema_version":
            "native_semantic_boundary_review_v0_1",
        "run_id": run_id,
        "narrative_sha256": narrative_hash,
        "source_claims_sha256":
            plan["source"]["claims_sha256"],
        "blinding":
            "token_probabilities_not_included",
        "evidence": evidence,
        "commentary": commentary,
        "sentences": review_rows,
    }

    markdown = [
        "# Probability-Blind Semantic Review",
        "",
        f"Run: `{run_id}`",
        "",
        "## Supplied evidence",
        "",
    ]

    for claim in evidence:
        markdown.append(
            "- " + claim["claim_text"]
        )

    markdown.extend([
        "",
        "## Commentary for review",
        "",
    ])

    for index, row in enumerate(review_rows, 1):
        markdown.extend([
            f"### Sentence {index}",
            "",
            row["text"],
            "",
            "**Semantic annotations:** Unreviewed",
            "",
            "**Boundary decision:** Unreviewed",
            "",
        ])

    markdown.extend([
        "## Review rules",
        "",
        "Assess semantic meaning and supplied evidence.",
        "Do not inspect token probabilities before review.",
        "Allow multiple or no boundaries.",
        "Record uncertainty rather than forcing a transition.",
        "",
    ])

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    (output_dir / "review_draft_v0_1.json").write_text(
        json.dumps(
            packet,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    (output_dir / "review_packet.md").write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )

    print("PROBABILITY-BLIND SEMANTIC REVIEW")
    print("-" * 60)
    print("Run:", run_id)
    print("Sentences:", len(review_rows))
    print("Evidence claims:", len(evidence))
    print("Reviewed decisions: 0")
    print("Token telemetry accessed: NO")
    print("Output:", output_dir)

    for index, row in enumerate(review_rows, 1):
        print(f"\n[{index}] {row['text']}")


if __name__ == "__main__":
    main()
