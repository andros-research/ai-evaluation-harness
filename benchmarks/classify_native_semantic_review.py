#!/usr/bin/env python3
"""Classify native-generation sentences without exposing token telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Existing semantic modules use sibling-level imports.
BENCHMARKS = ROOT / "benchmarks"

if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from build_semantic_proposer_prompt import (
    build_model_visible_guide,
    compact_evidence,
    load_corpus,
)

from run_semantic_classifier import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_TEMPERATURE,
    build_prompt,
    call_ollama,
    validate_proposal,
    reference_annotations,
)

from run_semantic_dashboard_batch import (
    build_live_target,
)


CLASSIFIER_CONFIG = "compact_balanced_v0_1"

COMPACT_GUIDE_PATH = (
    ROOT
    / "docs/v1_9_semantic_annotation_guide_compact_v0_1.md"
)

REVIEWED_FIXTURE_PATH = (
    ROOT
    / "benchmarks/fixtures/semantic_dashboard_reviewed_v0_1.jsonl"
)

BALANCED_EXAMPLE_IDS = [
    "mistral_semantic_conservative_t0__commentary_01",
    "llama3_semantic_analytical_t0__commentary_05",
    "semantic_harvest_004",
]

RUN_ID = "llama3_bf16_sep2026_12m_002"

REVIEW_ROOT = (
    ROOT
    / "benchmarks/results/semantic_boundary_reviews"
)

GENERATION_ROOT = (
    ROOT
    / "benchmarks/results/native_fred_generation"
)

OUTPUT_ROOT = (
    ROOT
    / "benchmarks/results/native_semantic_classifier"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(".json.tmp")

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def load_balanced_examples():
    reviewed = [
        json.loads(line)
        for line in REVIEWED_FIXTURE_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    reviewed_by_id = {
        row["claim_id"]: row
        for row in reviewed
    }

    corpus_by_id = {
        row["annotation_id"]: row
        for row in load_corpus()
    }

    examples = []

    for example_id in BALANCED_EXAMPLE_IDS:
        if example_id in reviewed_by_id:
            row = reviewed_by_id[example_id]

            if row["human_review"]["review_status"] not in (
                "accepted",
                "corrected",
            ):
                raise RuntimeError(
                    f"Example not reviewed: {example_id}"
                )

            annotations = row[
                "human_review"
            ]["reviewed_annotations"]

            # Preserve all supplied source claims, while
            # omitting redundant nested metadata from
            # the model-facing few-shot evidence.
            evidence = []

            for source in row["evidence"]:
                payload = source.get("payload") or {}

                evidence.append({
                    "evidence_id":
                        source["evidence_id"],
                    "source_type":
                        source["source_type"],
                    "claim_text":
                        source.get("claim_text")
                        or payload.get("claim_text"),
                    "source_series":
                        payload.get("source_series"),
                    "observation_date":
                        payload.get("source_observation_date"),
                    "comparison_window":
                        source.get("comparison_window"),
                    "current_value":
                        source.get("current_value"),
                    "prior_value":
                        source.get("prior_value"),
                    "delta_value":
                        source.get("delta_value"),
                    "direction":
                        source.get("direction"),
                })

            claim_span = row["claim_span"]

        else:
            row = corpus_by_id[example_id]

            claim_span = row["statement"]["text"]
            evidence = compact_evidence(row)
            annotations = reference_annotations(row)

        if not annotations or not evidence:
            raise RuntimeError(
                f"Incomplete reviewed example: {example_id}"
            )

        examples.append({
            "claim_span": claim_span,
            "evidence": evidence,
            "reviewed_annotations": annotations,
        })

    assert len(examples) == 3

    return BALANCED_EXAMPLE_IDS.copy(), examples


def prepare_targets(run_id: str):
    review_path = (
        REVIEW_ROOT
        / run_id
        / "review_draft_v0_1.json"
    )

    packet = read_json(review_path)

    plan = read_json(
        GENERATION_ROOT / run_id / "run_plan.json"
    )

    if (
        packet["run_id"] != run_id
        or plan["run_id"] != run_id
    ):
        raise RuntimeError("Run identity mismatch.")

    if packet["blinding"] != (
        "token_probabilities_not_included"
    ):
        raise RuntimeError(
            "Unexpected review blinding state."
        )

    source_path = Path(
        plan["source"]["claims_path"]
    )

    if not source_path.is_absolute():
        source_path = ROOT / source_path

    elif not source_path.is_file():
        # Translate host paths into the mounted
        # repository location inside Docker.
        marker = "/ai-lab/"
        recorded = source_path.as_posix()

        if marker not in recorded:
            raise FileNotFoundError(
                f"Cannot relocate source path: {recorded}"
            )

        relative = recorded.split(marker, 1)[1]
        source_path = ROOT / relative

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Source claims not found: {source_path}"
        )

    source_bytes = source_path.read_bytes()

    if sha256(source_bytes) != (
        plan["source"]["claims_sha256"]
    ):
        raise RuntimeError(
            "Frozen evidence hash mismatch."
        )

    claims = json.loads(
        source_bytes.decode("utf-8")
    )

    narrative_bytes = (
        GENERATION_ROOT
        / run_id
        / "fred_narrative.md"
    ).read_bytes()

    if sha256(narrative_bytes) != (
        packet["narrative_sha256"]
    ):
        raise RuntimeError(
            "Narrative hash mismatch."
        )

    if len(packet["sentences"]) != 5:
        raise RuntimeError(
            "Expected five commentary sentences."
        )

    example_ids, examples = load_balanced_examples()

    guide_source = COMPACT_GUIDE_PATH.read_text(
        encoding="utf-8"
    )

    prepared = []

    for sentence in packet["sentences"]:
        claim_text = sentence["text"]

        target = build_live_target(
            claim_text=claim_text,
            claims=claims,
        )

        if any(
            example["claim_span"] == claim_text
            for example in examples
        ):
            raise RuntimeError(
                "Target appears in few-shot examples."
            )

        guide = build_model_visible_guide(
            guide_source,
            target_text=claim_text,
        )

        prompt = build_prompt(
            guide=guide,
            examples=examples,
            target=target,
        )

        prepared.append({
            "sentence_id": sentence["sentence_id"],
            "claim_text": claim_text,
            "example_ids": example_ids,
            "prompt": prompt,
            "prompt_sha256": sha256(
                prompt.encode("utf-8")
            ),
        })

    return prepared, packet


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-id",
        default=RUN_ID,
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call the existing Ollama classifier.",
    )

    args = parser.parse_args()

    if (
        not args.run_id
        or args.run_id in (".", "..")
        or Path(args.run_id).name != args.run_id
    ):
        raise ValueError("Invalid run ID.")

    prepared, packet = prepare_targets(
        args.run_id
    )

    print("NATIVE SEMANTIC CLASSIFICATION")
    print("=" * 64)
    print("Run:", args.run_id)
    print("Classifier:", DEFAULT_MODEL)
    print("Configuration:", CLASSIFIER_CONFIG)
    print("Temperature:", DEFAULT_TEMPERATURE)
    print("Sentences:", len(prepared))
    print("Telemetry accessed: NO")

    for index, item in enumerate(prepared, 1):
        print()
        print(
            f"[{index}/{len(prepared)}] "
            f"{item['sentence_id']}"
        )
        print("Text:", item["claim_text"])
        print("Examples:", item["example_ids"])
        print(
            "Prompt characters:",
            len(item["prompt"]),
        )
        print(
            "Prompt SHA-256:",
            item["prompt_sha256"],
        )

    if not args.execute:
        print()
        print("DRY RUN: PASS")
        print("Ollama calls made: 0")
        print("Machine proposals written: 0")
        return

    output_dir = OUTPUT_ROOT / args.run_id

    successes = 0
    failures = 0

    for item in prepared:
        output = (
            output_dir
            / item["sentence_id"]
            / "machine_proposal_v0_1.json"
        )

        if output.exists():
            existing = read_json(output)

            if (
                existing.get("sentence_id")
                != item["sentence_id"]
                or existing.get("prompt_sha256")
                != item["prompt_sha256"]
            ):
                raise RuntimeError(
                    f"Conflicting existing proposal: {output}"
                )

            print("SKIP existing:", item["sentence_id"])

            if existing.get("status") == "success":
                successes += 1
            else:
                failures += 1

            continue

        record = {
            "schema_version":
                "native_semantic_machine_proposal_v0_1",
            "run_id": args.run_id,
            "sentence_id": item["sentence_id"],
            "claim_span": item["claim_text"],
            "narrative_sha256":
                packet["narrative_sha256"],
            "source_claims_sha256":
                packet["source_claims_sha256"],
            "classifier_model": DEFAULT_MODEL,
            "classifier_config": CLASSIFIER_CONFIG,
            "guide_sha256": sha256(
                COMPACT_GUIDE_PATH.read_bytes()
            ),
            "temperature": DEFAULT_TEMPERATURE,
            "few_shot_example_ids": item["example_ids"],
            "prompt_sha256": item["prompt_sha256"],
            "status": "pending",
            "semantic_annotations": [],
        }

        try:
            result = call_ollama(
                prompt=item["prompt"],
                model=DEFAULT_MODEL,
                temperature=DEFAULT_TEMPERATURE,
                ollama_host=DEFAULT_OLLAMA_HOST,
                timeout_s=600,
            )

            annotations = validate_proposal(
                proposal=result["proposal"],
                claim_span=item["claim_text"],
            )

            record["status"] = "success"
            record["semantic_annotations"] = annotations
            record["ollama_metadata"] = result[
                "ollama_metadata"
            ]

            successes += 1
            print("SUCCESS:", item["sentence_id"])

        except Exception as exc:
            record["status"] = "failed"
            record["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

            failures += 1
            print("FAILED:", item["sentence_id"])

        write_json_atomic(output, record)

    print()
    print("Successful proposals:", successes)
    print("Failed proposals:", failures)
    print("Output:", output_dir)
    print("Original review packet: UNCHANGED")
    print("Token telemetry accessed: NO")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
