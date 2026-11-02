#!/usr/bin/env python3

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
BENCH = ROOT / "benchmarks"
sys.path.insert(0, str(BENCH))

from build_semantic_proposer_prompt import build_model_visible_guide
from classify_native_semantic_review import (
    COMPACT_GUIDE_PATH,
    REVIEWED_FIXTURE_PATH,
    load_balanced_examples,
)
from run_semantic_classifier import build_prompt


OUT_PATH = (
    ROOT
    / "benchmarks/fixtures/"
      "semantic_classifier_expanded_benchmark_v0_1.json"
)

HASH_PATH = OUT_PATH.with_suffix(
    ".sha256"
)

CURRENT_KINDS = {
    "empirical_comparison",
    "unit_conversion",
    "temporal_pattern",
    "causal_explanation",
    "scope_generalization",
    "magnitude_judgment",
    "market_interpretation",
    "policy_interpretation",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(obj) -> str:
    payload = json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return sha256_bytes(payload)


rows = [
    json.loads(line)
    for line in REVIEWED_FIXTURE_PATH.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

example_ids, examples = load_balanced_examples()
example_id_set = set(example_ids)

guide_source = COMPACT_GUIDE_PATH.read_text(
    encoding="utf-8"
)

primary = []
legacy = []
excluded = []

for row in rows:
    claim_id = row["claim_id"]
    review = row.get("human_review", {})
    status = review.get("review_status")
    annotations = review.get(
        "reviewed_annotations"
    ) or []

    if claim_id in example_id_set:
        excluded.append({
            "claim_id": claim_id,
            "reason": "few_shot_example",
        })
        continue

    if status not in ("accepted", "corrected"):
        excluded.append({
            "claim_id": claim_id,
            "reason":
                f"review_status={status}",
        })
        continue

    if not annotations:
        excluded.append({
            "claim_id": claim_id,
            "reason": "no_reviewed_annotations",
        })
        continue

    target = {
        "statement": {
            "text": row["claim_span"],
        },
        "evidence": row["evidence"],
    }

    guide = build_model_visible_guide(
        guide_source,
        target_text=row["claim_span"],
    )

    prompt = build_prompt(
        guide=guide,
        examples=examples,
        target=target,
    )

    kinds = {
        a["claim_kind"]
        for a in annotations
    }

    unknown_kinds = sorted(
        kinds - CURRENT_KINDS
    )

    record = {
        "target_id":
            claim_id,

        "claim_span":
            row["claim_span"],

        "source_model":
            row.get("source_model"),

        "prompt_variant":
            row.get("prompt_variant"),

        "evidence":
            row["evidence"],

        "historical_prompt":
            prompt,

        "historical_prompt_sha256":
            sha256_bytes(
                prompt.encode("utf-8")
            ),

        # Human labels deliberately omitted.
        "human_reference_sha256":
            canonical_hash(
                annotations
            ),

        "human_annotation_count":
            len(annotations),
    }

    if unknown_kinds:
        record[
            "legacy_reason"
        ] = (
            "frozen_human_label_not_in_"
            "current_compact_ontology"
        )

        record[
            "legacy_claim_kinds"
        ] = unknown_kinds

        legacy.append(record)

    else:
        primary.append(record)


artifact = {
    "schema_version":
        "semantic_classifier_expanded_benchmark_v0_1",

    "classifier_config":
        "compact_balanced_v0_1",

    "guide_path":
        str(
            COMPACT_GUIDE_PATH.relative_to(ROOT)
        ),

    "guide_sha256":
        sha256_bytes(
            COMPACT_GUIDE_PATH.read_bytes()
        ),

    "few_shot_example_ids":
        example_ids,

    "primary_case_count":
        len(primary),

    "legacy_case_count":
        len(legacy),

    "excluded_case_count":
        len(excluded),

    "primary_cases":
        primary,

    "legacy_cases":
        legacy,

    "excluded":
        excluded,
}

payload = (
    json.dumps(
        artifact,
        indent=2,
        ensure_ascii=False,
    )
    + "\n"
).encode("utf-8")

OUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_PATH.write_bytes(payload)

digest = sha256_bytes(payload)

HASH_PATH.write_text(
    digest
    + "  "
    + OUT_PATH.name
    + "\n",
    encoding="utf-8",
)

print("=" * 80)
print("EXPANDED CLASSIFIER BENCHMARK FROZEN")
print("=" * 80)

print(
    "primary:",
    len(primary),
)
print(
    "legacy:",
    len(legacy),
)
print(
    "excluded:",
    len(excluded),
)

print("\nfew-shot:")
for x in example_ids:
    print(" ", x)

print("\nlegacy:")
for x in legacy:
    print(
        " ",
        x["target_id"],
        x["legacy_claim_kinds"],
    )

print("\nartifact:")
print(OUT_PATH)

print("\nSHA-256:")
print(digest)
