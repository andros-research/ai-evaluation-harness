#!/usr/bin/env python3

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path.cwd()
BENCH = ROOT / "benchmarks"
sys.path.insert(0, str(BENCH))

from run_semantic_classifier import (
    call_ollama,
    validate_proposal,
)


MANIFEST_PATH = (
    ROOT
    / "benchmarks/fixtures/"
      "semantic_classifier_expanded_benchmark_v0_1.json"
)

MANIFEST_HASH_PATH = MANIFEST_PATH.with_suffix(
    ".sha256"
)

OUT_ROOT = (
    ROOT
    / "benchmarks/results/"
      "semantic_classifier_expanded_v0_1/"
      "llama3_70b"
)

WORKING_PATH = (
    OUT_ROOT
    / "generation_working_v0_1.json"
)

FROZEN_PATH = (
    OUT_ROOT
    / "generation_frozen_v0_1.json"
)

FROZEN_HASH_PATH = (
    OUT_ROOT
    / "generation_frozen_v0_1.sha256"
)


MODEL = "llama3:70b"
TEMPERATURE = 0.0
NUM_CTX = 8192
NUM_PREDICT = 1024
OLLAMA_HOST = "http://127.0.0.1:11434"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -------------------------------------------------------
# Verify benchmark manifest.
# -------------------------------------------------------

manifest_bytes = MANIFEST_PATH.read_bytes()

actual_manifest_hash = sha256_bytes(
    manifest_bytes
)

expected_manifest_hash = (
    MANIFEST_HASH_PATH
    .read_text(encoding="utf-8")
    .strip()
    .split()[0]
)

if actual_manifest_hash != expected_manifest_hash:
    raise RuntimeError(
        "Benchmark manifest hash mismatch."
    )

manifest = json.loads(
    manifest_bytes.decode("utf-8")
)

targets = manifest["primary_cases"]

if len(targets) != 23:
    raise RuntimeError(
        f"Expected 23 primary cases, got {len(targets)}"
    )


# -------------------------------------------------------
# Resume support.
# -------------------------------------------------------

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

if WORKING_PATH.exists():
    working = json.loads(
        WORKING_PATH.read_text(
            encoding="utf-8"
        )
    )

    if (
        working["benchmark_manifest_sha256"]
        != actual_manifest_hash
    ):
        raise RuntimeError(
            "Working artifact belongs to another manifest."
        )

    records = working["runs"]

else:
    records = []

completed_ids = {
    r["target_id"]
    for r in records
}


def write_working():
    artifact = {
        "schema_version":
            "llama70b_expanded_classifier_generation_v0_1",

        "status":
            "working",

        "benchmark_manifest_sha256":
            actual_manifest_hash,

        "model":
            MODEL,

        "temperature":
            TEMPERATURE,

        "num_ctx":
            NUM_CTX,

        "num_predict":
            NUM_PREDICT,

        "runs":
            records,
    }

    WORKING_PATH.write_text(
        json.dumps(
            artifact,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


# -------------------------------------------------------
# Generation.
# -------------------------------------------------------

print("=" * 80)
print("LLAMA 3 70B — EXPANDED SEMANTIC CLASSIFIER")
print("=" * 80)

print("cases:", len(targets))
print("model:", MODEL)
print("temperature:", TEMPERATURE)
print("num_ctx:", NUM_CTX)
print("num_predict:", NUM_PREDICT)
print()

for index, target in enumerate(
    targets,
    start=1,
):
    target_id = target["target_id"]

    if target_id in completed_ids:
        print(
            f"[{index:02d}/{len(targets)}] "
            f"SKIP {target_id}"
        )
        continue

    prompt = target["historical_prompt"]

    prompt_hash = sha256_bytes(
        prompt.encode("utf-8")
    )

    if (
        prompt_hash
        != target["historical_prompt_sha256"]
    ):
        raise RuntimeError(
            f"Prompt hash mismatch: {target_id}"
        )

    print()
    print("-" * 80)
    print(
        f"[{index:02d}/{len(targets)}] "
        f"{target_id}"
    )
    print("-" * 80)

    start = time.perf_counter()

    record = {
        "target_id":
            target_id,

        "claim_span":
            target["claim_span"],

        "historical_prompt_sha256":
            prompt_hash,

        "status":
            "pending",

        "proposal":
            None,

        "semantic_annotations":
            [],

        "validation_error":
            None,

        "ollama_metadata":
            None,
    }

    try:
        result = call_ollama(
            prompt=prompt,
            model=MODEL,
            temperature=TEMPERATURE,
            ollama_host=OLLAMA_HOST,
            timeout_s=900,
            num_ctx=NUM_CTX,
            num_predict=NUM_PREDICT,
        )

        proposal = result["proposal"]

        annotations = validate_proposal(
            proposal=proposal,
            claim_span=target[
                "claim_span"
            ],
        )

        metadata = (
            result.get("metadata")
            if isinstance(result, dict)
            else None
        )

        record["status"] = "success"
        record["proposal"] = proposal
        record[
            "semantic_annotations"
        ] = annotations
        record[
            "ollama_metadata"
        ] = metadata

        # Explicitly detect the old truncation problem.
        if isinstance(metadata, dict):
            prompt_eval_count = (
                metadata.get(
                    "prompt_eval_count"
                )
            )

            record[
                "prompt_eval_count"
            ] = prompt_eval_count

            record[
                "context_truncation_detected"
            ] = (
                prompt_eval_count
                is not None
                and prompt_eval_count
                >= NUM_CTX
            )

        else:
            record[
                "prompt_eval_count"
            ] = None
            record[
                "context_truncation_detected"
            ] = None

    except Exception as exc:
        record["status"] = "failure"
        record["validation_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    elapsed = (
        time.perf_counter() - start
    )

    record[
        "elapsed_seconds"
    ] = elapsed

    records.append(record)
    completed_ids.add(target_id)

    write_working()

    print(
        "status:",
        record["status"],
    )
    print(
        "seconds:",
        round(elapsed, 2),
    )
    print(
        "prompt_eval_count:",
        record.get(
            "prompt_eval_count"
        ),
    )
    print(
        "truncation:",
        record.get(
            "context_truncation_detected"
        ),
    )

    if record["status"] == "success":
        for ann in record[
            "semantic_annotations"
        ]:
            print(
                " ",
                (
                    ann["claim_kind"],
                    ann["support_status"],
                    ann["assertion_strength"],
                ),
            )
    else:
        print(
            "ERROR:",
            record["validation_error"],
        )


# -------------------------------------------------------
# Freeze only if all 23 attempted.
# -------------------------------------------------------

if len(records) != len(targets):
    raise RuntimeError(
        f"Only {len(records)}/{len(targets)} cases recorded."
    )

artifact = {
    "schema_version":
        "llama70b_expanded_classifier_generation_v0_1",

    "status":
        "generation_frozen_unscored",

    "benchmark_manifest_sha256":
        actual_manifest_hash,

    "model":
        MODEL,

    "temperature":
        TEMPERATURE,

    "num_ctx":
        NUM_CTX,

    "num_predict":
        NUM_PREDICT,

    "runs":
        records,
}

payload = (
    json.dumps(
        artifact,
        indent=2,
        ensure_ascii=False,
    )
    + "\n"
).encode("utf-8")

FROZEN_PATH.write_bytes(payload)

digest = sha256_bytes(payload)

FROZEN_HASH_PATH.write_text(
    digest
    + "  "
    + FROZEN_PATH.name
    + "\n",
    encoding="utf-8",
)

print()
print("=" * 80)
print("LLAMA GENERATION FROZEN — UNSCORED")
print("=" * 80)

print(
    "successes:",
    sum(
        r["status"] == "success"
        for r in records
    ),
)

print(
    "failures:",
    sum(
        r["status"] != "success"
        for r in records
    ),
)

print(
    "truncation flags:",
    sum(
        r.get(
            "context_truncation_detected"
        ) is True
        for r in records
    ),
)

print("\nartifact:")
print(FROZEN_PATH)

print("\nSHA-256:")
print(digest)
