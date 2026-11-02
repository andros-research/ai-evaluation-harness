#!/usr/bin/env python3

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path.cwd()
BENCH = ROOT / "benchmarks"
sys.path.insert(0, str(BENCH))

from classify_native_semantic_review import REVIEWED_FIXTURE_PATH
from run_semantic_classifier import (
    annotation_key,
    compare_annotations,
)

MANIFEST_PATH = (
    ROOT
    / "benchmarks/fixtures/"
      "semantic_classifier_expanded_benchmark_v0_1.json"
)
MANIFEST_HASH_PATH = MANIFEST_PATH.with_suffix(".sha256")

LLAMA_PATH = (
    ROOT
    / "benchmarks/results/"
      "semantic_classifier_expanded_v0_1/"
      "llama3_70b/generation_frozen_v0_1.json"
)
LLAMA_HASH_PATH = LLAMA_PATH.with_suffix(".sha256")

OSS_PATH = (
    ROOT
    / "benchmarks/results/"
      "semantic_classifier_expanded_v0_1/"
      "gpt_oss_20b/generation_frozen_v0_1.json"
)
OSS_HASH_PATH = OSS_PATH.with_suffix(".sha256")

OUT_ROOT = (
    ROOT
    / "benchmarks/results/"
      "semantic_classifier_expanded_v0_1/scoring"
)
OUT_PATH = OUT_ROOT / "scored_comparison_v0_1.json"
OUT_HASH_PATH = OUT_ROOT / "scored_comparison_v0_1.sha256"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_file(path: Path, hash_path: Path) -> tuple[bytes, str]:
    payload = path.read_bytes()
    actual = sha256_bytes(payload)

    expected = (
        hash_path
        .read_text(encoding="utf-8")
        .strip()
        .split()[0]
    )

    if actual != expected:
        raise RuntimeError(
            f"Hash mismatch: {path}\n"
            f"expected={expected}\n"
            f"actual={actual}"
        )

    return payload, actual


def canonical_hash(obj) -> str:
    payload = json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return sha256_bytes(payload)


# ------------------------------------------------------------------
# Verify all frozen inputs.
# ------------------------------------------------------------------

manifest_bytes, manifest_hash = verify_file(
    MANIFEST_PATH,
    MANIFEST_HASH_PATH,
)

llama_bytes, llama_hash = verify_file(
    LLAMA_PATH,
    LLAMA_HASH_PATH,
)

oss_bytes, oss_hash = verify_file(
    OSS_PATH,
    OSS_HASH_PATH,
)

manifest = json.loads(manifest_bytes.decode("utf-8"))
llama = json.loads(llama_bytes.decode("utf-8"))
oss = json.loads(oss_bytes.decode("utf-8"))

print("FROZEN INPUT HASHES VERIFIED")
print("manifest:", manifest_hash)
print("llama:   ", llama_hash)
print("oss:     ", oss_hash)
print()


# ------------------------------------------------------------------
# Recover frozen human references and verify against manifest hashes.
# ------------------------------------------------------------------

reviewed_rows = [
    json.loads(line)
    for line in REVIEWED_FIXTURE_PATH.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

reviewed_by_id = {
    row["claim_id"]: row
    for row in reviewed_rows
}

targets = manifest["primary_cases"]

if len(targets) != 23:
    raise RuntimeError(
        f"Expected 23 primary cases, found {len(targets)}"
    )

human = {}

for target in targets:
    target_id = target["target_id"]

    annotations = (
        reviewed_by_id[target_id]
        ["human_review"]
        ["reviewed_annotations"]
    )

    actual_reference_hash = canonical_hash(
        annotations
    )

    expected_reference_hash = target[
        "human_reference_sha256"
    ]

    if actual_reference_hash != expected_reference_hash:
        raise RuntimeError(
            f"Human reference hash mismatch: {target_id}"
        )

    human[target_id] = annotations

print(
    "HUMAN REFERENCES VERIFIED:",
    len(human),
    "cases /",
    sum(len(x) for x in human.values()),
    "annotation triples",
)
print()


# ------------------------------------------------------------------
# Index frozen model runs.
# ------------------------------------------------------------------

llama_by_id = {
    r["target_id"]: r
    for r in llama["runs"]
}

oss_by_key = {
    (r["target_id"], r["reasoning_effort"]): r
    for r in oss["runs"]
}

conditions = [
    "llama3_70b",
    "gpt_oss_low",
    "gpt_oss_medium",
    "gpt_oss_high",
]


# ------------------------------------------------------------------
# Strict primary scoring.
# ------------------------------------------------------------------

def result_status(
    *,
    valid_proposal: bool,
    no_final: bool,
    comparison: dict,
) -> str:

    if not valid_proposal:
        if no_final:
            return "non_completion"
        return "protocol_failure"

    if comparison["exact_annotation_set_match"]:
        return "exact"

    if comparison["matched_annotations"]:
        return "partial"

    return "miss"


def duplicate_count(proposed: list[dict]) -> int:
    keys = [
        annotation_key(x)
        for x in proposed
    ]

    return len(keys) - len(set(keys))


def score_case(
    *,
    proposed: list[dict],
    reference: list[dict],
    valid_proposal: bool,
    no_final: bool = False,
    failure_reason=None,
) -> dict:

    comparison = compare_annotations(
        proposed=proposed,
        reference=reference,
    )

    return {
        "result_status": result_status(
            valid_proposal=valid_proposal,
            no_final=no_final,
            comparison=comparison,
        ),
        "valid_proposal": valid_proposal,
        "failure_reason": failure_reason,
        "matched_reference_annotations":
            len(comparison["matched_annotations"]),
        "reference_annotation_count":
            len(reference),
        "duplicate_proposed_annotations":
            duplicate_count(proposed),
        "comparison":
            comparison,
    }


results = {
    condition: {}
    for condition in conditions
}


# Llama.
for target in targets:
    target_id = target["target_id"]
    run = llama_by_id[target_id]
    reference = human[target_id]

    valid = (
        run.get("status") == "success"
        and isinstance(
            run.get("semantic_annotations"),
            list,
        )
    )

    proposed = (
        run["semantic_annotations"]
        if valid
        else []
    )

    results["llama3_70b"][target_id] = score_case(
        proposed=proposed,
        reference=reference,
        valid_proposal=valid,
        failure_reason=run.get("validation_error"),
    )


# GPT-OSS.
for target in targets:
    target_id = target["target_id"]
    reference = human[target_id]

    for effort in ("low", "medium", "high"):
        run = oss_by_key[(target_id, effort)]
        condition = f"gpt_oss_{effort}"

        valid = (
            run.get("parse_status") == "success"
            and isinstance(
                run.get("parsed_proposal"),
                dict,
            )
            and isinstance(
                run["parsed_proposal"].get("annotations"),
                list,
            )
        )

        proposed = (
            run["parsed_proposal"]["annotations"]
            if valid
            else []
        )

        no_final = (
            run.get("final_tokens", 0) == 0
            and run.get("reached_token_cap") is True
        )

        results[condition][target_id] = score_case(
            proposed=proposed,
            reference=reference,
            valid_proposal=valid,
            no_final=no_final,
            failure_reason=run.get("validation_error"),
        )

        results[condition][target_id].update({
            "analysis_tokens":
                run.get("analysis_tokens"),
            "final_tokens":
                run.get("final_tokens"),
            "generated_tokens":
                run.get("generated_tokens"),
            "elapsed_seconds":
                run.get("elapsed_seconds"),
        })


# ------------------------------------------------------------------
# Aggregate summaries.
# ------------------------------------------------------------------

summaries = {}

for condition in conditions:
    rows = list(
        results[condition].values()
    )

    summaries[condition] = {
        "cases": len(rows),

        "valid_proposals": sum(
            r["valid_proposal"]
            for r in rows
        ),

        "exact_cases": sum(
            r["result_status"] == "exact"
            for r in rows
        ),

        "partial_cases": sum(
            r["result_status"] == "partial"
            for r in rows
        ),

        "miss_cases": sum(
            r["result_status"] == "miss"
            for r in rows
        ),

        "protocol_failures": sum(
            r["result_status"] == "protocol_failure"
            for r in rows
        ),

        "non_completions": sum(
            r["result_status"] == "non_completion"
            for r in rows
        ),

        "matched_reference_annotations": sum(
            r["matched_reference_annotations"]
            for r in rows
        ),

        "total_reference_annotations": sum(
            r["reference_annotation_count"]
            for r in rows
        ),

        "proposal_only_annotations": sum(
            len(r["comparison"]["proposal_only"])
            for r in rows
        ),

        "reference_only_annotations": sum(
            len(r["comparison"]["reference_only"])
            for r in rows
        ),

        "duplicate_proposed_annotations": sum(
            r["duplicate_proposed_annotations"]
            for r in rows
        ),
    }


# ------------------------------------------------------------------
# Per-reference-label recall.
# ------------------------------------------------------------------

by_kind = {}

for condition in conditions:
    stats = defaultdict(
        lambda: {
            "matched": 0,
            "reference": 0,
        }
    )

    for target in targets:
        target_id = target["target_id"]

        reference = human[target_id]

        matched_set = {
            (
                x["claim_kind"],
                x["support_status"],
                x["assertion_strength"],
            )
            for x in results[
                condition
            ][target_id][
                "comparison"
            ][
                "matched_annotations"
            ]
        }

        for ann in reference:
            key = annotation_key(ann)
            kind = ann["claim_kind"]

            stats[kind]["reference"] += 1

            if key in matched_set:
                stats[kind]["matched"] += 1

    by_kind[condition] = dict(stats)


# ------------------------------------------------------------------
# Secondary salvage view:
# If GPT-OSS emitted parseable JSON + annotations but violated only
# the exact claim-span protocol, score those annotations separately.
# This does NOT alter the primary result.
# ------------------------------------------------------------------

semantic_only_salvage = {}

for effort in ("low", "medium", "high"):
    condition = f"gpt_oss_{effort}"

    matched = 0
    total = 0
    exact_cases = 0
    salvaged_cases = []

    for target in targets:
        target_id = target["target_id"]
        run = oss_by_key[(target_id, effort)]
        reference = human[target_id]

        proposed = []

        if (
            isinstance(
                run.get("parsed_proposal"),
                dict,
            )
            and isinstance(
                run["parsed_proposal"].get("annotations"),
                list,
            )
        ):
            proposed = (
                run["parsed_proposal"]["annotations"]
            )

        comparison = compare_annotations(
            proposed=proposed,
            reference=reference,
        )

        matched += len(
            comparison["matched_annotations"]
        )
        total += len(reference)

        if comparison[
            "exact_annotation_set_match"
        ]:
            exact_cases += 1

        if (
            run.get("parse_status") != "success"
            and proposed
        ):
            salvaged_cases.append({
                "target_id": target_id,
                "validation_error":
                    run.get("validation_error"),
                "comparison": comparison,
            })

    semantic_only_salvage[condition] = {
        "exact_cases": exact_cases,
        "cases": len(targets),
        "matched_reference_annotations": matched,
        "total_reference_annotations": total,
        "protocol_failed_cases_with_parseable_annotations":
            salvaged_cases,
    }


# ------------------------------------------------------------------
# Terminal summary.
# ------------------------------------------------------------------

print("=" * 112)
print("EXPANDED SEMANTIC CLASSIFIER — STRICT PRIMARY SCORE")
print("=" * 112)

print(
    f"{'CONDITION':22s}"
    f"{'VALID':>8s}"
    f"{'EXACT':>8s}"
    f"{'PARTIAL':>10s}"
    f"{'MISS':>8s}"
    f"{'PROTO':>8s}"
    f"{'NONCOMP':>10s}"
    f"{'LABELS':>12s}"
    f"{'DUPES':>8s}"
)

print("-" * 112)

for condition in conditions:
    s = summaries[condition]

    labels = (
        f"{s['matched_reference_annotations']}/"
        f"{s['total_reference_annotations']}"
    )

    print(
        f"{condition:22s}"
        f"{s['valid_proposals']:8d}"
        f"{s['exact_cases']:8d}"
        f"{s['partial_cases']:10d}"
        f"{s['miss_cases']:8d}"
        f"{s['protocol_failures']:8d}"
        f"{s['non_completions']:10d}"
        f"{labels:>12s}"
        f"{s['duplicate_proposed_annotations']:8d}"
    )


print()
print("=" * 112)
print("MATCHED REFERENCE LABELS BY CLAIM KIND")
print("=" * 112)

all_kinds = sorted({
    kind
    for condition in conditions
    for kind in by_kind[condition]
})

print(
    f"{'CLAIM KIND':28s}"
    + "".join(
        f"{c:>20s}"
        for c in conditions
    )
)

print("-" * (
    28 + 20 * len(conditions)
))

for kind in all_kinds:
    line = f"{kind:28s}"

    for condition in conditions:
        s = by_kind[condition].get(
            kind,
            {"matched": 0, "reference": 0},
        )

        cell = (
            f"{s['matched']}/"
            f"{s['reference']}"
        )

        line += f"{cell:>20s}"

    print(line)


print()
print("=" * 112)
print("NON-EXACT / FAILED CASE DETAILS")
print("=" * 112)

for target in targets:
    target_id = target["target_id"]

    statuses = {
        c: results[c][target_id]["result_status"]
        for c in conditions
    }

    if all(
        status == "exact"
        for status in statuses.values()
    ):
        continue

    print()
    print(target_id)

    print("  HUMAN:")
    for ann in human[target_id]:
        print(
            "   ",
            annotation_key(ann),
        )

    for condition in conditions:
        r = results[condition][target_id]

        print(
            f"\n  {condition}: "
            f"{r['result_status']}"
        )

        for x in r["comparison"]["matched_annotations"]:
            print(
                "    MATCH ",
                annotation_key(x),
            )

        for x in r["comparison"]["proposal_only"]:
            print(
                "    EXTRA ",
                annotation_key(x),
            )

        for x in r["comparison"]["reference_only"]:
            print(
                "    MISS  ",
                annotation_key(x),
            )

        if r["duplicate_proposed_annotations"]:
            print(
                "    DUPLICATES:",
                r["duplicate_proposed_annotations"],
            )

        if r["failure_reason"]:
            print(
                "    FAILURE:",
                r["failure_reason"],
            )


print()
print("=" * 112)
print("SEMANTIC-ONLY SALVAGE VIEW — GPT-OSS")
print("=" * 112)

for condition, s in semantic_only_salvage.items():
    print(
        f"{condition:22s} "
        f"exact={s['exact_cases']}/{s['cases']} "
        f"labels="
        f"{s['matched_reference_annotations']}/"
        f"{s['total_reference_annotations']} "
        f"salvaged_protocol_cases="
        f"{len(s['protocol_failed_cases_with_parseable_annotations'])}"
    )


# ------------------------------------------------------------------
# Freeze scoring artifact.
# ------------------------------------------------------------------

artifact = {
    "schema_version":
        "semantic_classifier_expanded_scored_v0_1",

    "source_hashes": {
        "benchmark_manifest": manifest_hash,
        "llama3_70b_generation": llama_hash,
        "gpt_oss_20b_generation": oss_hash,
    },

    "human_references_revealed":
        True,

    "primary_case_count":
        len(targets),

    "conditions":
        conditions,

    "results":
        results,

    "summaries":
        summaries,

    "by_claim_kind":
        by_kind,

    "semantic_only_salvage":
        semantic_only_salvage,
}

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

payload = (
    json.dumps(
        artifact,
        indent=2,
        ensure_ascii=False,
    )
    + "\n"
).encode("utf-8")

OUT_PATH.write_bytes(payload)

digest = sha256_bytes(payload)

OUT_HASH_PATH.write_text(
    digest
    + "  "
    + OUT_PATH.name
    + "\n",
    encoding="utf-8",
)

print()
print("=" * 112)
print("SCORED ARTIFACT FROZEN")
print("=" * 112)

print(OUT_PATH)
print(digest)
