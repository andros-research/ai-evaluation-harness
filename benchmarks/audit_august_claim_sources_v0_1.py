#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "benchmarks/fixtures"

OBSERVATION_SUFFIX = "__2026-08-01"

TARGET_SERIES = {
    "CPIAUCSL",
    "UNRATE",
    "FEDFUNDS",
    "GS10",
}


def walk(obj):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from walk(value)

    elif isinstance(obj, list):
        for value in obj:
            yield from walk(value)


def read_objects(path: Path):
    if path.suffix == ".jsonl":
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines():
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

    elif path.suffix == ".json":
        try:
            yield json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except json.JSONDecodeError:
            return


def value_from_claim(claim, key):
    if key in claim:
        return claim[key]

    supporting = claim.get(
        "supporting_values"
    )

    if isinstance(supporting, dict):
        return supporting.get(key)

    evidence = claim.get("evidence")

    if isinstance(evidence, dict):
        if key in evidence:
            return evidence[key]

    return None


def normalized_claim(claim):
    return {
        "claim_id":
            claim.get("claim_id"),

        "claim_text":
            (
                claim.get("claim_text")
                or claim.get("text")
            ),

        "source_series":
            (
                claim.get("source_series")
                or claim.get("series_id")
            ),

        "source_series_name":
            claim.get(
                "source_series_name"
            ),

        "direction":
            claim.get("direction"),

        "prior_value":
            value_from_claim(
                claim,
                "prior_value",
            ),

        "current_value":
            value_from_claim(
                claim,
                "current_value",
            ),

        "delta_value":
            value_from_claim(
                claim,
                "delta_value",
            ),

        "units":
            value_from_claim(
                claim,
                "units",
            ),
    }


def canonical_hash(obj):
    blob = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(
        blob
    ).hexdigest()


hits = defaultdict(
    lambda: defaultdict(set)
)

raw_examples = {}


for path in sorted(
    list(FIXTURES.rglob("*.json"))
    + list(FIXTURES.rglob("*.jsonl"))
):
    for root_obj in read_objects(path):
        for obj in walk(root_obj):

            claim_id = obj.get(
                "claim_id"
            )

            if not isinstance(
                claim_id,
                str,
            ):
                continue

            if not claim_id.endswith(
                OBSERVATION_SUFFIX
            ):
                continue

            if not any(
                series in claim_id
                for series
                in TARGET_SERIES
            ):
                continue

            normalized = (
                normalized_claim(obj)
            )

            digest = canonical_hash(
                normalized
            )

            hits[
                claim_id
            ][digest].add(
                str(
                    path.relative_to(
                        ROOT
                    )
                )
            )

            raw_examples[
                (claim_id, digest)
            ] = normalized


print("=" * 100)
print("AUGUST 2026 FROZEN CLAIM SOURCE AUDIT")
print("=" * 100)

print(
    "\nUnique target claim IDs:",
    len(hits),
)

for claim_id in sorted(hits):

    variants = hits[
        claim_id
    ]

    print()
    print("-" * 100)
    print(claim_id)
    print(
        "payload variants:",
        len(variants),
    )

    for i, (
        digest,
        sources,
    ) in enumerate(
        variants.items(),
        start=1,
    ):
        print()
        print(
            f"  VARIANT {i}"
        )
        print(
            "  sha256:",
            digest,
        )

        payload = raw_examples[
            (claim_id, digest)
        ]

        print(
            json.dumps(
                payload,
                indent=4,
                ensure_ascii=False,
            )
        )

        print(
            "  sources:"
        )

        for source in sorted(
            sources
        ):
            print(
                "   -",
                source,
            )


print()
print("=" * 100)

if len(hits) != 4:
    print(
        "WARNING: expected exactly "
        "4 August target claims."
    )

conflicts = {
    claim_id: variants
    for claim_id, variants
    in hits.items()
    if len(variants) > 1
}

if conflicts:
    print(
        "RESULT: payload conflicts "
        "exist and require review."
    )
else:
    print(
        "RESULT: each August claim "
        "has one canonical payload "
        "across frozen fixtures."
    )

print("=" * 100)
