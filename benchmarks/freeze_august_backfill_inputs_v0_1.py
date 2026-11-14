#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_ID = (
    "v1_9_3_aug2026_backfill_multi_model_v0_1"
)

OBSERVATION_DATE = "2026-08-01"
PRIOR_DATE = "2025-08-01"

SEP_EXPERIMENT_ID = (
    "v1_9_3_sep2026_multi_model_v0_1"
)

SEP_CLAIMS_PATH = (
    ROOT
    / "benchmarks/fixtures/macro_narrative_experiments"
    / SEP_EXPERIMENT_ID
    / "inputs/selected_fred_claims.json"
)

SOURCE_FIXTURE = (
    ROOT
    / "benchmarks/fixtures"
    / "semantic_dashboard_reviewed_v0_1.jsonl"
)

OUT_ROOT = (
    ROOT
    / "benchmarks/fixtures/macro_narrative_experiments"
    / EXPERIMENT_ID
)

INPUT_ROOT = OUT_ROOT / "inputs"

OUT_CLAIMS = (
    INPUT_ROOT
    / "selected_fred_claims.json"
)

OUT_PROVENANCE = (
    INPUT_ROOT
    / "august_backfill_provenance.json"
)

OUT_MANIFEST = (
    OUT_ROOT
    / "experiment_manifest.json"
)

TARGET_SERIES = (
    "CPIAUCSL",
    "UNRATE",
    "FEDFUNDS",
    "GS10",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_hash(obj) -> str:
    blob = json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return sha256_bytes(blob)


def walk(obj):
    if isinstance(obj, dict):
        yield obj

        for value in obj.values():
            yield from walk(value)

    elif isinstance(obj, list):
        for value in obj:
            yield from walk(value)


def extract_records(obj):
    if isinstance(obj, list):
        return obj, None

    if isinstance(obj, dict):
        for key in (
            "selected_claims",
            "claims",
            "records",
            "items",
        ):
            if isinstance(
                obj.get(key),
                list,
            ):
                return obj[key], key

    raise RuntimeError(
        "Could not identify selected-claims container."
    )


def load_august_claims():
    found = {}

    for line in SOURCE_FIXTURE.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        root_obj = json.loads(line)

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
                "__2026-08-01"
            ):
                continue

            series = (
                obj.get(
                    "source_series"
                )
            )

            if series not in TARGET_SERIES:
                continue

            found[series] = {
                "claim_id":
                    claim_id,

                "claim_text":
                    (
                        obj.get(
                            "claim_text"
                        )
                        or obj.get("text")
                    ),

                "source_series":
                    series,

                "source_series_name":
                    obj.get(
                        "source_series_name"
                    ),

                "direction":
                    obj.get(
                        "direction"
                    ),

                "prior_value":
                    obj.get(
                        "prior_value"
                    ),

                "current_value":
                    obj.get(
                        "current_value"
                    ),

                "delta_value":
                    obj.get(
                        "delta_value"
                    ),

                "units":
                    (
                        obj.get("units")
                        or obj.get(
                            "supporting_values",
                            {},
                        ).get("units")
                    ),
            }

    missing = (
        set(TARGET_SERIES)
        - set(found)
    )

    if missing:
        raise RuntimeError(
            f"Missing August claims: {sorted(missing)}"
        )

    return found


def main():
    INPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    sep_root = json.loads(
        SEP_CLAIMS_PATH.read_text(
            encoding="utf-8"
        )
    )

    sep_records, container_key = (
        extract_records(
            sep_root
        )
    )

    august = load_august_claims()

    templates = {}

    for claim in sep_records:
        series = claim.get(
            "source_series"
        )

        if series in TARGET_SERIES:
            templates[
                series
            ] = claim

    missing_templates = (
        set(TARGET_SERIES)
        - set(templates)
    )

    if missing_templates:
        raise RuntimeError(
            "September template missing series: "
            f"{sorted(missing_templates)}"
        )

    now = datetime.now(
        timezone.utc
    ).isoformat()

    rebuilt = []

    # Preserve September's deterministic selection order.
    for sep_claim in sep_records:

        series = sep_claim.get(
            "source_series"
        )

        if series not in TARGET_SERIES:
            continue

        source = august[
            series
        ]

        claim = copy.deepcopy(
            sep_claim
        )

        claim.update({
            "claim_id":
                source[
                    "claim_id"
                ],

            "claim_text":
                source[
                    "claim_text"
                ],

            "source_series":
                source[
                    "source_series"
                ],

            "source_series_name":
                source[
                    "source_series_name"
                ],

            "direction":
                source[
                    "direction"
                ],

            "prior_value":
                source[
                    "prior_value"
                ],

            "current_value":
                source[
                    "current_value"
                ],

            "delta_value":
                source[
                    "delta_value"
                ],

            "source_observation_date":
                OBSERVATION_DATE,

            "created_at":
                now,

            "selected_at":
                now,
        })

        if "supporting_values" in claim:
            claim[
                "supporting_values"
            ] = {
                "current_date":
                    OBSERVATION_DATE,

                "current_value":
                    source[
                        "current_value"
                    ],

                "delta_value":
                    source[
                        "delta_value"
                    ],

                "prior_date":
                    PRIOR_DATE,

                "prior_value":
                    source[
                        "prior_value"
                    ],

                "units":
                    source[
                        "units"
                    ],
            }

        rebuilt.append(
            claim
        )

    if len(rebuilt) != 4:
        raise RuntimeError(
            f"Expected 4 rebuilt claims; got {len(rebuilt)}"
        )

    # Preserve the September file's outer schema.
    if container_key is None:
        out_root = rebuilt
    else:
        out_root = copy.deepcopy(
            sep_root
        )

        out_root[
            container_key
        ] = rebuilt

    OUT_CLAIMS.write_text(
        json.dumps(
            out_root,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    normalized_claims = []

    for claim in rebuilt:
        normalized_claims.append({
            "claim_id":
                claim["claim_id"],

            "claim_text":
                claim["claim_text"],

            "source_series":
                claim[
                    "source_series"
                ],

            "source_series_name":
                claim.get(
                    "source_series_name"
                ),

            "direction":
                claim[
                    "direction"
                ],

            "prior_value":
                claim[
                    "prior_value"
                ],

            "current_value":
                claim[
                    "current_value"
                ],

            "delta_value":
                claim[
                    "delta_value"
                ],

            "units":
                claim.get(
                    "supporting_values",
                    {},
                ).get(
                    "units"
                ),
        })

    provenance = {
        "schema_version":
            "macro_backfill_input_provenance_v0_1",

        "experiment_id":
            EXPERIMENT_ID,

        "execution_mode":
            "retrospective_backfill",

        "observation_date":
            OBSERVATION_DATE,

        "comparison_window":
            "12m",

        "reference_experiment_id":
            SEP_EXPERIMENT_ID,

        "reference_claim_schema_path":
            str(
                SEP_CLAIMS_PATH.relative_to(
                    ROOT
                )
            ),

        "source_fixture":
            str(
                SOURCE_FIXTURE.relative_to(
                    ROOT
                )
            ),

        "source_fixture_sha256":
            sha256_path(
                SOURCE_FIXTURE
            ),

        "claim_count":
            len(rebuilt),

        "normalized_claim_sha256": {
            claim[
                "source_series"
            ]:
                canonical_hash(
                    claim
                )
            for claim
            in normalized_claims
        },

        "selected_claims_sha256":
            sha256_path(
                OUT_CLAIMS
            ),

        "created_at":
            now,

        "provenance_note": (
            "Historical August 2026 macro state "
            "recovered from frozen semantic fixtures. "
            "The four canonical claim payloads were "
            "audited across frozen fixtures before "
            "this retrospective backfill was created. "
            "No current FRED retrieval was used."
        ),
    }

    OUT_PROVENANCE.write_text(
        json.dumps(
            provenance,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version":
            "macro_narrative_backfill_experiment_manifest_v0_1",

        "experiment_id":
            EXPERIMENT_ID,

        "label":
            "August 2026 · retrospective backfill",

        "observation_date":
            OBSERVATION_DATE,

        "execution_mode":
            "retrospective_backfill",

        "status":
            "inputs_frozen",

        "selected_claim_count":
            4,

        "selected_claims_path":
            str(
                OUT_CLAIMS.relative_to(
                    ROOT
                )
            ),

        "selected_claims_sha256":
            provenance[
                "selected_claims_sha256"
            ],

        "reference_experiment_id":
            SEP_EXPERIMENT_ID,

        "planned_arms": [
            "llama3_8b_bf16",
            "gpt_oss_20b_low",
            "gpt_oss_20b_medium",
            "gpt_oss_20b_high",
        ],

        "review_policy": {
            "generations_per_arm":
                1,

            "human_review_before_semantic_classifier":
                True,

            "blind_model_identity":
                True,

            "semantic_classifier":
                "compact_balanced_v0_2_1",

            "semantic_classifier_reasoning":
                "low",
        },

        "created_at":
            now,
    }

    OUT_MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 90)
    print("AUGUST 2026 BACKFILL INPUT FREEZE")
    print("=" * 90)

    print(
        "experiment:",
        EXPERIMENT_ID,
    )

    print(
        "claims:",
        len(rebuilt),
    )

    print(
        "selected claims SHA:",
        sha256_path(
            OUT_CLAIMS
        ),
    )

    print()

    for claim in rebuilt:
        print(
            claim[
                "claim_id"
            ]
        )
        print(
            " ",
            claim[
                "claim_text"
            ]
        )

    print()
    print(
        "fixture root:",
        OUT_ROOT.relative_to(
            ROOT
        ),
    )


if __name__ == "__main__":
    main()
