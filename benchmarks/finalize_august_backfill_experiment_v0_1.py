#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.generate_fred_narrative_from_claims import (
    build_llm_prompt,
    load_selected_claims,
    validate_selected_claims,
)


AUG_ID = "v1_9_3_aug2026_backfill_multi_model_v0_1"
SEP_ID = "v1_9_3_sep2026_multi_model_v0_1"

BASE = (
    ROOT
    / "benchmarks/fixtures/macro_narrative_experiments"
)

AUG_ROOT = BASE / AUG_ID
SEP_ROOT = BASE / SEP_ID

CLAIMS_PATH = (
    AUG_ROOT
    / "inputs/selected_fred_claims.json"
)

PROMPT_PATH = (
    AUG_ROOT
    / "inputs/narrative_prompt.txt"
)

MANIFEST_PATH = (
    AUG_ROOT
    / "experiment_manifest.json"
)

SEP_MANIFEST_PATH = (
    SEP_ROOT
    / "experiment_manifest.json"
)

PROMPT_BUILDER_PATH = (
    ROOT
    / "benchmarks/generate_fred_narrative_from_claims.py"
)

PROMPT_VARIANT = "semantic_analytical"

# Preserve the same GPT-OSS Harmony system date
# used in the September comparison/replay.
GPT_OSS_SYSTEM_DATE = "2026-11-02"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def main() -> None:
    if not CLAIMS_PATH.exists():
        raise FileNotFoundError(CLAIMS_PATH)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    if not SEP_MANIFEST_PATH.exists():
        raise FileNotFoundError(SEP_MANIFEST_PATH)

    selected_claims = load_selected_claims(
        CLAIMS_PATH
    )

    validate_selected_claims(
        selected_claims
    )

    if len(selected_claims) != 4:
        raise RuntimeError(
            f"Expected 4 August claims; got {len(selected_claims)}"
        )

    prompt = build_llm_prompt(
        selected_claims,
        PROMPT_VARIANT,
    )

    PROMPT_PATH.write_text(
        prompt,
        encoding="utf-8",
    )

    manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    sep_manifest = json.loads(
        SEP_MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    # Copy the actual generation-arm contract from
    # September rather than reconstructing it by hand.
    generation_arms = sep_manifest.get(
        "generation_arms"
    )

    if not generation_arms:
        raise RuntimeError(
            "September manifest has no generation_arms."
        )

    manifest["schema_version"] = (
        "macro_narrative_experiment_v0_1"
    )

    manifest["purpose"] = (
        "Retrospective August 2026 backfill using "
        "the frozen v1.9.3 September multi-model "
        "generation contract."
    )

    manifest["execution_mode"] = (
        "retrospective_backfill"
    )

    manifest["observation_date"] = (
        "2026-08-01"
    )

    manifest["reference_experiment_id"] = (
        SEP_ID
    )

    manifest["input"] = {
        "source_claims_path":
            str(
                CLAIMS_PATH.relative_to(
                    ROOT
                )
            ),

        "frozen_claims_file":
            "inputs/selected_fred_claims.json",

        "claims_sha256":
            sha256_path(
                CLAIMS_PATH
            ),

        "claim_count":
            len(selected_claims),

        "provenance_file":
            "inputs/august_backfill_provenance.json",

        "retrieval_policy":
            "historical_recovery_from_frozen_fixtures",

        "current_fred_retrieval_used":
            False,
    }

    manifest["prompt"] = {
        "variant":
            PROMPT_VARIANT,

        "file":
            "inputs/narrative_prompt.txt",

        "sha256":
            sha256_bytes(
                prompt.encode(
                    "utf-8"
                )
            ),

        "characters":
            len(prompt),

        "builder":
            "benchmarks/"
            "generate_fred_narrative_from_claims.py",

        "builder_sha256":
            sha256_path(
                PROMPT_BUILDER_PATH
            ),
    }

    manifest["generation_arms"] = (
        generation_arms
    )

    manifest["generation_contract_reference"] = {
        "experiment_id":
            SEP_ID,

        "manifest_sha256":
            sha256_path(
                SEP_MANIFEST_PATH
            ),

        "same_prompt_variant":
            (
                sep_manifest.get(
                    "prompt",
                    {}
                ).get(
                    "variant"
                )
                == PROMPT_VARIANT
            ),

        "same_prompt_builder_sha256":
            (
                sep_manifest.get(
                    "prompt",
                    {}
                ).get(
                    "builder_sha256"
                )
                == manifest[
                    "prompt"
                ][
                    "builder_sha256"
                ]
            ),

        "gpt_oss_system_date":
            GPT_OSS_SYSTEM_DATE,
    }

    # Preserve review policy already frozen in
    # the August backfill manifest, but copy the
    # exact September form if one exists.
    for key in (
        "review_policy",
        "blind_review",
        "semantic_classifier",
    ):
        if key in sep_manifest:
            manifest[key] = (
                sep_manifest[key]
            )

    manifest["status"] = (
        "experiment_frozen_ready_for_generation"
    )

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 90)
    print("AUGUST BACKFILL EXPERIMENT FINALIZED")
    print("=" * 90)

    print("experiment:", AUG_ID)
    print(
        "claims SHA:",
        manifest["input"]["claims_sha256"],
    )
    print(
        "prompt SHA:",
        manifest["prompt"]["sha256"],
    )
    print(
        "prompt chars:",
        manifest["prompt"]["characters"],
    )
    print(
        "builder SHA:",
        manifest["prompt"]["builder_sha256"],
    )
    print(
        "same builder as September:",
        manifest[
            "generation_contract_reference"
        ][
            "same_prompt_builder_sha256"
        ],
    )
    print(
        "same prompt variant as September:",
        manifest[
            "generation_contract_reference"
        ][
            "same_prompt_variant"
        ],
    )
    print(
        "generation arms:",
        len(
            manifest["generation_arms"]
        ),
    )

    print()
    print("ARMS")

    for arm in manifest[
        "generation_arms"
    ]:
        print(
            " ",
            arm["arm_id"],
            "|",
            arm["model"],
            "| reasoning=",
            arm.get(
                "reasoning_effort"
            ),
            "| max_new_tokens=",
            arm.get(
                "max_new_tokens"
            ),
        )

    print()
    print(
        "fixture:",
        AUG_ROOT.relative_to(
            ROOT
        ),
    )


if __name__ == "__main__":
    main()
