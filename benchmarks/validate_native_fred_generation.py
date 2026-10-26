#!/usr/bin/env python3
"""Validate a persisted native FRED generation run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.generate_fred_narrative_from_claims import (
    load_selected_claims,
    validate_selected_claims,
    validate_narrative_citations,
)


ROOT = Path(__file__).resolve().parents[1]

RESULT_ROOT = (
    ROOT / "benchmarks/results/native_fred_generation"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate(run_id: str) -> dict:
    if (
        not run_id
        or run_id in (".", "..")
        or Path(run_id).name != run_id
    ):
        raise ValueError("Invalid run ID.")

    run = RESULT_ROOT / run_id

    plan = json.loads(
        (run / "run_plan.json").read_text()
    )

    checks = []
    failures = []

    def check(name: str, condition: bool) -> None:
        checks.append({
            "name": name,
            "passed": bool(condition),
        })

        if not condition:
            failures.append(name)

    try:
        prompt = (run / "prompt.txt").read_bytes()

        check(
            "Prompt SHA-256",
            sha256(prompt) == plan["prompt"]["sha256"],
        )

        check(
            "Run identity",
            plan["run_id"] == run_id,
        )

        check(
            "Prompt character count",
            len(prompt.decode("utf-8"))
            == plan["prompt"]["characters"],
        )

        claims_path = Path(
            plan["source"]["claims_path"]
        )

        check(
            "Source claims SHA-256",
            sha256(claims_path.read_bytes())
            == plan["source"]["claims_sha256"],
        )

        claims = load_selected_claims(claims_path)
        validate_selected_claims(claims)

        check(
            "Source claim count",
            len(claims) == plan["source"]["claim_count"],
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

        telemetry = json.loads(
            (
                run / "native_generation_telemetry.json"
            ).read_text()
        )

        check(
            "Narrative SHA-256",
            sha256(narrative_bytes)
            == metadata["narrative_sha256"],
        )

        check(
            "Model and decoding provenance",
            metadata["model_id"]
            == plan["inference"]["model_id"]
            and metadata["decoding"]
            == plan["inference"]
            and metadata["prompt_sha256"]
            == plan["prompt"]["sha256"],
        )

        prompt_ids = metadata["prompt_chat_token_ids"]

        check(
            "Chat prompt token count",
            len(prompt_ids)
            == metadata["prompt_chat_token_count"],
        )

        token_ids = metadata["generated_token_ids"]
        records = telemetry["tokens"]

        check(
            "Generated token counts",
            len(token_ids)
            == metadata["generated_token_count"]
            == len(records)
            and len(token_ids) > 0,
        )

        check(
            "Exact generated token identities",
            all(
                record["token_id"] == token_id
                and record["generation_token_index"] == i
                for i, (token_id, record)
                in enumerate(zip(token_ids, records))
            ),
        )

        check(
            "Generation-time telemetry provenance",
            telemetry["run_id"] == run_id
            and telemetry["telemetry_mode"]
            == "generation_time_raw_logits",
        )

        check(
            "Finite probability measurements",
            all(
                0 < record["probability"] <= 1
                and math.isfinite(record["logprob"])
                and math.isfinite(record["entropy_nats"])
                and record["entropy_nats"] >= 0
                for record in records
            ),
        )

        greedy = (
            plan["inference"]["do_sample"] is False
        )

        if greedy:
            check(
                "Greedy rank-1 consistency",
                all(
                    record["rank"] == 1
                    for record in records
                ),
            )

        validate_narrative_citations(
            narrative_text=narrative,
            selected_claims=claims,
        )

        check(
            "FRED citation contract",
            True,
        )

    except Exception as exc:
        failures.append(
            f"{type(exc).__name__}: {exc}"
        )

    report = {
        "schema_version":
            "native_generation_validation_v0_1",
        "run_id": run_id,
        "validated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
    }

    output = (
        run / "native_generation_validation.json"
    )

    # Write atomically so a partial report is not left
    # behind if writing is interrupted.
    temporary = output.with_suffix(".json.tmp")

    temporary.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    temporary.replace(output)

    print("NATIVE GENERATION VALIDATION")
    print("-" * 58)

    for item in checks:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"{status}: {item['name']}")

    for failure in failures:
        if failure not in [
            item["name"] for item in checks
        ]:
            print("FAIL:", failure)

    print()
    print("Overall:", report["status"])
    print("Report:", output)

    return report


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-id",
        default="llama3_bf16_smoke_001",
    )

    args = parser.parse_args()

    report = validate(args.run_id)

    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
