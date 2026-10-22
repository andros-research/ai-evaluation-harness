#!/usr/bin/env python3
"""Prepare a reproducible native FRED generation run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.generate_fred_narrative_from_claims import (
    build_llm_prompt,
    load_selected_claims,
    validate_selected_claims,
    SUPPORTED_PROMPT_VARIANTS,
)


ROOT = Path(__file__).resolve().parents[1]

BASELINE = (
    ROOT
    / "benchmarks/fixtures/"
    / "token_telemetry_llama3_semantic_conservative_v0_1.json"
)

RESULT_ROOT = (
    ROOT
    / "benchmarks/results/native_fred_generation"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def baseline_claims_path() -> Path:
    """Resolve the selected FRED claims from our baseline."""
    baseline = json.loads(BASELINE.read_text())

    narrative = Path(
        baseline["source_generation"]["narrative_path"]
    )

    if not narrative.is_absolute():
        narrative = ROOT / narrative

    return (
        narrative.parent.parent
        / "fred_claims/selected_fred_claims.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--claims-path",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--prompt-variant",
        choices=SUPPORTED_PROMPT_VARIANTS,
        default="semantic_conservative",
    )

    parser.add_argument(
        "--run-id",
        default="llama3_bf16_smoke_001",
    )

    args = parser.parse_args()

    claims_path = (
        args.claims_path
        if args.claims_path is not None
        else baseline_claims_path()
    ).resolve()

    selected_claims = load_selected_claims(
        claims_path
    )

    validate_selected_claims(
        selected_claims
    )

    prompt = build_llm_prompt(
        selected_claims,
        args.prompt_variant,
    )

    output_dir = RESULT_ROOT / args.run_id

    if (
        not args.run_id
        or Path(args.run_id).name != args.run_id
        or args.run_id in (".", "..")
    ):
        raise ValueError("Invalid run ID.")

    if output_dir.exists():
        raise FileExistsError(
            f"Run already exists: {output_dir}"
        )

    plan = {
        "schema_version":
            "native_fred_generation_plan_v0_1",
        "run_id":
            args.run_id,
        "source": {
            "claims_path": str(claims_path),
            "claims_sha256": sha256(
                claims_path.read_bytes()
            ),
            "claim_count": len(selected_claims),
        },
        "prompt": {
            "variant": args.prompt_variant,
            "sha256": sha256(
                prompt.encode("utf-8")
            ),
            "characters": len(prompt),
            "file": "prompt.txt",
        },
        "inference": {
            "backend": "transformers_pytorch",
            "model_id":
                "meta-llama/Meta-Llama-3-8B-Instruct",
            "dtype": "bfloat16",
            "chat_template": "model_native",
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": 512,
        },
        "planned_outputs": {
            "narrative": "fred_narrative.md",
            "generation_metadata":
                "native_generation_metadata.json",
            "generation_telemetry":
                "native_generation_telemetry.json",
        },
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    (output_dir / "prompt.txt").write_text(
        prompt,
        encoding="utf-8",
    )

    (output_dir / "run_plan.json").write_text(
        json.dumps(
            plan,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print("NATIVE GENERATION RUN PREPARED")
    print("-" * 55)
    print("Run:", args.run_id)
    print("Model:", plan["inference"]["model_id"])
    print("Claims:", len(selected_claims))
    print("Prompt chars:", len(prompt))
    print("Prompt SHA-256:", plan["prompt"]["sha256"])
    print("Output:", output_dir)
    print("GPU inference performed: NO")


if __name__ == "__main__":
    main()
