#!/usr/bin/env python3
"""
Generate a simple claim-cited FRED narrative from selected FRED claims.

v1.6.2 scaffold:
- reads selected FRED claim artifacts
- emits a constrained markdown narrative
- every empirical bullet cites exactly one source claim ID
- writes durable narrative and metadata artifacts

This first version is deterministic/template-based. Later versions may add
LLM-generated narrative text while preserving the same citation contract.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NARRATIVE_SCHEMA_VERSION = "fred_narrative_v0_1"
GENERATION_METHOD = "deterministic_claim_bullets"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "fred_claims"
    / "selected_fred_claims.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "results" / "fred_narratives"


REQUIRED_SELECTED_FIELDS = [
    "claim_id",
    "claim_text",
    "source_series",
    "source_observation_date",
    "claim_type",
    "metric_name",
    "comparison_window",
    "direction",
    "supporting_values",
    "selection_rank",
    "selection_method",
    "selection_schema_version",
]

DEFAULT_MODE = "deterministic"
SUPPORTED_MODES = ["deterministic", "llm"]
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3"
DEFAULT_TIMEOUT_S = 600

DEFAULT_PROMPT_VARIANT = "hardened"
SUPPORTED_PROMPT_VARIANTS = [
    "weak",
    "intermediate",
    "hardened",
]
DEFAULT_TEMPERATURE = 0.0


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def load_selected_claims(path: Path) -> list[dict]:
    """Load selected FRED claims from JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Selected claims file does not exist: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, list):
        raise ValueError(f"Expected list of selected claim records in {path}")

    return payload


def validate_selected_claims(selected_claims: list[dict]) -> None:
    """Validate selected claims before narrative generation."""
    errors: list[str] = []

    for idx, claim in enumerate(selected_claims):
        claim_id = claim.get("claim_id", f"<row {idx}>")

        for field in REQUIRED_SELECTED_FIELDS:
            if field not in claim:
                errors.append(f"{claim_id}: missing required field {field!r}")
                continue

            value = claim[field]

            if value is None:
                errors.append(f"{claim_id}: required field {field!r} is None")
                continue

            if isinstance(value, str) and not value.strip():
                errors.append(f"{claim_id}: required field {field!r} is empty")

        if not isinstance(claim.get("supporting_values"), dict):
            errors.append(f"{claim_id}: supporting_values must be a dict")

        if not isinstance(claim.get("selection_rank"), int):
            errors.append(f"{claim_id}: selection_rank must be an integer")

    if errors:
        joined = "\n".join(f"- {err}" for err in errors)
        raise ValueError(f"Selected claim validation failed:\n{joined}")


def render_claim_bullet(claim: dict) -> str:
    """Render one selected claim as a cited narrative bullet."""
    claim_text = claim["claim_text"].rstrip(".")
    claim_id = claim["claim_id"]
    return f"- {claim_text}. [CLAIMS: {claim_id}]"


def build_llm_prompt(
    selected_claims: list[dict],
    prompt_variant: str,
) -> str:
    """Build an LLM prompt using the requested contract-strength variant."""
    claim_blocks: list[str] = []

    for claim in selected_claims:
        supporting_values = claim.get(
            "supporting_values",
            {},
        )
        claim_blocks.append(
            "\n".join(
                [
                    f"claim_id: {claim.get('claim_id')}",
                    f"claim_text: {claim.get('claim_text')}",
                    f"source_series: {claim.get('source_series')}",
                    f"metric_name: {claim.get('metric_name')}",
                    f"comparison_window: {claim.get('comparison_window')}",
                    f"current_value: {claim.get('current_value')}",
                    f"prior_value: {claim.get('prior_value')}",
                    f"delta_value: {claim.get('delta_value')}",
                    f"direction: {claim.get('direction')}",
                    (
                        "supporting_values: "
                        f"{json.dumps(supporting_values, sort_keys=True)}"
                    ),
                ]
            )
        )

    claims_text = "\n\n---\n\n".join(
        claim_blocks
    )
    expected_bullet_count = len(
        selected_claims
    )

    if prompt_variant == "weak":
        return f"""You are generating a concise macro narrative from selected FRED claims.

Write a short markdown summary describing the supplied macro claims.

Include the relevant source claim ID with each statement so the output can be traced back to the evidence.

Do not invent facts that are not present in the selected claims.

Selected claims:

{claims_text}
"""

    if prompt_variant == "intermediate":
        return f"""You are generating a concise macro narrative from selected FRED claims.

Output requirements:

1. Write markdown only.
2. Begin with:
# FRED Macro Narrative
3. Then include:
## Claim-Cited Summary
4. Write exactly {expected_bullet_count} bullets.
5. Write one bullet for each selected claim.
6. Each bullet must include:
   - the direction
   - the prior value
   - the current value
   - the delta magnitude
7. End each bullet with a citation using:
   [CLAIMS: <exact claim_id>]
8. Do not add facts that are not present in the selected claims.

Selected claims:

{claims_text}
"""

    if prompt_variant == "hardened":
        return f"""You are generating a concise macro narrative from selected FRED claims.

Output requirements:

1. Write markdown only.
2. Begin with exactly this title:
# FRED Macro Narrative
3. Then include exactly this section heading:
## Claim-Cited Summary
4. Write exactly {expected_bullet_count} bullets.
5. Write one bullet for each selected claim.
6. Do not combine two claims into one bullet.
7. Each bullet must include:
   - the direction
   - the prior value
   - the current value
   - the delta magnitude
8. Each bullet must end with exactly one citation using this exact syntax:
   [CLAIMS: <exact claim_id>]
9. Copy each claim_id exactly as supplied below.
10. Do not use a bare citation such as:
    [fred__example__ID]
11. Do not omit the literal prefix:
    CLAIMS:
12. Do not wrap claim IDs in backticks, quotation marks, parentheses, or extra brackets.
13. Do not add causal interpretation, policy interpretation, market interpretation, or qualitative conclusions.
14. Do not add facts not present in the selected claims.
15. Do not add any section after the claim-cited bullets.

Required bullet template:

- <metric> <direction> by <delta magnitude>, from <prior value> to <current value>. [CLAIMS: <exact claim_id>]

Selected claims:

{claims_text}
"""

    raise ValueError(
        "Unsupported prompt variant: "
        f"{prompt_variant}"
    )


def build_narrative_markdown(
    selected_claims: list[dict],
    generated_at: str,
) -> str:
    """Build a deterministic claim-cited markdown narrative."""
    if not selected_claims:
        return (
            "# FRED Macro Narrative\n\n"
            f"Generated at: {generated_at}\n\n"
            "No selected FRED claims were available for narrative generation.\n"
        )

    comparison_windows = sorted(
        {claim.get("comparison_window") for claim in selected_claims if claim.get("comparison_window")}
    )
    observation_dates = sorted(
        {claim.get("source_observation_date") for claim in selected_claims if claim.get("source_observation_date")}
    )

    window_text = ", ".join(comparison_windows)
    date_text = ", ".join(observation_dates)

    bullets = "\n".join(render_claim_bullet(claim) for claim in selected_claims)

    return (
        "# FRED Macro Narrative\n\n"
        f"Generated at: {generated_at}\n\n"
        "## Context\n\n"
        f"This deterministic narrative uses selected FRED-native claims for comparison window(s): {window_text}.\n"
        f"The current source observation date is: {date_text}.\n\n"
        "## Claim-Cited Summary\n\n"
        f"{bullets}\n"
    )


def build_llm_narrative_markdown(
    *,
    selected_claims: list[dict],
    generated_at: str,
    model: str,
    ollama_host: str,
    timeout_s: int,
    prompt_variant: str,
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    """Build an LLM-generated claim-cited markdown narrative."""
    if not selected_claims:
        narrative = (
            "# FRED Macro Narrative\n\n"
            f"Generated at: {generated_at}\n\n"
            "No selected FRED claims were available for narrative generation.\n"
        )
        return narrative, {
            "llm_used": False,
            "model": model,
            "ollama_host": ollama_host,
            "prompt_variant": prompt_variant,
            "temperature": temperature,
            "elapsed_s": 0,
            "error": "",
        }

    prompt = build_llm_prompt(
        selected_claims,
        prompt_variant,
    )

    result = ollama_generate(
        host=ollama_host,
        model=model,
        prompt=prompt,
        options={
            "temperature": temperature,
            "top_p": 0.9,
            "num_predict": 512,
        },
        timeout_s=timeout_s,
    )

    if not result["ok"]:
        raise RuntimeError(f"LLM narrative generation failed: {result['error']}")

    narrative = result["response_text"].strip()

    return narrative + "\n", {
        "llm_used": True,
        "model": model,
        "ollama_host": ollama_host,
        "prompt_variant": prompt_variant,
        "temperature": temperature,
        "elapsed_s": result["elapsed_s"],
        "error": result["error"],
    }


def extract_claim_ids(selected_claims: list[dict]) -> list[str]:
    """Return claim IDs in selected order."""
    return [claim["claim_id"] for claim in selected_claims]


def normalize_extracted_claim_id(raw_claim_id: str) -> str:
    """Normalize claim IDs extracted from [CLAIMS: ...] blocks."""
    claim_id = raw_claim_id.strip()

    # Handle LLM mistakes like:
    # [CLAIMS: CLAIMS: fred__...]
    while claim_id.upper().startswith("CLAIMS:"):
        claim_id = claim_id.split(":", 1)[1].strip()

    # Remove common markdown/code punctuation around IDs.
    claim_id = claim_id.strip("`'\" ")

    return claim_id


def extract_cited_claim_ids(narrative_text: str) -> list[str]:
    """Extract claim IDs cited in [CLAIMS: ...] blocks."""
    cited: list[str] = []

    matches = re.findall(r"\[CLAIMS:\s*([^\]]+)\]", narrative_text)

    for match in matches:
        parts = [normalize_extracted_claim_id(part) for part in match.split(",")]
        cited.extend(part for part in parts if part)

    return cited


def format_claim_id_diagnostics(claim_ids: list[str]) -> str:
    """Format claim IDs for validation error diagnostics."""
    if not claim_ids:
        return "  (none)"
    return "\n".join(f"  - {claim_id}" for claim_id in claim_ids)


def validate_narrative_citations(
    *,
    narrative_text: str,
    selected_claims: list[dict],
) -> None:
    """Validate narrative citation coverage against selected claims."""
    selected_claim_ids = extract_claim_ids(selected_claims)
    selected_claim_id_set = set(selected_claim_ids)

    cited_claim_ids = extract_cited_claim_ids(narrative_text)
    cited_claim_id_set = set(cited_claim_ids)

    errors: list[str] = []

    missing_from_narrative = [
        claim_id for claim_id in selected_claim_ids if claim_id not in cited_claim_id_set
    ]
    if missing_from_narrative:
        errors.append(
            "Selected claim IDs missing from narrative citations: "
            + ", ".join(missing_from_narrative)
        )

    unknown_citations = [
        claim_id for claim_id in cited_claim_ids if claim_id not in selected_claim_id_set
    ]
    if unknown_citations:
        errors.append(
            "Narrative cites unknown claim IDs: "
            + ", ".join(unknown_citations)
        )

    if len(cited_claim_ids) != len(set(cited_claim_ids)):
        errors.append("Narrative contains duplicate claim citations.")

    if errors:
        diagnostic_lines = [
            *[f"- {err}" for err in errors],
            "",
            "Expected selected claim IDs:",
            format_claim_id_diagnostics(selected_claim_ids),
            "",
            "Extracted narrative claim IDs:",
            format_claim_id_diagnostics(cited_claim_ids),
            "",
            f"Expected selected claim count: {len(selected_claim_ids)}",
            f"Extracted narrative claim count: {len(cited_claim_ids)}",
        ]
        joined = "\n".join(diagnostic_lines)
        raise ValueError(f"FRED narrative citation validation failed:\n{joined}")


def write_json(path: Path, payload: object) -> None:
    """Write JSON with stable formatting."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    
    
def ollama_generate(
    *,
    host: str,
    model: str,
    prompt: str,
    options: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    """Generate text using a local Ollama model."""
    url = host.rstrip("/") + "/api/generate"
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    if options:
        payload["options"] = options

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    started = time.time()

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read()
            raw = json.loads(body.decode("utf-8", errors="replace"))
            text = raw.get("response") or ""
            return {
                "ok": True,
                "response_text": text,
                "raw": raw,
                "elapsed_s": round(time.time() - started, 3),
                "error": "",
            }
    except urllib.error.HTTPError as exc:
        try:
            error_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            error_body = ""
        return {
            "ok": False,
            "response_text": "",
            "raw": {},
            "elapsed_s": round(time.time() - started, 3),
            "error": f"HTTPError {exc.code}: {error_body}",
        }
    except Exception as exc:
        return {
            "ok": False,
            "response_text": "",
            "raw": {},
            "elapsed_s": round(time.time() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def write_narrative_artifacts(
    *,
    input_path: Path,
    output_dir: Path,
    mode: str,
    model: str,
    ollama_host: str,
    timeout_s: int,
    prompt_variant: str,
    temperature: float,
) -> None:
    """Generate and write FRED narrative artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_at = utc_now_iso()
    selected_claims = load_selected_claims(input_path)
    validate_selected_claims(selected_claims)

    if mode == "deterministic":
        narrative_md = build_narrative_markdown(
            selected_claims=selected_claims,
            generated_at=generated_at,
        )
        llm_metadata = {
            "llm_used": False,
            "model": None,
            "ollama_host": None,
            "prompt_variant": None,
            "temperature": None,
            "elapsed_s": 0,
            "error": "",
        }
    elif mode == "llm":
        narrative_md, llm_metadata = (
            build_llm_narrative_markdown(
                selected_claims=selected_claims,
                generated_at=generated_at,
                model=model,
                ollama_host=ollama_host,
                timeout_s=timeout_s,
                prompt_variant=prompt_variant,
                temperature=temperature,
            )
        )
    else:
        raise ValueError(f"Unsupported generation mode: {mode}")
    
    narrative_path = output_dir / "fred_narrative.md"
    metadata_path = output_dir / "fred_narrative_metadata.json"

    try:
        validate_narrative_citations(
            narrative_text=narrative_md,
            selected_claims=selected_claims,
        )
    except ValueError as exc:
        failed_narrative_path = output_dir / "fred_narrative_failed_validation.md"
        failed_metadata_path = output_dir / "fred_narrative_failed_validation_metadata.json"

        selected_claim_ids = extract_claim_ids(selected_claims)
        cited_claim_ids = extract_cited_claim_ids(narrative_md)

        failed_narrative_path.write_text(narrative_md, encoding="utf-8")

        failed_metadata = {
            "narrative_schema_version": NARRATIVE_SCHEMA_VERSION,
            "generation_method": "llm_claim_cited_narrative" if mode == "llm" else GENERATION_METHOD,
            "validation_error": str(exc),
            "input_file": str(input_path),
            "generated_at": generated_at,
            "mode": mode,
            "model": model if mode == "llm" else None,
            "prompt_variant": (
                prompt_variant
                if mode == "llm"
                else None
            ),
            "temperature": (
                temperature
                if mode == "llm"
                else None
            ),
            "llm_metadata": llm_metadata,
            "n_selected_claims": int(len(selected_claims)),
            "expected_claim_ids": selected_claim_ids,
            "extracted_cited_claim_ids": cited_claim_ids,
            "output_files": {
                "failed_narrative_md": str(failed_narrative_path),
                "failed_metadata_json": str(failed_metadata_path),
            },
        }

        write_json(failed_metadata_path, failed_metadata)

        print("Wrote failed FRED narrative validation debug artifacts:")
        print(f"  {failed_narrative_path}")
        print(f"  {failed_metadata_path}")

        raise

    narrative_path.write_text(narrative_md, encoding="utf-8")

    metadata = {
        "narrative_schema_version": NARRATIVE_SCHEMA_VERSION,
        "generation_method": "llm_claim_cited_narrative" if mode == "llm" else GENERATION_METHOD,
        "llm_metadata": llm_metadata,
        "input_file": str(input_path),
        "generated_at": generated_at,
        "mode": mode,
        "model": (
            model
            if mode == "llm"
            else None
        ),
        "prompt_variant": (
            prompt_variant
            if mode == "llm"
            else None
        ),
        "temperature": (
            temperature
            if mode == "llm"
            else None
        ),
        "n_selected_claims": int(len(selected_claims)),
        "used_claim_ids": extract_claim_ids(selected_claims),
        "cited_claim_ids": extract_cited_claim_ids(narrative_md),
        "citation_validation": {
            "all_selected_claims_cited": True,
            "all_citations_known": True,
            "duplicate_citations": False,
        },
        "output_files": {
            "narrative_md": str(narrative_path),
            "metadata_json": str(metadata_path),
        },
    }
    write_json(metadata_path, metadata)

    print("Wrote FRED narrative artifacts:")
    print(f"  {narrative_path}")
    print(f"  {metadata_path}")
    print(f"n_selected_claims={len(selected_claims)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a claim-cited FRED macro narrative."
    )
    parser.add_argument(
        "--input-claims",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input selected_fred_claims.json file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where FRED narrative artifacts will be written.",
    )
    parser.add_argument(
        "--mode",
        default=DEFAULT_MODE,
        choices=SUPPORTED_MODES,
        help="Narrative generation mode.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Local Ollama model to use when --mode llm.",
    )
    parser.add_argument(
        "--prompt-variant",
        default=DEFAULT_PROMPT_VARIANT,
        choices=SUPPORTED_PROMPT_VARIANTS,
        help=(
            "LLM prompt contract variant. "
            "Defaults to hardened."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=(
            "LLM sampling temperature. "
            "Defaults to 0.0."
        ),
    )
    parser.add_argument(
        "--ollama-host",
        default=DEFAULT_OLLAMA_HOST,
        help="Ollama host URL when --mode llm.",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=DEFAULT_TIMEOUT_S,
        help="Timeout in seconds for local LLM generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.temperature < 0:
        raise ValueError(
            "--temperature must be >= 0."
        )
    write_narrative_artifacts(
        input_path=args.input_claims,
        output_dir=args.output_dir,
        mode=args.mode,
        model=args.model,
        ollama_host=args.ollama_host,
        timeout_s=args.timeout_s,
        prompt_variant=args.prompt_variant,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()