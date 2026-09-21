#!/usr/bin/env python3

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from build_semantic_proposer_prompt import (
    GUIDE_PATH,
    annotations_for_record,
    build_example,
    build_model_prompt,
    build_reference_answer,
    load_corpus,
)


DEFAULT_MODEL = "llama3:70b"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"

EXAMPLE_IDS = [
    "semantic_pilot_004",
    "semantic_pilot_005",
    "semantic_challenge_002",
    "semantic_challenge_004",
    "semantic_harvest_004",
]


def write_json(
    path: Path,
    payload: object,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def validate_proposal(
    proposal: object,
    *,
    target_text: str,
) -> list[dict]:
    if not isinstance(proposal, dict):
        raise ValueError(
            "Proposal must be a JSON object."
        )

    units = proposal.get(
        "semantic_units"
    )

    if not isinstance(units, list):
        raise ValueError(
            "semantic_units must be an array."
        )

    if not units:
        raise ValueError(
            "semantic_units must not be empty."
        )

    allowed_support = {
        "supported",
        "contradicted",
        "insufficient_evidence",
    }

    allowed_strength = {
        "asserted",
        "qualified_inference",
        "qualified_possibility",
    }

    for index, unit in enumerate(units):
        prefix = (
            f"semantic_units[{index}]"
        )

        if not isinstance(unit, dict):
            raise ValueError(
                f"{prefix} must be an object."
            )

        for field in (
            "text",
            "claim_kind",
            "support_status",
            "assertion_strength",
            "rationale",
        ):
            value = unit.get(field)

            if (
                not isinstance(value, str)
                or not value.strip()
            ):
                raise ValueError(
                    f"{prefix}.{field} "
                    "must be a non-empty string."
                )

        if (
            unit["support_status"]
            not in allowed_support
        ):
            raise ValueError(
                f"{prefix}.support_status "
                "is unsupported."
            )

        if (
            unit["assertion_strength"]
            not in allowed_strength
        ):
            raise ValueError(
                f"{prefix}.assertion_strength "
                "is unsupported."
            )

        if unit["text"] not in target_text:
            raise ValueError(
                f"{prefix}.text is not an "
                "exact span of the target statement."
            )

    return units


def call_ollama(
    *,
    prompt: str,
    model: str,
    temperature: float,
    ollama_host: str,
    timeout_s: int,
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
        },
    }

    request = urllib.request.Request(
        (
            ollama_host.rstrip("/")
            + "/api/generate"
        ),
        data=json.dumps(
            payload
        ).encode("utf-8"),
        headers={
            "Content-Type":
                "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=timeout_s,
    ) as response:
        result = json.loads(
            response.read().decode(
                "utf-8"
            )
        )

    raw_response = result.get(
        "response"
    )

    if not isinstance(
        raw_response,
        str,
    ):
        raise ValueError(
            "Ollama response did not "
            "contain response text."
        )

    try:
        proposal = json.loads(
            raw_response
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Ollama response was not "
            f"valid JSON: {exc}"
        ) from exc

    return {
        "proposal": proposal,
        "ollama_metadata": {
            key: value
            for key, value
            in result.items()
            if key != "response"
        },
    }


def compare_units(
    *,
    proposed_units: list[dict],
    reference_units: list[dict],
) -> dict:
    count_match = (
        len(proposed_units)
        == len(reference_units)
    )

    unit_results = []

    max_units = max(
        len(proposed_units),
        len(reference_units),
    )

    for index in range(max_units):
        proposed = (
            proposed_units[index]
            if index
            < len(proposed_units)
            else None
        )

        reference = (
            reference_units[index]
            if index
            < len(reference_units)
            else None
        )

        if (
            proposed is None
            or reference is None
        ):
            unit_results.append({
                "unit_index": index,
                "proposal_present":
                    proposed is not None,
                "reference_present":
                    reference is not None,
                "span_match": False,
                "claim_kind_match": False,
                "support_status_match":
                    False,
                "assertion_strength_match":
                    False,
            })
            continue

        annotation = reference[
            "annotation"
        ]

        unit_results.append({
            "unit_index": index,
            "proposal_present": True,
            "reference_present": True,
            "span_match": (
                proposed["text"]
                == reference["text"]
            ),
            "claim_kind_match": (
                proposed["claim_kind"]
                == annotation[
                    "claim_kind"
                ]
            ),
            "support_status_match": (
                proposed[
                    "support_status"
                ]
                == annotation[
                    "support_status"
                ]
            ),
            "assertion_strength_match": (
                proposed[
                    "assertion_strength"
                ]
                == annotation[
                    "assertion_strength"
                ]
            ),
            "proposed_rationale": (
                proposed["rationale"]
            ),
            "reference_rationale": (
                annotation["rationale"]
            ),
        })

    scored_fields = []

    for result in unit_results:
        for field in (
            "span_match",
            "claim_kind_match",
            "support_status_match",
            "assertion_strength_match",
        ):
            scored_fields.append(
                bool(result[field])
            )

    exact_categorical_match = (
        count_match
        and all(scored_fields)
    )

    return {
        "unit_count_match":
            count_match,
        "proposed_unit_count":
            len(proposed_units),
        "reference_unit_count":
            len(reference_units),
        "unit_results":
            unit_results,
        "n_field_checks":
            len(scored_fields),
        "n_field_matches":
            sum(scored_fields),
        "exact_categorical_match":
            exact_categorical_match,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target-id",
        required=True,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
    )

    parser.add_argument(
        "--ollama-host",
        default=DEFAULT_OLLAMA_HOST,
    )

    parser.add_argument(
        "--timeout-s",
        type=int,
        default=600,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "benchmarks/results/"
            "semantic_proposer"
        ),
    )

    args = parser.parse_args()

    corpus = load_corpus()

    by_id = {
        row["annotation_id"]: row
        for row in corpus
    }

    if args.target_id not in by_id:
        raise SystemExit(
            f"Unknown target ID: "
            f"{args.target_id}"
        )

    target = by_id[
        args.target_id
    ]

    if (
        target["review"]["status"]
        != "reviewed"
    ):
        raise SystemExit(
            "Target must have a reviewed "
            "human reference."
        )

    if args.target_id in EXAMPLE_IDS:
        raise SystemExit(
            "STOP: target is part of the "
            "few-shot example set."
        )

    examples = [
        build_example(
            by_id[example_id]
        )
        for example_id
        in EXAMPLE_IDS
    ]

    guide = GUIDE_PATH.read_text(
        encoding="utf-8"
    )

    prompt = build_model_prompt(
        guide=guide,
        examples=examples,
        target=target,
    )

    reference = (
        build_reference_answer(
            target
        )
    )

    reference_rationales = [
        unit["annotation"][
            "rationale"
        ]
        for unit in reference[
            "semantic_units"
        ]
    ]

    for rationale in (
        reference_rationales
    ):
        if rationale in prompt:
            raise SystemExit(
                "STOP: held-out human "
                "rationale leaked into prompt."
            )

    run_id = (
        f"{args.target_id}"
        f"__{args.model.replace(':', '_')}"
        f"__t{str(args.temperature).replace('.', '_')}"
    )

    output_dir = (
        args.output_root
        / run_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prompt_path = (
        output_dir
        / "prompt.txt"
    )

    proposal_path = (
        output_dir
        / "proposal.json"
    )

    reference_path = (
        output_dir
        / "reference.json"
    )

    comparison_path = (
        output_dir
        / "comparison.json"
    )

    prompt_path.write_text(
        prompt,
        encoding="utf-8",
    )

    write_json(
        reference_path,
        reference,
    )

    print(
        "Running blind semantic proposer..."
    )
    print(
        "target:",
        args.target_id,
    )
    print(
        "model:",
        args.model,
    )
    print(
        "temperature:",
        args.temperature,
    )
    print()

    result = call_ollama(
        prompt=prompt,
        model=args.model,
        temperature=args.temperature,
        ollama_host=args.ollama_host,
        timeout_s=args.timeout_s,
    )

    proposal = result[
        "proposal"
    ]

    proposed_units = (
        validate_proposal(
            proposal,
            target_text=target[
                "statement"
            ]["text"],
        )
    )

    write_json(
        proposal_path,
        proposal,
    )

    comparison = compare_units(
        proposed_units=
            proposed_units,
        reference_units=
            reference[
                "semantic_units"
            ],
    )

    comparison_artifact = {
        "schema_version":
            "semantic_proposer_comparison_v0_1",
        "created_at":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "target_id":
            args.target_id,
        "model":
            args.model,
        "temperature":
            args.temperature,
        "few_shot_example_ids":
            EXAMPLE_IDS,
        "artifacts": {
            "prompt":
                str(prompt_path),
            "proposal":
                str(proposal_path),
            "reference":
                str(reference_path),
        },
        "comparison":
            comparison,
        "ollama_metadata":
            result[
                "ollama_metadata"
            ],
    }

    write_json(
        comparison_path,
        comparison_artifact,
    )

    print(
        "=== SEMANTIC PROPOSER "
        "COMPARISON ==="
    )
    print()

    print(
        "unit_count_match:",
        comparison[
            "unit_count_match"
        ],
    )

    print(
        "field_matches:",
        (
            f"{comparison['n_field_matches']}"
            f"/"
            f"{comparison['n_field_checks']}"
        ),
    )

    print(
        "exact_categorical_match:",
        comparison[
            "exact_categorical_match"
        ],
    )

    print()

    for result in comparison[
        "unit_results"
    ]:
        print(
            "unit",
            result["unit_index"],
        )

        for field in (
            "span_match",
            "claim_kind_match",
            "support_status_match",
            "assertion_strength_match",
        ):
            print(
                f"  {field}:",
                result[field],
            )

    print()
    print(
        "comparison artifact:",
        comparison_path,
    )


if __name__ == "__main__":
    main()
