#!/usr/bin/env python3

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from build_semantic_proposer_prompt import (
    GUIDE_PATH,
    build_model_visible_guide,
    compact_evidence,
    load_corpus,
)


DEFAULT_MODEL = "llama3:70b"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"

EXAMPLE_CANDIDATE_IDS = [
    "semantic_pilot_005",
    "semantic_challenge_001",
    "semantic_challenge_002",
    "semantic_harvest_004",
    "semantic_challenge_004",
    "semantic_harvest_002",
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


def reference_annotations(
    record: dict,
) -> list[dict]:
    units = (
        record.get("semantic_units")
        or []
    )

    # New claim-span representation.
    if units:
        if len(units) != 1:
            raise ValueError(
                "Classifier v0.1 expects exactly one "
                "claim-bearing span per target."
            )

        unit = units[0]

        if (
            unit["text"]
            != record["statement"]["text"]
        ):
            raise ValueError(
                "Classifier v0.1 requires the claim span "
                "to equal the full target statement."
            )

        annotations = unit.get(
            "annotations"
        )

        if annotations is not None:
            return annotations

        annotation = unit.get(
            "annotation"
        )

        if annotation is None:
            raise ValueError(
                "Semantic unit has no annotation."
            )

        return [
            annotation
        ]

    annotation = record.get(
        "annotation"
    )

    if annotation is None:
        raise ValueError(
            "Record has no reviewed annotation."
        )

    return [
        annotation
    ]


def select_examples(
    *,
    target_id: str,
    by_id: dict[str, dict],
) -> tuple[list[str], list[dict]]:
    target_text = by_id[
        target_id
    ]["statement"]["text"]

    example_ids = [
        example_id
        for example_id
        in EXAMPLE_CANDIDATE_IDS
        if (
            example_id != target_id
            and by_id[example_id][
                "statement"
            ]["text"] != target_text
        )
    ]

    examples = []

    for example_id in example_ids:
        record = by_id[
            example_id
        ]

        examples.append({
            "claim_span":
                record["statement"]["text"],
            "evidence":
                compact_evidence(record),
            "reviewed_annotations":
                reference_annotations(
                    record
                ),
        })

    return (
        example_ids,
        examples,
    )


def build_prompt(
    *,
    guide: str,
    examples: list[dict],
    target: dict,
) -> str:
    target_payload = {
        "claim_span":
            target["statement"]["text"],
        "evidence":
            compact_evidence(target),
    }

    sections = [
        (
            "You are classifying semantic moves in a "
            "fixed macroeconomic claim-bearing span."
        ),
        "",
        (
            "The claim span has already been selected. "
            "Do not split it, shorten it, or propose "
            "different text spans."
        ),
        "",
        (
            "Your job is to identify every distinct "
            "semantic annotation expressed by this claim "
            "relative to the supplied evidence."
        ),
        "",
        "Important principles:",
        (
            "- One claim span may contain more than one "
            "semantic move."
        ),
        (
            "- Treat this as exhaustive multi-label "
            "classification, not as a request to choose "
            "the single best label."
        ),
        (
            "- Do not stop after finding one applicable "
            "semantic move. Scan the complete claim again "
            "for additional commitments."
        ),
        (
            "- In particular, independently consider whether "
            "the claim contains an empirical comparison, "
            "unit conversion, temporal pattern, causal "
            "explanation, scope generalization, magnitude "
            "judgment, market interpretation, or policy "
            "interpretation."
        ),
        (
            "- A modifier or qualifier can add a distinct "
            "semantic commitment even when the underlying "
            "claim already receives another annotation."
        ),
        (
            "- Return one annotation for each distinct "
            "semantic move that materially changes what "
            "the reader is asked to accept."
        ),
        (
            "- Do not create duplicate annotations for "
            "the same semantic move."
        ),
        (
            "- Do not import outside facts to rescue "
            "an unsupported claim."
        ),
        (
            "- insufficient_evidence does not mean false."
        ),
        (
            "- Assertion strength is separate from "
            "evidential support."
        ),
        (
            "- Claim-kind vocabulary may be extended only "
            "when existing reviewed categories clearly "
            "do not fit."
        ),
        "",
        "ANNOTATION GUIDE",
        "================",
        guide.rstrip(),
        "",
        "REVIEWED EXAMPLES",
        "=================",
        json.dumps(
            examples,
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "TARGET",
        "======",
        json.dumps(
            target_payload,
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "Return JSON only using this shape:",
        "",
        "{",
        '  "claim_span": "<exact supplied claim span>",',
        '  "annotations": [',
        "    {",
        '      "claim_kind": "<claim kind>",',
        (
            '      "support_status": '
            '"supported | contradicted | insufficient_evidence",'
        ),
        (
            '      "assertion_strength": '
            '"asserted | qualified_inference | '
            'qualified_possibility",'
        ),
        (
            '      "rationale": '
            '"<brief evidence-relative rationale>"'
        ),
        "    }",
        "  ]",
        "}",
        "",
    ]

    return "\n".join(
        sections
    )


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
            "temperature":
                temperature,
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

    raw = result.get(
        "response"
    )

    if not isinstance(raw, str):
        raise ValueError(
            "Ollama response did not contain response text."
        )

    proposal = json.loads(
        raw
    )

    metadata = {
        key: value
        for key, value
        in result.items()
        if key not in {
            "response",
            "context",
        }
    }

    context = result.get(
        "context"
    )

    if isinstance(
        context,
        list,
    ):
        metadata[
            "context_token_count"
        ] = len(context)

    return {
        "proposal": proposal,
        "ollama_metadata": metadata,
    }


def validate_proposal(
    *,
    proposal: object,
    claim_span: str,
) -> list[dict]:
    if not isinstance(
        proposal,
        dict,
    ):
        raise ValueError(
            "Proposal must be an object."
        )

    if (
        proposal.get("claim_span")
        != claim_span
    ):
        raise ValueError(
            "Proposal changed the fixed claim span."
        )

    annotations = proposal.get(
        "annotations"
    )

    if (
        not isinstance(
            annotations,
            list,
        )
        or not annotations
    ):
        raise ValueError(
            "annotations must be a non-empty array."
        )

    for index, annotation in enumerate(
        annotations
    ):
        if not isinstance(
            annotation,
            dict,
        ):
            raise ValueError(
                f"annotations[{index}] must be an object."
            )

        for field in (
            "claim_kind",
            "support_status",
            "assertion_strength",
            "rationale",
        ):
            value = annotation.get(
                field
            )

            if (
                not isinstance(
                    value,
                    str,
                )
                or not value.strip()
            ):
                raise ValueError(
                    f"annotations[{index}].{field} "
                    "must be a non-empty string."
                )

    return annotations


def annotation_key(
    annotation: dict,
) -> tuple[str, str, str]:
    return (
        annotation[
            "claim_kind"
        ],
        annotation[
            "support_status"
        ],
        annotation[
            "assertion_strength"
        ],
    )


def compare_annotations(
    *,
    proposed: list[dict],
    reference: list[dict],
) -> dict:
    proposed_keys = [
        annotation_key(
            item
        )
        for item in proposed
    ]

    reference_keys = [
        annotation_key(
            item
        )
        for item in reference
    ]

    proposed_set = set(
        proposed_keys
    )

    reference_set = set(
        reference_keys
    )

    exact_match = (
        proposed_set
        == reference_set
        and len(proposed_keys)
        == len(reference_keys)
    )

    return {
        "proposed_annotation_count":
            len(proposed_keys),
        "reference_annotation_count":
            len(reference_keys),
        "annotation_count_match":
            len(proposed_keys)
            == len(reference_keys),
        "matched_annotations": [
            {
                "claim_kind": key[0],
                "support_status": key[1],
                "assertion_strength": key[2],
            }
            for key in sorted(
                proposed_set
                & reference_set
            )
        ],
        "proposal_only": [
            {
                "claim_kind": key[0],
                "support_status": key[1],
                "assertion_strength": key[2],
            }
            for key in sorted(
                proposed_set
                - reference_set
            )
        ],
        "reference_only": [
            {
                "claim_kind": key[0],
                "support_status": key[1],
                "assertion_strength": key[2],
            }
            for key in sorted(
                reference_set
                - proposed_set
            )
        ],
        "exact_annotation_set_match":
            exact_match,
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
            "semantic_classifier"
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

    claim_span = target[
        "statement"
    ]["text"]

    reference = (
        reference_annotations(
            target
        )
    )

    (
        example_ids,
        examples,
    ) = select_examples(
        target_id=args.target_id,
        by_id=by_id,
    )

    guide = build_model_visible_guide(
        GUIDE_PATH.read_text(
            encoding="utf-8"
        ),
        target_text=claim_span,
    )

    prompt = build_prompt(
        guide=guide,
        examples=examples,
        target=target,
    )

    if prompt.count(
        claim_span
    ) != 1:
        raise SystemExit(
            "STOP: held-out claim span must "
            "appear exactly once in prompt."
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
        {
            "target_id":
                args.target_id,
            "claim_span":
                claim_span,
            "annotations":
                reference,
        },
    )

    print(
        "Running blind semantic classifier..."
    )
    print(
        "target:",
        args.target_id,
    )
    print(
        "model:",
        args.model,
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

    proposed = validate_proposal(
        proposal=proposal,
        claim_span=claim_span,
    )

    write_json(
        proposal_path,
        proposal,
    )

    comparison = (
        compare_annotations(
            proposed=proposed,
            reference=reference,
        )
    )

    artifact = {
        "schema_version":
            "semantic_classifier_comparison_v0_1",
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
            example_ids,
        "claim_span":
            claim_span,
        "comparison":
            comparison,
        "ollama_metadata":
            result[
                "ollama_metadata"
            ],
    }

    write_json(
        comparison_path,
        artifact,
    )

    print(
        "=== SEMANTIC CLASSIFIER RESULT ==="
    )
    print()
    print(
        "annotation_count_match:",
        comparison[
            "annotation_count_match"
        ],
    )
    print(
        "exact_annotation_set_match:",
        comparison[
            "exact_annotation_set_match"
        ],
    )

    print()
    print(
        "matched:",
        comparison[
            "matched_annotations"
        ],
    )
    print(
        "proposal_only:",
        comparison[
            "proposal_only"
        ],
    )
    print(
        "reference_only:",
        comparison[
            "reference_only"
        ],
    )

    print()
    print(
        "comparison artifact:",
        comparison_path,
    )


if __name__ == "__main__":
    main()
