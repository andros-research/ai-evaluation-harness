#!/usr/bin/env python3

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from build_semantic_proposer_prompt import (
    GUIDE_PATH,
    build_model_visible_guide,
    compact_evidence,
    load_corpus,
)

from run_semantic_classifier import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_TEMPERATURE,
    EXAMPLE_CANDIDATE_IDS,
    build_prompt,
    call_ollama,
    reference_annotations,
    validate_proposal,
)


DEFAULT_COMPARISON_ROOT = Path(
    "benchmarks/results/"
    "semantic_model_comparisons/"
    "fred_semantic_discovery_20260920"
)

DEFAULT_OUTPUT_ROOT = Path(
    "benchmarks/results/"
    "semantic_dashboard_batch/"
    "fred_semantic_discovery_20260920_t0_v0_1"
)

RUN_LABELS = [
    "llama3_semantic_conservative_t0",
    "llama3_semantic_analytical_t0",
    "llama3_70b_semantic_conservative_t0",
    "llama3_70b_semantic_analytical_t0",
    "mistral_semantic_conservative_t0",
    "mistral_semantic_analytical_t0",
]


def load_jsonl(
    path: Path,
) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
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


def write_jsonl(
    path: Path,
    rows: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def extract_commentary(
    narrative: str,
) -> tuple[str, int]:
    marker = "## Commentary"

    marker_index = narrative.find(
        marker
    )

    if marker_index < 0:
        raise ValueError(
            "Narrative contains no Commentary section."
        )

    start = (
        marker_index
        + len(marker)
    )

    commentary = narrative[
        start:
    ]

    return (
        commentary,
        start,
    )


def split_sentences(
    text: str,
) -> list[dict]:
    """
    Conservative first-pass sentence splitting.

    A boundary requires sentence punctuation followed by
    whitespace and an uppercase letter. This avoids splitting
    decimal values such as 0.414.
    """
    boundary = re.compile(
        r"(?<=[.!?])\s+(?=[A-Z])"
    )

    spans = []

    cursor = 0

    for match in boundary.finditer(
        text
    ):
        spans.append(
            (
                cursor,
                match.start(),
            )
        )

        cursor = match.end()

    spans.append(
        (
            cursor,
            len(text),
        )
    )

    sentences = []

    for raw_start, raw_end in spans:
        raw = text[
            raw_start:raw_end
        ]

        if not raw.strip():
            continue

        left_trim = (
            len(raw)
            - len(raw.lstrip())
        )

        right_trim = (
            len(raw.rstrip())
        )

        start = (
            raw_start
            + left_trim
        )

        end = (
            raw_start
            + right_trim
        )

        sentence = text[
            start:end
        ]

        if not sentence:
            continue

        # Skip Mistral-style footnote-definition paragraphs.
        if re.match(
            r"^\[\d+\]\s",
            sentence,
        ):
            continue

        sentences.append({
            "text": sentence,
            "commentary_start": start,
            "commentary_end": end,
        })

    return sentences


def evidence_from_claims(
    claims: list[dict],
) -> list[dict]:
    return [
        {
            "evidence_id":
                claim["claim_id"],
            "source_type":
                "structured_claim",
            "source_ref": None,
            "payload":
                claim,
        }
        for claim in claims
    ]


def build_live_target(
    *,
    claim_text: str,
    claims: list[dict],
) -> dict:
    return {
        "statement": {
            "text":
                claim_text,
        },
        "evidence":
            evidence_from_claims(
                claims
            ),
    }


def build_examples(
    *,
    claim_text: str,
    corpus_by_id: dict[str, dict],
) -> tuple[list[str], list[dict]]:
    example_ids = []

    examples = []

    for example_id in (
        EXAMPLE_CANDIDATE_IDS
    ):
        record = corpus_by_id[
            example_id
        ]

        # Never show an identical statement as a reviewed example.
        if (
            record["statement"]["text"]
            == claim_text
        ):
            continue

        example_ids.append(
            example_id
        )

        examples.append({
            "claim_span":
                record["statement"]["text"],
            "evidence":
                compact_evidence(
                    record
                ),
            "reviewed_annotations":
                reference_annotations(
                    record
                ),
        })

    return (
        example_ids,
        examples,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--comparison-root",
        type=Path,
        default=DEFAULT_COMPARISON_ROOT,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
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
        "--max-sentences-per-run",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse existing dashboard claim results "
            "and classify only missing claim IDs."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    summary_path = (
        args.comparison_root
        / "summary"
        / "comparison_rows.jsonl"
    )

    rows = load_jsonl(
        summary_path
    )

    rows_by_label = {
        row["run_label"]: row
        for row in rows
    }

    corpus = load_corpus()

    corpus_by_id = {
        row["annotation_id"]: row
        for row in corpus
    }

    planned_claims = []

    for run_label in RUN_LABELS:
        if run_label not in rows_by_label:
            raise SystemExit(
                f"Missing comparison row: "
                f"{run_label}"
            )

        run = rows_by_label[
            run_label
        ]

        artifact_root = Path(
            run["artifact_root"]
        )

        narrative_path = (
            artifact_root
            / "fred_narratives"
            / "fred_narrative.md"
        )

        claims_path = (
            artifact_root
            / "fred_claims"
            / "selected_fred_claims.json"
        )

        if not narrative_path.is_file():
            print(
                "SKIP missing narrative:",
                run_label,
            )
            continue

        if not claims_path.is_file():
            print(
                "SKIP missing claims:",
                run_label,
            )
            continue

        narrative = (
            narrative_path.read_text(
                encoding="utf-8"
            )
        )

        commentary, commentary_offset = (
            extract_commentary(
                narrative
            )
        )

        sentences = split_sentences(
            commentary
        )

        claims = json.loads(
            claims_path.read_text(
                encoding="utf-8"
            )
        )

        for sentence_index, sentence in enumerate(
            sentences[
                :args.max_sentences_per_run
            ],
            start=1,
        ):
            planned_claims.append({
                "claim_id": (
                    f"{run_label}"
                    f"__commentary_"
                    f"{sentence_index:02d}"
                ),
                "run_label":
                    run_label,
                "source_model":
                    run.get("model"),
                "prompt_variant":
                    run.get(
                        "prompt_variant"
                    ),
                "generation_temperature":
                    run.get(
                        "temperature"
                    ),
                "process_ok":
                    run.get(
                        "process_ok"
                    ),
                "audit_pass":
                    run.get(
                        "audit_pass"
                    ),
                "claim_text":
                    sentence["text"],
                "source": {
                    "narrative_path":
                        str(
                            narrative_path
                        ),
                    "commentary_start":
                        sentence[
                            "commentary_start"
                        ],
                    "commentary_end":
                        sentence[
                            "commentary_end"
                        ],
                    "narrative_start": (
                        commentary_offset
                        + sentence[
                            "commentary_start"
                        ]
                    ),
                    "narrative_end": (
                        commentary_offset
                        + sentence[
                            "commentary_end"
                        ]
                    ),
                },
                "claims_path":
                    str(
                        claims_path
                    ),
                "source_claims":
                    claims,
            })

    print(
        "=== SEMANTIC DASHBOARD BATCH PLAN ==="
    )
    print()
    print(
        "runs:",
        len(
            {
                item["run_label"]
                for item in planned_claims
            }
        ),
    )
    print(
        "claim spans:",
        len(planned_claims),
    )

    for item in planned_claims:
        print()
        print(
            item["claim_id"]
        )
        print(
            " ",
            item["source_model"],
            "|",
            item["prompt_variant"],
            "| t=",
            item[
                "generation_temperature"
            ],
        )
        print(
            " ",
            item["claim_text"],
        )

    if args.dry_run:
        print()
        print(
            "DRY RUN: no classifier calls made."
        )
        return

    claims_artifact = (
        args.output_root
        / "dashboard_claims.jsonl"
    )

    existing_by_id = {}

    if (
        args.resume
        and claims_artifact.exists()
    ):
        existing_rows = load_jsonl(
            claims_artifact
        )

        existing_by_id = {
            row["claim_id"]: row
            for row in existing_rows
        }

        print()
        print(
            "resume existing claims:",
            len(existing_by_id),
        )

    output_by_id = dict(
        existing_by_id
    )

    planned_ids = [
        item["claim_id"]
        for item in planned_claims
    ]

    def current_output_rows():
        return [
            output_by_id[
                claim_id
            ]
            for claim_id
            in planned_ids
            if claim_id
            in output_by_id
        ]

    guide_source = (
        GUIDE_PATH.read_text(
            encoding="utf-8"
        )
    )

    for index, item in enumerate(
        planned_claims,
        start=1,
    ):
        if (
            item["claim_id"]
            in output_by_id
        ):
            print()
            print(
                "SKIP existing:",
                item["claim_id"],
            )
            continue

        print()
        print("=" * 78)
        print(
            f"Semantic claim "
            f"{index}/{len(planned_claims)}"
        )
        print(
            item["claim_id"]
        )
        print("=" * 78)

        target = build_live_target(
            claim_text=
                item["claim_text"],
            claims=
                item["source_claims"],
        )

        (
            example_ids,
            examples,
        ) = build_examples(
            claim_text=
                item["claim_text"],
            corpus_by_id=
                corpus_by_id,
        )

        guide = (
            build_model_visible_guide(
                guide_source,
                target_text=
                    item["claim_text"],
            )
        )

        prompt = build_prompt(
            guide=guide,
            examples=examples,
            target=target,
        )

        record = {
            "schema_version":
                "semantic_dashboard_claim_v0_1",
            "claim_id":
                item["claim_id"],
            "source_model":
                item["source_model"],
            "prompt_variant":
                item["prompt_variant"],
            "generation_temperature":
                item[
                    "generation_temperature"
                ],
            "process_ok":
                item["process_ok"],
            "audit_pass":
                item["audit_pass"],
            "claim_span":
                item["claim_text"],
            "source":
                item["source"],
            "evidence":
                compact_evidence(
                    target
                ),
            "semantic_review_status":
                "proposed",
            "semantic_classifier": {
                "model":
                    args.model,
                "temperature":
                    args.temperature,
                "few_shot_example_ids":
                    example_ids,
            },
        }

        try:
            result = call_ollama(
                prompt=prompt,
                model=args.model,
                temperature=
                    args.temperature,
                ollama_host=
                    args.ollama_host,
                timeout_s=
                    args.timeout_s,
            )

            proposal = result[
                "proposal"
            ]

            annotations = (
                validate_proposal(
                    proposal=proposal,
                    claim_span=
                        item[
                            "claim_text"
                        ],
                )
            )

            record[
                "semantic_annotations"
            ] = annotations

            record[
                "semantic_classifier"
            ][
                "ollama_metadata"
            ] = result[
                "ollama_metadata"
            ]

            record["status"] = (
                "success"
            )

            print(
                "annotations:",
                len(
                    annotations
                ),
            )

            for annotation in annotations:
                print(
                    " ",
                    annotation[
                        "claim_kind"
                    ],
                    "/",
                    annotation[
                        "support_status"
                    ],
                    "/",
                    annotation[
                        "assertion_strength"
                    ],
                )

        except Exception as exc:
            record["status"] = (
                "failure"
            )

            record[
                "semantic_annotations"
            ] = []

            record["error"] = {
                "type":
                    type(exc).__name__,
                "message":
                    str(exc),
            }

            print(
                "FAILED:",
                type(exc).__name__,
                str(exc),
            )

        output_by_id[
            item["claim_id"]
        ] = record

        # Checkpoint after every new claim
        # without discarding prior results.
        write_jsonl(
            claims_artifact,
            current_output_rows(),
        )

    output_rows = (
        current_output_rows()
    )

    manifest = {
        "schema_version":
            "semantic_dashboard_batch_v0_1",
        "created_at":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "source_comparison":
            str(
                args.comparison_root
            ),
        "semantic_classifier_model":
            args.model,
        "semantic_classifier_temperature":
            args.temperature,
        "run_labels":
            RUN_LABELS,
        "max_sentences_per_run":
            args.max_sentences_per_run,
        "n_claims":
            len(
                output_rows
            ),
        "n_success":
            sum(
                row["status"]
                == "success"
                for row in output_rows
            ),
        "n_failure":
            sum(
                row["status"]
                == "failure"
                for row in output_rows
            ),
        "claims_artifact":
            str(
                args.output_root
                / "dashboard_claims.jsonl"
            ),
    }

    write_json(
        args.output_root
        / "batch_manifest.json",
        manifest,
    )

    print()
    print(
        "=== BATCH COMPLETE ==="
    )
    print(
        "claims:",
        manifest["n_claims"],
    )
    print(
        "success:",
        manifest["n_success"],
    )
    print(
        "failure:",
        manifest["n_failure"],
    )
    print(
        "artifact:",
        manifest[
            "claims_artifact"
        ],
    )


if __name__ == "__main__":
    main()
