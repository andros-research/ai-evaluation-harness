#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

GUIDE_PATH = (
    REPO_ROOT
    / "docs"
    / "v1_9_semantic_annotation_guide.md"
)

FIXTURE_PATHS = [
    REPO_ROOT
    / "benchmarks"
    / "fixtures"
    / "semantic_annotation_reference_v0_1.jsonl",
    REPO_ROOT
    / "benchmarks"
    / "fixtures"
    / "semantic_challenge_v0_1.jsonl",
    REPO_ROOT
    / "benchmarks"
    / "fixtures"
    / "semantic_model_harvest_v0_1.jsonl",
]


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def load_corpus() -> list[dict]:
    rows = []

    for path in FIXTURE_PATHS:
        rows.extend(load_jsonl(path))

    return rows


def annotations_for_record(
    record: dict,
) -> list[dict]:
    semantic_units = (
        record.get("semantic_units")
        or []
    )

    if semantic_units:
        answers = []

        for unit in semantic_units:
            annotations = unit.get(
                "annotations"
            )

            if annotations is None:
                annotation = unit.get(
                    "annotation"
                )

                if annotation is None:
                    raise ValueError(
                        "Semantic unit has no annotation payload."
                    )

                annotations = [
                    annotation
                ]

            for annotation in annotations:
                answers.append({
                    "text": unit["text"],
                    "annotation": annotation,
                })

        return answers

    return [
        {
            "text": record["statement"]["text"],
            "annotation": record["annotation"],
        }
    ]


def compact_evidence(
    record: dict,
) -> list[dict]:
    """Preserve full evidence payload while exposing common fields."""
    compact = []

    for item in record["evidence"]:
        payload = item.get(
            "payload"
        )

        if not isinstance(
            payload,
            dict,
        ):
            payload = {}

        compact.append(
            {
                "evidence_id":
                    item["evidence_id"],
                "source_type":
                    item["source_type"],
                "claim_text":
                    payload.get(
                        "claim_text"
                    ),
                "metric_name":
                    payload.get(
                        "metric_name"
                    ),
                "current_value":
                    payload.get(
                        "current_value"
                    ),
                "prior_value":
                    payload.get(
                        "prior_value"
                    ),
                "delta_value":
                    payload.get(
                        "delta_value"
                    ),
                "direction":
                    payload.get(
                        "direction"
                    ),
                "comparison_window":
                    payload.get(
                        "comparison_window"
                    ),
                "payload":
                    payload,
            }
        )

    return compact


def build_example(record: dict) -> dict:
    return {
        "statement": record["statement"]["text"],
        "evidence": compact_evidence(record),
        "reviewed_answer": annotations_for_record(
            record
        ),
    }


def build_model_visible_guide(
    guide: str,
    *,
    target_text: str | None = None,
) -> str:
    """Remove record IDs and held-out target text from guidance."""
    visible = re.sub(
        r"semantic_(?:pilot|challenge|harvest)_\d+",
        "reviewed_example",
        guide,
    )

    if target_text:
        visible = visible.replace(
            target_text,
            "[held-out target statement removed]",
        )

    return visible


def build_target(record: dict) -> dict:
    return {
        "statement": record["statement"]["text"],
        "evidence": compact_evidence(record),
    }


def build_model_prompt(
    *,
    guide: str,
    examples: list[dict],
    target: dict,
) -> str:
    target_payload = build_target(target)

    sections = [
        (
            "You are proposing semantic annotations for "
            "macroeconomic model output."
        ),
        "",
        (
            "Your job is to assess the supplied statement only "
            "relative to the supplied evidence."
        ),
        "",
        "Important principles:",
        "- Separate direct observation from interpretation.",
        "- Do not import outside facts to rescue a claim.",
        "- insufficient_evidence does not mean false.",
        (
            "- contradicted means the supplied evidence conflicts "
            "with the claim."
        ),
        (
            "- assertion strength is separate from evidential "
            "support."
        ),
        (
            "- Use the smallest meaningful semantic unit when a "
            "statement contains multiple distinct claims."
        ),
        (
            "- Preserve exact wording from the target statement "
            "when identifying semantic units."
        ),
        (
            "- Claim-kind vocabulary may be extended only when "
            "existing reviewed categories clearly do not fit."
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
        "Return JSON only.",
        "",
        "Use this shape for a single semantic unit:",
        "",
        "{",
        '  "semantic_units": [',
        "    {",
        '      "text": "<exact target span>",',
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
        (
            "If multiple semantic units are necessary, return "
            "multiple entries in semantic_units."
        ),
        "",
    ]

    return "\n".join(sections)


FEW_SHOT_CANDIDATE_IDS = [
    "semantic_pilot_004",
    "semantic_pilot_005",
    "semantic_challenge_002",
    "semantic_challenge_004",
    "semantic_harvest_004",
    "semantic_harvest_002",
]

FEW_SHOT_EXAMPLE_COUNT = 5


def select_example_ids(
    target_id: str,
) -> list[str]:
    """Select reviewed examples while excluding the held-out target."""
    eligible = [
        example_id
        for example_id in FEW_SHOT_CANDIDATE_IDS
        if example_id != target_id
    ]

    if len(eligible) < FEW_SHOT_EXAMPLE_COUNT:
        raise ValueError(
            "Not enough eligible few-shot examples "
            "after excluding the held-out target."
        )

    return eligible[
        :FEW_SHOT_EXAMPLE_COUNT
    ]


def build_reference_answer(
    target: dict,
) -> dict:
    return {
        "annotation_id": target["annotation_id"],
        "semantic_units": annotations_for_record(
            target
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target-id",
        default="semantic_harvest_003",
    )

    parser.add_argument(
        "--prompt-output",
        type=Path,
        required=True,
        help=(
            "Path for the model-visible proposer prompt."
        ),
    )

    parser.add_argument(
        "--reference-output",
        type=Path,
        required=True,
        help=(
            "Path for the held-out human reference answer. "
            "This file must never be supplied to the proposer."
        ),
    )

    args = parser.parse_args()

    prompt_output = args.prompt_output.resolve()
    reference_output = (
        args.reference_output.resolve()
    )

    if prompt_output == reference_output:
        raise SystemExit(
            "STOP: prompt and reference outputs "
            "must be different files."
        )

    corpus = load_corpus()

    by_id = {
        row["annotation_id"]: row
        for row in corpus
    }

    if args.target_id not in by_id:
        raise SystemExit(
            f"Unknown target ID: {args.target_id}"
        )

    target = by_id[args.target_id]

    if target["review"]["status"] != "reviewed":
        raise SystemExit(
            "Target must have a reviewed human answer."
        )

    example_ids = select_example_ids(
        args.target_id
    )

    examples = [
        build_example(
            by_id[example_id]
        )
        for example_id in example_ids
    ]

    guide = build_model_visible_guide(
        GUIDE_PATH.read_text(
            encoding="utf-8"
        ),
        target_text=target[
            "statement"
        ]["text"],
    )

    model_prompt = build_model_prompt(
        guide=guide,
        examples=examples,
        target=target,
    )

    reference_answer = build_reference_answer(
        target
    )

    args.prompt_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.reference_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.prompt_output.write_text(
        model_prompt,
        encoding="utf-8",
    )

    args.reference_output.write_text(
        json.dumps(
            reference_answer,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Built blind semantic proposer inputs.")
    print()
    print(
        "target:",
        args.target_id,
    )
    print(
        "model-visible prompt:",
        args.prompt_output,
    )
    print(
        "held-out reference:",
        args.reference_output,
    )
    print()
    print(
        "The reference answer is not contained "
        "in the model-visible prompt."
    )


if __name__ == "__main__":
    main()
