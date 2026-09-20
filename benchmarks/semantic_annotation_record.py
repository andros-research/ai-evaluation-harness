from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "semantic_annotation_v0_1"

SUPPORT_STATUSES = {
    "supported",
    "contradicted",
    "insufficient_evidence",
}

ASSERTION_STRENGTHS = {
    "asserted",
    "qualified_possibility",
}

REVIEW_STATUSES = {
    "proposed",
    "reviewed",
}


def require_nonempty_string(
    value: object,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string."
        )

    return value


def validate_source_span(
    value: object,
) -> None:
    if value is None:
        return

    if not isinstance(value, dict):
        raise ValueError(
            "statement.source_span must be an object or null."
        )

    require_nonempty_string(
        value.get("path"),
        field_name="statement.source_span.path",
    )

    require_nonempty_string(
        value.get("sha256"),
        field_name="statement.source_span.sha256",
    )

    start = value.get("start")
    end = value.get("end")

    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError(
            "statement source offsets must be integers."
        )

    if start < 0 or end <= start:
        raise ValueError(
            "statement source offsets are invalid."
        )


def validate_generation(
    value: object,
) -> None:
    if value is None:
        return

    if not isinstance(value, dict):
        raise ValueError(
            "generation must be an object or null."
        )

    if value.get("model") is not None:
        require_nonempty_string(
            value["model"],
            field_name="generation.model",
        )

    if value.get("prompt_id") is not None:
        require_nonempty_string(
            value["prompt_id"],
            field_name="generation.prompt_id",
        )

    if value.get("prompt_variant") is not None:
        require_nonempty_string(
            value["prompt_variant"],
            field_name="generation.prompt_variant",
        )

    temperature = value.get("temperature")

    if (
        temperature is not None
        and not isinstance(temperature, (int, float))
    ):
        raise ValueError(
            "generation.temperature must be numeric or null."
        )


def validate_evidence_item(
    value: object,
    *,
    index: int,
) -> None:
    if not isinstance(value, dict):
        raise ValueError(
            f"evidence[{index}] must be an object."
        )

    require_nonempty_string(
        value.get("source_type"),
        field_name=f"evidence[{index}].source_type",
    )

    require_nonempty_string(
        value.get("evidence_id"),
        field_name=f"evidence[{index}].evidence_id",
    )

    source_ref = value.get("source_ref")

    if not isinstance(source_ref, dict):
        raise ValueError(
            f"evidence[{index}].source_ref must be an object."
        )

    if source_ref.get("path") is not None:
        require_nonempty_string(
            source_ref["path"],
            field_name=f"evidence[{index}].source_ref.path",
        )


def validate_annotation_record(
    record: object,
) -> None:
    if not isinstance(record, dict):
        raise ValueError(
            "Semantic annotation record must be an object."
        )

    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "Unexpected semantic annotation schema version."
        )

    require_nonempty_string(
        record.get("annotation_id"),
        field_name="annotation_id",
    )

    require_nonempty_string(
        record.get("origin"),
        field_name="origin",
    )

    statement = record.get("statement")

    if not isinstance(statement, dict):
        raise ValueError(
            "statement must be an object."
        )

    require_nonempty_string(
        statement.get("text"),
        field_name="statement.text",
    )

    validate_source_span(
        statement.get("source_span")
    )

    validate_generation(
        record.get("generation")
    )

    annotation = record.get("annotation")

    if not isinstance(annotation, dict):
        raise ValueError(
            "annotation must be an object."
        )

    # Intentionally open vocabulary:
    # concrete examples should drive claim-kind expansion.
    require_nonempty_string(
        annotation.get("claim_kind"),
        field_name="annotation.claim_kind",
    )

    support_status = annotation.get(
        "support_status"
    )

    if support_status not in SUPPORT_STATUSES:
        raise ValueError(
            "Unsupported annotation.support_status: "
            f"{support_status!r}"
        )

    assertion_strength = annotation.get(
        "assertion_strength"
    )

    if assertion_strength not in ASSERTION_STRENGTHS:
        raise ValueError(
            "Unsupported annotation.assertion_strength: "
            f"{assertion_strength!r}"
        )

    require_nonempty_string(
        annotation.get("rationale"),
        field_name="annotation.rationale",
    )

    evidence = record.get("evidence")

    if not isinstance(evidence, list) or not evidence:
        raise ValueError(
            "evidence must contain at least one item."
        )

    for index, item in enumerate(evidence):
        validate_evidence_item(
            item,
            index=index,
        )

    review = record.get("review")

    if not isinstance(review, dict):
        raise ValueError(
            "review must be an object."
        )

    review_status = review.get("status")

    if review_status not in REVIEW_STATUSES:
        raise ValueError(
            "Unsupported review.status: "
            f"{review_status!r}"
        )

    require_nonempty_string(
        review.get("reviewed_by"),
        field_name="review.reviewed_by",
    )


def validate_annotation_records(
    records: list[dict[str, Any]],
) -> None:
    seen_ids: set[str] = set()

    for record in records:
        validate_annotation_record(record)

        annotation_id = record["annotation_id"]

        if annotation_id in seen_ids:
            raise ValueError(
                f"Duplicate annotation_id: {annotation_id}"
            )

        seen_ids.add(annotation_id)
