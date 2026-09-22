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
    "qualified_inference",
    "qualified_possibility",
}

REVIEW_STATUSES = {
    "open",
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


def validate_annotation_payload(
    value: object,
    *,
    field_name: str,
) -> None:
    if not isinstance(value, dict):
        raise ValueError(
            f"{field_name} must be an object."
        )

    # Intentionally open vocabulary:
    # concrete examples should drive claim-kind expansion.
    require_nonempty_string(
        value.get("claim_kind"),
        field_name=f"{field_name}.claim_kind",
    )

    support_status = value.get("support_status")

    if support_status not in SUPPORT_STATUSES:
        raise ValueError(
            f"Unsupported {field_name}.support_status: "
            f"{support_status!r}"
        )

    assertion_strength = value.get(
        "assertion_strength"
    )

    if assertion_strength not in ASSERTION_STRENGTHS:
        raise ValueError(
            f"Unsupported {field_name}.assertion_strength: "
            f"{assertion_strength!r}"
        )

    require_nonempty_string(
        value.get("rationale"),
        field_name=f"{field_name}.rationale",
    )


def validate_semantic_units(
    value: object,
    *,
    statement_text: str,
    evidence_ids: set[str],
) -> None:
    if value is None:
        return

    if not isinstance(value, list):
        raise ValueError(
            "semantic_units must be an array when present."
        )

    seen_unit_ids: set[str] = set()

    for index, unit in enumerate(value):
        prefix = f"semantic_units[{index}]"

        if not isinstance(unit, dict):
            raise ValueError(
                f"{prefix} must be an object."
            )

        unit_id = require_nonempty_string(
            unit.get("unit_id"),
            field_name=f"{prefix}.unit_id",
        )

        if unit_id in seen_unit_ids:
            raise ValueError(
                f"Duplicate semantic unit ID: {unit_id}"
            )

        seen_unit_ids.add(unit_id)

        unit_text = require_nonempty_string(
            unit.get("text"),
            field_name=f"{prefix}.text",
        )

        span = unit.get("statement_span")

        if not isinstance(span, dict):
            raise ValueError(
                f"{prefix}.statement_span must be an object."
            )

        start = span.get("start")
        end = span.get("end")

        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(
                f"{prefix} offsets must be integers."
            )

        if (
            start < 0
            or end <= start
            or end > len(statement_text)
        ):
            raise ValueError(
                f"{prefix} has invalid statement offsets."
            )

        if statement_text[start:end] != unit_text:
            raise ValueError(
                f"{prefix} text does not match its statement span."
            )

        unit_evidence_ids = unit.get("evidence_ids")

        if (
            not isinstance(unit_evidence_ids, list)
            or not unit_evidence_ids
        ):
            raise ValueError(
                f"{prefix}.evidence_ids must be a non-empty array."
            )

        for evidence_id in unit_evidence_ids:
            require_nonempty_string(
                evidence_id,
                field_name=f"{prefix}.evidence_ids[]",
            )

            if evidence_id not in evidence_ids:
                raise ValueError(
                    f"{prefix} references unknown evidence: "
                    f"{evidence_id}"
                )

        annotation = unit.get(
            "annotation"
        )

        annotations = unit.get(
            "annotations"
        )

        if (
            annotation is not None
            and annotations is not None
        ):
            raise ValueError(
                f"{prefix} cannot contain both "
                "annotation and annotations."
            )

        if annotations is None:
            if annotation is None:
                raise ValueError(
                    f"{prefix} must contain annotation "
                    "or annotations."
                )

            annotations = [
                annotation
            ]

        if (
            not isinstance(annotations, list)
            or not annotations
        ):
            raise ValueError(
                f"{prefix}.annotations must be "
                "a non-empty array."
            )

        for annotation_index, item in enumerate(
            annotations
        ):
            validate_annotation_payload(
                item,
                field_name=(
                    f"{prefix}.annotations"
                    f"[{annotation_index}]"
                ),
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

    statement_text = require_nonempty_string(
        statement.get("text"),
        field_name="statement.text",
    )

    validate_source_span(
        statement.get("source_span")
    )

    validate_generation(
        record.get("generation")
    )

    evidence = record.get("evidence")

    if not isinstance(evidence, list) or not evidence:
        raise ValueError(
            "evidence must contain at least one item."
        )

    evidence_ids: list[str] = []

    for index, item in enumerate(evidence):
        validate_evidence_item(
            item,
            index=index,
        )
        evidence_ids.append(item["evidence_id"])

    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError(
            "Evidence IDs must be unique within a record."
        )

    semantic_units = record.get("semantic_units")

    validate_semantic_units(
        semantic_units,
        statement_text=statement_text,
        evidence_ids=set(evidence_ids),
    )

    has_units = bool(semantic_units)

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

    proposed_by = review.get("proposed_by")

    if proposed_by is not None:
        require_nonempty_string(
            proposed_by,
            field_name="review.proposed_by",
        )

    reviewed_by = review.get("reviewed_by")

    if review_status == "reviewed":
        require_nonempty_string(
            reviewed_by,
            field_name="review.reviewed_by",
        )
    elif reviewed_by is not None:
        raise ValueError(
            "review.reviewed_by must be null "
            "unless review.status is 'reviewed'."
        )

    annotation = record.get("annotation")

    if review_status == "open":
        if annotation is not None:
            raise ValueError(
                "annotation must be null when "
                "review.status is 'open'."
            )

        if has_units:
            raise ValueError(
                "open records cannot contain reviewed semantic units."
            )

        return

    if annotation is None and not has_units:
        raise ValueError(
            "A proposed or reviewed record must contain either "
            "a statement-level annotation or semantic units."
        )

    if annotation is not None:
        validate_annotation_payload(
            annotation,
            field_name="annotation",
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
