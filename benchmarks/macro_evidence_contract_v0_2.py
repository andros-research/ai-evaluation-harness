#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path


SCHEMA_VERSION = "macro_evidence_v0_2"

EVIDENCE_KINDS = {
    "prior_comparison",
    "period_change",
    "availability_note",
}

AVAILABILITY_STATUSES = {
    "available",
    "unavailable_source_gap",
}

DIRECTIONS = {
    "up",
    "down",
    "flat",
}


def load_macro_evidence(
    path: Path,
) -> dict:
    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    validate_macro_evidence(
        payload
    )

    return payload


def _is_number(value) -> bool:
    return (
        isinstance(
            value,
            (int, float),
        )
        and not isinstance(
            value,
            bool,
        )
    )


def validate_macro_evidence(
    payload: dict,
) -> None:
    errors = []

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "Macro evidence payload must be a dict."
        )

    if (
        payload.get(
            "schema_version"
        )
        != SCHEMA_VERSION
    ):
        errors.append(
            "schema_version must be "
            f"{SCHEMA_VERSION!r}"
        )

    observation_date = (
        payload.get(
            "observation_date"
        )
    )

    if (
        not isinstance(
            observation_date,
            str,
        )
        or not observation_date.strip()
    ):
        errors.append(
            "observation_date must be "
            "a non-empty string"
        )

    records = payload.get(
        "records"
    )

    if not isinstance(
        records,
        list,
    ):
        errors.append(
            "records must be a list"
        )
        records = []

    seen_ids = set()

    for index, record in enumerate(
        records
    ):
        prefix = (
            f"records[{index}]"
        )

        if not isinstance(
            record,
            dict,
        ):
            errors.append(
                f"{prefix}: must be a dict"
            )
            continue

        evidence_id = record.get(
            "evidence_id"
        )

        if (
            not isinstance(
                evidence_id,
                str,
            )
            or not evidence_id.strip()
        ):
            errors.append(
                f"{prefix}: invalid evidence_id"
            )

        elif evidence_id in seen_ids:
            errors.append(
                f"{prefix}: duplicate evidence_id "
                f"{evidence_id!r}"
            )

        else:
            seen_ids.add(
                evidence_id
            )

        kind = record.get(
            "evidence_kind"
        )

        if kind not in EVIDENCE_KINDS:
            errors.append(
                f"{prefix}: invalid evidence_kind "
                f"{kind!r}"
            )
            continue

        availability = record.get(
            "availability_status"
        )

        if (
            availability
            not in AVAILABILITY_STATUSES
        ):
            errors.append(
                f"{prefix}: invalid "
                "availability_status "
                f"{availability!r}"
            )

        for field in (
            "metric_name",
            "display_name",
            "source_type",
            "source_series",
            "source_observation_date",
            "comparison_window",
            "units",
            "supporting_values",
        ):
            if field not in record:
                errors.append(
                    f"{prefix}: missing "
                    f"{field!r}"
                )

        if not isinstance(
            record.get(
                "supporting_values"
            ),
            dict,
        ):
            errors.append(
                f"{prefix}: supporting_values "
                "must be a dict"
            )

        if kind == "prior_comparison":
            if availability != "available":
                errors.append(
                    f"{prefix}: prior_comparison "
                    "must be available"
                )

            for field in (
                "current_value",
                "prior_value",
                "delta_value",
            ):
                if not _is_number(
                    record.get(
                        field
                    )
                ):
                    errors.append(
                        f"{prefix}: {field} "
                        "must be numeric"
                    )

            if (
                record.get(
                    "direction"
                )
                not in DIRECTIONS
            ):
                errors.append(
                    f"{prefix}: invalid direction"
                )

        elif kind == "period_change":
            if availability != "available":
                errors.append(
                    f"{prefix}: period_change "
                    "must be available"
                )

            if not _is_number(
                record.get(
                    "current_value"
                )
            ):
                errors.append(
                    f"{prefix}: current_value "
                    "must be numeric"
                )

            if (
                record.get(
                    "direction"
                )
                not in DIRECTIONS
            ):
                errors.append(
                    f"{prefix}: invalid direction"
                )

            for field in (
                "prior_value",
                "delta_value",
            ):
                if (
                    record.get(
                        field
                    )
                    is not None
                ):
                    errors.append(
                        f"{prefix}: {field} "
                        "must be null for "
                        "period_change"
                    )

        elif kind == "availability_note":
            if (
                availability
                == "available"
            ):
                errors.append(
                    f"{prefix}: availability_note "
                    "must represent unavailable data"
                )

            for field in (
                "current_value",
                "prior_value",
                "delta_value",
                "direction",
            ):
                if (
                    record.get(
                        field
                    )
                    is not None
                ):
                    errors.append(
                        f"{prefix}: {field} "
                        "must be null for "
                        "availability_note"
                    )

            source_note = record.get(
                "source_note"
            )

            if (
                not isinstance(
                    source_note,
                    str,
                )
                or not source_note.strip()
            ):
                errors.append(
                    f"{prefix}: source_note "
                    "is required"
                )

    if errors:
        raise ValueError(
            "Macro evidence validation failed:\n"
            + "\n".join(
                f"- {error}"
                for error
                in errors
            )
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "path",
        type=Path,
    )

    args = parser.parse_args()

    payload = load_macro_evidence(
        args.path
    )

    print(
        "MACRO EVIDENCE VALID"
    )
    print(
        "schema:",
        payload[
            "schema_version"
        ],
    )
    print(
        "observation:",
        payload[
            "observation_date"
        ],
    )
    print(
        "records:",
        len(
            payload["records"]
        ),
    )
