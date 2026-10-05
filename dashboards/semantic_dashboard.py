#!/usr/bin/env python3

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st


DEFAULT_BATCH_ROOT = Path(
    "benchmarks/results/"
    "semantic_dashboard_batch/"
    "fred_semantic_discovery_20260920_t0_v0_1"
)

CLAIMS_PATH = (
    DEFAULT_BATCH_ROOT
    / "dashboard_claims.jsonl"
)

REVIEWS_PATH = (
    DEFAULT_BATCH_ROOT
    / "semantic_reviews.jsonl"
)

KNOWN_CLAIM_KINDS = {
    "empirical_comparison",
    "unit_conversion",
    "temporal_pattern",
    "causal_explanation",
    "scope_generalization",
    "magnitude_judgment",
    "market_interpretation",
    "policy_interpretation",
}

SUPPORT_STATUSES = [
    "supported",
    "contradicted",
    "insufficient_evidence",
]

ASSERTION_STRENGTHS = [
    "asserted",
    "qualified_inference",
    "qualified_possibility",
]

REVIEW_STATUSES = [
    "accepted",
    "corrected",
    "deferred",
]


def load_jsonl(
    path: Path,
) -> list[dict]:
    if not path.exists():
        return []

    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def write_jsonl_atomic(
    path: Path,
    rows: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = (
        path.parent
        / f".{path.name}.tmp"
    )

    temp_path.write_text(
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

    temp_path.replace(
        path
    )


def load_reviews() -> dict[str, dict]:
    return {
        row["claim_id"]: row
        for row in load_jsonl(
            REVIEWS_PATH
        )
    }


def save_review(
    review: dict,
) -> None:
    reviews = load_reviews()

    reviews[
        review["claim_id"]
    ] = review

    ordered = [
        reviews[key]
        for key in sorted(
            reviews
        )
    ]

    write_jsonl_atomic(
        REVIEWS_PATH,
        ordered,
    )


def reviews_as_jsonl() -> str:
    rows = load_jsonl(
        REVIEWS_PATH
    )

    if not rows:
        return ""

    return (
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        )
    )


def machine_annotations_df(
    row: dict,
) -> pd.DataFrame:
    annotations = row.get(
        "semantic_annotations",
        [],
    )

    return pd.DataFrame(
        [
            {
                "claim_kind":
                    item.get(
                        "claim_kind",
                        "",
                    ),
                "support_status":
                    item.get(
                        "support_status",
                        "",
                    ),
                "assertion_strength":
                    item.get(
                        "assertion_strength",
                        "",
                    ),
                "rationale":
                    item.get(
                        "rationale",
                        "",
                    ),
            }
            for item in annotations
        ]
    )


def review_annotations_df(
    *,
    row: dict,
    existing_review: dict | None,
) -> pd.DataFrame:
    if (
        existing_review
        and existing_review.get(
            "reviewed_annotations"
        )
    ):
        annotations = (
            existing_review[
                "reviewed_annotations"
            ]
        )
    else:
        annotations = row.get(
            "semantic_annotations",
            [],
        )

    return pd.DataFrame(
        [
            {
                "claim_kind":
                    item.get(
                        "claim_kind",
                        "",
                    ),
                "support_status":
                    item.get(
                        "support_status",
                        "",
                    ),
                "assertion_strength":
                    item.get(
                        "assertion_strength",
                        "",
                    ),
                "rationale":
                    item.get(
                        "rationale",
                        "",
                    ),
            }
            for item in annotations
        ],
        columns=[
            "claim_kind",
            "support_status",
            "assertion_strength",
            "rationale",
        ],
    )


def dataframe_to_annotations(
    frame: pd.DataFrame,
) -> list[dict]:
    annotations = []

    for record in frame.to_dict(
        orient="records"
    ):
        claim_kind = str(
            record.get(
                "claim_kind",
                "",
            )
        ).strip()

        if not claim_kind:
            continue

        support_status = str(
            record.get(
                "support_status",
                "",
            )
        ).strip()

        assertion_strength = str(
            record.get(
                "assertion_strength",
                "",
            )
        ).strip()

        rationale = str(
            record.get(
                "rationale",
                "",
            )
        ).strip()

        if (
            support_status
            not in SUPPORT_STATUSES
        ):
            raise ValueError(
                "Every reviewed annotation must "
                "have a valid support status."
            )

        if (
            assertion_strength
            not in ASSERTION_STRENGTHS
        ):
            raise ValueError(
                "Every reviewed annotation must "
                "have a valid assertion strength."
            )

        annotations.append({
            "claim_kind":
                claim_kind,
            "support_status":
                support_status,
            "assertion_strength":
                assertion_strength,
            "rationale":
                rationale,
        })

    return annotations


def review_status_for(
    claim_id: str,
    reviews: dict[str, dict],
) -> str:
    review = reviews.get(
        claim_id
    )

    if not review:
        return "unreviewed"

    return review.get(
        "review_status",
        "unreviewed",
    )


def render_annotation(
    annotation: dict,
) -> None:
    claim_kind = annotation.get(
        "claim_kind",
        "",
    )

    is_known = (
        claim_kind
        in KNOWN_CLAIM_KINDS
    )

    with st.container(
        border=True
    ):
        left, middle, right = (
            st.columns(
                [2, 1.4, 1.4]
            )
        )

        with left:
            if is_known:
                st.markdown(
                    f"**{claim_kind}**"
                )
            else:
                st.markdown(
                    f"**{claim_kind}**  "
                    ":orange-badge[NEW LABEL]"
                )

        with middle:
            st.caption(
                "Support"
            )
            st.write(
                annotation.get(
                    "support_status",
                    "",
                )
            )

        with right:
            st.caption(
                "Assertion strength"
            )
            st.write(
                annotation.get(
                    "assertion_strength",
                    "",
                )
            )

        rationale = annotation.get(
            "rationale"
        )

        if rationale:
            st.caption(
                "Rationale"
            )
            st.write(
                rationale
            )


def render_evidence(
    evidence_items: list[dict],
) -> None:
    for index, item in enumerate(
        evidence_items,
        start=1,
    ):
        payload = (
            item.get(
                "payload"
            )
            or {}
        )

        title = (
            payload.get(
                "source_series_name"
            )
            or payload.get(
                "metric_name"
            )
            or item.get(
                "source_type"
            )
            or item.get(
                "evidence_id"
            )
            or f"Evidence {index}"
        )

        with st.expander(
            title,
            expanded=True,
        ):
            claim_text = (
                payload.get(
                    "claim_text"
                )
            )

            if claim_text:
                st.write(
                    claim_text
                )

            observations = (
                payload.get(
                    "observations"
                )
            )

            if isinstance(
                observations,
                list,
            ):
                st.dataframe(
                    pd.DataFrame(
                        observations
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

            with st.expander(
                "Raw evidence payload"
            ):
                st.json(
                    payload
                )


def build_discovery_summary(
    rows: list[dict],
    reviews: dict[str, dict],
) -> dict:
    kind_counts = Counter()
    support_counts = Counter()
    unknown_counts = Counter()
    review_counts = Counter()

    annotation_count = 0

    for row in rows:
        for annotation in row.get(
            "semantic_annotations",
            [],
        ):
            annotation_count += 1

            kind = annotation.get(
                "claim_kind"
            )

            support = annotation.get(
                "support_status"
            )

            kind_counts[
                kind
            ] += 1

            support_counts[
                support
            ] += 1

            if (
                kind
                not in KNOWN_CLAIM_KINDS
            ):
                unknown_counts[
                    kind
                ] += 1

        review_counts[
            review_status_for(
                row["claim_id"],
                reviews,
            )
        ] += 1

    source_runs = {
        row["claim_id"].split(
            "__commentary_",
            1,
        )[0]
        for row in rows
    }

    return {
        "n_claims":
            len(rows),
        "n_annotations":
            annotation_count,
        "n_source_runs":
            len(
                source_runs
            ),
        "n_classifier_success":
            sum(
                row.get("status")
                == "success"
                for row in rows
            ),
        "kind_counts":
            kind_counts,
        "support_counts":
            support_counts,
        "unknown_counts":
            unknown_counts,
        "review_counts":
            review_counts,
    }


def model_prompt_summary(
    rows: list[dict],
) -> pd.DataFrame:
    output = []

    groups = {}

    for row in rows:
        key = (
            row.get(
                "source_model"
            ),
            row.get(
                "prompt_variant"
            ),
        )

        groups.setdefault(
            key,
            [],
        ).append(
            row
        )

    for (
        source_model,
        prompt_variant,
    ), group in groups.items():
        annotations = [
            annotation
            for row in group
            for annotation
            in row.get(
                "semantic_annotations",
                [],
            )
        ]

        output.append({
            "model":
                source_model,
            "prompt":
                prompt_variant,
            "claims":
                len(group),
            "annotations":
                len(
                    annotations
                ),
            "insufficient_evidence":
                sum(
                    annotation.get(
                        "support_status"
                    )
                    == "insufficient_evidence"
                    for annotation
                    in annotations
                ),
            "supported":
                sum(
                    annotation.get(
                        "support_status"
                    )
                    == "supported"
                    for annotation
                    in annotations
                ),
            "new_labels":
                sum(
                    annotation.get(
                        "claim_kind"
                    )
                    not in KNOWN_CLAIM_KINDS
                    for annotation
                    in annotations
                ),
        })

    return pd.DataFrame(
        output
    )


def select_claim(
    claim_id: str,
) -> None:
    st.session_state[
        "semantic_selected_claim_id"
    ] = claim_id


def render_model_output(
    *,
    run_rows: list[dict],
    selected_claim_id: str,
) -> None:
    """Render the full source narrative with analyzed claims clickable."""
    narrative_paths = {
        row.get(
            "source",
            {},
        ).get(
            "narrative_path"
        )
        for row in run_rows
        if row.get(
            "source",
            {},
        ).get(
            "narrative_path"
        )
    }

    if len(narrative_paths) != 1:
        st.warning(
            "Could not resolve exactly one "
            "source narrative for this run."
        )
        return

    narrative_path = Path(
        next(
            iter(
                narrative_paths
            )
        )
    )

    if not narrative_path.is_file():
        st.warning(
            f"Source narrative not found: "
            f"{narrative_path}"
        )
        return

    narrative = narrative_path.read_text(
        encoding="utf-8"
    )

    marker = "## Commentary"

    if marker not in narrative:
        st.markdown(
            narrative
        )
        return

    before_commentary, commentary = (
        narrative.split(
            marker,
            1,
        )
    )

    # Preserve the original title + claim-cited summary.
    st.markdown(
        before_commentary.strip()
    )

    st.markdown(
        "## Commentary"
    )

    ordered_rows = sorted(
        run_rows,
        key=lambda row: (
            row.get(
                "source",
                {},
            ).get(
                "commentary_start",
                0,
            )
        ),
    )

    cursor = 0

    for index, row in enumerate(
        ordered_rows,
        start=1,
    ):
        source = row.get(
            "source",
            {},
        )

        start = source.get(
            "commentary_start"
        )

        end = source.get(
            "commentary_end"
        )

        if (
            not isinstance(start, int)
            or not isinstance(end, int)
        ):
            continue

        gap = commentary[
            cursor:start
        ]

        if gap.strip():
            st.markdown(
                gap.strip()
            )

        is_selected = (
            row["claim_id"]
            == selected_claim_id
        )

        button_label = (
            row["claim_span"]
        )

        if is_selected:
            button_label = (
                "▸ "
                + button_label
            )

        st.button(
            button_label,
            key=(
                "model_output_claim_"
                + row["claim_id"]
            ),
            type="tertiary",
            help=(
                "Selected claim"
                if is_selected
                else
                "Click to inspect this claim"
            ),
            on_click=select_claim,
            args=(
                row["claim_id"],
            ),
        )

        cursor = end

    remainder = commentary[
        cursor:
    ]

    if remainder.strip():
        st.markdown(
            remainder.strip()
        )

        st.caption(
            "Additional commentary shown for context. "
            "The current discovery batch classified "
            "only the sampled claim sentences."
        )


st.set_page_config(
    page_title=(
        "Semantic Evaluation Lab"
    ),
    layout="wide",
)

st.title(
    "Semantic Evaluation Lab"
)

st.caption(
    "Where does an observation become an argument? "
    "Machine annotations are proposals for human review — "
    "not an oracle."
)


if not CLAIMS_PATH.exists():
    st.error(
        f"Dashboard claims artifact not found: {CLAIMS_PATH}"
    )
    st.stop()


rows = load_jsonl(
    CLAIMS_PATH
)

reviews = load_reviews()

if not rows:
    st.error(
        "No dashboard claims were found."
    )
    st.stop()


inspector_tab, summary_tab = (
    st.tabs([
        "Claim Inspector",
        "Discovery Summary",
    ])
)


with inspector_tab:
    top_left, top_right = (
        st.columns(
            [1, 1.5]
        )
    )

    models = sorted({
        row.get(
            "source_model"
        )
        for row in rows
    })

    with top_left:
        selected_model = (
            st.selectbox(
                "Source model",
                models,
            )
        )

    prompt_options = sorted({
        row.get(
            "prompt_variant"
        )
        for row in rows
        if row.get(
            "source_model"
        ) == selected_model
    })

    with top_right:
        selected_prompt = (
            st.selectbox(
                "Prompt",
                prompt_options,
            )
        )

    filtered = [
        row
        for row in rows
        if (
            row.get(
                "source_model"
            )
            == selected_model
            and row.get(
                "prompt_variant"
            )
            == selected_prompt
        )
    ]

    if not filtered:
        st.warning(
            "No claims found for this model / prompt."
        )
        st.stop()

    claim_ids = [
        row["claim_id"]
        for row in filtered
    ]

    selected_claim_id = (
        st.session_state.get(
            "semantic_selected_claim_id"
        )
    )

    if (
        selected_claim_id
        not in claim_ids
    ):
        selected_claim_id = (
            claim_ids[0]
        )

        st.session_state[
            "semantic_selected_claim_id"
        ] = selected_claim_id

    st.subheader(
        "Full model output"
    )

    st.caption(
        "Click an analyzed commentary sentence "
        "to inspect its semantic annotations."
    )

    with st.container(
        border=True
    ):
        render_model_output(
            run_rows=filtered,
            selected_claim_id=
                selected_claim_id,
        )

    row = next(
        item
        for item in filtered
        if item["claim_id"]
        == selected_claim_id
    )

    existing_review = (
        reviews.get(
            selected_claim_id
        )
    )

    meta_a, meta_b, meta_c, meta_d = (
        st.columns(4)
    )

    with meta_a:
        st.metric(
            "Generation temp",
            row.get(
                "generation_temperature"
            ),
        )

    with meta_b:
        audit_value = (
            "PASS"
            if row.get(
                "audit_pass"
            )
            else "FAIL"
        )

        st.metric(
            "Legacy audit",
            audit_value,
        )

    with meta_c:
        st.metric(
            "Machine annotations",
            len(
                row.get(
                    "semantic_annotations",
                    [],
                )
            ),
        )

    with meta_d:
        st.metric(
            "Human review",
            review_status_for(
                selected_claim_id,
                reviews,
            ).upper(),
        )

    st.subheader(
        "Claim-bearing span"
    )

    st.markdown(
        f"> {row['claim_span']}"
    )

    st.subheader(
        "Machine-proposed semantic annotations"
    )

    for annotation in row.get(
        "semantic_annotations",
        [],
    ):
        render_annotation(
            annotation
        )

    st.subheader(
        "Evidence"
    )

    render_evidence(
        row.get(
            "evidence",
            [],
        )
    )

    st.subheader(
        "Human review"
    )

    if existing_review:
        st.info(
            "A saved review already exists for this claim."
        )

    initial_status = (
        existing_review.get(
            "review_status"
        )
        if existing_review
        else "accepted"
    )

    initial_status_index = (
        REVIEW_STATUSES.index(
            initial_status
        )
        if initial_status
        in REVIEW_STATUSES
        else 0
    )

    with st.form(
        key=(
            "review_form_"
            + selected_claim_id
        )
    ):
        decision = st.radio(
            "Review decision",
            REVIEW_STATUSES,
            index=
                initial_status_index,
            horizontal=True,
        )

        st.caption(
            "For ACCEPTED, the machine annotations are "
            "copied into the reviewed result. "
            "For CORRECTED, edit the table below."
        )

        editable = (
            review_annotations_df(
                row=row,
                existing_review=
                    existing_review,
            )
        )

        edited = st.data_editor(
            editable,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "claim_kind":
                    st.column_config.TextColumn(
                        "Claim kind",
                        required=True,
                    ),
                "support_status":
                    st.column_config.SelectboxColumn(
                        "Support",
                        options=
                            SUPPORT_STATUSES,
                        required=True,
                    ),
                "assertion_strength":
                    st.column_config.SelectboxColumn(
                        "Assertion strength",
                        options=
                            ASSERTION_STRENGTHS,
                        required=True,
                    ),
                "rationale":
                    st.column_config.TextColumn(
                        "Rationale",
                        width="large",
                    ),
            },
        )

        note = st.text_area(
            "Reviewer note",
            value=(
                existing_review.get(
                    "review_note",
                    ""
                )
                if existing_review
                else ""
            ),
            placeholder=(
                "Why did you accept, correct, or defer this claim?"
            ),
        )

        save_clicked = (
            st.form_submit_button(
                "Save review",
                type="primary",
            )
        )

    if save_clicked:
        try:
            if decision == "accepted":
                reviewed_annotations = (
                    row.get(
                        "semantic_annotations",
                        [],
                    )
                )

            elif (
                decision
                == "corrected"
            ):
                reviewed_annotations = (
                    dataframe_to_annotations(
                        edited
                    )
                )

                if not reviewed_annotations:
                    raise ValueError(
                        "A corrected review must contain "
                        "at least one annotation."
                    )

            else:
                reviewed_annotations = None

            review = {
                "schema_version":
                    "semantic_review_v0_1",
                "claim_id":
                    selected_claim_id,
                "review_status":
                    decision,
                "reviewed_at":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),
                "source_model":
                    row.get(
                        "source_model"
                    ),
                "prompt_variant":
                    row.get(
                        "prompt_variant"
                    ),
                "generation_temperature":
                    row.get(
                        "generation_temperature"
                    ),
                "claim_span":
                    row.get(
                        "claim_span"
                    ),
                "machine_annotations":
                    row.get(
                        "semantic_annotations",
                        [],
                    ),
                "reviewed_annotations":
                    reviewed_annotations,
                "review_note":
                    note.strip(),
            }

            save_review(
                review
            )

            st.success(
                "Review saved to "
                f"{REVIEWS_PATH}"
            )

            st.rerun()

        except Exception as exc:
            st.error(
                str(exc)
            )

    st.divider()

    review_export = (
        reviews_as_jsonl()
    )

    st.download_button(
        "Download review ledger",
        data=review_export,
        file_name=(
            "semantic_reviews.jsonl"
        ),
        mime=(
            "application/x-ndjson"
        ),
        disabled=(
            not review_export
        ),
    )


with summary_tab:
    summary = (
        build_discovery_summary(
            rows,
            reviews,
        )
    )

    metric_a, metric_b, metric_c, metric_d = (
        st.columns(4)
    )

    metric_a.metric(
        "Claims",
        summary[
            "n_claims"
        ],
    )

    metric_b.metric(
        "Annotations",
        summary[
            "n_annotations"
        ],
    )

    metric_c.metric(
        "Source runs",
        summary[
            "n_source_runs"
        ],
    )

    metric_d.metric(
        "Classifier successes",
        summary[
            "n_classifier_success"
        ],
    )

    left, right = (
        st.columns(2)
    )

    with left:
        st.subheader(
            "Proposed claim kinds"
        )

        kind_df = pd.DataFrame(
            [
                {
                    "claim_kind":
                        key,
                    "count":
                        value,
                }
                for key, value
                in summary[
                    "kind_counts"
                ].most_common()
            ]
        )

        st.dataframe(
            kind_df,
            hide_index=True,
            use_container_width=True,
        )

        if not kind_df.empty:
            st.bar_chart(
                kind_df.set_index(
                    "claim_kind"
                )
            )

    with right:
        st.subheader(
            "Support status"
        )

        support_df = pd.DataFrame(
            [
                {
                    "support_status":
                        key,
                    "count":
                        value,
                }
                for key, value
                in summary[
                    "support_counts"
                ].most_common()
            ]
        )

        st.dataframe(
            support_df,
            hide_index=True,
            use_container_width=True,
        )

        if not support_df.empty:
            st.bar_chart(
                support_df.set_index(
                    "support_status"
                )
            )

    st.subheader(
        "New / out-of-ontology labels"
    )

    if summary[
        "unknown_counts"
    ]:
        unknown_df = pd.DataFrame(
            [
                {
                    "claim_kind":
                        key,
                    "count":
                        value,
                }
                for key, value
                in summary[
                    "unknown_counts"
                ].most_common()
            ]
        )

        st.dataframe(
            unknown_df,
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.success(
            "No out-of-ontology labels in this batch."
        )

    st.subheader(
        "Model / prompt comparison"
    )

    comparison_df = (
        model_prompt_summary(
            rows
        )
    )

    st.dataframe(
        comparison_df,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader(
        "Human review progress"
    )

    review_df = pd.DataFrame(
        [
            {
                "review_status":
                    key,
                "count":
                    value,
            }
            for key, value
            in summary[
                "review_counts"
            ].most_common()
        ]
    )

    st.dataframe(
        review_df,
        hide_index=True,
        use_container_width=True,
    )

    if not review_df.empty:
        st.bar_chart(
            review_df.set_index(
                "review_status"
            )
        )

    st.caption(
        "Discovery-batch counts are descriptive only. "
        "They are not stable estimates of model behavior."
    )
