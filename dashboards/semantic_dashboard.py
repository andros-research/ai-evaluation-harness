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

TOKEN_TELEMETRY_ROOT = Path(
    "benchmarks/results/token_telemetry"
)

FROZEN_TELEMETRY_ROOT = Path(
    "benchmarks/fixtures/"
    "token_telemetry_llama3_reviewed_v0_1"
)

FROZEN_TELEMETRY_PATH = Path(
    "benchmarks/fixtures/"
    "token_telemetry_llama3_semantic_conservative_v0_1.json"
)

FROZEN_TELEMETRY_CLAIM_ID = (
    "llama3_semantic_conservative_t0__commentary_02"
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

EXPLORATORY_TRANSITION_CUES = {
    "llama3_semantic_conservative_t0__commentary_02": {
        "token_index": 6,
        "label": "observation → interpretation",
        "token_text": "indicating",
    },
}


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


def load_token_telemetry(
    claim_id: str,
) -> dict | None:
    if (
        not claim_id
        or claim_id in (".", "..")
        or Path(claim_id).name != claim_id
    ):
        raise ValueError(
            f"Invalid telemetry claim_id: {claim_id!r}"
        )

    filename = "token_telemetry_v0_1.json"

    candidates = [
        TOKEN_TELEMETRY_ROOT / claim_id / filename,
        FROZEN_TELEMETRY_ROOT / claim_id / filename,
    ]

    if claim_id == FROZEN_TELEMETRY_CLAIM_ID:
        candidates.append(
            FROZEN_TELEMETRY_PATH
        )

    path = next(
        (candidate for candidate in candidates
         if candidate.is_file()),
        None,
    )

    if path is None:
        return None

    telemetry = json.loads(
        path.read_text(encoding="utf-8")
    )

    if telemetry.get("claim_id") != claim_id:
        raise RuntimeError(
            "Telemetry artifact claim_id mismatch: "
            f"{path}"
        )

    return telemetry


def token_display_label(
    token: dict,
) -> str:
    fragment = token.get(
        "claim_text_fragment",
        "",
    )

    if fragment:
        stripped = fragment.strip()

        if stripped:
            return stripped

        return fragment

    token_text = token.get(
        "token_text",
        "",
    )

    stripped = token_text.strip()

    return (
        stripped
        if stripped
        else token_text
    )


def token_hover_text(
    token: dict,
) -> str:
    probability = (
        100.0
        * float(
            token.get(
                "probability",
                0.0,
            )
        )
    )

    entropy = float(
        token.get(
            "entropy_nats",
            0.0,
        )
    )

    logprob = float(
        token.get(
            "logprob",
            0.0,
        )
    )

    lines = [
        (
            f"P(token): "
            f"{probability:.2f}%"
        ),
        (
            f"Entropy: "
            f"{entropy:.4f} nats"
        ),
        (
            f"Rank: "
            f"{token.get('rank', '')}"
        ),
        (
            f"LogP: "
            f"{logprob:.4f}"
        ),
        (
            f"Token ID: "
            f"{token.get('token_id', '')}"
        ),
    ]

    top_k = token.get(
        "top_k",
        [],
    )

    if top_k:
        lines.append("")
        lines.append(
            "Top alternatives:"
        )

        for alternative in top_k[:5]:
            lines.append(
                (
                    f"{alternative.get('rank', '')}. "
                    f"{alternative.get('token_text', '')!r} "
                    f"{100.0 * float(alternative.get('probability', 0.0)):.2f}%"
                )
            )

    return "\n".join(
        lines
    )


def select_token(
    *,
    state_key: str,
    token_index: int,
) -> None:
    st.session_state[
        state_key
    ] = token_index


def render_token_telemetry(
    *,
    telemetry: dict,
) -> None:
    st.subheader(
        "Token telemetry"
    )

    source = telemetry.get(
        "source_generation",
        {},
    )

    scoring = telemetry.get(
        "scoring",
        {},
    )

    st.caption(
        "Teacher-forced rescore of the reviewed historical "
        "generation. The original Ollama / Q4_0 logits were "
        "not captured; probabilities shown here come from "
        f"{scoring.get('model', 'the scoring model')} "
        f"using {scoring.get('dtype', 'unknown dtype')} "
        "PyTorch inference."
    )

    tokens = telemetry.get(
        "tokens",
        [],
    )

    if not tokens:
        st.warning(
            "Telemetry artifact contains no token records."
        )
        return

    claim_id = telemetry.get(
        "claim_id",
        "unknown",
    )

    token_state_key = (
        "token_telemetry_selected_"
        + claim_id
    )

    valid_token_indices = list(
        range(
            len(tokens)
        )
    )

    if (
        st.session_state.get(
            token_state_key
        )
        not in valid_token_indices
    ):
        st.session_state[
            token_state_key
        ] = 0

    st.markdown(
        "#### Token strip"
    )

    st.caption(
        "Click a token to inspect its probability "
        "distribution and top alternatives. "
        "Bar = observed-token probability; "
        "H = next-token entropy in nats."
    )

    column_weights = [
        max(
            1.0,
            min(
                3.0,
                0.8
                + 0.18
                * len(
                    token_display_label(
                        token
                    )
                ),
            ),
        )
        for token in tokens
    ]

    token_columns = st.columns(
        column_weights,
        gap="small",
    )

    selected_from_state = (
        st.session_state[
            token_state_key
        ]
    )

    for token_index, (
        column,
        token,
    ) in enumerate(
        zip(
            token_columns,
            tokens,
        )
    ):
        with column:
            label = (
                token_display_label(
                    token
                )
            )

            is_selected = (
                token_index
                == selected_from_state
            )

            st.button(
                label,
                key=(
                    "token_strip_"
                    + claim_id
                    + "_"
                    + str(token_index)
                ),
                type=(
                    "primary"
                    if is_selected
                    else "secondary"
                ),
                use_container_width=True,
                help=token_hover_text(
                    token
                ),
                on_click=select_token,
                kwargs={
                    "state_key":
                        token_state_key,
                    "token_index":
                        token_index,
                },
            )

            probability = float(
                token.get(
                    "probability",
                    0.0,
                )
            )

            probability = max(
                0.0,
                min(
                    1.0,
                    probability,
                ),
            )

            entropy = float(
                token.get(
                    "entropy_nats",
                    0.0,
                )
            )

            st.progress(
                probability
            )

            st.caption(
                f"{100.0 * probability:.1f}%"
                f" · H {entropy:.2f}"
            )

    transition_cue = (
        EXPLORATORY_TRANSITION_CUES.get(
            claim_id
        )
    )

    if transition_cue:
        transition_index = (
            transition_cue.get(
                "token_index"
            )
        )

        if (
            isinstance(
                transition_index,
                int,
            )
            and transition_index
            in valid_token_indices
        ):
            transition_columns = st.columns(
                column_weights,
                gap="small",
            )

            with transition_columns[
                transition_index
            ]:
                st.markdown(
                    (
                        "<div style='"
                        "text-align:center;"
                        "line-height:1.15;"
                        "margin-top:-0.30rem;"
                        "white-space:nowrap;"
                        "'>"
                        "<div style='"
                        "font-size:1.45rem;"
                        "'>↑</div>"
                        "<div style='"
                        "font-size:0.72rem;"
                        "font-weight:600;"
                        "opacity:0.85;"
                        "'>"
                        + transition_cue.get(
                            "label",
                            "",
                        )
                        + "</div>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

            st.caption(
                "Exploratory token-level cue only; "
                "human-reviewed semantic annotations "
                "still apply to the full claim-bearing span."
            )

    # --------------------------------------------------------
    # Reviewed semantic rail.
    #
    # These annotations currently apply to the reviewed
    # claim-bearing span as a whole. They are deliberately
    # not presented as exact token-local boundaries.
    # --------------------------------------------------------

    human_review = telemetry.get(
        "human_review",
        {},
    )

    reviewed_annotations = (
        human_review.get(
            "reviewed_annotations"
        )
        or []
    )

    if reviewed_annotations:
        st.markdown(
            "#### Reviewed semantic layer"
        )

        annotation_columns = st.columns(
            len(
                reviewed_annotations
            ),
            gap="small",
        )

        for column, annotation in zip(
            annotation_columns,
            reviewed_annotations,
        ):
            with column:
                with st.container(
                    border=True
                ):
                    st.markdown(
                        "**"
                        + annotation.get(
                            "claim_kind",
                            "unknown",
                        )
                        + "**"
                    )

                    st.markdown(
                        annotation.get(
                            "support_status",
                            "unknown",
                        )
                        + " · "
                        + annotation.get(
                            "assertion_strength",
                            "unknown",
                        )
                    )

                    rationale = (
                        annotation.get(
                            "rationale",
                            ""
                        )
                    )

                    if rationale:
                        st.caption(
                            rationale
                        )

        st.caption(
            "Human-reviewed annotations currently apply "
            "to the claim-bearing span as a whole. "
            "Exact token-local semantic boundaries are "
            "not yet adjudicated; token-level transitions "
            "remain an exploratory signal."
        )

    token_df = pd.DataFrame(
        [
            {
                "Index":
                    token.get(
                        "claim_token_index"
                    ),
                "Token":
                    token_display_label(
                        token
                    ),
                "P(token)":
                    (
                        f"{100.0 * token.get('probability', 0.0):.2f}%"
                    ),
                "LogP":
                    (
                        f"{token.get('logprob', 0.0):.4f}"
                    ),
                "Entropy":
                    (
                        f"{token.get('entropy_nats', 0.0):.4f}"
                    ),
                "Rank":
                    token.get(
                        "rank"
                    ),
            }
            for token in tokens
        ]
    )

    with st.expander(
        "Token telemetry table",
        expanded=False,
    ):
        st.dataframe(
            token_df,
            hide_index=True,
            use_container_width=True,
        )

    selected_token_index = st.selectbox(
        "Inspect token",
        options=list(
            range(
                len(tokens)
            )
        ),
        format_func=lambda idx: (
            f"{idx:02d} · "
            f"{token_display_label(tokens[idx])}"
        ),
        key=token_state_key,
    )

    selected = tokens[
        selected_token_index
    ]

    metric_a, metric_b, metric_c, metric_d, metric_e = (
        st.columns(5)
    )

    metric_a.metric(
        "Probability",
        (
            f"{100.0 * selected.get('probability', 0.0):.2f}%"
        ),
    )

    metric_b.metric(
        "Log probability",
        (
            f"{selected.get('logprob', 0.0):.4f}"
        ),
    )

    metric_c.metric(
        "Entropy",
        (
            f"{selected.get('entropy_nats', 0.0):.4f}"
        ),
    )

    metric_d.metric(
        "Vocabulary rank",
        selected.get(
            "rank",
            "",
        ),
    )

    metric_e.metric(
        "Token ID",
        selected.get(
            "token_id",
            "",
        ),
    )

    st.caption(
        "Model token: "
        f"{selected.get('token_text', '')!r}"
        " · Claim fragment: "
        f"{selected.get('claim_text_fragment', '')!r}"
        " · Context tokens before observed token: "
        f"{selected.get('context_token_count', '')}"
    )

    with st.expander(
        (
            "Top alternatives for "
            f"{selected.get('token_text', '')!r}"
        )
    ):
        top_k = selected.get(
            "top_k",
            [],
        )

        if not top_k:
            st.caption(
                "No top-k alternatives recorded."
            )
        else:
            top_k_df = pd.DataFrame(
                [
                    {
                        "Rank":
                            alternative.get(
                                "rank"
                            ),
                        "Model token":
                            repr(
                                alternative.get(
                                    "token_text",
                                    "",
                                )
                            ),
                        "Probability":
                            (
                                f"{100.0 * alternative.get('probability', 0.0):.3f}%"
                            ),
                    }
                    for alternative
                    in top_k
                ]
            )

            st.dataframe(
                top_k_df,
                hide_index=True,
                use_container_width=True,
            )

    context = telemetry.get(
        "context",
        {},
    )

    st.caption(
        "Scoring context: "
        f"{context.get('total_tokens_through_claim', '')} "
        "tokens through the end of the claim"
        " · Source model: "
        f"{source.get('model', '')}"
        " · Source temperature: "
        f"{source.get('temperature', '')}"
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


def reset_prompt_filter() -> None:
    """Reset prompt selection after a manual model change."""
    st.session_state.pop(
        "semantic_prompt_filter",
        None,
    )


def navigate_from_telemetry_table() -> None:
    """Synchronize the inspector with a selected corpus row."""
    event = st.session_state.get(
        "token_summary_table"
    )

    if event is None:
        return

    selected_rows = event["selection"]["rows"]

    if not selected_rows:
        return

    targets = st.session_state.get(
        "token_summary_targets",
        [],
    )

    row_index = selected_rows[0]

    if not 0 <= row_index < len(targets):
        return

    target = targets[row_index]

    st.session_state[
        "semantic_selected_claim_id"
    ] = target["claim_id"]

    st.session_state[
        "semantic_model_filter"
    ] = target["source_model"]

    st.session_state[
        "semantic_prompt_filter"
    ] = target["prompt_variant"]


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


inspector_tab, token_tab, semantic_tab = (
    st.tabs([
        "Claim Inspector",
        "Token Analytics",
        "Semantic Discovery",
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
                key="semantic_model_filter",
                on_change=reset_prompt_filter,
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
                key="semantic_prompt_filter",
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

    with st.expander(
        "Machine-proposed semantic annotations",
        expanded=False,
    ):
        for annotation in row.get(
            "semantic_annotations",
            [],
        ):
            render_annotation(
                annotation
            )

    with st.expander(
        "Evidence",
        expanded=False,
    ):
        render_evidence(
            row.get(
                "evidence",
                [],
            )
        )

    with st.expander(
        "Human review",
        expanded=False,
    ):
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

    token_telemetry = (
        load_token_telemetry(
            selected_claim_id
        )
    )

    if (
        token_telemetry
        and token_telemetry.get(
            "claim_id"
        )
        == selected_claim_id
    ):
        st.divider()

        render_token_telemetry(
            telemetry=
                token_telemetry,
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


with semantic_tab:
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


with token_tab:
    # --------------------------------------------------
    # Reviewed Llama 3 token telemetry corpus
    # --------------------------------------------------

    st.subheader(
        "Token telemetry — reviewed Llama 3 corpus"
    )

    summary_filename = (
        "token_telemetry_summary_v0_1.json"
    )

    telemetry_summary_path = next(
        (
            candidate
            for candidate in [
                TOKEN_TELEMETRY_ROOT / summary_filename,
                FROZEN_TELEMETRY_ROOT / summary_filename,
            ]
            if candidate.is_file()
        ),
        None,
    )

    if telemetry_summary_path is None:
        st.info(
            "Token telemetry summary not generated. "
            "Run benchmarks/summarize_token_telemetry.py."
        )

    else:
        telemetry_summary = json.loads(
            telemetry_summary_path.read_text(
                encoding="utf-8"
            )
        )

        claim_summaries = (
            telemetry_summary.get(
                "claims",
                [],
            )
        )

        if (
            telemetry_summary.get("claim_count")
            != len(claim_summaries)
        ):
            st.warning(
                "Telemetry summary claim count mismatch."
            )

        st.caption(
            f"{len(claim_summaries)} reviewed claims. "
            "Teacher-forced BF16 rescoring of "
            "historical Ollama Q4_0 generations. "
            "All metrics are descriptive."
        )

        table_rows = []

        for item in claim_summaries:
            lowest = item[
                "lowest_probability_token"
            ]

            highest = item[
                "highest_entropy_token"
            ]

            short_id = (
                item["claim_id"]
                .replace(
                    "llama3_semantic_",
                    "",
                )
                .replace(
                    "_t0__commentary_",
                    " / ",
                )
            )

            table_rows.append(
                {
                    "Claim": short_id,
                    "Mean P": item[
                        "mean_probability"
                    ],
                    "Mean H": item[
                        "mean_entropy_nats"
                    ],
                    "PPL": item[
                        "claim_perplexity"
                    ],
                    "Rank >1": (
                        f"{item['rank_gt_1_count']}"
                        f"/{item['token_count']}"
                    ),
                    "Lowest P token": (
                        f"{lowest['token_text'].strip()!r} "
                        f"({lowest['probability']:.1%})"
                    ),
                    "Highest H token": (
                        f"{highest['token_text'].strip()!r} "
                        f"({highest['entropy_nats']:.2f})"
                    ),
                    "Claim kinds": ", ".join(
                        item["claim_kinds"]
                    ),
                    "Support": ", ".join(
                        item["support_statuses"]
                    ),
                }
            )

        telemetry_df = pd.DataFrame(
            table_rows
        )

        st.session_state[
            "token_summary_targets"
        ] = [
            {
                "claim_id": item["claim_id"],
                "source_model": item["source_model"],
                "prompt_variant": item["prompt_variant"],
            }
            for item in claim_summaries
        ]

        st.dataframe(
            telemetry_df,
            hide_index=True,
            use_container_width=True,
            selection_mode="single-row",
            on_select=navigate_from_telemetry_table,
            key="token_summary_table",
            column_config={
                "Mean P": st.column_config.NumberColumn(
                    "Mean P",
                    format="%.3f",
                ),
                "Mean H": st.column_config.NumberColumn(
                    "Mean H",
                    format="%.3f",
                ),
                "PPL": st.column_config.NumberColumn(
                    "PPL",
                    format="%.3f",
                ),
            },
        )

        st.caption(
            "Select a table row, then open the "
            "Claim Inspector tab to explore its "
            "full narrative and token telemetry."
        )

        st.caption(
            "Mean P = arithmetic mean observed-token "
            "probability; Mean H = average next-token "
            "entropy in nats; PPL = exp(mean negative "
            "log probability); Rank >1 = historical "
            "tokens not ranked first by the BF16 scorer. "
            "Semantic labels apply to whole claims."
        )
