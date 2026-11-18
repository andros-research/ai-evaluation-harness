from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import plotly.graph_objects as go
import pandas as pd
import streamlit as st

try:
    from dashboards.experiment_registry import (
        DEFAULT_EXPERIMENT_ID,
        get_experiment,
        list_experiments,
    )
except ModuleNotFoundError:
    from experiment_registry import (
        DEFAULT_EXPERIMENT_ID,
        get_experiment,
        list_experiments,
    )

try:
    from dashboards.model_arm_adapter import (
        ARM_REGISTRY,
        MODEL_ARM_ORDER,
        load_model_arm,
    )
except ModuleNotFoundError:
    from model_arm_adapter import (
        ARM_REGISTRY,
        MODEL_ARM_ORDER,
        load_model_arm,
    )


try:
    from dashboards.macro_claim_inspector import (
        macro_sentence_options,
        prepare_macro_inspection,
    )
except ModuleNotFoundError:
    from macro_claim_inspector import (
        macro_sentence_options,
        prepare_macro_inspection,
    )


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_ID = "v1_9_3_sep2026_multi_model_v0_1"

FIXTURE_ROOT = (
    ROOT
    / "benchmarks/fixtures/macro_narrative_experiments"
    / EXPERIMENT_ID
)

RESULT_ROOT = (
    ROOT
    / "benchmarks/results/macro_narrative_experiments"
    / EXPERIMENT_ID
)

TELEMETRY_ROOT = (
    ROOT
    / "benchmarks/results/macro_token_telemetry_replay"
    / EXPERIMENT_ID
    / "replay_kernels_0_17_2_v0_1"
)

MODEL_ARMS = {
    arm_id: {
        "label":
            ARM_REGISTRY[
                arm_id
            ]["label"],

        "human_rank":
            ARM_REGISTRY[
                arm_id
            ]["human_rank"],

        "reasoning":
            ARM_REGISTRY[
                arm_id
            ]["reasoning"],
    }
    for arm_id
    in MODEL_ARM_ORDER
}



def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _as_records(obj):
    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in (
            "selected_claims",
            "claims",
            "records",
            "items",
        ):
            value = obj.get(key)

            if isinstance(value, list):
                return value

    return []


def _first_existing(*paths: Path):
    for path in paths:
        if path.exists():
            return path

    return None


def _generation_dir(arm_id: str):
    return (
        RESULT_ROOT
        / arm_id
        / "generation_001"
    )


def _load_generation_metadata(
    arm_id: str,
    experiment_id: str,
):
    return load_model_arm(
        arm_id,
        experiment_id,
    )["metadata"]



def _read_narrative(
    arm_id: str,
    experiment_id: str,
):
    return load_model_arm(
        arm_id,
        experiment_id,
    )["narrative"]



def _load_telemetry(
    arm_id: str,
    experiment_id: str,
):
    return load_model_arm(
        arm_id,
        experiment_id,
    )["telemetry"]



def _channel_tokens(payload, channel):
    if not payload:
        return []

    return [
        row
        for row in payload.get(
            "tokens",
            [],
        )
        if row.get("channel") == channel
    ]


def _mean(xs):
    return (
        statistics.fmean(xs)
        if xs
        else math.nan
    )


def _median(xs):
    return (
        statistics.median(xs)
        if xs
        else math.nan
    )


def _telemetry_summary(tokens):
    if not tokens:
        return {
            "tokens": 0,
            "mean_entropy": math.nan,
            "median_entropy": math.nan,
            "mean_probability": math.nan,
            "median_probability": math.nan,
            "mean_gap": math.nan,
            "p_lt_50": math.nan,
            "p_gt_99": math.nan,
        }

    entropy = [
        row["entropy_nats"]
        for row in tokens
    ]

    prob = [
        row["probability"]
        for row in tokens
    ]

    gap = [
        row["top1_top2_gap"]
        for row in tokens
    ]

    return {
        "tokens": len(tokens),
        "mean_entropy":
            _mean(entropy),
        "median_entropy":
            _median(entropy),
        "mean_probability":
            _mean(prob),
        "median_probability":
            _median(prob),
        "mean_gap":
            _mean(gap),
        "p_lt_50":
            sum(p < 0.50 for p in prob)
            / len(prob),
        "p_gt_99":
            sum(p > 0.99 for p in prob)
            / len(prob),
    }


def _fmt_number(value, digits=3):
    if value is None:
        return "—"

    try:
        if math.isnan(value):
            return "—"
    except TypeError:
        pass

    return f"{value:.{digits}f}"


def _claim_label(
    claim: dict,
) -> str:
    metric = str(
        claim.get("metric_name")
        or ""
    ).strip().lower()

    labels = {
        "cpi_yoy":
            "CPI YoY",

        "cpi_headline_mom":
            "Headline CPI m/m",

        "cpi_core_mom":
            "Core CPI m/m",

        "unrate":
            "Unemployment",

        "fedfunds":
            "Fed funds",

        "gs10":
            "10Y Treasury",
    }

    if metric in labels:
        return labels[metric]

    return (
        claim.get("source_series_name")
        or claim.get("source_series")
        or metric
        or "Unknown series"
    )

def _claim_value(claim, key):
    if key in claim:
        return claim[key]

    supporting = claim.get(
        "supporting_values",
        {}
    )

    if isinstance(supporting, dict):
        return supporting.get(key)

    return None


def _load_selected_claims(
    experiment_id: str,
):
    exp = get_experiment(
        experiment_id
    )

    path = (
        exp["fixture_root"]
        / "inputs/selected_fred_claims.json"
    )

    if not path.exists():
        return []

    return _as_records(
        _load_json(path)
    )



def _model_table(
    experiment_id: str,
):
    exp = get_experiment(
        experiment_id
    )

    rows = []

    for arm_id in exp["arm_order"]:
        arm = load_model_arm(
            arm_id,
            experiment_id,
        )

        telemetry = arm[
            "telemetry"
        ]

        analysis = _telemetry_summary(
            _channel_tokens(
                telemetry,
                "analysis",
            )
        )

        final = _telemetry_summary(
            _channel_tokens(
                telemetry,
                "final",
            )
        )

        rows.append({
            "Model":
                arm["label"],

            "Status":
                arm["status"],

            "Human rank":
                (
                    f"#{arm['human_rank']}"
                    if arm["human_rank"]
                    else "—"
                ),

            "Reasoning":
                arm["reasoning"],

            "Generated tokens":
                arm["generated_tokens"],

            "Analysis tokens":
                arm["analysis_tokens"],

            "Final tokens":
                arm["final_tokens"],

            "Final mean H":
                (
                    round(
                        final[
                            "mean_entropy"
                        ],
                        4,
                    )
                    if final["tokens"]
                    else None
                ),

            "Final mean P":
                (
                    round(
                        final[
                            "mean_probability"
                        ],
                        4,
                    )
                    if final["tokens"]
                    else None
                ),
        })

    return pd.DataFrame(
        rows
    )



def _render_macro_state(experiment_id: str):
    claims = _load_selected_claims(experiment_id)

    st.subheader("Macro state")

    if not claims:
        st.warning(
            "Frozen selected FRED claims "
            "were not found."
        )
        return

    # Scalable macro-state cards.
    # Keep four cards per row for readability,
    # but render every frozen state variable.
    cards_per_row = 4

    for row_start in range(
        0,
        len(claims),
        cards_per_row,
    ):
        row_claims = claims[
            row_start:
            row_start + cards_per_row
        ]

        cols = st.columns(
            cards_per_row
        )

        for col, claim in zip(
            cols,
            row_claims,
        ):
            label = _claim_label(
                claim
            )

            current = _claim_value(
                claim,
                "current_value",
            )

            delta = _claim_value(
                claim,
                "delta_value",
            )

            units = (
                claim.get("units")
                or claim.get(
                    "supporting_values",
                    {},
                ).get("units")
                or ""
            )

            if current is None:
                value = "—"
            else:
                value = f"{current:g}"

            if units:
                value = (
                    f"{value} {units}"
                )

            if delta is None:
                delta_display = None
            else:
                delta_display = (
                    f"{delta:+g}"
                    + (
                        f" {units}"
                        if units
                        else ""
                    )
                )

            col.metric(
                label,
                value,
                delta_display,
            )


    with st.expander(
        "Frozen deterministic claims",
        expanded=False,
    ):
        rows = []

        for claim in claims:
            rows.append({
                "Series":
                    _claim_label(claim),
                "Prior":
                    _claim_value(
                        claim,
                        "prior_value",
                    ),
                "Current":
                    _claim_value(
                        claim,
                        "current_value",
                    ),
                "Delta":
                    _claim_value(
                        claim,
                        "delta_value",
                    ),
                "Direction":
                    claim.get(
                        "direction",
                        "",
                    ),
                "Claim":
                    claim.get(
                        "claim_text",
                        claim.get(
                            "text",
                            "",
                        ),
                    ),
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )



def _prior_experiment_id(
    experiment_id: str,
):
    current = get_experiment(
        experiment_id
    )

    current_date = current[
        "observation_date"
    ]

    candidates = [
        candidate_id
        for candidate_id
        in list_experiments()
        if (
            get_experiment(
                candidate_id
            )["observation_date"]
            < current_date
        )
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda candidate_id:
            get_experiment(
                candidate_id
            )["observation_date"],
    )


def _longitudinal_rows(
    experiment_id: str,
):
    prior_id = _prior_experiment_id(
        experiment_id
    )

    if prior_id is None:
        return None, []

    current_claims = (
        _load_selected_claims(
            experiment_id
        )
    )

    prior_claims = (
        _load_selected_claims(
            prior_id
        )
    )

    def identity(
        claim: dict,
    ) -> str:
        return str(
            claim.get("metric_name")
            or claim.get("source_series")
            or ""
        ).strip().lower()

    current_by_metric = {
        identity(claim):
            claim
        for claim
        in current_claims
        if identity(claim)
    }

    prior_by_metric = {
        identity(claim):
            claim
        for claim
        in prior_claims
        if identity(claim)
    }

    rows = []

    for metric_id, current_claim in (
        current_by_metric.items()
    ):
        prior_claim = (
            prior_by_metric.get(
                metric_id
            )
        )

        # New metrics are not forced into a
        # comparison with some other series.
        if prior_claim is None:
            continue

        current_value = (
            _claim_value(
                current_claim,
                "current_value",
            )
        )

        prior_value = (
            _claim_value(
                prior_claim,
                "current_value",
            )
        )

        if (
            current_value is None
            or prior_value is None
        ):
            change = None
        else:
            change = (
                float(current_value)
                - float(prior_value)
            )

        rows.append({
            "Series":
                _claim_label(
                    current_claim
                ),

            "Metric":
                metric_id,

            "Prior observation":
                prior_value,

            "Current observation":
                current_value,

            "Observation change":
                change,
        })

    return prior_id, rows

def _render_longitudinal_delta(
    experiment_id: str,
):
    exp = get_experiment(
        experiment_id
    )

    prior_id, rows = (
        _longitudinal_rows(
            experiment_id
        )
    )

    st.markdown(
        "#### Change since prior observation"
    )

    if prior_id is None:
        st.caption(
            "No earlier registered observation "
            "is available for comparison."
        )
        return

    prior_exp = get_experiment(
        prior_id
    )

    st.caption(
        f"{prior_exp['label']} → "
        f"{exp['label']}. "
        "These are observation-to-observation "
        "changes in the current values, not the "
        "12-month claim deltas shown above."
    )

    display_rows = []

    for row in rows:
        change = row[
            "Observation change"
        ]

        display_rows.append({
            "Series":
                row["Series"],

            prior_exp["label"]:
                (
                    f"{row['Prior observation']:.3f}"
                    if row[
                        "Prior observation"
                    ] is not None
                    else "—"
                ),

            exp["label"]:
                (
                    f"{row['Current observation']:.3f}"
                    if row[
                        "Current observation"
                    ] is not None
                    else "—"
                ),

            "Δ observation":
                (
                    f"{change:+.3f} pp"
                    if change is not None
                    else "—"
                ),
        })

    st.dataframe(
        pd.DataFrame(
            display_rows
        ),
        use_container_width=True,
        hide_index=True,
    )

def _render_model_comparison(experiment_id: str):
    st.subheader(
        "Multi-model research"
    )

    st.caption(
        "One frozen generation per arm. "
        "Human rank reflects the frozen blind "
        "narrative review for this experiment."
    )

    st.dataframe(
        _model_table(experiment_id),
        use_container_width=True,
        hide_index=True,
    )

    exp = get_experiment(
        experiment_id
    )

    labels = {}

    for arm_id in exp["arm_order"]:
        arm = load_model_arm(
            arm_id,
            experiment_id,
        )

        labels[
            arm["label"]
        ] = arm_id

    selection = st.selectbox(
        "Inspect model",
        options=list(labels),
        index=0,
        key=f"macro_lab_model_{experiment_id}",
    )

    arm_id = labels[selection]

    narrative = _read_narrative(
        arm_id,
        experiment_id,
    )

    if narrative:
        st.markdown(
            narrative
        )
    else:
        st.info(
            "No final narrative was produced "
            "for this arm."
        )

    sentences = macro_sentence_options(
        experiment_id,
        arm_id,
    )

    if sentences:
        sentence_labels = {
            (
                f"{row['sentence_id']} · "
                f"{row['claim_span']}"
            ):
                row["sentence_id"]
            for row in sentences
        }

        sentence_selection = st.selectbox(
            "Inspect commentary sentence",
            options=list(
                sentence_labels
            ),
            key=(
                "macro_sentence_"
                f"{experiment_id}_"
                f"{arm_id}"
            ),
        )

        sentence_id = (
            sentence_labels[
                sentence_selection
            ]
        )

        if st.button(
            "Prepare in Claim Inspector",
            key=(
                "macro_prepare_inspector_"
                f"{experiment_id}_"
                f"{arm_id}_"
                f"{sentence_id}"
            ),
        ):
            prepare_macro_inspection(
                experiment_id,
                arm_id,
                sentence_id,
            )

            st.success(
                "Macro sentence prepared. "
                "Open the Claim Inspector tab "
                "for the frozen semantic view."
            )

    return arm_id


def _render_semantic_status(
    experiment_id: str,
):
    exp = get_experiment(
        experiment_id
    )

    metrics = exp[
        "semantic_evaluation"
    ]

    st.subheader(
        "Semantic evaluation"
    )

    st.caption(
        "Frozen out-of-sample evaluation "
        "against the human-reviewed "
        "commentary for this experiment."
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Strict triple recall",
        f"{metrics['strict_triple_recall']:.1%}",
    )

    c2.metric(
        "Strict triple precision",
        f"{metrics['strict_triple_precision']:.1%}",
    )

    c3.metric(
        "Claim-kind recall",
        f"{metrics['claim_kind_recall']:.1%}",
    )

    c4.metric(
        "Claim-kind precision",
        f"{metrics['claim_kind_precision']:.1%}",
    )

    st.caption(
        f"{metrics['protocol_valid']}/"
        f"{metrics['protocol_total']} "
        "protocol-valid cases; "
        + metrics["note"]
    )



def _render_token_inspector(
    arm_id: str,
    experiment_id: str,
):
    st.subheader(
        "Token telemetry"
    )

    telemetry = _load_telemetry(
        arm_id,
        experiment_id,
    )

    if not telemetry:
        st.info(
            "Compact recovered telemetry "
            "is not available for this arm "
            "in the v0.1 Macro Lab schema."
        )
        return

    analysis = _channel_tokens(
        telemetry,
        "analysis",
    )

    final = _channel_tokens(
        telemetry,
        "final",
    )

    a = _telemetry_summary(
        analysis
    )

    f = _telemetry_summary(
        final
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Analysis tokens",
        a["tokens"],
    )

    c2.metric(
        "Analysis mean entropy",
        _fmt_number(
            a["mean_entropy"],
            4,
        ),
    )

    c3.metric(
        "Final tokens",
        f["tokens"],
    )

    c4.metric(
        "Final mean entropy",
        _fmt_number(
            f["mean_entropy"],
            4,
        ),
    )

    channel_options = []

    if analysis:
        channel_options.append(
            "analysis"
        )

    if final:
        channel_options.append(
            "final"
        )

    if not channel_options:
        return

    channel = st.radio(
        "Channel",
        channel_options,
        horizontal=True,
        key=f"macro_channel_{arm_id}",
    )

    tokens = (
        analysis
        if channel == "analysis"
        else final
    )

    frame = pd.DataFrame({
        "index": [
            row[
                "generation_token_index"
            ]
            for row in tokens
        ],
        "token": [
            row["token_text"]
            .replace("\n", "\\n")
            for row in tokens
        ],
        "probability": [
            row["probability"]
            for row in tokens
        ],
        "entropy": [
            row["entropy_nats"]
            for row in tokens
        ],
        "top1_top2_gap": [
            row["top1_top2_gap"]
            for row in tokens
        ],
    })

    token_state_key = (
        f"macro_token_position_"
        f"{arm_id}_{channel}"
    )

    if token_state_key not in st.session_state:
        st.session_state[token_state_key] = 0

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.caption(
            "Observed-token probability"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=frame["index"],
                y=frame["probability"],
                mode="lines+markers",
                marker={"size": 6},
                customdata=frame[
                    [
                        "token",
                        "entropy",
                        "top1_top2_gap",
                    ]
                ],
                hovertemplate=(
                    "Index: %{x}<br>"
                    "Token: %{customdata[0]}<br>"
                    "Probability: %{y:.4f}<br>"
                    "Entropy: %{customdata[1]:.4f}<br>"
                    "Top1-Top2 gap: %{customdata[2]:.4f}"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            height=220,
            margin=dict(
                l=0,
                r=0,
                t=5,
                b=0,
            ),
            xaxis_title="Generated token index",
            yaxis_title="Probability",
            showlegend=False,
        )

        event = st.plotly_chart(
            fig,
            width="stretch",
            key=f"macro_prob_chart_{arm_id}_{channel}",
            on_select="rerun",
            selection_mode="points",
            config={
                "displayModeBar": False,
            },
        )

        points = []

        if event is not None:
            try:
                points = event.selection.points
            except Exception:
                try:
                    points = event["selection"]["points"]
                except Exception:
                    points = []

        if points:
            point = points[0]

            try:
                selected_generation_index = int(
                    point["x"]
                )
            except Exception:
                selected_generation_index = int(
                    point.x
                )

            matches = frame.index[
                frame["index"]
                == selected_generation_index
            ].tolist()

            if matches:
                st.session_state[
                    token_state_key
                ] = int(matches[0])

    with chart_col2:
        st.caption(
            "Next-token entropy · nats"
        )

        st.line_chart(
            frame.set_index(
                "index"
            )[["entropy"]],
            height=220,
        )

    with st.expander(
        "Token table",
        expanded=False,
    ):
        st.dataframe(
            frame,
            use_container_width=True,
            hide_index=True,
            height=360,
        )

    position = st.number_input(
        "Inspect token position",
        min_value=0,
        max_value=len(tokens) - 1,
        step=1,
        key=token_state_key,
    )

    selected = tokens[
        int(position)
    ]

    left = max(
        0,
        int(position) - 12,
    )

    right = min(
        len(tokens),
        int(position) + 13,
    )

    context = "".join(
        row["token_text"]
        for row in tokens[left:right]
    )

    st.code(
        context,
        language=None,
    )

    m1, m2, m3, m4 = (
        st.columns(4)
    )

    m1.metric(
        "Token",
        repr(
            selected["token_text"]
        ),
    )

    m2.metric(
        "Probability",
        f"{selected['probability']:.2%}",
    )

    m3.metric(
        "Entropy",
        f"{selected['entropy_nats']:.4f}",
    )

    m4.metric(
        "Top1–Top2 gap",
        f"{selected['top1_top2_gap']:.4f}",
    )

    top_k = pd.DataFrame(
        selected.get(
            "top_k",
            [],
        )
    )

    if not top_k.empty:
        st.markdown(
            "**Top alternatives**"
        )

        st.dataframe(
            top_k,
            use_container_width=True,
            hide_index=True,
        )

    if arm_id == "gpt_oss_20b_high":
        st.warning(
            "High reasoning reached the "
            "4,096-token cap with 4,093 "
            "analysis tokens and no final. "
            "The mature repetitive region "
            "had mean entropy 0.1004 and "
            "median chosen-token probability "
            "0.9994. This is preserved as a "
            "descriptive failure-mode result."
        )


def render_macro_lab():
    st.header(
        "Macro Lab"
    )

    st.caption(
        "Frozen macro state → "
        "multi-model narratives → "
        "semantic commitments → "
        "token uncertainty → "
        "longitudinal comparison."
    )

    experiment_ids = (
        list_experiments()
    )

    default_index = (
        experiment_ids.index(
            DEFAULT_EXPERIMENT_ID
        )
        if DEFAULT_EXPERIMENT_ID
        in experiment_ids
        else 0
    )

    experiment_id = st.selectbox(
        "Observation / experiment",
        options=experiment_ids,
        index=default_index,
        format_func=lambda x: (
            get_experiment(x)[
                "label"
            ]
        ),
        key="macro_lab_experiment",
    )

    exp = get_experiment(
        experiment_id
    )

    arms = [
        load_model_arm(
            arm_id,
            experiment_id,
        )
        for arm_id
        in exp["arm_order"]
    ]

    completed = sum(
        arm["completed"]
        for arm in arms
    )

    h1, h2, h3, h4 = (
        st.columns(4)
    )

    h1.metric(
        "Observation",
        exp[
            "observation_date"
        ],
    )

    h2.metric(
        "Experiment",
        exp["label"],
    )

    h3.metric(
        "Model arms",
        len(arms),
    )

    h4.metric(
        "Completed",
        f"{completed} / {len(arms)}",
    )

    st.caption(
        f"Experiment ID: "
        f"`{experiment_id}`"
    )

    st.divider()

    _render_macro_state(
        experiment_id
    )

    _render_longitudinal_delta(
        experiment_id
    )

    st.divider()

    arm_id = (
        _render_model_comparison(
            experiment_id
        )
    )

    st.divider()

    _render_semantic_status(
        experiment_id
    )

    st.divider()

    _render_token_inspector(
        arm_id,
        experiment_id,
    )

