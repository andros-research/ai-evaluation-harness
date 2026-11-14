from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from dashboards.experiment_registry import (
        get_experiment,
    )
    from dashboards.model_arm_adapter import (
        load_model_arm,
    )
except ModuleNotFoundError:
    from experiment_registry import (
        get_experiment,
    )
    from model_arm_adapter import (
        load_model_arm,
    )


ROOT = Path(__file__).resolve().parents[1]

STATE_EXPERIMENT = "macro_inspect_experiment_id"
STATE_ARM = "macro_inspect_arm_id"
STATE_SENTENCE = "macro_inspect_sentence_id"


def _read_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


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


def _load_human_reference(
    experiment_id: str,
):
    exp = get_experiment(
        experiment_id
    )

    path = (
        exp["fixture_root"]
        / "human_review_v0_1"
        / "human_macro_semantic_reference_v0_1.json"
    )

    if not path.exists():
        return {}

    return _read_json(path)


def _load_blind_mapping(
    experiment_id: str,
):
    """
    Resolve the post-freeze blind-label mapping from
    frozen artifacts without hard-coding model identities.
    """

    exp = get_experiment(
        experiment_id
    )

    reveal = (
        exp["fixture_root"]
        / "human_review_v0_1"
        / "mapping_reveal_v0_1.json"
    )

    if reveal.exists():
        payload = _read_json(
            reveal
        )

        mapping = payload.get(
            "blind_mapping",
            {},
        )

    else:
        mapping_path = (
            ROOT
            / "benchmarks/results/"
            "macro_narrative_reviews"
            / experiment_id
            / "blind_review_v0_1"
            / "BLIND_MAPPING_DO_NOT_OPEN.json"
        )

        if not mapping_path.exists():
            return {}

        payload = _read_json(
            mapping_path
        )

        mapping = payload.get(
            "mapping",
            {},
        )

    out = {}

    for blind_label, value in mapping.items():
        if isinstance(value, str):
            source_id = value

        elif isinstance(value, dict):
            source_id = (
                value.get("source_id")
                or value.get("arm_id")
                or value.get("model_id")
            )

        else:
            continue

        if source_id:
            out[source_id] = blind_label

    return out


def _blind_label_for_arm(
    experiment_id: str,
    arm_id: str,
):
    return _load_blind_mapping(
        experiment_id
    ).get(
        arm_id
    )


def macro_sentence_options(
    experiment_id: str,
    arm_id: str,
):
    reference = _load_human_reference(
        experiment_id
    )

    blind_label = (
        _blind_label_for_arm(
            experiment_id,
            arm_id,
        )
    )

    if not blind_label:
        return []

    narratives = reference.get(
        "narratives",
        [],
    )

    for narrative in narratives:
        if (
            narrative.get(
                "blind_label"
            )
            == blind_label
        ):
            return narrative.get(
                "commentary_sentences",
                [],
            )

    return []


def prepare_macro_inspection(
    experiment_id: str,
    arm_id: str,
    sentence_id: str,
):
    st.session_state[
        STATE_EXPERIMENT
    ] = experiment_id

    st.session_state[
        STATE_ARM
    ] = arm_id

    st.session_state[
        STATE_SENTENCE
    ] = sentence_id

    st.session_state[
        "claim_inspector_mode"
    ] = "macro"


def _load_scored_case(
    experiment_id: str,
    sentence_id: str,
):
    path = (
        ROOT
        / "benchmarks/results/"
        "macro_semantic_oos"
        / experiment_id
        / "gpt_oss_20b_low_v0_2_1"
        / "scored_comparison_v0_1.json"
    )

    if not path.exists():
        return None

    scored = _read_json(
        path
    )

    for case in scored.get(
        "cases",
        [],
    ):
        if (
            case.get("target_id")
            == sentence_id
        ):
            return case

    return None


def _load_evidence(
    experiment_id: str,
):
    exp = get_experiment(
        experiment_id
    )

    path = (
        exp["fixture_root"]
        / "inputs"
        / "selected_fred_claims.json"
    )

    if not path.exists():
        return []

    return _as_records(
        _read_json(path)
    )


def _annotation_frame(
    annotations,
):
    return pd.DataFrame([
        {
            "Kind":
                row.get(
                    "claim_kind",
                    "",
                ),

            "Support":
                row.get(
                    "support_status",
                    "",
                ),

            "Strength":
                row.get(
                    "assertion_strength",
                    "",
                ),

            "Proposition":
                row.get(
                    "proposition",
                    "",
                ),

            "Rationale":
                row.get(
                    "rationale",
                    "",
                ),
        }
        for row
        in annotations
    ])


def render_macro_claim_inspector():
    experiment_id = (
        st.session_state.get(
            STATE_EXPERIMENT
        )
    )

    arm_id = (
        st.session_state.get(
            STATE_ARM
        )
    )

    sentence_id = (
        st.session_state.get(
            STATE_SENTENCE
        )
    )

    if not (
        experiment_id
        and arm_id
        and sentence_id
    ):
        return False

    exp = get_experiment(
        experiment_id
    )

    arm = load_model_arm(
        arm_id,
        experiment_id,
    )

    sentences = macro_sentence_options(
        experiment_id,
        arm_id,
    )

    sentence = next(
        (
            row
            for row in sentences
            if row.get(
                "sentence_id"
            )
            == sentence_id
        ),
        None,
    )

    if sentence is None:
        st.warning(
            "The selected macro sentence could "
            "not be resolved from the frozen "
            "human reference."
        )

        return True

    st.subheader(
        "Macro experiment inspection"
    )

    st.caption(
        "Read-only longitudinal experiment view. "
        "Human semantic commitments come from the "
        "frozen blind review; classifier proposals "
        "remain machine-generated measurements."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Experiment",
        exp["label"],
    )

    c2.metric(
        "Model",
        arm["label"],
    )

    c3.metric(
        "Sentence",
        sentence_id,
    )

    blind_label = (
        _blind_label_for_arm(
            experiment_id,
            arm_id,
        )
    )

    st.caption(
        f"Observation: {exp['observation_date']} · "
        f"Blind review label: "
        f"{blind_label or '—'}"
    )

    st.markdown(
        "#### Claim-bearing sentence"
    )

    st.markdown(
        f"> {sentence.get('claim_span', '')}"
    )

    human_annotations = (
        sentence.get(
            "annotations",
            [],
        )
    )

    scored_case = _load_scored_case(
        experiment_id,
        sentence_id,
    )

    human_col, machine_col = (
        st.columns(2)
    )

    with human_col:
        st.markdown(
            "#### Frozen human reference"
        )

        if human_annotations:
            st.dataframe(
                _annotation_frame(
                    human_annotations
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No human semantic commitments "
                "were recorded."
            )

    with machine_col:
        st.markdown(
            "#### v0.2.1 classifier"
        )

        if scored_case:
            predicted = (
                scored_case.get(
                    "predicted_valid",
                    [],
                )
            )

            if predicted:
                st.dataframe(
                    _annotation_frame(
                        predicted
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(
                    "No valid classifier "
                    "annotations."
                )

            st.caption(
                "Sentence score: "
                f"{scored_case.get('status', '—')} · "
                "protocol valid: "
                f"{scored_case.get('protocol_valid', False)}"
            )

        else:
            st.info(
                "No frozen OOS classifier score "
                "was found for this sentence."
            )

    with st.expander(
        "Frozen macro evidence",
        expanded=False,
    ):
        evidence = _load_evidence(
            experiment_id
        )

        if evidence:
            evidence_rows = []

            for claim in evidence:
                evidence_rows.append({
                    "Metric":
                        claim.get(
                            "metric_name",
                            "",
                        ),

                    "Prior":
                        claim.get(
                            "prior_value"
                        ),

                    "Current":
                        claim.get(
                            "current_value"
                        ),

                    "Delta":
                        claim.get(
                            "delta_value"
                        ),

                    "Claim":
                        claim.get(
                            "claim_text",
                            "",
                        ),
                })

            st.dataframe(
                pd.DataFrame(
                    evidence_rows
                ),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander(
        "Full generated narrative",
        expanded=False,
    ):
        if arm["narrative"]:
            st.markdown(
                arm["narrative"]
            )
        else:
            st.info(
                "This arm produced no final "
                "narrative."
            )

    st.caption(
        "Sentence-local token alignment is not "
        "yet shown here. Whole-generation token "
        "telemetry remains available in Macro Lab."
    )

    if st.button(
        "Clear macro inspection",
        key="clear_macro_inspection",
    ):
        for key in (
            STATE_EXPERIMENT,
            STATE_ARM,
            STATE_SENTENCE,
            "claim_inspector_mode",
        ):
            st.session_state.pop(
                key,
                None,
            )

        st.rerun()

    return True
