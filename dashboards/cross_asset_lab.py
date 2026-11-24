from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from dashboards.fast_market_registry import (
        DEFAULT_FAST_MARKET_EXPERIMENT_ID,
        get_fast_market_experiment,
        list_fast_market_experiments,
        previous_fast_market_experiment,
    )
except ModuleNotFoundError:
    from fast_market_registry import (
        DEFAULT_FAST_MARKET_EXPERIMENT_ID,
        get_fast_market_experiment,
        list_fast_market_experiments,
        previous_fast_market_experiment,
    )


ROOT = Path(__file__).resolve().parents[1]


def _experiment_paths(
    experiment_id: str,
):
    return get_fast_market_experiment(
        experiment_id
    )



def _load_json(
    path: Path,
):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def _load_text(
    path: Path,
):
    return path.read_text(
        encoding="utf-8"
    )


def _fmt_number(
    value,
    decimals=2,
    suffix="",
):
    if value is None:
        return "—"

    return (
        f"{float(value):+.{decimals}f}"
        f"{suffix}"
    )


def _fmt_level(
    value,
    units,
):
    if value is None:
        return "—"

    if units == "percent":
        return f"{float(value):.2f}%"

    if units in {
        "usd_per_eur",
        "jpy_per_usd",
    }:
        return f"{float(value):.4f}"

    if units == "usd_per_barrel":
        return f"${float(value):.2f}"

    if units == "index":
        return f"{float(value):,.2f}"

    return f"{float(value):g}"


def _series_change(
    row,
    key,
):
    value = row.get(
        key
    )

    if value is None:
        return "—"

    suffix = (
        " bp"
        if row.get("transform")
        == "bp_change"
        else "%"
    )

    return (
        f"{float(value):+.2f}"
        f"{suffix}"
    )


def _parse_blind_narratives(
    packet: str,
):
    pattern = re.compile(
        r"^# NARRATIVE_([ABC])\s*$",
        re.MULTILINE,
    )

    matches = list(
        pattern.finditer(
            packet
        )
    )

    narratives = {}

    for index, match in enumerate(
        matches
    ):
        label = (
            "NARRATIVE_"
            + match.group(1)
        )

        start = match.end()

        if (
            index + 1
            < len(matches)
        ):
            end = matches[
                index + 1
            ].start()
        else:
            end = len(packet)

        text = packet[
            start:end
        ].strip()

        # Do not show the blank human-review
        # template as part of the model answer.
        text = text.split(
            "### Human review",
            1,
        )[0].strip()

        # Remove separator carried over
        # from the packet boundary.
        text = text.rstrip(
            "\n- "
        ).strip()

        narratives[
            label
        ] = text

    return narratives


@st.cache_data(
    show_spinner=False
)
def _load_cross_asset_data(
    experiment_id: str,
):
    paths = _experiment_paths(
        experiment_id
    )

    state = _load_json(
        paths[
            "state_json"
        ]
    )

    review_manifest = (
        _load_json(
            paths[
                "review_manifest"
            ]
        )
    )

    packet = _load_text(
        paths[
            "review_packet"
        ]
    )

    narratives = (
        _parse_blind_narratives(
            packet
        )
    )

    return (
        state,
        review_manifest,
        narratives,
    )


@st.cache_data(
    show_spinner=False
)
def _load_fast_state(
    experiment_id: str,
):
    paths = _experiment_paths(
        experiment_id
    )

    return _load_json(
        paths[
            "state_json"
        ]
    )



def _render_state_summary(
    state,
):
    series = state[
        "series_state"
    ]

    state_id = state[
        "source_state_id"
    ]

    manifest_path = (
        ROOT
        / "benchmarks"
        / "results"
        / "market_states"
        / "fred_cross_asset_core_v0_1"
        / state_id
        / "market_state_manifest.json"
    )

    observation_count = None

    if manifest_path.exists():
        manifest = _load_json(
            manifest_path
        )

        observation_count = (
            manifest.get(
                "observation_count"
            )
        )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Series",
        len(series),
    )

    c2.metric(
        "Raw observations",
        (
            observation_count
            if observation_count
            is not None
            else "—"
        ),
    )

    c3.metric(
        "Curve features",
        len(
            state[
                "curve_geometry"
            ]
        ),
    )

    c4.metric(
        "Yield decompositions",
        len(
            state[
                "yield_decompositions"
            ]
        ),
    )

    st.caption(
        "Frozen high-frequency state · "
        f"{state_id} · "
        f"as of {state['as_of_timestamp']}"
    )


def _render_series_state(
    state,
):
    st.subheader(
        "Fast market state"
    )

    st.caption(
        "Daily cross-asset features. "
        "Observation-count changes are "
        "series-specific; synchronized "
        "cross-series relationships are "
        "shown separately below."
    )

    rows = []

    for row in state[
        "series_state"
    ]:
        rows.append({
            "Series":
                row[
                    "display_name"
                ],

            "Latest":
                _fmt_level(
                    row[
                        "latest_value"
                    ],
                    row[
                        "units"
                    ],
                ),

            "Date":
                row[
                    "latest_observation_date"
                ],

            "Age":
                (
                    f"{row['age_days']}d"
                ),

            "1 obs":
                _series_change(
                    row,
                    "change_1obs",
                ),

            "5 obs":
                _series_change(
                    row,
                    "change_5obs",
                ),

            "20 obs":
                _series_change(
                    row,
                    "change_20obs",
                ),

            "63 obs":
                _series_change(
                    row,
                    "change_63obs",
                ),

            "252 obs":
                _series_change(
                    row,
                    "change_252obs",
                ),

            "Z20":
                (
                    "—"
                    if row[
                        "zscore_20"
                    ]
                    is None
                    else (
                        f"{row['zscore_20']:+.2f}"
                    )
                ),

            "Pct252":
                (
                    "—"
                    if row[
                        "percentile_252"
                    ]
                    is None
                    else (
                        f"{row['percentile_252']:.1f}"
                    )
                ),
        })

    st.dataframe(
        pd.DataFrame(
            rows
        ),
        hide_index=True,
        use_container_width=True,
    )


def _render_curve_geometry(
    state,
):
    st.subheader(
        "Synchronized curve geometry"
    )

    st.caption(
        "All cross-series changes below "
        "use dates shared by every "
        "component series."
    )

    rows = []

    for row in state[
        "curve_geometry"
    ]:
        rows.append({
            "Feature":
                row[
                    "display_name"
                ],

            "Date":
                row[
                    "common_date"
                ],

            "Level":
                _fmt_number(
                    row[
                        "level_bp"
                    ],
                    1,
                    " bp",
                ),

            "1 obs":
                _fmt_number(
                    row[
                        "change_1obs_bp"
                    ],
                    1,
                    " bp",
                ),

            "5 obs":
                _fmt_number(
                    row[
                        "change_5obs_bp"
                    ],
                    1,
                    " bp",
                ),

            "20 obs":
                _fmt_number(
                    row[
                        "change_20obs_bp"
                    ],
                    1,
                    " bp",
                ),

            "Z20":
                (
                    "—"
                    if row[
                        "zscore_20"
                    ]
                    is None
                    else (
                        f"{row['zscore_20']:+.2f}"
                    )
                ),

            "Pct252":
                (
                    "—"
                    if row[
                        "percentile_252"
                    ]
                    is None
                    else (
                        f"{row['percentile_252']:.1f}"
                    )
                ),
        })

    st.dataframe(
        pd.DataFrame(
            rows
        ),
        hide_index=True,
        use_container_width=True,
    )


def _render_decompositions(
    state,
):
    st.subheader(
        "Nominal / real / breakeven"
    )

    rows = []

    for row in state[
        "yield_decompositions"
    ]:
        latest = row[
            "latest"
        ]

        h = row[
            "horizons"
        ]

        rows.append({
            "Tenor":
                row[
                    "display_name"
                ].split(
                    " ",
                    1,
                )[0],

            "Date":
                row[
                    "common_date"
                ],

            "Nominal":
                (
                    f"{latest['nominal_percent']:.2f}%"
                ),

            "Real":
                (
                    f"{latest['real_percent']:.2f}%"
                ),

            "Breakeven":
                (
                    f"{latest['breakeven_percent']:.2f}%"
                ),

            "1obs N/R/BE":
                (
                    "—"
                    if h[
                        "obs_1"
                    ] is None
                    else (
                        f"{h['obs_1']['nominal_change_bp']:+.0f}"
                        " / "
                        f"{h['obs_1']['real_change_bp']:+.0f}"
                        " / "
                        f"{h['obs_1']['breakeven_change_bp']:+.0f}"
                        " bp"
                    )
                ),

            "5obs N/R/BE":
                (
                    "—"
                    if h[
                        "obs_5"
                    ] is None
                    else (
                        f"{h['obs_5']['nominal_change_bp']:+.0f}"
                        " / "
                        f"{h['obs_5']['real_change_bp']:+.0f}"
                        " / "
                        f"{h['obs_5']['breakeven_change_bp']:+.0f}"
                        " bp"
                    )
                ),

            "20obs N/R/BE":
                (
                    "—"
                    if h[
                        "obs_20"
                    ] is None
                    else (
                        f"{h['obs_20']['nominal_change_bp']:+.0f}"
                        " / "
                        f"{h['obs_20']['real_change_bp']:+.0f}"
                        " / "
                        f"{h['obs_20']['breakeven_change_bp']:+.0f}"
                        " bp"
                    )
                ),

            "Identity gap":
                (
                    f"{latest['identity_gap_bp']:+.2f} bp"
                ),
        })

    st.dataframe(
        pd.DataFrame(
            rows
        ),
        hide_index=True,
        use_container_width=True,
    )


def _render_blind_analysis(
    review_manifest,
    narratives,
):
    st.subheader(
        "Blind model analysis"
    )

    st.caption(
        "Model identities remain sealed. "
        "Only final answers are displayed; "
        "analysis channels and token telemetry "
        "remain hidden."
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Completed",
        review_manifest[
            "completed_narrative_count"
        ],
    )

    c2.metric(
        "Noncompleted",
        review_manifest[
            "noncompletion_count"
        ],
    )

    c3.metric(
        "Mapping",
        "SEALED",
    )

    c4.metric(
        "Human review",
        "Pending",
    )

    tabs = st.tabs([
        "Narrative A",
        "Narrative B",
        "Narrative C",
    ])

    for tab, label in zip(
        tabs,
        (
            "NARRATIVE_A",
            "NARRATIVE_B",
            "NARRATIVE_C",
        ),
    ):
        with tab:
            text = narratives.get(
                label
            )

            if text:
                st.markdown(
                    text
                )

            else:
                st.warning(
                    f"{label} not found."
                )

    st.info(
        "GPT-OSS high reasoning reached "
        "the frozen token cap without "
        "producing a final answer and is "
        "retained as a noncompletion."
    )


def _observation_delta(
    current: dict,
    previous: dict,
):
    current_value = float(
        current[
            "latest_value"
        ]
    )

    previous_value = float(
        previous[
            "latest_value"
        ]
    )

    transform = current.get(
        "transform"
    )

    if transform == "bp_change":
        return (
            current_value
            - previous_value
        ) * 100.0, "bp"

    if (
        transform == "pct_return"
        and previous_value != 0
    ):
        return (
            (
                current_value
                / previous_value
            )
            - 1.0
        ) * 100.0, "%"

    return None, ""


def _render_longitudinal_change(
    experiment_id: str,
    state: dict,
):
    previous_id = (
        previous_fast_market_experiment(
            experiment_id
        )
    )

    if previous_id is None:
        return

    previous_state = (
        _load_fast_state(
            previous_id
        )
    )

    current_exp = (
        get_fast_market_experiment(
            experiment_id
        )
    )

    previous_exp = (
        get_fast_market_experiment(
            previous_id
        )
    )

    st.subheader(
        "Change since prior fast-market observation"
    )

    st.caption(
        f"{previous_exp['label']} → "
        f"{current_exp['label']}. "
        "These compare the latest available "
        "level in each frozen weekly state. "
        "Source dates are shown explicitly "
        "because different markets may have "
        "different latest observations."
    )

    previous_series = {
        row["series_id"]:
            row
        for row in previous_state[
            "series_state"
        ]
    }

    series_rows = []

    for current in state[
        "series_state"
    ]:
        series_id = current[
            "series_id"
        ]

        previous = (
            previous_series.get(
                series_id
            )
        )

        if previous is None:
            continue

        delta, units = (
            _observation_delta(
                current,
                previous,
            )
        )

        if delta is None:
            delta_display = "—"
        else:
            delta_display = (
                f"{delta:+.2f} "
                f"{units}"
            )

        series_rows.append({
            "Series":
                current[
                    "display_name"
                ],

            "Prior":
                _fmt_level(
                    previous[
                        "latest_value"
                    ],
                    previous[
                        "units"
                    ],
                ),

            "Prior date":
                previous[
                    "latest_observation_date"
                ],

            "Current":
                _fmt_level(
                    current[
                        "latest_value"
                    ],
                    current[
                        "units"
                    ],
                ),

            "Current date":
                current[
                    "latest_observation_date"
                ],

            "Δ state":
                delta_display,
        })

    st.markdown(
        "##### Series levels"
    )

    st.dataframe(
        pd.DataFrame(
            series_rows
        ),
        hide_index=True,
        use_container_width=True,
    )

    previous_curves = {
        row["feature_id"]:
            row
        for row in previous_state[
            "curve_geometry"
        ]
    }

    curve_rows = []

    for current in state[
        "curve_geometry"
    ]:
        previous = (
            previous_curves.get(
                current[
                    "feature_id"
                ]
            )
        )

        if previous is None:
            continue

        delta = (
            float(
                current[
                    "level_bp"
                ]
            )
            - float(
                previous[
                    "level_bp"
                ]
            )
        )

        curve_rows.append({
            "Feature":
                current[
                    "display_name"
                ],

            "Prior":
                (
                    f"{previous['level_bp']:+.1f} bp"
                ),

            "Prior date":
                previous[
                    "common_date"
                ],

            "Current":
                (
                    f"{current['level_bp']:+.1f} bp"
                ),

            "Current date":
                current[
                    "common_date"
                ],

            "Δ state":
                (
                    f"{delta:+.1f} bp"
                ),
        })

    st.markdown(
        "##### Curve geometry"
    )

    st.dataframe(
        pd.DataFrame(
            curve_rows
        ),
        hide_index=True,
        use_container_width=True,
    )


def render_cross_asset_lab():
    experiments = (
        list_fast_market_experiments()
    )

    if not experiments:
        st.info(
            "No Fast Markets experiments "
            "are registered."
        )
        return

    st.divider()

    st.header(
        "Fast Markets"
    )

    st.caption(
        "Higher-frequency cross-asset state "
        "→ deterministic features "
        "→ synchronized geometry "
        "→ blinded model selection."
    )

    default_index = (
        experiments.index(
            DEFAULT_FAST_MARKET_EXPERIMENT_ID
        )
        if (
            DEFAULT_FAST_MARKET_EXPERIMENT_ID
            in experiments
        )
        else 0
    )

    experiment_id = st.selectbox(
        "Fast-market observation / experiment",
        options=experiments,
        index=default_index,
        format_func=lambda item:
            get_fast_market_experiment(
                item
            )[
                "label"
            ],
        key=
            "fast_market_experiment_selector",
    )

    paths = _experiment_paths(
        experiment_id
    )

    required = (
        paths["state_json"],
        paths["state_md"],
        paths["review_packet"],
        paths["review_manifest"],
    )

    if not all(
        path.exists()
        for path in required
    ):
        st.warning(
            "The selected Fast Markets "
            "experiment is incomplete."
        )
        return

    (
        state,
        review_manifest,
        narratives,
    ) = _load_cross_asset_data(
        experiment_id
    )

    _render_state_summary(
        state
    )

    _render_series_state(
        state
    )

    _render_curve_geometry(
        state
    )

    _render_decompositions(
        state
    )

    _render_longitudinal_change(
        experiment_id,
        state,
    )

    _render_blind_analysis(
        review_manifest,
        narratives,
    )

    with st.expander(
        "Frozen model-visible state packet",
        expanded=False,
    ):
        st.markdown(
            _load_text(
                paths[
                    "state_md"
                ]
            )
        )

    st.caption(
        "Experiment ID: "
        f"`{experiment_id}`"
    )

