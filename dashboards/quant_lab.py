#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]

LENS_ROOT = (
    ROOT
    / "benchmarks"
    / "results"
    / "quant_lab"
    / "treasury_5y_decomposition_v0_1"
)

LATEST_POINTER = (
    LENS_ROOT
    / "latest_run.json"
)


def _load_json(
    path: Path,
):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


@st.cache_data(
    show_spinner=False
)
def _load_quant_lens():
    pointer = _load_json(
        LATEST_POINTER
    )

    lens_path = Path(
        pointer[
            "quant_lens_json"
        ]
    )

    csv_path = Path(
        pointer[
            "rolling_windows_csv"
        ]
    )

    lens = _load_json(
        lens_path
    )

    windows = pd.read_csv(
        csv_path
    )

    return (
        pointer,
        lens,
        windows,
    )


def _fmt_bp(
    value,
):
    if value is None:
        return "—"

    return (
        f"{float(value):+.1f} bp"
    )


def _fmt_ratio(
    value,
):
    if value is None:
        return "—"

    return (
        f"{100.0 * float(value):.1f}%"
    )


def _fmt_percentile(
    value,
):
    if value is None:
        return "—"

    return (
        f"{float(value):.1f}"
    )


def _render_scatter(
    windows: pd.DataFrame,
):
    plot = windows.copy()

    plot[
        "offset_percent"
    ] = (
        100.0
        * plot[
            "offset_ratio"
        ]
    )

    plot[
        "status"
    ] = "Historical"

    plot.loc[
        plot.index[-1],
        "status",
    ] = "Current"

    historical = plot.iloc[
        :-1
    ].copy()

    current = plot.iloc[
        [-1]
    ].copy()

    tooltip = [
        alt.Tooltip(
            "start_date:N",
            title="Start",
        ),
        alt.Tooltip(
            "end_date:N",
            title="End",
        ),
        alt.Tooltip(
            "component_relation:N",
            title="Relation",
        ),
        alt.Tooltip(
            "nominal_change_bp:Q",
            title="Nominal Δ",
            format="+.1f",
        ),
        alt.Tooltip(
            "real_change_bp:Q",
            title="Real Δ",
            format="+.1f",
        ),
        alt.Tooltip(
            "breakeven_change_bp:Q",
            title="Breakeven Δ",
            format="+.1f",
        ),
        alt.Tooltip(
            "gross_component_move_bp:Q",
            title="Gross move",
            format=".1f",
        ),
        alt.Tooltip(
            "offset_percent:Q",
            title="Offset",
            format=".1f",
        ),
    ]

    base = alt.Chart(
        historical
    ).mark_circle(
        size=55,
        opacity=0.38,
        color="#8b949e",
    ).encode(
        x=alt.X(
            "gross_component_move_bp:Q",
            title=(
                "Gross component movement "
                "(bp)"
            ),
            scale=alt.Scale(
                zero=True
            ),
        ),
        y=alt.Y(
            "offset_percent:Q",
            title="Offset ratio (%)",
            scale=alt.Scale(
                domain=[
                    0,
                    100,
                ]
            ),
        ),
        tooltip=tooltip,
    )

    current_point = alt.Chart(
        current
    ).mark_point(
        size=260,
        filled=True,
        color="#f59e0b",
        stroke="white",
        strokeWidth=1.5,
    ).encode(
        x=alt.X(
            "gross_component_move_bp:Q",
        ),
        y=alt.Y(
            "offset_percent:Q",
        ),
        tooltip=tooltip,
    )

    chart = (
        base
        + current_point
    ).properties(
        height=470,
    ).interactive()

    st.altair_chart(
        chart,
        use_container_width=True,
    )


def render_quant_lab():
    st.header(
        "Quant Lab"
    )

    st.caption(
        "Small, deterministic quantitative "
        "lenses for testing market-state "
        "questions. No model interpretation "
        "is used to calculate the results."
    )

    if not LATEST_POINTER.exists():
        st.info(
            "No Quant Lab lens has been "
            "built yet."
        )
        return

    lens_name = st.selectbox(
        "Lens",
        [
            (
                "5Y nominal / real / "
                "breakeven decomposition"
            )
        ],
        key="quant_lab_lens",
    )

    (
        pointer,
        lens,
        windows,
    ) = _load_quant_lens()

    current = lens[
        "current"
    ]

    historical = lens[
        "historical"
    ]

    st.caption(
        f"Source state: "
        f"`{pointer['state_id']}`"
    )

    st.subheader(
        "Current 20-observation window"
    )

    st.caption(
        f"{current['start_date']} → "
        f"{current['end_date']}"
    )

    (
        c1,
        c2,
        c3,
        c4,
        c5,
    ) = st.columns(5)

    c1.metric(
        "Nominal",
        _fmt_bp(
            current[
                "nominal_change_bp"
            ]
        ),
    )

    c2.metric(
        "Real",
        _fmt_bp(
            current[
                "real_change_bp"
            ]
        ),
    )

    c3.metric(
        "Breakeven",
        _fmt_bp(
            current[
                "breakeven_change_bp"
            ]
        ),
    )

    c4.metric(
        "Gross move",
        (
            f"{current['gross_component_move_bp']:.1f} bp"
        ),
    )

    c5.metric(
        "Offset ratio",
        _fmt_ratio(
            current[
                "offset_ratio"
            ]
        ),
    )

    st.caption(
        "Identity check: "
        f"Δ nominal = Δ real + Δ breakeven · "
        f"gap = "
        f"{current['identity_gap_bp']:+.2f} bp · "
        f"components = "
        f"{current['component_relation']}"
    )

    st.subheader(
        "Historical context"
    )

    h1, h2, h3 = (
        st.columns(3)
    )

    h1.metric(
        "Opposing windows",
        _fmt_ratio(
            historical[
                "opposing_window_frequency"
            ]
        ),
        help=(
            "Share of prior rolling "
            "20-observation windows in which "
            "real yields and breakevens moved "
            "in opposite directions."
        ),
    )

    h2.metric(
        "Offset percentile | opposing",
        _fmt_percentile(
            historical[
                "current_offset_percentile_opposing"
            ]
        ),
        help=(
            "Historical percentile of the "
            "current offset ratio, conditional "
            "on prior windows in which real "
            "yields and breakevens also moved "
            "in opposite directions."
        ),
    )

    h3.metric(
        "Gross-move percentile",
        _fmt_percentile(
            historical[
                "current_gross_move_percentile"
            ]
        ),
        help=(
            "Historical percentile of "
            "|Δreal| + |Δbreakeven|."
        ),
    )

    st.caption(
        f"Reference set: "
        f"{historical['prior_window_count']} "
        "prior rolling windows. "
        "The current window is excluded from "
        "the percentile reference distributions."
    )

    st.subheader(
        "Historical decomposition map"
    )

    st.caption(
        "Each dot is one rolling "
        "20-observation window. "
        "Right = larger gross movement in "
        "real yields and breakevens. "
        "Up = more cancellation between them. "
        "The current window is highlighted."
    )

    _render_scatter(
        windows
    )

    st.markdown(
        """
**How to read the current point**

- Moving **right** means more is happening underneath the nominal Treasury yield.
- Moving **up** means more of that component movement cancels before appearing in the nominal move.
- The present state is much more unusual on the **gross-movement** dimension than on the **conditional cancellation** dimension.
"""
    )

    with st.expander(
        "Methodology and limitations",
        expanded=False,
    ):
        st.markdown(
            f"""
**Question**

{lens['question']}

**Identity**

`Δ nominal = Δ real + Δ breakeven`

**Gross component movement**

`|Δ real| + |Δ breakeven|`

**Offset ratio**

`1 - |Δ nominal| / (|Δ real| + |Δ breakeven|)`

An offset ratio of 0 means the two components reinforce one another. Values approaching 1 mean increasingly strong cancellation.

All rate-product changes are measured in **basis points**.

The historical observations are rolling 20-observation windows and therefore overlap heavily. They are useful as a descriptive historical distribution, but they are **not independent statistical samples**.

{lens['interpretation_boundary']}
"""
        )

    with st.expander(
        "Current lens artifact",
        expanded=False,
    ):
        st.json(
            lens,
            expanded=False,
        )
