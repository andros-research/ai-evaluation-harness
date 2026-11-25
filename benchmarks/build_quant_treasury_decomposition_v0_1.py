#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_STATE_POINTER = (
    ROOT
    / "benchmarks"
    / "results"
    / "market_states"
    / "cross_asset_core_v0_2"
    / "latest_run.json"
)

OUTPUT_ROOT = (
    ROOT
    / "benchmarks"
    / "results"
    / "quant_lab"
    / "treasury_5y_decomposition_v0_1"
)

SERIES = {
    "nominal":
        "ust_5y",

    "real":
        "tips_5y_real",

    "breakeven":
        "breakeven_5y",
}


def sha256_path(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def write_json(
    path: Path,
    payload,
):
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def sign_label(
    value: float,
    *,
    tolerance: float = 1e-10,
) -> str:
    if value > tolerance:
        return "positive"

    if value < -tolerance:
        return "negative"

    return "zero"


def component_relation(
    real_bp: float,
    breakeven_bp: float,
) -> str:
    real_sign = sign_label(
        real_bp
    )

    be_sign = sign_label(
        breakeven_bp
    )

    if (
        real_sign == "zero"
        or be_sign == "zero"
    ):
        return "neutral"

    if real_sign == be_sign:
        return "reinforcing"

    return "opposing"


def quantile(
    values: list[float],
    q: float,
):
    if not values:
        return None

    ordered = sorted(
        float(value)
        for value in values
    )

    if len(ordered) == 1:
        return ordered[0]

    position = (
        q
        * (
            len(ordered)
            - 1
        )
    )

    lower = math.floor(
        position
    )

    upper = math.ceil(
        position
    )

    if lower == upper:
        return ordered[
            lower
        ]

    weight = (
        position
        - lower
    )

    return (
        ordered[lower]
        * (
            1.0
            - weight
        )
        + ordered[upper]
        * weight
    )


def empirical_percentile(
    history: list[float],
    current: float,
):
    if not history:
        return None

    count = sum(
        value <= current
        for value in history
    )

    return (
        100.0
        * count
        / len(history)
    )


def distribution_summary(
    values: list[float],
):
    if not values:
        return {
            "count": 0,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
        }

    return {
        "count":
            len(values),

        "p10":
            quantile(
                values,
                0.10,
            ),

        "p25":
            quantile(
                values,
                0.25,
            ),

        "p50":
            quantile(
                values,
                0.50,
            ),

        "p75":
            quantile(
                values,
                0.75,
            ),

        "p90":
            quantile(
                values,
                0.90,
            ),
    }


def load_state(
    state_arg: str | None,
):
    if state_arg:
        state_path = Path(
            state_arg
        ).resolve()

    else:
        pointer = json.loads(
            DEFAULT_STATE_POINTER.read_text()
        )

        state_path = Path(
            pointer[
                "market_state_path"
            ]
        ).resolve()

    state = json.loads(
        state_path.read_text()
    )

    return (
        state_path,
        state,
    )


def observations_by_date(
    state,
    series_id: str,
):
    rows = {}

    for obs in state[
        "observations"
    ]:
        if (
            obs[
                "series_id"
            ]
            != series_id
        ):
            continue

        if (
            obs.get(
                "status"
            )
            != "observed"
        ):
            continue

        rows[
            obs[
                "observation_date"
            ]
        ] = float(
            obs[
                "value"
            ]
        )

    return rows


def build_windows(
    state,
    horizon: int,
):
    nominal = observations_by_date(
        state,
        SERIES[
            "nominal"
        ],
    )

    real = observations_by_date(
        state,
        SERIES[
            "real"
        ],
    )

    breakeven = observations_by_date(
        state,
        SERIES[
            "breakeven"
        ],
    )

    common_dates = sorted(
        set(nominal)
        & set(real)
        & set(breakeven)
    )

    if (
        len(common_dates)
        <= horizon
    ):
        raise RuntimeError(
            "Not enough synchronized "
            "observations for horizon "
            f"{horizon}."
        )

    windows = []

    for end_index in range(
        horizon,
        len(common_dates),
    ):
        start_index = (
            end_index
            - horizon
        )

        start_date = (
            common_dates[
                start_index
            ]
        )

        end_date = (
            common_dates[
                end_index
            ]
        )

        # Treasury source values are
        # percentage-point levels.
        # Convert changes to basis points.
        nominal_bp = (
            nominal[
                end_date
            ]
            - nominal[
                start_date
            ]
        ) * 100.0

        real_bp = (
            real[
                end_date
            ]
            - real[
                start_date
            ]
        ) * 100.0

        breakeven_bp = (
            breakeven[
                end_date
            ]
            - breakeven[
                start_date
            ]
        ) * 100.0

        identity_gap_bp = (
            nominal_bp
            - real_bp
            - breakeven_bp
        )

        if (
            abs(
                identity_gap_bp
            )
            > 1e-6
        ):
            raise RuntimeError(
                "Treasury decomposition "
                "identity failed: "
                f"{start_date} -> "
                f"{end_date}: "
                f"{identity_gap_bp} bp"
            )

        gross_component_bp = (
            abs(
                real_bp
            )
            + abs(
                breakeven_bp
            )
        )

        if (
            gross_component_bp
            > 1e-12
        ):
            net_to_gross_ratio = (
                abs(
                    nominal_bp
                )
                / gross_component_bp
            )

            offset_ratio = (
                1.0
                - net_to_gross_ratio
            )

            real_abs_share = (
                abs(
                    real_bp
                )
                / gross_component_bp
            )

            breakeven_abs_share = (
                abs(
                    breakeven_bp
                )
                / gross_component_bp
            )

        else:
            net_to_gross_ratio = None
            offset_ratio = None
            real_abs_share = None
            breakeven_abs_share = None

        windows.append({
            "start_date":
                start_date,

            "end_date":
                end_date,

            "horizon_observations":
                horizon,

            "nominal_change_bp":
                nominal_bp,

            "real_change_bp":
                real_bp,

            "breakeven_change_bp":
                breakeven_bp,

            "identity_gap_bp":
                identity_gap_bp,

            "component_relation":
                component_relation(
                    real_bp,
                    breakeven_bp,
                ),

            "gross_component_move_bp":
                gross_component_bp,

            "net_nominal_move_abs_bp":
                abs(
                    nominal_bp
                ),

            "net_to_gross_ratio":
                net_to_gross_ratio,

            "offset_ratio":
                offset_ratio,

            "real_abs_share":
                real_abs_share,

            "breakeven_abs_share":
                breakeven_abs_share,
        })

    return (
        common_dates,
        windows,
    )


def fmt_bp(
    value,
):
    if value is None:
        return "—"

    return (
        f"{value:+.1f} bp"
    )


def fmt_pct(
    value,
):
    if value is None:
        return "—"

    return (
        f"{100.0 * value:.1f}%"
    )


def fmt_percentile(
    value,
):
    if value is None:
        return "—"

    return (
        f"{value:.1f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Quant Lab v0.1: "
            "5Y Treasury nominal / real / "
            "breakeven rolling decomposition."
        )
    )

    parser.add_argument(
        "--state",
        default=None,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    state_path, state = (
        load_state(
            args.state
        )
    )

    common_dates, windows = (
        build_windows(
            state,
            args.horizon,
        )
    )

    current = (
        windows[-1]
    )

    prior = (
        windows[:-1]
    )

    prior_opposing = [
        row
        for row in prior
        if (
            row[
                "component_relation"
            ]
            == "opposing"
        )
    ]

    prior_reinforcing = [
        row
        for row in prior
        if (
            row[
                "component_relation"
            ]
            == "reinforcing"
        )
    ]

    prior_neutral = [
        row
        for row in prior
        if (
            row[
                "component_relation"
            ]
            == "neutral"
        )
    ]

    offset_history = [
        row[
            "offset_ratio"
        ]
        for row in prior
        if (
            row[
                "offset_ratio"
            ]
            is not None
        )
    ]

    opposing_offset_history = [
        row[
            "offset_ratio"
        ]
        for row in prior_opposing
        if (
            row[
                "offset_ratio"
            ]
            is not None
        )
    ]

    gross_history = [
        row[
            "gross_component_move_bp"
        ]
        for row in prior
    ]

    real_abs_history = [
        abs(
            row[
                "real_change_bp"
            ]
        )
        for row in prior
    ]

    breakeven_abs_history = [
        abs(
            row[
                "breakeven_change_bp"
            ]
        )
        for row in prior
    ]

    current_offset = (
        current[
            "offset_ratio"
        ]
    )

    summary = {
        "schema_version":
            "quant_lab_treasury_decomposition_v0_1",

        "lens_id":
            "treasury_5y_decomposition_v0_1",

        "question":
            (
                "Across rolling 20-observation "
                "windows, how are nominal Treasury "
                "moves decomposed between real-yield "
                "and breakeven moves, and how unusual "
                "is the current opposing-direction "
                "configuration?"
            ),

        "source": {
            "state_id":
                state[
                    "state_id"
                ],

            "state_path":
                str(
                    state_path
                ),

            "state_sha256":
                sha256_path(
                    state_path
                ),

            "as_of_timestamp":
                state[
                    "as_of_timestamp"
                ],
        },

        "parameters": {
            "tenor":
                "5Y",

            "horizon_observations":
                args.horizon,

            "series":
                SERIES,

            "change_unit":
                "basis_points",

            "window_definition":
                (
                    "Difference between the "
                    "current synchronized observation "
                    "and the synchronized observation "
                    "20 common dates earlier."
                ),

            "percentile_reference":
                (
                    "Prior rolling windows only; "
                    "the current window is excluded."
                ),
        },

        "definitions": {
            "identity":
                (
                    "nominal_change_bp = "
                    "real_change_bp + "
                    "breakeven_change_bp"
                ),

            "gross_component_move_bp":
                (
                    "|real_change_bp| + "
                    "|breakeven_change_bp|"
                ),

            "offset_ratio":
                (
                    "1 - |nominal_change_bp| / "
                    "gross_component_move_bp"
                ),

            "offset_ratio_interpretation":
                (
                    "0 means components reinforce; "
                    "values approaching 1 mean "
                    "increasing cancellation."
                ),
        },

        "current":
            current,

        "historical": {
            "prior_window_count":
                len(
                    prior
                ),

            "opposing_window_count":
                len(
                    prior_opposing
                ),

            "opposing_window_frequency":
                (
                    len(
                        prior_opposing
                    )
                    / len(
                        prior
                    )
                    if prior
                    else None
                ),

            "reinforcing_window_count":
                len(
                    prior_reinforcing
                ),

            "neutral_window_count":
                len(
                    prior_neutral
                ),

            "offset_ratio_distribution":
                distribution_summary(
                    offset_history
                ),

            "opposing_offset_ratio_distribution":
                distribution_summary(
                    opposing_offset_history
                ),

            "current_offset_percentile_all":
                (
                    empirical_percentile(
                        offset_history,
                        current_offset,
                    )
                    if (
                        current_offset
                        is not None
                    )
                    else None
                ),

            "current_offset_percentile_opposing":
                (
                    empirical_percentile(
                        opposing_offset_history,
                        current_offset,
                    )
                    if (
                        current_offset
                        is not None
                        and opposing_offset_history
                    )
                    else None
                ),

            "current_gross_move_percentile":
                empirical_percentile(
                    gross_history,
                    current[
                        "gross_component_move_bp"
                    ],
                ),

            "current_abs_real_move_percentile":
                empirical_percentile(
                    real_abs_history,
                    abs(
                        current[
                            "real_change_bp"
                        ]
                    ),
                ),

            "current_abs_breakeven_move_percentile":
                empirical_percentile(
                    breakeven_abs_history,
                    abs(
                        current[
                            "breakeven_change_bp"
                        ]
                    ),
                ),
        },

        "interpretation_boundary": (
            "Descriptive lens only. "
            "No causal interpretation, "
            "forecast, regression, or "
            "statistical-significance claim."
        ),
    }

    state_id = (
        state[
            "state_id"
        ]
    )

    out = (
        OUTPUT_ROOT
        / state_id
    )

    out.mkdir(
        parents=True,
        exist_ok=False,
    )

    json_path = (
        out
        / "quant_lens.json"
    )

    csv_path = (
        out
        / "rolling_windows.csv"
    )

    md_path = (
        out
        / "quant_lens.md"
    )

    write_json(
        json_path,
        summary,
    )

    csv_fields = list(
        windows[0].keys()
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=
                csv_fields,
        )

        writer.writeheader()

        writer.writerows(
            windows
        )

    h = summary[
        "historical"
    ]

    lines = [
        "# Quant Lab — Treasury 5Y Decomposition v0.1",
        "",
        "## Question",
        "",
        summary[
            "question"
        ],
        "",
        "## Current 20-observation window",
        "",
        (
            f"- Window: "
            f"{current['start_date']} "
            f"→ {current['end_date']}"
        ),
        (
            f"- Nominal: "
            f"{fmt_bp(current['nominal_change_bp'])}"
        ),
        (
            f"- Real: "
            f"{fmt_bp(current['real_change_bp'])}"
        ),
        (
            f"- Breakeven: "
            f"{fmt_bp(current['breakeven_change_bp'])}"
        ),
        (
            f"- Identity gap: "
            f"{fmt_bp(current['identity_gap_bp'])}"
        ),
        (
            f"- Component relation: "
            f"{current['component_relation']}"
        ),
        (
            f"- Gross component movement: "
            f"{current['gross_component_move_bp']:.1f} bp"
        ),
        (
            f"- Offset ratio: "
            f"{fmt_pct(current['offset_ratio'])}"
        ),
        "",
        "## Historical context",
        "",
        (
            f"- Prior rolling windows: "
            f"{h['prior_window_count']}"
        ),
        (
            f"- Opposing-direction frequency: "
            f"{fmt_pct(h['opposing_window_frequency'])}"
        ),
        (
            f"- Current offset percentile vs all prior windows: "
            f"{fmt_percentile(h['current_offset_percentile_all'])}"
        ),
        (
            f"- Current offset percentile vs prior opposing windows: "
            f"{fmt_percentile(h['current_offset_percentile_opposing'])}"
        ),
        (
            f"- Current gross component-move percentile: "
            f"{fmt_percentile(h['current_gross_move_percentile'])}"
        ),
        (
            f"- Current |real| move percentile: "
            f"{fmt_percentile(h['current_abs_real_move_percentile'])}"
        ),
        (
            f"- Current |breakeven| move percentile: "
            f"{fmt_percentile(h['current_abs_breakeven_move_percentile'])}"
        ),
        "",
        "## Method",
        "",
        (
            "All changes are basis points. "
            "Each row uses dates common to the 5Y nominal, "
            "5Y real and derived 5Y breakeven series."
        ),
        "",
        (
            "Offset ratio = "
            "1 - |Δnominal| / "
            "(|Δreal| + |Δbreakeven|)."
        ),
        "",
        (
            "The current window is excluded from "
            "the historical reference distribution."
        ),
        "",
        "## Boundary",
        "",
        summary[
            "interpretation_boundary"
        ],
        "",
    ]

    md_path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    write_json(
        OUTPUT_ROOT
        / "latest_run.json",
        {
            "lens_id":
                summary[
                    "lens_id"
                ],

            "state_id":
                state_id,

            "quant_lens_json":
                str(
                    json_path
                ),

            "rolling_windows_csv":
                str(
                    csv_path
                ),

            "quant_lens_markdown":
                str(
                    md_path
                ),

            "quant_lens_sha256":
                sha256_path(
                    json_path
                ),

            "rolling_windows_sha256":
                sha256_path(
                    csv_path
                ),

            "markdown_sha256":
                sha256_path(
                    md_path
                ),
        },
    )

    print("=" * 80)
    print(
        "QUANT LAB LENS BUILT"
    )
    print("=" * 80)

    print(
        "Lens:",
        summary[
            "lens_id"
        ],
    )

    print(
        "State:",
        state_id,
    )

    print(
        "Window:",
        current[
            "start_date"
        ],
        "->",
        current[
            "end_date"
        ],
    )

    print()
    print(
        "CURRENT DECOMPOSITION"
    )

    print(
        "  nominal:",
        fmt_bp(
            current[
                "nominal_change_bp"
            ]
        ),
    )

    print(
        "  real:",
        fmt_bp(
            current[
                "real_change_bp"
            ]
        ),
    )

    print(
        "  breakeven:",
        fmt_bp(
            current[
                "breakeven_change_bp"
            ]
        ),
    )

    print(
        "  relation:",
        current[
            "component_relation"
        ],
    )

    print(
        "  gross:",
        f"{current['gross_component_move_bp']:.1f} bp",
    )

    print(
        "  offset:",
        fmt_pct(
            current[
                "offset_ratio"
            ]
        ),
    )

    print()
    print(
        "HISTORICAL CONTEXT"
    )

    print(
        "  prior windows:",
        h[
            "prior_window_count"
        ],
    )

    print(
        "  opposing frequency:",
        fmt_pct(
            h[
                "opposing_window_frequency"
            ]
        ),
    )

    print(
        "  offset pctile / all:",
        fmt_percentile(
            h[
                "current_offset_percentile_all"
            ]
        ),
    )

    print(
        "  offset pctile / opposing:",
        fmt_percentile(
            h[
                "current_offset_percentile_opposing"
            ]
        ),
    )

    print(
        "  gross move pctile:",
        fmt_percentile(
            h[
                "current_gross_move_percentile"
            ]
        ),
    )

    print()
    print(
        "JSON:",
        json_path,
    )

    print(
        "CSV:",
        csv_path,
    )

    print(
        "Markdown:",
        md_path,
    )


if __name__ == "__main__":
    main()
