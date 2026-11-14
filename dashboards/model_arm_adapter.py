from __future__ import annotations

import json
from pathlib import Path

try:
    from dashboards.experiment_registry import (
        DEFAULT_EXPERIMENT_ID,
        get_experiment,
    )
except ModuleNotFoundError:
    from experiment_registry import (
        DEFAULT_EXPERIMENT_ID,
        get_experiment,
    )


ARM_DEFINITIONS = {
    "gpt_oss_20b_medium": {
        "label": "GPT-OSS 20B · medium",
        "backend": "gpt_oss",
        "reasoning": "medium",
    },

    "llama3_8b_bf16": {
        "label": "Llama 3 8B · BF16",
        "backend": "native_fred",
        "reasoning": "—",
    },

    "gpt_oss_20b_low": {
        "label": "GPT-OSS 20B · low",
        "backend": "gpt_oss",
        "reasoning": "low",
    },

    "gpt_oss_20b_high": {
        "label": "GPT-OSS 20B · high",
        "backend": "gpt_oss",
        "reasoning": "high",
    },
}


# Compatibility aliases for existing dashboard code.
_default_exp = get_experiment(
    DEFAULT_EXPERIMENT_ID
)

MODEL_ARM_ORDER = list(
    _default_exp["arm_order"]
)

ARM_REGISTRY = {
    arm_id: {
        **definition,
        "human_rank":
            _default_exp[
                "model_arms"
            ][arm_id][
                "human_rank"
            ],
    }
    for arm_id, definition
    in ARM_DEFINITIONS.items()
}


def _read_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def _read_text(path: Path):
    if not path.exists():
        return ""

    return path.read_text(
        encoding="utf-8"
    ).strip()


def _probability_gap(row):
    existing = row.get(
        "top1_top2_gap"
    )

    if existing is not None:
        return float(existing)

    top_k = row.get(
        "top_k",
        [],
    )

    probs = []

    for candidate in top_k:
        try:
            probs.append(
                float(
                    candidate[
                        "probability"
                    ]
                )
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

    if len(probs) < 2:
        return 0.0

    return (
        probs[0]
        - probs[1]
    )


def _normalize_token(
    row,
    *,
    channel,
):
    out = dict(row)

    out["channel"] = (
        row.get("channel")
        or channel
    )

    out[
        "generation_token_index"
    ] = int(
        row[
            "generation_token_index"
        ]
    )

    out["token_id"] = int(
        row["token_id"]
    )

    out["probability"] = float(
        row["probability"]
    )

    out["logprob"] = float(
        row["logprob"]
    )

    out["entropy_nats"] = float(
        row["entropy_nats"]
    )

    if row.get("rank") is not None:
        out["rank"] = int(
            row["rank"]
        )

    out[
        "top1_top2_gap"
    ] = _probability_gap(
        row
    )

    normalized_top_k = []

    for rank, candidate in enumerate(
        row.get(
            "top_k",
            [],
        ),
        start=1,
    ):
        item = dict(
            candidate
        )

        item["rank"] = int(
            item.get(
                "rank",
                rank,
            )
        )

        if item.get(
            "token_id"
        ) is not None:
            item[
                "token_id"
            ] = int(
                item[
                    "token_id"
                ]
            )

        if item.get(
            "probability"
        ) is not None:
            item[
                "probability"
            ] = float(
                item[
                    "probability"
                ]
            )

        normalized_top_k.append(
            item
        )

    out["top_k"] = (
        normalized_top_k
    )

    return out


def _load_gpt_narrative(
    run_dir,
    metadata,
):
    for filename in (
        "narrative.md",
        "narrative.txt",
        "final.md",
        "final.txt",
        "output.md",
        "output.txt",
    ):
        text = _read_text(
            run_dir / filename
        )

        if text:
            return text

    for key in (
        "narrative",
        "narrative_text",
        "final",
        "final_text",
        "output_text",
    ):
        value = metadata.get(
            key
        )

        if (
            isinstance(
                value,
                str,
            )
            and value.strip()
        ):
            return value.strip()

    if run_dir.exists():
        for path in sorted(
            list(
                run_dir.glob(
                    "*.md"
                )
            )
            + list(
                run_dir.glob(
                    "*.txt"
                )
            )
        ):
            if (
                "analysis"
                in path.name.lower()
            ):
                continue

            text = _read_text(
                path
            )

            if text:
                return text

    return ""


def _load_gpt_arm(
    arm_id,
    experiment_id,
):
    exp = get_experiment(
        experiment_id
    )

    definition = (
        ARM_DEFINITIONS[
            arm_id
        ]
    )

    exp_arm = (
        exp["model_arms"][
            arm_id
        ]
    )

    run_dir = (
        exp[
            "gpt_result_root"
        ]
        / arm_id
        / "generation_001"
    )

    metadata_path = (
        run_dir
        / "generation_metadata.json"
    )

    telemetry_layout = exp.get(
        "gpt_telemetry_layout",
        "replay_arm_dir",
    )

    if telemetry_layout == "generation_dir":
        # Newer experiments capture compact telemetry
        # during the original GPT-OSS generation.
        telemetry_path = (
            run_dir
            / "token_telemetry.json"
        )

    elif telemetry_layout == "replay_arm_dir":
        # Historical September experiment recovered
        # telemetry through exact deterministic replay.
        telemetry_path = (
            exp[
                "gpt_telemetry_root"
            ]
            / arm_id
            / "token_telemetry.json"
        )

    else:
        raise ValueError(
            "Unknown GPT telemetry layout: "
            f"{telemetry_layout}"
        )

    metadata = (
        _read_json(
            metadata_path
        )
        if metadata_path.exists()
        else {}
    )

    raw_telemetry = (
        _read_json(
            telemetry_path
        )
        if telemetry_path.exists()
        else {}
    )

    tokens = [
        _normalize_token(
            row,
            channel=row.get(
                "channel",
                "final",
            ),
        )
        for row
        in raw_telemetry.get(
            "tokens",
            [],
        )
    ]

    analysis = [
        row
        for row in tokens
        if row[
            "channel"
        ] == "analysis"
    ]

    final = [
        row
        for row in tokens
        if row[
            "channel"
        ] == "final"
    ]

    generated_count = int(
        metadata.get(
            "generated_token_count",
            len(tokens),
        )
    )

    analysis_count = int(
        metadata.get(
            "analysis_token_count",
            len(analysis),
        )
    )

    final_count = int(
        metadata.get(
            "final_token_count",
            len(final),
        )
    )

    reached_cap = bool(
        metadata.get(
            "reached_token_cap",
            False,
        )
    )

    if final_count > 0:
        narrative = (
            _load_gpt_narrative(
                run_dir,
                metadata,
            )
        )
    else:
        narrative = ""

    completed = (
        final_count > 0
        and bool(narrative)
    )

    if completed:
        status = "Complete"
    elif reached_cap:
        status = (
            "Noncomplete · token cap"
        )
    else:
        status = "Noncomplete"

    return {
        "experiment_id":
            experiment_id,

        "arm_id":
            arm_id,

        "label":
            definition[
                "label"
            ],

        "backend":
            definition[
                "backend"
            ],

        "human_rank":
            exp_arm[
                "human_rank"
            ],

        "reasoning":
            definition[
                "reasoning"
            ],

        "status":
            status,

        "completed":
            completed,

        "narrative":
            narrative,

        "generated_tokens":
            generated_count,

        "analysis_tokens":
            analysis_count,

        "final_tokens":
            final_count,

        "reached_token_cap":
            reached_cap,

        "telemetry": {
            "tokens":
                tokens,
        },

        "metadata": {
            "generated_token_count":
                generated_count,

            "analysis_token_count":
                analysis_count,

            "final_token_count":
                final_count,

            "reached_token_cap":
                reached_cap,
        },

        "raw_metadata":
            metadata,

        "provenance": {
            "generation_dir":
                str(run_dir),

            "metadata_path":
                str(
                    metadata_path
                ),

            "telemetry_path":
                str(
                    telemetry_path
                ),

            "backend":
                definition[
                    "backend"
                ],
        },
    }


def _load_native_fred_arm(
    arm_id,
    experiment_id,
):
    exp = get_experiment(
        experiment_id
    )

    definition = (
        ARM_DEFINITIONS[
            arm_id
        ]
    )

    exp_arm = (
        exp["model_arms"][
            arm_id
        ]
    )

    run_root = Path(
        exp_arm[
            "run_root"
        ]
    )

    metadata_path = (
        run_root
        / "native_generation_metadata.json"
    )

    telemetry_path = (
        run_root
        / "native_generation_telemetry.json"
    )

    narrative_path = (
        run_root
        / "fred_narrative.md"
    )

    metadata = _read_json(
        metadata_path
    )

    raw_telemetry = _read_json(
        telemetry_path
    )

    narrative = _read_text(
        narrative_path
    )

    tokens = [
        _normalize_token(
            row,
            channel="final",
        )
        for row
        in raw_telemetry.get(
            "tokens",
            [],
        )
    ]

    generated_count = int(
        metadata[
            "generated_token_count"
        ]
    )

    completed = (
        generated_count > 0
        and bool(narrative)
    )

    return {
        "experiment_id":
            experiment_id,

        "arm_id":
            arm_id,

        "label":
            definition[
                "label"
            ],

        "backend":
            definition[
                "backend"
            ],

        "human_rank":
            exp_arm[
                "human_rank"
            ],

        "reasoning":
            definition[
                "reasoning"
            ],

        "status":
            (
                "Complete"
                if completed
                else "Noncomplete"
            ),

        "completed":
            completed,

        "narrative":
            narrative,

        "generated_tokens":
            generated_count,

        "analysis_tokens":
            0,

        "final_tokens":
            generated_count,

        "reached_token_cap":
            False,

        "telemetry": {
            "tokens":
                tokens,
        },

        "metadata": {
            "generated_token_count":
                generated_count,

            "analysis_token_count":
                0,

            "final_token_count":
                generated_count,

            "reached_token_cap":
                False,
        },

        "raw_metadata":
            metadata,

        "provenance": {
            "generation_dir":
                str(run_root),

            "metadata_path":
                str(
                    metadata_path
                ),

            "telemetry_path":
                str(
                    telemetry_path
                ),

            "narrative_path":
                str(
                    narrative_path
                ),

            "backend":
                definition[
                    "backend"
                ],

            "telemetry_mode":
                raw_telemetry.get(
                    "telemetry_mode"
                ),
        },
    }


def load_model_arm(
    arm_id,
    experiment_id=DEFAULT_EXPERIMENT_ID,
):
    if arm_id not in ARM_DEFINITIONS:
        raise KeyError(
            f"Unknown model arm: {arm_id}"
        )

    exp = get_experiment(
        experiment_id
    )

    if (
        arm_id
        not in exp["model_arms"]
    ):
        raise KeyError(
            f"Arm {arm_id!r} is not "
            f"registered for "
            f"{experiment_id!r}"
        )

    backend = (
        ARM_DEFINITIONS[
            arm_id
        ]["backend"]
    )

    if backend == "gpt_oss":
        return _load_gpt_arm(
            arm_id,
            experiment_id,
        )

    if backend == "native_fred":
        return (
            _load_native_fred_arm(
                arm_id,
                experiment_id,
            )
        )

    raise ValueError(
        f"Unsupported backend: "
        f"{backend}"
    )


def load_all_model_arms(
    experiment_id=DEFAULT_EXPERIMENT_ID,
):
    exp = get_experiment(
        experiment_id
    )

    return {
        arm_id: load_model_arm(
            arm_id,
            experiment_id,
        )
        for arm_id
        in exp["arm_order"]
    }
