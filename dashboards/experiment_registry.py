from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EXPERIMENT_ID = (
    "v1_9_3_sep2026_multi_model_v0_1"
)


EXPERIMENTS = {
    DEFAULT_EXPERIMENT_ID: {
        "label": "September 2026",
        "observation_date": "2026-09-01",

        "fixture_root": (
            ROOT
            / "benchmarks/fixtures/"
            "macro_narrative_experiments/"
            "v1_9_3_sep2026_multi_model_v0_1"
        ),

        "gpt_result_root": (
            ROOT
            / "benchmarks/results/"
            "macro_narrative_experiments/"
            "v1_9_3_sep2026_multi_model_v0_1"
        ),

        "gpt_telemetry_root": (
            ROOT
            / "benchmarks/results/"
            "macro_token_telemetry_replay/"
            "v1_9_3_sep2026_multi_model_v0_1/"
            "replay_kernels_0_17_2_v0_1"
        ),

        "arm_order": [
            "gpt_oss_20b_medium",
            "llama3_8b_bf16",
            "gpt_oss_20b_low",
            "gpt_oss_20b_high",
        ],

        "model_arms": {
            "gpt_oss_20b_medium": {
                "human_rank": 1,
            },

            "llama3_8b_bf16": {
                "human_rank": 2,
                "run_root": (
                    ROOT
                    / "benchmarks/results/"
                    "native_fred_generation/"
                    "v1_9_3_sep2026_"
                    "llama3_8b_bf16_001"
                ),
            },

            "gpt_oss_20b_low": {
                "human_rank": 3,
            },

            "gpt_oss_20b_high": {
                "human_rank": None,
            },
        },

        "semantic_evaluation": {
            "strict_triple_recall": 0.600,
            "strict_triple_precision": 0.500,
            "claim_kind_recall": 0.771,
            "claim_kind_precision": 0.643,
            "protocol_valid": 12,
            "protocol_total": 12,

            "note": (
                "Scope generalized strongly, while "
                "support/assertion calibration and "
                "over-decomposition remain the main "
                "classifier weaknesses. Machine "
                "annotations remain proposals for "
                "human review."
            ),
        },
    },

    "v1_9_3_aug2026_backfill_multi_model_v0_1": {
        "label": "August 2026",
        "observation_date": "2026-08-01",

        "fixture_root": (
            ROOT
            / "benchmarks/fixtures/"
            "macro_narrative_experiments/"
            "v1_9_3_aug2026_backfill_multi_model_v0_1"
        ),

        "gpt_result_root": (
            ROOT
            / "benchmarks/results/"
            "macro_narrative_experiments/"
            "v1_9_3_aug2026_backfill_multi_model_v0_1"
        ),

        # August captured GPT-OSS telemetry natively during
        # generation. September instead uses the replay tree.
        "gpt_telemetry_layout": "generation_dir",

        "arm_order": [
            "gpt_oss_20b_low",
            "gpt_oss_20b_medium",
            "llama3_8b_bf16",
            "gpt_oss_20b_high",
        ],

        "model_arms": {
            "gpt_oss_20b_low": {
                "human_rank": 1,
            },

            "gpt_oss_20b_medium": {
                "human_rank": 2,
            },

            "llama3_8b_bf16": {
                "human_rank": 3,
                "run_root": (
                    ROOT
                    / "benchmarks/results/"
                    "native_fred_generation/"
                    "v1_9_3_aug2026_backfill_"
                    "llama3_8b_bf16_001"
                ),
            },

            "gpt_oss_20b_high": {
                "human_rank": None,
            },
        },

        "semantic_evaluation": {
            "strict_triple_recall": 0.556,
            "strict_triple_precision": 0.588,
            "claim_kind_recall": 0.722,
            "claim_kind_precision": 0.765,
            "protocol_valid": 11,
            "protocol_total": 11,

            "note": (
                "Second frozen OOS evaluation of "
                "compact_balanced_v0_2_1. Performance "
                "remained broadly consistent with September: "
                "strict recall 55.6%, strict precision 58.8%, "
                "and claim-kind recall 72.2%. Temporal-pattern "
                "recognition remained a notable weakness. "
                "Machine annotations remain proposals for "
                "human review."
            ),
        },
    },
}


def get_experiment(
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
):
    try:
        return EXPERIMENTS[
            experiment_id
        ]
    except KeyError as exc:
        raise KeyError(
            f"Unknown experiment: {experiment_id}"
        ) from exc


def list_experiments():
    return sorted(
        EXPERIMENTS,
        key=lambda experiment_id: (
            EXPERIMENTS[
                experiment_id
            ]["observation_date"]
        ),
        reverse=True,
    )
