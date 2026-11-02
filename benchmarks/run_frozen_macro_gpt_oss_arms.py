#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_ROOT = (
    ROOT
    / "benchmarks"
    / "fixtures"
    / "macro_narrative_experiments"
)

RESULT_ROOT = (
    ROOT
    / "benchmarks"
    / "results"
    / "macro_narrative_experiments"
)

MODEL_PATH = "/models/gpt-oss-20b"

OSS_ARM_IDS = (
    "gpt_oss_20b_low",
    "gpt_oss_20b_medium",
    "gpt_oss_20b_high",
)


# ------------------------------------------------------------------
# GB10 / PyTorch FlexAttention workaround already validated during
# v0.2 classifier development.
# ------------------------------------------------------------------

FLEX_KERNEL_OPTIONS = {
    "FORCE_USE_FLEX_ATTENTION": True,
}

import transformers.integrations.flex_attention as _flex_mod

_original_compile_friendly_flex_attention = (
    _flex_mod.compile_friendly_flex_attention
)


def _patched_compile_friendly_flex_attention(
    query,
    key,
    value,
    training=False,
    **kwargs,
):
    opts = dict(
        kwargs.get("kernel_options")
        or {}
    )

    opts.update(
        FLEX_KERNEL_OPTIONS
    )

    kwargs["kernel_options"] = opts

    return (
        _original_compile_friendly_flex_attention(
            query,
            key,
            value,
            training=training,
            **kwargs,
        )
    )


_flex_mod.compile_friendly_flex_attention = (
    _patched_compile_friendly_flex_attention
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pkg_version(name: str):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def write_json(
    path: Path,
    obj: dict,
) -> None:
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def extract_channels(
    tokenizer,
    generated_ids,
):
    ids = generated_ids.tolist()

    channel_id = (
        tokenizer.convert_tokens_to_ids(
            "<|channel|>"
        )
    )

    message_id = (
        tokenizer.convert_tokens_to_ids(
            "<|message|>"
        )
    )

    end_id = (
        tokenizer.convert_tokens_to_ids(
            "<|end|>"
        )
    )

    return_id = (
        tokenizer.convert_tokens_to_ids(
            "<|return|>"
        )
    )

    channels = []
    i = 0

    while i < len(ids):

        if ids[i] != channel_id:
            i += 1
            continue

        try:
            message_pos = ids.index(
                message_id,
                i + 1,
            )
        except ValueError:
            break

        channel_name = tokenizer.decode(
            ids[i + 1:message_pos],
            skip_special_tokens=False,
        ).strip()

        content_start = message_pos + 1

        endings = []

        for terminator in (
            end_id,
            return_id,
        ):
            try:
                endings.append(
                    ids.index(
                        terminator,
                        content_start,
                    )
                )
            except ValueError:
                pass

        content_end = (
            min(endings)
            if endings
            else len(ids)
        )

        content_ids = ids[
            content_start:content_end
        ]

        channels.append({
            "channel":
                channel_name,

            "token_count":
                len(content_ids),

            "text":
                tokenizer.decode(
                    content_ids,
                    skip_special_tokens=True,
                ),
        })

        i = content_end + 1

    return channels


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment-id",
        required=True,
    )

    args = parser.parse_args()

    experiment_dir = (
        EXPERIMENT_ROOT
        / args.experiment_id
    )

    manifest_path = (
        experiment_dir
        / "experiment_manifest.json"
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    prompt_path = (
        experiment_dir
        / manifest["prompt"]["file"]
    )

    raw_prompt_bytes = (
        prompt_path.read_bytes()
    )

    raw_prompt = (
        raw_prompt_bytes.decode("utf-8")
    )

    actual_prompt_sha = (
        sha256_bytes(raw_prompt_bytes)
    )

    expected_prompt_sha = (
        manifest["prompt"]["sha256"]
    )

    if (
        actual_prompt_sha
        != expected_prompt_sha
    ):
        raise RuntimeError(
            "Frozen prompt hash mismatch."
        )

    arms_by_id = {
        arm["arm_id"]: arm
        for arm
        in manifest["generation_arms"]
    }

    arms = [
        arms_by_id[arm_id]
        for arm_id
        in OSS_ARM_IDS
    ]

    for arm in arms:

        if (
            arm["model"]
            != "openai/gpt-oss-20b"
        ):
            raise RuntimeError(
                "Unexpected model in OSS arm."
            )

        if (
            arm["reasoning_effort"]
            not in (
                "low",
                "medium",
                "high",
            )
        ):
            raise RuntimeError(
                "Invalid reasoning effort."
            )

        if (
            arm["do_sample"]
            is not False
        ):
            raise RuntimeError(
                "Expected greedy decoding."
            )

        if (
            arm["max_new_tokens"]
            != 4096
        ):
            raise RuntimeError(
                "Expected 4096-token budget."
            )

    result_base = (
        RESULT_ROOT
        / args.experiment_id
    )

    result_base.mkdir(
        parents=True,
        exist_ok=True,
    )

    for arm in arms:

        arm_dir = (
            result_base
            / arm["arm_id"]
            / "generation_001"
        )

        if arm_dir.exists():
            raise FileExistsError(
                f"Refusing to overwrite: "
                f"{arm_dir}"
            )

    print("=" * 80)
    print(
        "GPT-OSS FROZEN MACRO "
        "NARRATIVE EXPERIMENT"
    )
    print("=" * 80)

    print(
        "Experiment:",
        args.experiment_id,
    )

    print(
        "Raw prompt SHA-256:",
        actual_prompt_sha,
    )

    print(
        "Raw prompt characters:",
        len(raw_prompt),
    )

    print(
        "Arms:",
        ", ".join(
            a["arm_id"]
            for a in arms
        ),
    )

    print("\nLoading tokenizer...")

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )
    )

    print("Loading GPT-OSS 20B...")

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            dtype="auto",
            device_map="auto",
            attn_implementation=
                "flex_attention",
        )
    )

    model.eval()

    runtime = {
        "torch":
            torch.__version__,

        "transformers":
            transformers.__version__,

        "accelerate":
            pkg_version("accelerate"),

        "kernels":
            pkg_version("kernels"),

        "cuda":
            torch.version.cuda,

        "gpu":
            torch.cuda.get_device_name(0),

        "compute_capability":
            list(
                torch.cuda
                .get_device_capability(0)
            ),

        "attention_implementation":
            getattr(
                model.config,
                "_attn_implementation",
                None,
            ),

        "flex_kernel_options":
            FLEX_KERNEL_OPTIONS,
    }

    print(runtime)

    # --------------------------------------------------------------
    # Warmup using the same workaround, but not the research prompt.
    # --------------------------------------------------------------

    warmup = (
        tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content":
                    "Reply with the word OK.",
            }],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            reasoning_effort="low",
        )
        .to(model.device)
    )

    print("\nWarm-up...")

    with torch.inference_mode():
        _ = model.generate(
            **warmup,
            max_new_tokens=32,
            do_sample=False,
        )

    torch.cuda.synchronize()

    # --------------------------------------------------------------
    # One frozen greedy generation per reasoning level.
    # --------------------------------------------------------------

    for arm in arms:

        arm_id = arm["arm_id"]

        effort = (
            arm["reasoning_effort"]
        )

        arm_dir = (
            result_base
            / arm_id
            / "generation_001"
        )

        arm_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        messages = [{
            "role": "user",
            "content": raw_prompt,
        }]

        rendered = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort=effort,
            )
        )

        inputs = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                reasoning_effort=effort,
            )
            .to(model.device)
        )

        prompt_tokens = int(
            inputs["input_ids"]
            .shape[-1]
        )

        rendered_sha = (
            sha256_bytes(
                rendered.encode("utf-8")
            )
        )

        print()
        print("-" * 80)
        print(
            f"{arm_id} "
            f"(reasoning={effort})"
        )
        print("-" * 80)

        print(
            "Prompt tokens:",
            prompt_tokens,
        )

        print(
            "Rendered Harmony SHA-256:",
            rendered_sha,
        )

        torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.inference_mode():

            output = model.generate(
                **inputs,
                max_new_tokens=
                    arm["max_new_tokens"],
                do_sample=False,
            )

        torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )

        generated_ids = output[
            0,
            prompt_tokens:
        ]

        channels = extract_channels(
            tokenizer,
            generated_ids,
        )

        analysis_text = "\n".join(
            c["text"]
            for c in channels
            if c["channel"]
            == "analysis"
        ).strip()

        final_text = "\n".join(
            c["text"]
            for c in channels
            if c["channel"]
            == "final"
        ).strip()

        analysis_tokens = sum(
            c["token_count"]
            for c in channels
            if c["channel"]
            == "analysis"
        )

        final_tokens = sum(
            c["token_count"]
            for c in channels
            if c["channel"]
            == "final"
        )

        generated_tokens = len(
            generated_ids
        )

        reached_cap = (
            generated_tokens
            >= arm["max_new_tokens"]
        )

        # Freeze prompt bytes alongside the run.
        (
            arm_dir
            / "narrative_prompt.txt"
        ).write_bytes(
            raw_prompt_bytes
        )

        (
            arm_dir
            / "analysis.txt"
        ).write_text(
            analysis_text,
            encoding="utf-8",
        )

        (
            arm_dir
            / "fred_narrative.md"
        ).write_text(
            final_text,
            encoding="utf-8",
        )

        metadata = {
            "schema_version":
                "gpt_oss_macro_generation_v0_1",

            "experiment_id":
                args.experiment_id,

            "arm_id":
                arm_id,

            "generation_index":
                1,

            "model":
                arm["model"],

            "reasoning_effort":
                effort,

            "raw_prompt_sha256":
                actual_prompt_sha,

            "rendered_harmony_sha256":
                rendered_sha,

            "prompt_chat_token_count":
                prompt_tokens,

            "generated_token_count":
                generated_tokens,

            "analysis_token_count":
                analysis_tokens,

            "final_token_count":
                final_tokens,

            "reached_token_cap":
                reached_cap,

            "elapsed_seconds":
                elapsed,

            "tokens_per_second":
                (
                    generated_tokens
                    / elapsed
                ),

            "analysis_sha256":
                sha256_bytes(
                    analysis_text
                    .encode("utf-8")
                ),

            "narrative_sha256":
                sha256_bytes(
                    final_text
                    .encode("utf-8")
                ),

            "runtime":
                runtime,

            "decoding": {
                "do_sample":
                    False,

                "max_new_tokens":
                    arm[
                        "max_new_tokens"
                    ],
            },

            "review_policy": {
                "analysis_hidden_from_initial_review":
                    True,

                "telemetry_reviewed_before_human_annotation":
                    False,
            },
        }

        write_json(
            arm_dir
            / "generation_metadata.json",
            metadata,
        )

        print(
            "Generated tokens:",
            generated_tokens,
        )

        print(
            "Analysis tokens:",
            analysis_tokens,
        )

        print(
            "Final tokens:",
            final_tokens,
        )

        print(
            "Seconds:",
            round(elapsed, 3),
        )

        print(
            "Token cap:",
            reached_cap,
        )

        print(
            "Narrative SHA-256:",
            metadata[
                "narrative_sha256"
            ],
        )

        if not final_text:
            print(
                "WARNING: no final channel "
                "was produced."
            )

    print()
    print("=" * 80)
    print(
        "ALL THREE GPT-OSS "
        "GENERATIONS FROZEN"
    )
    print("=" * 80)

    print(
        "Human review performed: NO"
    )

    print(
        "Analysis channels inspected: NO"
    )


if __name__ == "__main__":
    main()
