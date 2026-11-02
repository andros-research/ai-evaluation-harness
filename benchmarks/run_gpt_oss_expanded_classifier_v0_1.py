#!/usr/bin/env python3

import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path.cwd()

MODEL_PATH = "/models/gpt-oss-20b"

MANIFEST_PATH = (
    ROOT
    / "benchmarks/fixtures/"
      "semantic_classifier_expanded_benchmark_v0_1.json"
)

MANIFEST_HASH_PATH = MANIFEST_PATH.with_suffix(".sha256")

OUT_ROOT = (
    ROOT
    / "benchmarks/results/"
      "semantic_classifier_expanded_v0_1/"
      "gpt_oss_20b"
)

WORKING_PATH = OUT_ROOT / "generation_working_v0_1.json"
FROZEN_PATH = OUT_ROOT / "generation_frozen_v0_1.json"
FROZEN_HASH_PATH = OUT_ROOT / "generation_frozen_v0_1.sha256"

EFFORTS = ["low", "medium", "high"]
MAX_NEW_TOKENS = 2048


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def extract_channels(tokenizer, generated_ids):
    ids = generated_ids.tolist()

    channel_id = tokenizer.convert_tokens_to_ids("<|channel|>")
    message_id = tokenizer.convert_tokens_to_ids("<|message|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end|>")
    return_id = tokenizer.convert_tokens_to_ids("<|return|>")

    channels = []
    i = 0

    while i < len(ids):
        if ids[i] != channel_id:
            i += 1
            continue

        try:
            message_pos = ids.index(message_id, i + 1)
        except ValueError:
            break

        channel_name = tokenizer.decode(
            ids[i + 1:message_pos],
            skip_special_tokens=False,
        ).strip()

        content_start = message_pos + 1
        endings = []

        try:
            endings.append(ids.index(end_id, content_start))
        except ValueError:
            pass

        try:
            endings.append(ids.index(return_id, content_start))
        except ValueError:
            pass

        content_end = min(endings) if endings else len(ids)
        content_ids = ids[content_start:content_end]

        channels.append({
            "channel": channel_name,
            "token_count": len(content_ids),
            "text": tokenizer.decode(
                content_ids,
                skip_special_tokens=True,
            ),
        })

        i = content_end + 1

    return channels


# ------------------------------------------------------------------
# Verify frozen benchmark manifest.
# ------------------------------------------------------------------

manifest_bytes = MANIFEST_PATH.read_bytes()
actual_manifest_hash = sha256_bytes(manifest_bytes)

expected_manifest_hash = (
    MANIFEST_HASH_PATH
    .read_text(encoding="utf-8")
    .strip()
    .split()[0]
)

if actual_manifest_hash != expected_manifest_hash:
    raise RuntimeError("Benchmark manifest hash mismatch.")

manifest = json.loads(manifest_bytes.decode("utf-8"))
targets = manifest["primary_cases"]

if len(targets) != 23:
    raise RuntimeError(
        f"Expected 23 primary cases, found {len(targets)}"
    )

print("=" * 80)
print("GPT-OSS-20B — EXPANDED SEMANTIC CLASSIFIER")
print("=" * 80)
print("cases:", len(targets))
print("efforts:", EFFORTS)
print("cells:", len(targets) * len(EFFORTS))
print("max_new_tokens:", MAX_NEW_TOKENS)
print("manifest SHA-256:", actual_manifest_hash)


# ------------------------------------------------------------------
# Load model once.
# ------------------------------------------------------------------

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    dtype="auto",
    device_map="auto",
)

model.eval()

runtime = {
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "accelerate": pkg_version("accelerate"),
    "kernels": pkg_version("kernels"),
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "compute_capability": list(
        torch.cuda.get_device_capability(0)
    ),
}

print(runtime)


# ------------------------------------------------------------------
# Warm-up.
# ------------------------------------------------------------------

warmup_messages = [{
    "role": "user",
    "content": 'Return JSON only: {"ok": true}',
}]

warmup_inputs = tokenizer.apply_chat_template(
    warmup_messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    reasoning_effort="low",
).to(model.device)

print("\nWarm-up...")

with torch.inference_mode():
    _ = model.generate(
        **warmup_inputs,
        max_new_tokens=32,
        do_sample=False,
    )

torch.cuda.synchronize()


# ------------------------------------------------------------------
# Resume state.
# ------------------------------------------------------------------

OUT_ROOT.mkdir(parents=True, exist_ok=True)

if WORKING_PATH.exists():
    working = json.loads(
        WORKING_PATH.read_text(encoding="utf-8")
    )

    if (
        working["benchmark_manifest_sha256"]
        != actual_manifest_hash
    ):
        raise RuntimeError(
            "Working artifact belongs to another manifest."
        )

    runs = working["runs"]
else:
    runs = []

completed = {
    (r["target_id"], r["reasoning_effort"])
    for r in runs
}


def write_working():
    artifact = {
        "schema_version":
            "gpt_oss_expanded_classifier_generation_v0_1",
        "status": "working",
        "benchmark_manifest_sha256":
            actual_manifest_hash,
        "model": "openai/gpt-oss-20b",
        "model_path": MODEL_PATH,
        "reasoning_efforts": EFFORTS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "runtime": runtime,
        "runs": runs,
    }

    WORKING_PATH.write_text(
        json.dumps(
            artifact,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )


# ------------------------------------------------------------------
# 23 × 3 generations.
# ------------------------------------------------------------------

total_cells = len(targets) * len(EFFORTS)
cell_index = 0

for target in targets:
    prompt = target["historical_prompt"]

    actual_prompt_hash = sha256_bytes(
        prompt.encode("utf-8")
    )

    if (
        actual_prompt_hash
        != target["historical_prompt_sha256"]
    ):
        raise RuntimeError(
            f"Prompt hash mismatch: {target['target_id']}"
        )

    for effort in EFFORTS:
        cell_index += 1
        key = (target["target_id"], effort)

        if key in completed:
            print(
                f"[{cell_index:02d}/{total_cells}] "
                f"SKIP {target['target_id']} / {effort}"
            )
            continue

        print()
        print("-" * 90)
        print(
            f"[{cell_index:02d}/{total_cells}] "
            f"{target['target_id']} / {effort}"
        )
        print("-" * 90)

        # Same exact historical classifier payload for every model.
        messages = [{
            "role": "user",
            "content": prompt,
        }]

        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=effort,
        )

        harmony_prompt_sha256 = sha256_bytes(
            rendered.encode("utf-8")
        )

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            reasoning_effort=effort,
        ).to(model.device)

        prompt_tokens = inputs["input_ids"].shape[-1]

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        generated_ids = outputs[0][prompt_tokens:]

        channels = extract_channels(
            tokenizer,
            generated_ids,
        )

        raw_generation = tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
        )

        analysis_text = "\n".join(
            c["text"]
            for c in channels
            if c["channel"] == "analysis"
        ).strip()

        final_text = "\n".join(
            c["text"]
            for c in channels
            if c["channel"] == "final"
        ).strip()

        analysis_tokens = sum(
            c["token_count"]
            for c in channels
            if c["channel"] == "analysis"
        )

        final_tokens = sum(
            c["token_count"]
            for c in channels
            if c["channel"] == "final"
        )

        reached_token_cap = (
            len(generated_ids) >= MAX_NEW_TOKENS
        )

        parse_status = "success"
        parsed_proposal = None
        validation_error = None

        try:
            parsed_proposal = json.loads(final_text)

            # Minimal structural checks here.
            if not isinstance(parsed_proposal, dict):
                raise ValueError(
                    "Final JSON must be an object."
                )

            if parsed_proposal.get("claim_span") != target["claim_span"]:
                raise ValueError(
                    "claim_span does not exactly match target."
                )

            annotations = parsed_proposal.get("annotations")

            if not isinstance(annotations, list):
                raise ValueError(
                    "annotations must be a list."
                )

        except Exception as exc:
            parse_status = "failure"
            validation_error = (
                f"{type(exc).__name__}: {exc}"
            )

        record = {
            "target_id": target["target_id"],
            "claim_span": target["claim_span"],
            "reasoning_effort": effort,

            "historical_prompt_sha256":
                actual_prompt_hash,

            "harmony_prompt_sha256":
                harmony_prompt_sha256,

            "prompt_tokens": prompt_tokens,
            "generated_tokens": len(generated_ids),

            "analysis_tokens": analysis_tokens,
            "final_tokens": final_tokens,

            "elapsed_seconds": elapsed,
            "tokens_per_second":
                len(generated_ids) / elapsed,

            "reached_token_cap": reached_token_cap,

            "parse_status": parse_status,
            "parsed_proposal": parsed_proposal,
            "validation_error": validation_error,

            "analysis_text": analysis_text,
            "final_text": final_text,
            "raw_generation": raw_generation,
        }

        runs.append(record)
        completed.add(key)
        write_working()

        print("prompt tokens:   ", prompt_tokens)
        print("generated tokens:", len(generated_ids))
        print("analysis tokens: ", analysis_tokens)
        print("final tokens:    ", final_tokens)
        print("seconds:         ", round(elapsed, 3))
        print(
            "tokens/sec:      ",
            round(len(generated_ids) / elapsed, 2),
        )
        print("parse:           ", parse_status)
        print("token cap:       ", reached_token_cap)

        if validation_error:
            print(
                "validation error:",
                validation_error,
            )

        if parsed_proposal is not None:
            print("annotations:")
            for ann in parsed_proposal.get(
                "annotations", []
            ):
                print(
                    " ",
                    (
                        ann.get("claim_kind"),
                        ann.get("support_status"),
                        ann.get("assertion_strength"),
                    ),
                )


# ------------------------------------------------------------------
# Freeze only when all 69 cells exist.
# ------------------------------------------------------------------

if len(runs) != total_cells:
    raise RuntimeError(
        f"Expected {total_cells} runs, found {len(runs)}."
    )

artifact = {
    "schema_version":
        "gpt_oss_expanded_classifier_generation_v0_1",

    "status":
        "generation_frozen_unscored",

    "benchmark_manifest_sha256":
        actual_manifest_hash,

    "model":
        "openai/gpt-oss-20b",

    "model_path":
        MODEL_PATH,

    "reasoning_efforts":
        EFFORTS,

    "max_new_tokens":
        MAX_NEW_TOKENS,

    "runtime":
        runtime,

    "runs":
        runs,
}

payload = (
    json.dumps(
        artifact,
        indent=2,
        ensure_ascii=False,
    ) + "\n"
).encode("utf-8")

FROZEN_PATH.write_bytes(payload)

digest = sha256_bytes(payload)

FROZEN_HASH_PATH.write_text(
    digest
    + "  "
    + FROZEN_PATH.name
    + "\n",
    encoding="utf-8",
)

print()
print("=" * 80)
print("GPT-OSS GENERATION FROZEN — UNSCORED")
print("=" * 80)

print("cells:", len(runs))

for effort in EFFORTS:
    subset = [
        r for r in runs
        if r["reasoning_effort"] == effort
    ]

    print()
    print(effort)
    print(
        "  parse successes:",
        sum(r["parse_status"] == "success" for r in subset),
    )
    print(
        "  parse failures:",
        sum(r["parse_status"] != "success" for r in subset),
    )
    print(
        "  token-cap hits:",
        sum(r["reached_token_cap"] for r in subset),
    )
    print(
        "  mean analysis tokens:",
        round(
            sum(r["analysis_tokens"] for r in subset)
            / len(subset),
            1,
        ),
    )

print("\nartifact:")
print(FROZEN_PATH)

print("\nSHA-256:")
print(digest)
