#!/usr/bin/env python3
"""Native BF16 FRED generation with generation-time token telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RESULT_ROOT = (
    ROOT / "benchmarks/results/native_fred_generation"
)

OUTPUTS = (
    "fred_narrative.md",
    "native_generation_metadata.json",
    "native_generation_telemetry.json",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_and_validate(run_id: str):
    if (
        not run_id
        or run_id in (".", "..")
        or Path(run_id).name != run_id
    ):
        raise ValueError(f"Invalid run ID: {run_id!r}")

    run_dir = RESULT_ROOT / run_id

    plan = json.loads(
        (run_dir / "run_plan.json").read_text(
            encoding="utf-8"
        )
    )

    raw_prompt = (
        run_dir / "prompt.txt"
    ).read_bytes()

    if plan["run_id"] != run_id:
        raise RuntimeError("Run ID mismatch.")

    if sha256(raw_prompt) != plan["prompt"]["sha256"]:
        raise RuntimeError("Prompt hash mismatch.")

    settings = plan["inference"]

    if (
        settings["model_id"]
        != "meta-llama/Meta-Llama-3-8B-Instruct"
    ):
        raise RuntimeError(
            "This executor currently supports only Llama 3 8B."
        )

    if (
        settings["dtype"] != "bfloat16"
        or settings["do_sample"] is not False
        or settings["temperature"] != 0.0
    ):
        raise RuntimeError(
            "Expected BF16 greedy decoding."
        )

    if (
        not isinstance(settings["max_new_tokens"], int)
        or settings["max_new_tokens"] < 1
    ):
        raise RuntimeError("Invalid max_new_tokens.")

    for filename in OUTPUTS:
        if (run_dir / filename).exists():
            raise FileExistsError(
                f"Refusing to overwrite {run_dir / filename}"
            )

    return run_dir, plan, raw_prompt.decode("utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )


def generate(run_dir: Path, plan: dict, prompt: str) -> None:
    # Import GPU dependencies only after preflight.
    import torch
    import transformers
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. Use the NVIDIA container."
        )

    model_id = plan["inference"]["model_id"]

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        clean_up_tokenization_spaces=False,
    )

    encoding = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    prompt_ids = (
        encoding["input_ids"]
        if hasattr(encoding, "keys")
        else encoding
    ).to("cuda")

    print("Loading Llama 3 BF16...")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
        low_cpu_mem_usage=True,
    )

    model.eval()

    print("Generating and capturing raw logits...")

    with torch.inference_mode():
        result = model.generate(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            do_sample=False,
            max_new_tokens=plan["inference"]["max_new_tokens"],
            return_dict_in_generate=True,
            output_logits=True,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = result.sequences[
        0, prompt_ids.shape[1]:
    ].tolist()

    if (
        not new_ids
        or result.logits is None
        or len(result.logits) != len(new_ids)
    ):
        raise RuntimeError(
            "Generated tokens and captured logits differ."
        )

    records = []

    for index, (token_id, step) in enumerate(
        zip(new_ids, result.logits)
    ):
        logits = step[0].float()

        logprobs = torch.log_softmax(
            logits, dim=-1
        )

        probs = torch.exp(logprobs)

        top_probs, top_ids = torch.topk(
            probs, k=10
        )

        top_k = []

        for rank, (candidate_id, candidate_prob) in enumerate(
            zip(
                top_ids.tolist(),
                top_probs.tolist(),
            ),
            start=1,
        ):
            top_k.append({
                "rank": rank,
                "token_id": int(candidate_id),
                "token_text": tokenizer.decode(
                    [candidate_id],
                    clean_up_tokenization_spaces=False,
                ),
                "probability": float(candidate_prob),
            })

        records.append({
            "generation_token_index": index,
            "token_id": int(token_id),
            "token_text": tokenizer.decode(
                [token_id],
                clean_up_tokenization_spaces=False,
            ),
            "probability": float(
                probs[token_id].item()
            ),
            "logprob": float(
                logprobs[token_id].item()
            ),
            "entropy_nats": float(
                -(probs * logprobs).sum().item()
            ),
            "rank": (
                int(
                    (logits > logits[token_id])
                    .sum().item()
                ) + 1
            ),
            "top_k": top_k,
        })

    narrative = tokenizer.decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    metadata = {
        "schema_version":
            "native_fred_generation_metadata_v0_1",
        "run_id": plan["run_id"],
        "backend": "transformers_pytorch",
        "model_id": model_id,
        "dtype": "bfloat16",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "model_commit_hash": getattr(
            model.config, "_commit_hash", None
        ),
        "prompt_sha256": plan["prompt"]["sha256"],
        "prompt_chat_token_count": int(
            prompt_ids.shape[1]
        ),
        "prompt_chat_token_ids":
            prompt_ids[0].tolist(),
        "generated_token_ids": new_ids,
        "generated_token_count": len(new_ids),
        "narrative_sha256": sha256(
            narrative.encode("utf-8")
        ),
        "decoding": plan["inference"],
        "source_claims": plan["source"],
        "citation_validation": "not_run",
    }

    telemetry = {
        "schema_version":
            "native_generation_telemetry_v0_1",
        "run_id": plan["run_id"],
        "telemetry_mode":
            "generation_time_raw_logits",
        "full_raw_logit_vectors_persisted": False,
        "review_visibility":
            "research_only_not_for_blind_annotation",
        "tokens": records,
    }

    (run_dir / OUTPUTS[0]).write_text(
        narrative,
        encoding="utf-8",
    )

    write_json(
        run_dir / OUTPUTS[1],
        metadata,
    )

    write_json(
        run_dir / OUTPUTS[2],
        telemetry,
    )

    print()
    print("NATIVE GENERATION COMPLETE")
    print("Run:", plan["run_id"])
    print("Generated tokens:", len(new_ids))
    print("Narrative:", run_dir / OUTPUTS[0])
    print("Metadata:", run_dir / OUTPUTS[1])
    print("Telemetry:", run_dir / OUTPUTS[2])
    print("Semantic review performed: NO")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-id",
        default="llama3_bf16_smoke_001",
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
    )

    args = parser.parse_args()

    run_dir, plan, prompt = load_and_validate(
        args.run_id
    )

    print("PRE-FLIGHT OK:", args.run_id)
    print(
        "Prompt SHA-256:",
        plan["prompt"]["sha256"],
    )
    print(
        "Model:",
        plan["inference"]["model_id"],
    )
    print(
        "Max new tokens:",
        plan["inference"]["max_new_tokens"],
    )

    if args.preflight:
        print("GPU loading: SKIPPED")
        return

    generate(run_dir, plan, prompt)


if __name__ == "__main__":
    main()
