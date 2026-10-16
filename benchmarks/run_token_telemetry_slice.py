#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.generate_fred_narrative_from_claims import (
    build_llm_prompt,
    load_selected_claims,
)


MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
SOURCE_MODEL = "llama3"

DEFAULT_CLAIM_ID = (
    "llama3_semantic_conservative_t0__commentary_02"
)

REVIEW_FIXTURE = (
    REPO_ROOT
    / "benchmarks/fixtures/"
    / "semantic_dashboard_reviewed_v0_1.jsonl"
)

def token_telemetry_output_path(
    claim_id: str,
) -> Path:
    return (
        REPO_ROOT
        / "benchmarks/results/token_telemetry/"
        / claim_id
        / "token_telemetry_v0_1.json"
    )


def load_reviewed_claims() -> list[dict]:
    rows = [
        json.loads(line)
        for line in REVIEW_FIXTURE.read_text().splitlines()
        if line.strip()
    ]

    if not rows:
        raise RuntimeError(
            "Reviewed semantic fixture is empty."
        )

    return rows


def load_reviewed_claim(
    claim_id: str,
) -> dict:
    matches = [
        row
        for row in load_reviewed_claims()
        if row["claim_id"] == claim_id
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one reviewed claim for "
            f"{claim_id!r}; found {len(matches)}"
        )

    return matches[0]


def repo_relative_source_path(
    absolute_source: str,
) -> Path:
    marker = "/ai-lab/"

    if marker not in absolute_source:
        raise ValueError(
            "Could not map historical source path "
            f"into current repo: {absolute_source}"
        )

    relative = absolute_source.split(
        marker,
        1,
    )[1]

    return REPO_ROOT / relative


def load_inference_runtime():
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        clean_up_tokenization_spaces=False,
    )

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
        low_cpu_mem_usage=True,
    )

    model.eval()

    return tokenizer, model


def score_reviewed_claim(
    *,
    reviewed: dict,
    tokenizer,
    model,
) -> Path:
    claim_id = reviewed["claim_id"]

    output_path = (
        token_telemetry_output_path(
            claim_id
        )
    )

    claim_span = reviewed["claim_span"]
    prompt_variant = reviewed["prompt_variant"]

    narrative_path = repo_relative_source_path(
        reviewed["source"]["narrative_path"]
    )

    run_root = narrative_path.parent.parent

    claims_path = (
        run_root
        / "fred_claims"
        / "selected_fred_claims.json"
    )

    selected_claims = load_selected_claims(
        claims_path
    )

    prompt = build_llm_prompt(
        selected_claims,
        prompt_variant,
    )

    narrative = narrative_path.read_text()

    claim_start = narrative.index(claim_span)
    claim_end = claim_start + len(claim_span)

    # Score the exact historical generation through
    # the end of the reviewed claim.
    generated_text = narrative[:claim_end]

    # Original user prompt + Llama-3 assistant header.
    chat_encoding = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    prompt_ids = chat_encoding["input_ids"]

    # Tokenize the generated narrative as one continuous
    # sequence so token boundaries match historical context.
    generated_encoding = tokenizer(
        generated_text,
        add_special_tokens=False,
        return_tensors="pt",
        return_offsets_mapping=True,
    )

    generated_ids = (
        generated_encoding["input_ids"]
    )

    offsets = (
        generated_encoding["offset_mapping"][0]
        .tolist()
    )

    target_generation_indices = [
        idx
        for idx, (start, end) in enumerate(offsets)
        if (
            end > claim_start
            and start < claim_end
        )
    ]

    if not target_generation_indices:
        raise RuntimeError(
            "No generated tokens overlap claim span."
        )

    full_ids = torch.cat(
        [
            prompt_ids,
            generated_ids,
        ],
        dim=1,
    ).to("cuda")

    print(
        "Running teacher-forced forward pass..."
    )

    with torch.inference_mode():
        outputs = model(
            input_ids=full_ids,
            use_cache=False,
        )

    prompt_token_count = prompt_ids.shape[1]

    token_records = []

    for claim_token_index, gen_idx in enumerate(
        target_generation_indices
    ):
        token_id = int(
            generated_ids[0, gen_idx].item()
        )

        # Position of this token in the complete
        # prompt + assistant sequence.
        full_position = (
            prompt_token_count
            + gen_idx
        )

        # Causal LM logits at position n-1 predict
        # the token actually observed at position n.
        logits = (
            outputs.logits[
                0,
                full_position - 1,
            ]
            .float()
        )

        log_probs = torch.log_softmax(
            logits,
            dim=-1,
        )

        probs = torch.exp(log_probs)

        probability = float(
            probs[token_id].item()
        )

        logprob = float(
            log_probs[token_id].item()
        )

        rank = (
            int(
                (
                    logits
                    > logits[token_id]
                )
                .sum()
                .item()
            )
            + 1
        )

        entropy = float(
            -(
                probs
                * log_probs
            )
            .sum()
            .item()
        )

        top_probs, top_ids = torch.topk(
            probs,
            k=10,
        )

        top_k = []

        for alt_rank, (
            alt_id,
            alt_prob,
        ) in enumerate(
            zip(
                top_ids.tolist(),
                top_probs.tolist(),
            ),
            start=1,
        ):
            top_k.append(
                {
                    "rank": alt_rank,
                    "token_id": int(alt_id),
                    "token_text":
                        tokenizer.decode(
                            [alt_id],
                            clean_up_tokenization_spaces=False,
                        ),
                    "probability":
                        float(alt_prob),
                }
            )

        char_start, char_end = offsets[
            gen_idx
        ]

        overlap_start = max(
            char_start,
            claim_start,
        )

        overlap_end = min(
            char_end,
            claim_end,
        )

        claim_text_fragment = narrative[
            overlap_start:overlap_end
        ]

        token_records.append(
            {
                "claim_token_index":
                    claim_token_index,
                "generation_token_index":
                    int(gen_idx),
                "full_sequence_position":
                    int(full_position),
                "context_token_count":
                    int(full_position),
                "token_id":
                    token_id,
                "token_text":
                    tokenizer.decode(
                        [token_id],
                        clean_up_tokenization_spaces=False,
                    ),
                "claim_text_fragment":
                    claim_text_fragment,
                "narrative_char_start":
                    int(char_start),
                "narrative_char_end":
                    int(char_end),
                "claim_char_start":
                    int(
                        overlap_start
                        - claim_start
                    ),
                "claim_char_end":
                    int(
                        overlap_end
                        - claim_start
                    ),
                "probability":
                    probability,
                "logprob":
                    logprob,
                "entropy_nats":
                    entropy,
                "rank":
                    rank,
                "top_k":
                    top_k,
            }
        )

    claim_text_from_tokens = "".join(
        record["claim_text_fragment"]
        for record in token_records
    )

    if claim_text_from_tokens != claim_span:
        raise RuntimeError(
            "Token-to-claim alignment failed: "
            f"{claim_text_from_tokens!r} "
            f"!= {claim_span!r}"
        )

    target_ids = [
        record["token_id"]
        for record in token_records
    ]

    target_decoded = tokenizer.decode(
        target_ids,
        clean_up_tokenization_spaces=False,
    )

    artifact = {
        "schema_version":
            "token_telemetry_v0_1",
        "claim_id":
            claim_id,
        "telemetry_mode":
            "teacher_forced_rescore",
        "historical_logits_recovered":
            False,
        "source_generation": {
            "model":
                reviewed["source_model"],
            "backend":
                "ollama",
            "quantization":
                "Q4_0",
            "temperature":
                reviewed[
                    "generation_temperature"
                ],
            "prompt_variant":
                prompt_variant,
            "narrative_path":
                str(
                    narrative_path.relative_to(
                        REPO_ROOT
                    )
                ),
        },
        "scoring": {
            "model":
                MODEL_ID,
            "backend":
                "transformers_pytorch",
            "dtype":
                "bfloat16",
            "device":
                "cuda",
            "vocab_size":
                len(tokenizer),
        },
        "context": {
            "raw_prompt_chars":
                len(prompt),
            "raw_prompt_sha256":
                hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
            "prompt_chat_tokens":
                int(prompt_token_count),
            "generated_tokens_through_claim":
                int(generated_ids.shape[1]),
            "total_tokens_through_claim":
                int(full_ids.shape[1]),
        },
        "claim": {
            "text":
                claim_span,
            "narrative_char_start":
                claim_start,
            "narrative_char_end":
                claim_end,
            "token_count":
                len(token_records),
            "decoded_token_slice":
                target_decoded,
            "aligned_claim_text":
                claim_text_from_tokens,
        },
        "human_review":
            reviewed["human_review"],
        "tokens":
            token_records,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            artifact,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    print()
    print("TOKEN TELEMETRY ARTIFACT")
    print("-" * 72)
    print("output:", output_path)
    print(
        "claim tokens:",
        len(token_records),
    )
    print(
        "decoded slice:",
        repr(target_decoded),
    )

    print()
    print(
        f"{'IDX':>3}  "
        f"{'TOKEN':<22} "
        f"{'P':>10} "
        f"{'LOGP':>10} "
        f"{'H':>9} "
        f"{'RANK':>5}"
    )
    print("-" * 72)

    for record in token_records:
        print(
            f"{record['claim_token_index']:>3}  "
            f"{record['token_text']!r:<22} "
            f"{record['probability']:>10.6f} "
            f"{record['logprob']:>10.4f} "
            f"{record['entropy_nats']:>9.4f} "
            f"{record['rank']:>5}"
        )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Teacher-forced token telemetry for "
            "human-reviewed Llama 3 claims."
        )
    )

    selection = (
        parser.add_mutually_exclusive_group()
    )

    selection.add_argument(
        "--claim-id",
        default=None,
        help=(
            "One reviewed Llama 3 claim_id to score. "
            "If omitted, uses the original v1.9.1 "
            "regression specimen."
        ),
    )

    selection.add_argument(
        "--all-reviewed",
        action="store_true",
        help=(
            "Score every reviewed claim whose "
            "source_model is llama3."
        ),
    )

    args = parser.parse_args()

    if args.all_reviewed:
        reviewed_claims = [
            row
            for row in load_reviewed_claims()
            if row.get("source_model")
            == SOURCE_MODEL
        ]
    else:
        claim_id = (
            args.claim_id
            or DEFAULT_CLAIM_ID
        )

        reviewed_claims = [
            load_reviewed_claim(
                claim_id
            )
        ]

    if not reviewed_claims:
        raise RuntimeError(
            "No reviewed Llama 3 claims selected."
        )

    wrong_model = [
        row["claim_id"]
        for row in reviewed_claims
        if row.get("source_model")
        != SOURCE_MODEL
    ]

    if wrong_model:
        raise RuntimeError(
            "This scorer only supports source_model "
            f"{SOURCE_MODEL!r}; incompatible claims: "
            + ", ".join(wrong_model)
        )

    print(
        f"Selected {len(reviewed_claims)} "
        "reviewed claim(s):"
    )

    for reviewed in reviewed_claims:
        print(
            " -",
            reviewed["claim_id"],
        )

    print()

    tokenizer, model = (
        load_inference_runtime()
    )

    total = len(reviewed_claims)

    for index, reviewed in enumerate(
        reviewed_claims,
        start=1,
    ):
        print()
        print(
            f"[{index}/{total}] "
            f"{reviewed['claim_id']}"
        )

        score_reviewed_claim(
            reviewed=reviewed,
            tokenizer=tokenizer,
            model=model,
        )


if __name__ == "__main__":
    main()
