import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/models/gpt-oss-20b"
OUT_PATH = Path(
    "benchmarks/results/gpt_oss_greedy_repro_v0_1.json"
)

N_RUNS = 5
MAX_NEW_TOKENS = 256
REASONING_EFFORT = "low"

MESSAGES = [
    {
        "role": "developer",
        "content": (
            "You are a semantic analysis classifier. "
            "Analyze exactly the supplied macroeconomic statement. "
            "In the final answer, return JSON only with the keys "
            "'observation', 'transition', and 'interpretation'."
        ),
    },
    {
        "role": "user",
        "content": (
            "The unemployment rate fell, indicating a stronger labor market."
        ),
    },
]


def pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def first_difference(a, b):
    n = min(len(a), len(b))

    for i in range(n):
        if a[i] != b[i]:
            return i

    if len(a) != len(b):
        return n

    return None


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

# IMPORTANT:
# Render ONCE. The Harmony template contains the current date,
# so we freeze the actual rendered prompt before any repetitions.
rendered_prompt = tokenizer.apply_chat_template(
    MESSAGES,
    tokenize=False,
    add_generation_prompt=True,
    reasoning_effort=REASONING_EFFORT,
)

inputs = tokenizer(
    rendered_prompt,
    return_tensors="pt",
    add_special_tokens=False,
)

prompt_ids = inputs["input_ids"][0].tolist()
prompt_hash = hashlib.sha256(
    json.dumps(prompt_ids).encode("utf-8")
).hexdigest()

print("\n--- FROZEN PROMPT ---")
print(rendered_prompt)
print(f"Prompt tokens: {len(prompt_ids)}")
print(f"Prompt token SHA-256: {prompt_hash}")

print("\nLoading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    dtype="auto",
    device_map="auto",
)

model.eval()

inputs = {
    k: v.to(model.device)
    for k, v in inputs.items()
}

print("\nRuntime:")
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("accelerate:", pkg_version("accelerate"))
print("kernels:", pkg_version("kernels"))
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print(
    "deterministic algorithms:",
    torch.are_deterministic_algorithms_enabled(),
)

# One discarded warm-up run so run #1 does not pay all startup costs.
print("\nWarm-up generation...")
with torch.inference_mode():
    _ = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
    )

torch.cuda.synchronize()

records = []

for run_idx in range(N_RUNS):
    print(f"\n{'=' * 72}")
    print(f"RUN {run_idx + 1}/{N_RUNS}")
    print("=" * 72)

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        result = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated_ids = result.sequences[0][len(prompt_ids):]
    ids = generated_ids.tolist()

    token_metrics = []

    # Capture the local branching structure at every generated token.
    for step, score_tensor in enumerate(result.scores):
        logits = score_tensor[0].float()
        probs = torch.softmax(logits, dim=-1)

        top_probs, top_ids = torch.topk(probs, k=2)

        p1 = top_probs[0].item()
        p2 = top_probs[1].item()

        # Full-vocabulary entropy in nats.
        entropy = -(
            probs * torch.log(probs.clamp_min(1e-30))
        ).sum().item()

        chosen_id = ids[step]

        token_metrics.append(
            {
                "step": step,
                "chosen_id": chosen_id,
                "chosen_token": tokenizer.decode(
                    [chosen_id],
                    skip_special_tokens=False,
                ),
                "top1_id": top_ids[0].item(),
                "top1_token": tokenizer.decode(
                    [top_ids[0].item()],
                    skip_special_tokens=False,
                ),
                "top1_probability": p1,
                "top2_id": top_ids[1].item(),
                "top2_token": tokenizer.decode(
                    [top_ids[1].item()],
                    skip_special_tokens=False,
                ),
                "top2_probability": p2,
                "top1_top2_gap": p1 - p2,
                "entropy_nats": entropy,
            }
        )

    raw = tokenizer.decode(
        ids,
        skip_special_tokens=False,
    )

    record = {
        "run": run_idx + 1,
        "generated_token_count": len(ids),
        "elapsed_seconds": elapsed,
        "tokens_per_second": len(ids) / elapsed,
        "generated_ids": ids,
        "raw_generation": raw,
        "token_metrics": token_metrics,
    }

    records.append(record)

    print("generated tokens:", len(ids))
    print("elapsed seconds:", round(elapsed, 3))
    print(
        "tokens/sec:",
        round(len(ids) / elapsed, 2),
    )

    print("\nRAW GENERATION")
    print(raw)

    # Free the very large per-step score tensors before next run.
    del result
    torch.cuda.empty_cache()


reference = records[0]["generated_ids"]

print("\n" + "=" * 72)
print("REPRODUCIBILITY SUMMARY")
print("=" * 72)

all_identical = True

for record in records:
    ids = record["generated_ids"]
    diff = first_difference(reference, ids)

    if diff is None:
        print(
            f"Run {record['run']}: EXACT MATCH "
            f"({len(ids)} tokens)"
        )
    else:
        all_identical = False

        print(
            f"Run {record['run']}: FIRST DIFFERENCE "
            f"AT GENERATED TOKEN {diff}"
        )

        if diff < len(reference):
            print(
                "  reference:",
                reference[diff],
                repr(
                    tokenizer.decode(
                        [reference[diff]],
                        skip_special_tokens=False,
                    )
                ),
            )

        if diff < len(ids):
            print(
                "  this run:",
                ids[diff],
                repr(
                    tokenizer.decode(
                        [ids[diff]],
                        skip_special_tokens=False,
                    )
                ),
            )

print("\nAll five identical:", all_identical)


# Even if all runs match, identify the most plausible branch points
# in run 1: smallest top1-vs-top2 gaps and highest entropy.
ref_metrics = records[0]["token_metrics"]

smallest_gaps = sorted(
    ref_metrics,
    key=lambda x: x["top1_top2_gap"],
)[:10]

highest_entropy = sorted(
    ref_metrics,
    key=lambda x: x["entropy_nats"],
    reverse=True,
)[:10]


print("\n" + "=" * 72)
print("RUN 1 — TEN SMALLEST TOP1/TOP2 GAPS")
print("=" * 72)

for x in smallest_gaps:
    print(
        f"{x['step']:4d}  "
        f"{x['chosen_token']!r:18s}  "
        f"P1={x['top1_probability']:.6f}  "
        f"P2={x['top2_probability']:.6f}  "
        f"gap={x['top1_top2_gap']:.6f}  "
        f"H={x['entropy_nats']:.4f}"
    )


print("\n" + "=" * 72)
print("RUN 1 — TEN HIGHEST-ENTROPY POSITIONS")
print("=" * 72)

for x in highest_entropy:
    print(
        f"{x['step']:4d}  "
        f"{x['chosen_token']!r:18s}  "
        f"P1={x['top1_probability']:.6f}  "
        f"P2={x['top2_probability']:.6f}  "
        f"gap={x['top1_top2_gap']:.6f}  "
        f"H={x['entropy_nats']:.4f}"
    )


artifact = {
    "experiment": "gpt_oss_greedy_repro_v0_1",
    "model_path": MODEL_PATH,
    "reasoning_effort": REASONING_EFFORT,
    "n_runs": N_RUNS,
    "max_new_tokens": MAX_NEW_TOKENS,
    "rendered_prompt": rendered_prompt,
    "prompt_token_ids": prompt_ids,
    "prompt_token_sha256": prompt_hash,
    "runtime": {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": pkg_version("accelerate"),
        "kernels": pkg_version("kernels"),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(
            torch.cuda.get_device_capability(0)
        ),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
    },
    "all_identical": all_identical,
    "runs": records,
}


OUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_PATH.write_text(
    json.dumps(
        artifact,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print(f"\nSaved: {OUT_PATH}")
