import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/models/gpt-oss-20b"
OUT_PATH = Path(
    "benchmarks/results/gpt_oss_reasoning_effort_probe_v0_2.json"
)

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

EFFORTS = ["low", "medium", "high"]


def extract_channels(tokenizer, generated_ids):
    """
    Read Harmony channel boundaries directly from the generated token IDs.
    Returns channel text plus the exact number of content tokens generated
    inside each channel.
    """
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
            ids[i + 1 : message_pos],
            skip_special_tokens=False,
        ).strip()

        content_start = message_pos + 1

        terminators = [
            p for p in (
                ids.index(end_id, content_start)
                if end_id in ids[content_start:]
                else None,
                ids.index(return_id, content_start)
                if return_id in ids[content_start:]
                else None,
            )
            if p is not None
        ]

        if not terminators:
            content_end = len(ids)
        else:
            content_end = min(terminators)

        content_ids = ids[content_start:content_end]

        channels.append(
            {
                "channel": channel_name,
                "token_count": len(content_ids),
                "text": tokenizer.decode(
                    content_ids,
                    skip_special_tokens=True,
                ),
            }
        )

        i = content_end + 1

    return channels


print("Loading tokenizer...")
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

results = []

for effort in EFFORTS:
    print(f"\n{'=' * 72}")
    print(f"REASONING EFFORT: {effort}")
    print("=" * 72)

    rendered = tokenizer.apply_chat_template(
        MESSAGES,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=effort,
    )

    inputs = tokenizer.apply_chat_template(
        MESSAGES,
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
            max_new_tokens=1536,
            do_sample=False,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated_ids = outputs[0][prompt_tokens:]
    raw = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
    )

    channels = extract_channels(tokenizer, generated_ids)

    record = {
        "reasoning_effort": effort,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": len(generated_ids),
        "elapsed_seconds": elapsed,
        "tokens_per_second": len(generated_ids) / elapsed,
        "rendered_prompt": rendered,
        "raw_generation": raw,
        "channels": channels,
    }

    results.append(record)

    print("\nRendered reasoning line:")
    for line in rendered.splitlines():
        if line.startswith("Reasoning:"):
            print(line)

    print(f"\nPrompt tokens:    {prompt_tokens}")
    print(f"Generated tokens: {len(generated_ids)}")
    print(f"Elapsed seconds:  {elapsed:.3f}")
    print(
        f"Tokens/sec:       "
        f"{len(generated_ids) / elapsed:.2f}"
    )

    for channel in channels:
        print(
            f"\n[{channel['channel'].upper()}] "
            f"{channel['token_count']} tokens"
        )
        print(channel["text"])

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8",
)

print(f"\nSaved: {OUT_PATH}")
