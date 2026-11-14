#!/usr/bin/env python3
"""Frozen out-of-sample semantic proposals for the September v1.9.3 macro lab.

Commands:
  prepare: CPU/host; project 12 unlabeled targets from the frozen human review.
  generate: NVIDIA container; run GPT-OSS low on exactly those prepared targets.

No human gold annotations or model-identity mapping enter the target prompts.
"""
from __future__ import annotations

import argparse
import os
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"
sys.path.insert(0, str(BENCH))

EXPERIMENT = os.environ.get(
    "MACRO_OOS_EXPERIMENT_ID",
    "v1_9_3_sep2026_multi_model_v0_1",
)
EXP_DIR = BENCH / "fixtures/macro_narrative_experiments" / EXPERIMENT
RESULT_DIR = BENCH / "results/macro_semantic_oos" / EXPERIMENT / "gpt_oss_20b_low_v0_2_1"
CONFIG = BENCH / "configs/semantic_classifier_compact_v0_2_1.json"
REVIEWED = BENCH / "fixtures/semantic_dashboard_reviewed_v0_1.jsonl"
FROZEN_REFERENCE = EXP_DIR / "human_review_v0_1/human_macro_semantic_reference_v0_1.json"
REFERENCE_MANIFEST = EXP_DIR / "human_review_v0_1/human_reference_freeze_manifest_v0_1.json"
SELECTED_CLAIMS = EXP_DIR / "inputs/selected_fred_claims.json"
EXPERIMENT_MANIFEST = EXP_DIR / "experiment_manifest.json"

INPUT_FILE = RESULT_DIR / "evaluation_input_v0_1.json"
DEVELOPER_FILE = RESULT_DIR / "developer_prompt_v0_1.txt"
WORKING_FILE = RESULT_DIR / "generation_working_v0_1.json"
FROZEN_FILE = RESULT_DIR / "generation_frozen_v0_1.json"
FROZEN_SHA_FILE = RESULT_DIR / "generation_frozen_v0_1.sha256"

MODEL_PATH = "/models/gpt-oss-20b"
MAX_NEW_TOKENS = 2048
EXPECTED_REFERENCE_SHA = json.loads(
    REFERENCE_MANIFEST.read_text(
        encoding="utf-8"
    )
)["reference_sha256"]
EXPECTED_PREVIOUS_DEVELOPER_SHA = "36ce7c8a4977104b316bb23ca09a92627be5bbcb1a96c73846d2aecaee289ce7"

OUTPUT_CONTRACT = """
Return JSON only using exactly this shape:

{
  "claim_span": "<exact supplied claim span>",
  "annotations": [
    {
      "claim_kind": "<claim kind>",
      "support_status": "supported | contradicted | insufficient_evidence",
      "assertion_strength": "asserted | qualified_inference | qualified_possibility",
      "rationale": "<brief evidence-relative rationale>"
    }
  ]
}

The claim_span must exactly reproduce the supplied target text.
Do not add prose outside the JSON object.
""".strip()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_bytes(obj) -> bytes:
    return (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def write_atomic(path: Path, raw: bytes) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(raw)
    temp.replace(path)


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def build_unchanged_developer_prompt(cfg: dict) -> str:
    """Mirror the already-run v0.2.1 developer construction byte-for-byte."""
    from build_semantic_proposer_prompt import compact_evidence

    guide = (ROOT / cfg["guide_path"]).read_text(encoding="utf-8")
    rows = [json.loads(line) for line in REVIEWED.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {row["claim_id"]: row for row in rows}
    examples = []
    for claim_id in cfg["few_shot_example_ids"]:
        row = by_id[claim_id]
        review = row["human_review"]
        if review["review_status"] not in ("accepted", "corrected"):
            raise RuntimeError(f"Unreviewed demonstration: {claim_id}")
        examples.append({
            "claim_span": row["claim_span"],
            "evidence": compact_evidence(row),
            "reviewed_annotations": review["reviewed_annotations"],
        })

    return "\n\n".join([
        "# Semantic classifier instructions",
        guide.strip(),
        "# Reviewed examples",
        json.dumps(examples, indent=2, ensure_ascii=False),
        "# Output contract",
        OUTPUT_CONTRACT,
    ])


def make_frozen_evidence(claims: list[dict]) -> dict:
    """Use the already frozen FRED selected claims, not hand-entered numbers."""
    if len(claims) != 4:
        raise RuntimeError(f"Expected four selected FRED claims, got {len(claims)}")
    return {
        "source_type": "fred",
        "comparison_window": "12m",
        "evidence_policy": "Frozen selected FRED claims; only supplied evidence counts.",
        "selected_claims": [
            {key: claim[key] for key in (
                "claim_id", "claim_text", "source_series", "metric_name",
                "comparison_window", "current_value", "prior_value",
                "delta_value", "direction", "supporting_values"
            )} for claim in claims
        ],
    }


def prepare() -> None:
    if RESULT_DIR.exists():
        raise FileExistsError(f"Refusing to replace an existing evaluation: {RESULT_DIR}")

    ref_bytes = FROZEN_REFERENCE.read_bytes()
    ref_sha = sha(ref_bytes)
    if ref_sha != EXPECTED_REFERENCE_SHA:
        raise RuntimeError(f"Reference SHA mismatch: {ref_sha}")
    ref = json.loads(ref_bytes)
    manifest = load_json(REFERENCE_MANIFEST)
    if manifest["status"] != "source_verified_frozen" or manifest["reference_sha256"] != ref_sha:
        raise RuntimeError("Human reference freeze manifest mismatch")
    if ref["experiment_id"] != EXPERIMENT:
        raise RuntimeError("Wrong reference experiment")

    experiment = load_json(EXPERIMENT_MANIFEST)
    claim_bytes = SELECTED_CLAIMS.read_bytes()
    claims_sha = sha(claim_bytes)
    if claims_sha != experiment["input"]["claims_sha256"]:
        raise RuntimeError("Selected-claims SHA differs from frozen experiment")
    if claims_sha != ref["frozen_source"]["selected_fred_claims_sha256"]:
        raise RuntimeError("Selected claims differ from human review evidence")
    raw_prompt_sha = sha((EXP_DIR / experiment["prompt"]["file"]).read_bytes())
    if raw_prompt_sha != ref["frozen_source"]["narrative_prompt_sha256"]:
        raise RuntimeError("Narrative prompt differs from reference")

    cfg_bytes = CONFIG.read_bytes()
    cfg = json.loads(cfg_bytes)
    if cfg["config_id"] != "compact_balanced_v0_2_1" or cfg["reasoning_effort"] != "low":
        raise RuntimeError("Unexpected classifier config")
    guide_path = ROOT / cfg["guide_path"]
    developer = build_unchanged_developer_prompt(cfg)
    developer_sha = sha(developer.encode("utf-8"))
    if developer_sha != EXPECTED_PREVIOUS_DEVELOPER_SHA:
        raise RuntimeError(
            "v0.2.1 developer prompt is not byte-identical to frozen regression. "
            f"Actual={developer_sha}; expected={EXPECTED_PREVIOUS_DEVELOPER_SHA}. "
            "Stop and inspect; don't silently tune or weaken this guard."
        )

    evidence = make_frozen_evidence(json.loads(claim_bytes))
    targets = []
    for narrative in ref["narratives"]:
        label = narrative["blind_label"]
        for sentence in narrative["commentary_sentences"]:
            # Deliberately access ONLY sentence_id and claim_span, never gold annotations.
            claim_span = sentence["claim_span"]
            user_content = "\n".join([
                "# Target",
                json.dumps({"claim_span": claim_span, "evidence": evidence}, indent=2, ensure_ascii=False),
            ])
            targets.append({
                "target_id": sentence["sentence_id"],
                "blind_label": label,
                "claim_span": claim_span,
                "user_content": user_content,
                "user_sha256": sha(user_content.encode("utf-8")),
            })

    if (
        not targets
        or len({
            t["target_id"]
            for t in targets
        }) != len(targets)
    ):
        raise RuntimeError(
            "Expected nonempty unique commentary targets"
        )
    if {t["blind_label"] for t in targets} != {"NARRATIVE_A", "NARRATIVE_B", "NARRATIVE_C"}:
        raise RuntimeError("Unexpected narrative labels")
    # No human gold labels are persisted in the model-facing input.
    plan = {
        "schema_version": "macro_semantic_oos_input_v0_1",
        "experiment_id": EXPERIMENT,
        "model": "openai/gpt-oss-20b",
        "classifier_config_id": cfg["config_id"],
        "reasoning_effort": "low",
        "max_new_tokens": MAX_NEW_TOKENS,
        "inputs": {
            "human_reference_sha256": ref_sha,
            "source_claims_sha256": claims_sha,
            "source_narrative_prompt_sha256": raw_prompt_sha,
            "classifier_config_sha256": sha(cfg_bytes),
            "classifier_guide_sha256": sha(guide_path.read_bytes()),
            "reviewed_examples_fixture_sha256": sha(REVIEWED.read_bytes()),
            "developer_sha256": developer_sha,
        },
        "targets": targets,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=False)
    INPUT_FILE.write_bytes(dump_bytes(plan))
    DEVELOPER_FILE.write_bytes(developer.encode("utf-8"))
    print("OOS EVALUATION INPUT FROZEN — UNLABELED")
    print("Targets:", len(targets), "| human gold annotations excluded: YES")
    print("Developer SHA-256:", developer_sha)
    print("Evidence claims SHA-256:", claims_sha)
    print("Reference SHA-256:", ref_sha)
    print("Prepared input SHA-256:", sha(INPUT_FILE.read_bytes()))
    print("Output:", RESULT_DIR)
    print("GPU used: NO")


def extract_channels(tokenizer, generated_ids):
    """Same token-delimited Harmony extraction used in the v0.2 runner."""
    ids = generated_ids.tolist()
    channel_id = tokenizer.convert_tokens_to_ids("<|channel|>")
    message_id = tokenizer.convert_tokens_to_ids("<|message|>")
    terminators = [tokenizer.convert_tokens_to_ids(x) for x in ("<|end|>", "<|return|>")]
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
        name = tokenizer.decode(ids[i+1:message_pos], skip_special_tokens=False).strip()
        start = message_pos + 1
        endings = []
        for end in terminators:
            try:
                endings.append(ids.index(end, start))
            except ValueError:
                pass
        stop = min(endings) if endings else len(ids)
        contents = ids[start:stop]
        channels.append({
            "channel": name,
            "token_count": len(contents),
            "text": tokenizer.decode(contents, skip_special_tokens=True),
        })
        i = stop + 1
    return channels


def generate() -> None:
    # Imports intentionally deferred: 'prepare' runs on the host without PyTorch/CUDA.
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from run_semantic_classifier import validate_proposal
    import transformers.integrations.flex_attention as flex_module

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: run generate inside the NVIDIA container")
    input_raw = INPUT_FILE.read_bytes()
    plan = json.loads(input_raw)
    developer = DEVELOPER_FILE.read_text(encoding="utf-8")
    if sha(developer.encode("utf-8")) != plan["inputs"]["developer_sha256"]:
        raise RuntimeError("Prepared developer prompt hash mismatch")
    if (
        plan["classifier_config_id"]
        != "compact_balanced_v0_2_1"
        or not plan["targets"]
    ):
        raise RuntimeError(
            "Invalid prepared evaluation input"
        )
    if sha(FROZEN_REFERENCE.read_bytes()) != plan["inputs"]["human_reference_sha256"]:
        raise RuntimeError("Frozen reference changed since preparation")
    if sha(SELECTED_CLAIMS.read_bytes()) != plan["inputs"]["source_claims_sha256"]:
        raise RuntimeError("Frozen claims changed since preparation")
    if FROZEN_FILE.exists():
        raise FileExistsError(f"Already frozen: {FROZEN_FILE}")

    # Known-working GPT-OSS/GB10 workaround from the frozen v0.2.1 regression.
    flex_kernel_options = {"FORCE_USE_FLEX_ATTENTION": True}
    original_flex = flex_module.compile_friendly_flex_attention
    def patched_flex(query, key, value, training=False, **kwargs):
        opts = dict(kwargs.get("kernel_options") or {})
        opts.update(flex_kernel_options)
        kwargs["kernel_options"] = opts
        return original_flex(query, key, value, training=training, **kwargs)
    flex_module.compile_friendly_flex_attention = patched_flex

    print(
        "GPT-OSS v0.2.1 OUT-OF-SAMPLE — "
        f"{len(plan['targets'])} FROZEN COMMENTARY SENTENCES"
    )
    print("Developer prompt hash:", plan["inputs"]["developer_sha256"])
    print("Input hash:", sha(input_raw))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, dtype="auto", device_map="auto",
        attn_implementation="flex_attention",
    )
    model.eval()
    runtime = {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "accelerate": package_version("accelerate"), "kernels": package_version("kernels"),
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "flex_kernel_options": flex_kernel_options,
    }
    print("Runtime:", runtime)

    warmup = tokenizer.apply_chat_template(
        [{"role": "user", "content": 'Return JSON only: {"ok": true}'}],
        tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True, reasoning_effort="low",
    ).to(model.device)
    with torch.inference_mode():
        model.generate(**warmup, max_new_tokens=32, do_sample=False)
    torch.cuda.synchronize()

    runs = []
    if WORKING_FILE.exists():
        working = load_json(WORKING_FILE)
        if working["prepared_input_sha256"] != sha(input_raw):
            raise RuntimeError("Existing work belongs to a different frozen input")
        if working["runtime"] != runtime:
            raise RuntimeError("Runtime differs from existing work; do not mix runs")
        runs = working["runs"]
    completed = {r["target_id"] for r in runs}
    expected_ids = [t["target_id"] for t in plan["targets"]]
    if len(completed) != len(runs) or not completed.issubset(set(expected_ids)):
        raise RuntimeError("Corrupt/duplicate existing working run IDs")

    def write_work():
        write_atomic(WORKING_FILE, dump_bytes({
            "schema_version": "macro_semantic_oos_generation_v0_1",
            "status": "working_unscored",
            "experiment_id": EXPERIMENT,
            "model": "openai/gpt-oss-20b",
            "classifier_config_id": plan["classifier_config_id"],
            "reasoning_effort": "low",
            "prepared_input_sha256": sha(input_raw),
            "runtime": runtime,
            "runs": runs,
        }))

    for index, target in enumerate(plan["targets"], 1):
        tid = target["target_id"]
        if tid in completed:
            print(
                f"[{index:02d}/{len(plan['targets']):02d}] "
                f"SKIP {tid}"
            )
            continue
        if sha(target["user_content"].encode("utf-8")) != target["user_sha256"]:
            raise RuntimeError(f"Target user-content hash mismatch: {tid}")
        messages = [
            {"role": "developer", "content": developer},
            {"role": "user", "content": target["user_content"]},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, reasoning_effort="low",
        )
        tensors = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True, reasoning_effort="low",
        ).to(model.device)
        count = tensors["input_ids"].shape[-1]
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**tensors, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        generated_ids = output[0][count:]
        channels = extract_channels(tokenizer, generated_ids)
        analysis = "\n".join(c["text"] for c in channels if c["channel"] == "analysis").strip()
        final = "\n".join(c["text"] for c in channels if c["channel"] == "final").strip()
        proposed = None
        error = None
        try:
            parsed = json.loads(final)
            anns = validate_proposal(proposal=parsed, claim_span=target["claim_span"])
            proposed = {"claim_span": parsed["claim_span"], "annotations": anns}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        record = {
            "target_id": tid, "blind_label": target["blind_label"],
            "claim_span": target["claim_span"],
            "raw_user_sha256": target["user_sha256"],
            "harmony_prompt_sha256": sha(rendered.encode("utf-8")),
            "prompt_tokens": int(count), "generated_tokens": len(generated_ids),
            "analysis_tokens": sum(c["token_count"] for c in channels if c["channel"] == "analysis"),
            "final_tokens": sum(c["token_count"] for c in channels if c["channel"] == "final"),
            "elapsed_seconds": seconds,
            "reached_token_cap": len(generated_ids) >= MAX_NEW_TOKENS,
            "parse_status": "success" if proposed is not None else "failure",
            "validation_error": error, "validated_proposal": proposed,
            # Save the model's local analysis for future research, but do not print it.
            "analysis_text": analysis, "final_text": final,
        }
        runs.append(record)
        completed.add(tid)
        write_work()
        print(
            f"[{index:02d}/{len(plan['targets']):02d}] "
            f"{tid}: parse={record['parse_status']}, "
            f"generated={len(generated_ids)}, "
            f"analysis={record['analysis_tokens']}, "
            f"cap={record['reached_token_cap']}, "
            f"seconds={seconds:.1f}",
            flush=True,
        )

    if [r["target_id"] for r in runs] != expected_ids:
        raise RuntimeError("Runs incomplete/out of frozen target order; not freezing")
    frozen = {
        "schema_version": "macro_semantic_oos_generation_v0_1",
        "status": "generation_frozen_unscored",
        "experiment_id": EXPERIMENT,
        "model": "openai/gpt-oss-20b",
        "classifier_config_id": plan["classifier_config_id"],
        "reasoning_effort": "low",
        "prepared_input_sha256": sha(input_raw),
        "runtime": runtime,
        "runs": runs,
    }
    raw = dump_bytes(frozen)
    write_atomic(FROZEN_FILE, raw)
    FROZEN_SHA_FILE.write_text(f"{sha(raw)}  {FROZEN_FILE.name}\n", encoding="utf-8")
    print("OOS GENERATION FROZEN — UNSCORED")
    print("Cases:", len(runs), "| parsed:", sum(r["parse_status"] == "success" for r in runs),
          "| cap hits:", sum(r["reached_token_cap"] for r in runs))
    print("SHA-256:", sha(raw))
    print("Artifact:", FROZEN_FILE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "generate"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        generate()


if __name__ == "__main__":
    main()
