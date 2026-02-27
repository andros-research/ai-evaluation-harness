#!/usr/bin/env python3
"""
run_suite.py

Research-grade(ish) local harness runner for Ollama models via HTTP API.

- Reads suite.yml (optional) for models/prompts/options/reps
- Falls back to a built-in default suite if suite.yml is missing
- Executes each (prompt_id x model x rep) via POST /api/generate
- Writes:
    results_<ts>/
        metrics.csv
        metadata.json
        <prompt_id>/
            prompt.txt
            <model>.rep01.txt
            <model>.rep01.json
            ...
- Tracks failures properly (exit_code, stderr, error)
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # We'll handle missing pyyaml gracefully.


# ----------------------------
# Config defaults
# ----------------------------

DEFAULT_MODELS = ["mistral", "llama3", "llama3:70b"]

DEFAULT_PROMPTS: Dict[str, str] = {
    "attention_5sent": "Explain attention in transformers in 5 sentences.",
    "json_strict": "Return ONLY valid JSON with keys: summary (string), bullets (array of 3 strings).",
    "reasoning_trap": (
        "You have a bond with DV01=100k and convexity=12k. Rates shift +10bp then +10bp. "
        "Return ONLY a single line with the approximate PnL number (no words, no units)."
    ),
    "follow_instr": (
        "Write exactly 2 sentences. Sentence 1 must be 8 words. Sentence 2 must be 12 words."
    ),
    "style_hemi": (
        "Write a Hemingway-lite paragraph about a quiet office morning. "
        "No metaphors. No adjectives longer than 8 letters."
    ),
    "verbosity_drift": (
        "In one paragraph, explain what a yield curve is, for a smart non-expert."
    ),
}

DEFAULT_REPS = 3

# API options that affect output behavior.
DEFAULT_OPTIONS: Dict[str, Any] = {
    "temperature": 0.2,
    "top_p": 0.9,
    "num_predict": 256,  # cap output tokens
}

DEFAULT_TIMEOUT_S = 600

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


# ----------------------------
# Helpers
# ----------------------------

_SENT_SPLIT = re.compile(r"[.!?]+\s+")


def now_run_id() -> str:
    # matches your prior style: results_YYYY-MM-DD_HH-MM-SS
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def safe_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def count_sentences(text: str) -> int:
    t = text.strip()
    if not t:
        return 0
    return len([s for s in _SENT_SPLIT.split(t) if s.strip()])


def basic_counts(text: str) -> Dict[str, int]:
    if not text:
        return {"chars": 0, "words": 0, "lines": 0}
    return {
        "chars": len(text),
        "words": len(text.split()),
        "lines": text.count("\n") + 1,
    }


def slugify(s: str) -> str:
    # keep it simple and stable: lowercase + safe chars
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


@dataclass
class RunResult:
    exit_code: int
    stdout: str
    stderr: str
    meta: Dict[str, Any]


def ollama_generate(
    model: str,
    prompt: str,
    options: Optional[Dict[str, Any]] = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> RunResult:
    """
    Call Ollama generate API (non-streaming).
    """
    url = f"{OLLAMA_HOST}/api/generate"
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if options:
        payload["options"] = options

    t0 = time.time()
    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        elapsed = time.time() - t0

        if r.status_code != 200:
            return RunResult(
                exit_code=2,
                stdout="",
                stderr=f"HTTP {r.status_code}: {r.text[:500]}",
                meta={"elapsed_s": elapsed},
            )

        data = r.json()
        text = data.get("response", "")
        return RunResult(
            exit_code=0,
            stdout=text,
            stderr="",
            meta={"elapsed_s": elapsed, "raw": data},
        )

    except requests.Timeout:
        elapsed = time.time() - t0
        return RunResult(
            exit_code=124,
            stdout="",
            stderr=f"Timeout after {timeout_s}s",
            meta={"elapsed_s": elapsed},
        )
    except Exception as e:
        elapsed = time.time() - t0
        return RunResult(
            exit_code=1,
            stdout="",
            stderr=f"Exception: {type(e).__name__}: {e}",
            meta={"elapsed_s": elapsed},
        )


def load_suite_yaml(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    if yaml is None:
        raise RuntimeError(
            "suite.yml exists but PyYAML is not installed. "
            "Install with: python -m pip install pyyaml"
        )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def normalize_models(x: Any) -> List[str]:
    if x is None:
        return DEFAULT_MODELS[:]
    if isinstance(x, list):
        return [str(m) for m in x]
    raise ValueError("models must be a list")


def normalize_prompts(x: Any) -> Dict[str, str]:
    if x is None:
        return dict(DEFAULT_PROMPTS)
    if isinstance(x, dict):
        return {slugify(str(k)): str(v) for k, v in x.items()}
    raise ValueError("prompts must be a mapping of id -> text")


def normalize_reps(x: Any) -> int:
    if x is None:
        return DEFAULT_REPS
    r = int(x)
    if r < 1:
        raise ValueError("reps must be >= 1")
    return r


def normalize_options(x: Any) -> Dict[str, Any]:
    if x is None:
        return dict(DEFAULT_OPTIONS)
    if isinstance(x, dict):
        # merge defaults + overrides
        merged = dict(DEFAULT_OPTIONS)
        merged.update(x)
        return merged
    raise ValueError("options must be a mapping")


# ----------------------------
# Minimal checks (Day 5 style)
# ----------------------------

def check_json_strict(text: str) -> Tuple[int, int, str]:
    """
    Expect ONLY JSON with required keys.
    """
    total = 1
    t = text.strip()
    if not t:
        return (0, total, "json=FAIL(empty)")
    try:
        obj = json.loads(t)
    except Exception as e:
        return (0, total, f"json=FAIL(parse:{type(e).__name__})")
    ok = isinstance(obj, dict) and "summary" in obj and "bullets" in obj
    if not ok:
        return (0, total, "json=FAIL(keys)")
    if not isinstance(obj["summary"], str):
        return (0, total, "json=FAIL(summary_type)")
    if not (isinstance(obj["bullets"], list) and len(obj["bullets"]) == 3 and all(isinstance(b, str) for b in obj["bullets"])):
        return (0, total, "json=FAIL(bullets_type)")
    return (1, total, "json=OK")


def check_attention_5sent(text: str) -> Tuple[int, int, str]:
    total = 1
    n = count_sentences(text)
    ok = (n > 0) and (n <= 5)
    return (1 if ok else 0, total, f"max_sentences_5={'OK' if ok else 'FAIL'}(sentences={n})")


def check_follow_instr(text: str) -> Tuple[int, int, str]:
    """
    Exactly 2 sentences; sentence1=8 words; sentence2=12 words.
    """
    total = 1
    t = text.strip()
    if not t:
        return (0, total, "exact_match=FAIL(empty)")
    # Split on sentence end. This is intentionally simple; good enough for first harness.
    parts = re.split(r"[.!?]+", t)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) != 2:
        return (0, total, f"exact_match=FAIL(sentences={len(parts)})")
    w1 = len(parts[0].split())
    w2 = len(parts[1].split())
    ok = (w1 == 8 and w2 == 12)
    return (1 if ok else 0, total, f"exact_match={'OK' if ok else 'FAIL'}(w1={w1},w2={w2})")


def check_reasoning_trap(text: str) -> Tuple[int, int, str]:
    """
    Numeric-only single-line output.
    """
    total = 1
    t = text.strip()
    if not t:
        return (0, total, "numeric_only=FAIL(empty)")
    # Allow signs, decimals
    ok = bool(re.fullmatch(r"[+-]?\d+(\.\d+)?", t))
    return (1 if ok else 0, total, f"numeric_only={'OK' if ok else 'FAIL'}")


def run_checks(prompt_id: str, text: str, exit_code: int) -> Tuple[int, int, str]:
    """
    Returns: (passed, total, detail)
    """
    # If the model call failed, don't score content checks.
    if exit_code != 0:
        return (0, 0, "skipped_due_to_error")
    if prompt_id == "json_strict":
        return check_json_strict(text)
    if prompt_id == "attention_5sent":
        return check_attention_5sent(text)
    if prompt_id == "follow_instr":
        return check_follow_instr(text)
    if prompt_id == "reasoning_trap":
        return check_reasoning_trap(text)
    # style_hemi / verbosity_drift: no strict checks yet
    return (0, 0, "no_checks")


# ----------------------------
# Main runner
# ----------------------------

def main() -> None:
    root = Path(__file__).resolve().parent
    suite_path = root / "suite.yml"

    suite = load_suite_yaml(suite_path) or {}
    models = normalize_models(suite.get("models"))
    prompts = normalize_prompts(suite.get("prompts"))
    reps = normalize_reps(suite.get("reps"))
    options = normalize_options(suite.get("options"))
    timeout_s = int(suite.get("timeout_s", DEFAULT_TIMEOUT_S))

    run_id = now_run_id()
    outdir = root / f"results_{run_id}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ollama_host": OLLAMA_HOST,
        "models": models,
        "reps": reps,
        "options": options,
        "timeout_s": timeout_s,
        "suite_file": "suite.yml" if suite_path.exists() else None,
    }
    safe_write_text(outdir / "metadata.json", json.dumps(metadata, indent=2))

    metrics_path = outdir / "metrics.csv"
    fields = [
        "run_id",
        "ts",
        "prompt_id",
        "rep",
        "model",
        "elapsed_s",
        "chars",
        "words",
        "lines",
        "words_per_s",
        "checks_passed",
        "checks_total",
        "checks_detail",
        "exit_code",
        "stderr",
        "error",
    ]

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        print(f"Running benchmark suite -> {outdir.name}")
        for prompt_id, prompt_text in prompts.items():
            prompt_id = slugify(prompt_id)
            prompt_dir = outdir / f"{prompt_id}_"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            safe_write_text(prompt_dir / "prompt.txt", prompt_text)

            print(f"Prompt: {prompt_id}")
            for model in models:
                print(f"  Model: {model}")
                # Optional warmup per (prompt,model) — don’t log
                _ = ollama_generate(model=model, prompt="warmup", options=options, timeout_s=timeout_s)

                for rep in range(1, reps + 1):
                    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    res = ollama_generate(model=model, prompt=prompt_text, options=options, timeout_s=timeout_s)

                    out_text = res.stdout or ""
                    err_text = res.stderr or ""

                    # Write raw output + raw json metadata for debugging
                    safe_write_text(prompt_dir / f"{model}.rep{rep:02d}.txt", out_text)
                    safe_write_text(prompt_dir / f"{model}.rep{rep:02d}.json", json.dumps(res.meta.get("raw", {}), indent=2))

                    counts = basic_counts(out_text)
                    elapsed = float(res.meta.get("elapsed_s", 0.0))
                    wps = (counts["words"] / elapsed) if elapsed > 0 else 0.0

                    passed, total, detail = run_checks(prompt_id, out_text, res.exit_code)

                    row = {
                        "run_id": run_id,
                        "ts": ts,
                        "prompt_id": prompt_id,
                        "rep": rep,
                        "model": model,
                        "elapsed_s": round(elapsed, 3),
                        "chars": counts["chars"],
                        "words": counts["words"],
                        "lines": counts["lines"],
                        "words_per_s": round(wps, 3),
                        "checks_passed": passed,
                        "checks_total": total,
                        "checks_detail": detail,
                        "exit_code": res.exit_code,
                        "stderr": err_text[:200].replace("\n", "\\n"),
                        "error": "" if res.exit_code == 0 else "ollama_generate_failed",
                    }
                    w.writerow(row)

    print(f"Done. Results saved in {outdir.name}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()

