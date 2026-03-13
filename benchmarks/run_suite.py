
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Ensure experiment runs save to parent of parent
from pathlib import Path
import os

BENCH_DIR = Path(__file__).resolve().parent          # /home/joe/ai-lab/benchmarks
PROJECT_ROOT = BENCH_DIR.parent                     # /home/joe/ai-lab
os.chdir(PROJECT_ROOT)
print(f"Working directory set to: {PROJECT_ROOT}")


def resolve_config_path(p: str) -> Path:
    """
    Accept:
      - absolute paths
      - relative to CWD (repo root)
      - relative to benchmarks/ (common case)
      - relative to script dir (benchmarks/)
    """
    cand = Path(p)
    if cand.is_absolute() and cand.exists():
        return cand

    # relative to current working dir (repo root)
    cand1 = (Path.cwd() / p)
    if cand1.exists():
        return cand1

    # relative to benchmarks directory
    cand2 = (BENCH_DIR / p)
    if cand2.exists():
        return cand2

    # last attempt: relative to script dir (same as BENCH_DIR)
    cand3 = (Path(__file__).resolve().parent / p)
    if cand3.exists():
        return cand3

    # if nothing worked, return the CWD-based path (for a good error)
    return cand1

# -------------------------
# YAML loading (best effort)
# -------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        raise RuntimeError("PyYAML not installed. Install with: conda install -y pyyaml")
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError("YAML root must be a mapping")
    return obj

# -------------------------
# Day 6: Valid-run gating
# -------------------------
def classify_failure(exit_code: int, words: int, error: str = "", stderr: str = "", timed_out: bool = False) -> str:
    if timed_out:
        return "timeout"
    if error:
        return "error"
    if stderr:
        # only treat as failure if exit_code != 0; otherwise stderr could be warnings
        if exit_code != 0:
            return "stderr"
    if exit_code != 0:
        return "nonzero_exit"
    if words <= 0:
        return "empty_output"
    return "ok"

def compute_ok(exit_code: int, words: int) -> int:
    return int(exit_code == 0 and words > 0)


# -------------------------
# Lightweight text stats
# -------------------------
_WORD_RE = re.compile(r"\S+")

def count_words(s: str) -> int:
    return len(_WORD_RE.findall(s or ""))

def count_lines(s: str) -> int:
    if not s:
        return 0
    return s.count("\n") + 1

def safe_slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "prompt"



# -------------------------
# Checks (only run when ok=1)
# -------------------------
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_TOKEN_RE = re.compile(r"\b[\w'-]+\b")

def split_sentences_basic(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENTENCE_END_RE.split(text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def count_words_clean(text: str) -> int:
    return len(_WORD_TOKEN_RE.findall(text or ""))

def check_max_sentences(text: str, max_sentences: int) -> Tuple[bool, str]:
    # naive but stable: split on .!? plus newlines
    chunks = re.split(r"[.!?]+\s+|\n+", (text or "").strip())
    sentences = [c for c in (c.strip() for c in chunks) if c]
    ok = len(sentences) <= max_sentences
    return ok, f"max_sentences_{max_sentences}={'OK' if ok else 'FAIL'}(sentences={len(sentences)})"

def check_exact_sentences_and_wordcounts(text: str, s1_words: int, s2_words: int) -> Tuple[bool, str]:
    sentences = split_sentences_basic(text)

    if len(sentences) != 2:
        return False, f"exact_match=FAIL(sentences={len(sentences)},expected=2)"

    w1 = count_words_clean(sentences[0])
    w2 = count_words_clean(sentences[1])

    ok = (w1 == s1_words and w2 == s2_words)
    return ok, f"exact_match={'OK' if ok else 'FAIL'}(w1={w1},w2={w2},expected={s1_words}/{s2_words})"

def check_json_strict(text: str, required: Dict[str, str], rules: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """
    required: key -> type ("string","number","object","array","boolean")
    rules: optional stricter rules, e.g.
      {
        "nonempty_strings": ["summary"],
        "array_lengths": {"bullets": 3},
        "array_item_types": {"bullets": "string"},
        "nonempty_array_items": ["bullets"],
      }
    """
    rules = rules or {}

    try:
        obj = json.loads(text)
    except Exception as e:
        return False, f"json_parse=FAIL({type(e).__name__})"

    if not isinstance(obj, dict):
        return False, "json_object=FAIL(not_object)"

    for k, t in required.items():
        if k not in obj:
            return False, f"json_keys=FAIL(missing={k})"

        v = obj[k]

        if t == "string" and not isinstance(v, str):
            return False, f"json_type=FAIL({k}!=string)"
        if t == "number" and not isinstance(v, (int, float)):
            return False, f"json_type=FAIL({k}!=number)"
        if t == "object" and not isinstance(v, dict):
            return False, f"json_type=FAIL({k}!=object)"
        if t == "array" and not isinstance(v, list):
            return False, f"json_type=FAIL({k}!=array)"
        if t == "boolean" and not isinstance(v, bool):
            return False, f"json_type=FAIL({k}!=boolean)"

    # Optional stricter rules
    for k in rules.get("nonempty_strings", []):
        if k in obj and isinstance(obj[k], str) and not obj[k].strip():
            return False, f"json_value=FAIL({k}=empty_string)"

    for k, expected_len in rules.get("array_lengths", {}).items():
        if k in obj and isinstance(obj[k], list) and len(obj[k]) != int(expected_len):
            return False, f"json_array=FAIL({k}_len={len(obj[k])},expected={expected_len})"

    for k, item_type in rules.get("array_item_types", {}).items():
        if k in obj and isinstance(obj[k], list):
            for i, item in enumerate(obj[k]):
                if item_type == "string" and not isinstance(item, str):
                    return False, f"json_array=FAIL({k}[{i}]!=string)"
                if item_type == "number" and not isinstance(item, (int, float)):
                    return False, f"json_array=FAIL({k}[{i}]!=number)"
                if item_type == "object" and not isinstance(item, dict):
                    return False, f"json_array=FAIL({k}[{i}]!=object)"
                if item_type == "boolean" and not isinstance(item, bool):
                    return False, f"json_array=FAIL({k}[{i}]!=boolean)"

    for k in rules.get("nonempty_array_items", []):
        if k in obj and isinstance(obj[k], list):
            for i, item in enumerate(obj[k]):
                if isinstance(item, str) and not item.strip():
                    return False, f"json_array=FAIL({k}[{i}]=empty_string)"

    return True, "json_strict=OK"

def check_numeric_only_final_line(text: str) -> Tuple[bool, str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return False, "numeric_only=FAIL(empty)"
    last = lines[-1]
    ok = bool(re.fullmatch(r"[-+]?\d+(\.\d+)?", last))
    return ok, f"numeric_only={'OK' if ok else 'FAIL'}"

def run_checks(prompt_id: str, text: str, suite: Dict[str, Any]) -> Tuple[int, int, str]:
    """
    suite supports either:
      - suite['checks'][prompt_id] = list of check dicts
      - prompt as dict with 'checks' field if prompts are structured
    """
    checks_spec: List[Dict[str, Any]] = []
    if isinstance(suite.get("checks"), dict) and isinstance(suite["checks"].get(prompt_id), list):
        checks_spec = suite["checks"][prompt_id]
    # else no checks

    if not checks_spec:
        return 0, 0, "no_checks"

    passed = 0
    details: List[str] = []
    total = 0

    for chk in checks_spec:
        total += 1
        ctype = (chk.get("type") or "").strip()
        ok = False
        msg = "unknown_check"
        if ctype == "max_sentences":
            ok, msg = check_max_sentences(text, int(chk.get("max", 5)))
        elif ctype == "follow_instr_2sent_wordcounts":
            ok, msg = check_exact_sentences_and_wordcounts(text, int(chk.get("s1_words", 8)), int(chk.get("s2_words", 12)))
        elif ctype == "json_strict":
            required = chk.get("required", {"summary": "string", "bullets": "array"})
            rules = chk.get("rules", {})
            ok, msg = check_json_strict(text, required, rules)
        elif ctype == "numeric_only_final_line":
            ok, msg = check_numeric_only_final_line(text)
        else:
            ok = False
            msg = f"unknown_check=FAIL(type={ctype})"

        if ok:
            passed += 1
        details.append(msg)

    return passed, total, ";".join(details)


# -------------------------
# Ollama API
# -------------------------
@dataclass
class OllamaResult:
    response_text: str
    raw_json: Dict[str, Any]
    elapsed_s: float
    exit_code: int
    stderr: str
    error: str
    timed_out: bool


def ollama_generate(host: str, model: str, prompt: str, options: Dict[str, Any], timeout_s: int) -> OllamaResult:
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if options:
        payload["options"] = options

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    t0 = time.time()
    timed_out = False
    stderr = ""
    error = ""
    exit_code = 0
    raw: Dict[str, Any] = {}
    text = ""

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read()
            raw = json.loads(body.decode("utf-8", errors="replace"))
            text = (raw.get("response") or "")
    except urllib.error.HTTPError as e:
        exit_code = 1
        error = f"HTTPError {e.code}"
        try:
            stderr = e.read().decode("utf-8", errors="replace")
        except Exception:
            stderr = ""
    except urllib.error.URLError as e:
        exit_code = 1
        # URLError wraps timeouts too
        error = f"URLError {getattr(e, 'reason', e)}"
        if "timed out" in str(error).lower():
            timed_out = True
    except TimeoutError:
        exit_code = 1
        timed_out = True
        error = "timeout"
    except Exception as e:
        exit_code = 1
        error = f"{type(e).__name__}: {e}"

    elapsed_s = time.time() - t0
    return OllamaResult(
        response_text=text,
        raw_json=raw if isinstance(raw, dict) else {},
        elapsed_s=elapsed_s,
        exit_code=exit_code,
        stderr=stderr,
        error=error,
        timed_out=timed_out,
    )

def normalize_prompts(prompts_obj: Any) -> Dict[str, str]:
    """
    Accept:
      prompts:
        id1: "text"
        id2: "text"
    OR
      prompts:
        - id: id1
          text: "..."
        - id: id2
          text: "..."
    """
    if isinstance(prompts_obj, dict):
        out: Dict[str, str] = {}
        for k, v in prompts_obj.items():
            if isinstance(v, str):
                out[str(k)] = v
            elif isinstance(v, dict) and isinstance(v.get("text"), str):
                out[str(k)] = v["text"]
            else:
                raise ValueError("prompts must be mapping id -> text (or id -> {text:...})")
        return out

    if isinstance(prompts_obj, list):
        out = {}
        for item in prompts_obj:
            if not isinstance(item, dict):
                raise ValueError("prompt list entries must be mappings")
            pid = item.get("id")
            txt = item.get("text")
            if not isinstance(pid, str) or not isinstance(txt, str):
                raise ValueError("prompt list entries need {id: str, text: str}")
            out[pid] = txt
        return out

    raise ValueError("prompts must be a mapping of id -> text (or a list of {id,text})")


def now_run_id() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def main() -> None:
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--suite", default="suite.yml", help="Suite YAML file (prompts + optional checks)")
    # ap.add_argument("--runner", default="", help="Optional runner YAML file (models/reps/options/timeout/host)")
    # args = ap.parse_args()

    # suite = load_yaml(args.suite)
    # prompts = normalize_prompts(suite.get("prompts"))

    # runner: Dict[str, Any] = {}
    # if args.runner:
    #     runner = load_yaml(args.runner)
        
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="suite.yml", help="Suite YAML file")
    ap.add_argument("--runner", default="", help="Optional runner YAML file")
    args = ap.parse_args()

    BASE_DIR = Path(__file__).resolve().parent
    SUITE_PATH = BASE_DIR / args.suite
    suite = load_yaml(str(SUITE_PATH))
    
    suite_path = resolve_config_path(args.suite)
    # suite = load_yaml(suite_path)
    
    prompts = normalize_prompts(suite.get("prompts"))

    runner: Dict[str, Any] = {}
    runner_path: Path | None = None
    if args.runner:
        runner_path = resolve_config_path(args.runner)
        runner = load_yaml(runner_path)

    # Runner settings: fall back to suite.yml if present; then defaults
    host = runner.get("ollama_host") or suite.get("ollama_host") or "http://127.0.0.1:11434"
    models = runner.get("models") or suite.get("models") or ["mistral", "llama3"]
    reps = int(runner.get("reps") or suite.get("reps") or 5)
    timeout_s = int(runner.get("timeout_s") or suite.get("timeout_s") or 600)
    options = runner.get("options") or suite.get("options") or {"temperature": 0.2, "top_p": 0.9, "num_predict": 256}

    if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
        raise ValueError("models must be a list[str]")

    run_id = now_run_id()
    # outdir = f"results_{run_id}"
    outdir = str(BENCH_DIR / "results" / "raw_runs" / f"results_{run_id}")
    responses_path = os.path.join(outdir, "responses.jsonl")
    os.makedirs(outdir, exist_ok=True)

    # Persist metadata.json (paper trail)
    meta = {
        "run_id": run_id,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "ollama_host": host,
        "models": models,
        "reps": reps,
        "options": options,
        "timeout_s": timeout_s,
        "suite_file": str(suite_path),
        "runner_file": str(runner_path) if runner_path else "",
    }
    with open(os.path.join(outdir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # CSV (keep existing columns, append new ones at end)
    metrics_path = os.path.join(outdir, "metrics.csv")
    fieldnames = [
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
        "checks_ok",
        "overall_ok",
        "stderr",
        "error",
        "exit_code",
        "ok",
        "failure_type",
        "response_hash",
    ]

    print(f"Running benchmark suite -> {outdir}")
    with open(metrics_path, "w", newline="", encoding="utf-8") as csvf, \
         open(responses_path, "a", encoding="utf-8") as respf:
        w = csv.DictWriter(csvf, fieldnames=fieldnames)
        w.writeheader()

        for pid, ptxt in prompts.items():
            prompt_dir = os.path.join(outdir, safe_slug(pid) + "_")
            os.makedirs(prompt_dir, exist_ok=True)
            # Save prompt text for auditability
            with open(os.path.join(prompt_dir, "prompt.txt"), "w", encoding="utf-8") as pf:
                pf.write(ptxt)

            print(f"Prompt: {pid}")

            for model in models:
                print(f"  Model: {model}")
                for rep in range(1, reps + 1):
                    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                    res = ollama_generate(host, model, ptxt, options, timeout_s)
                    out_text = res.response_text or ""
                    import hashlib
                    def hash_text(s: str) -> str:
                        return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]
                    chars = len(out_text)
                    words = count_words(out_text)
                    lines = count_lines(out_text)
                    words_per_s = (words / res.elapsed_s) if res.elapsed_s > 0 else 0.0

                    ok = compute_ok(res.exit_code, words)
                    failure_type = classify_failure(
                        res.exit_code, words,
                        error=res.error,
                        stderr=res.stderr,
                        timed_out=res.timed_out
                    )

                    # Persist artifacts (always)
                    rep_tag = f"rep{rep:02d}"
                    base = f"{model}.{rep_tag}"
                    txt_path = os.path.join(prompt_dir, base + ".txt")
                    json_path = os.path.join(prompt_dir, base + ".json")

                    with open(txt_path, "w", encoding="utf-8") as tf:
                        tf.write(out_text)

                    raw = res.raw_json if isinstance(res.raw_json, dict) else {}
                    if raw:
                        raw.setdefault("model", model)
                        raw.setdefault("prompt_id", pid)
                        raw.setdefault("rep", rep)
                    with open(json_path, "w", encoding="utf-8") as jf:
                        json.dump(raw if raw else {"model": model, "prompt_id": pid, "rep": rep, "response": out_text}, jf, indent=2)

                    # Valid-run gating: only score checks if ok==1
                    if ok:
                        checks_passed, checks_total, checks_detail = run_checks(pid, out_text, suite)
                    else:
                        checks_passed, checks_total, checks_detail = 0, 0, "skipped_invalid_run"
                    checks_ok = int(checks_total > 0 and checks_passed == checks_total)
                    overall_ok = int(ok == 1 and (checks_total == 0 or checks_ok == 1))
                    # Possibly stricter definition for later
                    # overall_ok = int(ok == 1 and checks_ok == 1)

                    row = {
                        "run_id": run_id,
                        "ts": ts,
                        "prompt_id": pid,
                        "rep": rep,
                        "model": model,
                        "elapsed_s": round(res.elapsed_s, 3),
                        "chars": chars,
                        "words": words,
                        "lines": lines,
                        "words_per_s": round(words_per_s, 3),
                        "checks_passed": checks_passed,
                        "checks_total": checks_total,
                        "checks_detail": checks_detail,
                        "checks_ok": checks_ok,
                        "overall_ok": overall_ok,
                        "stderr": (res.stderr or "").replace("\n", "\\n")[:5000],
                        "error": (res.error or "")[:1000],
                        "exit_code": res.exit_code,
                        "ok": ok,
                        "failure_type": failure_type,
                        "response_hash": hash_text(out_text),
                    }
                    # inside the loops, after you compute row:
                    resp_record = {
                        "run_id": run_id,
                        "ts": ts,
                        "prompt_id": pid,
                        "rep": rep,
                        "model": model,
                        "prompt": ptxt,              # optional (handy early on; can remove later)
                        "response_text": out_text,
                        "raw": raw,                  # full ollama JSON (if you want it)
                        "elapsed_s": res.elapsed_s,
                        "exit_code": res.exit_code,
                        "ok": ok,
                        "checks_ok": checks_ok,
                        "overall_ok": overall_ok,
                        "failure_type": failure_type,
                        "stderr": res.stderr,
                        "error": res.error,
                        "timed_out": res.timed_out,
                        "response_hash": hash_text(out_text),
                    }
                    respf.write(json.dumps(resp_record, ensure_ascii=False) + "\n")
                    respf.flush()
                    w.writerow(row)
                    csvf.flush()

    print(f"Done.\nMetrics: {os.path.abspath(metrics_path)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
