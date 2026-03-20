from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_selected_claims(selected_payload: dict[str, Any]) -> dict[str, Any]:
    claims = selected_payload.get("selected_claims", [])

    compact_claims = []
    for c in claims:
        compact_claims.append(
            {
                "claim_id": c.get("claim_id"),
                "claim_type": c.get("claim_type"),
                "prompt_id": c.get("prompt_id"),
                "model": c.get("model"),
                "comparison_experiment": c.get("comparison_experiment"),
                "baseline_experiment": c.get("baseline_experiment"),
                "baseline_value": c.get("baseline_value"),
                "comparison_value": c.get("comparison_value"),
                "delta_value": c.get("delta_value"),
                "label": c.get("label"),
                "claim_strength": c.get("claim_strength"),
            }
        )

    return {
        "suite_name": selected_payload.get("suite_name"),
        "metric": selected_payload.get("metric"),
        "baseline_experiment": selected_payload.get("baseline_experiment"),
        "comparison_experiments": selected_payload.get("comparison_experiments", []),
        "selection_policy": selected_payload.get("selection_policy", {}),
        "selection_summary": selected_payload.get("selection_summary", {}),
        "selected_claims": compact_claims,
    }


def build_prompt(selected_payload: dict[str, Any]) -> str:
    summary = summarize_selected_claims(selected_payload)

    instruction = """You are writing a concise benchmark note from validated, pre-selected claims.

You will be given a structured set of claims derived from language model benchmark comparisons.
These claims have already passed validation and selection.

Your task:
1. Summarize the most important observed changes.
2. Highlight tradeoffs or asymmetries across prompts or models.
3. Note any meaningful invariances or saturated behaviors.
4. Keep the tone analytical, cautious, and concrete.

Rules:
- Use only the supplied claims.
- Do not introduce any facts, metrics, or causes not present in the input.
- Do not generalize beyond the supplied claims.
- If multiple claims repeat the same pattern, compress them into a concise observation.
- Prefer specific references to prompt_id, model, and experiment names.
- Keep hypotheses minimal and cautious.

Output exactly in this structure:

Observations:
- ...

Tradeoffs:
- ...

Invariances:
- ...

Cautions:
- ...
"""

    payload = json.dumps(summary, indent=2)
    return f"{instruction}\nInput data:\n{payload}\n"


def run_ollama(model: str, prompt: str, temperature: float, num_predict: int) -> str:
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        raise RuntimeError(
            "Could not find 'ollama' on PATH. "
            "Run `which ollama` in your shell and either add it to PATH "
            "or hardcode the full executable path in run_ollama()."
        )

    env = os.environ.copy()
    env["OLLAMA_TEMPERATURE"] = str(temperature)
    env["OLLAMA_NUM_PREDICT"] = str(num_predict)

    cmd = [ollama_path, "run", model]

    result = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ollama run failed with code {result.returncode}\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout.strip()


def save_outputs(
    selected_claims_path: Path,
    prompt: str,
    narrative: str,
    model: str,
    temperature: float,
    num_predict: int,
) -> tuple[Path, Path]:
    stem = selected_claims_path.stem.replace("__selected_claims", "")
    out_dir = selected_claims_path.parent

    json_path = out_dir / f"{stem}__narrative_from_claims.json"
    md_path = out_dir / f"{stem}__narrative_from_claims.md"

    payload = {
        "source_selected_claims_json": str(selected_claims_path),
        "model": model,
        "temperature": temperature,
        "num_predict": num_predict,
        "prompt": prompt,
        "narrative": narrative,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Narrative from Selected Claims\n\n")
        f.write(f"- source_selected_claims_json: `{selected_claims_path}`\n")
        f.write(f"- model: `{model}`\n")
        f.write(f"- temperature: `{temperature}`\n")
        f.write(f"- num_predict: `{num_predict}`\n\n")
        f.write("## Narrative\n\n")
        f.write(narrative)
        f.write("\n")

    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-claims-json", required=True)
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-predict", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selected_claims_path = Path(args.selected_claims_json).resolve()
    selected_payload = load_json(selected_claims_path)
    prompt = build_prompt(selected_payload)

    narrative = run_ollama(
        model=args.model,
        prompt=prompt,
        temperature=args.temperature,
        num_predict=args.num_predict,
    )

    json_path, md_path = save_outputs(
        selected_claims_path=selected_claims_path,
        prompt=prompt,
        narrative=narrative,
        model=args.model,
        temperature=args.temperature,
        num_predict=args.num_predict,
    )

    print(f"Saved narrative JSON: {json_path}")
    print(f"Saved narrative markdown: {md_path}")


if __name__ == "__main__":
    main()