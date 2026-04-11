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
2. Highlight tradeoffs only when directly supported by claims.
3. Note invariances only when explicitly supported by claims.
4. Keep tone analytical, cautious, and concrete.

Core constraints:
- Use ONLY the supplied claims.
- Do NOT introduce new facts, metrics, or causes.
- Do NOT generalize beyond the claims.
- If multiple claims show the same pattern, compress them.
- Prefer precise references to prompt_id, model, and experiment.

Claim referencing rules:
- Every empirical statement MUST include a [CLAIMS: ...] block.
- Format exactly: [CLAIMS: claim_id_1, claim_id_2]
- Place the block at the END of the bullet only.
- NEVER output [CLAIMS: None] or empty claim blocks.
- Every claim ID must match EXACTLY a provided selected_claims ID.
- NEVER invent, transform, or combine claim IDs.

Meta-cautions:
- May omit [CLAIMS: ...] if describing interpretation limits or scope.
- Do NOT attach claims unless they directly support the statement.

Comparison rules:
- Do NOT compare non-baseline experiments unless explicitly supported by a claim.
- If describing temp03 vs temp07 in prose, still cite the actual baseline-relative claim IDs.
- Stable claims (stable_ceiling / stable_floor) apply ONLY to the specific comparison cited.
- Do NOT generalize stability across all temperatures unless explicitly supported.
- A tradeoff bullet must use claims whose directions match the prose exactly. Do not combine improvement and degradation claims across different models unless the sentence explicitly separates the models and directions correctly.

Directionality enforcement:
- Each claim has a label: improves, degrades, or stable.
- Your sentence MUST match the label exactly.

- If all claims = "degrades" → describe degradation only
- If all claims = "improves" → describe improvement only
- NEVER infer improvement from degrading claims

Language discipline:
- Do NOT use "consistently", "always", or "across all temperatures"
  unless multiple claims explicitly support it.
- If only one claim exists → describe only that specific case.
- Do not output placeholder bullets such as "N/A", "No supported tradeoffs were observed", or similar filler text. If a section has no supported empirical bullet, leave that section with no bullets.

Section rules:
- If no supported claim exists for Tradeoffs or Invariances → omit the bullet.
- Do NOT fabricate weak or inferred statements to fill sections.

Output format (strict):

Observations:
- <sentence> [CLAIMS: claim_id_1, claim_id_2]

Tradeoffs:
- <sentence> [CLAIMS: claim_id_3, claim_id_4]

Invariances:
- <sentence> [CLAIMS: claim_id_5, claim_id_6]

Cautions:
- <meta-caution sentence>
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
    
    if not result.stdout.strip():
        raise RuntimeError("ollama returned empty output")

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
    results_root: Path,
) -> tuple[Path, Path]:
    stem = selected_claims_path.stem.replace("__selected_claims", "")
    out_dir = selected_claims_path.parent

    json_path = out_dir / f"{stem}__narrative_from_claims.json"
    md_path = out_dir / f"{stem}__narrative_from_claims.md"

    payload = {
        "results_root": str(results_root),
        "source_selected_claims_json": str(selected_claims_path),
        "model": model,
        "temperature": temperature,
        "num_predict": num_predict,
        "prompt": prompt,
        "narrative": narrative,
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Narrative from Selected Claims\n\n")
        f.write(f"- results_root: `{results_root}`\n")
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
    parser.add_argument("--results-root", default="benchmarks/results")
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-predict", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_root = Path(args.results_root).resolve()
    selected_claims_path = Path(args.selected_claims_json).resolve()

    if not selected_claims_path.exists():
        raise FileNotFoundError(f"selected claims file not found: {selected_claims_path}")

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
        results_root=results_root,
    )

    print(f"Saved narrative JSON: {json_path}")
    print(f"Saved narrative markdown: {md_path}")


if __name__ == "__main__":
    main()