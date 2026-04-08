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
2. Highlight tradeoffs or asymmetries only when they are directly supported by the supplied claims.
3. Note invariances or saturated behaviors only when explicitly supported by the supplied claims.
4. Keep the tone analytical, cautious, and concrete.

Rules:
- Use only the supplied claims.
- Do not introduce any facts, metrics, or causes not present in the input.
- Do not generalize beyond the supplied claims.
- If multiple claims repeat the same pattern, compress them into a concise observation.
- Prefer specific references to prompt_id, model, and experiment names.
- Keep hypotheses minimal and cautious.
- Every bullet in Observations, Tradeoffs, and Invariances must include one or more supporting claim IDs.
- Format claim references exactly as: [CLAIMS: claim_id_1, claim_id_2]
- Always use [CLAIMS: ...] even for a single claim.
- Place the [CLAIMS: ...] block at the end of the bullet only.
- Do not place claim references at the beginning or middle of a bullet.
- Do not write any empirical statement without at least one [CLAIMS: ...] block.
- Cautions may omit claim IDs if they are clearly meta-cautions about interpretation limits.
- Preserve experiment names exactly as written (e.g. temp03, temp07). Do not rewrite them as numeric temperatures unless they are copied exactly from the claims.
- Every claim ID cited in [CLAIMS: ...] must appear verbatim in the supplied selected_claims list.
- Never invent, infer, extrapolate, rewrite, or transform claim IDs.
- Do not create new claim IDs by combining two existing claims into a synthetic comparison form.
- If you describe a comparison between temp03 and temp07 in prose, cite the underlying selected claim IDs exactly as provided, even if those IDs are baseline-relative to temp0.
- Copy claim IDs exactly from the supplied selected_claims list. Do not generate claim IDs from naming patterns.
- If no suitable selected claim supports an invariance, omit the invariance bullet rather than fabricating support.
- If no supported claim exists for a section, omit that bullet rather than filling the section with a weak or inferred statement.
- If a comparative statement would require a claim ID that does not exist verbatim in selected_claims, keep the prose grounded but cite only the real selected claim IDs.
- Before writing each [CLAIMS: ...] block, verify that every cited claim ID appears exactly in the supplied selected_claims list.
- Meta-cautions may omit claim IDs when they describe interpretation limits, sample-size limitations, or scope boundaries rather than empirical findings.
- Do not attach [CLAIMS: ...] to a meta-caution unless the cited IDs are real selected_claims and directly support the statement.
- Do not compare one non-baseline experiment directly against another unless an exact selected claim for that comparison exists.
- Do not cite claim IDs that are not present verbatim in selected_claims.
- Do not write placeholder claim blocks such as [CLAIMS: ] or [CLAIMS: None].
- If no supported bullet exists for Tradeoffs or Invariances, write no bullet under that section.
- A stable_ceiling or stable_floor claim supports only the specific comparison experiment cited; do not generalize it to “all temperatures” unless both comparison experiments are explicitly cited.
- Do not combine stable and degrading claims into a single invariance statement.

Output exactly in this structure:

Observations:
- <observation sentence> [CLAIMS: claim_id_1, claim_id_2]

Tradeoffs:
- <tradeoff sentence> [CLAIMS: claim_id_3, claim_id_4]

Invariances:
- <invariance sentence> [CLAIMS: claim_id_5, claim_id_6]

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