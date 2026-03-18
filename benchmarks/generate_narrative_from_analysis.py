from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_analysis(analysis: dict[str, Any], top_n: int = 12) -> dict[str, Any]:
    comparisons = analysis.get("comparisons", [])
    summary: list[dict[str, Any]] = []

    for comp in comparisons:
        comp_name = comp.get("comparison_experiment")
        cells = comp.get("cells", [])

        top_cells = sorted(
            cells,
            key=lambda x: abs(float(x.get("delta_pass_rate", 0.0))),
            reverse=True,
        )[:top_n]

        summary.append(
            {
                "comparison_experiment": comp_name,
                "top_cells": [
                    {
                        "prompt_id": c.get("prompt_id"),
                        "model": c.get("model"),
                        "baseline_pass_rate": c.get("baseline_pass_rate"),
                        "comparison_pass_rate": c.get("comparison_pass_rate"),
                        "delta_pass_rate": c.get("delta_pass_rate"),
                        "label": c.get("label"),
                    }
                    for c in top_cells
                ],
            }
        )

    stable_cells = []
    for cell in analysis.get("cells", []):
        labels = cell.get("labels", {})
        if any(v in {"stable_always_pass", "stable_always_fail"} for v in labels.values()):
            stable_cells.append(
                {
                    "prompt_id": cell.get("prompt_id"),
                    "model": cell.get("model"),
                    "baseline_pass_rate": cell.get("baseline_pass_rate"),
                    "labels": labels,
                }
            )

    return {
        "suite_name": analysis.get("suite_name"),
        "metric": analysis.get("metric"),
        "baseline_experiment": analysis.get("baseline_experiment"),
        "comparison_experiments": analysis.get("comparison_experiments", []),
        "top_comparison_cells": summary,
        "stable_cells": stable_cells[:24],
    }


def build_prompt(analysis: dict[str, Any]) -> str:
    summary = summarize_analysis(analysis)

    instruction = """You are analyzing benchmark comparison results from language model evaluations.

You will be given structured experimental analysis for one suite.
The analysis compares one baseline experiment against one or more comparison experiments.

Your task:
1. Identify the most important stable patterns.
2. Identify tradeoffs across prompts or models.
3. Identify invariances.
4. Identify anomalies or asymmetries.
5. Propose cautious hypotheses grounded only in the observed data.

Rules:
- Use only the provided data.
- Do not invent causes that are not supported by the analysis.
- If evidence is weak or mixed, say so.
- Prefer concrete references to prompt_id, model, and experiment names.
- Keep the writing concise and analytical.

Output exactly in this structure:

Observations:
- ...

Tradeoffs:
- ...

Invariances:
- ...

Anomalies:
- ...

Hypotheses:
- ...
"""

    payload = json.dumps(
        {
            "summary": summary,
            "raw_analysis": analysis,
        },
        indent=2,
    )

    return f"{instruction}\nInput data:\n{payload}\n"

import os
import shutil
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
    analysis_path: Path,
    prompt: str,
    narrative: str,
    model: str,
    temperature: float,
    num_predict: int,
) -> tuple[Path, Path]:
    stem = analysis_path.stem.replace("__analysis", "")
    out_dir = analysis_path.parent

    json_path = out_dir / f"{stem}__narrative.json"
    md_path = out_dir / f"{stem}__narrative.md"

    payload = {
        "source_analysis_json": str(analysis_path),
        "model": model,
        "temperature": temperature,
        "num_predict": num_predict,
        "prompt": prompt,
        "narrative": narrative,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Narrative v0\n\n")
        f.write(f"- source_analysis_json: `{analysis_path}`\n")
        f.write(f"- model: `{model}`\n")
        f.write(f"- temperature: `{temperature}`\n")
        f.write(f"- num_predict: `{num_predict}`\n\n")
        f.write("## Narrative\n\n")
        f.write(narrative)
        f.write("\n")

    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-json", required=True)
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--num-predict", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    analysis_path = Path(args.analysis_json).resolve()
    analysis = load_json(analysis_path)
    prompt = build_prompt(analysis)
    narrative = run_ollama(
        model=args.model,
        prompt=prompt,
        temperature=args.temperature,
        num_predict=args.num_predict,
    )

    json_path, md_path = save_outputs(
        analysis_path=analysis_path,
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