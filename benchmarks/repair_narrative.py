from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

REPAIR_SUFFIX_RE = re.compile(r"__narrative_repair_(\d+)$")

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def get_selected_claims_list(selected_payload: dict[str, Any]) -> list[dict[str, Any]]:
    claims = selected_payload.get("selected_claims")
    if isinstance(claims, list):
        return claims

    claims = selected_payload.get("validated_claims")
    if isinstance(claims, list):
        return claims

    return []


def index_claims_by_id(claims: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(c["claim_id"]): c
        for c in claims
        if isinstance(c, dict) and "claim_id" in c
    }


def extract_narrative_text(narrative_payload: dict[str, Any]) -> str:
    """
    Extract narrative text from a few possible narrative JSON shapes.
    """

    # Common direct text fields
    for key in [
        "narrative_text",
        "narrative_markdown",
        "markdown",
        "narrative",
        "text",
        "output_text",
        "output",
    ]:
        value = narrative_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # If sections are stored as a dict of lists or strings
    sections = narrative_payload.get("sections")
    if isinstance(sections, dict):
        rendered = []
        for section_name in ["observations", "tradeoffs", "invariances", "cautions"]:
            value = sections.get(section_name)
            if not value:
                continue

            rendered.append(f"{section_name.capitalize()}:")
            if isinstance(value, list):
                for item in value:
                    rendered.append(f"- {str(item).strip()}")
            elif isinstance(value, str):
                for line in value.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("- "):
                        rendered.append(line)
                    else:
                        rendered.append(f"- {line}")
            rendered.append("")

        if rendered:
            return "\n".join(rendered).strip()

    # Fallback: look for direct top-level section strings/lists
    rendered = []
    for section_name in ["observations", "tradeoffs", "invariances", "cautions"]:
        value = narrative_payload.get(section_name)
        if not value:
            continue

        rendered.append(f"{section_name.capitalize()}:")
        if isinstance(value, list):
            for item in value:
                rendered.append(f"- {str(item).strip()}")
        elif isinstance(value, str):
            for line in value.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("- "):
                    rendered.append(line)
                else:
                    rendered.append(f"- {line}")
        rendered.append("")

    if rendered:
        return "\n".join(rendered).strip()

    raise ValueError(
        f"Could not find narrative text in narrative JSON. Available keys: {sorted(narrative_payload.keys())}"
    )


def get_base_artifact_stem(narrative_json: Path) -> tuple[str, int]:
    """
    Returns:
        base_stem: artifact stem without repair suffix
        next_repair_idx: next repair number to write
    """

    name = narrative_json.name

    # Original narrative
    if name.endswith("__narrative_from_claims.json"):
        base_stem = name.replace("__narrative_from_claims.json", "")
        return base_stem, 1

    stem = narrative_json.stem

    # Already-numbered repaired narrative
    m = REPAIR_SUFFIX_RE.search(stem)
    if m:
        repair_idx = int(m.group(1))
        base_stem = REPAIR_SUFFIX_RE.sub("", stem)
        return base_stem, repair_idx + 1

    # Fallback for old-style repaired artifact names
    if stem.endswith("__narrative_repaired"):
        base_stem = stem.replace("__narrative_repaired", "")
        return base_stem, 2

    # Last-resort fallback
    return stem, 1


def derive_output_paths(narrative_json: Path) -> dict[str, Path]:
    base_stem, next_repair_idx = get_base_artifact_stem(narrative_json)
    repair_tag = f"__narrative_repair_{next_repair_idx:02d}"

    parent = narrative_json.parent
    return {
        "json": parent / f"{base_stem}{repair_tag}.json",
        "md": parent / f"{base_stem}{repair_tag}.md",
        "meta": parent / f"{base_stem}{repair_tag}__repair_metadata.json",
    }


def format_claim_subset(claim_ids: list[str], claims_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [claims_by_id[cid] for cid in claim_ids if cid in claims_by_id]


def build_instruction() -> str:
    return """You are revising a benchmark note to improve evidence coverage.

You will be given:
1. The current narrative
2. The selected claims
3. A small set of important unused claim IDs that should be incorporated if possible

Your task:
- Revise the narrative to incorporate the target unused claims if they add meaningful new information
- Preserve the existing section structure and concise analytical style
- Prefer minimal edits over wholesale rewriting
- Do not introduce any facts not supported by the selected claims
- Do not invent, rewrite, or transform claim IDs
- Every empirical bullet must cite real selected claim IDs verbatim
- If a target claim is redundant with an existing bullet, merge it into a revised observation or tradeoff instead of adding unnecessary new bullets
- If a target claim does not add meaningful new information, you may leave it unused, but only if the current narrative already captures its content well
- Cautions may omit claim IDs if they are clearly meta-cautions
- For every empirical bullet, the model names in prose must exactly match the models referenced by the cited claim IDs.
- Do not cite a mistral claim in a bullet that describes llama3.
- Do not merge llama3 and llama3:70b into one bullet unless the prose explicitly names both.

Important:
- All cited claim IDs must be copied exactly from the provided selected_claims
- Do not create synthetic comparison IDs
- If comparing two temperatures in prose, cite the underlying real selected claim IDs
- The model, prompt_id, and experiment names in prose must be consistent with the cited claim IDs.
- If a cited claim refers to mistral, the prose must refer to mistral, not llama3.
- If a cited claim refers to llama3:70b, the prose must preserve that distinction and not collapse it into llama3.
- If a bullet cites a target claim, the prose in that bullet must explicitly name the same model family as the cited claim ID.
- Do not satisfy target-claim inclusion by attaching the target claim ID to a bullet whose prose primarily describes a different model.
- If no valid revision can incorporate a target claim without violating these rules, omit it and add a short note under Cautions explaining that the target claim could not be incorporated cleanly.
- Do not write placeholder bullets like "None (...)" in Observations, Tradeoffs, or Invariances. Omit the bullet instead.
- Do not merge target claims into an existing bullet if doing so would combine different models in one generalized statement.
- If a target claim belongs to a different model than the existing cited claims in a bullet, create a separate bullet instead of merging.
- When in doubt, prefer adding a new model-specific observation over compressing across models.
- Never combine llama3 and mistral into the same empirical bullet unless every cited claim in that bullet clearly supports a cross-model comparison stated explicitly in the prose.
- Cross-model comparisons are only valid if the prose explicitly compares the same prompt_id and experiment across models, and all cited claims correspond to that exact comparison.
- If a target claim cannot be incorporated cleanly into an existing bullet without changing the cited model names, add a new bullet for that target claim.


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


def build_prompt(
    *,
    current_narrative_text: str,
    selected_claims: list[dict[str, Any]],
    target_claim_ids: list[str],
    target_claims: list[dict[str, Any]],
) -> str:
    instruction = build_instruction()

    payload = {
        "target_claim_ids": target_claim_ids,
        "target_claims": target_claims,
        "selected_claims": selected_claims,
    }

    return (
        f"{instruction}\n\n"
        f"CURRENT NARRATIVE:\n"
        f"{current_narrative_text}\n\n"
        f"REPAIR TARGET CLAIM IDS:\n"
        f"{json.dumps(target_claim_ids, indent=2)}\n\n"
        f"REPAIR TARGET CLAIM OBJECTS:\n"
        f"{json.dumps(target_claims, indent=2)}\n\n"
        f"FULL SELECTED CLAIMS:\n"
        f"{json.dumps(selected_claims, indent=2)}\n\n"
        f"Return only the revised benchmark note in the required structure."
    )


def call_ollama_generate(
    *,
    model: str,
    prompt: str,
    temperature: float,
    num_predict: int,
) -> str:
    cmd = [
        "ollama",
        "run",
        model,
    ]

    env = os.environ.copy()
    env["OLLAMA_NUM_PREDICT"] = str(num_predict)
    env["OLLAMA_TEMPERATURE"] = str(temperature)

    proc = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"Ollama generation failed.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    return proc.stdout.strip()


SECTION_RE = re.compile(
    r"^(Observations|Tradeoffs|Invariances|Cautions):\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


def parse_repaired_text_to_sections(text: str) -> dict[str, list[str]]:
    current_section = None
    sections: dict[str, list[str]] = {
        "observations": [],
        "tradeoffs": [],
        "invariances": [],
        "cautions": [],
    }

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        m = SECTION_RE.match(line)
        if m:
            current_section = m.group(1).lower()
            continue

        if current_section is None:
            continue

        if line.startswith("- "):
            sections[current_section].append(line[2:].strip())
        else:
            # continuation line: append to previous bullet if any
            if sections[current_section]:
                sections[current_section][-1] += " " + line

    return sections


def infer_metadata_from_selected_payload(
    selected_payload: dict[str, Any],
    selected_claims_path: Path,
) -> dict[str, Any]:
    suite_name = selected_payload.get("suite_name", "")
    metric = selected_payload.get("metric", "")
    baseline_experiment = selected_payload.get("baseline_experiment", "")
    comparison_experiments = selected_payload.get("comparison_experiments", [])

    # Fallback: derive from filename if fields are missing
    if not suite_name or not metric:
        stem = selected_claims_path.name.replace("__selected_claims.json", "")
        parts = stem.split("__")

        if len(parts) >= 4:
            # expected shape:
            # suite_instruction__temp0__vs__temp03_temp07__checks_pass_rate
            suite_name = suite_name or parts[0]
            baseline_experiment = baseline_experiment or parts[1]

            # crude but workable fallback
            if "checks_pass_rate" in stem and not metric:
                metric = "checks_pass_rate"

            # comparison experiments fallback from temp03_temp07
            if not comparison_experiments and len(parts) >= 4:
                maybe_comps = parts[3]
                if isinstance(maybe_comps, str) and maybe_comps.startswith("temp"):
                    comparison_experiments = maybe_comps.split("_")

    return {
        "suite_name": suite_name,
        "metric": metric,
        "baseline_experiment": baseline_experiment,
        "comparison_experiments": comparison_experiments,
    }
    

def build_repaired_payload(
    *,
    repaired_text: str,
    original_narrative_payload: dict[str, Any],
    selected_payload: dict[str, Any],
    selected_claims_path: Path,
    parent_narrative_json: Path,
    repair_iteration: int,
    target_claim_ids: list[str],
    model: str,
    temperature: float,
    num_predict: int,
) -> dict[str, Any]:
    sections = parse_repaired_text_to_sections(repaired_text)
    meta = infer_metadata_from_selected_payload(selected_payload, selected_claims_path)

    return {
        "suite_name": meta["suite_name"],
        "metric": meta["metric"],
        "baseline_experiment": meta["baseline_experiment"],
        "comparison_experiments": meta["comparison_experiments"],
        "source_selected_claims_json": str(selected_claims_path),
        "source_narrative_json": str(parent_narrative_json),
        "parent_narrative_json": str(parent_narrative_json),
        "repair_iteration": repair_iteration,
        "repair_mode": "targeted_unused_claim_inclusion",
        "target_claim_ids": target_claim_ids,
        "model": model,
        "temperature": temperature,
        "num_predict": num_predict,
        "prompt": build_instruction(),
        "narrative": repaired_text,
        "sections": sections,
    }


def expected_model_tokens_for_claim_id(claim_id: str) -> list[str]:
    cid = claim_id.lower()
    if "__mistral__" in cid:
        return ["mistral"]
    if "__llama3_70b__" in cid:
        return ["70b"]
    if "__llama3__" in cid:
        return ["llama3"]
    return []


def validate_repaired_sections(
    sections: dict[str, list[str]],
    target_claim_ids: list[str],
) -> list[str]:
    errors: list[str] = []

    empirical_sections = ["observations", "tradeoffs", "invariances"]

    # 1. No "None (...)" bullets in empirical sections
    for section_name in empirical_sections:
        for bullet in sections.get(section_name, []):
            if bullet.strip().lower().startswith("none"):
                errors.append(
                    f"{section_name}: found placeholder bullet starting with 'None' -> {bullet}"
                )

    # 2. For each target claim, if cited in a bullet, prose should mention the expected model token
    for section_name in empirical_sections:
        for bullet in sections.get(section_name, []):
            bullet_lower = bullet.lower()
            for claim_id in target_claim_ids:
                if claim_id in bullet:
                    expected_tokens = expected_model_tokens_for_claim_id(claim_id)
                    if expected_tokens and not any(tok in bullet_lower for tok in expected_tokens):
                        errors.append(
                            f"{section_name}: bullet cites {claim_id} but prose does not mention expected model token(s) {expected_tokens}: {bullet}"
                        )

                    # Optional stricter check:
                    if "__mistral__" in claim_id.lower() and "llama3" in bullet_lower:
                        errors.append(
                            f"{section_name}: bullet cites mistral claim but mentions llama3: {bullet}"
                        )

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair an existing benchmark narrative by incorporating target unused claims."
    )
    parser.add_argument("--selected-claims-json", required=True, help="Path to selected claims JSON.")
    parser.add_argument("--narrative-json", required=True, help="Path to current narrative JSON.")
    parser.add_argument(
        "--target-claim-ids",
        nargs="+",
        required=True,
        help="One or more target unused claim IDs to incorporate if possible.",
    )
    parser.add_argument("--model", default="mistral", help="Ollama model name.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature.")
    parser.add_argument("--num-predict", type=int, default=768, help="Max generated tokens.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selected_claims_path = Path(args.selected_claims_json).resolve()
    narrative_json_path = Path(args.narrative_json).resolve()
    target_claim_ids = [str(x).strip() for x in args.target_claim_ids if str(x).strip()]

    selected_payload = load_json(selected_claims_path)
    narrative_payload = load_json(narrative_json_path)

    selected_claims = get_selected_claims_list(selected_payload)
    claims_by_id = index_claims_by_id(selected_claims)

    missing_targets = [cid for cid in target_claim_ids if cid not in claims_by_id]
    if missing_targets:
        raise ValueError(
            "Some target claim IDs were not found in selected claims:\n"
            + "\n".join(missing_targets)
        )
        
    print("Narrative JSON keys:", sorted(narrative_payload.keys()))

    current_narrative_text = extract_narrative_text(narrative_payload)
    target_claims = format_claim_subset(target_claim_ids, claims_by_id)

    _, next_repair_idx = get_base_artifact_stem(narrative_json_path)
    output_paths = derive_output_paths(narrative_json_path)

    prompt = build_prompt(
        current_narrative_text=current_narrative_text,
        selected_claims=selected_claims,
        target_claim_ids=target_claim_ids,
        target_claims=target_claims,
    )

    repaired_text = call_ollama_generate(
        model=args.model,
        prompt=prompt,
        temperature=args.temperature,
        num_predict=args.num_predict,
    )
    
    missing_targets_in_output = [cid for cid in target_claim_ids if cid not in repaired_text]
    if missing_targets_in_output:
        raise ValueError(
            "Repaired output did not include all target claim IDs:\n"
            + "\n".join(missing_targets_in_output)
        )

    repaired_payload = build_repaired_payload(
        repaired_text=repaired_text,
        original_narrative_payload=narrative_payload,
        selected_payload=selected_payload,
        selected_claims_path=selected_claims_path,
        parent_narrative_json=narrative_json_path,
        repair_iteration=next_repair_idx,
        target_claim_ids=target_claim_ids,
        model=args.model,
        temperature=args.temperature,
        num_predict=args.num_predict,
    )
    
    validation_errors = validate_repaired_sections(
        repaired_payload["sections"],
        target_claim_ids,
    )

    if validation_errors:
        raise ValueError(
            "Repaired output failed validation:\n- " + "\n- ".join(validation_errors)
        )


    repair_metadata = {
        "selected_claims_json": str(selected_claims_path),
        "narrative_json": str(narrative_json_path),
        "parent_narrative_json": str(narrative_json_path),
        "repair_iteration": next_repair_idx,
        "target_claim_ids": target_claim_ids,
        "model": args.model,
        "temperature": args.temperature,
        "num_predict": args.num_predict,
        "output_json": str(output_paths["json"]),
        "output_md": str(output_paths["md"]),
    }

    save_json(repaired_payload, output_paths["json"])
    save_text(repaired_text, output_paths["md"])
    save_json(repair_metadata, output_paths["meta"])

    print(f"Saved repaired narrative JSON: {output_paths['json']}")
    print(f"Saved repaired narrative markdown: {output_paths['md']}")
    print(f"Saved repair metadata JSON: {output_paths['meta']}")


if __name__ == "__main__":
    main()