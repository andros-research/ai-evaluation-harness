from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("benchmarks")
NARRATIVES_DIR = ROOT / "results" / "narratives"
AGG_DIR = ROOT / "results" / "aggregated"
DEFAULT_RESULTS_ROOT = ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full repair → audit → parse → summarize → compare loop."
    )
    parser.add_argument(
        "--selected-claims-json",
        required=True,
        help="Path to selected claims JSON",
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Run-scoped results root, e.g. benchmarks/results/runs/v1_4_4_scaled",
    )
    parser.add_argument(
        "--narrative-json",
        required=True,
        help="Path to parent/original narrative JSON to repair",
    )
    parser.add_argument(
        "--original-artifact",
        required=True,
        help="Artifact name of the original baseline narrative for comparison",
    )
    parser.add_argument(
        "--target-claim-ids",
        nargs="+",
        required=True,
        help="Target unused claim IDs to incorporate",
    )
    parser.add_argument(
        "--repair-strategies",
        nargs="*",
        default=[],
        help="Optional repair strategies to pass through to repair_narrative.py.",
    )
    parser.add_argument(
        "--model",
        default="mistral",
        help="Ollama model for repair generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Repair generation temperature",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=768,
        help="Repair generation max tokens",
    )
    parser.add_argument(
        "--summarize-script",
        default=str(ROOT / "summarize_audits.py"),
        help="Path to summarize_audits.py",
    )
    parser.add_argument(
        "--repair-script",
        default=str(ROOT / "repair_narrative.py"),
        help="Path to repair_narrative.py",
    )
    parser.add_argument(
        "--audit-script",
        default=str(ROOT / "audit_narrative.py"),
        help="Path to audit_narrative.py",
    )
    parser.add_argument(
        "--parse-script",
        default=str(ROOT / "parse_narrative_claims.py"),
        help="Path to parse_narrative_claims.py",
    )
    parser.add_argument(
        "--compare-script",
        default=str(ROOT / "compare_repair_runs.py"),
        help="Path to compare_repair_runs.py",
    )
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    print("\n>>>", " ".join(cmd))
    proc = subprocess.run(cmd, text=True, capture_output=True)

    if proc.stdout:
        print(proc.stdout.strip())
    if proc.stderr:
        print(proc.stderr.strip())

    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}:\n{' '.join(cmd)}"
        )

    return proc


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_repaired_paths_from_metadata(metadata_path: Path) -> dict[str, Path]:
    meta = load_json(metadata_path)

    output_json = Path(meta["output_json"])
    output_md = Path(meta["output_md"])

    repaired_stem = output_json.stem
    parent = output_json.parent

    return {
        "repaired_json": output_json,
        "repaired_md": output_md,
        "repair_metadata_json": metadata_path,
        "repaired_audit_json": parent / f"{repaired_stem}__audit.json",
        "repaired_parsed_json": parent / f"{repaired_stem}__parsed_narrative.json",
        "repaired_artifact_name": repaired_stem,
    }


def newest_repair_metadata_for_parent(narratives_dir: Path, parent_narrative_json: Path) -> Path:
    """
    Find the newest repair metadata file whose parent_narrative_json / narrative_json matches the input narrative.
    """
    candidates = sorted(narratives_dir.glob("*__repair_metadata.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    parent_str = str(parent_narrative_json.resolve())

    for path in candidates:
        try:
            meta = load_json(path)
        except Exception:
            continue

        parent_meta = str(meta.get("parent_narrative_json", "")).strip()
        narrative_meta = str(meta.get("narrative_json", "")).strip()

        if parent_meta == parent_str or narrative_meta == parent_str:
            return path

    raise FileNotFoundError(
        f"Could not find repair metadata for parent narrative: {parent_narrative_json}"
    )


def main() -> None:
    args = parse_args()
    
    results_root = Path(args.results_root).resolve()
    agg_dir = results_root / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)

    selected_claims_json = Path(args.selected_claims_json).resolve()
    narrative_json = Path(args.narrative_json).resolve()
    repair_parent_dir = narrative_json.parent.resolve()

    repair_cmd = [
        "python",
        args.repair_script,
        "--selected-claims-json",
        str(selected_claims_json),
        "--results-root", 
        str(results_root),
        "--narrative-json",
        str(narrative_json),
        "--target-claim-ids",
        *args.target_claim_ids,
        "--repair-strategies",
        *args.repair_strategies,
        "--model",
        args.model,
        "--temperature",
        str(args.temperature),
        "--num-predict",
        str(args.num_predict),
    ]
    run_cmd(repair_cmd)

    metadata_path = newest_repair_metadata_for_parent(repair_parent_dir, narrative_json)
    repaired = infer_repaired_paths_from_metadata(metadata_path)

    audit_cmd = [
        "python",
        args.audit_script,
        "--selected-claims-json",
        str(selected_claims_json),
        "--results-root", 
        str(results_root),
        "--narrative-json",
        str(repaired["repaired_json"]),
    ]
    run_cmd(audit_cmd)

    parse_cmd = [
        "python",
        args.parse_script,
        "--selected-claims-json",
        str(selected_claims_json),
        "--results-root", 
        str(results_root),
        "--narrative-json",
        str(repaired["repaired_json"]),
    ]
    run_cmd(parse_cmd)

    summarize_cmd = [
        "python",
        args.summarize_script,
        "--results-root", 
        str(results_root),
    ]
    run_cmd(summarize_cmd)

    compare_cmd = [
        "python",
        args.compare_script,
        "--results-root", 
        str(results_root),
        "--original-artifact",
        args.original_artifact,
        "--repaired-artifact",
        repaired["repaired_artifact_name"],
    ]
    run_cmd(compare_cmd)

    comparison_summary = load_json((agg_dir / "repair_comparison_summary.json").resolve())

    print("\n=== REPAIR EVAL COMPLETE ===")
    print("Original artifact:", args.original_artifact)
    print("Repaired artifact:", repaired["repaired_artifact_name"])
    print("Repaired narrative JSON:", repaired["repaired_json"])
    print("Repaired audit JSON:", repaired["repaired_audit_json"])
    print("Repaired parsed JSON:", repaired["repaired_parsed_json"])
    print("Repair metadata JSON:", repaired["repair_metadata_json"])
    print("Repair success:", comparison_summary.get("repair_success"))
    print(
        "Used-claim ratio: {before} -> {after} (delta {delta})".format(
            before=comparison_summary.get("claim_coverage", {}).get("before", {}).get("used_claim_ratio"),
            after=comparison_summary.get("claim_coverage", {}).get("after", {}).get("used_claim_ratio"),
            delta=comparison_summary.get("claim_coverage", {}).get("delta", {}).get("used_claim_ratio_delta"),
        )
    )


if __name__ == "__main__":
    main()