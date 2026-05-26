#!/usr/bin/env python3
"""
Run the full FRED evidence loop.

v1.6.6 scaffold:
- builds deterministic FRED claims
- selects narrative-eligible claims
- generates a claim-cited narrative
- audits the narrative
- plans repairs from the audit
- builds a traceability summary
- writes one run-level metadata artifact

This runner coordinates existing scripts. It does not replace the individual
pipeline steps, which remain independently runnable and auditable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


RUN_SCHEMA_VERSION = "fred_evidence_loop_run_v0_1"
RUN_METHOD = "subprocess_artifact_chain"

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_CONTEXT = REPO_ROOT / "benchmarks" / "data" / "fred_macro_context.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "results" / "fred_runs"


PIPELINE_STEPS = [
    {
        "step_name": "build_fred_claims",
        "script": "benchmarks/build_fred_claims.py",
    },
    {
        "step_name": "select_fred_claims",
        "script": "benchmarks/select_fred_claims.py",
    },
    {
        "step_name": "generate_fred_narrative_from_claims",
        "script": "benchmarks/generate_fred_narrative_from_claims.py",
    },
    {
        "step_name": "audit_fred_narrative",
        "script": "benchmarks/audit_fred_narrative.py",
    },
    {
        "step_name": "plan_fred_narrative_repair",
        "script": "benchmarks/plan_fred_narrative_repair.py",
    },
    {
        "step_name": "build_fred_traceability_summary",
        "script": "benchmarks/build_fred_traceability_summary.py",
    },
]


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def safe_timestamp() -> str:
    """Return a filesystem-safe UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: object) -> None:
    """Write JSON with stable formatting."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_command(command: list[str], cwd: Path) -> dict:
    """Run one subprocess command and capture execution metadata."""
    started_at = utc_now_iso()

    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )

    finished_at = utc_now_iso()

    return {
        "command": command,
        "started_at": started_at,
        "finished_at": finished_at,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "ok": result.returncode == 0,
    }


def build_step_command(
    *,
    step_name: str,
    script: str,
    input_context: Path,
    comparison_window: str,
) -> list[str]:
    """Build command for a named pipeline step."""
    command = [sys.executable, script]

    if step_name == "build_fred_claims":
        command.extend(
            [
                "--input-context",
                str(input_context),
                "--comparison-window",
                comparison_window,
            ]
        )

    return command


def read_json_if_exists(path: Path) -> object | None:
    """Read JSON if the artifact exists; otherwise return None."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_output_summary() -> dict:
    """Collect summary information from downstream artifacts when available."""
    claims_meta = read_json_if_exists(
        REPO_ROOT / "benchmarks" / "results" / "fred_claims" / "fred_claims_metadata.json"
    )
    selected_meta = read_json_if_exists(
        REPO_ROOT
        / "benchmarks"
        / "results"
        / "fred_claims"
        / "selected_fred_claims_metadata.json"
    )
    narrative_meta = read_json_if_exists(
        REPO_ROOT
        / "benchmarks"
        / "results"
        / "fred_narratives"
        / "fred_narrative_metadata.json"
    )
    audit_meta = read_json_if_exists(
        REPO_ROOT
        / "benchmarks"
        / "results"
        / "fred_audits"
        / "fred_narrative_audit.json"
    )
    repair_meta = read_json_if_exists(
        REPO_ROOT
        / "benchmarks"
        / "results"
        / "fred_repairs"
        / "fred_repair_plan.json"
    )
    traceability_meta = read_json_if_exists(
        REPO_ROOT
        / "benchmarks"
        / "results"
        / "fred_traceability"
        / "fred_traceability_summary_metadata.json"
    )

    return {
        "claims": {
            "n_claims": claims_meta.get("n_claims") if isinstance(claims_meta, dict) else None,
            "series_included": claims_meta.get("series_included") if isinstance(claims_meta, dict) else None,
            "comparison_window": claims_meta.get("comparison_window") if isinstance(claims_meta, dict) else None,
        },
        "selected_claims": {
            "n_selected_claims": selected_meta.get("n_selected_claims")
            if isinstance(selected_meta, dict)
            else None,
            "selection_method": selected_meta.get("selection_method")
            if isinstance(selected_meta, dict)
            else None,
        },
        "narrative": {
            "n_selected_claims": narrative_meta.get("n_selected_claims")
            if isinstance(narrative_meta, dict)
            else None,
            "generation_method": narrative_meta.get("generation_method")
            if isinstance(narrative_meta, dict)
            else None,
            "citation_validation": narrative_meta.get("citation_validation")
            if isinstance(narrative_meta, dict)
            else None,
        },
        "audit": {
            "audit_pass": audit_meta.get("audit_pass") if isinstance(audit_meta, dict) else None,
            "n_bullets": audit_meta.get("n_bullets") if isinstance(audit_meta, dict) else None,
            "n_citations": audit_meta.get("n_citations") if isinstance(audit_meta, dict) else None,
            "errors": audit_meta.get("errors") if isinstance(audit_meta, dict) else None,
        },
        "repair": {
            "repair_needed": repair_meta.get("repair_needed")
            if isinstance(repair_meta, dict)
            else None,
            "n_repair_actions": repair_meta.get("n_repair_actions")
            if isinstance(repair_meta, dict)
            else None,
        },
        "traceability": {
            "n_traceability_rows": traceability_meta.get("n_traceability_rows")
            if isinstance(traceability_meta, dict)
            else None,
            "n_cited_claims": traceability_meta.get("n_cited_claims")
            if isinstance(traceability_meta, dict)
            else None,
        },
    }


def run_fred_evidence_loop(
    *,
    input_context: Path,
    comparison_window: str,
    output_dir: Path,
    stop_on_failure: bool = True,
) -> None:
    """Run the full FRED evidence loop and write run metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"fred_evidence_loop_{safe_timestamp()}"
    run_started_at = utc_now_iso()

    step_results: list[dict] = []

    print(f"Starting FRED evidence loop: {run_id}")
    print(f"input_context={input_context}")
    print(f"comparison_window={comparison_window}")

    for step in PIPELINE_STEPS:
        step_name = step["step_name"]
        script = step["script"]

        command = build_step_command(
            step_name=step_name,
            script=script,
            input_context=input_context,
            comparison_window=comparison_window,
        )

        print(f"\n=== Running step: {step_name} ===")
        print(" ".join(command))

        result = run_command(command, cwd=REPO_ROOT)
        step_results.append(
            {
                "step_name": step_name,
                "script": script,
                **result,
            }
        )

        if result["stdout"]:
            print(result["stdout"], end="" if result["stdout"].endswith("\n") else "\n")

        if result["stderr"]:
            print(result["stderr"], end="" if result["stderr"].endswith("\n") else "\n")

        if not result["ok"]:
            print(f"Step failed: {step_name} returncode={result['returncode']}")
            if stop_on_failure:
                break

    run_finished_at = utc_now_iso()
    overall_ok = all(step["ok"] for step in step_results)
    completed_steps = [step["step_name"] for step in step_results if step["ok"]]
    failed_steps = [step["step_name"] for step in step_results if not step["ok"]]

    output_summary = collect_output_summary() if overall_ok else {}

    run_metadata = {
        "run_schema_version": RUN_SCHEMA_VERSION,
        "run_method": RUN_METHOD,
        "run_id": run_id,
        "run_started_at": run_started_at,
        "run_finished_at": run_finished_at,
        "input_context": str(input_context),
        "comparison_window": comparison_window,
        "overall_ok": overall_ok,
        "stop_on_failure": stop_on_failure,
        "n_steps": len(PIPELINE_STEPS),
        "n_steps_run": len(step_results),
        "completed_steps": completed_steps,
        "failed_steps": failed_steps,
        "step_results": step_results,
        "output_summary": output_summary,
    }

    run_metadata_path = output_dir / f"{run_id}.json"
    latest_metadata_path = output_dir / "latest_fred_evidence_loop_run.json"

    write_json(run_metadata_path, run_metadata)
    write_json(latest_metadata_path, run_metadata)

    print("\nWrote FRED evidence loop run metadata:")
    print(f"  {run_metadata_path}")
    print(f"  {latest_metadata_path}")
    print(f"overall_ok={overall_ok}")
    print(f"completed_steps={len(completed_steps)}")
    print(f"failed_steps={len(failed_steps)}")

    if not overall_ok:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full FRED evidence loop."
    )
    parser.add_argument(
        "--input-context",
        type=Path,
        default=DEFAULT_INPUT_CONTEXT,
        help="Structured FRED macro context JSON.",
    )
    parser.add_argument(
        "--comparison-window",
        default="12m",
        choices=["6m", "12m", "24m"],
        help="Comparison window to use for FRED claims.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where run metadata artifacts will be written.",
    )
    parser.add_argument(
        "--no-stop-on-failure",
        action="store_true",
        help="Continue running downstream steps even if a prior step fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_fred_evidence_loop(
        input_context=args.input_context,
        comparison_window=args.comparison_window,
        output_dir=args.output_dir,
        stop_on_failure=not args.no_stop_on_failure,
    )


if __name__ == "__main__":
    main()