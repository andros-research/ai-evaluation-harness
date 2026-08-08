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
- builds a human-readable demo report

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


RUN_SCHEMA_VERSION = "fred_evidence_loop_run_v0_2"
RUN_METHOD = "subprocess_artifact_chain"

DEFAULT_NARRATIVE_MODE = "deterministic"
DEFAULT_NARRATIVE_MODEL = "llama3"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_NARRATIVE_TIMEOUT_S = 600
SUPPORTED_NARRATIVE_MODES = ["deterministic", "llm"]
DEFAULT_PROMPT_VARIANT = "hardened"
DEFAULT_TEMPERATURE = 0.0

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_CONTEXT = (
    REPO_ROOT / "benchmarks" / "data" / "fred_macro_context.json"
)
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "benchmarks" / "results"
DEFAULT_OUTPUT_DIR = DEFAULT_ARTIFACT_ROOT / "fred_runs"


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
DEMO_REPORT_STEP = {
    "step_name": "build_fred_demo_report",
    "script": "benchmarks/build_fred_demo_report.py",
}


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


def build_artifact_paths(artifact_root: Path) -> dict[str, Path]:
    """Build all FRED evidence-loop artifact paths from one root directory."""
    claims_dir = artifact_root / "fred_claims"
    narratives_dir = artifact_root / "fred_narratives"
    audits_dir = artifact_root / "fred_audits"
    repairs_dir = artifact_root / "fred_repairs"
    traceability_dir = artifact_root / "fred_traceability"
    demo_dir = artifact_root / "fred_demo"
    runs_dir = artifact_root / "fred_runs"

    return {
        "artifact_root": artifact_root,

        "claims_dir": claims_dir,
        "claims_json": claims_dir / "fred_claims.json",
        "claims_metadata": claims_dir / "fred_claims_metadata.json",
        "selected_claims_json": claims_dir / "selected_fred_claims.json",
        "selected_claims_metadata": (
            claims_dir / "selected_fred_claims_metadata.json"
        ),

        "narratives_dir": narratives_dir,
        "narrative_md": narratives_dir / "fred_narrative.md",
        "narrative_metadata": (
            narratives_dir / "fred_narrative_metadata.json"
        ),

        "audits_dir": audits_dir,
        "audit_json": audits_dir / "fred_narrative_audit.json",

        "repairs_dir": repairs_dir,
        "repair_plan_json": repairs_dir / "fred_repair_plan.json",

        "traceability_dir": traceability_dir,
        "traceability_json": (
            traceability_dir / "fred_traceability_summary.json"
        ),
        "traceability_metadata": (
            traceability_dir / "fred_traceability_summary_metadata.json"
        ),

        "demo_dir": demo_dir,
        "demo_report_metadata": (
            demo_dir / "fred_demo_report_metadata.json"
        ),

        "runs_dir": runs_dir,
    }


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
    narrative_mode: str,
    narrative_model: str,
    narrative_prompt_variant: str,
    narrative_temperature: float,
    ollama_host: str,
    narrative_timeout_s: int,
    artifact_paths: dict[str, Path],
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
                "--output-dir",
                str(artifact_paths["claims_dir"]),
            ]
        )

    elif step_name == "select_fred_claims":
        command.extend(
            [
                "--input-claims",
                str(artifact_paths["claims_json"]),
                "--output-dir",
                str(artifact_paths["claims_dir"]),
            ]
        )

    elif step_name == "generate_fred_narrative_from_claims":
        command.extend(
            [
                "--input-claims",
                str(artifact_paths["selected_claims_json"]),
                "--output-dir",
                str(artifact_paths["narratives_dir"]),
                "--mode",
                narrative_mode,
            ]
        )

        if narrative_mode == "llm":
            command.extend(
                [
                    "--model",
                    narrative_model,
                    "--ollama-host",
                    ollama_host,
                    "--timeout-s",
                    str(narrative_timeout_s),
                    "--prompt-variant",
                    narrative_prompt_variant,
                    "--temperature",
                    str(narrative_temperature),
                ]
            )

    elif step_name == "audit_fred_narrative":
        command.extend(
            [
                "--narrative",
                str(artifact_paths["narrative_md"]),
                "--selected-claims",
                str(artifact_paths["selected_claims_json"]),
                "--output-dir",
                str(artifact_paths["audits_dir"]),
            ]
        )

    elif step_name == "plan_fred_narrative_repair":
        command.extend(
            [
                "--audit",
                str(artifact_paths["audit_json"]),
                "--narrative",
                str(artifact_paths["narrative_md"]),
                "--selected-claims",
                str(artifact_paths["selected_claims_json"]),
                "--output-dir",
                str(artifact_paths["repairs_dir"]),
            ]
        )

    elif step_name == "build_fred_traceability_summary":
        command.extend(
            [
                "--claims",
                str(artifact_paths["claims_json"]),
                "--selected-claims",
                str(artifact_paths["selected_claims_json"]),
                "--audit",
                str(artifact_paths["audit_json"]),
                "--repair-plan",
                str(artifact_paths["repair_plan_json"]),
                "--output-dir",
                str(artifact_paths["traceability_dir"]),
            ]
        )

    else:
        raise ValueError(f"Unsupported pipeline step: {step_name}")

    return command


def read_json_if_exists(path: Path) -> object | None:
    """Read JSON if the artifact exists; otherwise return None."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_output_summary(
    artifact_paths: dict[str, Path],
    *,
    include_demo_report: bool = True,
) -> dict:
    """Collect summary information from downstream artifacts when available."""
    claims_meta = read_json_if_exists(
        artifact_paths["claims_metadata"]
    )
    selected_meta = read_json_if_exists(
        artifact_paths["selected_claims_metadata"]
    )
    narrative_meta = read_json_if_exists(
        artifact_paths["narrative_metadata"]
    )
    audit_meta = read_json_if_exists(
        artifact_paths["audit_json"]
    )
    repair_meta = read_json_if_exists(
        artifact_paths["repair_plan_json"]
    )
    traceability_meta = read_json_if_exists(
        artifact_paths["traceability_metadata"]
    )

    demo_report_meta = (
        read_json_if_exists(artifact_paths["demo_report_metadata"])
        if include_demo_report
        else None
    )

    return {
        "claims": {
            "n_claims": (
                claims_meta.get("n_claims")
                if isinstance(claims_meta, dict)
                else None
            ),
            "series_included": (
                claims_meta.get("series_included")
                if isinstance(claims_meta, dict)
                else None
            ),
            "comparison_window": (
                claims_meta.get("comparison_window")
                if isinstance(claims_meta, dict)
                else None
            ),
        },
        "selected_claims": {
            "n_selected_claims": (
                selected_meta.get("n_selected_claims")
                if isinstance(selected_meta, dict)
                else None
            ),
            "selection_method": (
                selected_meta.get("selection_method")
                if isinstance(selected_meta, dict)
                else None
            ),
        },
        "narrative": {
            "n_selected_claims": (
                narrative_meta.get("n_selected_claims")
                if isinstance(narrative_meta, dict)
                else None
            ),
            "generation_method": (
                narrative_meta.get("generation_method")
                if isinstance(narrative_meta, dict)
                else None
            ),
            "citation_validation": (
                narrative_meta.get("citation_validation")
                if isinstance(narrative_meta, dict)
                else None
            ),
        },
        "audit": {
            "audit_pass": (
                audit_meta.get("audit_pass")
                if isinstance(audit_meta, dict)
                else None
            ),
            "n_bullets": (
                audit_meta.get("n_bullets")
                if isinstance(audit_meta, dict)
                else None
            ),
            "n_citations": (
                audit_meta.get("n_citations")
                if isinstance(audit_meta, dict)
                else None
            ),
            "errors": (
                audit_meta.get("errors")
                if isinstance(audit_meta, dict)
                else None
            ),
        },
        "repair": {
            "repair_needed": (
                repair_meta.get("repair_needed")
                if isinstance(repair_meta, dict)
                else None
            ),
            "n_repair_actions": (
                repair_meta.get("n_repair_actions")
                if isinstance(repair_meta, dict)
                else None
            ),
        },
        "traceability": {
            "n_traceability_rows": (
                traceability_meta.get("n_traceability_rows")
                if isinstance(traceability_meta, dict)
                else None
            ),
            "n_cited_claims": (
                traceability_meta.get("n_cited_claims")
                if isinstance(traceability_meta, dict)
                else None
            ),
        },
        "demo_report": {
            "overall_ok": (
                demo_report_meta.get("overall_ok")
                if isinstance(demo_report_meta, dict)
                else None
            ),
            "narrative_mode": (
                demo_report_meta.get("narrative_mode")
                if isinstance(demo_report_meta, dict)
                else None
            ),
            "audit_pass": (
                demo_report_meta.get("audit_pass")
                if isinstance(demo_report_meta, dict)
                else None
            ),
            "repair_needed": (
                demo_report_meta.get("repair_needed")
                if isinstance(demo_report_meta, dict)
                else None
            ),
        },
    }
    

def build_run_metadata(
    *,
    run_id: str,
    run_started_at: str,
    run_finished_at: str,
    input_context: Path,
    artifact_root: Path,
    comparison_window: str,
    narrative_mode: str,
    narrative_model: str,
    narrative_prompt_variant: str,
    narrative_temperature: float,
    ollama_host: str,
    narrative_timeout_s: int,
    stop_on_failure: bool,
    step_results: list[dict],
    output_summary: dict,
) -> dict:
    """Build run metadata from current step results."""
    overall_ok = all(step["ok"] for step in step_results)
    completed_steps = [step["step_name"] for step in step_results if step["ok"]]
    failed_steps = [step["step_name"] for step in step_results if not step["ok"]]

    return {
        "run_schema_version": RUN_SCHEMA_VERSION,
        "run_method": RUN_METHOD,
        "run_id": run_id,
        "run_started_at": run_started_at,
        "run_finished_at": run_finished_at,
        "input_context": str(input_context),
        "artifact_root": str(artifact_root),
        "comparison_window": comparison_window,
        "narrative_mode": narrative_mode,
        "narrative_model": narrative_model if narrative_mode == "llm" else None,
        "narrative_prompt_variant": (
            narrative_prompt_variant
            if narrative_mode == "llm"
            else None
        ),
        "narrative_temperature": (
            narrative_temperature
            if narrative_mode == "llm"
            else None
        ),
        "ollama_host": ollama_host if narrative_mode == "llm" else None,
        "narrative_timeout_s": narrative_timeout_s if narrative_mode == "llm" else None,
        "overall_ok": overall_ok,
        "stop_on_failure": stop_on_failure,
        "n_steps": len(PIPELINE_STEPS) + 1,
        "n_steps_run": len(step_results),
        "completed_steps": completed_steps,
        "failed_steps": failed_steps,
        "step_results": step_results,
        "output_summary": output_summary,
    }    


def run_fred_evidence_loop(
    *,
    input_context: Path,
    comparison_window: str,
    artifact_root: Path,
    output_dir: Path,
    narrative_mode: str,
    narrative_model: str,
    narrative_prompt_variant: str,
    narrative_temperature: float,
    ollama_host: str,
    narrative_timeout_s: int,
    stop_on_failure: bool = True,
) -> None:
    """Run the full FRED evidence loop and write run metadata."""
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = build_artifact_paths(artifact_root)

    run_id = f"fred_evidence_loop_{safe_timestamp()}"
    run_started_at = utc_now_iso()

    step_results: list[dict] = []

    print(f"Starting FRED evidence loop: {run_id}")
    print(f"input_context={input_context}")
    print(f"comparison_window={comparison_window}")
    print(f"artifact_root={artifact_root}")
    print(f"narrative_mode={narrative_mode}")
    if narrative_mode == "llm":
        print(f"narrative_model={narrative_model}")
        print(f"ollama_host={ollama_host}")
        print(
            "narrative_prompt_variant="
            f"{narrative_prompt_variant}"
        )
        print(
            "narrative_temperature="
            f"{narrative_temperature}"
        )

    for step in PIPELINE_STEPS:
        step_name = step["step_name"]
        script = step["script"]

        command = build_step_command(
            step_name=step_name,
            script=script,
            input_context=input_context,
            comparison_window=comparison_window,
            narrative_mode=narrative_mode,
            narrative_model=narrative_model,
            narrative_prompt_variant=narrative_prompt_variant,
            narrative_temperature=narrative_temperature,
            ollama_host=ollama_host,
            narrative_timeout_s=narrative_timeout_s,
            artifact_paths=artifact_paths,
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

    run_metadata_path = output_dir / f"{run_id}.json"
    latest_metadata_path = output_dir / "latest_fred_evidence_loop_run.json"

    # First write current run metadata after the core evidence-loop steps.
    # This lets the demo report read the current run, not the previous run.
    output_summary = (
        collect_output_summary(
            artifact_paths,
            include_demo_report=False,
        )
        if overall_ok
        else {}
    )

    run_metadata = build_run_metadata(
        run_id=run_id,
        run_started_at=run_started_at,
        run_finished_at=run_finished_at,
        input_context=input_context,
        artifact_root=artifact_root,
        comparison_window=comparison_window,
        narrative_mode=narrative_mode,
        narrative_model=narrative_model,
        narrative_prompt_variant=narrative_prompt_variant,
        narrative_temperature=narrative_temperature,
        ollama_host=ollama_host,
        narrative_timeout_s=narrative_timeout_s,
        stop_on_failure=stop_on_failure,
        step_results=step_results,
        output_summary=output_summary,
    )

    write_json(run_metadata_path, run_metadata)
    write_json(latest_metadata_path, run_metadata)

    # Then build the screen-friendly demo report from the current run metadata.
    if overall_ok:
        step_name = DEMO_REPORT_STEP["step_name"]
        script = DEMO_REPORT_STEP["script"]
        command = [
            sys.executable,
            script,
            "--run-metadata",
            str(latest_metadata_path),
            "--claims",
            str(artifact_paths["claims_json"]),
            "--selected-claims",
            str(artifact_paths["selected_claims_json"]),
            "--narrative",
            str(artifact_paths["narrative_md"]),
            "--narrative-metadata",
            str(artifact_paths["narrative_metadata"]),
            "--audit",
            str(artifact_paths["audit_json"]),
            "--repair-plan",
            str(artifact_paths["repair_plan_json"]),
            "--traceability",
            str(artifact_paths["traceability_json"]),
            "--output-dir",
            str(artifact_paths["demo_dir"]),
        ]

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

    # Rebuild metadata after demo report step and include demo summary.
    run_finished_at = utc_now_iso()
    overall_ok = all(step["ok"] for step in step_results)
    output_summary = (
        collect_output_summary(artifact_paths)
        if overall_ok
        else {}
    )

    run_metadata = build_run_metadata(
        run_id=run_id,
        run_started_at=run_started_at,
        run_finished_at=run_finished_at,
        input_context=input_context,
        artifact_root=artifact_root,
        comparison_window=comparison_window,
        narrative_mode=narrative_mode,
        narrative_model=narrative_model,
        narrative_prompt_variant=narrative_prompt_variant,
        narrative_temperature=narrative_temperature,
        ollama_host=ollama_host,
        narrative_timeout_s=narrative_timeout_s,
        stop_on_failure=stop_on_failure,
        step_results=step_results,
        output_summary=output_summary,
    )

    write_json(run_metadata_path, run_metadata)
    write_json(latest_metadata_path, run_metadata)

    completed_steps = [step["step_name"] for step in step_results if step["ok"]]
    failed_steps = [step["step_name"] for step in step_results if not step["ok"]]

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
        "--narrative-mode",
        default=DEFAULT_NARRATIVE_MODE,
        choices=SUPPORTED_NARRATIVE_MODES,
        help="Narrative generation mode to pass to generate_fred_narrative_from_claims.py.",
    )
    parser.add_argument(
        "--narrative-model",
        default=DEFAULT_NARRATIVE_MODEL,
        help="Local Ollama model to use when --narrative-mode llm.",
    )
    parser.add_argument(
        "--narrative-prompt-variant",
        default=DEFAULT_PROMPT_VARIANT,
        choices=[
            "weak",
            "intermediate",
            "hardened",
        ],
        help=(
            "Prompt contract variant for "
            "LLM narrative generation."
        ),
    )

    parser.add_argument(
        "--narrative-temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=(
            "Sampling temperature for "
            "LLM narrative generation."
        ),
    )
    parser.add_argument(
        "--ollama-host",
        default=DEFAULT_OLLAMA_HOST,
        help="Ollama host URL when --narrative-mode llm.",
    )
    parser.add_argument(
        "--narrative-timeout-s",
        type=int,
        default=DEFAULT_NARRATIVE_TIMEOUT_S,
        help="Timeout in seconds for LLM narrative generation.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help=(
            "Root directory for all FRED evidence-loop artifacts. "
            "Defaults to benchmarks/results."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional override for run metadata output. "
            "Defaults to <artifact-root>/fred_runs."
        ),
    )
    parser.add_argument(
        "--no-stop-on-failure",
        action="store_true",
        help="Continue running downstream steps even if a prior step fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.narrative_temperature < 0:
        raise ValueError(
            "--narrative-temperature must be >= 0."
        )

    artifact_root = args.artifact_root
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else artifact_root / "fred_runs"
    )

    run_fred_evidence_loop(
        input_context=args.input_context,
        comparison_window=args.comparison_window,
        artifact_root=artifact_root,
        output_dir=output_dir,
        narrative_mode=args.narrative_mode,
        narrative_model=args.narrative_model,
        narrative_prompt_variant=(
            args.narrative_prompt_variant
        ),
        narrative_temperature=(
            args.narrative_temperature
        ),
        ollama_host=args.ollama_host,
        narrative_timeout_s=args.narrative_timeout_s,
        stop_on_failure=not args.no_stop_on_failure,
    )


if __name__ == "__main__":
    main()