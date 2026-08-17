#!/usr/bin/env python3
"""
Run one frozen FRED evidence context across multiple narrative modes/models.

Responsibilities:
- load a JSON comparison configuration
- allocate an isolated experiment/batch directory
- snapshot the input context once
- run the existing FRED evidence loop in isolated artifact roots
- continue after individual run failures
- preserve orchestration logs
- write one incremental comparison manifest
- build normalized comparison summary artifacts after execution
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic


COMPARISON_SCHEMA_VERSION = (
    "fred_model_comparison_v0_2"
)

SUPPORTED_MODES = {"deterministic", "llm"}
SUPPORTED_COMPARISON_WINDOWS = {"6m", "12m", "24m"}

SAFE_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")

REPO_ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_RUNNER = (
    REPO_ROOT
    / "benchmarks"
    / "run_fred_evidence_loop.py"
)

SUMMARY_BUILDER = (
    REPO_ROOT
    / "benchmarks"
    / "build_fred_model_comparison_summary.py"
)

DEFAULT_CONFIG = (
    REPO_ROOT
    / "benchmarks"
    / "configs"
    / "fred_model_comparison_smoke.json"
)

DEFAULT_RESULTS_ROOT = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "model_comparisons"
)

DEFAULT_PROMPT_VARIANT = "hardened"
SUPPORTED_PROMPT_VARIANTS = {
    "weak",
    "intermediate",
    "hardened",
}
DEFAULT_TEMPERATURE = 0.0


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json_if_exists(path: Path) -> dict | None:
    """Read a JSON object if it exists."""
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def resolve_repo_path(path_value: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def allocate_comparison_directory(
    *,
    results_root: Path,
    comparison_family_id: str,
) -> tuple[str, Path, int]:
    """
    Allocate a unique comparison directory.

    The first batch uses the unsuffixed family ID.
    Later batches use __batch_NNN suffixes.
    """
    results_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    batch_number = 1

    while True:
        if batch_number == 1:
            comparison_id = (
                comparison_family_id
            )
        else:
            comparison_id = (
                f"{comparison_family_id}"
                f"__batch_{batch_number:03d}"
            )

        comparison_dir = (
            results_root
            / comparison_id
        )

        try:
            comparison_dir.mkdir(
                parents=False,
                exist_ok=False,
            )

            return (
                comparison_id,
                comparison_dir,
                batch_number,
            )

        except FileExistsError:
            batch_number += 1


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def git_commit() -> str | None:
    """Return the current Git commit when available."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        return None

    return result.stdout.strip() or None


def validate_label(
    label: str,
    *,
    field_name: str,
) -> str:
    """Validate a comparison or run label for safe directory use."""
    if not label or not SAFE_LABEL_PATTERN.fullmatch(label):
        raise ValueError(
            f"{field_name} must match "
            f"{SAFE_LABEL_PATTERN.pattern}: {label!r}"
        )

    return label


def load_config(path: Path) -> dict:
    """Load and validate a model-comparison configuration."""
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError(
            "Comparison config must be a JSON object."
        )

    required_fields = {
        "comparison_id",
        "input_context",
        "comparison_window",
        "runs",
    }

    missing = sorted(required_fields - payload.keys())
    if missing:
        raise ValueError(
            f"Missing required config fields: {missing}"
        )

    validate_label(
        str(payload["comparison_id"]),
        field_name="comparison_id",
    )

    comparison_window = str(payload["comparison_window"])
    if comparison_window not in SUPPORTED_COMPARISON_WINDOWS:
        raise ValueError(
            "comparison_window must be one of "
            f"{sorted(SUPPORTED_COMPARISON_WINDOWS)}"
        )

    runs = payload["runs"]
    if not isinstance(runs, list) or not runs:
        raise ValueError(
            "runs must be a non-empty JSON array."
        )

    seen_labels: set[str] = set()

    for index, run in enumerate(runs, start=1):
        if not isinstance(run, dict):
            raise ValueError(
                f"runs[{index}] must be a JSON object."
            )

        label = validate_label(
            str(run.get("label", "")),
            field_name=f"runs[{index}].label",
        )

        if label in seen_labels:
            raise ValueError(
                f"Duplicate run label: {label}"
            )

        seen_labels.add(label)

        mode = str(run.get("mode", ""))
        if mode not in SUPPORTED_MODES:
            raise ValueError(
                f"runs[{index}].mode must be one of "
                f"{sorted(SUPPORTED_MODES)}"
            )

        if mode == "llm" and not run.get("model"):
            raise ValueError(
                f"runs[{index}].model is required "
                "for llm mode."
            )

        prompt_variant = run.get(
            "prompt_variant",
            DEFAULT_PROMPT_VARIANT,
        )

        temperature = run.get(
            "temperature",
            DEFAULT_TEMPERATURE,
        )

        if mode == "llm":
            if (
                prompt_variant
                not in SUPPORTED_PROMPT_VARIANTS
            ):
                raise ValueError(
                    f"runs[{index}].prompt_variant "
                    "must be one of "
                    f"{sorted(SUPPORTED_PROMPT_VARIANTS)}"
                )

            if (
                isinstance(temperature, bool)
                or not isinstance(
                    temperature,
                    (int, float),
                )
                or temperature < 0
            ):
                raise ValueError(
                    f"runs[{index}].temperature "
                    "must be a non-negative number."
                )

        repetitions = run.get("repetitions", 1)
        if (
            not isinstance(repetitions, int)
            or repetitions < 1
        ):
            raise ValueError(
                f"runs[{index}].repetitions must be "
                "a positive integer."
            )

    timeout_s = payload.get(
        "narrative_timeout_s",
        600,
    )

    if not isinstance(timeout_s, int) or timeout_s < 1:
        raise ValueError(
            "narrative_timeout_s must be "
            "a positive integer."
        )

    return payload


def expand_run_specs(config: dict) -> list[dict]:
    """Expand configured runs into one record per repetition."""
    specs: list[dict] = []

    for run in config["runs"]:
        repetitions = run.get("repetitions", 1)

        for repetition in range(
            1,
            repetitions + 1,
        ):
            specs.append(
                {
                    "label": run["label"],
                    "mode": run["mode"],
                    "model": run.get("model"),
                    "prompt_variant": (
                        run.get(
                            "prompt_variant",
                            DEFAULT_PROMPT_VARIANT,
                        )
                        if run["mode"] == "llm"
                        else None
                    ),
                    "temperature": (
                        float(
                            run.get(
                                "temperature",
                                DEFAULT_TEMPERATURE,
                            )
                        )
                        if run["mode"] == "llm"
                        else None
                    ),
                    "repetition": repetition,
                }
            )

    return specs


def build_evidence_command(
    *,
    config: dict,
    frozen_context: Path,
    artifact_root: Path,
    run_spec: dict,
) -> list[str]:
    """Build one invocation of the existing evidence-loop runner."""
    command = [
        sys.executable,
        str(EVIDENCE_RUNNER),
        "--input-context",
        str(frozen_context),
        "--comparison-window",
        str(config["comparison_window"]),
        "--artifact-root",
        str(artifact_root),
        "--narrative-mode",
        str(run_spec["mode"]),
    ]

    if run_spec["mode"] == "llm":
        command.extend(
            [
                "--narrative-model",
                str(run_spec["model"]),
                "--narrative-prompt-variant",
                str(
                    run_spec[
                        "prompt_variant"
                    ]
                ),
                "--narrative-temperature",
                str(
                    run_spec[
                        "temperature"
                    ]
                ),
                "--ollama-host",
                str(
                    config.get(
                        "ollama_host",
                        "http://127.0.0.1:11434",
                    )
                ),
                "--narrative-timeout-s",
                str(
                    config.get(
                        "narrative_timeout_s",
                        600,
                    )
                ),
            ]
        )

    return command


def run_streaming_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
) -> dict:
    """Run a subprocess, stream output, and preserve a log."""
    started_at = utc_now_iso()
    started_clock = monotonic()

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        with log_path.open(
            "w",
            encoding="utf-8",
        ) as log_handle:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )

            if process.stdout is None:
                raise RuntimeError(
                    "Unable to capture comparison-run output."
                )

            for line in process.stdout:
                print(line, end="")
                log_handle.write(line)
                log_handle.flush()

            returncode = process.wait()

    except OSError as exc:
        message = (
            f"Unable to start command: {exc}\n"
        )
        print(message, end="")

        log_path.write_text(
            message,
            encoding="utf-8",
        )

        returncode = 127

    finished_at = utc_now_iso()
    elapsed_seconds = round(
        monotonic() - started_clock,
        3,
    )

    return {
        "command": command,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": elapsed_seconds,
        "returncode": returncode,
        "process_ok": returncode == 0,
        "log_path": str(log_path),
    }


def summarize_manifest_runs(
    run_results: list[dict],
    expected_runs: int,
) -> dict:
    """Build lightweight orchestration counts."""
    return {
        "n_expected_runs": expected_runs,
        "n_attempted_runs": len(run_results),
        "n_process_ok": sum(
            result.get("process_ok") is True
            for result in run_results
        ),
        "n_process_failed": sum(
            result.get("process_ok") is False
            for result in run_results
        ),
        "n_run_completed": sum(
            result.get("run_completed") is True
            for result in run_results
        ),
        "n_audit_pass": sum(
            result.get("audit_pass") is True
            for result in run_results
        ),
        "n_repair_needed": sum(
            result.get("repair_needed") is True
            for result in run_results
        ),
        "n_accepted_output": sum(
            result.get("accepted_output") is True
            for result in run_results
        ),
    }


def build_run_result(
    *,
    run_spec: dict,
    artifact_root: Path,
    process_result: dict,
) -> dict:
    """Combine orchestration and inner evidence-loop metadata."""
    metadata_path = (
        artifact_root
        / "fred_runs"
        / "latest_fred_evidence_loop_run.json"
    )

    inner_run = read_json_if_exists(
        metadata_path
    )

    output_summary = (
        inner_run.get("output_summary", {})
        if isinstance(inner_run, dict)
        else {}
    )

    audit_summary = (
        output_summary.get("audit", {})
        if isinstance(output_summary, dict)
        else {}
    )

    repair_summary = (
        output_summary.get("repair", {})
        if isinstance(output_summary, dict)
        else {}
    )

    failed_steps = (
        inner_run.get("failed_steps", [])
        if isinstance(inner_run, dict)
        else []
    )

    failure_stage = None

    if failed_steps:
        failure_stage = failed_steps[0]
    elif not process_result["process_ok"]:
        failure_stage = "evidence_runner_process"

    audit_pass = (
        audit_summary.get("audit_pass")
        if isinstance(audit_summary, dict)
        else None
    )

    repair_needed = (
        repair_summary.get("repair_needed")
        if isinstance(repair_summary, dict)
        else None
    )

    return {
        "run_label": run_spec["label"],
        "mode": run_spec["mode"],
        "model": run_spec["model"],
        "prompt_variant": (
            run_spec["prompt_variant"]
        ),
        "temperature": (
            run_spec["temperature"]
        ),
        "repetition": run_spec["repetition"],
        "artifact_root": str(artifact_root),
        "run_metadata_path": (
            str(metadata_path)
            if metadata_path.exists()
            else None
        ),
        "inner_run_id": (
            inner_run.get("run_id")
            if isinstance(inner_run, dict)
            else None
        ),
        "process_ok": process_result["process_ok"],
        "returncode": process_result["returncode"],
        "run_completed": (
            inner_run.get("overall_ok") is True
            if isinstance(inner_run, dict)
            else False
        ),
        "audit_pass": audit_pass,
        "repair_needed": repair_needed,
        "accepted_output": (
            audit_pass is True
            and repair_needed is False
        ),
        "failed_steps": failed_steps,
        "failure_stage": failure_stage,
        "started_at": process_result["started_at"],
        "finished_at": process_result["finished_at"],
        "elapsed_seconds": (
            process_result["elapsed_seconds"]
        ),
        "command": process_result["command"],
        "orchestration_log": (
            process_result["log_path"]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a frozen FRED context across "
            "multiple models."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="JSON model-comparison configuration.",
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=(
            "Root directory for "
            "model-comparison experiments."
        ),
    )

    parser.add_argument(
        "--comparison-id",
        help="Optional comparison ID override.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = resolve_repo_path(
        args.config
    )

    results_root = resolve_repo_path(
        args.results_root
    )

    config = load_config(
        config_path
    )

    comparison_family_id = validate_label(
        args.comparison_id
        or str(config["comparison_id"]),
        field_name="comparison_id",
    )

    (
        comparison_id,
        comparison_dir,
        batch_number,
    ) = allocate_comparison_directory(
        results_root=results_root,
        comparison_family_id=(
            comparison_family_id
        ),
    )

    print(
        "Allocated comparison experiment:"
    )
    print(
        f"  family_id="
        f"{comparison_family_id}"
    )
    print(
        f"  comparison_id="
        f"{comparison_id}"
    )
    print(
        f"  batch_number="
        f"{batch_number}"
    )
    print(
        f"  comparison_dir="
        f"{comparison_dir}"
    )

    source_context = resolve_repo_path(
        config["input_context"]
    )

    if not source_context.exists():
        raise FileNotFoundError(
            f"Input context not found: {source_context}"
        )

    frozen_input_dir = (
        comparison_dir
        / "inputs"
    )

    frozen_input_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frozen_context = (
        frozen_input_dir
        / source_context.name
    )

    shutil.copy2(
        source_context,
        frozen_context,
    )

    normalized_config = {
        **config,
        "comparison_family_id": (
            comparison_family_id
        ),
        "comparison_id": comparison_id,
        "batch_number": batch_number,
        "source_config": str(config_path),
        "source_input_context": str(
            source_context
        ),
        "frozen_input_context": str(
            frozen_context
        ),
    }

    write_json(
        comparison_dir
        / "comparison_config.json",
        normalized_config,
    )

    run_specs = expand_run_specs(
        config
    )

    manifest_path = (
        comparison_dir
        / "comparison_manifest.json"
    )

    manifest = {
        "comparison_schema_version": (
            COMPARISON_SCHEMA_VERSION
        ),
        "comparison_family_id": (
            comparison_family_id
        ),
        "comparison_id": comparison_id,
        "batch_number": batch_number,
        "status": "running",
        "comparison_started_at": utc_now_iso(),
        "comparison_finished_at": None,
        "comparison_window": (
            config["comparison_window"]
        ),
        "source_config": str(config_path),
        "frozen_input_context": str(
            frozen_context
        ),
        "context_sha256": sha256_file(
            frozen_context
        ),
        "git_commit": git_commit(),
        "run_results": [],
        "summary": summarize_manifest_runs(
            [],
            len(run_specs),
        ),
    }

    write_json(
        manifest_path,
        manifest,
    )

    for index, run_spec in enumerate(
        run_specs,
        start=1,
    ):
        repetition_dir = (
            f"repetition_"
            f"{run_spec['repetition']:03d}"
        )

        artifact_root = (
            comparison_dir
            / "runs"
            / run_spec["label"]
            / repetition_dir
        )

        log_path = (
            artifact_root
            / "orchestration.log"
        )

        model_display = (
            run_spec["model"]
            or "deterministic"
        )

        controls_display = ""

        if run_spec["mode"] == "llm":
            controls_display = (
                f" | prompt="
                f"{run_spec['prompt_variant']}"
                f" | temp="
                f"{run_spec['temperature']}"
            )

        print()
        print("=" * 78)
        print(
            f"Comparison run "
            f"{index}/{len(run_specs)}: "
            f"{run_spec['label']} | "
            f"{model_display} | "
            f"{repetition_dir}"
            f"{controls_display}"
        )
        print(
            f"artifact_root={artifact_root}"
        )
        print("=" * 78)

        command = build_evidence_command(
            config=config,
            frozen_context=frozen_context,
            artifact_root=artifact_root,
            run_spec=run_spec,
        )

        try:
            process_result = (
                run_streaming_command(
                    command,
                    cwd=REPO_ROOT,
                    log_path=log_path,
                )
            )

            run_result = build_run_result(
                run_spec=run_spec,
                artifact_root=artifact_root,
                process_result=process_result,
            )

        except Exception as exc:
            run_result = {
                "run_label": run_spec["label"],
                "mode": run_spec["mode"],
                "model": run_spec["model"],
                "prompt_variant": (
                    run_spec["prompt_variant"]
                ),
                "temperature": (
                    run_spec["temperature"]
                ),
                "repetition": (
                    run_spec["repetition"]
                ),
                "artifact_root": str(
                    artifact_root
                ),
                "process_ok": False,
                "returncode": None,
                "run_completed": False,
                "audit_pass": None,
                "repair_needed": None,
                "accepted_output": False,
                "failed_steps": [],
                "failure_stage": (
                    "comparison_runner_exception"
                ),
                "exception_type": (
                    type(exc).__name__
                ),
                "exception_message": str(exc),
            }

            print(
                "Comparison runner caught "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

        manifest["run_results"].append(
            run_result
        )

        manifest["summary"] = (
            summarize_manifest_runs(
                manifest["run_results"],
                len(run_specs),
            )
        )

        # Incremental write preserves completed results
        # even if a later model fails or the run stops.
        write_json(
            manifest_path,
            manifest,
        )

    manifest["status"] = "completed"
    manifest["comparison_finished_at"] = (
        utc_now_iso()
    )

    manifest["summary"] = (
        summarize_manifest_runs(
            manifest["run_results"],
            len(run_specs),
        )
    )

    write_json(
        manifest_path,
        manifest,
    )

    summary_log_path = (
        comparison_dir
        / "summary"
        / "summary_build.log"
    )

    summary_command = [
        sys.executable,
        str(SUMMARY_BUILDER),
        "--comparison-dir",
        str(comparison_dir),
    ]

    print()
    print("=" * 78)
    print(
        "Building normalized comparison "
        "summary artifacts"
    )
    print("=" * 78)

    summary_result = run_streaming_command(
        summary_command,
        cwd=REPO_ROOT,
        log_path=summary_log_path,
    )

    manifest["summary_artifacts"] = {
        "status": (
            "completed"
            if summary_result["process_ok"]
            else "failed"
        ),
        "returncode": summary_result["returncode"],
        "started_at": summary_result["started_at"],
        "finished_at": summary_result["finished_at"],
        "elapsed_seconds": summary_result[
            "elapsed_seconds"
        ],
        "summary_dir": str(
            comparison_dir / "summary"
        ),
        "log_path": summary_result["log_path"],
    }

    write_json(
        manifest_path,
        manifest,
    )

    print()
    print(
        "Wrote FRED model comparison manifest:"
    )
    print(f"  {manifest_path}")

    for key, value in (
        manifest["summary"].items()
    ):
        print(f"{key}={value}")

    print()
    print(
        "Comparison experiment complete:"
    )
    print(
        f"  comparison_id="
        f"{comparison_id}"
    )
    print(
        f"  family_id="
        f"{comparison_family_id}"
    )
    print(
        f"  batch_number="
        f"{batch_number}"
    )
    print(
        f"  summary_status="
        f"{manifest['summary_artifacts']['status']}"
    )
    print(
        f"  comparison_dir="
        f"{comparison_dir}"
    )

    has_run_failures = (
        manifest["summary"][
            "n_process_failed"
        ]
        > 0
    )

    summary_failed = (
        manifest[
            "summary_artifacts"
        ]["status"]
        != "completed"
    )

    if (
        has_run_failures
        or summary_failed
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()