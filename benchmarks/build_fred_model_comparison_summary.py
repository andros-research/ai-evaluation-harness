#!/usr/bin/env python3
"""
Build normalized and human-readable artifacts from a FRED model comparison.

Inputs:
- comparison_manifest.json
- comparison_config.json
- each run's latest FRED evidence-loop metadata

Outputs:
- comparison_rows.jsonl
- comparison_summary.csv
- comparison_summary.json
- comparison_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


SUMMARY_SCHEMA_VERSION = "fred_model_comparison_summary_v0_1"

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RESULTS_ROOT = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "model_comparisons"
)


CSV_FIELDS = [
    "comparison_id",
    "comparison_window",
    "context_sha256",
    "run_label",
    "mode",
    "model",
    "prompt_variant",
    "temperature",
    "repetition",
    "process_ok",
    "returncode",
    "run_completed",
    "audit_pass",
    "repair_needed",
    "accepted_output",
    "failure_stage",
    "failed_steps",
    "elapsed_seconds",
    "n_steps_run",
    "n_claims",
    "n_selected_claims",
    "generation_method",
    "citation_validation",
    "n_bullets",
    "n_citations",
    "audit_errors",
    "n_repair_actions",
    "n_traceability_rows",
    "n_cited_claims",
    "demo_overall_ok",
    "inner_run_id",
    "artifact_root",
    "run_metadata_path",
    "orchestration_log",
    "started_at",
    "finished_at",
]


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_path(path: Path) -> Path:
    """Resolve a relative path from the repository root."""
    return path if path.is_absolute() else REPO_ROOT / path


def read_json(path: Path) -> dict:
    """Read and validate a JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a JSON object: {path}"
        )

    return payload


def read_json_if_exists(path: Path | None) -> dict | None:
    """Read a JSON object if the path exists."""
    if path is None or not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def write_json(path: Path, payload: object) -> None:
    """Write stable formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(
    path: Path,
    rows: list[dict],
) -> None:
    """Write one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True)
                + "\n"
            )


def csv_value(value: object) -> object:
    """Convert complex values into stable CSV strings."""
    if value is None:
        return ""

    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        )

    return value


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    """Write normalized rows as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: csv_value(row.get(field))
                    for field in CSV_FIELDS
                }
            )


def resolve_manifest_path(
    path_value: str | None,
    *,
    comparison_dir: Path,
) -> Path | None:
    """Resolve a path stored in a comparison manifest."""
    if not path_value:
        return None

    path = Path(path_value)

    if path.is_absolute():
        return path

    repo_candidate = REPO_ROOT / path
    if repo_candidate.exists():
        return repo_candidate

    return comparison_dir / path


def config_by_label(
    config: dict | None,
) -> dict[str, dict]:
    """Index configured run definitions by label."""
    if not isinstance(config, dict):
        return {}

    runs = config.get("runs", [])
    if not isinstance(runs, list):
        return {}

    indexed: dict[str, dict] = {}

    for run in runs:
        if not isinstance(run, dict):
            continue

        label = run.get("label")
        if isinstance(label, str):
            indexed[label] = run

    return indexed


def section(
    output_summary: dict,
    name: str,
) -> dict:
    """Return one output-summary section safely."""
    value = output_summary.get(name, {})
    return value if isinstance(value, dict) else {}


def build_normalized_row(
    *,
    manifest: dict,
    comparison_dir: Path,
    config_runs: dict[str, dict],
    run_result: dict,
) -> dict:
    """Build one normalized run-level comparison row."""
    run_label = str(run_result.get("run_label", ""))

    configured_run = config_runs.get(
        run_label,
        {},
    )

    metadata_path = resolve_manifest_path(
        run_result.get("run_metadata_path"),
        comparison_dir=comparison_dir,
    )

    inner_run = read_json_if_exists(
        metadata_path
    )

    output_summary = (
        inner_run.get("output_summary", {})
        if isinstance(inner_run, dict)
        else {}
    )

    if not isinstance(output_summary, dict):
        output_summary = {}

    claims = section(
        output_summary,
        "claims",
    )

    selected_claims = section(
        output_summary,
        "selected_claims",
    )

    narrative = section(
        output_summary,
        "narrative",
    )

    audit = section(
        output_summary,
        "audit",
    )

    repair = section(
        output_summary,
        "repair",
    )

    traceability = section(
        output_summary,
        "traceability",
    )

    demo_report = section(
        output_summary,
        "demo_report",
    )

    failed_steps = run_result.get(
        "failed_steps",
        [],
    )

    if not isinstance(failed_steps, list):
        failed_steps = []

    model = run_result.get("model")

    if (
        model is None
        and isinstance(inner_run, dict)
    ):
        model = inner_run.get(
            "narrative_model"
        )

    return {
        "comparison_id": manifest.get(
            "comparison_id"
        ),
        "comparison_window": manifest.get(
            "comparison_window"
        ),
        "context_sha256": manifest.get(
            "context_sha256"
        ),
        "git_commit": manifest.get(
            "git_commit"
        ),

        "run_label": run_label,
        "mode": run_result.get("mode"),
        "model": model,
        "prompt_variant": configured_run.get(
            "prompt_variant"
        ),
        "temperature": configured_run.get(
            "temperature"
        ),
        "repetition": run_result.get(
            "repetition"
        ),

        "process_ok": run_result.get(
            "process_ok"
        ),
        "returncode": run_result.get(
            "returncode"
        ),
        "run_completed": run_result.get(
            "run_completed"
        ),
        "audit_pass": run_result.get(
            "audit_pass"
        ),
        "repair_needed": run_result.get(
            "repair_needed"
        ),
        "accepted_output": run_result.get(
            "accepted_output"
        ),
        "failure_stage": run_result.get(
            "failure_stage"
        ),
        "failed_steps": failed_steps,

        "elapsed_seconds": run_result.get(
            "elapsed_seconds"
        ),
        "started_at": run_result.get(
            "started_at"
        ),
        "finished_at": run_result.get(
            "finished_at"
        ),

        "run_schema_version": (
            inner_run.get("run_schema_version")
            if isinstance(inner_run, dict)
            else None
        ),
        "run_method": (
            inner_run.get("run_method")
            if isinstance(inner_run, dict)
            else None
        ),
        "n_steps_run": (
            inner_run.get("n_steps_run")
            if isinstance(inner_run, dict)
            else None
        ),

        "n_claims": claims.get(
            "n_claims"
        ),
        "series_included": claims.get(
            "series_included"
        ),
        "n_selected_claims": (
            selected_claims.get(
                "n_selected_claims"
            )
        ),
        "selection_method": (
            selected_claims.get(
                "selection_method"
            )
        ),

        "generation_method": narrative.get(
            "generation_method"
        ),
        "citation_validation": narrative.get(
            "citation_validation"
        ),

        "n_bullets": audit.get(
            "n_bullets"
        ),
        "n_citations": audit.get(
            "n_citations"
        ),
        "audit_errors": audit.get(
            "errors"
        ),

        "n_repair_actions": repair.get(
            "n_repair_actions"
        ),

        "n_traceability_rows": (
            traceability.get(
                "n_traceability_rows"
            )
        ),
        "n_cited_claims": traceability.get(
            "n_cited_claims"
        ),

        "demo_overall_ok": demo_report.get(
            "overall_ok"
        ),

        "inner_run_id": run_result.get(
            "inner_run_id"
        ),
        "artifact_root": run_result.get(
            "artifact_root"
        ),
        "run_metadata_path": (
            str(metadata_path)
            if metadata_path is not None
            else None
        ),
        "orchestration_log": run_result.get(
            "orchestration_log"
        ),
    }


def rate(
    numerator: int,
    denominator: int,
) -> float | None:
    """Return a rounded rate or None."""
    if denominator == 0:
        return None

    return round(
        numerator / denominator,
        4,
    )


def summarize_rows(
    rows: list[dict],
) -> dict:
    """Build aggregate counts, rates, and latency statistics."""
    n_runs = len(rows)

    n_process_ok = sum(
        row.get("process_ok") is True
        for row in rows
    )

    n_run_completed = sum(
        row.get("run_completed") is True
        for row in rows
    )

    n_audit_pass = sum(
        row.get("audit_pass") is True
        for row in rows
    )

    n_repair_needed = sum(
        row.get("repair_needed") is True
        for row in rows
    )

    n_accepted_output = sum(
        row.get("accepted_output") is True
        for row in rows
    )

    audit_values = [
        row.get("audit_pass")
        for row in rows
        if isinstance(row.get("audit_pass"), bool)
    ]

    repair_values = [
        row.get("repair_needed")
        for row in rows
        if isinstance(row.get("repair_needed"), bool)
    ]

    n_audit_evaluated = len(audit_values)
    n_audit_pass = sum(
        value is True
        for value in audit_values
    )

    n_repair_evaluated = len(repair_values)
    n_repair_needed = sum(
        value is True
        for value in repair_values
    )

    latencies = [
        float(row["elapsed_seconds"])
        for row in rows
        if isinstance(
            row.get("elapsed_seconds"),
            (int, float),
        )
    ]

    return {
        "n_runs": n_runs,
        "n_process_ok": n_process_ok,
        "n_process_failed": (
            n_runs - n_process_ok
        ),
        "n_run_completed": n_run_completed,
        "n_audit_pass": n_audit_pass,
        "n_repair_needed": n_repair_needed,
        "n_accepted_output": n_accepted_output,

        "process_ok_rate": rate(
            n_process_ok,
            n_runs,
        ),
        "run_completion_rate": rate(
            n_run_completed,
            n_runs,
        ),
        "audit_pass_rate": rate(
            n_audit_pass,
            n_audit_evaluated,
        ),
        "repair_needed_rate": rate(
            n_repair_needed,
            n_repair_evaluated,
        ),
        "accepted_output_rate": rate(
            n_accepted_output,
            n_runs,
        ),

        "n_audit_evaluated": n_audit_evaluated,
        "n_audit_pass": n_audit_pass,
        "n_repair_evaluated": n_repair_evaluated,
        "n_repair_needed": n_repair_needed,

        "latency_seconds": {
            "n_observations": len(latencies),
            "minimum": (
                round(min(latencies), 3)
                if latencies
                else None
            ),
            "mean": (
                round(mean(latencies), 3)
                if latencies
                else None
            ),
            "maximum": (
                round(max(latencies), 3)
                if latencies
                else None
            ),
        },
    }


def build_group_summaries(
    rows: list[dict],
) -> list[dict]:
    """Build summaries grouped by run label."""
    grouped: dict[str, list[dict]] = {}

    for row in rows:
        label = str(row.get("run_label", ""))
        grouped.setdefault(label, []).append(row)

    summaries: list[dict] = []

    for label in sorted(grouped):
        group_rows = grouped[label]
        first = group_rows[0]

        summaries.append(
            {
                "run_label": label,
                "mode": first.get("mode"),
                "model": first.get("model"),
                "prompt_variant": first.get(
                    "prompt_variant"
                ),
                "temperature": first.get(
                    "temperature"
                ),
                **summarize_rows(group_rows),
            }
        )

    return summaries


def markdown_value(value: object) -> str:
    """Format a value safely for Markdown tables."""
    if value is None:
        return ""

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, (dict, list)):
        text = json.dumps(
            value,
            sort_keys=True,
        )
    else:
        text = str(value)

    return (
        text
        .replace("|", "\\|")
        .replace("\n", " ")
    )


def build_markdown_report(
    *,
    summary: dict,
    rows: list[dict],
) -> str:
    """Build a human-readable comparison report."""
    overall = summary["overall"]

    lines = [
        "# FRED Model Comparison Report",
        "",
        f"- Comparison ID: `{summary['comparison_id']}`",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Comparison window: `{summary.get('comparison_window')}`",
        f"- Context SHA-256: `{summary.get('context_sha256')}`",
        f"- Git commit: `{summary.get('git_commit')}`",
        "",
        "## Overall Result",
        "",
        f"- Runs: **{overall['n_runs']}**",
        f"- Process success: **{overall['n_process_ok']} / {overall['n_runs']}**",
        f"- Completed evidence loops: **{overall['n_run_completed']} / {overall['n_runs']}**",
        (
            f"- Audit passes: "
            f"**{overall['n_audit_pass']} / "
            f"{overall['n_audit_evaluated']} evaluated**"
        ),
        (
            f"- Repairs needed: "
            f"**{overall['n_repair_needed']} / "
            f"{overall['n_repair_evaluated']} evaluated**"
        ),
        f"- Accepted outputs: **{overall['n_accepted_output']} / {overall['n_runs']}**",
        "",
        "## Run Results",
        "",
        "| Label | Model | Rep | Process | Completed | Audit | Repair | Accepted | Seconds | Failure stage |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_value(
                        row.get("run_label")
                    ),
                    markdown_value(
                        row.get("model")
                        or row.get("mode")
                    ),
                    markdown_value(
                        row.get("repetition")
                    ),
                    markdown_value(
                        row.get("process_ok")
                    ),
                    markdown_value(
                        row.get("run_completed")
                    ),
                    markdown_value(
                        row.get("audit_pass")
                    ),
                    markdown_value(
                        row.get("repair_needed")
                    ),
                    markdown_value(
                        row.get("accepted_output")
                    ),
                    markdown_value(
                        row.get("elapsed_seconds")
                    ),
                    markdown_value(
                        row.get("failure_stage")
                    ),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Group Summary",
            "",
            "| Label | Model | Runs | Process rate | Audit rate | Repair rate | Acceptance rate | Mean seconds |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for group in summary["by_run_label"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_value(
                        group.get("run_label")
                    ),
                    markdown_value(
                        group.get("model")
                        or group.get("mode")
                    ),
                    markdown_value(
                        group.get("n_runs")
                    ),
                    markdown_value(
                        group.get("process_ok_rate")
                    ),
                    markdown_value(
                        group.get("audit_pass_rate")
                    ),
                    markdown_value(
                        group.get("repair_needed_rate")
                    ),
                    markdown_value(
                        group.get("accepted_output_rate")
                    ),
                    markdown_value(
                        group.get(
                            "latency_seconds",
                            {},
                        ).get("mean")
                    ),
                ]
            )
            + " |"
        )

    failed_rows = [
        row
        for row in rows
        if row.get("process_ok") is not True
        or row.get("accepted_output") is not True
    ]

    lines.extend(
        [
            "",
            "## Failures and Rejections",
            "",
        ]
    )

    if not failed_rows:
        lines.append(
            "No process failures or rejected outputs were recorded."
        )
    else:
        for row in failed_rows:
            lines.extend(
                [
                    f"### {row.get('run_label')} / repetition {row.get('repetition')}",
                    "",
                    f"- Process OK: `{row.get('process_ok')}`",
                    f"- Run completed: `{row.get('run_completed')}`",
                    f"- Audit pass: `{row.get('audit_pass')}`",
                    f"- Repair needed: `{row.get('repair_needed')}`",
                    f"- Accepted output: `{row.get('accepted_output')}`",
                    f"- Failure stage: `{row.get('failure_stage')}`",
                    f"- Failed steps: `{row.get('failed_steps')}`",
                    f"- Audit errors: `{row.get('audit_errors')}`",
                    "",
                ]
            )

    lines.extend(
        [
            "## Artifact Locations",
            "",
        ]
    )

    for row in rows:
        lines.append(
            f"- `{row.get('run_label')}` repetition "
            f"{row.get('repetition')}: "
            f"`{row.get('artifact_root')}`"
        )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build normalized artifacts from a "
            "FRED model comparison."
        )
    )

    parser.add_argument(
        "--comparison-dir",
        type=Path,
        required=True,
        help=(
            "Comparison experiment directory containing "
            "comparison_manifest.json."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. Defaults to "
            "<comparison-dir>/summary."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    comparison_dir = resolve_repo_path(
        args.comparison_dir
    )

    manifest_path = (
        comparison_dir
        / "comparison_manifest.json"
    )

    config_path = (
        comparison_dir
        / "comparison_config.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Comparison manifest not found: {manifest_path}"
        )

    manifest = read_json(
        manifest_path
    )

    config = (
        read_json(config_path)
        if config_path.exists()
        else None
    )

    run_results = manifest.get(
        "run_results",
        [],
    )

    if not isinstance(run_results, list):
        raise ValueError(
            "comparison_manifest.json run_results "
            "must be a JSON array."
        )

    indexed_config = config_by_label(
        config
    )

    rows = [
        build_normalized_row(
            manifest=manifest,
            comparison_dir=comparison_dir,
            config_runs=indexed_config,
            run_result=run_result,
        )
        for run_result in run_results
        if isinstance(run_result, dict)
    ]

    output_dir = (
        resolve_repo_path(args.output_dir)
        if args.output_dir is not None
        else comparison_dir / "summary"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows_jsonl_path = (
        output_dir
        / "comparison_rows.jsonl"
    )

    summary_csv_path = (
        output_dir
        / "comparison_summary.csv"
    )

    summary_json_path = (
        output_dir
        / "comparison_summary.json"
    )

    report_path = (
        output_dir
        / "comparison_report.md"
    )

    summary = {
        "summary_schema_version": (
            SUMMARY_SCHEMA_VERSION
        ),
        "comparison_id": manifest.get(
            "comparison_id"
        ),
        "generated_at": utc_now_iso(),
        "source_manifest": str(
            manifest_path
        ),
        "source_config": (
            str(config_path)
            if config_path.exists()
            else None
        ),
        "comparison_status": manifest.get(
            "status"
        ),
        "comparison_window": manifest.get(
            "comparison_window"
        ),
        "context_sha256": manifest.get(
            "context_sha256"
        ),
        "git_commit": manifest.get(
            "git_commit"
        ),
        "overall": summarize_rows(
            rows
        ),
        "by_run_label": (
            build_group_summaries(rows)
        ),
        "rows": rows,
    }

    write_jsonl(
        rows_jsonl_path,
        rows,
    )

    write_csv(
        summary_csv_path,
        rows,
    )

    write_json(
        summary_json_path,
        summary,
    )

    report_path.write_text(
        build_markdown_report(
            summary=summary,
            rows=rows,
        ),
        encoding="utf-8",
    )

    print(
        "Wrote FRED model comparison summary artifacts:"
    )
    print(f"  {rows_jsonl_path}")
    print(f"  {summary_csv_path}")
    print(f"  {summary_json_path}")
    print(f"  {report_path}")

    overall = summary["overall"]

    print(
        f"n_runs={overall['n_runs']}"
    )
    print(
        f"n_process_ok={overall['n_process_ok']}"
    )
    print(
        f"n_audit_pass={overall['n_audit_pass']}"
    )
    print(
        "n_repair_needed="
        f"{overall['n_repair_needed']}"
    )
    print(
        "n_accepted_output="
        f"{overall['n_accepted_output']}"
    )


if __name__ == "__main__":
    main()
