#!/usr/bin/env python3
"""
Build a human-readable FRED evidence loop demo report.

v1.7.0 demo-focused version:
- reads latest FRED evidence-loop artifacts
- summarizes the run, selected claims, generated narrative, audit, repair plan,
  and traceability map
- writes a markdown report suitable for a 3-5 minute screen walkthrough
- makes the demo thesis, validation boundary, and current limitation visible
- writes metadata for downstream dashboard/demo use

This script does not generate new claims or narratives. It packages the latest
artifacts into a screen-friendly report.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEMO_REPORT_SCHEMA_VERSION = "fred_demo_report_v0_2"
DEMO_REPORT_METHOD = "v1_7_demo_markdown_summary"

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RUN_METADATA_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "fred_runs"
    / "latest_fred_evidence_loop_run.json"
)

DEFAULT_CLAIMS_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "fred_claims"
    / "fred_claims.json"
)

DEFAULT_SELECTED_CLAIMS_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "fred_claims"
    / "selected_fred_claims.json"
)

DEFAULT_NARRATIVE_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "fred_narratives"
    / "fred_narrative.md"
)

DEFAULT_NARRATIVE_METADATA_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "fred_narratives"
    / "fred_narrative_metadata.json"
)

DEFAULT_AUDIT_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "fred_audits"
    / "fred_narrative_audit.json"
)

DEFAULT_REPAIR_PLAN_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "fred_repairs"
    / "fred_repair_plan.json"
)

DEFAULT_TRACEABILITY_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "fred_traceability"
    / "fred_traceability_summary.json"
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "fred_demo"
)


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    """Load a JSON artifact."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_text(path: Path) -> str:
    """Load a text artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Text file does not exist: {path}")
    return path.read_text(encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    """Write JSON with stable formatting."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def bool_text(value: object) -> str:
    """Render booleans in a readable way."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "n/a"
    return str(value)


def short_claim_id(claim_id: str | None) -> str:
    """Return a compact claim label for tables."""
    if not claim_id:
        return ""
    return claim_id.replace("fred__", "")


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """Render a simple markdown table."""
    if not rows:
        return "_No rows available._\n"

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"

    row_lines = []
    for row in rows:
        clean = [str(item).replace("\n", " ").strip() for item in row]
        row_lines.append("| " + " | ".join(clean) + " |")

    return "\n".join([header_line, separator_line, *row_lines]) + "\n"


def render_run_summary(
    *,
    run_metadata: dict,
    narrative_metadata: dict,
    audit: dict,
    repair_plan: dict,
    traceability_rows: list[dict],
) -> str:
    """Render the run summary section."""
    output_summary = run_metadata.get("output_summary", {})

    rows = [
        ["Overall OK", bool_text(run_metadata.get("overall_ok"))],
        ["Completed core steps before report", len(run_metadata.get("completed_steps", []))],
        ["Failed steps", len(run_metadata.get("failed_steps", []))],
        ["Comparison window", run_metadata.get("comparison_window", "n/a")],
        ["Narrative mode", run_metadata.get("narrative_mode", narrative_metadata.get("generation_mode", "n/a"))],
        ["Narrative model", run_metadata.get("narrative_model") or "n/a"],
        ["Generation method", narrative_metadata.get("generation_method", "n/a")],
        ["LLM used", bool_text(narrative_metadata.get("llm_metadata", {}).get("llm_used"))],
        ["Audit pass", bool_text(audit.get("audit_pass"))],
        ["Repair needed", bool_text(repair_plan.get("repair_needed"))],
        ["Traceability rows", len(traceability_rows)],
        ["Cited claims", sum(1 for row in traceability_rows if row.get("was_cited"))],
    ]

    # Keep output_summary referenced indirectly so future fields can be added
    # without changing the human-readable contract.
    _ = output_summary

    return markdown_table(["Field", "Value"], rows)


def render_selected_claims(selected_claims: list[dict]) -> str:
    """Render selected claims as a markdown table."""
    rows = []

    for claim in selected_claims:
        rows.append(
            [
                claim.get("selection_rank", ""),
                claim.get("source_series", ""),
                claim.get("metric_name", ""),
                claim.get("comparison_window", ""),
                claim.get("direction", ""),
                claim.get("current_value", ""),
                claim.get("prior_value", ""),
                claim.get("delta_value", ""),
                claim.get("claim_text", ""),
            ]
        )

    return markdown_table(
        [
            "Rank",
            "Series",
            "Metric",
            "Window",
            "Direction",
            "Current",
            "Prior",
            "Delta",
            "Claim",
        ],
        rows,
    )


def render_audit_summary(audit: dict, repair_plan: dict) -> str:
    """Render audit and repair status."""
    rows = [
        ["Audit pass", bool_text(audit.get("audit_pass"))],
        ["Bullets", audit.get("n_bullets", "n/a")],
        ["Citations", audit.get("n_citations", "n/a")],
        ["Unknown citations", len(audit.get("unknown_citations", []))],
        ["Bullets missing citations", audit.get("n_bullets_missing_citations", "n/a")],
        ["Content mismatches", audit.get("n_bullets_with_content_mismatches", "n/a")],
        ["Content issue counts", json.dumps(audit.get("content_issue_counts", {}), sort_keys=True)],
        ["Repair needed", bool_text(repair_plan.get("repair_needed"))],
        ["Repair actions", repair_plan.get("n_repair_actions", "n/a")],
    ]

    return markdown_table(["Check", "Result"], rows)


def render_traceability(traceability_rows: list[dict]) -> str:
    """Render source-to-narrative traceability as a markdown table."""
    rows = []

    for row in traceability_rows:
        rows.append(
            [
                row.get("source_series", ""),
                short_claim_id(row.get("claim_id")),
                bool_text(row.get("was_selected")),
                row.get("selection_rank", ""),
                bool_text(row.get("was_cited")),
                row.get("audit_citation_status", ""),
                row.get("repair_action_count", 0),
                row.get("narrative_bullet_text", ""),
            ]
        )

    return markdown_table(
        [
            "Series",
            "Claim",
            "Selected",
            "Rank",
            "Cited",
            "Audit status",
            "Repair actions",
            "Narrative bullet",
        ],
        rows,
    )


def render_compact_traceability(traceability_rows: list[dict]) -> str:
    """Render a compact traceability view for screen walkthroughs."""
    if not traceability_rows:
        return "_No traceability rows available._\n"

    lines = []
    for row in traceability_rows:
        series = row.get("source_series", "unknown_series")
        rank = row.get("selection_rank", "n/a")
        cited = "cited" if row.get("was_cited") else "not cited"
        status = row.get("audit_citation_status") or "unknown"
        repairs = row.get("repair_action_count", 0)

        if repairs:
            repair_text = f"{repairs} repair action(s)"
        else:
            repair_text = "no repair action"

        lines.append(
            f"- `{series}` → selected rank `{rank}` → {cited} → "
            f"audit `{status}` → {repair_text}"
        )

    return "\n".join(lines) + "\n"


def normalize_narrative_for_report(narrative_text: str) -> str:
    """Demote embedded narrative headings so the report has one clean outline."""
    replacements = {
        "# FRED Macro Narrative": "**FRED Macro Narrative**",
        "## Claim-Cited Summary": "**Claim-cited summary**",
        "## Macro Narrative": "**Macro narrative**",
    }

    cleaned_lines = []
    for line in narrative_text.strip().splitlines():
        stripped = line.strip()
        cleaned_lines.append(replacements.get(stripped, line))

    return "\n".join(cleaned_lines).strip()


def render_demo_workflow() -> str:
    """Render the evidence-loop workflow diagram."""
    return (
        "```text\n"
        "FRED macro context\n"
        "  -> deterministic source-grounded claims\n"
        "  -> selected evidence claims\n"
        "  -> deterministic or local LLM narrative\n"
        "  -> citation + numeric/directional audit\n"
        "  -> repair plan, if needed\n"
        "  -> source-to-narrative traceability\n"
        "  -> screen-readable demo report\n"
        "```\n"
    )


def render_validation_boundary(
    *,
    run_metadata: dict,
    narrative_metadata: dict,
) -> str:
    """Render the current validation boundary and interpretation-risk caveat."""
    mode = run_metadata.get("narrative_mode") or narrative_metadata.get("generation_mode")

    if mode == "llm":
        return (
            "The audit currently verifies the parts of the narrative that can be "
            "checked directly against the structured claim layer: citation coverage, "
            "numeric values, and direction.\n\n"
            "It does **not** yet fully validate interpretive or market-facing language. "
            "For example, phrases such as `significant`, `deterioration`, "
            "`policy tightening`, or `market expectations` may be plausible, but they "
            "are not separately proven by the current audit layer.\n\n"
            "That limitation is intentional for this MVP. v1.7.0 demonstrates factual "
            "traceability first, while making the remaining interpretation risk visible "
            "rather than hiding it.\n"
        )

    return (
        "The deterministic narrative is template-based, so interpretation risk is "
        "more limited in this run. The same audit boundary still matters for future "
        "LLM runs: citation coverage, numeric values, and direction can be checked "
        "now, while richer interpretation-risk classification remains a later layer.\n"
    )


def render_five_minute_takeaway() -> str:
    """Render the final demo takeaway section."""
    return (
        "A normal AI demo usually shows only the final answer. This demo shows the "
        "workflow around the answer. The model is asked to write inside a controlled "
        "evidence loop, and the surrounding harness records what evidence was used, "
        "what the narrative said, what the audit checked, whether repair was needed, "
        "and how each narrative claim traces back to source data.\n\n"
        "The current system is small, but the pattern is the point: context engineering, "
        "structured artifacts, validation, repair planning, and traceability around an LLM.\n"
    )


def build_demo_report_markdown(
    *,
    generated_at: str,
    run_metadata: dict,
    claims: list[dict],
    selected_claims: list[dict],
    narrative_text: str,
    narrative_metadata: dict,
    audit: dict,
    repair_plan: dict,
    traceability_rows: list[dict],
) -> str:
    """Build the human-readable demo report."""
    run_id = run_metadata.get("run_id", "unknown_run")
    narrative_mode = run_metadata.get(
        "narrative_mode",
        narrative_metadata.get("generation_mode", "n/a"),
    )

    return (
        "# CPI/FRED Evidence Loop Demo Report\n\n"
        f"Generated at: {generated_at}\n\n"
        "## 1. Five-minute demo thesis\n\n"
        "This demo turns structured FRED macro context into source-grounded claims, "
        "uses those claims to generate a claim-cited narrative, audits the narrative "
        "for citation, numeric, and directional consistency, plans repair if needed, "
        "and records traceability back to the source evidence.\n\n"
        "The point is not that an LLM can write a macro paragraph. The point is that "
        "the LLM is only one component inside a controlled evidence loop.\n\n"
        "## 2. Workflow at a glance\n\n"
        f"{render_demo_workflow()}\n"
        "## 3. Why this matters\n\n"
        "Most AI demos show only the final answer. This report shows the machinery "
        "around the answer: source construction, claim selection, citation discipline, "
        "numeric and directional audit, repair planning, and traceability.\n\n"
        "That harness is what makes the output inspectable.\n\n"
        "## 4. Run summary\n\n"
        f"Run ID: `{run_id}`\n\n"
        f"Narrative mode: `{narrative_mode}`\n\n"
        f"Input claims: `{len(claims)}`\n\n"
        f"Selected claims: `{len(selected_claims)}`\n\n"
        f"{render_run_summary(run_metadata=run_metadata, narrative_metadata=narrative_metadata, audit=audit, repair_plan=repair_plan, traceability_rows=traceability_rows)}\n"
        "## 5. Selected source claims\n\n"
        "These are the structured claims made available to the narrative step. "
        "They are generated from the source macro context before the narrative is written.\n\n"
        f"{render_selected_claims(selected_claims)}\n"
        "## 6. Generated narrative\n\n"
        "This is the narrative output produced from the selected claims. In LLM mode, "
        "this is where useful language and risky interpretation can both appear.\n\n"
        f"{normalize_narrative_for_report(narrative_text)}\n\n"
        "## 7. What the audit and repair layer found\n\n"
        "The current audit checks factual grounding against the claim layer. If the "
        "audit fails, the repair planner records proposed fixes instead of silently "
        "accepting the narrative.\n\n"
        f"{render_audit_summary(audit, repair_plan)}\n"
        "## 8. Current limitation: interpretation risk\n\n"
        f"{render_validation_boundary(run_metadata=run_metadata, narrative_metadata=narrative_metadata)}\n"
        "## 9. Source-to-narrative traceability\n\n"
        "Each row connects a source claim to whether it was selected, cited, audited, "
        "and represented in the generated narrative.\n\n"
        "### Compact walkthrough view\n\n"
        f"{render_compact_traceability(traceability_rows)}\n"
        "### Full traceability table\n\n"
        f"{render_traceability(traceability_rows)}\n"
        "## 10. Five-minute takeaway\n\n"
        f"{render_five_minute_takeaway()}"
    )


def validate_demo_inputs(
    *,
    run_metadata: dict,
    claims: list[dict],
    selected_claims: list[dict],
    narrative_text: str,
    narrative_metadata: dict,
    audit: dict,
    repair_plan: dict,
    traceability_rows: list[dict],
) -> None:
    """Validate that demo inputs are minimally usable."""
    errors: list[str] = []

    if not isinstance(run_metadata, dict):
        errors.append("run_metadata must be an object")
    if not isinstance(claims, list):
        errors.append("claims must be a list")
    if not isinstance(selected_claims, list):
        errors.append("selected_claims must be a list")
    if not narrative_text.strip():
        errors.append("narrative_text is empty")
    if not isinstance(narrative_metadata, dict):
        errors.append("narrative_metadata must be an object")
    if not isinstance(audit, dict):
        errors.append("audit must be an object")
    if not isinstance(repair_plan, dict):
        errors.append("repair_plan must be an object")
    if not isinstance(traceability_rows, list):
        errors.append("traceability_rows must be a list")

    if isinstance(run_metadata, dict) and run_metadata.get("overall_ok") is not True:
        errors.append("latest run metadata does not show overall_ok=true")

    if isinstance(audit, dict) and "audit_pass" not in audit:
        errors.append("audit artifact missing audit_pass")

    if isinstance(repair_plan, dict) and "repair_needed" not in repair_plan:
        errors.append("repair plan missing repair_needed")

    if errors:
        joined = "\n".join(f"- {err}" for err in errors)
        raise ValueError(f"FRED demo report input validation failed:\n{joined}")


def write_demo_report_artifacts(
    *,
    run_metadata_path: Path,
    claims_path: Path,
    selected_claims_path: Path,
    narrative_path: Path,
    narrative_metadata_path: Path,
    audit_path: Path,
    repair_plan_path: Path,
    traceability_path: Path,
    output_dir: Path,
) -> None:
    """Build and write demo report artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_at = utc_now_iso()

    run_metadata = load_json(run_metadata_path)
    claims = load_json(claims_path)
    selected_claims = load_json(selected_claims_path)
    narrative_text = load_text(narrative_path)
    narrative_metadata = load_json(narrative_metadata_path)
    audit = load_json(audit_path)
    repair_plan = load_json(repair_plan_path)
    traceability_rows = load_json(traceability_path)

    validate_demo_inputs(
        run_metadata=run_metadata,
        claims=claims,
        selected_claims=selected_claims,
        narrative_text=narrative_text,
        narrative_metadata=narrative_metadata,
        audit=audit,
        repair_plan=repair_plan,
        traceability_rows=traceability_rows,
    )

    report_markdown = build_demo_report_markdown(
        generated_at=generated_at,
        run_metadata=run_metadata,
        claims=claims,
        selected_claims=selected_claims,
        narrative_text=narrative_text,
        narrative_metadata=narrative_metadata,
        audit=audit,
        repair_plan=repair_plan,
        traceability_rows=traceability_rows,
    )

    report_path = output_dir / "fred_demo_report.md"
    metadata_path = output_dir / "fred_demo_report_metadata.json"

    report_path.write_text(report_markdown, encoding="utf-8")

    metadata = {
        "demo_report_schema_version": DEMO_REPORT_SCHEMA_VERSION,
        "demo_report_method": DEMO_REPORT_METHOD,
        "generated_at": generated_at,
        "run_id": run_metadata.get("run_id"),
        "overall_ok": run_metadata.get("overall_ok"),
        "narrative_mode": run_metadata.get("narrative_mode", narrative_metadata.get("generation_mode")),
        "narrative_model": run_metadata.get("narrative_model"),
        "generation_method": narrative_metadata.get("generation_method"),
        "audit_pass": audit.get("audit_pass"),
        "repair_needed": repair_plan.get("repair_needed"),
        "n_claims": len(claims),
        "n_selected_claims": len(selected_claims),
        "n_traceability_rows": len(traceability_rows),
        "demo_focus": "screen_readable_cpi_fred_evidence_loop",
        "includes_interpretation_risk_section": True,
        "intended_demo_read_time_minutes": "3-5",
        "input_files": {
            "run_metadata_json": str(run_metadata_path),
            "claims_json": str(claims_path),
            "selected_claims_json": str(selected_claims_path),
            "narrative_md": str(narrative_path),
            "narrative_metadata_json": str(narrative_metadata_path),
            "audit_json": str(audit_path),
            "repair_plan_json": str(repair_plan_path),
            "traceability_json": str(traceability_path),
        },
        "output_files": {
            "demo_report_md": str(report_path),
            "metadata_json": str(metadata_path),
        },
    }

    write_json(metadata_path, metadata)

    print("Wrote FRED demo report artifacts:")
    print(f"  {report_path}")
    print(f"  {metadata_path}")
    print(f"n_claims={len(claims)}")
    print(f"n_selected_claims={len(selected_claims)}")
    print(f"n_traceability_rows={len(traceability_rows)}")
    print(f"overall_ok={metadata['overall_ok']}")
    print(f"audit_pass={metadata['audit_pass']}")
    print(f"repair_needed={metadata['repair_needed']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a human-readable FRED evidence loop demo report."
    )
    parser.add_argument(
        "--run-metadata",
        type=Path,
        default=DEFAULT_RUN_METADATA_PATH,
        help="Input latest_fred_evidence_loop_run.json file.",
    )
    parser.add_argument(
        "--claims",
        type=Path,
        default=DEFAULT_CLAIMS_PATH,
        help="Input fred_claims.json file.",
    )
    parser.add_argument(
        "--selected-claims",
        type=Path,
        default=DEFAULT_SELECTED_CLAIMS_PATH,
        help="Input selected_fred_claims.json file.",
    )
    parser.add_argument(
        "--narrative",
        type=Path,
        default=DEFAULT_NARRATIVE_PATH,
        help="Input fred_narrative.md file.",
    )
    parser.add_argument(
        "--narrative-metadata",
        type=Path,
        default=DEFAULT_NARRATIVE_METADATA_PATH,
        help="Input fred_narrative_metadata.json file.",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=DEFAULT_AUDIT_PATH,
        help="Input fred_narrative_audit.json file.",
    )
    parser.add_argument(
        "--repair-plan",
        type=Path,
        default=DEFAULT_REPAIR_PLAN_PATH,
        help="Input fred_repair_plan.json file.",
    )
    parser.add_argument(
        "--traceability",
        type=Path,
        default=DEFAULT_TRACEABILITY_PATH,
        help="Input fred_traceability_summary.json file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where demo report artifacts will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_demo_report_artifacts(
        run_metadata_path=args.run_metadata,
        claims_path=args.claims,
        selected_claims_path=args.selected_claims,
        narrative_path=args.narrative,
        narrative_metadata_path=args.narrative_metadata,
        audit_path=args.audit,
        repair_plan_path=args.repair_plan,
        traceability_path=args.traceability,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()