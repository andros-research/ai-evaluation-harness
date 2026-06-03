# FRED Demo Report v0.1

## Purpose

The FRED demo report layer packages the latest FRED evidence-loop artifacts into a human-readable markdown report.

Earlier v1.6 layers produced durable JSON, CSV, and markdown artifacts for claims, selection, narrative generation, audit, repair planning, traceability, and run metadata. The demo report is a presentation layer over those artifacts.

The purpose of this layer is to make the evidence loop screen-friendly for review, debugging, and future demo use.

## Pipeline Position

```text
fred_macro_context.json
  -> build_fred_claims.py
  -> fred_claims.csv/json/metadata.json
  -> select_fred_claims.py
  -> selected_fred_claims.csv/json/metadata.json
  -> generate_fred_narrative_from_claims.py
  -> fred_narrative.md
  -> fred_narrative_metadata.json
  -> audit_fred_narrative.py
  -> fred_narrative_audit.json
  -> plan_fred_narrative_repair.py
  -> fred_repair_plan.json
  -> build_fred_traceability_summary.py
  -> fred_traceability_summary.csv/json/metadata.json
  -> build_fred_demo_report.py
  -> fred_demo_report.md
  -> fred_demo_report_metadata.json
```

## Runner Position

The FRED evidence-loop runner now builds the demo report after the six core evidence-loop steps.

The runner first writes current run metadata so the report reads the current run rather than the previous run. It then builds the demo report and rewrites the final run metadata with demo-report summary fields included.

This avoids a stale-run bug where the report could otherwise lag one run behind the active runner state.

## Default Command

Deterministic mode:

```bash
python benchmarks/run_fred_evidence_loop.py \
  --input-context benchmarks/data/fred_macro_context.json \
  --comparison-window 12m
```

LLM mode:

```bash
python benchmarks/run_fred_evidence_loop.py \
  --input-context benchmarks/data/fred_macro_context.json \
  --comparison-window 12m \
  --narrative-mode llm \
  --narrative-model llama3 \
  --ollama-host http://127.0.0.1:11434
```

Standalone report builder:

```bash
python benchmarks/build_fred_demo_report.py
```

## Input Artifacts

Default inputs:

```text
benchmarks/results/fred_runs/latest_fred_evidence_loop_run.json
benchmarks/results/fred_claims/fred_claims.json
benchmarks/results/fred_claims/selected_fred_claims.json
benchmarks/results/fred_narratives/fred_narrative.md
benchmarks/results/fred_narratives/fred_narrative_metadata.json
benchmarks/results/fred_audits/fred_narrative_audit.json
benchmarks/results/fred_repairs/fred_repair_plan.json
benchmarks/results/fred_traceability/fred_traceability_summary.json
```

## Output Artifacts

Default outputs:

```text
benchmarks/results/fred_demo/fred_demo_report.md
benchmarks/results/fred_demo/fred_demo_report_metadata.json
```

## Report Sections

The markdown report includes:

| Section | Description |
|---|---|
| `What this system does` | Human-readable explanation of the evidence loop. |
| `Run summary` | Current run ID, narrative mode, model, audit status, repair status, and traceability count. |
| `Selected source claims` | Table of selected source-grounded FRED claims. |
| `Generated narrative` | Deterministic or LLM-generated narrative text. |
| `Audit and repair result` | Citation, numeric/directional audit, and repair-plan summary. |
| `Source-to-narrative traceability` | Table mapping source claims to cited narrative bullets. |
| `Important caveat` | Current validation boundary, especially around LLM interpretation risk. |
| `Why this matters` | Explanation of the harness pattern. |

## Metadata

The demo report metadata records:

| Field | Description |
|---|---|
| `demo_report_schema_version` | Demo report schema version, initially `fred_demo_report_v0_1`. |
| `demo_report_method` | Report construction method, initially `latest_artifact_markdown_summary`. |
| `generated_at` | UTC timestamp when the report was generated. |
| `run_id` | Evidence-loop run ID summarized by the report. |
| `overall_ok` | Whether the summarized run succeeded. |
| `narrative_mode` | Narrative generation mode. |
| `narrative_model` | LLM model if applicable. |
| `generation_method` | Narrative generation method. |
| `audit_pass` | Whether the narrative audit passed. |
| `repair_needed` | Whether repair was needed. |
| `n_claims` | Number of FRED claims. |
| `n_selected_claims` | Number of selected claims. |
| `n_traceability_rows` | Number of traceability rows. |
| `input_files` | Source artifact paths. |
| `output_files` | Report artifact paths. |

## Current Limitations

v0.1 does not yet:

- render an HTML report
- render a PDF report
- create screenshots or charts
- compare deterministic and LLM outputs side by side
- include multiple runs in one report
- include multiple comparison windows
- include interpretation-risk classifications
- include dashboard visualizations
- include token/probabilistic telemetry

These are intentionally deferred.

## Demo Use

This report is designed to support a short on-screen walkthrough:

```text
Here is the source data converted into claims.
Here is the generated narrative.
Here is the audit result.
Here is the repair-plan status.
Here is the traceability table tying each narrative bullet back to a claim.
Here is what the system still cannot validate yet.
```

The report is not meant to hide limitations. The caveat section is part of the demo because it shows the current boundary between factual validation and interpretive judgment.

## Future Extensions

Likely future extensions include:

```text
v1.7.0 - CPI/FRED MVP demo loop
v1.7.x - deterministic vs LLM side-by-side comparison
v1.7.x - interpretation-risk warning layer
v1.8.x - broader source ingestion
v2.0 - token/probabilistic telemetry
```

Longer-term, the demo layer may incorporate:

- Streamlit dashboard view
- HTML export
- PDF export
- side-by-side run comparison
- multi-model comparison
- interpretation-risk highlighting
- model profile links
- repair-before/after comparison