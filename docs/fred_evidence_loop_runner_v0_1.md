# FRED Evidence Loop Runner v0.1

## Purpose

The FRED evidence loop runner coordinates the full v1.6 artifact spine with a single command.

This layer turns the individual FRED pipeline scripts into one reproducible workflow. The runner does not replace the individual steps. Each stage remains independently runnable and auditable. The runner exists to orchestrate the chain, capture step-level execution metadata, and produce a run-level summary artifact.

## Pipeline Position

```text
run_fred_evidence_loop.py
  -> build_fred_claims.py
  -> select_fred_claims.py
  -> generate_fred_narrative_from_claims.py
  -> audit_fred_narrative.py
  -> plan_fred_narrative_repair.py
  -> build_fred_traceability_summary.py
  -> latest_fred_evidence_loop_run.json
```

## Default Command

```bash
python benchmarks/run_fred_evidence_loop.py \
  --input-context benchmarks/data/fred_macro_context.json \
  --comparison-window 12m
```

## Inputs

Default input context:

```text
benchmarks/data/fred_macro_context.json
```

This file is produced upstream by `build_fred_prompt_context.py`.

The runner currently supports comparison windows:

```text
6m
12m
24m
```

## Output Artifacts

The runner coordinates the existing output artifacts:

```text
benchmarks/results/fred_claims/fred_claims.csv
benchmarks/results/fred_claims/fred_claims.json
benchmarks/results/fred_claims/fred_claims_metadata.json

benchmarks/results/fred_claims/selected_fred_claims.csv
benchmarks/results/fred_claims/selected_fred_claims.json
benchmarks/results/fred_claims/selected_fred_claims_metadata.json

benchmarks/results/fred_narratives/fred_narrative.md
benchmarks/results/fred_narratives/fred_narrative_metadata.json

benchmarks/results/fred_audits/fred_narrative_audit.json

benchmarks/results/fred_repairs/fred_repair_plan.json

benchmarks/results/fred_traceability/fred_traceability_summary.csv
benchmarks/results/fred_traceability/fred_traceability_summary.json
benchmarks/results/fred_traceability/fred_traceability_summary_metadata.json
```

It also writes run-level metadata:

```text
benchmarks/results/fred_runs/fred_evidence_loop_<timestamp>.json
benchmarks/results/fred_runs/latest_fred_evidence_loop_run.json
```

## Run Method

v0.1 uses subprocess orchestration:

```text
run_method = subprocess_artifact_chain
```

Each pipeline step is executed as a standalone Python script. The runner captures:

- command
- start timestamp
- finish timestamp
- return code
- stdout
- stderr
- step success flag

## Run Metadata

The run metadata artifact records:

| Field | Description |
|---|---|
| `run_schema_version` | Run metadata schema version, initially `fred_evidence_loop_run_v0_1`. |
| `run_method` | Runner method, initially `subprocess_artifact_chain`. |
| `run_id` | Timestamped run identifier. |
| `run_started_at` | UTC timestamp when the run began. |
| `run_finished_at` | UTC timestamp when the run completed. |
| `input_context` | Input FRED context file. |
| `comparison_window` | Comparison window used by the claim builder. |
| `overall_ok` | Whether all executed steps succeeded. |
| `stop_on_failure` | Whether the runner stops after the first failed step. |
| `n_steps` | Number of configured pipeline steps. |
| `n_steps_run` | Number of steps actually run. |
| `completed_steps` | Step names that completed successfully. |
| `failed_steps` | Step names that failed. |
| `step_results` | Full step-level execution metadata. |
| `output_summary` | Compact summary from downstream artifacts. |

## Output Summary

When the run succeeds, the runner collects a compact summary across the artifact chain:

```text
claims
selected_claims
narrative
audit
repair
traceability
```

Example expected summary for the current deterministic CPI/FRED path:

```text
n_claims = 4
n_selected_claims = 4
audit_pass = true
repair_needed = false
n_traceability_rows = 4
```

## Failure Behavior

By default, the runner stops after the first failed step:

```text
stop_on_failure = true
```

This can be disabled with:

```bash
python benchmarks/run_fred_evidence_loop.py \
  --input-context benchmarks/data/fred_macro_context.json \
  --comparison-window 12m \
  --no-stop-on-failure
```

In normal demo and release-gate usage, the default stop-on-failure behavior should be preferred.

## Current Limitations

v0.1 does not yet:

- isolate each run into a fully separate output directory
- pass custom output paths to every downstream script
- clean previous artifacts before running
- support multiple comparison windows in one run
- support LLM narrative mode
- enforce numeric/directional audit checks
- emit a human-readable run report
- render dashboard views
- integrate with model profiles or repair policies

These are intentionally deferred.

## Future Extensions

Likely future extensions include:

```text
v1.6.7 - numeric and directional audit checks
v1.6.8 - optional LLM narrative mode under existing citation constraints
v1.6.9 - dashboard/demo-readiness pass
v1.7.0 - CPI demo loop
```

Longer-term, the runner may incorporate:

- run-specific output directories
- multi-window runs
- artifact cleanup controls
- config-file driven workflows
- model selection / routing
- repair execution
- dashboard launch hooks
- token/probabilistic telemetry in v2.0