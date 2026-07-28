# v1.8.0 Cross-Model Comparison Runner

## Purpose

v1.8.0 introduces a config-driven runner for executing the same evidence-loop
task across multiple models and controlled experimental settings.

The goal is to replace the current manual process:

```text
run one model
→ inspect output
→ copy artifacts
→ run next model
→ manually compare reports
```

with a reproducible comparison workflow:

```text
load comparison configuration
→ freeze evidence context
→ run configured models and repetitions
→ preserve isolated artifacts
→ normalize run-level results
→ generate comparison summaries
```

This creates the experimental foundation for:

- repeated cross-model evaluation,
- longitudinal model profiles,
- prompt-contract robustness testing,
- repair-loop evaluation,
- interpretation-risk metrics,
- token and log-probability telemetry.

## Initial task scope

The first comparison task remains the current FRED evidence loop:

- source context: local generated FRED macro context,
- comparison window: 12 months,
- selected claims: CPI, unemployment, fed funds, and the 10-year Treasury yield,
- deterministic baseline,
- local LLM narrative generation,
- citation and numeric/directional audit,
- repair planning,
- traceability,
- demo report.

No new macro data sources are required for v1.8.0.

## Minimum viable experiment configuration

The comparison runner should accept a configuration describing:

```yaml
comparison_id: fred_12m_contract_smoke_test

input_context: benchmarks/data/fred_macro_context.json
comparison_window: 12m

runs:
  - mode: deterministic
    label: deterministic

  - mode: llm
    model: llama3
    label: llama3

  - mode: llm
    model: mistral
    label: mistral

  - mode: llm
    model: llama3:70b
    label: llama3_70b

temperature: 0.0
repetitions: 1
timeout_s: 600
ollama_host: http://127.0.0.1:11434
```

Later versions may support multiple temperatures, prompt variants, windows, and
repetition counts within one configuration.

## Proposed artifact structure

Each comparison must preserve runs independently.

```text
benchmarks/results/model_comparisons/
  <comparison_id>/
    comparison_config.json
    comparison_manifest.json

    runs/
      deterministic/
        repetition_001/
          run.json
          narrative.md
          narrative_metadata.json
          audit.json
          repair_plan.json
          traceability.json
          demo_report.md

      llama3/
        repetition_001/
          ...

      mistral/
        repetition_001/
          ...

      llama3_70b/
        repetition_001/
          ...

    comparison_rows.jsonl
    comparison_summary.csv
    comparison_summary.json
    comparison_report.md
```

The comparison runner must never overwrite another model's artifacts.

## Normalized comparison row

Each completed or failed model run should produce one normalized record.

Initial fields:

```text
comparison_id
run_id
run_label
mode
model
comparison_window
temperature
repetition
run_started_at
run_finished_at
elapsed_seconds
run_completed
failure_stage
audit_pass
repair_needed
n_claims
n_selected_claims
n_bullets
n_citations
n_unknown_citations
n_content_mismatches
content_issue_counts
n_repair_actions
prompt_hash
context_hash
git_commit
artifact_directory
```

Future fields may include:

```text
output_token_count
interpretive_span_count
interpretation_risk_score
logprob_summary
repair_attempt_count
repair_success
```

## Operational semantics

The comparison runner should distinguish between:

```text
run_completed
audit_pass
accepted_output
repair_needed
```

A workflow may complete successfully even when the resulting narrative fails
audit.

For example:

```text
run_completed = true
audit_pass = false
accepted_output = false
repair_needed = true
```

One model failure must not stop the remaining comparison runs.

The runner should:

1. record the failed stage,
2. preserve all available artifacts,
3. write a normalized failed-run row,
4. continue to the next configured run.

## Acceptance criteria

v1.8.0 is complete when:

1. One configuration launches deterministic, llama3, Mistral, and llama3:70b.
2. All models use the same frozen FRED context and comparison window.
3. Each run receives an isolated artifact directory.
4. One failed model does not terminate the comparison experiment.
5. Every run produces a normalized comparison row.
6. The runner writes JSONL, JSON, CSV, and Markdown comparison artifacts.
7. The comparison report shows pass/fail, repair status, issue counts, and elapsed time.
8. Artifacts include context hash, prompt hash, Git commit, model, temperature, and repetition.
9. A one-repetition four-run smoke test completes successfully.
10. Results are suitable for later dashboard ingestion.

## First smoke test

The initial release gate is intentionally small:

```text
1 deterministic run
1 llama3 run
1 Mistral run
1 llama3:70b run
temperature = 0.0
comparison window = 12m
```

Expected total:

```text
4 runs
```

This validates orchestration and artifact handling before generating a larger
dataset.

## First dataset pilot

After the smoke test, run a controlled contract-robustness pilot:

```text
Models:
- llama3
- mistral
- llama3:70b

Prompt variants:
- original weak contract
- intermediate contract
- current hardened contract

Temperatures:
- 0.0
- 0.7

Repetitions:
- 5

Comparison window:
- 12m
```

Total:

```text
3 models × 3 prompt variants × 2 temperatures × 5 repetitions
= 90 LLM runs
```

The deterministic baseline should also be run once for the frozen evidence
context.

## Pilot research questions

The initial dataset should help answer:

1. How often does each model satisfy the narrative contract?
2. How much does prompt hardening improve pass rates?
3. Which failure types recur by model?
4. Does higher temperature increase structural or factual failures?
5. How variable are outputs under identical evidence?
6. Does llama3:70b show more consistent contract compliance?
7. Which failures can be repaired deterministically?
8. Which failures require model regeneration?
9. Do models differ mainly in factual compliance, structure, or interpretation?

## Relationship to model profiles

The normalized comparison dataset should eventually support a model-profile
summary containing:

- total runs,
- contract pass rate,
- audit pass rate,
- repair-needed rate,
- issue-code frequency,
- median and percentile latency,
- output-length distribution,
- performance by temperature,
- performance by prompt variant,
- repeated-run consistency.

Later versions may add:

- interpretation behavior,
- uncertainty metrics,
- token-level probability characteristics,
- repair responsiveness.

## Dashboard direction

A comparison dashboard tab is a likely follow-on milestone, but experiment
execution must remain outside the dashboard.

The dashboard should read durable comparison artifacts and provide:

- experiment selector,
- model and prompt filters,
- pass-rate summary,
- repair-rate summary,
- issue counts by model,
- latency comparison,
- run-level table,
- side-by-side narratives,
- expandable audit and repair artifacts,
- links to traceability details.

The dashboard should not own orchestration logic.

## Non-goals

v1.8.0 does not yet include:

- executable narrative repair,
- automatic re-audit after repair,
- interpretation-risk classification,
- token or log-probability capture,
- new macro data sources,
- Fed speech ingestion,
- parallel model execution,
- large-scale dashboard redesign.

## Likely follow-on milestones

```text
v1.8.0
Cross-model comparison runner and normalized artifacts

v1.8.1
Repeated-run dataset generation and dashboard comparison tab

v1.8.2
Controlled fault injection and executable factual repair

v1.9
Interpretation taxonomy and interpretation-aware audit

v2.0
Token and probability telemetry
```