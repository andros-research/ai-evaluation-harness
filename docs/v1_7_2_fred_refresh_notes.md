# v1.7.2 FRED Data Refresh Notes

## Purpose

v1.7.2 validates that the FRED evidence loop continues to operate correctly after refreshing the underlying macro data.

The goal of this version was not to expand the number of data sources or add a new workflow layer. It was to:

1. regenerate the local FRED macro context,
2. rerun the existing evidence loop against a newer data snapshot,
3. compare behavior with the earlier v1.7.0 and v1.7.1 runs,
4. investigate any new model-specific failures,
5. improve observability and prompt reliability where needed.

This became a useful “wake the system back up” exercise after a period away from the project.

## Refreshed FRED context

The FRED prompt-context builder was rerun using:

```bash
python benchmarks/build_fred_prompt_context.py
```

The builder fetches live FRED observations for four configured series:

- `CPIAUCSL`
- `UNRATE`
- `FEDFUNDS`
- `GS10`

It constructs:

- a latest snapshot,
- a 6-month comparison,
- a 12-month comparison,
- a 24-month comparison.

The refreshed local context used a latest observation date of:

```text
2026-06-01
```

The generated file is:

```text
benchmarks/data/fred_macro_context.json
```

This file is intentionally ignored by Git through the existing `data/` ignore rule. It is treated as a generated local input rather than a tracked source artifact.

## Refreshed 12-month claims

The refreshed 12-month comparison produced four selected claims:

| Series | Prior | Current | Delta | Direction |
| --- | ---: | ---: | ---: | --- |
| CPIAUCSL | 2.68 | 4.167 | +1.487 | up |
| UNRATE | 4.10 | 4.20 | +0.10 | up |
| FEDFUNDS | 4.33 | 3.63 | -0.70 | down |
| GS10 | 4.38 | 4.47 | +0.09 | up |

The GS10 claim changed direction relative to the earlier demo snapshot. The refreshed claim layer correctly represented the new source data.

## Deterministic release-gate result

The deterministic evidence loop completed successfully against the refreshed context.

Result:

```text
overall_ok=True
failed_steps=0
audit_pass=True
repair_needed=False
n_traceability_rows=4
n_cited_claims=4
```

All four claims were selected, cited, audited as supported, and represented in the traceability output.

The deterministic narrative preserved:

- prior value,
- current value,
- delta magnitude,
- direction,
- exact claim citation.

## Initial llama3 failure after refresh

The first refreshed-data llama3 run failed during narrative citation validation.

The validator reported that all four selected claim IDs were missing from the narrative citations.

At first, this appeared to suggest that the model had failed to use the selected claims. Additional diagnostics showed that this interpretation was incorrect.

The model had reproduced all four correct claim IDs, but used the wrong citation wrapper:

```text
[fred__claim_id]
```

instead of the required contract:

```text
[CLAIMS: fred__claim_id]
```

The same output also:

- combined CPI and unemployment into one bullet,
- omitted prior values,
- added unsupported interpretation,
- produced fewer bullets than selected claims.

This was therefore not primarily an evidence-selection failure. It was a serialization and output-contract failure.

## Failed-output diagnostics

The narrative generator was updated to improve failure observability.

When citation validation fails, the error now records:

- expected selected claim IDs,
- extracted narrative claim IDs,
- expected citation count,
- extracted citation count.

The script also preserves failed model output in:

```text
benchmarks/results/fred_narratives/fred_narrative_failed_validation.md
```

and writes supporting debug metadata to:

```text
benchmarks/results/fred_narratives/fred_narrative_failed_validation_metadata.json
```

This change turned an opaque validation error into an inspectable model failure.

## Preserved failure fixture

The original llama3 failure was copied into a tracked fixture directory:

```text
benchmarks/fixtures/fred_narrative_failures/
```

The preserved files include:

```text
llama3_2026-07-09_bare-citation-format.md
llama3_2026-07-09_bare-citation-format.metadata.json
README.md
```

The fixture records a representative case where the model was semantically grounded in the selected evidence but failed the exact machine-readable output contract.

This fixture can later support:

- regression tests,
- parser tests,
- repair-layer tests,
- model-behavior documentation.

## Prompt hardening

`build_llm_prompt()` was strengthened to make the narrative contract explicit.

The revised prompt now requires:

1. exactly one bullet per selected claim,
2. exactly the expected number of bullets,
3. prior value in every bullet,
4. current value in every bullet,
5. delta magnitude in every bullet,
6. direction in every bullet,
7. exact `[CLAIMS: <claim_id>]` syntax,
8. no bare `[claim_id]` citations,
9. no combined claims,
10. no unsupported causal or market interpretation,
11. no extra narrative sections after the controlled bullets.

The prompt also includes a concrete required bullet template and a negative example of the invalid bare-citation format.

## llama3 result after prompt hardening

After the prompt update, the standalone llama3 narrative generator completed successfully.

It produced:

- four bullets,
- four exact claim citations,
- one claim per bullet,
- prior and current values,
- delta and direction,
- no unsupported interpretation.

The full refreshed-data llama3 evidence loop then completed successfully:

```text
overall_ok=True
completed_steps=7
failed_steps=0
audit_pass=True
repair_needed=False
n_traceability_rows=4
```

The final saved llama3 report was generated from run:

```text
fred_evidence_loop_20260723T015903Z
```

The report correctly identified:

```text
Narrative model: llama3
Audit pass: true
Repair needed: false
Cited claims: 4
```

The generated narrative contained four machine-auditable bullets with exact claim IDs and no content mismatches.

This demonstrated that the failure was recoverable through a clearer output contract without weakening the citation validator.

## Final model-comparison gate

The refreshed-data release gate was rerun across:

- deterministic mode,
- llama3,
- Mistral,
- llama3:70b.

The hardened prompt materially improved contract compliance.

### Deterministic

Result:

```text
audit_pass=True
repair_needed=False
content_mismatches=0
```

The deterministic mode remained the stable reference baseline.

### llama3

The previously observed refreshed-data citation-wrapper failure was corrected after prompt hardening.

The final llama3 run completed the full evidence loop successfully with:

```text
audit_pass=True
repair_needed=False
content_mismatches=0
cited_claims=4
```

The resulting narrative was structurally compliant and fully traceable.

Its wording remained somewhat mechanical, for example:

```text
CPI year-over-year inflation up by...
```

rather than:

```text
CPI year-over-year inflation increased by...
```

This is a prose-quality issue rather than an evidence-contract failure.

### Mistral

Mistral had previously exhibited a repeatable failure mode:

- correct citations,
- correct deltas,
- missing prior values,
- missing current values in the controlled claim-cited bullets.

Under the hardened prompt, Mistral produced four complete claim-cited bullets containing prior value, current value, delta, direction, and exact citation syntax.

Result:

```text
audit_pass=True
repair_needed=False
content_mismatches=0
```

The earlier Mistral failure signature did not recur in this release-gate run.

### llama3:70b

llama3:70b also produced four complete, contract-following bullets.

Result:

```text
audit_pass=True
repair_needed=False
content_mismatches=0
```

The output closely followed the structured claim text and required citation format.

## Model-comparison interpretation

The final outputs were more similar than the earlier v1.7.1 comparison.

This is an expected consequence of tightening the prompt contract. The models had less freedom to vary in structure, interpretation, or citation style.

The comparison therefore shifted from:

> Which model naturally follows the weakly specified evidence contract?

to:

> Can a stronger workflow contract make multiple local models reliably produce machine-auditable output?

For this narrow four-claim task, the answer was yes.

This does not mean that the models are equivalent. It means that a sufficiently explicit prompt reduced the model-specific failure modes visible in this particular workflow.

## Current limitations

The current audit validates:

- citation coverage,
- known claim IDs,
- prior values,
- current values,
- delta values,
- direction.

It does not yet fully validate:

- qualitative significance,
- policy interpretation,
- causal inference,
- market interpretation,
- narrative usefulness beyond contract compliance.

The final LLM outputs are accurate and auditable, but intentionally constrained and somewhat mechanical.

This is acceptable for v1.7.2. The objective of this version was refreshed-data reliability and evidence-contract validation, not polished macro prose.

## Main lesson

The refreshed FRED data did not break the core evidence loop.

Instead, the refresh exposed a new LLM output-contract failure. The harness stopped the invalid artifact, diagnostics made the failure observable, the failed output was preserved, and prompt hardening restored a complete passing workflow.

The key distinction was:

> The model used the right evidence, but initially serialized it in the wrong form.

That distinction is central to the project. A plausible or semantically grounded response is not automatically a valid workflow artifact. The surrounding system defines the contract, validates it, and preserves failures for analysis.

## v1.7.2 completion status

Completed:

- refreshed FRED macro context,
- deterministic refreshed-data run,
- initial llama3 failure investigation,
- failed-output diagnostics,
- failed narrative fixture,
- prompt hardening,
- successful full llama3 rerun,
- successful deterministic release gate,
- successful Mistral release gate,
- successful llama3:70b release gate,
- correctly saved llama3 comparison report,
- saved model-comparison reports.

Remaining before tagging:

- review Git status,
- commit this document,
- push the final branch state,
- tag v1.7.2.

## Suggested release description

v1.7.2 refreshes the local FRED macro context and validates the complete evidence loop against a newer June 2026 snapshot. It adds durable diagnostics for failed narrative citation validation, preserves a representative llama3 failure fixture, strengthens the LLM output contract, and verifies successful deterministic and multi-model claim-cited runs with audit and traceability.