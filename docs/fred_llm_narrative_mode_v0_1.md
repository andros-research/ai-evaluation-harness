# FRED LLM Narrative Mode v0.1

## Purpose

The FRED LLM narrative mode adds an optional local LLM generation path to the FRED evidence loop.

Earlier narrative generation was deterministic and template-based. v0.1 preserves deterministic generation as the default while adding an opt-in LLM mode using a local Ollama model.

The purpose of this layer is to introduce model-generated prose inside the existing claim, citation, audit, repair, and traceability guardrails.

## Default Behavior

Deterministic mode remains the default:

```bash
python benchmarks/generate_fred_narrative_from_claims.py
```

Equivalent explicit command:

```bash
python benchmarks/generate_fred_narrative_from_claims.py \
  --mode deterministic
```

The full runner also defaults to deterministic mode:

```bash
python benchmarks/run_fred_evidence_loop.py \
  --input-context benchmarks/data/fred_macro_context.json \
  --comparison-window 12m
```

## LLM Mode

LLM mode is opt-in:

```bash
python benchmarks/generate_fred_narrative_from_claims.py \
  --mode llm \
  --model llama3 \
  --ollama-host http://127.0.0.1:11434
```

The full FRED evidence loop can also run in LLM mode:

```bash
python benchmarks/run_fred_evidence_loop.py \
  --input-context benchmarks/data/fred_macro_context.json \
  --comparison-window 12m \
  --narrative-mode llm \
  --narrative-model llama3 \
  --ollama-host http://127.0.0.1:11434
```

## Pipeline Position

```text
selected_fred_claims.json
  -> generate_fred_narrative_from_claims.py
     -> deterministic mode OR llm mode
  -> fred_narrative.md
  -> fred_narrative_metadata.json
  -> audit_fred_narrative.py
  -> fred_narrative_audit.json
  -> plan_fred_narrative_repair.py
  -> fred_repair_plan.json
  -> build_fred_traceability_summary.py
  -> fred_traceability_summary.csv/json/metadata.json
```

## LLM Provider

v0.1 uses local Ollama generation.

Default settings:

```text
ollama_host = http://127.0.0.1:11434
model = llama3
temperature = 0.0
top_p = 0.9
num_predict = 512
timeout_s = 600
```

## LLM Prompt Contract

The LLM prompt instructs the model to:

- write markdown only
- include `# FRED Macro Narrative`
- include `## Claim-Cited Summary`
- use one bullet per selected claim
- cite every bullet with exactly one `[CLAIMS: claim_id]` block
- preserve current value, prior value, delta magnitude, and direction
- avoid unsupported causal interpretation
- avoid adding facts not present in the selected claims
- avoid citing unknown claim IDs

## Metadata

The narrative metadata records whether LLM mode was used:

| Field | Description |
|---|---|
| `generation_mode` | `deterministic` or `llm`. |
| `generation_method` | `deterministic_claim_bullets` or `llm_claim_cited_narrative`. |
| `llm_metadata.llm_used` | Whether an LLM was used. |
| `llm_metadata.model` | Local model used for LLM generation. |
| `llm_metadata.ollama_host` | Ollama host used for generation. |
| `llm_metadata.elapsed_s` | LLM generation elapsed time. |
| `llm_metadata.error` | Error string, if any. |

The runner metadata also records:

| Field | Description |
|---|---|
| `narrative_mode` | Narrative generation mode passed through the runner. |
| `narrative_model` | LLM model used when runner mode is `llm`. |
| `ollama_host` | Ollama host used when runner mode is `llm`. |
| `narrative_timeout_s` | LLM timeout used when runner mode is `llm`. |

## Guardrails

LLM mode is evaluated by the same downstream guardrails as deterministic mode:

```text
citation validation
numeric/directional audit
repair planning
traceability summary
runner metadata
```

This means the LLM is allowed to generate prose, but the artifact chain still verifies whether the output remains claim-cited and source-consistent.

## Current Limitations

v0.1 does not yet:

- use LLM mode by default
- support multiple LLM providers
- isolate LLM outputs into separate run directories
- retry failed generations
- automatically repair failed LLM narratives
- classify unsupported interpretation
- compare deterministic and LLM narratives side by side
- expose LLM mode in a dashboard
- collect token-level probabilities

These are intentionally deferred.

## Known Risk: Interpretive Additions

LLM-generated prose may add interpretive language beyond the selected claims.

Examples of interpretation risk include:

```text
significant upward trend
labor market deterioration
monetary policy tightening
market expectations
```

These phrases may be plausible, but they are not always directly supported by the selected FRED claim itself.

Future audit layers may classify this as interpretation risk rather than immediate failure.

## Future Extensions

Likely future extensions include:

```text
v1.6.9 - dashboard/demo-readiness pass
v1.7.0 - CPI demo loop
v1.7.x - interpretation audit / warning layer
v1.8.x - broader source ingestion
v2.0 - token/probabilistic telemetry
```

Longer-term, LLM narrative mode may incorporate:

- constrained JSON intermediate output
- side-by-side deterministic vs LLM comparison
- automatic repair on failed audit
- interpretation-risk classification
- model profile routing
- token-level confidence diagnostics