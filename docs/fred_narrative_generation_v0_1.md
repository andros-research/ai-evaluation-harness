# FRED Narrative Generation v0.1

## Purpose

The FRED narrative generation layer converts selected FRED-native claims into a durable, claim-cited narrative artifact.

This is the first narrative step after the v1.6.1 FRED claim selection layer. The initial v0.1 generator is intentionally deterministic and template-based. It does not yet use an LLM to synthesize prose. The goal is to establish the narrative artifact contract before introducing model variability.

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
  -> future narrative audit
  -> future repair
```

## Input Artifacts

Default input:

```text
benchmarks/results/fred_claims/selected_fred_claims.json
```

This file is produced by:

```bash
python benchmarks/select_fred_claims.py
```

A typical upstream sequence is:

```bash
python benchmarks/build_fred_claims.py \
  --input-context benchmarks/data/fred_macro_context.json \
  --comparison-window 12m

python benchmarks/select_fred_claims.py
```

## Output Artifacts

Default outputs:

```text
benchmarks/results/fred_narratives/fred_narrative.md
benchmarks/results/fred_narratives/fred_narrative_metadata.json
```

## Narrative Contract

The v0.1 narrative generator must preserve source traceability.

Each empirical bullet must include an explicit claim citation using the format:

```text
[CLAIMS: claim_id]
```

Example:

```text
- CPI year-over-year inflation increased by 0.904 percentage points versus the prior observation, from 2.382 to 3.286. [CLAIMS: fred__cpi_yoy__CPIAUCSL__prior_comparison__2026-03-01]
```

## Generation Method

v0.1 uses deterministic claim bullets:

```text
generation_method = deterministic_claim_bullets
```

Each selected claim is converted into one markdown bullet. The original `claim_text` is preserved, and the corresponding `claim_id` is appended as a citation.

## Metadata

The narrative metadata artifact records:

| Field | Description |
|---|---|
| `narrative_schema_version` | Narrative artifact schema version, initially `fred_narrative_v0_1`. |
| `generation_method` | Narrative generation method, initially `deterministic_claim_bullets`. |
| `input_file` | Path to the selected FRED claims artifact. |
| `generated_at` | UTC timestamp when the narrative was generated. |
| `n_selected_claims` | Number of selected claims available to the generator. |
| `used_claim_ids` | Claim IDs used in the narrative. |
| `output_files` | Paths to generated narrative artifacts. |

## Current Limitations

v0.1 does not yet:

- use an LLM for prose synthesis
- combine multiple claims into a single sentence
- rank or group claims by macro theme
- produce polished paragraph-style commentary
- audit citation coverage
- repair unsupported narrative bullets
- apply model-specific style or risk controls
- apply a final user voice layer

These are intentionally deferred.

## Future Extensions

Likely future extensions include:

```text
v1.6.3 - FRED-native narrative audit
v1.6.4 - narrative repair using selected FRED claims
v1.6.5 - dashboard traceability from source series to claims to narrative bullets
v1.7.0 - CPI demo loop using selected evidence, narrative, audit, and repair artifacts
```

Longer-term, the narrative generator may incorporate:

- LLM-generated paragraph synthesis
- strict claim citation constraints
- model profile routing
- repair-policy-aware generation
- narrative style controls
- final voice-layer transformation
- token/probabilistic telemetry in v2.0