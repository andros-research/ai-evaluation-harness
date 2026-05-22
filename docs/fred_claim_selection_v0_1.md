# FRED Claim Selection v0.1

## Purpose

The FRED claim selection layer converts deterministic FRED-native claim artifacts into a selected evidence set for downstream narrative generation.

This is the first selection step after the v1.6.0 FRED-native claim layer. It does not yet rank claims by importance, novelty, macro relevance, or narrative usefulness. The initial v0.1 selector intentionally uses a simple eligibility filter so the pipeline remains deterministic and auditable.

## Pipeline Position

```text
fred_macro_context.json
  -> build_fred_claims.py
  -> fred_claims.csv/json/metadata.json
  -> select_fred_claims.py
  -> selected_fred_claims.csv/json/metadata.json
  -> future narrative generation
  -> future narrative audit
  -> future repair
```

## Input Artifacts

Default input:

```text
benchmarks/results/fred_claims/fred_claims.json
```

This file is produced by:

```bash
python benchmarks/build_fred_claims.py \
  --input-context benchmarks/data/fred_macro_context.json \
  --comparison-window 12m
```

## Output Artifacts

Default outputs:

```text
benchmarks/results/fred_claims/selected_fred_claims.csv
benchmarks/results/fred_claims/selected_fred_claims.json
benchmarks/results/fred_claims/selected_fred_claims_metadata.json
```

## Selection Rule

v0.1 uses a deterministic eligibility filter:

```text
eligible_for_narrative == True
```

All eligible claims are selected. Input order is preserved.

## Added Selection Fields

Each selected claim preserves the original FRED claim fields and adds:

| Field | Description |
|---|---|
| `selection_rank` | 1-based rank based on preserved input order. |
| `claim_schema_version` | Original claim schema version copied from `schema_version`. |
| `selection_method` | Selection rule used, initially `eligible_for_narrative_filter`. |
| `selection_schema_version` | Selected-claim schema version, initially `fred_selected_claims_v0_1`. |
| `selected_at` | UTC timestamp when selection artifacts were generated. |

## Current Limitations

v0.1 does not yet:

- rank claims by macro importance
- remove redundant claims
- select across multiple comparison windows at once
- use model profile information
- use repair policy information
- optimize for narrative structure
- perform cross-series reasoning

These are intentionally deferred.

## Future Extensions

Likely future extensions include:

```text
v1.6.2 - narrative generation from selected FRED claims
v1.6.3 - FRED-native narrative audit
v1.6.4 - dashboard traceability from source series to selected claims
v1.7.0 - CPI demo loop using selected evidence artifacts
```

Longer-term, the selector may incorporate:

- claim type priority
- comparison-window priority
- macro theme grouping
- redundancy checks
- model-specific narrative risk controls
- repair-policy-aware selection