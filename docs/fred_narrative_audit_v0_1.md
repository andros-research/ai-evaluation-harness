# FRED Narrative Audit v0.1

## Purpose

The FRED narrative audit layer independently validates a generated FRED narrative against the selected FRED claim set.

This is the first audit step after the v1.6.2 FRED narrative generation layer. The initial v0.1 auditor focuses only on citation coverage and claim ID validity. It does not yet evaluate numeric accuracy, semantic faithfulness, direction correctness, or unsupported macro interpretation.

The purpose of this layer is to separate narrative generation from narrative verification.

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
  -> future repair
```

## Input Artifacts

Default narrative input:

```text
benchmarks/results/fred_narratives/fred_narrative.md
```

Default selected-claims input:

```text
benchmarks/results/fred_claims/selected_fred_claims.json
```

These files are produced by the upstream sequence:

```bash
python benchmarks/build_fred_claims.py \
  --input-context benchmarks/data/fred_macro_context.json \
  --comparison-window 12m

python benchmarks/select_fred_claims.py

python benchmarks/generate_fred_narrative_from_claims.py
```

## Output Artifacts

Default output:

```text
benchmarks/results/fred_audits/fred_narrative_audit.json
```

## Audit Method

v0.1 uses a citation coverage audit:

```text
audit_method = citation_coverage_audit
```

The auditor extracts all markdown bullet lines from the narrative and inspects claim citation blocks of the form:

```text
[CLAIMS: claim_id]
```

It then checks:

- whether each bullet has a claim citation
- whether every cited claim ID exists in the selected claim set
- whether duplicate claim citations appear
- whether every selected claim is cited, when strict selected-claim coverage is enabled

## Audit Output Fields

The audit artifact records:

| Field | Description |
|---|---|
| `audit_schema_version` | Audit schema version, initially `fred_narrative_audit_v0_1`. |
| `audit_method` | Audit method, initially `citation_coverage_audit`. |
| `audit_pass` | Boolean pass/fail result. |
| `errors` | List of top-level audit error categories. |
| `strict_selected_claim_coverage` | Whether all selected claims must be cited. |
| `n_selected_claims` | Number of selected claims loaded. |
| `n_bullets` | Number of markdown bullets extracted from the narrative. |
| `n_citations` | Total number of claim citations extracted. |
| `n_unique_citations` | Number of unique cited claim IDs. |
| `selected_claim_ids` | Claim IDs from the selected claim set. |
| `cited_claim_ids` | Claim IDs cited by the narrative. |
| `unknown_citations` | Cited IDs not found in selected claims. |
| `duplicate_citations` | Claim IDs cited more than once. |
| `selected_claims_missing_from_narrative` | Selected claims not cited in the narrative. |
| `bullet_audits` | Per-bullet citation audit records. |

## Bullet-Level Audit Records

Each bullet audit includes:

| Field | Description |
|---|---|
| `bullet_text` | Full markdown bullet text. |
| `cited_claim_ids` | Claim IDs cited by the bullet. |
| `has_claim_citation` | Whether the bullet includes a `[CLAIMS: ...]` block. |
| `unknown_claim_ids` | Cited IDs not found in the selected claim set. |
| `citation_status` | `supported` if citation coverage passes, otherwise `failed`. |
| `issue_type` | Failure category such as `missing_claim_citation` or `unknown_claim_id`. |

## Current Limitations

v0.1 does not yet:

- verify numeric values in narrative text against supporting values
- verify direction words such as increased/decreased against claim direction
- detect unsupported macro interpretation
- detect causal language
- compare narrative text semantically against claim text
- recommend repairs
- generate repaired narrative artifacts
- audit LLM-generated paragraph prose beyond citation coverage

These are intentionally deferred.

## Future Extensions

Likely future extensions include:

```text
v1.6.4 - FRED narrative repair scaffold
v1.6.5 - source-to-claim-to-narrative traceability summary
v1.6.6 - numeric and directional audit checks
v1.7.0 - CPI demo loop using claim, selection, narrative, audit, and repair artifacts
```

Longer-term, the audit layer may incorporate:

- numeric tolerance checks
- direction consistency checks
- claim text overlap diagnostics
- unsupported interpretation detection
- causal language checks
- repair recommendations
- model-profile-aware audit severity
- dashboard traceability views