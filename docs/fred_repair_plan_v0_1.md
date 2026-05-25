# FRED Repair Plan v0.1

## Purpose

The FRED repair plan layer converts a FRED narrative audit artifact into an explicit repair plan.

This is the first repair step after the v1.6.3 FRED narrative audit layer. The initial v0.1 repair planner does not rewrite the narrative. It only determines whether repair is needed and maps audit failure categories to planned repair actions.

The purpose of this layer is to create a durable handoff between audit and future repair execution.

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
  -> future repaired narrative artifact
```

## Input Artifacts

Default audit input:

```text
benchmarks/results/fred_audits/fred_narrative_audit.json
```

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

python benchmarks/audit_fred_narrative.py
```

## Output Artifacts

Default output:

```text
benchmarks/results/fred_repairs/fred_repair_plan.json
```

## Repair Method

v0.1 uses a citation-audit repair planner:

```text
repair_method = citation_audit_repair_plan
```

The planner reads the audit result and determines whether any repair actions are needed.

If the audit passes:

```text
repair_needed = false
repair_actions = []
```

If the audit fails, the planner maps audit issues to planned repair actions.

## Initial Repair Action Mapping

| Audit Issue | Planned Repair Action |
|---|---|
| `missing_claim_citation` | `add_valid_claim_citation_or_remove_bullet` |
| `unknown_claim_id` | `replace_unknown_claim_id_or_remove_citation` |
| `selected_claim_missing_from_narrative` | `add_claim_bullet_or_relax_strict_coverage` |
| `duplicate_claim_citation` | `deduplicate_claim_citation` |
| unknown issue | `manual_review_required` |

## Repair Plan Fields

The repair plan artifact records:

| Field | Description |
|---|---|
| `repair_plan_schema_version` | Repair plan schema version, initially `fred_repair_plan_v0_1`. |
| `repair_method` | Repair planning method, initially `citation_audit_repair_plan`. |
| `planned_at` | UTC timestamp when the repair plan was generated. |
| `repair_needed` | Whether the audit result requires repair. |
| `audit_pass` | Pass/fail status copied from the audit artifact. |
| `audit_errors` | Top-level audit error categories. |
| `n_selected_claims` | Number of selected claims available. |
| `n_known_claim_ids` | Number of known selected claim IDs. |
| `n_repair_actions` | Number of planned repair actions. |
| `repair_actions` | List of planned repair actions. |
| `inputs_summary` | Summary of input artifact metadata. |
| `input_files` | Paths to audit, narrative, and selected-claims inputs. |
| `output_files` | Path to the generated repair plan artifact. |

## Repair Action Records

Each repair action may include:

| Field | Description |
|---|---|
| `action_id` | Stable action identifier within the plan. |
| `action_scope` | Scope of action, such as `bullet` or `narrative`. |
| `bullet_index` | Bullet index when action applies to a specific bullet. |
| `issue_type` | Audit issue requiring repair. |
| `repair_action` | Planned repair action. |
| `bullet_text` | Bullet text, when applicable. |
| `claim_id` | Claim ID, when applicable. |
| `cited_claim_ids` | Claim IDs cited by the affected bullet. |
| `unknown_claim_ids` | Unknown claim IDs, when applicable. |
| `status` | Initial status, usually `planned`. |

## Current Limitations

v0.1 does not yet:

- rewrite the narrative
- apply repairs automatically
- generate a repaired narrative artifact
- verify numeric or directional content
- decide between alternative valid repairs
- use LLMs for repair
- use model profile information
- use repair policy recommendations from earlier behavioral profiling

These are intentionally deferred.

## Future Extensions

Likely future extensions include:

```text
v1.6.5 - source-to-claim-to-narrative traceability summary
v1.6.6 - numeric and directional audit checks
v1.6.7 - optional LLM narrative mode under existing citation constraints
v1.6.8 - applied repair for failed narrative audits
v1.7.0 - CPI demo loop using claim, selection, narrative, audit, and repair artifacts
```

Longer-term, the repair layer may incorporate:

- automatic citation insertion
- removal or rewriting of unsupported bullets
- missing-claim bullet generation
- numeric correction
- direction correction
- constrained LLM repair
- repair success/failure telemetry
- repair strategy selection informed by model profiles