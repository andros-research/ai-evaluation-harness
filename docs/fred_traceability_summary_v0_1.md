# FRED Traceability Summary v0.1

## Purpose

The FRED traceability summary layer joins the v1.6 artifact spine into a single claim-level evidence map.

This layer converts the pipeline from a sequence of independent artifacts into an inspectable source-to-narrative trace. It shows how each FRED-native claim moves through selection, narrative citation, audit, and repair planning.

The purpose of this layer is to make the end-to-end chain visible for dashboarding, debugging, and future demo use.

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
```

## Input Artifacts

Default claims input:

```text
benchmarks/results/fred_claims/fred_claims.json
```

Default selected-claims input:

```text
benchmarks/results/fred_claims/selected_fred_claims.json
```

Default audit input:

```text
benchmarks/results/fred_audits/fred_narrative_audit.json
```

Default repair-plan input:

```text
benchmarks/results/fred_repairs/fred_repair_plan.json
```

These files are produced by the upstream sequence:

```bash
python benchmarks/build_fred_claims.py \
  --input-context benchmarks/data/fred_macro_context.json \
  --comparison-window 12m

python benchmarks/select_fred_claims.py

python benchmarks/generate_fred_narrative_from_claims.py

python benchmarks/audit_fred_narrative.py

python benchmarks/plan_fred_narrative_repair.py
```

## Output Artifacts

Default outputs:

```text
benchmarks/results/fred_traceability/fred_traceability_summary.csv
benchmarks/results/fred_traceability/fred_traceability_summary.json
benchmarks/results/fred_traceability/fred_traceability_summary_metadata.json
```

## Traceability Method

v0.1 uses a claim-selection-narrative-audit join:

```text
traceability_method = claim_selection_narrative_audit_join
```

Each traceability row starts from a source-grounded claim and joins downstream state:

```text
claim
  -> selected claim rank
  -> narrative bullet citation
  -> audit citation status
  -> repair status
```

## Traceability Row Fields

Each row may include:

| Field | Description |
|---|---|
| `claim_id` | Stable FRED-native claim identifier. |
| `source_type` | Source type, initially `fred`. |
| `source_series` | FRED series ID. |
| `source_series_name` | Human-readable series name. |
| `source_observation_date` | Current observation date. |
| `claim_type` | Claim type, such as `prior_comparison`. |
| `metric_name` | Metric name, such as `cpi_yoy`. |
| `comparison_window` | Comparison window, such as `12m`. |
| `current_value` | Current source value. |
| `prior_value` | Prior comparison value. |
| `delta_value` | Difference between current and prior values. |
| `direction` | Direction of change. |
| `claim_text` | Human-readable claim statement. |
| `eligible_for_narrative` | Whether the claim was eligible for narrative use. |
| `was_selected` | Whether the claim appeared in the selected evidence set. |
| `selection_rank` | Rank in selected evidence set. |
| `selection_method` | Selection method used. |
| `was_cited` | Whether the claim was cited in the narrative. |
| `citation_count` | Number of narrative citations for the claim. |
| `narrative_bullet_index` | First cited bullet index. |
| `narrative_bullet_text` | First cited bullet text. |
| `audit_citation_status` | Bullet-level citation audit status. |
| `audit_issue_type` | Bullet-level audit issue, if any. |
| `repair_needed` | Whether the repair plan says repair is needed. |
| `repair_action_count` | Number of repair actions linked to the claim. |
| `repair_action_ids` | Repair action IDs linked to the claim. |

## Metadata

The metadata artifact records:

| Field | Description |
|---|---|
| `traceability_schema_version` | Traceability schema version, initially `fred_traceability_summary_v0_1`. |
| `traceability_method` | Traceability construction method. |
| `generated_at` | UTC timestamp when the summary was generated. |
| `n_claims` | Number of input claims. |
| `n_selected_claims` | Number of selected claims. |
| `n_traceability_rows` | Number of traceability rows. |
| `n_cited_claims` | Number of claims cited in the narrative. |
| `n_uncited_claims` | Number of claims not cited in the narrative. |
| `audit_pass` | Audit pass/fail status. |
| `repair_needed` | Whether repair is needed. |
| `input_files` | Paths to source artifacts. |
| `output_files` | Paths to generated traceability artifacts. |

## Current Limitations

v0.1 does not yet:

- include raw source context values beyond claim-level fields
- include unreconciled narrative text outside bullets
- support multiple narratives in one run
- support multiple comparison windows in one summary
- include numeric/directional audit results
- include applied repair status
- render a dashboard view
- score narrative quality or importance

These are intentionally deferred.

## Future Extensions

Likely future extensions include:

```text
v1.6.6 - workflow runner for the full FRED artifact chain
v1.6.7 - numeric and directional audit checks
v1.6.8 - optional LLM narrative mode under citation constraints
v1.6.9 - v1.7 readiness and CPI demo packaging
v1.7.0 - CPI demo loop
```

Longer-term, the traceability layer may incorporate:

- dashboard tables
- source-value drilldowns
- multi-window comparisons
- audit severity summaries
- repair status summaries
- LLM narrative variants
- model profile links
- token/probabilistic telemetry in v2.0