# FRED Numeric and Directional Audit v0.1

## Purpose

The FRED numeric and directional audit layer extends the narrative audit beyond citation coverage.

Earlier versions of the FRED narrative audit verified that each narrative bullet cited known selected claims. v0.1 numeric and directional checks add a basic content-preservation layer: cited bullets should preserve the core numeric values and direction of the underlying selected claim.

The purpose of this layer is to catch simple factual drift before introducing optional LLM-generated narrative mode.

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

## Audit Method

The audit still uses the existing audit artifact:

```text
audit_method = citation_coverage_audit
```

However, v0.1 now adds claim-content checks inside each bullet audit.

For each cited claim, the auditor checks whether the bullet contains:

```text
current_value
prior_value
delta_value
direction term
```

The auditor records these checks under:

```text
bullet_audits[].content_audits
```

## Content Audit Fields

Each claim-level content audit record includes:

| Field | Description |
|---|---|
| `claim_id` | Cited claim being checked. |
| `current_value` | Current value from the selected claim. |
| `prior_value` | Prior comparison value from the selected claim. |
| `delta_value` | Difference between current and prior values. |
| `direction` | Structured direction field from the selected claim. |
| `current_value_present` | Whether current value appears in the bullet text. |
| `prior_value_present` | Whether prior value appears in the bullet text. |
| `delta_value_present` | Whether delta magnitude appears in the bullet text. |
| `direction_term_present` | Whether an acceptable direction word appears in the bullet text. |
| `content_audit_pass` | Whether all content checks pass. |
| `content_issues` | List of content audit failures. |

## Direction Terms

v0.1 uses simple directional word matching.

For upward moves:

```text
increased
rose
higher
up
```

For downward moves:

```text
decreased
fell
lower
down
```

For flat moves:

```text
unchanged
flat
little changed
```

## Signed Delta Handling

Structured claims store signed deltas.

For example:

```text
current_value = 3.64
prior_value = 4.33
delta_value = -0.69
direction = down
```

Natural-language prose usually expresses the delta as an unsigned magnitude while the direction word carries the sign:

```text
Effective federal funds rate decreased by 0.69 percentage points...
```

Therefore, v0.1 delta matching checks the absolute magnitude of numeric deltas. The sign is validated separately through the direction term.

This avoids incorrectly failing bullets that say `decreased by 0.69` when the structured delta is `-0.69`.

## New Audit Summary Fields

v0.1 adds:

| Field | Description |
|---|---|
| `n_bullets_with_content_mismatches` | Number of bullets with numeric or directional content issues. |
| `content_issue_counts` | Counts by content issue type. |

If any content mismatches are found, the top-level audit `errors` list includes:

```text
claim_content_mismatches
```

## Current Content Issue Types

Current issue types include:

| Issue | Meaning |
|---|---|
| `missing_current_value` | Bullet does not contain the selected claim current value. |
| `missing_prior_value` | Bullet does not contain the selected claim prior value. |
| `missing_delta_value` | Bullet does not contain the selected claim delta magnitude. |
| `missing_or_inconsistent_direction` | Bullet does not contain an accepted direction term. |

## Current Limitations

v0.1 does not yet:

- parse numbers robustly from arbitrary prose
- handle alternate rounded values beyond simple formatting
- infer semantic equivalence of direction words beyond a small term list
- validate units
- detect unsupported causal interpretation
- detect unsupported adjectives such as `clearly`, `sharply`, or `surprisingly`
- validate multi-claim synthesis
- evaluate non-bullet paragraph prose
- use model confidence or token-level probabilities

These are intentionally deferred.

## Future Extensions

Likely future extensions include:

```text
v1.6.8 - optional LLM narrative mode under citation and content-audit constraints
v1.6.9 - dashboard/demo-readiness pass
v1.7.0 - CPI demo loop
```

Longer-term, the numeric/directional audit layer may incorporate:

- tolerance-based numeric matching
- unit-aware validation
- phrase-level direction classification
- semantic entailment checks
- unsupported interpretation detection
- model-profile-aware severity
- token/probabilistic uncertainty diagnostics in v2.0