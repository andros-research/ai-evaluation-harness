# v1.8.2 Model Profiles Walkthrough

## Purpose

v1.8.2 converts repeated controlled FRED model-comparison experiments into reproducible empirical model profiles.

The version begins from the experiment infrastructure built in v1.8.0 and v1.8.1:

- isolated model-comparison artifacts
- repeated cross-model runs
- controlled prompt variants
- controlled temperature settings
- incremental manifests
- normalized comparison rows
- cumulative compatible-batch aggregation
- experiment-dashboard inspection

v1.8.2 changes the emphasis from running the laboratory to using the laboratory.

Its core question is:

> Given a frozen repeated-run population, what stable and reproducible behavioral differences can be measured across models, prompt specificity, and temperature?

The resulting profiles are deliberately empirical.

They report measured behavior, failure mechanisms, controlled sensitivity, and repeatability without assigning model personalities, roles, or generalized behavioral labels.

---

## Frozen Experimental Population

The initial v1.8.2 population is intentionally frozen.

Comparison family:

`fred_90_run_pilot_20260808`

Reference comparison:

`fred_90_run_pilot_20260808__batch_033`

Population:

- 33 completed compatible batches
- 90 attempted runs per batch
- 2,970 attempted runs total
- 3 models
- 3 prompt variants
- 2 temperatures
- 5 repetitions per experimental cell per batch
- 18 experimental cells
- 165 observations per model × prompt × temperature cell

Models:

- `llama3`
- `llama3:70b`
- `mistral`

Prompt variants:

- `weak`
- `intermediate`
- `hardened`

Temperatures:

- `0.0`
- `0.7`

The design is exactly balanced:

```text
33 batches
× 3 models
× 3 prompt variants
× 2 temperatures
× 5 repetitions
= 2,970 runs
```

Each experimental cell therefore contains:

```text
33 batches × 5 repetitions = 165 observations
```

---

## Population Compatibility Rules

v1.8.2 reuses the experiment-identity semantics originally developed for the v1.8.1 dashboard.

Batches can be pooled only when they share:

- inferred `comparison_family_id`
- `context_sha256`
- `comparison_window`
- exact experimental design signature

The experimental design signature includes:

- run label
- mode
- model
- prompt variant
- temperature
- repetitions

Legacy batch support is preserved through inference helpers for family ID and batch number.

The shared non-UI helpers live in:

`benchmarks/fred_model_comparison_utils.py`

Unlike the live dashboard population, the model-profile population is intentionally stricter:

- completed batches only
- normalized `comparison_rows.jsonl` only
- no fallback to incremental manifest rows
- normalized row count must equal expected configured runs

This distinction is deliberate.

The dashboard may answer:

> What is happening in the experiment population right now?

The empirical profile artifact answers:

> What did the completed controlled experiment population show?

---

## Canonical Input

The canonical analytical input is:

`summary/comparison_rows.jsonl`

for each compatible completed comparison batch.

The JSONL representation is preferred over the CSV export because it preserves the full normalized row schema, including fields not present in the CSV such as provenance and run metadata.

The profile builder does not use the dashboard as an analytical dependency.

Streamlit remains a presentation layer.

---

## Population Validation

The completed-compatible-population collector was validated against the frozen population.

Expected result:

```text
family: fred_90_run_pilot_20260808
included batches: 33
excluded compatible batches: 0
population rows: 2970
```

Condition balance:

```text
model:
    llama3        990
    llama3:70b    990
    mistral       990

prompt:
    weak          990
    intermediate  990
    hardened      990

temperature:
    0.0          1485
    0.7          1485
```

All 18 model × prompt × temperature cells contain exactly 165 observations.

---

## Outcome Semantics

v1.8.2 preserves the workflow distinction established in v1.7.

### Generation/process failure

```text
process_ok = False
run_completed = False
audit_pass = None
repair_needed = None
accepted_output = False
```

The run never reached audit.

### Audit failure

```text
process_ok = True
run_completed = True
audit_pass = False
repair_needed = True
accepted_output = False
```

The workflow completed, but the generated output failed deterministic audit.

### Accepted output

```text
process_ok = True
run_completed = True
audit_pass = True
repair_needed = False
accepted_output = True
```

This means:

> not audited is not equivalent to audited and failed.

Nullable audit and repair fields remain nullable throughout the profile pipeline.

---

## Frozen Population Outcome Accounting

Across all 2,970 runs:

```text
accepted                         1,643
audit_failure                      288
generation_contract_failure      1,039
other_process_failure                0
completed_unaccepted_other           0
                               -------
                                 2,970
```

The state accounting closes exactly.

Global measurements:

```text
Process completion:
1,931 / 2,970 = 65.0%

Audit pass:
1,643 / 1,931 evaluated = 85.1%

Acceptance:
1,643 / 2,970 = 55.3%

Repair needed:
288 / 1,931 evaluated = 14.9%
```

No run falls into either anomaly/catch-all outcome state.

---

## Empirical Profile Builder

Primary builder:

`benchmarks/build_fred_model_profiles.py`

Canonical output:

`benchmarks/results/model_profiles/fred_model_profiles.json`

Schema:

`fred_model_profiles_v0_1`

Generated results remain reproducible artifacts under `benchmarks/results` and are not committed wholesale.

The profile builder reports several layers of measurement.

---

## Direct Measurements

For each model and each model × prompt × temperature cell:

- `n_attempted`
- `n_process_ok`
- `n_audit_evaluated`
- `n_audit_pass`
- `n_repair_evaluated`
- `n_repair_needed`
- `n_accepted`

Rates use explicit denominators.

### Process completion rate

```text
process_ok = True
-----------------
all attempted runs
```

### Audit pass rate

```text
audit_pass = True
-----------------
audit_pass is not null
```

### Acceptance rate

```text
accepted_output = True
----------------------
all attempted runs
```

### Repair rate

```text
repair_needed = True
--------------------
repair_needed is not null
```

The builder also records mean and median elapsed seconds.

---

## Outcome-Stage Characterization

Each run is classified into one mutually exclusive observed outcome stage:

- `accepted`
- `generation_contract_failure`
- `other_process_failure`
- `audit_failure`
- `completed_unaccepted_other`

The last two non-primary catch-all states remain explicit so unexpected orchestration behavior cannot be silently absorbed into another category.

Across the frozen population both catch-all states are zero.

---

## Audit-Error Incidence

Audit failures remain multi-label.

Across the 288 audit-failed runs:

```text
claim_content_mismatches
231 / 288 = 80.2%

bullets_missing_claim_citations
224 / 288 = 77.8%

citations_found_but_no_bullets_extracted
42 / 288 = 14.6%
```

Percentages may sum above 100% because multiple errors may occur in the same failed output.

This is intentional.

---

# Experimental Findings

## Weak Prompt

The weak prompt produces a universal generation-contract failure regime in this experiment.

Across all models and both temperatures:

```text
990 / 990 weak-prompt runs
ended in generation-contract failure
```

No weak-prompt run reached deterministic audit.

Therefore:

- process completion = 0%
- acceptance = 0%
- audit pass = undefined
- repair rate = undefined

The weak prompt is useful as an intentionally underspecified control.

Human-readable output may still appear reasonable, but it does not satisfy the private machine-readable citation/output contract.

---

## Hardened Prompt

The hardened prompt produces a universal accepted regime in the frozen experiment population.

Across all models and both temperatures:

```text
990 / 990 hardened-prompt runs accepted
```

Therefore:

- process completion = 100%
- audit pass = 100%
- acceptance = 100%
- repair rate = 0%

The hardened regime is therefore saturated and provides relatively little model differentiation.

---

## Intermediate Prompt

The intermediate prompt is the primary discriminating regime.

It is specific enough to allow successful outputs but not so defensive that all three models saturate.

### llama3

At temperature 0.0:

```text
process completion: 100%
audit pass:         100%
acceptance:         100%
repair:               0%
```

At temperature 0.7:

```text
process completion: 100%
audit pass:          82.4%
acceptance:          82.4%
repair:              17.6%
```

Counts at t=0.7:

```text
165 attempted
165 process-complete
136 accepted
29 audit failures
0 generation-contract failures
```

Higher temperature therefore changes downstream audit behavior without changing generation/process completion.

---

### llama3:70b

At both temperatures:

```text
process completion: 100%
audit pass:         100%
acceptance:         100%
repair:               0%
```

All 330 intermediate-prompt runs were accepted.

No measurable intermediate temperature effect appears in this experiment.

---

### mistral

At temperature 0.0:

```text
process completion: 100%
audit pass:           0%
acceptance:           0%
repair:             100%
```

Counts:

```text
165 attempted
165 process-complete
165 audit failures
0 accepted
```

Every audit failure contained both:

- `bullets_missing_claim_citations`
- `claim_content_mismatches`

At temperature 0.7:

```text
process completion: 70.3%
audit pass:         19.0% of evaluated outputs
acceptance:         13.3% of attempted runs
repair:             81.0% of evaluated outputs
```

Counts:

```text
165 attempted
116 process-complete
49 generation-contract failures
94 audit failures
22 accepted
```

Higher temperature therefore simultaneously:

- reduces process completion
- creates accepted outputs absent at t=0.0
- increases conditional audit success among outputs reaching audit

This is why v1.8.2 does not collapse temperature sensitivity into one scalar score.

---

# Controlled Sensitivity

## Temperature Effects

Temperature effects are measured as:

```text
t=0.7 minus t=0.0
```

within a fixed model and prompt regime.

Metrics:

- process completion delta
- audit pass delta
- acceptance delta
- repair delta

### llama3 / intermediate

```text
process:       +0.0 pp
audit:        -17.6 pp
acceptance:   -17.6 pp
repair:       +17.6 pp
```

### llama3:70b / intermediate

```text
process:       +0.0 pp
audit:         +0.0 pp
acceptance:    +0.0 pp
repair:        +0.0 pp
```

### mistral / intermediate

```text
process:      -29.7 pp
audit:        +19.0 pp
acceptance:   +13.3 pp
repair:       -19.0 pp
```

Undefined audit or repair comparisons remain null rather than being converted into zero.

---

## Prompt-Transition Effects

Prompt sensitivity is measured through transparent adjacent transitions rather than an arbitrary sensitivity score.

Transitions:

```text
weak -> intermediate
intermediate -> hardened
```

Each transition is measured at fixed temperature.

---

### llama3

At t=0.0:

```text
weak -> intermediate
process:       +100.0 pp
acceptance:    +100.0 pp

intermediate -> hardened
process:         +0.0 pp
audit:           +0.0 pp
acceptance:      +0.0 pp
repair:          +0.0 pp
```

At t=0.7:

```text
weak -> intermediate
process:       +100.0 pp
acceptance:     +82.4 pp

intermediate -> hardened
process:         +0.0 pp
audit:          +17.6 pp
acceptance:     +17.6 pp
repair:         -17.6 pp
```

---

### llama3:70b

At both temperatures:

```text
weak -> intermediate
process:       +100.0 pp
acceptance:    +100.0 pp

intermediate -> hardened
no measured change
```

The intermediate prompt already reaches the observed ceiling.

---

### mistral

At t=0.0:

```text
weak -> intermediate
process:       +100.0 pp
acceptance:      +0.0 pp

intermediate -> hardened
process:         +0.0 pp
audit:         +100.0 pp
acceptance:    +100.0 pp
repair:        -100.0 pp
```

This separates two distinct effects.

The weak-to-intermediate transition completely solves generation/process completion while solving none of the final acceptance problem.

The intermediate-to-hardened transition then resolves the downstream audit failure.

At t=0.7:

```text
weak -> intermediate
process:        +70.3 pp
acceptance:     +13.3 pp

intermediate -> hardened
process:        +29.7 pp
audit:          +81.0 pp
acceptance:     +86.7 pp
repair:         -81.0 pp
```

These measurements provide an empirical foundation for later work on instruction-specificity thresholds without defining an arbitrary threshold in v1.8.2.

---

# Batch Repeatability

Aggregate rates alone do not show whether failures are broadly recurring or concentrated in a few unusual batches.

v1.8.2 therefore measures batch-level behavior across all 33 repeated batches.

The experimental unit was validated:

```text
33 batches × 18 cells = 594 batch-cells

all 594 batch-cells contain exactly 5 runs
```

The profile artifact records:

- number of batches
- repetitions per batch
- process-rate distribution
- process success-count distribution
- acceptance-rate distribution
- acceptance success-count distribution
- audit evaluated/pass distribution

For fixed-denominator rates it records:

- mean
- median
- population standard deviation
- minimum
- maximum

Audit retains evaluated/pass pairs rather than naively averaging rates with different evaluated denominators.

---

## llama3 Intermediate t=0.7 Repeatability

Accepted runs per five-run batch:

```text
2/5:  1 batch
3/5:  6 batches
4/5: 14 batches
5/5: 12 batches
```

Batch acceptance:

```text
mean:   82.4%
median: 80.0%
SD:     16.1 percentage points
range:  40.0% to 100.0%
```

Process completion:

```text
5/5 in all 33 batches
```

Twenty-one of 33 batches contain at least one audit failure.

The aggregate temperature sensitivity is therefore broadly recurrent rather than driven by only a few pathological batches.

---

## llama3:70b Intermediate Repeatability

At both temperatures:

```text
5/5 accepted in all 33 batches
```

Batch acceptance SD:

```text
0.0 percentage points
```

The observed intermediate behavior is fully stable across the repeated batches.

---

## mistral Intermediate t=0.0 Repeatability

Across all 33 batches:

```text
process:    5/5
accepted:   0/5
audit:      5 evaluated / 0 passed
```

Batch acceptance SD:

```text
0.0 percentage points
```

The audit-failure regime is perfectly reproducible across the frozen experiment population.

---

## mistral Intermediate t=0.7 Repeatability

Accepted runs per batch:

```text
0/5: 15 batches
1/5: 14 batches
2/5:  4 batches
```

Process-complete runs per batch:

```text
1/5:  1 batch
2/5:  4 batches
3/5: 11 batches
4/5: 11 batches
5/5:  6 batches
```

Batch acceptance:

```text
mean:   13.3%
median: 20.0%
SD:     13.6 percentage points
range:   0.0% to 40.0%
```

Eighteen of 33 batches produce at least one accepted output.

No batch produces more than two accepted runs.

The small successful-output region introduced at higher temperature is therefore distributed across many batches rather than concentrated in a handful of anomalous experiments.

---

# Batch-Stability Validation

The persisted batch-stability layer was reconciled against the original condition-level profile metrics across all 18 experimental cells.

Validation included:

- 33 batches per condition
- 5 repetitions per batch
- complete process count distribution
- complete acceptance count distribution
- reconstructed process totals
- reconstructed accepted totals
- reconstructed audit-evaluated totals
- reconstructed audit-pass totals
- mean process rate equals aggregate process rate
- mean acceptance rate equals aggregate acceptance rate

Result:

```text
conditions checked: 18
errors: 0

ALL BATCH-STABILITY CHECKS PASSED
```

---

# Human-Readable Profile Renderer

Presentation is intentionally separated from measurement.

Renderer:

`benchmarks/render_fred_model_profiles.py`

Input:

`fred_model_profiles.json`

Outputs:

```text
fred_model_profiles.md
fred_model_profile_llama3.md
fred_model_profile_llama3_70b.md
fred_model_profile_mistral.md
```

The renderer does not recompute experiment metrics.

It translates the canonical profile artifact into Markdown containing:

- profile snapshot
- concise observed-behavior summary
- condition matrix
- outcome-stage counts
- temperature effects
- prompt-transition effects
- batch repeatability
- audit failure details

Generated Markdown remains under `benchmarks/results/model_profiles`.

The renderer explicitly avoids behavioral personality or role labels.

---

# Observed Behavior Summaries

The final rendered summaries provide concise evidence-backed descriptions.

## llama3

- 330/330 weak-prompt runs end in generation-contract failure.
- Intermediate t=0.0 accepts 165/165.
- Intermediate t=0.7 accepts 136/165.
- Intermediate t=0.7 has 29 audit failures and no generation failures.
- All 33 intermediate t=0.7 batches produce at least one accepted run.
- 12/33 intermediate t=0.7 batches are fully accepted.
- 330/330 hardened-prompt runs are accepted.

## llama3:70b

- 330/330 weak-prompt runs end in generation-contract failure.
- 330/330 intermediate-prompt runs are accepted.
- Intermediate behavior is unchanged between t=0.0 and t=0.7.
- All 33 intermediate t=0.7 batches are fully accepted.
- 330/330 hardened-prompt runs are accepted.

## mistral

- 330/330 weak-prompt runs end in generation-contract failure.
- Intermediate t=0.0 accepts 0/165 despite 165/165 process completion.
- Intermediate t=0.7 accepts 22/165.
- Intermediate t=0.7 contains 49 generation-contract failures and 94 audit failures.
- 18/33 intermediate t=0.7 batches produce at least one accepted run.
- 0/33 intermediate t=0.7 batches are fully accepted.
- 330/330 hardened-prompt runs are accepted.

These statements are bounded descriptions of this frozen controlled task.

They are not intended as universal claims about the models.

---

# Architecture at v1.8.2 Closeout

```text
Repeated controlled experiment runs
            ↓
normalized comparison_rows.jsonl
            ↓
completed-compatible population collector
            ↓
fred_model_profiles.json
canonical empirical measurement layer
            ↓
render_fred_model_profiles.py
presentation-only renderer
            ↓
human-readable empirical model profiles
```

The architecture deliberately separates:

```text
experiment execution
measurement
interpretation/presentation
```

This separation allows later versions to add repair outcomes, semantic measurements, or token-level telemetry without collapsing those concerns into the experimental runner.

---

# Deliberate Non-Goals

v1.8.2 does not attempt to:

- assign model personalities
- classify models as anchor/explorer/etc.
- define an arbitrary scalar temperature-sensitivity score
- define an arbitrary scalar prompt-sensitivity score
- define a formal instruction threshold
- infer causal explanations for model differences
- perform semantic interpretation of generated claims
- execute repair actions
- measure repair effectiveness
- perform token-level or internal-model interpretability

Those questions remain downstream work.

---

# Key Source Files

```text
benchmarks/fred_model_comparison_utils.py

benchmarks/build_fred_model_profiles.py

benchmarks/render_fred_model_profiles.py

dashboards/eval_dashboard.py
```

Generated profile artifacts:

```text
benchmarks/results/model_profiles/
```

---

# Representative v1.8.2 Commits

```text
e85566c  Extract FRED comparison identity helpers
287f5d1  Share FRED comparison directory discovery
362b128  Add completed FRED comparison population collector
f3049b5  Build empirical FRED model profiles
5bbe5f4  Add FRED profile outcome and audit failure metrics
e4fc6bc  Add FRED profile temperature effects
8610097  Add FRED profile prompt transition effects
ff86e43  Add FRED profile batch stability metrics
6e45b23  Render empirical FRED model profiles
d348443  Add observed behavior summaries to FRED profiles
c8c750a  Fix duplicate FRED profile sections
```

---

# v1.8.2 Closeout Statement

> v1.8.2 converts repeated controlled experiments into empirical model behavioral profiles.

The version establishes a reproducible characterization layer with:

- frozen population provenance
- balanced experimental cells
- explicit denominator semantics
- outcome-stage taxonomy
- audit failure incidence
- temperature sensitivity
- prompt-specificity sensitivity
- repeated-batch stability
- deterministic human-readable profile rendering

The resulting profiles show not only whether each model succeeds, but where it fails, how controlled interventions change behavior, and whether those signatures recur across repeated experiments.

---

# Next Version: v1.8.3 — ACT

v1.8.3 should move from observing failures to executing interventions.

Core loop:

```text
original output
      ↓
deterministic audit
      ↓
observed failure type
      ↓
selected repair strategy
      ↓
repaired output
      ↓
re-audit
      ↓
measured repair outcome
```

The first milestone should remain deliberately narrow:

> known fault → known repair → deterministic re-audit → measurable improvement

Initial repair targets should come from failures already observed in v1.8.2, including:

- `bullets_missing_claim_citations`
- `citations_found_but_no_bullets_extracted`
- `claim_content_mismatches`

Potential repair strategies may include:

- restore exact citation contract
- restore bullet structure
- rebind bullet content to cited claim evidence

The repair layer should persist:

- original audit errors
- selected repair strategy
- repair actions
- repaired output
- repaired audit errors
- acceptance before repair
- acceptance after repair
- repair success
- residual failures
- newly introduced failures

Only after generic and failure-specific repair are measurable should the empirical model profiles influence repair selection.

---

# Longer-Term Path

```text
v1.8.2  CHARACTERIZE
Repeated controlled behavior
→ empirical profiles

v1.8.3  ACT
Observed failure
→ repair intervention
→ re-audit
→ measured repair outcome

v1.9    INTERPRET
Evidence
→ model statement
→ added meaning
→ semantic support/failure

v1.9.x  SEMANTIC REPAIR
Unsupported interpretation
→ targeted correction
→ semantic re-audit

v2.0    LOOK INSIDE
Behavioral contrasts
→ token probabilities / logprobs
→ uncertainty / divergence / decision points

later
internal representations
→ sparse-autoencoder or related mechanistic analysis
```

The guiding principle remains the same:

> Measure behavior first, intervene second, interpret meaning third, and only then look inside the model for mechanisms.
