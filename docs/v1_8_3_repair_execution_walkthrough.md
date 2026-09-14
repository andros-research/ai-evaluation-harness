# v1.8.3 Repair Execution Walkthrough

## Purpose

v1.8.3 turns repair from a recommendation into a deterministic, auditable action layer.

The version begins from the frozen empirical population characterized in v1.8.2:

- repeated controlled FRED experiments
- explicit process-success and generation-failure semantics
- deterministic narrative audits
- durable repair plans
- reproducible model/prompt/temperature cells
- a frozen 2,970-run experimental population

v1.8.2 asked:

> Given a frozen repeated-run population, what stable and reproducible behavioral differences can be measured?

v1.8.3 asks:

> Given a known audited failure, can the system select a defensible repair, execute it, independently re-audit the result, and measure whether the intervention actually improved the output?

The version is deliberately conservative.

A repair is not considered successful merely because:

- an executor can modify the narrative
- one local audit symptom disappears
- the repaired output looks subjectively better

Instead, v1.8.3 separates:

- repair applicability
- repair execution
- target-error resolution
- full audit acceptance
- residual failures
- newly observed failures
- unmasked failures
- genuinely introduced failures
- explicit abstention

The central milestone is:

> known fault → known repair → deterministic re-audit → measurable outcome

---

## Frozen Experimental Population

v1.8.3 reuses the frozen v1.8.2 FRED population.

Comparison family:

`fred_90_run_pilot_20260808`

Population:

- 33 compatible batches
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

The frozen population contains:

```text
2,970 attempted runs
├── 1,039 upstream narrative-validation failures
└── 1,931 workflow-ready runs
    ├── 1,643 initial audit passes
    └──   288 initial audit failures
```

The first physical batch predates the later batch naming convention.

It is stored as:

`fred_90_run_pilot_20260808`

Later batches are stored as:

`fred_90_run_pilot_20260808__batch_002`

through:

`fred_90_run_pilot_20260808__batch_033`

Population discovery therefore explicitly supports both the original unsuffixed family root and later numbered batch roots.

---

## ACT Architecture

At v1.8.3 closeout, the repair system is:

```text
original narrative
      ↓
deterministic audit
      ↓
repair plan
      ↓
deterministic strategy selector
      ↓
capability registry
      ↓
repair executor
      ↓
repaired narrative
      ↓
independent re-audit
      ↓
repair outcome evaluator
      ↓
single-run ACT workflow
      ↓
population runner
      ↓
population aggregate
      ↓
generic invariant validator
```

The architecture separates:

```text
failure observation

repair eligibility

repair execution

repair evaluation

population measurement

validation
```

No layer is allowed to silently redefine the semantics of another.

---

# Repair Strategies

## Strategy #1 — Structural Normalization

Strategy:

`normalize_uncited_detail_bullets`

Source:

`benchmarks/apply_fred_narrative_repair.py`

The original failure grammar consisted of uncited Markdown detail bullets such as:

```text
- Direction: up
- Prior value: 2.68%
- Current value: 4.167%
- Delta magnitude: 1.487 percentage points
```

These detail bullets were often subordinate to an already-cited empirical summary.

The repair removes only the Markdown bullet marker:

```text
- Prior value: 2.68%
```

becomes:

```text
Prior value: 2.68%
```

The substantive text is preserved.

The initial executor demonstrated that structural normalization could resolve:

`bullets_missing_claim_citations`

while leaving independent content failures visible.

### Population-Derived Eligibility Guard

The first full-population ACT run revealed that the original Strategy #1 eligibility contract was too permissive.

It asked, effectively:

> Can at least one supported detail bullet be normalized?

This permitted two undesirable cases:

1. partial normalization, where only some target-error bullets were supported
2. over-normalization, where every audited bullet was demoted out of audit scope

The first authoritative population run produced:

```text
211 normalization selections
185 targeted successes
26 targeted failures
3 introduced-error runs
```

The 26 failures separated cleanly into:

- incomplete target coverage
- failure to preserve an auditable cited representation

The final Strategy #1 capability contract therefore requires:

```text
1. Every audited missing_claim_citation bullet must be covered
   by a supported normalization action.

AND

2. At least one citation-bearing audited bullet representation
   must remain after normalization.
```

A retrospective population diagnostic produced a perfect split:

```text
historical Strategy #1 selections: 211

actual targeted successes: 185
actual targeted failures:   26

new eligibility = true:     185
new eligibility = false:     26

false positives: 0
false negatives: 0
```

The guarded executor therefore abstains on those 26 cases.

This changes the meaning of repair eligibility from:

> I can perform an action here.

to:

> I expect this repair to resolve its declared target while preserving a usable audited representation.

---

## Strategy #2 — Evidence Relocation

Strategy:

`relocate_existing_claim_citations`

Source:

`benchmarks/relocate_fred_claim_citations.py`

This strategy applies when:

- an uncited audited bullet contains a complete empirical fingerprint
- exactly one selected claim matches that fingerprint
- the required citation already exists elsewhere
- the citation source is outside audited bullet scope
- evidence can be relocated without inventing or duplicating provenance

Claim identity is established conservatively from:

- prior value
- current value
- delta magnitude

using strict numeric-token matching.

Direction is deliberately excluded from identity matching.

The strategy preserves the citation multiset exactly.

Conceptually:

```text
valid evidence
currently attached to the wrong narrative span
        ↓
move existing evidence
        ↓
bind evidence to the auditable empirical span
```

The executor does not:

- invent citations
- duplicate evidence
- rewrite substantive prose
- repair semantic direction vocabulary
- guess when identity is ambiguous

---

## Strategy #3 — Claim Representation Consolidation

Strategy:

`consolidate_duplicate_claim_representations`

Source:

`benchmarks/consolidate_fred_claim_representations.py`

Population analysis exposed narratives containing multiple incomplete representations of the same underlying claim.

Two supported subtypes emerged.

### Subtype A

A cited audited source is deficient, while an uncited destination contains the complete empirical representation.

```text
deficient cited representation
+
complete uncited representation
        ↓
retain complete representation
transfer existing citation
remove deficient duplicate
```

### Subtype B

A valid cited representation already passes, while another uncited detail representation is redundant.

```text
valid cited representation
+
redundant uncited representation
        ↓
retain valid cited representation
remove redundant duplicate
```

The governing principle is:

> When multiple spans represent the same claim, consolidate them into one fully auditable representation rather than repairing each representation independently.

Citation conservation remains mandatory.

---

# Repair Outcome Evaluation

Source:

`benchmarks/evaluate_fred_narrative_repair.py`

The repair evaluator compares:

```text
before audit
repair execution
after audit
```

and distinguishes:

- target error present before
- target error resolved
- targeted repair success
- full audit success
- resolved errors
- residual errors
- newly observed errors
- unmasked errors
- introduced errors

This distinction became important during citation relocation.

A repair may remove an upstream blocker and expose a content failure that was already latent in unchanged substantive text.

Such an error is:

`unmasked`

rather than:

`introduced`

The evaluator therefore avoids treating every after-only error as repair-induced damage.

At the same time, genuinely repair-induced regressions remain classified as introduced errors.

---

# Deterministic Strategy Selection

Source:

`benchmarks/select_fred_repair_strategy.py`

The selector probes existing repair executors in memory.

Selection policy:

```text
exactly one eligible strategy
    → select it

zero eligible strategies
    → abstain

multiple eligible strategies
    → abstain as ambiguous
```

No strategy precedence is encoded.

The selector does not choose a repair because one strategy is considered globally better than another.

It selects only when the capability contracts themselves yield one uniquely defensible action.

This makes abstention a first-class successful outcome.

---

# Capability Registry

The single-run workflow owns a small repair capability registry.

Each capability records:

- executor module
- whether selected claims are required
- declared target error

Conceptually:

```text
repair strategy
    ↓
capability metadata
    ├── executor
    ├── input requirements
    └── target error
```

The workflow therefore does not contain scattered logic such as:

```text
if strategy == X:
    target_error = Y
```

It asks the registry what contract the selected capability claims to address.

This design allows later semantic repair capabilities to register new target errors without redesigning the orchestration layer.

---

# Single-Run ACT Workflow

Source:

`benchmarks/run_fred_repair_workflow.py`

The workflow performs:

```text
load inputs
    ↓
select strategy
    ↓
execute selected capability
    ↓
independent re-audit
    ↓
evaluate before/after outcome
    ↓
persist workflow result
```

Non-execution outcomes are also durable.

Supported workflow outcomes include:

- `repair_executed`
- `no_repair_needed`
- `no_supported_deterministic_repair`
- `ambiguous_repair_selection`

A successful workflow execution does not imply:

- a repair occurred
- the target was resolved
- the final narrative passed audit

`workflow_status`

and:

`workflow_outcome`

remain distinct.

---

# Preserving Audit Semantics

The post-repair audit reuses:

`strict_selected_claim_coverage`

from the original before-audit artifact.

The workflow does not silently recreate audit configuration from a default.

This preserves evaluation semantics across before/after comparison.

---

# Population Runner

Source:

`benchmarks/run_fred_repair_population.py`

The population runner scales the single-run ACT workflow over the frozen experiment.

Its responsibilities are operational:

```text
discover runs
    ↓
classify readiness
    ↓
run ACT when workflow-ready
    ↓
record upstream failures separately
    ↓
persist one population record per original run
    ↓
continue after individual workflow failures
    ↓
write population manifest
```

Recognized readiness states include:

- `workflow_ready`
- `upstream_validation_failed`
- `population_input_error`

The 1,039 upstream validation failures are classified from durable failed-validation artifacts rather than inferred merely from missing downstream files.

The runner supports:

- full denominator preservation
- per-run provenance
- restartability through `--resume`
- bounded smoke testing through `--limit`
- per-run exception capture
- continuation after individual workflow failure

---

# First Full-Population ACT Run

The first full population run completed all 2,970 observations with:

```text
infrastructure/input failures: 0
```

Initial ACT results:

```text
2,970 total attempts

1,039 upstream validation failures

1,931 workflow-ready
├── 1,643 no repair needed
└──   288 audit failures
      ├── 216 repair executed
      └──  72 unsupported

216 repairs executed
190 targeted repair successes
 26 targeted repair failures
  4 full audit successes
  3 introduced-error runs
```

Strategy selections:

```text
211 normalize
  4 consolidate
  1 relocate
```

The population run therefore became a diagnostic instrument for the repair system itself.

It exposed the overly permissive Strategy #1 contract described above.

---

# Guarded Authoritative ACT Population

After tightening Strategy #1 eligibility, the complete population was rerun from committed source revision:

`05dde69`

Authoritative guarded population root:

`benchmarks/results/repair_population/fred_90_run_pilot_20260808__v1_8_3_act_guarded`

Final population:

```text
2,970 attempted runs

1,039 upstream validation failures

1,931 workflow-ready
├── 1,643 initially accepted
└──   288 initial audit failures
      ├── 190 uniquely and safely repairable
      └──  98 deterministic abstentions
```

Final strategy selections:

```text
185 normalize
  4 consolidate
  1 relocate
```

Repair outcomes:

```text
190 repairs executed
190 targeted repair successes
  0 targeted repair failures
  4 full audit successes
  0 introduced-error runs
```

The before/after population result is:

```text
INITIAL ACT CONTRACT

216 interventions
190 targeted successes
 26 targeted failures
  3 introduced-error runs
  4 full audit successes


GUARDED ACT CONTRACT

190 interventions
190 targeted successes
  0 targeted failures
  0 introduced-error runs
  4 full audit successes
```

The system removed 26 interventions while losing zero genuine repair successes.

This is the central v1.8.3 population result.

---

# Final Acceptance

Initial full audit acceptance:

```text
1,643 / 1,931 workflow-ready runs
```

ACT promoted four additional narratives to full audit acceptance:

```text
1,647 final accepted
```

The more important repair-layer result is:

```text
190 / 190 selected repairs resolved their declared target
```

Most repairs did not make the entire narrative acceptable because independent residual failures remained visible.

v1.8.3 therefore treats:

```text
targeted repair success
```

and:

```text
full narrative acceptance
```

as distinct measurements.

---

# Cell-Level ACT Behavior

The guarded population contains 18 balanced cells with 165 observations each.

Most ACT activity is concentrated in intermediate-prompt cells.

## Hardened prompts

All hardened cells are fully accepted before ACT.

ACT has nothing to do.

## Weak prompts

Weak-prompt runs fail upstream narrative validation rather than reaching the audit/repair layer.

## llama3:70b intermediate

Both temperatures are fully accepted before ACT.

## llama3 intermediate t=0.0

165/165 runs are accepted before ACT.

## llama3 intermediate t=0.7

```text
136 initial passes
29 audit failures

1 consolidation repair
28 unsupported

1 full audit success
```

## mistral intermediate t=0.0

```text
165 audit failures
165 structural normalizations
165 targeted repair successes
0 full audit successes
```

This cell cleanly demonstrates the distinction between:

- successful target repair
- full narrative acceptance

## mistral intermediate t=0.7

```text
49 upstream validation failures
22 initial passes
94 audit failures

24 repair executions
├── 20 normalize
├──  3 consolidate
└──  1 relocate

70 unsupported

24 targeted repair successes
3 full audit successes
0 introduced errors
```

This cell contains most of the heterogeneous failure grammar that motivated the richer repair capabilities and guard conditions.

---

# Population Aggregate

Source:

`benchmarks/aggregate_fred_repair_population.py`

The population aggregate reduces durable per-run artifacts into analysis-ready outputs.

Generated artifacts include:

```text
fred_repair_population_aggregate.json

fred_repair_population_rows.jsonl

fred_repair_population_rows.csv

fred_repair_population_cells.csv
```

The row layer preserves one record per original observation.

The aggregate includes:

- provenance
- model
- prompt variant
- temperature
- repetition
- readiness state
- original audit state
- repair-needed state
- selector outcome
- selected strategy
- repair execution
- target resolution
- post-repair audit state
- full repair success
- residual errors
- newly observed errors
- unmasked errors
- introduced errors

The 18-cell summary preserves the original balanced experimental design.

---

# Generic Population Validation

Source:

`benchmarks/validate_fred_repair_population.py`

The validator checks the durable aggregate package rather than rereading the original experiment tree.

It validates contracts and internal consistency rather than hard-coding the scientific outcome.

Examples include:

- aggregate row count matches source population
- source paths are unique
- logical run identities are unique
- analysis metadata is complete
- cell summaries reconcile exactly to rows
- upstream-failure states obey their contract
- workflow-ready rows contain valid before-audit state
- initial passes map to no-repair outcomes
- initial failures route to repair or abstention
- selected strategies exist in the live capability registry
- selected repairs resolve their declared target
- unsupported runs abstain cleanly
- ambiguous selections abstain cleanly
- repair safety holds
- full success implies post-repair audit pass
- aggregate summaries reconcile to rows
- strategy summaries reconcile to rows
- logical CSV record counts match JSONL population

The only externally supplied expectation is the source-code commit.

This is a provenance assertion, not an expected scientific result.

The authoritative v1.8.3 aggregate passed:

```text
20 / 20 generic validation checks
overall_pass = True
```

---

# Architecture at v1.8.3 Closeout

```text
controlled generation
      ↓
original narrative
      ↓
deterministic audit
      ↓
repair plan
      ↓
deterministic selector
      ↓
capability registry
      ↓
repair executor
      ↓
independent re-audit
      ↓
repair evaluator
      ↓
single-run ACT workflow
      ↓
population runner
      ↓
aggregate
      ↓
generic validator
```

The architecture now separates:

```text
generation

audit

failure classification

capability eligibility

action

evaluation

population measurement

validation
```

This separation is intended to remain useful when later versions introduce semantic repair capabilities.

---

# Deliberate Non-Goals

v1.8.3 does not attempt to:

- repair every audit failure
- maximize repair coverage
- rank strategies by arbitrary precedence
- use an LLM to improvise unrestricted repairs
- hide unsupported failures
- treat target resolution as equivalent to full acceptance
- weaken the auditor to make repaired outputs pass
- perform semantic claim interpretation
- determine whether causal or policy conclusions are justified
- perform token-level or internal-model interpretability

Unsupported failures remain visible.

Abstention is a valid system outcome.

---

# Key Source Files

```text
benchmarks/apply_fred_narrative_repair.py

benchmarks/evaluate_fred_narrative_repair.py

benchmarks/relocate_fred_claim_citations.py

benchmarks/consolidate_fred_claim_representations.py

benchmarks/select_fred_repair_strategy.py

benchmarks/run_fred_repair_workflow.py

benchmarks/run_fred_repair_population.py

benchmarks/aggregate_fred_repair_population.py

benchmarks/validate_fred_repair_population.py
```

Authoritative population artifacts:

```text
benchmarks/results/repair_population/
fred_90_run_pilot_20260808__v1_8_3_act_guarded/
```

---

# Representative v1.8.3 Commits

```text
790a4d2  Add deterministic FRED repair executor

60e0a2d  Add FRED repair outcome evaluator

5cf0123  Add deterministic FRED citation relocation repair

7479da2  Distinguish unmasked errors in FRED repair evaluation

cc2a469  Add deterministic FRED claim consolidation repair

1bb22ed  Add deterministic FRED repair strategy selector

b78058b  Add deterministic FRED repair workflow runner

7f2320c  Add deterministic FRED repair population runner

05dde69  Tighten FRED structural repair eligibility

d2f62c9  Add FRED repair population aggregate

7080819  Add generic FRED repair population validation
```

---

# v1.8.3 Closeout Statement

> v1.8.3 turns repair from a recommendation into a deterministic, auditable action layer.

The version establishes:

- executable repair primitives
- explicit repair capability contracts
- provenance-preserving evidence operations
- deterministic strategy selection
- conservative abstention
- independent post-repair audit
- targeted-success versus full-success semantics
- unmasked versus introduced error semantics
- single-run ACT orchestration
- population-scale execution
- restartable population processing
- analysis-ready aggregate artifacts
- generic population validation
- population-derived repair safety refinement

The final guarded population demonstrates:

```text
190 selected repairs

190 targeted repair successes

0 targeted repair failures

0 introduced-error runs

4 full audit successes

98 unsupported audit failures preserved as abstentions
```

The strongest v1.8.3 result is not that the system repairs more outputs.

It is that the system learned to intervene less often and with stronger guarantees.

---

# Fresh-Data Acceptance Experiment

Before beginning v1.9 development, the tagged v1.8.3 system should be tested on fresh FRED evidence.

The experiment should be treated as out-of-sample acceptance testing rather than additional v1.8.3 development.

A deliberately small design is sufficient.

Suggested initial cells:

```text
llama3 intermediate t0
llama3 intermediate t07

mistral intermediate t0
mistral intermediate t07
```

with approximately five repetitions per cell.

The lifecycle should preserve both raw model behavior and repaired system behavior:

```text
fresh evidence
      ↓
original narrative
      ↓
initial audit
      ↓
repair plan
      ↓
ACT
      ↓
post-repair audit
      ↓
final acceptance state
```

Original artifacts must remain intact.

This allows separate measurement of:

```text
RAW MODEL PERFORMANCE

and

FULL SYSTEM PERFORMANCE
```

Unexpected fresh-data failure grammars should result in deterministic abstention rather than improvised repair.

---

# Next Version: v1.9 — INTERPRET

v1.9 moves from structural/evidence correctness toward semantic interpretation.

The central question becomes:

> What meaning does the model add beyond the evidence, and is that added meaning justified?

Initial semantic categories may include:

- empirical restatement
- magnitude judgment
- trend interpretation
- scope generalization
- causal interpretation
- policy interpretation
- market interpretation
- unsupported generalization
- contradiction

Conceptually:

```text
evidence
    ↓
model statement
    ↓
added meaning
    ↓
semantic category
    ↓
support / failure assessment
```

The v1.8.3 orchestration architecture should remain reusable.

Future semantic capabilities can register:

- their executor
- required evidence inputs
- declared semantic target error

without redesigning the select → execute → re-audit → evaluate control plane.

---

# Longer-Term Path

```text
v1.8.2  CHARACTERIZE

Repeated controlled behavior
→ empirical model profiles


v1.8.3  ACT

Observed failure
→ deterministic capability selection
→ repair intervention
→ re-audit
→ measured repair outcome


fresh-data acceptance test

Tagged ACT system
→ unseen evidence
→ out-of-sample repair behavior


v1.9  INTERPRET

Evidence
→ model statement
→ added meaning
→ semantic support/failure


v1.9.x  SEMANTIC REPAIR

Unsupported interpretation
→ targeted semantic correction
→ semantic re-audit


v2.0  LOOK INSIDE

Behavioral contrasts
→ token probabilities / logprobs
→ uncertainty / divergence / decision points


later

internal representations
→ sparse-autoencoder or related mechanistic analysis
```
