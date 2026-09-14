# Roadmap

The project develops a local evaluation and interpretability harness for model-generated macro/rates analysis.

The long-term arc is:

```text
generate
→ audit
→ characterize
→ act
→ interpret
→ repair semantics
→ inspect token-level behavior
→ investigate internal representations
```

---

## Completed Foundations

### v1.6 — STRUCTURE

Structured FRED claims and deterministic evidence fields.

Core ideas:

- explicit claim IDs
- prior/current values
- comparison windows
- deltas
- claim strength
- narrative eligibility

---

### v1.7 — EVIDENCE LOOP

End-to-end deterministic evidence workflow:

```text
structured claims
→ claim selection
→ LLM narrative
→ deterministic audit
→ repair plan
→ traceability
→ demo artifact
```

Key semantic distinction:

```text
run_completed
audit_pass
accepted_output
repair_needed
```

Generation failure is not equivalent to audit failure.

---

### v1.7.2 — PROMPT HARDENING

Hardened narrative contract plus a real citation-contract failure fixture.

This established a controlled behavioral contrast between:

- weak prompts
- intermediate prompts
- hardened prompts

---

### v1.8.0 — CROSS-MODEL INFRASTRUCTURE

Controlled model-comparison execution.

Added:

- multiple models
- prompt variants
- temperature controls
- isolated artifacts
- reproducible comparison identity

---

### v1.8.1 — SCALE + OBSERVE

Repeated experiment execution and population inspection.

Added:

- repeated batches
- incremental manifests
- normalized comparison rows
- cumulative compatible-batch aggregation
- dashboard inspection

---

### v1.8.2 — CHARACTERIZE

Status:

`COMPLETE`

Frozen population:

```text
33 batches
90 runs per batch
2,970 attempted runs
18 balanced experiment cells
165 observations per cell
```

Purpose:

> Convert repeated controlled experiments into reproducible empirical model profiles.

Added:

- completed-compatible population collection
- empirical model profiles
- process/acceptance outcome measurement
- audit-failure incidence
- prompt sensitivity
- temperature sensitivity
- repeated-batch stability
- deterministic human-readable profile rendering

Closeout document:

`docs/v1_8_2_model_profiles_walkthrough.md`

---

### v1.8.3 — ACT

Status:

`COMPLETE`

Purpose:

> Turn repair from a recommendation into a deterministic, auditable action layer.

Core architecture:

```text
audit
→ repair plan
→ deterministic selector
→ capability registry
→ executor
→ independent re-audit
→ repair evaluator
→ single-run workflow
→ population runner
→ aggregate
→ generic validator
```

Deterministic repair capabilities:

- structural normalization
- citation relocation
- claim-representation consolidation

Final guarded frozen-population result:

```text
2,970 attempted runs

1,039 upstream validation failures

1,931 workflow-ready
├── 1,643 initially accepted
└──   288 audit failures
      ├── 190 uniquely/safely repairable
      └──  98 deterministic abstentions

190 repairs executed
190 targeted repair successes
0 targeted repair failures
0 introduced-error runs
4 full audit successes
```

Key finding:

> Repair eligibility must represent a capability contract, not merely action availability.

Population evidence showed that an overly permissive structural repair could partially resolve a target or destroy the global audited representation.

Tightening Strategy #1 eligibility:

- removed 26 interventions
- preserved all 190 genuine successes
- eliminated all observed targeted failures
- eliminated all introduced-error runs

Closeout document:

`docs/v1_8_3_repair_execution_walkthrough.md`

---

## Acceptance Bridge — Fresh FRED Data

After tagging v1.8.3 and before beginning v1.9 development, run a small out-of-sample acceptance experiment using fresh FRED evidence.

Suggested initial cells:

```text
llama3 intermediate t0
llama3 intermediate t07
mistral intermediate t0
mistral intermediate t07
```

Suggested initial repetitions:

```text
5 per cell
20 total runs
```

Preserve:

```text
original generation
initial audit
repair decision
repair execution / abstention
post-repair audit
final acceptance state
```

Measure separately:

```text
raw model performance

versus

full system performance
```

The tagged v1.8.3 code should remain frozen during this acceptance experiment.

Unexpected failure grammars should produce abstention rather than ad hoc repair changes.

---

## v1.9 — INTERPRET

Status:

`NEXT`

Purpose:

> Measure the meaning a model adds beyond its evidence.

The next analytical unit is no longer merely the citation or numeric claim.

It is the relationship:

```text
evidence
→ model statement
→ added meaning
```

Initial semantic taxonomy candidates:

- `empirical_restatement`
- `magnitude_judgment`
- `trend_interpretation`
- `scope_generalization`
- `causal_interpretation`
- `policy_interpretation`
- `market_interpretation`
- `unsupported_generalization`
- `contradiction`

Initial goals:

- define semantic units
- classify narrative spans
- distinguish evidence restatement from interpretation
- measure unsupported added meaning
- preserve source provenance
- build deterministic or constrained semantic validators where possible
- characterize semantic failures before attempting semantic repair

The v1.8.3 control plane should be reused:

```text
observe semantic failure
→ identify eligible capability
→ act or abstain
→ semantic re-audit
→ measure outcome
```

---

## v1.9.x — SEMANTIC REPAIR

After semantic failure classes are empirically characterized:

```text
unsupported interpretation
→ constrained correction
→ semantic re-audit
→ measured outcome
```

Potential repair primitives may include:

- semantic narrowing
- unsupported causal-claim removal
- evidence rebinding
- scope correction
- contradiction correction
- explicit abstention

LLM-proposed repairs may become appropriate here, but only behind constrained evidence and validation contracts.

---

## v2.0 — LOOK INSIDE

Add token-level telemetry where technically available.

Potential measurements:

- token log probabilities
- alternative-token probabilities
- entropy
- uncertainty
- divergence across prompt conditions
- decision-point localization
- repair-related token shifts

Goal:

> Connect observable behavioral differences to local token-level decision behavior.

---

## Later — INTERNAL REPRESENTATIONS

Longer-term mechanistic work may include:

- hidden-state analysis
- representation comparison
- sparse autoencoders
- feature discovery
- causal interventions on internal activations

This work should be motivated by stable behavioral contrasts discovered in earlier versions rather than performed without a concrete empirical target.

---

# Current Sequence

```text
v1.8.2  CHARACTERIZE        COMPLETE

v1.8.3  ACT                 COMPLETE
          ↓
fresh-data acceptance test
          ↓

v1.9    INTERPRET           NEXT
          ↓

v1.9.x  SEMANTIC REPAIR
          ↓

v2.0    LOOK INSIDE
          ↓

later   INTERNAL REPRESENTATIONS
```
