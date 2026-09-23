# v1.9 Semantic Annotation Guide

## Purpose

v1.9 INTERPRET asks:

> Where does an observation become an argument?

The goal is to identify what a model statement asks the reader to accept beyond direct evidence, and to assess whether the supplied evidence supports that commitment.

This guide is intentionally narrow.

It is derived from the first reviewed semantic pilot:

`benchmarks/fixtures/fred_semantic_pilot_v0_1.jsonl`

The pilot contains eight human-reviewed examples spanning:

- observed model outputs
- deterministic transformations
- contradicted empirical statements
- temporal-pattern claims
- causal explanations
- asserted versus qualified claims

This guide does not attempt to define a complete semantic ontology.

---

## Core Principle

An interpretive move occurs when a statement asks the reader to accept something beyond:

1. a direct restatement of supplied evidence, or
2. an explicitly checkable transformation of supplied evidence.

The central object is therefore not a suspicious word or phrase.

It is the relationship:

```text
supplied evidence
      ↓
model statement
      ↓
additional commitment
```

A statement should be assessed relative to the evidence available to it.

---

# 1. Annotation Unit

The annotation unit is:

> The smallest meaningful claim whose evidential support can be assessed without destroying an important relationship expressed by the statement.

Examples of relationships that should remain intact include:

- comparison
- qualification
- causation
- temporal pattern
- magnitude judgment

The initial v0.1 pilot used one statement-sized annotation unit per example.

The current semantic record uses claim-bearing spans. Each span preserves exact source text and may receive one or more evidence-relative semantic annotations. Distinct semantic moves do not need to map one-to-one to separate text spans.

Example:

```text
"Unemployment fell from 4.3% to 4.1%, demonstrating a broad
and sustained improvement in labor-market conditions."
```

Potential future units include:

```text
1. unemployment fell from 4.3% to 4.1%
2. the improvement was broad
3. the improvement was sustained
```

The exact source text should always remain preserved.

---

# 2. Claim Kind

`claim_kind` describes what operation or type of claim the statement makes.

It does not determine whether the statement is supported.

Current reviewed claim kinds include:

## `empirical_comparison`

A direct comparison of supplied observations.

Example:

```text
Unemployment rate decreased by 0.2 percentage points.
```

This may be supported even if it does not reproduce every field required by an older structural audit.

Reference examples:

- `semantic_pilot_001`
- `semantic_pilot_002`
- `semantic_pilot_004`

---

## `unit_conversion`

A deterministic transformation of supplied values.

Example:

```text
A decline of 0.2 percentage points is a decline of 20 basis points.
```

The transformation must be explicitly checkable.

Reference example:

- `semantic_pilot_003`

---

## `temporal_pattern`

A claim about the path or sequence of observations across time.

Example:

```text
Each successive monthly unemployment-rate observation was lower.
```

Two endpoints establish a net change.

They do not establish the path between those endpoints.

The same temporal statement can therefore be unsupported under sparse evidence and supported under richer evidence.

Reference examples:

- `semantic_pilot_005`
- `semantic_pilot_006`

---

## `causal_explanation`

A statement connecting one observed change to another through a causal relationship.

Example:

```text
The decline in the federal funds rate caused the fall in unemployment.
```

Observing two changes over the same period does not by itself establish causality.

Reference examples:

- `semantic_pilot_007`
- `semantic_pilot_008`

---

## `scope_generalization`

A statement that moves from a specific supplied measure to a broader condition or category.

Example:

```text
Evidence:
Unemployment rate decreased from 4.3 to 4.1.

Statement:
The labor market strengthened.
```

A single unemployment-rate comparison may be relevant to the broader labor-market claim without being sufficient to establish it.

Reviewed examples include:

- `semantic_challenge_001`
- `semantic_challenge_003`
- `semantic_challenge_004`
- `semantic_harvest_001`
- `semantic_harvest_002`

---

## `magnitude_judgment`

A qualitative characterization of the size or importance of an observed change.

Example:

```text
The labor market softened only modestly.
```

Words such as `modestly`, `sharply`, `substantially`, or `materially` require an appropriate benchmark. A supplied numerical change does not automatically establish the qualitative magnitude judgment.

Reviewed example:

- `semantic_challenge_004`

---

## `market_interpretation`

A statement that interprets observed data or market prices as indicating market beliefs, expectations, sentiment, motives, or related latent market states.

Example:

```text
The rise in the 10-year yield suggests markets became more concerned about inflation.
```

A market-price move can be directly observed while the proposed belief or motive remains insufficiently established.

Reviewed examples include:

- `semantic_challenge_002`
- `semantic_harvest_003`
- `semantic_harvest_005`
- `semantic_harvest_006`

---

## `policy_interpretation`

A statement that interprets evidence in terms of policy stance, policy response, or possible policy behavior.

Example:

```text
Higher inflation may prompt the Federal Reserve to maintain a hawkish stance.
```

Evidence about inflation or interest rates does not by itself establish a policy reaction, policy intention, or broader characterization of policy stance.

Reviewed example:

- `semantic_harvest_004`

---

# 3. Support Status

`support_status` describes the relationship between a statement and the supplied evidence.

It is separate from:

- formatting compliance
- completeness
- usefulness
- assertion strength
- whether the statement might be true in the world

Current values are:

## `supported`

The supplied evidence is sufficient to establish the statement under the defined evidence scope.

Example:

```text
Evidence:
Unemployment moved from 4.3 to 4.1.

Statement:
Unemployment decreased by 0.2 percentage points.
```

Reference:

- `semantic_pilot_001`
- `semantic_pilot_002`

A statement may be supported while failing an older completeness contract.

This is the distinction demonstrated by `semantic_pilot_002`.

---

## `contradicted`

The supplied evidence directly conflicts with the statement.

Example:

```text
Evidence:
4.3 → 4.1

Statement:
Unemployment was higher at the end of the period.
```

Reference:

- `semantic_pilot_004`

Contradiction is stronger than lack of support.

---

## `insufficient_evidence`

The evidence does not establish the statement, but also does not necessarily establish its negation.

Example:

```text
Evidence:
August 2025 = 4.3
August 2026 = 4.1

Statement:
Every successive monthly observation declined.
```

The endpoints establish a net decline but not the monthly path.

Reference:

- `semantic_pilot_005`

A hypothetical statement can therefore be possible without being supported.

---

# 4. Evidence Sensitivity

Semantic support is a relationship between statement and evidence.

It is not an intrinsic property of the sentence.

The key contrast is:

```text
semantic_pilot_005
```

and:

```text
semantic_pilot_006
```

Both use the same temporal statement.

With only endpoints:

```text
support_status = insufficient_evidence
```

With an explicitly synthetic monthly series showing a decline at every step:

```text
support_status = supported
```

This establishes a core v1.9 principle:

> Do not classify statements merely by vocabulary or grammatical form.

A temporal claim is not automatically unsupported.

A causal phrase is not automatically false.

The available evidence determines the support relationship.

---

# 5. Assertion Strength

`assertion_strength` records how strongly the statement commits to its claim.

It is separate from support.

Current reviewed assertion-strength values include:

## `asserted`

The statement presents the claim as established.

Example:

```text
The decline in the federal funds rate caused the fall in unemployment.
```

Reference:

- `semantic_pilot_007`

---

## `qualified_inference`

The statement presents an interpretation as indicated or suggested by the evidence rather than as an established fact.

Typical constructions include:

```text
suggests
might suggest
indicating
might reflect
```

Example:

```text
The rise in the 10-year yield suggests markets became more concerned about inflation.
```

This differs from `qualified_possibility` because the statement is primarily describing an inference from observed evidence, rather than merely saying that an outcome or explanation may be possible.

Reviewed examples include:

- `semantic_challenge_002`
- `semantic_harvest_001`
- `semantic_harvest_003`
- `semantic_harvest_005`
- `semantic_harvest_006`

---

## `qualified_possibility`

The statement presents a possible explanation rather than an established conclusion.

Example:

```text
The decline in the federal funds rate may have contributed
to the fall in unemployment.
```

Reference:

- `semantic_pilot_008`

Both examples currently receive:

```text
support_status = insufficient_evidence
```

because the supplied endpoint comparisons do not establish the causal contribution.

However, the claims do not have the same strength.

This distinction must remain visible.

A qualified possibility should not silently become an established causal claim.

Likewise, lack of support for a qualified possibility is not evidence that the possibility is false.

---

# 6. Compliance Is Not Semantic Support

The legacy FRED auditor and the v1.9 semantic layer answer different questions.

Legacy audit:

> Does the narrative satisfy the required structural and claim-content contract?

Semantic annotation:

> Does the supplied evidence support what this statement asks the reader to accept?

These judgments can differ.

The clearest reference pair is:

```text
semantic_pilot_001
semantic_pilot_002
```

Both correctly report the same unemployment decline.

The Llama3 example includes:

- direction
- prior value
- current value
- delta

and passes the legacy audit.

The Mistral example reports the correct decline but omits prior and current values from the bullet itself.

It therefore fails the legacy per-bullet completeness requirement.

Its semantic statement remains supported.

Conceptually:

```text
structural compliance ≠ semantic support
```

and:

```text
semantic support ≠ analytical usefulness
```

These dimensions should remain separate.

---

# 7. Evidence Scope

For the first pilot, judgments are deliberately narrow.

Permitted support includes:

- supplied evidence items
- explicitly checkable arithmetic
- explicitly defined unit conversions
- context already encoded in the cited claim

Do not silently import:

- outside economic explanations
- unstated historical data
- assumptions about intermediate observations
- causal mechanisms
- forecasts
- general world knowledge used to rescue an unsupported claim

Later tasks may deliberately supply richer evidence.

If richer evidence changes the support relationship, the annotation should change accordingly.

---

# 8. Constructed Examples

Constructed examples are allowed when they test a specific semantic distinction.

They must be clearly marked as constructed or synthetic.

They must not be presented as observed model behavior.

The v0.1 pilot uses constructed cases to test:

- contradiction
- deterministic conversion
- sparse versus rich temporal evidence
- asserted versus qualified causal language

Observed-model examples and constructed challenge examples should remain distinguishable in later analysis.

---

# 9. Uncertainty and Review

A reviewer should not force a confident judgment merely to complete the schema.

When the distinction is genuinely unclear:

1. preserve the exact statement
2. preserve the supplied evidence
3. document the unresolved issue in review notes
4. keep the annotation provisional until the criterion is clarified

Uncertainty about the annotation framework is evidence about the framework.

It should not be hidden.

The goal is reproducibility, not artificial certainty.

---

# 10. Current Reference Set

The first reviewed fixture contains eight examples:

| ID | Claim kind | Support | Assertion strength | Origin |
|---|---|---|---|---|
| 001 | empirical_comparison | supported | asserted | observed |
| 002 | empirical_comparison | supported | asserted | observed |
| 003 | unit_conversion | supported | asserted | constructed |
| 004 | empirical_comparison | contradicted | asserted | constructed |
| 005 | temporal_pattern | insufficient_evidence | asserted | constructed |
| 006 | temporal_pattern | supported | asserted | constructed |
| 007 | causal_explanation | insufficient_evidence | asserted | constructed |
| 008 | causal_explanation | insufficient_evidence | qualified_possibility | constructed |

These examples are a starting reference set.

They are not a complete taxonomy of macroeconomic interpretation.

---

# 11. v1.9 Working Principle

The semantic layer should help answer:

```text
What did the model observe?

What did it calculate?

What did it infer?

What additional commitment did the wording introduce?

What evidence supports that commitment?

What evidence would be required if the current evidence is insufficient?
```

The objective is not to eliminate interpretation.

A useful analytical system should be able to move beyond raw numbers.

The objective is to make that movement visible and auditable.

---

# Current Corpus and Next Step

The current reviewed corpus contains:

```text
8 reference records
4 challenge records
6 naturally occurring model-output records
18 reviewed records total
20 semantic annotation units
```

The corpus now includes both statement-level and multi-unit annotations and has survived constructed challenges as well as naturally generated model language.

The working rule remains:

> Preserve interesting complexity without immediately expanding the ontology.

Do not add distinctions merely because more labels are imaginable. Add them when concrete examples show that the existing framework cannot represent something important.

The next phase is the first constrained semantic proposer:

```text
reviewed guide + reviewed examples + supplied evidence
                         ↓
                 model proposal
                         ↓
            held-out human comparison
```

The proposer is not an oracle. Its purpose is to test whether the reviewed semantic framework can be reproduced consistently enough to help bootstrap annotation of larger model-output populations.
