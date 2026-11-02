# Semantic Annotation Guide — Compact v0.1

## Task

Identify every distinct semantic commitment in the exact supplied
claim-bearing span, relative only to the supplied evidence.

One span may receive multiple annotations. Do not split, shorten,
rewrite, or replace the target span. Do not duplicate equivalent
annotations or invent unsupported category extensions.

Return claim_kind, support_status, assertion_strength, and a
specific evidence-sensitive rationale for each annotation.

## Claim kinds

empirical_comparison:
A directly stated, checkable comparison between supplied values,
dates, observations, or quantities. A movement between two
observations is a comparison, not automatically a sustained trend.

unit_conversion:
An explicit mathematical conversion of units or scale supported
by supplied values.

temporal_pattern:
A claim about persistence, trend, acceleration, deceleration,
recurrence, or evolution over time. Two endpoints generally
establish a change, not a continued trend or changing pace.

causal_explanation:
An asserted or qualified explanatory relationship in which one
factor produces or contributes to another. Co-movement alone
does not establish causation.

scope_generalization:
An inference extending limited evidence to a broader condition,
population, market, or economic environment.

magnitude_judgment:
A qualitative assessment of the size, intensity, significance,
or strength of a measured change or condition.

market_interpretation:
An inference about investor expectations, sentiment, positioning,
valuation, pricing motives, or other market behavior beyond a
direct observation.

policy_interpretation:
An inference about monetary or fiscal policy intent, stance,
reaction, consequences, or prospective decisions.

Use more than one kind when the span makes distinct commitments.
A qualifying phrase can introduce an additional semantic move.

## Evidential support

supported:
The supplied evidence directly establishes the claim, or provides
adequate explicit support for the stated inference. Explain the
specific supporting relationship.

contradicted:
The supplied evidence conflicts with the stated proposition.
Identify the conflicting observation or calculation.

insufficient_evidence:
The supplied evidence does not establish or contradict the claim.
The claim may be plausible or true, but would require information
not supplied.

Do not import outside facts, familiar macroeconomic relationships,
unstated observations, or model world knowledge to rescue a claim.

Distinguish direct numerical support from support for a broader
interpretation. A valid calculation does not automatically support
the economic explanation attached to it.

## Assertion strength

asserted:
The proposition is presented as a direct claim without meaningful
qualification.

qualified_inference:
The statement indicates an inferential interpretation with
language such as "suggests", "indicating", "may indicate",
or "might reflect".

qualified_possibility:
The statement presents an outcome or consequence as possible,
contingent, or prospective, using language such as "may prompt",
"could lead", or "might happen".

Determine assertion strength independently for each annotation.
Hedging does not establish evidential support.

## Review principles

Compliance with an instruction or citation requirement is not
semantic support.

Evaluate what the reader is actually asked to accept, including
modifiers, magnitude judgments, trends, and causal implications.

Do not label every historical comparison as a temporal pattern.
Do not infer a general economic condition from one isolated move.
Do not treat insufficient evidence as a false statement.

When distinctions overlap, preserve all materially different
semantic commitments without manufacturing redundant labels.

The classifier proposes annotations for human review; its output
is not ground truth.
