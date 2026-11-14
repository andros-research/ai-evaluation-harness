# Semantic Annotation Guide — Compact v0.2.1

## Task

Identify every distinct semantic commitment in the exact supplied
claim-bearing span, relative only to the supplied evidence.

One span may receive multiple annotations. Do not split, shorten,
rewrite, or replace the target span.

Do not stop after identifying the most obvious annotation.

Return one annotation for every materially distinct semantic
commitment expressed by the span.

Do not duplicate equivalent annotations or invent unsupported
category extensions.

Return claim_kind, support_status, assertion_strength, and a
specific evidence-sensitive rationale for each annotation.


## Required annotation procedure

Evaluate each target in three separate stages.

### Stage 1 — Identify claim kinds

Before judging evidential support, independently test whether the
span contains each of the following semantic moves:

1. empirical_comparison
2. unit_conversion
3. temporal_pattern
4. scope_generalization
5. magnitude_judgment
6. causal_explanation
7. market_interpretation
8. policy_interpretation

Do not stop after one category matches.

A single phrase may create more than one semantic commitment.

For example:

"The unemployment rate fell, indicating a strong labor market."

may contain:

- an empirical comparison about unemployment;
- a magnitude judgment in the word "strong";
- a scope generalization from one unemployment statistic to the
  broader labor market.

These are independently assessable commitments and should not be
collapsed into one generic interpretation label.


### Stage 2 — Judge evidential support separately

For every identified semantic commitment, determine whether the
supplied evidence:

- directly supports it;
- contradicts it; or
- is insufficient to establish it.

Support for one annotation does not transfer automatically to
another annotation in the same span.

A numerical observation may be fully supported while an attached
magnitude judgment, broader generalization, trend, or interpretation
remains insufficiently supported.


### Stage 3 — Judge assertion strength separately

For every annotation, determine whether the proposition is:

- asserted;
- a qualified inference; or
- a qualified possibility.

Assertion strength is independent of claim kind and evidential
support.


## Claim kinds

### empirical_comparison

A directly stated, checkable comparison between supplied values,
dates, observations, or quantities.

A movement between two observations is a comparison, not
automatically a sustained trend.

Example:

"The unemployment rate fell from 4.3% to 4.1%."

This is an empirical comparison.


### unit_conversion

An explicit mathematical conversion of units or scale supported
by supplied values.


### temporal_pattern

A claim about persistence, trend, acceleration, deceleration,
recurrence, continuation, or evolution over time.

Two endpoints generally establish a change, not a continued trend,
persistent condition, acceleration, or changing pace.

Words and phrases such as the following may signal a temporal
pattern:

- continued
- persistent
- sustained
- accelerating
- slowing
- increasingly
- remains
- trend
- momentum

Example:

"CPI increased."

may be only an empirical comparison.

"CPI continued its upward trend."

adds a temporal-pattern commitment that requires evidence beyond
a single pair of endpoints.


### scope_generalization

An inference extending limited evidence from a narrower metric,
observation, population, instrument, or series to a broader
condition, system, market, or economic environment.

Explicitly ask:

"Is the sentence moving from what was directly measured to
something broader?"

Common forms include:

- unemployment rate -> labor market
- one inflation series -> inflation pressures
- one yield -> bond market conditions
- one market price -> investor expectations
- one economic indicator -> overall economy
- one policy rate -> broader policy stance

Example:

"The unemployment rate fell."

does not by itself require a scope-generalization annotation.

"The unemployment rate fell, indicating a stronger labor market."

moves from one unemployment statistic to the broader labor market
and therefore contains a scope-generalization commitment.

Scope generalization may coexist with magnitude judgment,
empirical comparison, market interpretation, policy
interpretation, or other claim kinds.


### magnitude_judgment

A qualitative assessment of the size, intensity, significance,
degree, or strength of a measured change or condition.

Explicitly inspect adjectives and adverbs that characterize
magnitude rather than merely report the observation.

Common examples include:

- strong
- weak
- modest
- substantial
- significant
- elevated
- limited
- severe
- mild
- robust
- sharp
- large
- small

Example:

"The unemployment rate fell."

is not necessarily a magnitude judgment.

"The unemployment rate fell, indicating a strong labor market."

contains a magnitude judgment because "strong" asks the reader to
accept a qualitative assessment of degree or condition.

A magnitude judgment can be insufficiently supported even when the
underlying empirical comparison is directly supported.


### causal_explanation

An asserted or qualified explanatory relationship in which one
factor produces, contributes to, drives, explains, causes, or
otherwise generates another outcome.

Co-movement alone does not establish causation.

Do not use causal_explanation merely because one fact is followed
by another. There must be an explanatory relationship.


### market_interpretation

An inference about investor expectations, sentiment, positioning,
valuation, pricing motives, risk appetite, or other market behavior
beyond a direct observation.

Example:

"The 10-year Treasury yield increased."

may be an empirical comparison.

"The 10-year Treasury yield increased, potentially reflecting
expectations of future rate hikes."

adds a market-interpretation commitment.


### policy_interpretation

An inference about monetary or fiscal policy intent, stance,
reaction, consequences, or prospective decisions.

Example:

"The effective federal funds rate declined."

may be an empirical comparison.

"The effective federal funds rate declined, potentially signaling
a more accommodative monetary policy."

adds a policy-interpretation commitment.


## Contrastive multi-label examples

### Example A — observation only

"The unemployment rate fell from 4.3% to 4.1%."

Possible annotation:

- empirical_comparison

Do not add scope_generalization or magnitude_judgment merely
because unemployment is an economically important variable.


### Example B — observation plus magnitude plus scope

"The unemployment rate fell, indicating a strong labor market."

Possible distinct annotations:

- empirical_comparison
- magnitude_judgment
- scope_generalization

"strong" adds a magnitude assessment.

"labor market" broadens the claim from one unemployment statistic
to a wider economic condition.


### Example C — observation plus trend plus broader condition

"CPI rose, showing persistent inflation pressures."

Possible distinct annotations:

- empirical_comparison
- temporal_pattern
- scope_generalization

The observed CPI move does not by itself establish persistence.

The phrase "inflation pressures" may also broaden beyond the
specific measured observation.


### Example D — observation plus interpretation

"The 10-year Treasury yield rose, potentially reflecting market
expectations of further rate hikes."

Possible distinct annotations:

- empirical_comparison
- market_interpretation

Do not collapse the observed yield move and the explanation of
market expectations into one annotation.


## Evidential support

### supported

The supplied evidence directly establishes the claim, or provides
adequate explicit support for the stated inference.

Explain the specific supporting relationship.


### contradicted

The supplied evidence conflicts with the stated proposition.

Identify the conflicting observation or calculation.


### insufficient_evidence

The supplied evidence does not establish or contradict the claim.

The claim may be plausible or true, but would require information
not supplied.

Do not import outside facts, familiar macroeconomic relationships,
unstated observations, or model world knowledge to rescue a claim.

Distinguish direct numerical support from support for a broader
interpretation.

A valid calculation does not automatically support:

- a magnitude judgment;
- a temporal pattern;
- a scope generalization;
- a causal explanation;
- a market interpretation; or
- a policy interpretation.


## Assertion strength

### asserted

The proposition is presented as a direct claim without meaningful
qualification.


### qualified_inference

The statement presents an inferential interpretation using language
such as:

- suggests
- indicating
- may indicate
- might reflect
- signaling
- consistent with

This category normally describes a qualified interpretation of
what the evidence means.


### qualified_possibility

The statement presents an outcome, consequence, or event as
possible, contingent, or prospective, using language such as:

- may prompt
- could lead
- might happen
- could result
- may cause

This category normally describes a possible outcome rather than
an interpretation of an already observed relationship.


Determine assertion strength independently for each annotation.

Hedging does not establish evidential support.



## Assertion-strength locality and precedence

Determine assertion strength from the language governing the specific
semantic commitment being annotated, not from the overall rhetorical
tone of the sentence.

Do not propagate a hedge, inference marker, evidential phrase, or
qualifier from one proposition to every annotation in the span.

### asserted

Use asserted when the specific proposition is stated directly as true,
present, or occurring, even if another clause explains, supports, or
interprets that proposition.

Example:

"Inflationary pressures are building, as evidenced by higher CPI."

The temporal-pattern commitment "inflationary pressures are building"
is asserted.

The phrase "as evidenced by" supplies justification for the proposition;
it does not make the temporal proposition itself a qualified inference.

### qualified_possibility

Use qualified_possibility when the specific proposition is framed as a
possible, contingent, prospective, or uncertain outcome.

Typical constructions include:

- could be a sign of
- may prompt
- could lead to
- might cause
- may result in
- potentially producing

Example:

"The yield rise could be a sign of increased optimism."

The market-interpretation commitment is qualified_possibility because
the proposition itself is explicitly framed as one possible explanation.

### qualified_inference

Use qualified_inference when the proposition is presented as an
interpretation of observed evidence: what the evidence suggests,
indicates, reflects, signals, or is consistent with.

Typical constructions include:

- suggests
- indicates
- may indicate
- might reflect
- signals
- is consistent with

Example:

"The yield rise might reflect expectations of higher rates."

The market-interpretation commitment is qualified_inference. Although
"might" hedges the sentence, "reflect expectations" expresses an
inferential interpretation of what the observed market move means.

### Local decision rule

Apply assertion strength independently to every annotation.

For each semantic commitment, ask:

1. Is this proposition directly stated as true or occurring?
   -> asserted

2. Is this proposition explicitly one possible outcome, explanation,
   or consequence?
   -> qualified_possibility

3. Is this proposition an interpretation of what observed evidence
   means or implies?
   -> qualified_inference

Do not determine assertion strength by keyword alone.

In particular:

- "might reflect" can be qualified_inference;
- "could be a sign of" can be qualified_possibility;
- an asserted proposition does not become qualified merely because
  another clause says "as evidenced by", "because", or otherwise
  supplies justification.


## Final exhaustive scan

Before returning the answer, reread the complete claim span and ask:

1. Did I identify every directly stated empirical comparison?
2. Did the claim assert persistence, trend, or changing pace?
3. Did it move from a narrow observation to a broader system or
   condition?
4. Did it characterize anything as strong, weak, significant,
   modest, elevated, severe, or otherwise judge magnitude?
5. Did it explain why something happened?
6. Did it interpret market expectations, sentiment, pricing, or
   behavior?
7. Did it interpret policy stance, intent, reaction, or possible
   consequences?
8. Did a qualifier introduce an additional semantic commitment?
9. Did I accidentally merge two independently assessable claims?
10. Did I accidentally duplicate the same semantic move?

Only after this exhaustive scan should the annotation set be
considered complete.


## Review principles

Compliance with an instruction or citation requirement is not
semantic support.

Evaluate what the reader is actually asked to accept, including
modifiers, magnitude judgments, trends, broader scope claims, and
causal implications.

Do not label every historical comparison as a temporal pattern.

Do not infer a general economic condition from one isolated move
without separately identifying the scope-generalization step.

Do not treat a qualitative adjective as mere writing style when it
materially changes the proposition being asserted.

Do not treat insufficient evidence as a false statement.

When distinctions overlap, preserve all materially different
semantic commitments without manufacturing redundant labels.

The classifier proposes annotations for human review; its output
is not ground truth.
