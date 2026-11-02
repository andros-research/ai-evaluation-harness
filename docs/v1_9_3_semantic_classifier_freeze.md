# v1.9.3 Semantic Classifier Freeze

## Selected configuration

`compact_balanced_v0_2_1`

Parent: `compact_balanced_v0_2`

Model: `openai/gpt-oss-20b`

Reasoning effort: `low`

## Development result

The v0.2 redesign materially improved the primary v1.9.2 failure modes
on the 21 non-demonstration development cases:

- strict triple recall: 51.1% -> 71.1%
- strict triple precision: 45.1% -> 52.5%
- scope-generalization claim-kind recall: 23.1% -> 92.3%
- magnitude-judgment claim-kind recall: 50.0% -> 100.0%
- empirical-comparison claim-kind recall: 93.8% -> 100.0%

The remaining systematic weakness was assertion-strength calibration,
especially:

- asserted vs qualified_inference
- qualified_inference vs qualified_possibility

v0.2.1 made one narrow rule-level clarification concerning assertion
strength locality and precedence. It changed neither the ontology nor
the four few-shot demonstrations.

A ten-case targeted regression showed partial improvement:

- several market-interpretation strength errors were corrected;
- one temporal-pattern asserted/inference error was corrected;
- policy strength improved in one mixed case;
- three controls remained stable;
- some temporal and causal assertion-strength errors remained;
- one causal commitment was omitted in a targeted case.

## Decision

Freeze v0.2.1 as the classifier entering fresh macro evaluation.

Do not perform additional prompt tuning on the existing development
corpus.

Remaining assertion-strength errors are treated as a known model /
calibration limitation and will be evaluated on fresh out-of-sample
macro narratives.

Any future semantic-guide revision must occur only after the fresh
evaluation is frozen and reviewed.

## Runtime

Working long-prompt configuration on NVIDIA GB10:

- PyTorch 2.14.0a0 NVIDIA 26.08
- Transformers 5.22.1
- CUDA 13.4 container runtime
- Flex Attention
- FORCE_USE_FLEX_ATTENTION=True
- approximately 14.7k prompt tokens under v0.2.1
