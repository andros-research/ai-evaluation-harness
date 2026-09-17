# v1.8.3 Fresh-FRED Acceptance and Post-Release Regression

## Experiment

The tagged v1.8.3 release was tested on a fresh FRED context without
changing generation, audit, selection, or repair logic during the experiment.

Source revision:
`171959229e0068be16fda86d37868364974d6bfc`

Design: five batches, four intermediate-prompt cells, five repetitions
per cell per batch. The models were llama3 and mistral at temperatures
0.0 and 0.7: 100 attempts, with 25 observations per cell.

## Frozen acceptance result

The original population contained 90 workflow-ready observations and
10 upstream validation failures. Of the workflow-ready observations,
45 initially passed audit and 45 failed.

ACT produced 6 targeted repair successes, 38 unsupported abstentions,
and 1 workflow failure. No repair produced a full audit pass, and no
introduced errors were reported among the completed repairs.

Final accepted narratives therefore remained 45.

The workflow failure occurred during structural-normalizer probing:
the auditor recognized 20 bullets, while the normalizer indexed only
16. A fresh in-memory audit reproduced the stored audit's ordered
bullet texts and errors.

## Reporting correction

Commit `8d1fa01` makes workflow-outcome aggregation tolerate null outcomes.
The failed observation remains null at row level; the summary uses
the string category "None". No observation is dropped or relabeled.

The original acceptance package, reported with this correction, records
18 passed validator checks and 2 failed checks. Both failures reflect
the same workflow failure:

- audit_failure_routing_contract
- infrastructure_and_data_anomalies

This original verdict remains preserved.

## Parser correction

Commit `47a6c7b` aligns the normalizer's bullet-line indexing with the
auditor's extract_bullets() recognition.

The supported normalization grammar, eligibility guards, and genuine
narrative/audit mismatch checks remain unchanged.

Focused tests confirmed mixed-marker indexing, clean abstention on the
formerly crashing specimen, rejection of a genuinely inconsistent audit,
and continued operation of a known successful historical normalization.

## Saved-population regression

The same 100 saved observations were replayed into a separate output root.
No new model generations were performed.

| Outcome | Frozen acceptance | Parser regression |
|---|---:|---:|
| Population records completed | 99 | 100 |
| Workflow failures | 1 | 0 |
| No repair needed | 45 | 45 |
| Unsupported abstentions | 38 | 39 |
| Repairs executed | 6 | 6 |
| Upstream validation failures | 10 | 10 |
| Targeted repair successes | 6 | 6 |
| Full audit successes after repair | 0 | 0 |
| Introduced-error runs | 0 | 0 |

All ten saved-population regression checks passed. Only the known
observation changed its recorded outcome: workflow failure became a
clean unsupported abstention. The same six observations were repaired,
and their repaired narratives were byte-for-byte unchanged.

The replay preceded the parser commit. Its candidate_base_commit.txt
and candidate.patch preserve the tested source state; the correction
was subsequently committed as 47a6c7b.

These are observed results on the saved population, not guarantees
about every future input. The replay is a post-release regression,
not a replacement out-of-sample acceptance result.

## Artifact locations

Experiment root:
`benchmarks/results/fresh_fred_acceptance/v1_8_3_20260913/`

Original ACT:
`act_population_100/`

Original acceptance reporting and validation:
`act_population_100/aggregate_nullsafe/`

Post-release replay:
`act_population_100_parser_regression/`

Replay verdict:
`act_population_100_parser_regression/regression_check.json`

## Handoff

Keep the v1.8.3 tag and original acceptance artifacts unchanged.
Land the reporting and parser corrections, then begin v1.9 INTERPRET.
Further ACT enhancements remain backlog items unless needed by a
demonstrated blocker.
