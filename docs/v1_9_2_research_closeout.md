# v1.9.2 Research Closeout

## Native semantic-token experiments and model comparison

v1.9.2 moved the AI Evaluation Harness from a reproducible token-telemetry
baseline into a controlled experimental system for studying relationships
between semantic interpretation and native model-generation behavior.

## What v1.9.2 established

### 1. Native generation-time token telemetry

The project replaced the earlier historical teacher-forced rescoring setup
with direct BF16 generation-time telemetry.

The validated Llama 3 run preserved selected-token probability, log
probability, entropy, vocabulary rank, and top alternatives during native
generation.

### 2. Probability-blind semantic boundary experiment

Five fresh September-2026 macro commentary sentences were human-reviewed
before token telemetry was revealed.

Three observation-to-interpretation boundaries were frozen.

All three boundary tokens showed elevated entropy relative to other comma
continuations. Four-token pre/post entropy increased across all three
boundaries. After removing the transition token itself, two of three
retained elevated post-boundary entropy.

The Treasury-yield-to-market-expectations transition produced the strongest
local signal.

These findings remain descriptive and hypothesis-generating. Punctuation,
lexical branching, prompt/evidence repetition, non-independent tokens, and
small sample size remain important confounds.

### 3. gpt-oss-20b added as a second model architecture

The official openai/gpt-oss-20b checkpoint was installed locally on the
NVIDIA GB10 Spark and run directly with Hugging Face / PyTorch.

Observed architecture:

- 24 transformer layers
- hidden size 2880
- 64 attention heads
- 8 KV heads
- alternating sliding/full attention
- 32 local experts per MoE layer
- 4 experts selected per token
- 131072 maximum positions
- Harmony system/developer/user/assistant protocol
- explicit low / medium / high reasoning effort

The work distinguished:
- autoregressive token generation,
- MoE routing inside transformer layers,
- Harmony conversation/channel structure,
- explicit reasoning-token generation,
- higher-level model/system routing.

### 4. Reasoning-effort experiment

Using the same classifier-shaped task:

- low reasoning produced short analysis traces;
- medium produced substantially longer analysis;
- high frequently produced extremely long and repetitive analysis.

A fixed-prompt reproducibility experiment generated the exact same 105-token
sequence across five consecutive greedy runs in one loaded process.

Saved top-token probabilities, top-two margins, and entropy were also
identical across all five runs.

### 5. Three-case controlled classifier challenge

Historical classifier prompts were reconstructed byte-for-byte and verified
against their frozen SHA-256 values before reuse.

gpt-oss-20b low, medium, and high reasoning were compared with the historical
Llama 3 70B classifier.

The pilot suggested that low reasoning could improve some semantic judgments,
while high reasoning frequently spent its generation budget deliberating
rather than returning the requested structured answer.

### 6. Expanded frozen classifier benchmark

A new primary benchmark was frozen from the human-reviewed semantic corpus.

Primary set:
- 23 claims
- 52 frozen annotation triples

One legacy causal_claim case was excluded from the primary benchmark because
that frozen label is not part of the current compact ontology.

Three few-shot examples and one deferred review case were also excluded.

A fresh Llama 3 70B classifier baseline was generated under the corrected
compact classifier setup rather than relying on the older historically
truncated classifier runs.

gpt-oss-20b was then run at low, medium, and high reasoning against the same
frozen classifier tasks.

## Expanded benchmark results

Strict primary scoring:

| Condition | Valid | Exact | Partial | Miss | Protocol | Noncomplete | Matched labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama 3 70B | 23 | 4 | 16 | 3 | 0 | 0 | 26 / 52 |
| gpt-oss low | 22 | 5 | 13 | 4 | 1 | 0 | 26 / 52 |
| gpt-oss medium | 20 | 4 | 13 | 3 | 3 | 0 | 24 / 52 |
| gpt-oss high | 4 | 3 | 1 | 0 | 1 | 18 | 5 / 52 |

The semantic-only salvage view for the one low-reasoning protocol failure
raised gpt-oss-low to 27 / 52 matched labels.

## Main classifier finding

The largest result was not a clear Llama-versus-gpt-oss winner.

Instead, the benchmark exposed strong ontology-specific failure modes:

- empirical_comparison: Llama 17/17; OSS-low 16/17
- causal_explanation: Llama 2/2; OSS-low 1/2
- policy_interpretation: both 4/6
- temporal_pattern: Llama 1/4; OSS-low 2/4
- magnitude_judgment: Llama 0/6; OSS-low 1/6
- scope_generalization: Llama 1/15; OSS-low 1/15

The dominant bottleneck is therefore now the semantic task/prompt design,
especially reliable decomposition of broader scope claims and qualitative
magnitude judgments.

## Reasoning-effort finding

For this constrained semantic-classification task:

- low reasoning completed almost every case and had the strongest gpt-oss
  semantic result;
- medium reasoning consumed substantially more analysis tokens without
  improving aggregate semantic fidelity;
- high reasoning was operationally unsuitable under a 2048-token generation
  budget, failing to produce a final answer in 18 of 23 cases.

Reasoning effort therefore showed strongly non-monotonic value for this
narrow task.

## Version boundary

v1.9.2 closes the experimental stage in which semantic labels and native
token telemetry became usable research instruments across multiple model
architectures.

The next version should not continue tuning these same frozen cases.

## v1.9.3 direction — Multi-Model Macro Lab

v1.9.3 begins by redesigning the semantic classifier as v0.2 using the
v1.9.2 corpus as a development/diagnostic set.

Primary objectives:

1. strengthen scope_generalization and magnitude_judgment identification;
2. explicitly teach exhaustive multi-label decomposition;
3. use gpt-oss/Harmony natively rather than preserving the old Llama-era
   monolithic classifier interface;
4. generate new macro narratives from identical frozen evidence with both
   Llama 3 BF16 and gpt-oss-20b;
5. human-review those narratives probability-blind;
6. use those new annotations as an out-of-sample classifier-v0.2 evaluation;
7. capture native token telemetry for both generator models;
8. build the first multi-model dashboard view.

Longer-term data expansion should add weekly H.4.1 and other weekly/monthly
macro data with explicit observation, publication, retrieval, and frequency
metadata.

Future generator comparisons may add Glimmer or another modern open-weight
model.

## Toward v2.0

The v2.0 threshold remains a repeatable macro research engine:

new macro state
→ multiple model narratives
→ semantic hypotheses
→ quantitative analytical triggers
→ evidence tests
→ revised conclusions
→ auditable multi-model dashboard.

Technology remains subordinate to the research objective: improve the
quality, explicitness, and revisability of macroeconomic judgment.
