# Narrative from Selected Claims

- source_selected_claims_json: `/home/joe/ai-lab/benchmarks/results/narratives/suite_reasoning__temp0__vs__temp03_temp07__checks_pass_rate__selected_claims.json`
- model: `mistral`
- temperature: `0.1`
- num_predict: `512`

## Narrative

Observations:
- The Mistral model degrades on the 'attention_5sent' prompt at temperature 0.7 compared to the baseline [CLAIMS: suite_reasoning__attention_5sent__mistral__temp07_vs_temp0].
- The Mistral model consistently fails on the 'reasoning_trap' prompt across both temperature levels, indicating a stable floor of failure [CLAIMS: suite_reasoning__reasoning_trap__mistral__temp03_vs_temp0, suite_reasoning__reasoning_trap__mistral__temp07_vs_temp0].
- The Llama3:70b model demonstrates a stable ceiling of success on the 'attention_5sent' prompt across both temperature levels [CLAIMS: suite_reasoning__attention_5sent__llama3_70b__temp03_vs_temp0, suite_reasoning__attention_5sent__llama3_70b__temp07_vs_temp0].
- The Mistral model slightly degrades on the 'attention_5sent' prompt at temperature 0.3 compared to the baseline [CLAIMS: suite_reasoning__attention_5sent__mistral__temp03_vs_temp0].

Tradeoffs:
- Increasing temperature from 0.3 to 0.7 in the Mistral model worsens performance on the 'attention_5sent' prompt [CLAIMS: suite_reasoning__attention_5sent__mistral__temp03_vs_temp0, suite_reasoning__attention_5sent__mistral__temp07_vs_temp0].

Invariances:
- The Llama3:70b model maintains a stable ceiling of success on the 'attention_5sent' prompt across both temperature levels [CLAIMS: suite_reasoning__attention_5sent__llama3_70b__temp03_vs_temp0, suite_reasoning__attention_5sent__llama3_70b__temp07_vs_temp0].
- The Mistral model consistently fails on the 'reasoning_trap' prompt across both temperature levels [CLAIMS: suite_reasoning__reasoning_trap__mistral__temp03_vs_temp0, suite_reasoning__reasoning_trap__mistral__temp07_vs_temp0].

Cautions:
- The analysis is limited to the provided claims and may not fully capture all differences or similarities between models and prompts.
- Interpretations should be cautious as changes in performance metrics do not necessarily correlate with real-world performance improvements or declines.
