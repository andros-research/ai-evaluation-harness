# Narrative v0

- source_analysis_json: `/home/joe/ai-lab/benchmarks/results/narratives/suite_instruction__temp0__vs__temp03_temp07__checks_pass_rate__analysis.json`
- model: `mistral`
- temperature: `0.3`
- num_predict: `512`

## Narrative

Observations:
- For the prompt `attention_5sent`, all models show stable results, always passing or always failing, across both comparison experiments and the baseline.
- The prompt `verbosity_drift` shows improvement in checks pass rate for LLama3 and Mistral under experiment temp07 compared to the baseline, but degrades for experiment temp03. For LLama3:70b, the trend is reversed.
- The prompt `style_hemi` shows a consistent degradation in checks pass rate for both LLama3 and Mistral under both comparison experiments compared to the baseline, with an exception that LLama3:70b improves under experiment temp03.

Tradeoffs:
- Increasing model size (from LLama3 to LLama3:70b) can lead to degraded performance in prompts like `verbosity_drift` and `style_hemi`, but improved performance in others like `verbosity_drift`.
- The choice between LLama3 and Mistral might be a tradeoff depending on the specific prompt. For example, for `attention_5sent`, both models show stable performance, while for `style_hemi` and `verbosity_drift`, Mistral generally degrades more than LLama3.

Invariances:
- The prompt `follow_instr` shows a consistent failure to pass checks across all experiments and models.

Anomalies:
- An anomaly is the improvement observed in the `verbosity_drift` prompt for LLama3:70b under experiment temp03 compared to the baseline, while it degrades under experiment temp07 and for all other models.

Hypotheses:
- The improvements observed in the `verbosity_drift` prompt for LLama3:70b under experiment temp03 could be due to a unique interaction between this model and this specific prompt in this experiment, potentially related to its larger size or specific training data.
- The degradation in performance of LLama3 and Mistral for the `style_hemi` prompt across both comparison experiments compared to the baseline might indicate that these models struggle with complex language styles like hemimeria.
- The consistent failure of all models on the `follow_instr` prompt suggests that this benchmark may not adequately test following instructions, or there could be an issue with the instructions themselves. Further investigation is needed to confirm these hypotheses.
