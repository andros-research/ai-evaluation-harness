# v1.7.1 Model Comparison Notes

## Purpose

Document observed model behavior in the CPI/FRED evidence-loop demo.

## Models tested

| Mode / Model | Result | Audit | Repair | Demo role |
| --- | --- | --- | --- | --- |
| deterministic | pass | true | false | stable baseline |
| llama3 | pass | true | false | useful local LLM narrative |
| mistral | fail | false | true | real audit/repair demo |
| llama3:70b | pass | true | false | stronger contract-following comparison |

## Key observation

The most interesting result is not that one model “wins.” The interesting result is that different models produce different failure signatures under the same evidence contract.

## Mistral failure mode

Mistral cited the correct claims and included the delta values, but the claim-cited summary bullets omitted prior/current values. It later included some current values in a separate freeform macro narrative section, but the audit treats the claim-cited bullets as the controlled auditable units.

This means the model put relevant information in the wrong place.

## Why this matters

A human skimming the output might think it was basically fine. The harness catches that the output failed the stricter evidence contract.

This demonstrates the value of workflow-level validation around LLM output.

## Demo talking point

“Looks fine” is not the same thing as “passed the evidence contract.”