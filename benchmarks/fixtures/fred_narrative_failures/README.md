# FRED narrative failure fixtures

These fixtures preserve representative LLM outputs that failed the narrative
generation contract.

## llama3_2026-07-09_bare-citation-format

The model reproduced all four correct claim IDs, but:

- used bare `[claim_id]` citations instead of `[CLAIMS: claim_id]`
- combined multiple claims into one bullet
- omitted prior values
- introduced unsupported interpretation

The strict citation parser therefore extracted zero valid citations.
