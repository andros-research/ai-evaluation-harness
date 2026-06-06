# v1.7.0 CPI/FRED Demo Loop

## Purpose

v1.7.0 turns the completed v1.6 FRED evidence loop into a screen-readable MVP demo.

The goal is not to add major new infrastructure. The goal is to make the existing claim → narrative → audit → repair → traceability workflow understandable to a smart outsider in 3–5 minutes.

## Demo thesis

The LLM is not the whole system. The interesting part is the harness around the model:

FRED context → deterministic source claims → selected claims → claim-cited narrative → audit → repair planning → traceability → demo report

## Target audience

The first target audience is a smart non-specialist collaborator, such as Luke.

The report should show both:

1. useful output
2. current limitations

## v1.7.0 acceptance criteria

v1.7.0 passes if one command can produce a complete demo report that:

1. clearly states what the system does
2. explains why the workflow is interesting
3. shows the selected source claims
4. shows the generated narrative
5. shows audit status
6. shows repair status
7. shows traceability from narrative back to source claims
8. clearly distinguishes factual validation from interpretation validation
9. includes a visible limitation / interpretation-risk section
10. can be understood on screen in roughly 3–5 minutes

## Non-goals for v1.7.0

v1.7.0 should not add:

- new data sources
- Fed speech ingestion
- full dashboard redesign
- token-level telemetry
- mechanistic interpretability
- generalized weekly note workflow
- complex interpretation-risk classification

Those belong in later v1.7.x, v1.8, or v2.0 work.

## Release gate

Before tagging v1.7.0:

- deterministic evidence-loop run passes
- LLM evidence-loop run passes
- completed_steps equals expected step count
- failed_steps = 0
- overall_ok = True
- demo report is generated from current run metadata
- demo report identifies runner mode and demo mode correctly
- report includes explicit limitation / interpretation-risk language