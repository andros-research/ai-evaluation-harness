# Methodology (Phase 1)

This project is a reproducible evaluation harness for local LLMs (via Ollama).

## Core ideas
- Fixed prompt suites + multi-rep runs to characterize stochastic output distributions.
- Log both structured metrics (CSV) and raw response corpora (JSONL/text) for auditing.
- Treat invalid runs as first-class data (exit_code, empty output, timeout, etc).
- Keep a “paper trail” per run: metadata, prompts, settings, artifacts.

## Outputs per run
- metrics.csv: one row per (prompt, model, rep)
- metadata.json: run configuration (models, options, timeouts, suite)
- per-prompt artifacts: prompt.txt, response text, raw JSON
- responses.jsonl: append-only corpus for downstream embedding + clustering (Phase 2)
