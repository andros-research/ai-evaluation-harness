# ai-evaluation-harness

A reproducible evaluation harness for measuring LLM behavior via:
- prompt suites
- multi-rep runs (distributional measurement)
- metrics logging (CSV)
- response corpora (JSONL/text)
- dashboard analytics (Streamlit)

## Why this exists
I’m building a research-grade “behavioral observatory” for local models as a foundation for:
- reliability + drift monitoring
- failure mode discovery
- future agent evaluation (macro “situation monitor” project)

## Repo layout
- `benchmarks/` — suite config + runner (writes `results_*`)
- `dashboards/` — Streamlit dashboard for exploring metrics
- `docs/` — methodology + roadmap

## Status
Phase 1 is functional; Phase 2 will introduce embeddings, clustering, and drift analysis.

## Running
(Instructions will be added once Phase 1 is stabilized and packaged.)
