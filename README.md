# ai-evaluation-harness

A reproducible evaluation harness for studying LLM behavior through controlled prompt suites, repeated runs, and comparative experiment analysis.

## v1.0 milestone

This repository’s v1.0 milestone establishes the core behavioral harness:

- modular YAML prompt suites
- reusable runner configs for controlled inference sweeps
- multi-rep runs for distributional measurement
- aggregated benchmark outputs via `runs_master`
- Streamlit dashboard analytics, including:
  - prompt × model summaries
  - pass-rate heatmaps
  - within-suite experiment comparison tables
  - signed delta heatmaps

This version is focused on **behavioral measurement and comparison**.  
Later versions extend this toward narrative generation, validation, and richer telemetry.

## Why this exists

The goal of this project is to build a practical local-model evaluation lab for measuring:

- reliability across prompt types
- behavioral drift under inference changes
- tradeoffs across constraints like structure, style, verbosity, and attention
- foundations for later interpretability / telemetry work

Temperature is treated here not just as “randomness,” but as a behavioral control knob that can shift model performance across competing dimensions.

## Repository layout

benchmarks/   Core benchmark suites, runners, aggregation, and analysis helpers
dashboards/   Streamlit dashboard for evaluation analytics
docs/         Research notes, images, and project documentation
experiments/  Scratch / future experimental work

## Quick start

1. Create the environment
`conda env create -f environment.yml`
`conda activate ai-lab`

2. Run a benchmark suite
From the repository root:
`cd benchmarks`
`python run_suite.py --suite suite_instruction.yml --runner runner_temp0.yml`

3. Aggregate outputs
`python aggregate_runs.py`
This produces the aggregated `runs_master` artifacts used by the dashboard and later analysis steps.

4. Launch the dashboard
From the repository root:
`streamlit run dashboards/eval_dashboard.py`

## Dashboard preview

Within-suite experiment comparison
Screenshot to be added under `docs/images`
Prompt × model pass-rate heatmap
Screenshot to be added under `docs/images`

## Example early findings

Across repeated temperature sweeps on the initial suites, the harness has already surfaced several stable patterns:

- some structural output constraints are largely temperature-insensitive
- some instruction-following failures appear saturated across all tested temperatures
- style and verbosity constraints can move in opposite directions as temperature rises
- some attention constraints remain stable at low temperature and degrade only at higher temperature

These are early behavioral results rather than final scientific claims, but they demonstrate that the harness can detect real tradeoff surfaces in model behavior.

## Version roadmap

- v1.0 — behavioral harness complete
- v1.1 — early narrative generation over analysis payloads
- v2.0 — richer telemetry inputs and PyTorch-backed execution

## Notes

Generated benchmark artifacts can become large quickly. The public repository is intended to keep the core framework, selected examples, and documentation clean, while broader local result corpora remain excluded from version control unless explicitly curated.

## License

Currently unlicensed / all rights reserved unless and until a license is added.