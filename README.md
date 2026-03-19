# ai-evaluation-harness

A reproducible evaluation harness for studying LLM behavior through controlled prompt suites, repeated runs, and comparative experiment analysis — designed for local model experimentation and systematic behavioral measurement.

This project is designed to systematically measure how local LLM behavior changes under controlled inference conditions (e.g., temperature sweeps), with a focus on identifying tradeoffs across competing constraints like structure, instruction following, and verbosity.

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

## How this can be used

From a risk perspective, this harness can be used to map how model behavior shifts under controlled inference changes (e.g., temperature sweeps), similar to stress testing a system across different regimes. 

For example, a model that performs well on structured outputs at low temperature may degrade in instruction-following or verbosity control as temperature rises. By running repeated evaluations across prompt types and aggregating results, the harness surfaces these tradeoffs explicitly, allowing a user to identify stable operating regions and failure modes.

In practice, this can inform model selection, prompt design, and guardrail strategies by making behavioral reliability measurable rather than anecdotal.

## Repository layout

```
benchmarks/
  Core benchmark suites, runners, aggregation, and analysis helpers

dashboards/
  Streamlit dashboard for evaluation analytics

docs/
  Research notes, images, and project documentation

experiments/
  Scratch / future experimental work
```

## Quick start

1. Create the environment
```bash
conda env create -f environment.yml
conda activate ai-lab
```

2. Run a benchmark suite
From the repository root:
```bash
cd benchmarks
python run_suite.py --suite suite_instruction.yml --runner runner_temp0.yml
```

3. Aggregate outputs
```bash
python aggregate_runs.py
```
This produces the aggregated `runs_master` artifacts used by the dashboard and later analysis steps.

4. Launch the dashboard
From the repository root:
```bash
streamlit run dashboards/eval_dashboard.py
```

## Dashboard preview

### Prompt × model pass-rate heatmap
<p align="center">
  <img src="docs/images/dashboard-passrate-heatmap.png" width="800"/>
</p>
Pass-rate heatmap across prompt × model combinations, showing constraint-specific degradation patterns across temperature sweeps.

### Within-suite experiment comparison
<p align="center">
  <img src="docs/images/dashboard-experiment-comparison.png" width="800"/>
</p>

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