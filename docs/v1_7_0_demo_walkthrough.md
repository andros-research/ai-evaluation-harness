# v1.7.0 CPI/FRED Demo Walkthrough

## Purpose

This walkthrough demonstrates the v1.7.0 CPI/FRED evidence loop.

The goal is not to show that an LLM can write a macro paragraph. The goal is to show a small AI workflow where the model is only one component inside a controlled evidence loop.

The system:

```text
FRED macro context
  -> deterministic source-grounded claims
  -> selected evidence claims
  -> deterministic or local LLM narrative
  -> citation + numeric/directional audit
  -> repair plan, if needed
  -> source-to-narrative traceability
  -> screen-readable demo report
```

## One-sentence demo framing

Here’s a tiny version of what I mean by an AI workflow. It takes structured macro data, converts it into claims, asks a local model to write from those claims, audits whether the model stayed grounded, and shows exactly what it can and cannot verify. Right now it catches citation, numeric, and directional issues, but it does not yet fully validate interpretation. That gap is the next interesting layer.

## Demo setup

Start from the repo root:

```bash
conda activate ai-lab
cd ~/ai-lab
git status
```

Expected branch:

```text
v1.7.0/cpi-demo-loop
```

## Demo step 1: deterministic baseline

Run:

```bash
python benchmarks/run_fred_evidence_loop.py \
  --input-context benchmarks/data/fred_macro_context.json \
  --comparison-window 12m
```

What this shows:

* the full pipeline can run without an LLM narrative
* the deterministic version is stable and boring
* the system generates claims, selects claims, writes a constrained narrative, audits it, plans repair if needed, and records traceability

Expected result:

```text
overall_ok=True
completed_steps=7
failed_steps=0
audit_pass=True
repair_needed=False
```

Talking point:

The deterministic version is not the most interesting prose, but it establishes the controlled baseline. It proves the evidence loop works before adding LLM behavior.

## Demo step 2: LLM narrative with llama3

Run:

```bash
python benchmarks/run_fred_evidence_loop.py \
  --input-context benchmarks/data/fred_macro_context.json \
  --comparison-window 12m \
  --narrative-mode llm \
  --narrative-model llama3 \
  --ollama-host http://127.0.0.1:11434
```

Then inspect:

```bash
cat benchmarks/results/fred_demo/fred_demo_report.md
```

What this shows:

* the model writes a claim-cited narrative
* the audit verifies citations, numeric values, and direction
* the traceability layer maps narrative bullets back to source claims
* interpretation risk remains visible

Expected result:

```text
audit_pass=True
repair_needed=False
```

Talking point:

This is the useful version. The LLM writes something more natural than the deterministic template, but the output is still constrained by the claim layer and checked by the audit.

## Demo step 3: failure case with mistral

Run:

```bash
python benchmarks/run_fred_evidence_loop.py \
  --input-context benchmarks/data/fred_macro_context.json \
  --comparison-window 12m \
  --narrative-mode llm \
  --narrative-model mistral \
  --ollama-host http://127.0.0.1:11434
```

Then inspect:

```bash
cat benchmarks/results/fred_demo/fred_demo_report.md
```

Expected result:

```text
audit_pass=False
repair_needed=True
n_repair_actions=4
```

What this shows:

* the model cites the correct claims
* the model gets direction and delta values mostly right
* but the claim-cited summary omits the prior/current values required by the audit contract
* the repair planner catches the issue instead of silently accepting the output

Talking point:

This is a useful failure. Mistral did not hallucinate the citations and did not obviously write nonsense. But it failed the stricter evidence contract. The claim-cited bullets were supposed to include prior and current values, and they only included deltas. The model later included some values in a separate macro narrative paragraph, but the harness does not give it credit for putting required evidence in the wrong place.

That is the point of the audit layer: “looks fine” is not the same thing as “passed the evidence contract.”

## Demo step 4: larger model comparison with llama3:70b

Run:

```bash
python benchmarks/run_fred_evidence_loop.py \
  --input-context benchmarks/data/fred_macro_context.json \
  --comparison-window 12m \
  --narrative-mode llm \
  --narrative-model llama3:70b \
  --ollama-host http://127.0.0.1:11434
```

Then inspect:

```bash
cat benchmarks/results/fred_demo/fred_demo_report.md
```

Expected result:

```text
audit_pass=True
repair_needed=False
```

Talking point:

The larger model generally follows the evidence contract more cleanly here. This does not mean it is universally better, but it gives a simple example of why the harness can be used to compare model behavior under the same structured task.

## What to point at in the report

In `benchmarks/results/fred_demo/fred_demo_report.md`, focus on:

1. **Five-minute demo thesis**
   Explains the point of the system.

2. **Workflow at a glance**
   Shows the evidence loop.

3. **Selected source claims**
   Shows the structured claim layer before narrative generation.

4. **Generated narrative**
   Shows the model output.

5. **Audit and repair result**
   Shows whether the model satisfied the evidence contract.

6. **Current limitation: interpretation risk**
   Explains what the audit does and does not yet validate.

7. **Compact traceability view**
   Shows claim selection, citation, audit status, and repair status in a screen-readable format.

## Current validation boundary

The current audit checks:

* citation coverage
* unknown citations
* missing citations
* numeric values
* direction
* required prior/current values in claim-cited bullets

The current audit does not yet fully validate:

* interpretive adjectives
* market-facing language
* causal explanations
* policy implications
* whether phrases like “significant,” “deterioration,” or “market expectations” are justified

This limitation is intentional in v1.7.0. The system validates factual traceability first and makes interpretation risk visible rather than hiding it.

## Demo close

The current system is intentionally small. But the pattern is the point.

A normal AI demo shows a final answer. This demo shows the workflow around the answer: structured inputs, claim construction, model generation, audit, repair planning, and traceability.

That is the larger thesis: the real value is not just smarter chatbots, but controlled AI workflows around models.
