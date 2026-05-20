# FRED Claim Schema v0.1

## Purpose

The FRED-native claim layer converts structured macroeconomic source data into durable, auditable claim artifacts.

This layer is the first v1.6 step toward moving the harness from behavioral model profiling into source-grounded macro evidence. The goal is to create claims that can later be selected, cited, narrated, audited, repaired, and displayed in the dashboard.

A FRED-native claim is a structured statement derived from one or more FRED time series observations. Each claim records the source series, observation date, metric, comparison window, direction, supporting values, and narrative eligibility.

## Design Principles

1. **Source-grounded first**  
   Claims should be tied directly to observed macro data before any model-generated narrative is produced.

2. **Deterministic before generative**  
   The initial v0.1 claim builder should use deterministic rules, not LLM-generated claim text.

3. **Stable claim IDs**  
   Claim IDs should be deterministic, human-readable, and stable across reruns when the underlying source data has not changed.

4. **Narrative-ready but audit-friendly**  
   Claims should be readable enough to support narrative generation, while preserving enough structured metadata for downstream audit and repair.

5. **Small schema first**  
   v0.1 should avoid overengineering. Additional fields can be added later as the CPI demo loop, dashboard, and token telemetry layers mature.

## Relationship to Existing Harness

Earlier claim artifacts focused primarily on model behavior, prompt deltas, and narrative audit support.

The FRED-native claim layer introduces a separate source-evidence track:

```text
FRED structured data
  → deterministic macro claims
  → selected FRED claims
  → macro narrative generation
  → narrative audit
  → repair
  → dashboard traceability