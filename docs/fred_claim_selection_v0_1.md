# FRED Claim Selection v0.1

## Purpose

The FRED claim selection layer converts deterministic FRED-native claim artifacts into a selected evidence set for downstream narrative generation.

This is the first selection step after the v1.6.0 FRED-native claim layer. It does not yet rank claims by importance, novelty, macro relevance, or narrative usefulness. The initial v0.1 selector intentionally uses a simple eligibility filter so the pipeline remains deterministic and auditable.

## Pipeline Position

```text
fred_macro_context.json
  -> build_fred_claims.py
  -> fred_claims.csv/json/metadata.json
  -> select_fred_claims.py
  -> selected_fred_claims.csv/json/metadata.json
  -> future narrative generation
  -> future narrative audit
  -> future repair