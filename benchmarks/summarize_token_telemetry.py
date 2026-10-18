#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean, median


REPO_ROOT = Path(__file__).resolve().parents[1]

REVIEW_FIXTURE = (
    REPO_ROOT
    / "benchmarks/fixtures/"
    / "semantic_dashboard_reviewed_v0_1.jsonl"
)

TELEMETRY_ROOT = (
    REPO_ROOT
    / "benchmarks/results/token_telemetry"
)

OUTPUT_PATH = (
    TELEMETRY_ROOT
    / "token_telemetry_summary_v0_1.json"
)

SOURCE_MODEL = "llama3"


def load_reviewed_claims() -> list[dict]:
    return [
        json.loads(line)
        for line in REVIEW_FIXTURE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def telemetry_path(
    claim_id: str,
) -> Path:
    return (
        TELEMETRY_ROOT
        / claim_id
        / "token_telemetry_v0_1.json"
    )


def compact_token(
    token: dict,
) -> dict:
    return {
        "claim_token_index":
            token["claim_token_index"],
        "token_text":
            token["token_text"],
        "claim_text_fragment":
            token["claim_text_fragment"],
        "probability":
            token["probability"],
        "logprob":
            token["logprob"],
        "entropy_nats":
            token["entropy_nats"],
        "rank":
            token["rank"],
    }


def summarize_claim(
    reviewed: dict,
) -> dict:
    claim_id = reviewed["claim_id"]
    path = telemetry_path(claim_id)

    if not path.exists():
        raise RuntimeError(
            f"Missing telemetry artifact: {path}"
        )

    artifact = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if artifact.get("claim_id") != claim_id:
        raise RuntimeError(
            f"claim_id mismatch for {claim_id}"
        )

    claim = artifact["claim"]

    if (
        claim.get("aligned_claim_text")
        != claim.get("text")
    ):
        raise RuntimeError(
            f"Alignment mismatch for {claim_id}"
        )

    tokens = artifact["tokens"]

    if not tokens:
        raise RuntimeError(
            f"No tokens for {claim_id}"
        )

    probabilities = [
        float(token["probability"])
        for token in tokens
    ]

    logprobs = [
        float(token["logprob"])
        for token in tokens
    ]

    entropies = [
        float(token["entropy_nats"])
        for token in tokens
    ]

    rank_gt_1 = [
        token
        for token in tokens
        if int(token["rank"]) > 1
    ]

    mean_logprob = fmean(
        logprobs
    )

    mean_negative_logprob = (
        -mean_logprob
    )

    claim_perplexity = math.exp(
        mean_negative_logprob
    )

    lowest_probability_token = min(
        tokens,
        key=lambda token:
            float(token["probability"]),
    )

    highest_entropy_token = max(
        tokens,
        key=lambda token:
            float(token["entropy_nats"]),
    )

    annotations = (
        artifact
        .get("human_review", {})
        .get("reviewed_annotations", [])
    )

    return {
        "claim_id":
            claim_id,
        "claim_text":
            claim["text"],
        "prompt_variant":
            artifact[
                "source_generation"
            ]["prompt_variant"],
        "source_model":
            artifact[
                "source_generation"
            ]["model"],
        "review_status":
            artifact
            .get("human_review", {})
            .get("review_status"),
        "claim_kinds":
            sorted(
                {
                    annotation["claim_kind"]
                    for annotation
                    in annotations
                }
            ),
        "support_statuses":
            sorted(
                {
                    annotation["support_status"]
                    for annotation
                    in annotations
                }
            ),
        "assertion_strengths":
            sorted(
                {
                    annotation[
                        "assertion_strength"
                    ]
                    for annotation
                    in annotations
                }
            ),
        "reviewed_annotations":
            annotations,
        "token_count":
            len(tokens),
        "mean_probability":
            fmean(probabilities),
        "median_probability":
            median(probabilities),
        "mean_logprob":
            mean_logprob,
        "mean_negative_logprob":
            mean_negative_logprob,
        "claim_perplexity":
            claim_perplexity,
        "mean_entropy_nats":
            fmean(entropies),
        "max_entropy_nats":
            max(entropies),
        "rank_gt_1_count":
            len(rank_gt_1),
        "rank_gt_1_share":
            (
                len(rank_gt_1)
                / len(tokens)
            ),
        "lowest_probability_token":
            compact_token(
                lowest_probability_token
            ),
        "highest_entropy_token":
            compact_token(
                highest_entropy_token
            ),
    }


def main() -> None:
    reviewed_claims = [
        row
        for row in load_reviewed_claims()
        if row.get("source_model")
        == SOURCE_MODEL
    ]

    summaries = [
        summarize_claim(reviewed)
        for reviewed in reviewed_claims
    ]

    artifact = {
        "schema_version":
            "token_telemetry_summary_v0_1",
        "source_model":
            SOURCE_MODEL,
        "claim_count":
            len(summaries),
        "claims":
            summaries,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            artifact,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "TOKEN TELEMETRY SUMMARY"
    )
    print("-" * 100)

    print(
        f"{'CLAIM':<52} "
        f"{'MEAN P':>8} "
        f"{'MEAN H':>8} "
        f"{'PPL':>8} "
        f"{'R>1':>6}"
    )
    print("-" * 100)

    for summary in summaries:
        print(
            f"{summary['claim_id']:<52} "
            f"{summary['mean_probability']:>8.3f} "
            f"{summary['mean_entropy_nats']:>8.3f} "
            f"{summary['claim_perplexity']:>8.3f} "
            f"{summary['rank_gt_1_count']:>3}"
            f"/{summary['token_count']:<2}"
        )

    print()
    print(
        "output:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
