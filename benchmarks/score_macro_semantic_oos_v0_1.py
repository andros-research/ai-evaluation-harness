#!/usr/bin/env python3
"""Score frozen GPT-OSS v0.2.1 OOS proposals against the frozen human reference.

Run only AFTER the generation_frozen_v0_1.json artifact has been created.
No model-identity mapping, hidden model analysis, or token telemetry is opened.
"""
from __future__ import annotations

import csv
import hashlib
import os
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = os.environ.get(
    "MACRO_OOS_EXPERIMENT_ID",
    "v1_9_3_sep2026_multi_model_v0_1",
)
EXP_DIR = ROOT / "benchmarks/fixtures/macro_narrative_experiments" / EXPERIMENT
RUN_DIR = ROOT / "benchmarks/results/macro_semantic_oos" / EXPERIMENT / "gpt_oss_20b_low_v0_2_1"
REFERENCE = EXP_DIR / "human_review_v0_1/human_macro_semantic_reference_v0_1.json"
FREEZE_MANIFEST = EXP_DIR / "human_review_v0_1/human_reference_freeze_manifest_v0_1.json"
INPUT_FILE = RUN_DIR / "evaluation_input_v0_1.json"
GENERATION = RUN_DIR / "generation_frozen_v0_1.json"
REPORT_JSON = RUN_DIR / "scored_comparison_v0_1.json"
REPORT_MD = RUN_DIR / "scored_comparison_v0_1.md"
EXPECTED_REFERENCE_SHA = json.loads(
    FREEZE_MANIFEST.read_text(
        encoding="utf-8"
    )
)["reference_sha256"]

ALLOWED_KINDS = {
    "empirical_comparison", "unit_conversion", "temporal_pattern",
    "scope_generalization", "magnitude_judgment", "causal_explanation",
    "market_interpretation", "policy_interpretation",
}
ALLOWED_SUPPORT = {"supported", "contradicted", "insufficient_evidence"}
ALLOWED_STRENGTH = {"asserted", "qualified_inference", "qualified_possibility"}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def triple(a: dict) -> tuple[str, str, str]:
    return a.get("claim_kind"), a.get("support_status"), a.get("assertion_strength")


def valid(a: dict) -> bool:
    return (isinstance(a, dict)
            and a.get("claim_kind") in ALLOWED_KINDS
            and a.get("support_status") in ALLOWED_SUPPORT
            and a.get("assertion_strength") in ALLOWED_STRENGTH)


def matched_parts(gold: list[dict], predicted: list[dict]) -> dict:
    g = Counter(triple(a) for a in gold)
    p = Counter(triple(a) for a in predicted)
    matched = g & p
    extras = p - g
    missing = g - p
    gkind = Counter(k for k, _, _ in g.elements())
    pkind = Counter(k for k, _, _ in p.elements())
    kind_matched = gkind & pkind
    kind_extra = pkind - gkind
    kind_missing = gkind - pkind
    return {
        "gold": g, "predicted": p,
        "matched": matched, "extras": extras, "missing": missing,
        "kind_matched": kind_matched, "kind_extra": kind_extra, "kind_missing": kind_missing,
    }


def pct(num: int, denom: int) -> str:
    return f"{100 * num / denom:.1f}%" if denom else "n/a"


def expand(c: Counter):
    return [list(x) for x in c.elements()]


def main():
    for p in (REPORT_JSON, REPORT_MD):
        if p.exists():
            raise FileExistsError(f"Scored output exists: {p}; refusing overwrite")

    if sha(REFERENCE.read_bytes()) != EXPECTED_REFERENCE_SHA:
        raise RuntimeError("Frozen human reference checksum changed")
    human = read_json(REFERENCE)
    manifest = read_json(FREEZE_MANIFEST)
    if (
        manifest.get("status")
        not in {
            "source_verified_frozen",
            "frozen",
        }
        or manifest["reference_sha256"]
        != EXPECTED_REFERENCE_SHA
    ):
        raise RuntimeError(
            "Human freeze manifest does not match"
        )
    plan_raw = INPUT_FILE.read_bytes()
    plan = json.loads(plan_raw)
    generation = read_json(GENERATION)
    if generation["status"] != "generation_frozen_unscored":
        raise RuntimeError("Generation was not frozen")
    if generation["prepared_input_sha256"] != sha(plan_raw):
        raise RuntimeError("Generation was produced from another input")
    case_count = len(
        plan["targets"]
    )

    if (
        not case_count
        or len(
            generation["runs"]
        ) != case_count
    ):
        raise RuntimeError(
            "Unexpected target/generation count"
        )

    gold_by_id = {}
    for narrative in human["narratives"]:
        for sentence in narrative["commentary_sentences"]:
            gold_by_id[sentence["sentence_id"]] = {
                "blind_label": narrative["blind_label"],
                "claim_span": sentence["claim_span"],
                "annotations": sentence["annotations"],
            }
    gold_annotation_count = sum(
        len(
            g["annotations"]
        )
        for g in gold_by_id.values()
    )

    if (
        len(gold_by_id)
        != case_count
        or not gold_annotation_count
    ):
        raise RuntimeError(
            "Frozen reference target count mismatch"
        )

    if set(gold_by_id) != {x["target_id"] for x in generation["runs"]}:
        raise RuntimeError("Generated IDs don't match frozen human reference IDs")

    totals = Counter()
    per_narrative = defaultdict(Counter)
    by_kind = defaultdict(Counter)
    cases = []
    for target, run in zip(plan["targets"], generation["runs"]):
        tid = target["target_id"]
        gold = gold_by_id[tid]
        if run["target_id"] != tid or target["claim_span"] != gold["claim_span"] or run["claim_span"] != gold["claim_span"]:
            raise RuntimeError(f"Exact-span or target-order mismatch: {tid}")
        if run["blind_label"] != gold["blind_label"]:
            raise RuntimeError(f"Blind-label mismatch: {tid}")
        protocol_valid = run["parse_status"] == "success" and isinstance(run.get("validated_proposal"), dict)
        proposed = run["validated_proposal"] if protocol_valid else None
        invalid = []
        predicted = []
        if proposed is not None:
            if proposed.get("claim_span") != gold["claim_span"]:
                protocol_valid = False
            else:
                for ann in proposed.get("annotations", []):
                    if valid(ann):
                        predicted.append(ann)
                    else:
                        invalid.append(ann)
        if not protocol_valid:
            predicted = []
            invalid = []
        parts = matched_parts(gold["annotations"], predicted)
        matched_n = sum(parts["matched"].values())
        kind_matched_n = sum(parts["kind_matched"].values())
        extra_n = sum(parts["extras"].values())
        missing_n = sum(parts["missing"].values())
        exact = protocol_valid and not invalid and not extra_n and not missing_n
        status = "exact" if exact else "partial" if matched_n else "miss"
        narrative_id = gold["blind_label"]

        for counter in (totals, per_narrative[narrative_id]):
            counter["cases"] += 1
            counter[status] += 1
            counter["protocol_success"] += int(protocol_valid)
            counter["protocol_failure"] += int(not protocol_valid)
            counter["gold_triples"] += len(gold["annotations"])
            counter["valid_predicted_triples"] += len(predicted)
            counter["matched_triples"] += matched_n
            counter["extra_valid_triples"] += extra_n
            counter["missing_triples"] += missing_n
            counter["invalid_annotations"] += len(invalid)
            counter["kind_matched"] += kind_matched_n
            counter["gold_kinds"] += len(gold["annotations"])
            counter["predicted_kinds"] += len(predicted)

        for annotation in gold["annotations"]:
            by_kind[annotation["claim_kind"]]["gold"] += 1
        for annotation in predicted:
            by_kind[annotation["claim_kind"]]["predicted"] += 1
        for kind, count in parts["kind_matched"].items():
            by_kind[kind]["kind_matches"] += count
        for (kind, _, _), count in parts["matched"].items():
            by_kind[kind]["strict_triple_matches"] += count
        for kind, count in parts["kind_extra"].items():
            by_kind[kind]["extra_kinds"] += count

        cases.append({
            "target_id": tid,
            "blind_label": narrative_id,
            "claim_span": gold["claim_span"],
            "status": status,
            "protocol_valid": protocol_valid,
            "parse_status": run["parse_status"],
            "validation_error": run.get("validation_error"),
            "gold": [
                {"claim_kind": a["claim_kind"], "support_status": a["support_status"],
                 "assertion_strength": a["assertion_strength"], "proposition": a.get("proposition"),
                 "rationale": a.get("rationale")}
                for a in gold["annotations"]
            ],
            "predicted_valid": predicted,
            "invalid_annotations": invalid,
            "strict_matched": expand(parts["matched"]),
            "strict_missing": expand(parts["missing"]),
            "strict_extra": expand(parts["extras"]),
            "matched_kind_occurrences": kind_matched_n,
        })

    aggregate = dict(totals)
    if (
        aggregate[
            "gold_triples"
        ] != gold_annotation_count
        or aggregate[
            "cases"
        ] != case_count
    ):
        raise RuntimeError(
            "Unexpected gold denominators"
        )
    result = {
        "schema_version": "macro_semantic_oos_scored_v0_1",
        "experiment_id": EXPERIMENT,
        "scorer_policy": {
            "strict_triples": "Exact claim_kind/support_status/assertion_strength, multiplicity-aware",
            "claim_kind": "Occurrence recall/precision; independently of support and strength",
            "repeated_triples": "A repeated triple can represent distinct propositions; multiplicity preserved",
            "protocol_failures": "Zero predicted labels for failed proposals",
            "invalid_labels": "Reported separately; excluded from valid predicted labels",
            "human_reference_method": "collaborative_human_assisted_review_with_partial_blinding",
            "classifier_inputs_consumed_mapping_or_telemetry": False,
        },
        "input_sha256": sha(plan_raw),
        "generation_sha256": sha(GENERATION.read_bytes()),
        "human_reference_sha256": sha(REFERENCE.read_bytes()),
        "aggregate": aggregate,
        "by_narrative": {k: dict(v) for k, v in sorted(per_narrative.items())},
        "by_claim_kind": {k: dict(v) for k, v in sorted(by_kind.items())},
        "cases": cases,
    }
    write_json(REPORT_JSON, result)

    lines = [
        "# v1.9.3 Out-of-sample semantic classifier score",
        "",
        "Model: GPT-OSS 20B / reasoning low / frozen `compact_balanced_v0_2_1`.",
        (
            "Human reference: collaborative and partially blinded; "
            f"{case_count} commentary sentences, "
            f"{gold_annotation_count} gold commitments."
        ),
        (
            "Classifier generation used blind narrative IDs and did not "
            "consume model-identity mapping, hidden analysis traces, "
            "or token telemetry."
        ),
        "",
        "## Overall",
        "",
        (
            f"- Case exact/partial/miss: "
            f"{totals['exact']}/"
            f"{totals['partial']}/"
            f"{totals['miss']} "
            f"({case_count} total)"
        ),
        (
            f"- Protocol successes: "
            f"{totals['protocol_success']}/"
            f"{case_count}; "
            f"invalid ontology annotations: "
            f"{totals['invalid_annotations']}"
        ),
        f"- Strict triple recall: {totals['matched_triples']}/{totals['gold_triples']} = {pct(totals['matched_triples'], totals['gold_triples'])}",
        f"- Strict triple precision: {totals['matched_triples']}/{totals['valid_predicted_triples']} = {pct(totals['matched_triples'], totals['valid_predicted_triples'])}",
        f"- Claim-kind occurrence recall: {totals['kind_matched']}/{totals['gold_kinds']} = {pct(totals['kind_matched'], totals['gold_kinds'])}",
        f"- Claim-kind occurrence precision: {totals['kind_matched']}/{totals['predicted_kinds']} = {pct(totals['kind_matched'], totals['predicted_kinds'])}",
        f"- Extra valid / missing strict triples: {totals['extra_valid_triples']} / {totals['missing_triples']}",
        "",
        "## By blind narrative",
        "",
        "| Narrative | Cases | Exact | Partial | Miss | Matched / gold triples | Invalid |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for n, v in sorted(per_narrative.items()):
        lines.append(f"| {n} | {v['cases']} | {v['exact']} | {v['partial']} | {v['miss']} | {v['matched_triples']}/{v['gold_triples']} | {v['invalid_annotations']} |")
    lines += [
        "", "## By claim kind", "",
        "| Claim kind | Gold | Recognized | Strict triple | Extra kind occurrences |",
        "|---|---:|---:|---:|---:|",
    ]
    for kind, v in sorted(by_kind.items()):
        lines.append(f"| {kind} | {v['gold']} | {v['kind_matches']} | {v['strict_triple_matches']} | {v['extra_kinds']} |")
    lines += ["", "## Sentence-level disagreements", ""]
    for c in cases:
        if c["status"] == "exact":
            continue
        lines.append(f"### {c['target_id']} — {c['status']}")
        lines.append("")
        lines.append(f"> {c['claim_span']}")
        lines.append("")
        lines.append("Missing triples:")
        lines.append("- " + ("; ".join(map(str, c["strict_missing"])) if c["strict_missing"] else "none"))
        lines.append("Extra triples:")
        lines.append("- " + ("; ".join(map(str, c["strict_extra"])) if c["strict_extra"] else "none"))
        if c["invalid_annotations"]:
            lines.append(f"Invalid ontology: `{json.dumps(c['invalid_annotations'], ensure_ascii=False)}`")
        if not c["protocol_valid"]:
            lines.append(f"Protocol failure: {c['validation_error']}")
        lines.append("")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("OOS CLASSIFIER SCORING COMPLETE")
    print(
        f"Cases: {case_count} | "
        f"exact {totals['exact']} "
        f"partial {totals['partial']} "
        f"miss {totals['miss']}"
    )

    print(
        f"Protocol: "
        f"{totals['protocol_success']}/"
        f"{case_count} | "
        f"invalid ontology: "
        f"{totals['invalid_annotations']}"
    )
    print(f"Strict triple: {totals['matched_triples']}/{totals['gold_triples']} "
          f"(recall {pct(totals['matched_triples'], totals['gold_triples'])}; "
          f"precision {pct(totals['matched_triples'], totals['valid_predicted_triples'])})")
    print(f"Claim-kind occurrence: {totals['kind_matched']}/{totals['gold_kinds']} "
          f"(recall {pct(totals['kind_matched'], totals['gold_kinds'])})")
    print("By kind: gold, recognized, strict")
    for k, v in sorted(by_kind.items()):
        print(f"  {k}: {v['gold']}, {v['kind_matches']}, {v['strict_triple_matches']}")
    print("Scored JSON:", REPORT_JSON)
    print("Readable report:", REPORT_MD)
    print("Scored JSON SHA-256:", sha(REPORT_JSON.read_bytes()))


if __name__ == "__main__":
    main()
