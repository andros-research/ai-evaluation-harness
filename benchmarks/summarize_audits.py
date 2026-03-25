from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


NARRATIVES_DIR = Path("benchmarks/results/narratives")
AGGREGATED_DIR = Path("benchmarks/results/aggregated")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stringify_list(values: list[Any]) -> str:
    return "|".join(str(v) for v in values)


def safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def list_set(values: list[Any]) -> set[str]:
    return {str(v) for v in values if str(v).strip()}


def compute_claim_overlap(
    claim_ids: list[Any], matched_claim_ids: list[Any]
) -> dict[str, Any]:
    cited = list_set(claim_ids)
    matched = list_set(matched_claim_ids)

    overlap = cited & matched
    extra_matched = matched - cited
    unused_cited = cited - matched

    overlap_ratio = 0.0
    if cited:
        overlap_ratio = len(overlap) / len(cited)

    return {
        "claim_id_overlap_count": len(overlap),
        "claim_id_overlap_ratio": round(overlap_ratio, 4),
        "extra_matched_claim_ids": sorted(extra_matched),
        "unused_cited_claim_ids": sorted(unused_cited),
        "n_extra_matched_claim_ids": len(extra_matched),
        "n_unused_cited_claim_ids": len(unused_cited),
    }


def infer_model_from_linked_claims(linked_claims: list[dict[str, Any]]) -> str:
    models = sorted(
        {str(c.get("model")) for c in linked_claims if c.get("model") is not None}
    )
    if len(models) == 1:
        return models[0]
    if len(models) > 1:
        return "MULTI"
    return ""


def infer_prompt_from_linked_claims(linked_claims: list[dict[str, Any]]) -> str:
    prompts = sorted(
        {str(c.get("prompt_id")) for c in linked_claims if c.get("prompt_id") is not None}
    )
    if len(prompts) == 1:
        return prompts[0]
    if len(prompts) > 1:
        return "MULTI"
    return ""


def extract_artifact_paths(narratives_dir: Path) -> list[dict[str, Path]]:
    """
    Find parsed narrative artifacts and pair them with sibling audit JSONs.
    """
    artifact_pairs: list[dict[str, Path]] = []

    for parsed_path in sorted(narratives_dir.rglob("*__parsed_narrative.json")):
        audit_path = parsed_path.with_name(
            parsed_path.name.replace("__parsed_narrative.json", "__audit.json")
        )
        if not audit_path.exists():
            continue

        artifact_pairs.append(
            {
                "parsed_path": parsed_path,
                "audit_path": audit_path,
            }
        )

    return artifact_pairs


def row_from_items(
    *,
    parsed_payload: dict[str, Any],
    audit_payload: dict[str, Any],
    parsed_item: dict[str, Any],
    audit_item: dict[str, Any],
    parsed_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    linked_claims = safe_list(parsed_item.get("linked_claims"))
    claim_ids = safe_list(parsed_item.get("claim_ids"))
    unknown_claim_ids = safe_list(parsed_item.get("unknown_claim_ids"))
    matched_claim_ids = safe_list(audit_item.get("matched_claim_ids"))

    overlap_info = compute_claim_overlap(claim_ids, matched_claim_ids)

    summary = (
        audit_payload.get("summary", {})
        if isinstance(audit_payload.get("summary"), dict)
        else {}
    )

    model = infer_model_from_linked_claims(linked_claims)
    prompt_id = infer_prompt_from_linked_claims(linked_claims)

    supported = audit_item.get("supported")
    issue_type = str(audit_item.get("issue_type", "")).strip()

    if issue_type == "meta_caution":
        audit_status = "meta_caution"
    elif supported is True:
        audit_status = "supported"
    elif supported is False:
        audit_status = "flagged"
    else:
        audit_status = "unknown"

    row = {
        # artifact identity
        "artifact_name": parsed_path.name.replace("__parsed_narrative.json", ""),
        "parsed_path": str(parsed_path),
        "audit_path": str(audit_path),

        # artifact metadata
        "suite_name": parsed_payload.get("suite_name", ""),
        "metric": parsed_payload.get("metric", ""),
        "baseline_experiment": parsed_payload.get("baseline_experiment", ""),
        "comparison_experiments": stringify_list(
            safe_list(parsed_payload.get("comparison_experiments"))
        ),

        # inferred item-level metadata
        "model": model,
        "prompt_id": prompt_id,

        # item identity
        "item_id": parsed_item.get("item_id", ""),
        "section": parsed_item.get("section", ""),
        "bullet_text": audit_item.get("text", ""),
        "clean_text": parsed_item.get("clean_text", ""),
        "raw_text": parsed_item.get("raw_text", ""),

        # item support / traceability
        "audit_status": audit_status,
        "supported": "" if supported is None else supported,
        "issue_type": issue_type,
        "notes": audit_item.get("notes", ""),

        "claim_ids": stringify_list(claim_ids),
        "matched_claim_ids": stringify_list(matched_claim_ids),
        "unknown_claim_ids": stringify_list(unknown_claim_ids),

        "n_claim_ids": len(claim_ids),
        "n_matched_claim_ids": len(matched_claim_ids),
        "n_unknown_claim_ids": len(unknown_claim_ids),
        "missing_claim_refs": len(claim_ids) == 0,

        # overlap / support discipline diagnostics
        "claim_id_overlap_count": overlap_info["claim_id_overlap_count"],
        "claim_id_overlap_ratio": overlap_info["claim_id_overlap_ratio"],
        "extra_matched_claim_ids": stringify_list(
            overlap_info["extra_matched_claim_ids"]
        ),
        "unused_cited_claim_ids": stringify_list(
            overlap_info["unused_cited_claim_ids"]
        ),
        "n_extra_matched_claim_ids": overlap_info["n_extra_matched_claim_ids"],
        "n_unused_cited_claim_ids": overlap_info["n_unused_cited_claim_ids"],

        # artifact summary fields copied down for easy grouping
        "fidelity_score_artifact": audit_payload.get("fidelity_score", ""),
        "n_items_artifact": summary.get("n_items", ""),
        "n_scored_items_artifact": summary.get("n_scored_items", ""),
        "n_meta_cautions_artifact": summary.get("n_meta_cautions", ""),
        "n_supported_artifact": summary.get("n_supported", ""),
        "n_flagged_artifact": summary.get("n_flagged", ""),
    }
    return row


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_bullets = len(rows)

    status_counts = Counter(row["audit_status"] for row in rows)

    issue_type_counts = Counter(
        row["issue_type"]
        for row in rows
        if row["audit_status"] == "flagged"
        and row["issue_type"]
        and row["issue_type"] != "meta_caution"
    )

    artifacts = {}
    fidelity_by_suite_accum: dict[str, list[float]] = defaultdict(list)
    fidelity_by_model_accum: dict[str, list[float]] = defaultdict(list)
    flagged_by_suite = Counter()
    flagged_by_model = Counter()
    flagged_by_prompt = Counter()
    
    overlap_ratios = []
    total_extra_matched = 0
    total_unused_cited = 0
    bullets_with_extra_matched = 0
    bullets_with_unused_cited = 0

    for row in rows:
        artifact_name = str(row["artifact_name"])
        if artifact_name not in artifacts:
            fidelity = row["fidelity_score_artifact"]
            try:
                fidelity_value = float(fidelity)
            except (TypeError, ValueError):
                fidelity_value = None

            artifacts[artifact_name] = {
                "suite_name": row["suite_name"],
                "metric": row["metric"],
                "baseline_experiment": row["baseline_experiment"],
                "comparison_experiments": row["comparison_experiments"],
                "fidelity_score": fidelity_value,
            }

            if fidelity_value is not None:
                suite_name = str(row["suite_name"])
                if suite_name:
                    fidelity_by_suite_accum[suite_name].append(fidelity_value)

        if row["audit_status"] == "flagged":
            if row["suite_name"]:
                flagged_by_suite[str(row["suite_name"])] += 1
            if row["model"]:
                flagged_by_model[str(row["model"])] += 1
            if row["prompt_id"]:
                flagged_by_prompt[str(row["prompt_id"])] += 1

        if row["model"]:
            artifact_name = str(row["artifact_name"])
            fidelity = artifacts.get(artifact_name, {}).get("fidelity_score")
            if isinstance(fidelity, float):
                fidelity_by_model_accum[str(row["model"])].append(fidelity)

        try:
            overlap_ratios.append(float(row["claim_id_overlap_ratio"]))
        except (TypeError, ValueError):
            pass

        total_extra_matched += int(row.get("n_extra_matched_claim_ids", 0) or 0)
        total_unused_cited += int(row.get("n_unused_cited_claim_ids", 0) or 0)

        if int(row.get("n_extra_matched_claim_ids", 0) or 0) > 0:
            bullets_with_extra_matched += 1
        if int(row.get("n_unused_cited_claim_ids", 0) or 0) > 0:
            bullets_with_unused_cited += 1

    fidelity_by_suite = {
        k: round(sum(v) / len(v), 4) for k, v in sorted(fidelity_by_suite_accum.items()) if v
    }
    fidelity_by_model = {
        k: round(sum(v) / len(v), 4) for k, v in sorted(fidelity_by_model_accum.items()) if v
    }

    top_flagged_bullets = []
    for row in rows:
        if row["audit_status"] != "flagged":
            continue
        top_flagged_bullets.append(
            {
                "artifact_name": row["artifact_name"],
                "suite_name": row["suite_name"],
                "model": row["model"],
                "prompt_id": row["prompt_id"],
                "section": row["section"],
                "bullet_text": row["bullet_text"],
                "issue_type": row["issue_type"],
                "notes": row["notes"],
                "claim_ids": row["claim_ids"].split("|") if row["claim_ids"] else [],
                "matched_claim_ids": row["matched_claim_ids"].split("|") if row["matched_claim_ids"] else [],
            }
        )

    top_flagged_bullets = top_flagged_bullets[:25]
    
    
    top_extra_matched_bullets = []
    for row in sorted(
        rows,
        key=lambda r: int(r.get("n_extra_matched_claim_ids", 0) or 0),
        reverse=True,
    ):
        if int(row.get("n_extra_matched_claim_ids", 0) or 0) <= 0:
            continue
        top_extra_matched_bullets.append(
            {
                "artifact_name": row["artifact_name"],
                "suite_name": row["suite_name"],
                "model": row["model"],
                "prompt_id": row["prompt_id"],
                "section": row["section"],
                "bullet_text": row["bullet_text"],
                "claim_ids": row["claim_ids"].split("|") if row["claim_ids"] else [],
                "matched_claim_ids": row["matched_claim_ids"].split("|") if row["matched_claim_ids"] else [],
                "extra_matched_claim_ids": row["extra_matched_claim_ids"].split("|")
                if row["extra_matched_claim_ids"]
                else [],
                "claim_id_overlap_ratio": row["claim_id_overlap_ratio"],
            }
        )

    top_extra_matched_bullets = top_extra_matched_bullets[:25]

    return {
        "support_mode": "heuristic_matched_claims",
        "total_artifacts": len(artifacts),
        "total_bullets": total_bullets,
        "status_counts": dict(status_counts),
        "issue_type_counts": dict(issue_type_counts),
        "fidelity_by_suite": fidelity_by_suite,
        "fidelity_by_model": fidelity_by_model,
        "flagged_count_by_suite": dict(flagged_by_suite),
        "flagged_count_by_model": dict(flagged_by_model),
        "flagged_count_by_prompt": dict(flagged_by_prompt),

        "support_diagnostics": {
            "mean_claim_id_overlap_ratio": round(
                sum(overlap_ratios) / len(overlap_ratios), 4
            ) if overlap_ratios else 0.0,
            "total_extra_matched_claim_ids": total_extra_matched,
            "total_unused_cited_claim_ids": total_unused_cited,
            "bullets_with_extra_matched_claims": bullets_with_extra_matched,
            "bullets_with_unused_cited_claims": bullets_with_unused_cited,
        },

        "top_flagged_bullets": top_flagged_bullets,
        "top_extra_matched_bullets": top_extra_matched_bullets,
    }


def build_rows(narratives_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    artifact_pairs = extract_artifact_paths(narratives_dir)

    for pair in artifact_pairs:
        parsed_path = pair["parsed_path"]
        audit_path = pair["audit_path"]

        parsed_payload = load_json(parsed_path)
        audit_payload = load_json(audit_path)

        parsed_items = safe_list(parsed_payload.get("items"))
        audit_items = safe_list(audit_payload.get("audit_items"))

        if len(parsed_items) != len(audit_items):
            print(
                f"WARNING: item count mismatch for {parsed_path.name} "
                f"(parsed={len(parsed_items)}, audit={len(audit_items)}). "
                "Will align by position up to min length."
            )

        n = min(len(parsed_items), len(audit_items))
        for i in range(n):
            row = row_from_items(
                parsed_payload=parsed_payload,
                audit_payload=audit_payload,
                parsed_item=parsed_items[i],
                audit_item=audit_items[i],
                parsed_path=parsed_path,
                audit_path=audit_path,
            )
            rows.append(row)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate parsed narrative + audit artifacts into bullet-level audit tables."
    )
    parser.add_argument(
        "--narratives-dir",
        default=str(NARRATIVES_DIR),
        help="Directory containing narrative artifacts.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(AGGREGATED_DIR / "audit_items.csv"),
        help="Output CSV path for bullet-level audit rows.",
    )
    parser.add_argument(
        "--output-json",
        default=str(AGGREGATED_DIR / "audit_summary.json"),
        help="Output JSON path for summary metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    narratives_dir = Path(args.narratives_dir).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_json = Path(args.output_json).resolve()

    rows = build_rows(narratives_dir)
    summary = summarize_rows(rows)

    save_csv(rows, output_csv)
    save_json(summary, output_json)

    print(f"Scanned narratives dir: {narratives_dir}")
    print(f"Saved audit items CSV: {output_csv}")
    print(f"Saved audit summary JSON: {output_json}")
    print(
        "Artifacts={total_artifacts} | Bullets={total_bullets} | "
        "Supported={supported} | Flagged={flagged} | Meta={meta}".format(
            total_artifacts=summary.get("total_artifacts", 0),
            total_bullets=summary.get("total_bullets", 0),
            supported=summary.get("status_counts", {}).get("supported", 0),
            flagged=summary.get("status_counts", {}).get("flagged", 0),
            meta=summary.get("status_counts", {}).get("meta_caution", 0),
        )
    )


if __name__ == "__main__":
    main()