#!/usr/bin/env python3
"""Verify agreed blind human macro reference against frozen source and seal it.

Run from the ai-lab repository. Does not access model mappings, analysis, or telemetry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

EXPERIMENT = 'v1_9_3_sep2026_multi_model_v0_1'
REFERENCE_NAME = 'human_macro_semantic_reference_v0_1.json'
MANIFEST_NAME = 'human_reference_freeze_manifest_v0_1.json'
ALLOWED_KINDS = {
    'empirical_comparison', 'unit_conversion', 'temporal_pattern',
    'scope_generalization', 'magnitude_judgment', 'causal_explanation',
    'market_interpretation', 'policy_interpretation',
}
ALLOWED_SUPPORT = {'supported', 'contradicted', 'insufficient_evidence'}
ALLOWED_STRENGTH = {'asserted', 'qualified_inference', 'qualified_possibility'}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def normalize_for_comparison(text: str) -> str:
    """Ignore harmless hyphen/whitespace rendering, not substantive wording."""
    table = str.maketrans({'‑': '-', '–': '-', '—': '-', '−': '-', '‐': '-', '\u00a0': ' '})
    return ' '.join(unicodedata.normalize('NFKC', text).translate(table).split())


def extract(packet: str, label: str) -> tuple[list[str], int]:
    match = re.search(
        rf'(?ms)^## {re.escape(label)}[ \t]*\n(.*?)(?=^---[ \t]*$|\Z)',
        packet,
    )
    if match is None:
        raise ValueError(f'Missing blind section {label}')
    section = match.group(1)
    prefix, sep, commentary = section.partition('## Commentary')
    if not sep:
        raise ValueError(f'Missing commentary heading for {label}')
    bullet_count = len(re.findall(r'(?m)^[-*][ \t]+', prefix))
    commentary = ' '.join(commentary.split())
    sentences = re.split(r'(?<=[.!?])[ \t]+', commentary)
    return sentences, bullet_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repo_root.resolve()
    out = (root / 'benchmarks/fixtures/macro_narrative_experiments'
           / EXPERIMENT / 'human_review_v0_1')
    ref_path = out / REFERENCE_NAME
    ref = json.loads(ref_path.read_text(encoding='utf-8'))
    assert ref['experiment_id'] == EXPERIMENT
    assert ref['schema_version'] == 'human_macro_semantic_reference_v0_1'

    packet_rel = ref['frozen_source']['blind_review_packet_path']
    packet_path = root / packet_rel
    packet_bytes = packet_path.read_bytes()
    packet_hash = sha(packet_bytes)
    if packet_hash != ref['frozen_source']['blind_review_packet_sha256']:
        raise ValueError(f'Blind packet SHA mismatch: {packet_hash}')
    packet = packet_bytes.decode('utf-8')
    claims_path = (root / 'benchmarks/fixtures/macro_narrative_experiments'
                   / EXPERIMENT / 'inputs/selected_fred_claims.json')
    claims_hash = sha(claims_path.read_bytes())
    if claims_hash != ref['frozen_source']['selected_fred_claims_sha256']:
        raise ValueError(f'Claims SHA mismatch: {claims_hash}')
    prompt_path = (root / 'benchmarks/fixtures/macro_narrative_experiments'
                   / EXPERIMENT / 'inputs/narrative_prompt.txt')
    prompt_hash = sha(prompt_path.read_bytes())
    if prompt_hash != ref['frozen_source']['narrative_prompt_sha256']:
        raise ValueError(f'Prompt SHA mismatch: {prompt_hash}')

    count_sent = count_anns = 0
    ids = set()
    for row in ref['narratives']:
        label = row['blind_label']
        actual_sentences, bullet_count = extract(packet, label)
        expected = row['commentary_sentences']
        if bullet_count != 4:
            raise ValueError(f'{label}: {bullet_count} summary bullets (expected 4)')
        if len(actual_sentences) != len(expected):
            raise ValueError(f'{label}: {len(actual_sentences)} source sentences vs {len(expected)} references')
        for i, (actual, reviewed) in enumerate(zip(actual_sentences, expected), 1):
            sentence_id = f'{label[-1]}_S{i:02d}'
            if reviewed['sentence_id'] != sentence_id:
                raise ValueError(f'{label}: unexpected sentence ID {reviewed["sentence_id"]}')
            if normalize_for_comparison(actual) != normalize_for_comparison(reviewed['claim_span']):
                raise ValueError(
                    f'{sentence_id} exact-span mismatch:\n'
                    f'SOURCE: {actual!r}\nREFERENCE: {reviewed["claim_span"]!r}'
                )
            if not reviewed['annotations']:
                raise ValueError(f'{sentence_id}: no annotations')
            count_sent += 1
            for j, a in enumerate(reviewed['annotations'], 1):
                ann_id = f'{sentence_id}_A{j:02d}'
                if a.get('annotation_id') != ann_id or ann_id in ids:
                    raise ValueError(f'{sentence_id}: duplicate/incorrect annotation ID {a.get("annotation_id")}')
                ids.add(ann_id)
                if (a.get('claim_kind') not in ALLOWED_KINDS or
                    a.get('support_status') not in ALLOWED_SUPPORT or
                    a.get('assertion_strength') not in ALLOWED_STRENGTH):
                    raise ValueError(f'{ann_id}: invalid ontology field(s): {a}')
                if not a.get('rationale') or not a.get('proposition'):
                    raise ValueError(f'{ann_id}: missing proposition/rationale')
                count_anns += 1

    if (count_sent, count_anns) != (12, 35):
        raise ValueError(f'Unexpected counts: {count_sent} sentences, {count_anns} annotations')
    if (ref['totals']['commentary_sentences'], ref['totals']['semantic_annotations']) != (12, 35):
        raise ValueError('Reference declared totals do not match')
    if (out / MANIFEST_NAME).exists():
        raise FileExistsError('Already frozen: refusing to overwrite manifest')

    review_md = out / 'human_macro_semantic_reference_v0_1.md'
    manifest = {
        'schema_version': 'human_macro_semantic_reference_freeze_manifest_v0_1',
        'status': 'source_verified_frozen',
        'experiment_id': EXPERIMENT,
        'source_packet_path': packet_rel,
        'source_packet_sha256': packet_hash,
        'selected_fred_claims_sha256': claims_hash,
        'narrative_prompt_sha256': prompt_hash,
        'reference_file': REFERENCE_NAME,
        'reference_sha256': sha(ref_path.read_bytes()),
        'readable_review_file': review_md.name,
        'readable_review_sha256': sha(review_md.read_bytes()),
        'narratives': 3,
        'commentary_sentences': count_sent,
        'semantic_annotations': count_anns,
        'blind_mapping_accessed': False,
        'model_analysis_or_telemetry_accessed': False,
        'review_method': 'collaborative_human_assisted_review_with_partial_blinding',
        'limitations': ref['review_provenance']['limitations'],
    }
    manifest_path = out / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (out / 'human_macro_semantic_reference_v0_1.sha256').write_text(
        manifest['reference_sha256'] + '  ' + REFERENCE_NAME + '\n', encoding='utf-8'
    )
    print('HUMAN REFERENCE SOURCE-VERIFIED AND FROZEN')
    print('Experiment:', EXPERIMENT)
    print('Narratives:', 3, '| commentary sentences:', count_sent, '| annotations:', count_anns)
    print('Reference SHA-256:', manifest['reference_sha256'])
    print('Source packet SHA-256:', packet_hash)
    print('Manifest:', manifest_path.relative_to(root))
    print('No mapping, analysis trace, or token telemetry was accessed.')


if __name__ == '__main__':
    main()
