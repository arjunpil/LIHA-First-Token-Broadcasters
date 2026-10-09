"""Verify the archived Qwen2.5-1.5B-Instruct L22H6 monolingual phase run.

Run from the repository root:
    python tools/check_l22h6_monolingual.py

Reads only the frozen CSVs. No Hugging Face downloads or GPU required.
"""

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results' / 'qwen-l22h6-mechanism'
CONDITIONS = ('clean', 'L22H6_prefill', 'L22H6_decode', 'L22H6_all', 'L22H8_control')
LANGUAGES = ('it', 'fr', 'de', 'es')


def as_bool(s):
    if s not in ('True', 'False'):
        raise ValueError(f'Unexpected boolean string: {s!r}')
    return s == 'True'


def exact_two_sided_binomial(a, b):
    """Exact two-sided binomial p for matched discordant pairs, null p=0.5."""
    n = a + b
    if not n:
        return 1.0
    lower = sum(math.comb(n, k) for k in range(min(a, b) + 1))
    return min(1.0, 2.0 * lower / 2**n)


def main():
    with (BASE / 'monolingual_prompt_manifest.csv').open(newline='', encoding='utf-8') as f:
        manifest = {int(row['prompt_id']): row for row in csv.DictReader(f)}
    with (BASE / 'monolingual_phase_outputs.csv').open(newline='', encoding='utf-8') as f:
        outputs = list(csv.DictReader(f))
    with (BASE / 'monolingual_phase_summary.csv').open(newline='', encoding='utf-8') as f:
        archived_summary = {(r['language'], r['condition']):r for r in csv.DictReader(f)}

    assert len(manifest) == 96, len(manifest)
    assert len(outputs) == 480, len(outputs)
    assert len(archived_summary) == 20, len(archived_summary)
    assert Counter(row['language'] for row in manifest.values()) == Counter({l:24 for l in LANGUAGES})
    assert len({row['prompt'] for row in manifest.values()}) == 96

    rows = defaultdict(dict)
    for item in outputs:
        pid = int(item['prompt_id'])
        assert pid in manifest
        orig = manifest[pid]
        assert item['language'] == orig['language']
        assert item['source'] == orig['source']
        assert item['prompt'] == orig['prompt']
        cond = item['condition']
        assert cond in CONDITIONS and cond not in rows[pid]
        assert item['task'] == 'monolingual'
        assert int(item['generated_tokens']) >= 1
        if cond == 'clean':
            assert int(item['edited_calls']) == 0
        elif cond == 'L22H6_prefill':
            assert int(item['prefill_calls']) == int(item['edited_calls']) == 1
        elif cond == 'L22H6_decode':
            assert int(item['edited_calls']) == int(item['decode_calls']) > 0
        elif cond == 'L22H6_all' or cond == 'L22H8_control':
            assert int(item['edited_calls']) == int(item['prefill_calls']) + int(item['decode_calls']) > 0
        assert item['passed'] in ('True','False')
        assert item['skipped'] in ('True','False')
        rows[pid][cond] = item
    assert len(rows) == 96
    assert all(set(x) == set(CONDITIONS) for x in rows.values())
    assert all(as_bool(manifest[i]['upstream_baseline_pass']) for i in manifest)
    assert not any(as_bool(manifest[i]['upstream_baseline_skipped']) for i in manifest)
    assert all(rows[i]['clean']['first_token_id'] == rows[i]['L22H6_decode']['first_token_id'] for i in rows)
    assert not any(as_bool(r['skipped']) for r in outputs)

    print('Records: 480, prompts: 96, conditions: 5; no missing/duplicate entries')
    print('Every prompt baseline-correct in saved upstream labels; every output scorable')
    print('Decode-only first generated token matches clean on 96/96 prompts')
    counts = Counter((r['language'], r['source']) for r in manifest.values())
    print('Source distribution:', ', '.join(f'{l}/{s}={n}' for (l,s),n in sorted(counts.items())))
    print('Language   clean  control  prefill  decode  full   paired_exact_p')
    for lang in (*LANGUAGES, 'ALL'):
        subset = [r for pid, r in rows.items() if lang == 'ALL' or manifest[pid]['language'] == lang]
        tally = {cond:sum(as_bool(r[cond]['passed']) for r in subset) for cond in CONDITIONS}
        pre_only = sum(as_bool(r['L22H6_prefill']['passed']) and not as_bool(r['L22H6_decode']['passed']) for r in subset)
        dec_only = sum(as_bool(r['L22H6_decode']['passed']) and not as_bool(r['L22H6_prefill']['passed']) for r in subset)
        if lang != 'ALL':
            for cond in CONDITIONS:
                sr = archived_summary[lang, cond]
                assert int(sr['passes']) == tally[cond]
                assert int(sr['n']) == 24 and int(sr['unscorable']) == 0
                assert math.isclose(float(sr['pass_rate_scorable']), tally[cond]/24)
        print(f'{lang:>8} {tally["clean"]:>7} {tally["L22H8_control"]:>8} {tally["L22H6_prefill"]:>8} {tally["L22H6_decode"]:>7} {tally["L22H6_all"]:>5}   {exact_two_sided_binomial(pre_only,dec_only):.10g}')
    print('Saved summaries and paired comparisons verified.')

if __name__ == '__main__':
    main()
