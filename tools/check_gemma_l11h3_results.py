#!/usr/bin/env python3
"""Validate saved Gemma-3-1B L11H3 outputs without running inference."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results" / "gemma-l11h3-mechanism"
MODES = ("clean", "target_word_mask", "nearby_word_mask", "control_head_word_mask", "head_zero_decode")
EXPECTED = {"clean": 12, "target_word_mask": 5, "nearby_word_mask": 12,
            "control_head_word_mask": 12, "head_zero_decode": 4}


def records(name):
    with (ROOT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def yes(value):
    assert value in ("True", "False"), repr(value)
    return value == "True"


def main():
    manifest = records("prompt_manifest.csv")
    outputs = records("condition_outputs.csv")
    paired = records("paired_outcomes.csv")
    attention = records("attention_by_prompt.csv")
    summary = records("summary.csv")
    att_summary = records("attention_summary.csv")
    metadata = json.loads((ROOT / "run_metadata.json").read_text(encoding="utf-8"))
    assert len(manifest) == len(attention) == 96
    assert len(outputs) == 480 and len(paired) == 384
    assert len(summary) == 25 and len(att_summary) == 5
    prompts = {row["prompt_id"]: row for row in manifest}
    assert len(prompts) == 96
    assert Counter(row["language"] for row in manifest) == {"de": 24, "es": 24, "fr": 24, "it": 24}
    assert metadata["resolved_revision"] == "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
    assert metadata["layer"] == 11 and metadata["head"] == 3 and metadata["control_head"] == 0
    groups = defaultdict(dict)
    for row in outputs:
        pid, mode = row["prompt_id"], row["condition"]
        assert pid in prompts and mode in MODES and mode not in groups[pid]
        assert row["language"] == prompts[pid]["language"]
        groups[pid][mode] = row
    assert all(set(d) == set(MODES) for d in groups.values())
    for mode in MODES:
        curr = [g[mode] for g in groups.values()]
        scorable = [r for r in curr if not yes(r["skipped"])]
        assert len(scorable) == 95
        assert sum(yes(r["passed"]) for r in scorable) == EXPECTED[mode]
        agg = next(r for r in summary if r["language"] == "ALL" and r["condition"] == mode)
        assert int(agg["n"]) == 96 and int(agg["scorable"]) == 95
        assert int(agg["passes"]) == EXPECTED[mode]
    new_failures = Counter()
    new_passes = Counter()
    for pid, d in groups.items():
        first_tokens = {r["first_token_id"] for r in d.values()}
        assert len(first_tokens) == 1, pid
        for mode in MODES[1:]:
            if yes(d["clean"]["passed"]) and not yes(d[mode]["passed"]):
                new_failures[mode] += 1
            if not yes(d["clean"]["passed"]) and yes(d[mode]["passed"]):
                new_passes[mode] += 1
    assert new_failures == {"target_word_mask": 7, "head_zero_decode": 8}
    assert not new_passes
    assert all(yes(p["first_token_matches_clean"]) for p in paired)
    a = next(r for r in att_summary if r["language"] == "ALL")
    assert abs(float(a["last_prompt_h3_target_mass_mean"]) - 0.13557) < 1e-7
    assert abs(float(a["last_prompt_h3_nearby_mass_mean"]) - 0.00858) < 1e-7
    assert {r["prompt_id"] for r in attention} == set(prompts)
    print("PASS: 96 prompts, 480 outputs, 384 paired records; totals, attention means, and first tokens agree.")
    print("Passes / 95 scorable:", EXPECTED)
    print("New baseline-pass failures:", dict(new_failures))


if __name__ == "__main__":
    main()
