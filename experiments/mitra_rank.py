"""Rank heads from MITra's published language-head scores, with the rule of their src/representation/top_head.py:
average each direction over examples, take its top head, and rank heads by how many directions they top. Also reports
our head's rank by the score averaged over directions.

The scores are results/translation_task/logprobs_diff_lang/<model>:<source>:<target>.pt in github.com/Blyzi/mitra
(commit daa721b), for Qwen3-1.7B-Base, Llama-3.2-1B and gemma-3-1b-pt, English to and from fra, spa, por, jpn, zho,
swh, wol, hin, arb and rus. They are not copied here; download them into one directory and pass it:
python mitra_rank.py <dir>"""
import sys
from pathlib import Path

import torch

D = Path(sys.argv[1])
OURS = {"Qwen3-1.7B-Base": (18, 12), "Llama-3.2-1B": (8, 25), "gemma-3-1b-pt": (11, 3)}

for model, ours in OURS.items():
    files = sorted(D.glob(f"{model}:*.pt"))
    means, n_ex = {}, {}
    for f in files:
        t = torch.load(f, map_location="cpu", weights_only=True)
        assert isinstance(t, torch.Tensor) and t.dim() == 3, (f.name, type(t))
        _, src, tgt = f.stem.split(":")
        means[(src, tgt)] = t.float().mean(-1)
        n_ex[(src, tgt)] = t.shape[-1]
    L, H = next(iter(means.values())).shape
    tops = {}
    for k, m in means.items():
        i = m.argmax().item()
        tops.setdefault((i // H, i % H), []).append(k)
    ranked = sorted(tops.items(), key=lambda x: -len(x[1]))
    print(f"== {model}: {L} layers x {H} heads, {len(files)} directions, examples per direction "
          f"{min(n_ex.values())}-{max(n_ex.values())}")
    print("  top head per direction, counted (their rule):")
    for (l, h), ks in ranked[:5]:
        en_x = sum(s == "eng_Latn" for s, _ in ks)
        print(f"    L{l}H{h}: {len(ks)} of {len(files)} directions ({en_x} en->X)")
    print(f"  ours L{ours[0]}H{ours[1]}: top in {len(tops.get(ours, []))} directions")
    for name, keys in (("all directions", list(means)), ("en->X", [k for k in means if k[0] == "eng_Latn"]),
                       ("X->en", [k for k in means if k[1] == "eng_Latn"])):
        avg = torch.stack([means[k] for k in keys]).mean(0)
        order = avg.flatten().argsort(descending=True).tolist()
        rank = order.index(ours[0] * H + ours[1]) + 1
        best = order[0]
        print(f"  mean over {name} ({len(keys)}): ours rank {rank} of {L * H}, score {avg[ours].item():+.3f}; "
              f"top L{best // H}H{best % H} {avg.flatten()[best].item():+.3f}")
    en_fr = means.get(("eng_Latn", "fra_Latn"))
    if en_fr is not None:
        order = en_fr.flatten().argsort(descending=True).tolist()
        per_ex = torch.load(D / f"{model}:eng_Latn:fra_Latn.pt", map_location="cpu", weights_only=True)
        tops_ex = per_ex.float().flatten(0, 1).argmax(0).bincount(minlength=L * H)
        top_ex = ", ".join(f"L{i // H}H{i % H} {tops_ex[i].item()}" for i in tops_ex.argsort(descending=True)[:3]
                           if tops_ex[i] > 0)
        print(f"  en->fr only: ours rank {order.index(ours[0] * H + ours[1]) + 1}, "
              f"top L{order[0] // H}H{order[0] % H}, examples {n_ex[('eng_Latn', 'fra_Latn')]}; "
              f"top head per example: {top_ex}")
