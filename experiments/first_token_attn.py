import csv
import json
import sys

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


@torch.no_grad()
def main(prompts="out/gpt2/prompts.csv", out="out/gpt2/first_token_attn.json", bs=50):
    torch.set_num_threads(4)
    rows = list(csv.DictReader(open(prompts, encoding="utf-8")))
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
    last, allpos, n = torch.zeros(12, 12), torch.zeros(12, 12), 0
    for s in range(0, len(rows), bs):
        b = tok([r["prompt"] for r in rows[s:s + bs]], return_tensors="pt", padding=True)
        att = model(**b, output_attentions=True).attentions
        lens = b["attention_mask"].sum(1)
        for i, length in enumerate(lens.tolist()):
            for l, a in enumerate(att):
                last[l] += a[i, :, length - 1, 0]
                allpos[l] += a[i, :, 1:length, 0].mean(-1) if length > 1 else a[i, :, 0, 0]
            n += 1
    json.dump({"last": (last / n).tolist(), "all": (allpos / n).tolist(), "n": n}, open(out, "w"))
    print("done", n)


if __name__ == "__main__":
    main(*sys.argv[1:])
