"""Run a LIHA experiment script with extra Qwen2.5 sizes and optional multi-GPU loading.

The team's scripts stay unchanged; this launcher only patches them at run time:
  1. adds Qwen3-8B/14B and Qwen2.5-7B/14B/32B/72B (base and instruct) to sweep.MODELS
  2. when more than one GPU is visible (or BIG_DEVICE_MAP=1), loads the model with
     device_map="auto" and turns .to()/.cuda() on the dispatched model into no-ops
  3. makes sweep.nll move the targets to the logits' device (needed when the model spans GPUs)

Usage (from the repo root):
  python big/run_big.py sweep    --model qwen2.5-32b-instruct --modes head ...
  python big/run_big.py lcb      --model qwen2.5-32b-instruct --heads L22H6 ...
  python big/run_big.py followup run --model qwen2.5-32b-instruct ...
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "prompts"), str(ROOT)]
os.chdir(ROOT)

import torch  # noqa: E402
import transformers  # noqa: E402

import sweep  # noqa: E402

sweep.MODELS.update({
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B", "float32"),  # small keys for a local smoke test
    "qwen2.5-1.5b-instruct": ("Qwen/Qwen2.5-1.5B-Instruct", "float32"),
    # Qwen3 = last non-hybrid generation; fp32 like the team's qwen3-1.7b / 4b runs (8B ~33GB, 14B ~59GB)
    "qwen3-8b": ("Qwen/Qwen3-8B-Base", "float32"),
    "qwen3-8b-instruct": ("Qwen/Qwen3-8B", "float32"),
    "qwen3-14b": ("Qwen/Qwen3-14B-Base", "float32"),
    "qwen3-14b-instruct": ("Qwen/Qwen3-14B", "float32"),
    "qwen2.5-7b": ("Qwen/Qwen2.5-7B", "bfloat16"),
    "qwen2.5-7b-instruct": ("Qwen/Qwen2.5-7B-Instruct", "bfloat16"),
    "qwen2.5-14b": ("Qwen/Qwen2.5-14B", "bfloat16"),
    "qwen2.5-14b-instruct": ("Qwen/Qwen2.5-14B-Instruct", "bfloat16"),
    "qwen2.5-32b": ("Qwen/Qwen2.5-32B", "bfloat16"),
    "qwen2.5-32b-instruct": ("Qwen/Qwen2.5-32B-Instruct", "bfloat16"),
    "qwen2.5-72b": ("Qwen/Qwen2.5-72B", "bfloat16"),
    "qwen2.5-72b-instruct": ("Qwen/Qwen2.5-72B-Instruct", "bfloat16"),
})

MULTI = torch.cuda.device_count() > 1 or os.environ.get("BIG_DEVICE_MAP") == "1"

if MULTI:
    _auto = transformers.AutoModelForCausalLM
    _orig_fp = _auto.from_pretrained

    def _from_pretrained(*args, **kwargs):
        kwargs.setdefault("device_map", "auto")
        return _orig_fp(*args, **kwargs)

    _auto.from_pretrained = _from_pretrained

    _P = transformers.PreTrainedModel
    _orig_to, _orig_cuda = _P.to, _P.cuda

    def _to(self, *args, **kwargs):
        return self if getattr(self, "hf_device_map", None) else _orig_to(self, *args, **kwargs)

    def _cuda(self, *args, **kwargs):
        return self if getattr(self, "hf_device_map", None) else _orig_cuda(self, *args, **kwargs)

    _P.to, _P.cuda = _to, _cuda


@torch.no_grad()
def _nll(model, tok, sents, prefix, bs=64):
    tok.padding_side = "right"
    total, n = 0.0, 0
    for s in range(0, len(sents), bs):
        b = tok([prefix + x for x in sents[s:s + bs]], return_tensors="pt", padding=True).to(model.device)
        logits = model(**b).logits[:, :-1].float()
        target = b["input_ids"][:, 1:].to(logits.device)
        mask = b["attention_mask"][:, 1:].bool().to(logits.device)
        lp = torch.log_softmax(logits, -1).gather(-1, target[..., None])[..., 0]
        total -= lp[mask].sum().item()
        n += mask.sum().item()
    return total / n


sweep.nll = _nll


@torch.no_grad()
def _prompt_means(model, tok, key, prompts, heads, dh, bs):
    """followup.prompt_means with the mask moved to each layer's device (needed when the model spans GPUs)."""
    from followup import blocks
    store = {}
    layers = sorted({l for l, _ in heads})
    hooks = [blocks(model, key)[l][1].register_forward_pre_hook(lambda m, args, l=l: store.__setitem__(l, args[0]))
             for l in layers]
    tok.padding_side = "left"
    total, n = {lh: 0 for lh in heads}, 0
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(model.device)
        model.base_model(**b)
        mask = b["attention_mask"][..., None].float()
        for l, h in heads:
            x = store[l][..., h * dh:(h + 1) * dh].float()
            total[(l, h)] = total[(l, h)] + (x * mask.to(x.device)).sum((0, 1))
        n += mask.sum().item()
    for hook in hooks:
        hook.remove()
    return {lh: t / n for lh, t in total.items()}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    name = sys.argv[1].removesuffix(".py").split("/")[-1]
    sys.argv = [f"experiments/{name}.py"] + sys.argv[2:]
    print(f"[run_big] {name} | GPUs visible: {torch.cuda.device_count()} | device_map: {MULTI}", flush=True)
    mod = __import__(name)
    for m in (mod, sys.modules.get("followup")):
        if m is not None and hasattr(m, "prompt_means"):
            m.prompt_means = _prompt_means
    mod.main()


if __name__ == "__main__":
    main()
