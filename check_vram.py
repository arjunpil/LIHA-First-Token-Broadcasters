"""
check_vram.py — Run this BEFORE qwen_experiment.py to confirm VRAM is sufficient.

Loads Qwen2.5-1.5B-Instruct, runs one full generation pass (40 tokens),
then runs one ablation pass — the exact same operations as the real sweep.
Reports peak VRAM at each stage.

If this completes without OOM: you're safe to run the full sweep.
If this OOMs: prints the exact memory numbers so we can decide what to cut.

Run:
  python check_vram.py
"""

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME  = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.float16 if torch.cuda.is_available() else torch.float32
MAX_TOKENS  = 40

def vram_str():
    if not torch.cuda.is_available():
        return "no GPU"
    alloc   = torch.cuda.memory_allocated()  / 1e9
    reserved = torch.cuda.memory_reserved()  / 1e9
    total   = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"alloc={alloc:.2f}GB  reserved={reserved:.2f}GB  total={total:.2f}GB"

def checkpoint(label):
    print(f"  [{label}]  VRAM: {vram_str()}")

print("="*55)
print("VRAM CHECK for Qwen2.5-1.5B-Instruct")
print("="*55)

if not torch.cuda.is_available():
    print("ERROR: No CUDA GPU detected.")
    print("Make sure you're running this on your RTX 3060 machine,")
    print("not inside a cloud notebook or CPU-only environment.")
    exit(1)

props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name}")
print(f"VRAM total: {props.total_memory/1e9:.1f} GB")
print()

# ── Step 1: Load model ────────────────────────────────────────────────────────
print("Step 1: Loading model...")
checkpoint("before load")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    checkpoint("after load")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
except torch.cuda.OutOfMemoryError:
    print("  OOM on model load.")
    print("  DIAGNOSIS: 1.5B in float16 needs ~3GB. Your GPU has"
          f" {props.total_memory/1e9:.1f}GB.")
    print("  FIX: Close other GPU processes and retry.")
    exit(1)

# ── Step 2: Single generation pass ───────────────────────────────────────────
print("\nStep 2: Single generation (40 tokens)...")
prompt = "Le temps aujourd'hui est très"
messages = [{"role": "user", "content": prompt}]
formatted = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(formatted, return_tensors="pt").to(DEVICE)

try:
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
        )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    checkpoint("after generation")
    print(f"  Generated: '{generated[:80]}'")
except torch.cuda.OutOfMemoryError:
    print("  OOM during generation.")
    print("  FIX: Try MAX_TOKENS=20, or reduce --n-prompts to 10.")
    exit(1)

# ── Step 3: One ablation hook pass ───────────────────────────────────────────
print("\nStep 3: One ablation hook pass (layer 0, head 0)...")
n_heads   = model.config.num_attention_heads   # 28 for Qwen 1.5B
head_dim  = model.config.hidden_size // n_heads

def ablation_hook(module, inp, out):
    result = out.clone()
    result[:, :, :head_dim] *= 0.0   # zero head 0
    return result

handle = model.model.layers[0].self_attn.o_proj.register_forward_hook(ablation_hook)

try:
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
        )
    checkpoint("after ablated generation")
    handle.remove()
except torch.cuda.OutOfMemoryError:
    handle.remove()
    print("  OOM during ablated generation.")
    print("  NOTE: Hooks add minimal overhead. This OOM is unexpected.")
    print("  FIX: Reduce MAX_TOKENS or use --n-prompts 10 in qwen_experiment.py")
    exit(1)

# ── Step 4: Peak memory summary ───────────────────────────────────────────────
peak_alloc = torch.cuda.max_memory_allocated()  / 1e9
peak_res   = torch.cuda.max_memory_reserved()   / 1e9
total      = props.total_memory / 1e9
headroom   = total - peak_res

print()
print("="*55)
print("RESULT")
print("="*55)
print(f"  Peak VRAM allocated : {peak_alloc:.2f} GB")
print(f"  Peak VRAM reserved  : {peak_res:.2f} GB")
print(f"  Total VRAM          : {total:.2f} GB")
print(f"  Headroom remaining  : {headroom:.2f} GB")
print()

if headroom > 1.0:
    print("✓ SAFE TO RUN FULL SWEEP")
    print(f"  {headroom:.1f}GB headroom is sufficient.")
    print()
    print("  Run the full sweep with:")
    print("  python qwen_experiment.py --model instruct --resume --layers 0 3")
elif headroom > 0.3:
    print("⚠ MARGINAL — likely fine but monitor closely")
    print(f"  Only {headroom:.1f}GB headroom. Longer prompts may OOM.")
    print()
    print("  Run with reduced prompts as a precaution:")
    print("  python qwen_experiment.py --model instruct --resume --n-prompts 10 --layers 0 3")
else:
    print("✗ LIKELY TO OOM on full sweep")
    print(f"  Only {headroom:.1f}GB headroom — not enough for 25 prompts × 40 tokens.")
    print()
    print("  Options:")
    print("  1. Reduce prompts:  --n-prompts 10")
    print("  2. Reduce tokens:   edit MAX_NEW_TOKENS = 20 in qwen_experiment.py")
    print("  3. Use Qwen2.5-0.5B-Instruct (fits in ~1GB)")

# Cleanup
del model
gc.collect()
torch.cuda.empty_cache()
print()
print(f"  Final VRAM after cleanup: {vram_str()}")