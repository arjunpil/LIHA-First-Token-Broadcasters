import argparse
import csv
import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import flores
from sweep import ablated, blocks, generate, nll


LANGS = ("en", "fr", "de", "es", "it")
HERE = Path(__file__).resolve().parent


def load_rows(path, per_lang):
    seen = {}
    rows = []

    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lang = row["language"]
            seen[lang] = seen.get(lang, 0) + 1
            if seen[lang] <= per_lang:
                rows.append(row)

    if not rows:
        raise RuntimeError(f"No prompts loaded from {path}")

    return rows


def parse_heads(spec, n_layers, n_heads):
    if spec is None:
        return [
            (layer, head)
            for layer in range(n_layers)
            for head in range(n_heads)
        ]

    heads = []

    for item in spec.split(","):
        item = item.strip()

        if not item.startswith("L") or "H" not in item:
            raise ValueError(
                f"Bad head '{item}'. Use format like L6H10,L6H1"
            )

        layer_s, head_s = item[1:].split("H", 1)
        layer = int(layer_s)
        head = int(head_s)

        if not (0 <= layer < n_layers):
            raise ValueError(f"Layer out of range: {item}")

        if not (0 <= head < n_heads):
            raise ValueError(f"Head out of range: {item}")

        heads.append((layer, head))

    return heads


@torch.no_grad()
def compute_mean_activations(model, tokenizer, prompts, batch_size):
    """
    Compute the clean dataset mean at the GPT-2 c_proj input.

    The c_proj input is the concatenation of the individual attention-head
    outputs, so replacing one head slice here performs genuine mean ablation
    at the same location where the corrected zero-ablation runner intervenes.
    """
    layer_blocks = blocks(model, "gpt2")
    hidden_size = model.config.hidden_size

    sums = [
        torch.zeros(
            hidden_size,
            device=model.device,
            dtype=torch.float32,
        )
        for _ in layer_blocks
    ]

    state = {"mask": None}
    handles = []

    def make_hook(layer):
        def hook(module, args):
            x = args[0].detach()

            mask = (
                state["mask"][:, : x.shape[1]]
                .to(device=x.device, dtype=x.dtype)
                .unsqueeze(-1)
            )

            sums[layer].add_(
                (x * mask).sum(dim=(0, 1)).float()
            )

        return hook

    for layer, (_, proj) in enumerate(layer_blocks):
        handles.append(
            proj.register_forward_pre_hook(make_hook(layer))
        )

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    total_tokens = 0

    try:
        for start in range(0, len(prompts), batch_size):
            batch = tokenizer(
                prompts[start : start + batch_size],
                return_tensors="pt",
                padding=True,
            ).to(model.device)

            state["mask"] = batch["attention_mask"]
            total_tokens += int(
                batch["attention_mask"].sum().item()
            )

            model.transformer(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                return_dict=True,
            )

    finally:
        for handle in handles:
            handle.remove()

        tokenizer.padding_side = old_padding_side
        state["mask"] = None

    if total_tokens == 0:
        raise RuntimeError("Mean pass saw zero valid tokens")

    means = torch.stack(
        [layer_sum / total_tokens for layer_sum in sums]
    ).cpu()

    return means, total_tokens


@contextmanager
def mean_ablated(
    model,
    layer,
    head,
    head_dim,
    means,
):
    """
    Replace exactly one head's c_proj-input slice by its clean dataset mean.
    """
    _, proj = blocks(model, "gpt2")[layer]

    start = head * head_dim
    end = (head + 1) * head_dim

    target = means[layer, start:end].to(model.device)

    def hook(module, args):
        x = args[0].clone()

        x[..., start:end] = target.to(dtype=x.dtype)

        return (x,) + tuple(args[1:])

    handle = proj.register_forward_pre_hook(hook)

    try:
        yield
    finally:
        handle.remove()


def existing_conditions(path):
    if not path.exists():
        return set()

    conditions = set()

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                conditions.add(json.loads(line)["cond"])

    return conditions


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Matched GPT-2 mean-vs-zero head ablation using the "
            "corrected c_proj-input intervention."
        )
    )

    parser.add_argument(
        "--prompts",
        default=str(HERE / "prompts_european.csv"),
    )

    parser.add_argument(
        "--per-lang",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--heads",
        default=None,
        help=(
            "Comma-separated subset such as L6H10,L6H1. "
            "Default is all 144 GPT-2 heads."
        ),
    )

    parser.add_argument(
        "--modes",
        default="mean,zero",
        help="Comma-separated mean and/or zero.",
    )

    parser.add_argument(
        "--bs",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--mean-bs",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--n-loss",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--out",
        default=str(HERE / "out" / "gpt2_mean_zero"),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    modes = [
        mode.strip()
        for mode in args.modes.split(",")
        if mode.strip()
    ]

    invalid_modes = [
        mode
        for mode in modes
        if mode not in {"mean", "zero"}
    ]

    if invalid_modes:
        raise ValueError(
            f"Unsupported modes: {invalid_modes}"
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(
        args.prompts,
        args.per_lang,
    )

    prompts_path = out_dir / "prompts.csv"
    means_path = out_dir / "mean_activations.pt"
    output_path = out_dir / "gens.jsonl"
    signature_path = out_dir / "run_signature.json"

    prompt_payload = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    prompt_sha256 = hashlib.sha256(
        prompt_payload
    ).hexdigest()

    run_signature = {
        "model": "gpt2",
        "prompt_sha256": prompt_sha256,
        "per_lang": args.per_lang,
        "n_prompts": len(rows),
        "batch_size": args.bs,
        "mean_batch_size": args.mean_bs,
        "max_new_tokens": args.max_new_tokens,
        "n_loss": args.n_loss,
        "mean_definition": (
            "clean global dataset mean at GPT-2 c_proj "
            "input over valid prompt tokens"
        ),
    }

    if args.resume:
        cached_artifacts = [
            candidate.name
            for candidate in (
                means_path,
                output_path,
            )
            if candidate.exists()
        ]

        if (
            cached_artifacts
            and not signature_path.exists()
        ):
            raise RuntimeError(
                "Refusing to resume cached outputs without "
                "run_signature.json. Use a new --out directory."
            )

        if signature_path.exists():
            previous = json.loads(
                signature_path.read_text(
                    encoding="utf-8"
                )
            )

            mismatches = {
                key: (
                    previous.get(key),
                    value,
                )
                for key, value
                in run_signature.items()
                if previous.get(key) != value
            }

            if mismatches:
                details = "; ".join(
                    f"{key}: cached={old!r}, "
                    f"requested={new!r}"
                    for key, (old, new)
                    in mismatches.items()
                )

                raise RuntimeError(
                    "Refusing to resume an incompatible run: "
                    + details
                )

    signature_path.write_text(
        json.dumps(
            run_signature,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    with open(
        prompts_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = (
        tokenizer.pad_token
        or tokenizer.eos_token
    )

    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        dtype=torch.float32,
        attn_implementation="eager",
    ).cuda().eval()

    config = model.config
    n_heads = config.num_attention_heads
    head_dim = config.hidden_size // n_heads

    heads = parse_heads(
        args.heads,
        config.num_hidden_layers,
        n_heads,
    )

    prompts = [
        row["prompt"]
        for row in rows
    ]

    order = sorted(
        range(len(prompts)),
        key=lambda i: len(
            tokenizer(prompts[i]).input_ids
        ),
    )

    dev = flores("dev")

    loss_sentences = {
        lang: [
            text.strip()
            for text in dev[lang][: args.n_loss]
        ]
        for lang in LANGS
    }

    prefix = tokenizer.bos_token or ""

    if "mean" in modes:
        if means_path.exists() and args.resume:
            payload = torch.load(
                means_path,
                map_location="cpu",
            )

            means = payload["means"]
            mean_token_count = payload["token_count"]

            print(
                f"Loaded cached means from {means_path} "
                f"({mean_token_count} tokens)"
            )

        else:
            print(
                f"Computing mean activations on "
                f"{len(prompts)} prompts..."
            )

            means, mean_token_count = (
                compute_mean_activations(
                    model,
                    tokenizer,
                    prompts,
                    args.mean_bs,
                )
            )

            torch.save(
                {
                    "means": means,
                    "token_count": mean_token_count,
                },
                means_path,
            )

            print(
                f"Saved means from {mean_token_count} "
                f"valid tokens."
            )

    else:
        means = None
        mean_token_count = None

    completed = (
        existing_conditions(output_path)
        if args.resume
        else set()
    )

    file_mode = (
        "a"
        if args.resume and output_path.exists()
        else "w"
    )

    def compute_losses():
        return {
            lang: nll(
                model,
                tokenizer,
                sentences,
                prefix,
            )
            for lang, sentences
            in loss_sentences.items()
        }

    def record(file, condition, texts, losses):
        file.write(
            json.dumps(
                {
                    "cond": condition,
                    "texts": texts,
                    "nll": losses,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        file.flush()

    start_time = time.time()

    with open(
        output_path,
        file_mode,
        encoding="utf-8",
    ) as f:

        if "base" not in completed:
            print("Running baseline...")

            record(
                f,
                "base",
                generate(
                    model,
                    tokenizer,
                    prompts,
                    order,
                    args.bs,
                    args.max_new_tokens,
                ),
                compute_losses(),
            )

        for mode in modes:
            for layer, head in heads:
                condition = (
                    f"{mode}:L{layer}H{head}"
                )

                if condition in completed:
                    print(f"Skipping {condition}")
                    continue

                if mode == "mean":
                    context = mean_ablated(
                        model,
                        layer,
                        head,
                        head_dim,
                        means,
                    )

                else:
                    context = ablated(
                        model,
                        "gpt2",
                        "head",
                        layer,
                        head,
                        head_dim,
                    )

                print(f"Running {condition}...")

                with context:
                    texts = generate(
                        model,
                        tokenizer,
                        prompts,
                        order,
                        args.bs,
                        args.max_new_tokens,
                    )

                    losses = compute_losses()

                record(
                    f,
                    condition,
                    texts,
                    losses,
                )

                elapsed = (
                    time.time() - start_time
                )

                print(
                    f"Finished {condition}; "
                    f"{elapsed:.1f}s elapsed"
                )

    prompt_path = Path(args.prompts).resolve()
    try:
        prompt_metadata = str(
            prompt_path.relative_to(HERE.parent.resolve())
        )
    except ValueError:
        prompt_metadata = prompt_path.name

    metadata = {
        "model": "gpt2",
        "prompt_file": prompt_metadata,
        "prompt_sha256": prompt_sha256,
        "run_signature_file": signature_path.name,
        "per_lang": args.per_lang,
        "n_prompts": len(prompts),
        "heads": [
            f"L{layer}H{head}"
            for layer, head in heads
        ],
        "modes": modes,
        "max_new_tokens": args.max_new_tokens,
        "n_loss": args.n_loss,
        "mean_definition": (
            "clean global dataset mean at "
            "GPT-2 c_proj input over valid prompt tokens"
        ),
        "mean_token_count": mean_token_count,
    }

    with open(
        out_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        f"Finished. Outputs are in {out_dir}"
    )


if __name__ == "__main__":
    main()
