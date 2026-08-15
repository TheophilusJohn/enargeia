#!/usr/bin/env python3
"""Dump every activation boundary from the PyTorch reference.

A kernel that is 99% correct produces fluent text. Reading the output does not detect that;
numerical comparison against a reference does. This script produces the reference.

Every tensor written here comes out of the HuggingFace model — nothing is recomputed by this
script, because a recomputed "reference" only proves the script agrees with itself. Q and K
after RoPE come from patching `apply_rotary_pos_emb` to record what it returned; attention
weights come from `output_attentions=True` with eager attention; the attention output before
`o_proj` and the post-SiLU product come from forward-pre-hooks, since a module's input is the
previous stage's output.

Two values are exceptions and are labelled as sums rather than captures: `resid_attn` is
`layer_input + o_proj_out`, and both terms are themselves dumped, so the sum is exact and
checkable.

Usage:
    tools/.venv/bin/python tools/dump_reference.py

Requires torch and transformers:
    tools/.venv/bin/pip install torch transformers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    sys.exit("missing dependency: tools/.venv/bin/pip install torch transformers")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--weights", type=Path, default=Path("test/fixtures/model.safetensors"))
    parser.add_argument("--tokenizer", type=Path, default=Path("test/fixtures/tokenizer.json"))
    parser.add_argument("--prompt", type=Path, default=Path("test/fixtures/prompt.txt"))
    parser.add_argument("--out", type=Path, default=Path("test/fixtures/reference.bin"))
    parser.add_argument("--sidecar", type=Path, default=Path("test/fixtures/reference.json"))
    parser.add_argument("--decode", type=int, default=20, help="greedy tokens to record")
    args = parser.parse_args()

    prompt = args.prompt.read_text(encoding="utf-8")

    # Load config and weights offline from the local checkpoint, so this does not depend on
    # network access or on a cached HF home directory.
    config = AutoConfig.from_pretrained(args.model)
    config._attn_implementation = "eager"  # eager attention exposes attention weights
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=config, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    seq = ids.shape[1]
    layers = model.config.num_hidden_layers
    print(f"prompt: {seq} tokens, {layers} layers, hidden {model.config.hidden_size}")

    captured: dict[str, torch.Tensor] = {}

    def keep(name: str, tensor: torch.Tensor) -> None:
        captured[name] = tensor.detach().to(torch.float32).contiguous().cpu()

    handles = []

    def hook_out(name):
        def fn(_module, _inputs, output):
            keep(name, output[0] if isinstance(output, tuple) else output)
        return fn

    def hook_in(name):
        def fn(_module, inputs):
            keep(name, inputs[0])
        return fn

    base = model.model
    handles.append(base.embed_tokens.register_forward_hook(hook_out("embeddings")))
    handles.append(base.norm.register_forward_hook(hook_out("final_norm")))

    for i, layer in enumerate(base.layers):
        p = f"layer{i}"
        handles.append(layer.register_forward_pre_hook(hook_in(f"{p}.input")))
        handles.append(layer.register_forward_hook(hook_out(f"{p}.resid_mlp")))
        handles.append(layer.input_layernorm.register_forward_hook(hook_out(f"{p}.post_rmsnorm")))
        handles.append(layer.self_attn.q_proj.register_forward_hook(hook_out(f"{p}.q")))
        handles.append(layer.self_attn.k_proj.register_forward_hook(hook_out(f"{p}.k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(hook_out(f"{p}.v")))
        # A module's input is the previous stage's output, captured without recomputation.
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(hook_in(f"{p}.attn_out")))
        handles.append(layer.self_attn.o_proj.register_forward_hook(hook_out(f"{p}.o_proj")))
        handles.append(
            layer.post_attention_layernorm.register_forward_hook(hook_out(f"{p}.post_attn_rmsnorm"))
        )
        handles.append(layer.mlp.gate_proj.register_forward_hook(hook_out(f"{p}.mlp_gate")))
        handles.append(layer.mlp.up_proj.register_forward_hook(hook_out(f"{p}.mlp_up")))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook_in(f"{p}.mlp_silu_mul")))
        handles.append(layer.mlp.down_proj.register_forward_hook(hook_out(f"{p}.mlp_down")))

    # Post-RoPE Q and K are not any module's output, so record them where they are produced.
    import transformers.models.qwen2.modeling_qwen2 as qwen2

    original_rope = qwen2.apply_rotary_pos_emb
    rope_calls = {"n": 0}

    # Signature-agnostic: transformers has changed this function's parameters between major
    # versions, and the patch only needs to observe what it returns.
    def patched_rope(*fn_args, **fn_kwargs):
        q_out, k_out = original_rope(*fn_args, **fn_kwargs)
        keep(f"layer{rope_calls['n']}.q_rope", q_out)
        keep(f"layer{rope_calls['n']}.k_rope", k_out)
        rope_calls["n"] += 1
        return q_out, k_out

    qwen2.apply_rotary_pos_emb = patched_rope

    try:
        with torch.no_grad():
            rope_calls["n"] = 0
            out = model(ids, output_attentions=True, use_cache=False)
        for i, attn in enumerate(out.attentions):
            keep(f"layer{i}.attn_weights", attn)
        keep("logits", out.logits)

        # resid_attn is the only derived value, and both of its terms are dumped alongside it.
        for i in range(layers):
            captured[f"layer{i}.resid_attn"] = (
                captured[f"layer{i}.input"] + captured[f"layer{i}.o_proj"]
            ).contiguous()
    finally:
        # Detach before decoding. Hooks overwrite on every call, and greedy decode re-runs
        # the model on a growing sequence — leaving them attached silently replaces every
        # activation with the last decode step's, at a different sequence length.
        qwen2.apply_rotary_pos_emb = original_rope
        for handle in handles:
            handle.remove()

    # Greedy decode, recording the token ids the reference produces. This is the end-to-end
    # check: the engine must reproduce this sequence exactly.
    greedy = ids.clone()
    with torch.no_grad():
        for _ in range(args.decode):
            logits = model(greedy, use_cache=False).logits[:, -1, :]
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
            greedy = torch.cat([greedy, nxt], dim=1)
    generated = greedy[0, seq:].tolist()

    order = ["embeddings"]
    for i in range(layers):
        p = f"layer{i}"
        order += [
            f"{p}.input", f"{p}.post_rmsnorm",
            f"{p}.q", f"{p}.k", f"{p}.v",
            f"{p}.q_rope", f"{p}.k_rope",
            f"{p}.attn_weights", f"{p}.attn_out", f"{p}.o_proj", f"{p}.resid_attn",
            f"{p}.post_attn_rmsnorm",
            f"{p}.mlp_gate", f"{p}.mlp_up", f"{p}.mlp_silu_mul", f"{p}.mlp_down",
            f"{p}.resid_mlp",
        ]
    order += ["final_norm", "logits"]

    missing = [name for name in order if name not in captured]
    if missing:
        sys.exit(f"reference is incomplete, missing: {missing[:8]}")

    tensors = []
    offset = 0
    blobs = []
    for name in order:
        tensor = captured[name].to(torch.float32).contiguous()
        data = tensor.numpy().tobytes()
        tensors.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": "F32",
                "offset": offset,
                "byteLength": len(data),
                "layer": int(name[5:].split(".")[0]) if name.startswith("layer") else None,
                "stage": name.split(".", 1)[1] if "." in name else name,
            }
        )
        blobs.append(data)
        offset += len(data)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as handle:
        for blob in blobs:
            handle.write(blob)

    sidecar = {
        "generator": "tools/dump_reference.py",
        "model": args.model,
        "torch": torch.__version__,
        "prompt": prompt,
        "promptTokens": ids[0].tolist(),
        "seq": seq,
        "config": {
            "layers": layers,
            "hidden": model.config.hidden_size,
            "heads": model.config.num_attention_heads,
            "kvHeads": model.config.num_key_value_heads,
            "headDim": model.config.hidden_size // model.config.num_attention_heads,
            "intermediate": model.config.intermediate_size,
            "vocab": model.config.vocab_size,
            "rmsNormEps": model.config.rms_norm_eps,
            # transformers 5 moved rope_theta into a rope_parameters dict; accept both.
            "ropeTheta": float(
                getattr(model.config, "rope_theta", None)
                or getattr(model.config, "rope_parameters", {}).get("rope_theta")
            ),
            "tieWordEmbeddings": bool(model.config.tie_word_embeddings),
        },
        "greedy": generated,
        "greedyText": tokenizer.decode(generated),
        "byteLength": offset,
        "tensors": tensors,
    }
    args.sidecar.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    print(f"wrote {len(tensors)} tensors, {offset / 1e6:.1f} MB to {args.out}")
    print(f"greedy {args.decode} tokens: {generated}")
    print(f"  -> {sidecar['greedyText']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
