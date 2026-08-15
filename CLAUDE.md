# Enargeia

A browser-native LLM inference engine. Hand-written WGSL compute kernels, int4 block-wise
quantized weights, no ML framework. Runs Qwen2.5-0.5B entirely client-side.

The site is at enargeia.dev. The whole point of the project is that the machinery is
visible while it runs — not that it's the fastest engine, and not that the model is good.

## Non-negotiable constraints

These exist because violating them destroys the thing that makes the project worth
building. If a task seems to require breaking one, stop and say so rather than working
around it.

1. **No ML framework, no WebGPU wrapper.** No ONNX Runtime, no transformers.js, no
   WebLLM, no tfjs, no wgpu-matrix. Every kernel is authored in this repo. Utility
   libraries for the *site* (Lenis, GSAP, Three.js) are fine — they never touch `src/gpu`,
   `src/kernels`, `src/model`, or `src/runtime`.

2. **No CPU readback in the decode loop.** `mapAsync` costs a round trip of roughly a
   millisecond. Sampling, argmax, and top-p all run as compute shaders; only the chosen
   token ID crosses back to JavaScript. One readback per token is the budget.

3. **Never dequantize into memory.** Unpacking int4 weights into an fp32 buffer defeats
   the entire optimization — the win is fewer bytes on the bus. Dequantization happens in
   registers, inside the matmul, as each nibble is consumed.

4. **Parity before performance.** No optimization lands without the parity harness passing.
   See the `enargeia-parity` skill. A kernel that is 99% correct produces plausible text
   and is undetectable without numerical comparison.

5. **Determinism.** Greedy decode on a fixed prompt produces byte-identical output across
   runs and devices. If it doesn't, there is a race — nearly always a missing
   `workgroupBarrier()`. Treat non-determinism as a correctness bug, never as noise.

## Layering

Imports flow one direction only. `gpu` knows nothing about transformers; `kernels` knows
nothing about the model; `ui` never touches a GPUBuffer.

```
ui  →  runtime  →  model  →  kernels  →  gpu
```

```
src/
  gpu/        device init, capability detection, buffer pool, pipeline cache, timing
  kernels/    *.wgsl + typed dispatch wrappers. One file per kernel.
  model/      safetensors + .enargeia loaders, config, graph construction
  tokenizer/  byte-level BPE
  runtime/    prefill, decode loop, KV cache, sampling orchestration
  ui/         chat, inspector panels
tools/        python — quantizer, reference activation dumper
test/         parity fixtures, kernel unit tests
```

## Skills

Consult these before working in their area. They are in `.claude/skills/`.

| Skill | Use when |
|---|---|
| `enargeia-architecture` | Adding modules, wiring the graph, touching GPU resource lifetime |
| `enargeia-kernels` | Writing or changing any `.wgsl` file |
| `enargeia-parity` | Before and after any optimization; when output looks subtly wrong |
| `enargeia-web` | Any work on the site, landing page, or inspector panels |

## Stack

Vite, TypeScript strict, no framework in the engine. Python with PyTorch and safetensors
for `tools/`. Cloudflare Pages deploys `main` automatically.

Tests are Vitest in **browser mode** (Playwright, Chromium) — `npm test`. Node has no
WebGPU, so a kernel test outside a browser would only be testing a mock. `npm run bench`
serves `bench.html`, which gates on the CPU reference before it reports a number.

## Target model

Qwen2.5-0.5B-Instruct. 24 layers, hidden 896, 14 query heads, 2 KV heads, head dim 64,
MLP intermediate 4864, vocab 151,936, tied embeddings.

The embedding table (151,936 × 896 ≈ 136M params) is the largest single tensor and the
first thing to exceed `maxStorageBufferBindingSize` on many devices. It must be split
across bindings. Assume this rather than discovering it.

## Working style

Terse. State what changed and what it measured. Skip preamble, skip summaries of work
already visible in the diff. When something is uncertain, say which measurement would
settle it rather than speculating.
