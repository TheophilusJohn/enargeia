# Enargeia

A browser-native LLM inference engine. Hand-written WGSL compute kernels, int4 block-wise
quantized weights, no ML framework. Qwen2.5-0.5B-Instruct runs entirely on the visitor's GPU.

**[enargeia.dev](https://enargeia.dev)** — load it and watch the kernels while they run.

*Enargeia* is a term from ancient rhetoric for description vivid enough that the audience sees
the thing rather than hears about it. The point of the project is that the machinery is visible
while it works.

## The constraint

No ONNX Runtime, no transformers.js, no WebLLM, no tfjs, no `wgpu-matrix`. Every kernel is
authored in this repository: 21 WGSL files, about 1,800 lines of shader and 9,600 lines of
TypeScript. The site is allowed libraries — Three.js, GSAP, Lenis — and they never import from
`src/gpu`, `src/kernels`, `src/model` or `src/runtime`.

Three more rules that shaped most of the code:

- **No CPU readback in the decode loop.** `mapAsync` costs about a millisecond of round trip.
  Sampling, top-p and the repetition penalty all run as compute shaders; only the chosen token
  id crosses back to JavaScript. One readback per token is the budget.
- **Never dequantize into memory.** Unpacking int4 weights into an fp32 buffer would move the
  bytes the quantization exists to avoid. Dequantization happens in registers, inside the
  matmul, as each nibble is consumed.
- **Determinism is correctness.** Greedy decode on a fixed prompt is byte-identical across runs.
  Non-determinism is treated as a race — nearly always a missing `workgroupBarrier()` — never as
  noise.

## Measured

Apple M2, headless Chromium over CDP unless a row says otherwise. Method and full history in
[BENCH.md](BENCH.md); a foreground tab shares the GPU with the compositor and reads about 15%
lower, which is why the live figure is quoted separately.

| | int4 | fp32 |
|---|---|---|
| decode, 512-token context | **45.5 tok/s** | 26.9 |
| decode, 2048-token context | **43.1 tok/s** | 21.6 |
| decode, live site, foreground tab, sampled | **38.2 tok/s** | — |
| prefill, 128 tokens | **1105 tok/s** | 225 |
| prefill, 2048 tokens, chunked at 256 | **708 tok/s** | 212 |
| time to first token, 32-token prompt | **219 ms** | — |
| weights resident | **334.9 MiB** | 1884.6 MiB |
| KV cache at 2048, f16 | 24.0 MiB | 48.0 MiB |
| activation scratch | 99.1 MiB | — |
| **live residency, everything** | **458.0 MiB** | — |
| perplexity, 1,256 held-out positions | 36.21 | 31.90 |
| cold first load over the CDN → first token | **14.6 s** | — |
| second visit, from cache | **1.4 s** | — |

334.9 MiB over 494.0M parameters is 5.69 bits per weight — int4 everywhere with an int8
embedding table, plus per-block scales.

One decode step is **412 compute passes**, or 460 where `shader-f16` lets the KV cache be half
precision. Per-kernel share, from `timestamp-query` on the live site: projections 60.8%, the
tied LM head 24.7%, sampling 9.3%, attention 2.6%, everything else under 3%.

### The matmul, 1024³ fp32

Medians of three runs **in one session**, all five kernels against the same operand buffers, so
a speedup is never two different sets of random numbers.

| kernel | GFLOP/s (wall) | GFLOP/s (GPU-side) | vs naive |
|---|---|---|---|
| `matmul_naive` | 222.4 | 214.7 | — |
| `matmul_tiled` — 16×16 tiles | 518.2 | 547.2 | 2.36× |
| `matmul_tiled4` — 4×1 strip | 838.9 | 899.7 | 3.82× |
| `matmul_tiled8` — 8×1 strip | 870.8 | 928.8 | 3.97× |
| `matmul_block42` — **4×2 block** | **1141.7** | **1240.8** | **5.20×** |

The GPU-side 1240.8 GFLOP/s is roughly **34% of 3.6 TFLOP/s**, Apple's stated fp32 peak for the
10-core M2 GPU. That denominator is a vendor figure rather than something measured here, so
treat the percentage as an order-of-magnitude statement about headroom, not a precise ratio.

The fifth rung is the interesting one: going from a 4×1 strip to an 8×1 strip bought 4%, and
going 2D instead — same thread count, 0.75 shared loads per MAC — bought 36%.

## Findings worth reading

The measurements that changed what got built, including the ones that went the wrong way.
Longer form in [BENCH.md](BENCH.md) and [docs/DECISIONS.md](docs/DECISIONS.md).

**The two largest optimizations were not on the optimization list.** M6 specified five changes.
Two measured null, one had its premise refuted before it was built, one was bounded below the
noise floor by arithmetic and skipped. The wins came from elsewhere: right-sizing prefill
dispatch geometry, which was launching 5.5M workgroups for a 32-token prompt (**11.6× at 128
tokens**), and parallelising the attention history reduction (**+50.4% decode at 2048**).

**Decode is flat in context.** Fitting inter-token time against position: `26.06 ms + 3.89
µs/position` before, `23.24 ms − 0.15 µs/position` after. The KV cache was supposed to deliver
that in M5 and did not, because the kernel reading the cache still walked it one position per
thread.

**Vary the suspected cause before optimizing against it.** This check has caught three wrong
mechanism assumptions here: an f16 KV cache predicted at +21.9% that measured +3.4%; an
over-dispatch diagnosis that turned out to be two mechanisms, not one; and a footprint claim
that was simply wrong. It is now written into the kernels skill, to be run *before* the build.

**A cost that only appears under settings no test uses is invisible to the whole test suite.**
Every harness measures with greedy decode, because that is what parity requires — and greedy
skips the repetition penalty entirely. Under the chat surface's real settings the penalty was
**2.7× the cost of everything else in decoding combined**, because it rescanned the whole history
for every one of ~36 passes over the vocabulary. Applied once, in place: 71.4 ms/token → 24.4.

**Sizing to a build-time maximum, twice.** Prefill allocated its attention scratch as
`heads × maxContext × maxContext` — 512 MiB of 773.8 for two buffers. Chunking prefill into
256-query passes took total scratch to **99.1 MiB and residency from 1132.6 to 458.0**. Both
predictions about its speed were wrong, in opposite directions (a fit said +13%, the first
measurement said −10%); measured properly against a same-session control, it is flat within
2.7%.

**Loopback hides design faults.** Every load number in this repo was originally measured against
a local dev server, where a round trip is free. Over a real CDN the loader's 630 serial range
requests cost **78.3 seconds**, and the quantized path had never used the Cache API at all, so
every visit re-downloaded 335 MB. Coalescing into 17 spans: **14.6 s cold, 1.4 s warm.** The
progress readout was rating bytes against summed per-request durations, and told a visitor
16 minutes remained on a 54-second download.

**Where int4 actually costs quality.** An ablation on the tied embedding: keeping that one
tensor at int8 costs 66 MiB and recovers **99.95%** of what a full fp32 exemption recovers. But
the hypothesis it was testing — that the output projection dominates the damage — was only a
third right. The embedding accounts for 37% of the degradation and the other 169 quantized
tensors for 63%.

**Wrong answers that look like right answers.** Three shaders silently failed to compile
(`active` and `shared` are WGSL reserved words) and dispatched nothing. A determinism self-test
passed vacuously because a struct alignment mismatch meant the kernel never ran, and zeros agree
perfectly. A tied LM head exceeding `maxComputeWorkgroupsPerDimension` invalidated the whole
command buffer and produced a perplexity of exactly 151,936 — uniform over the vocabulary.
Each is now guarded by a test that would have caught it.

**One sample is not a measurement.** Lighthouse scored the same deployment 88 and 84 on single
runs and **98 as a median of three**. A "chunking is 10% slower" result came from comparing
against a number recorded in an earlier session, where run-to-run variance is 8%. The harnesses
now take medians and carry same-session controls.

**Screenshots cannot falsify a claim about behaviour.** The pinned pipeline section never
actually pinned — in any browser — and looked correct in exactly one frame: the first one, which
is where every screenshot had been taken. Nested scroll containers could only be dragged, never
wheeled, because Lenis was taking the events. Both survived a five-width, two-engine visual
sweep. `npm run sweep` now ends with behavioural checks that start from an injected fault, so
each has been shown to fail before it is trusted to pass.

## Layout

Imports flow one direction. `gpu` knows nothing about transformers; `kernels` knows nothing
about the model; `ui` never touches a `GPUBuffer`.

```
ui  →  runtime  →  model  →  kernels  →  gpu
```

```
src/
  gpu/        device init, capability detection, buffer pool, pipeline cache
  kernels/    *.wgsl + typed dispatch wrappers, one file per kernel
  model/      safetensors and .enargeia loaders, config, graph construction
  tokenizer/  byte-level BPE
  runtime/    prefill, decode loop, KV cache, sampling orchestration
  ui/         chat, inspector panels, landing page
tools/        python quantizer and reference dumper; measurement harnesses
test/         parity fixtures, kernel unit tests
```

## Running it

```bash
npm install
npx playwright install chromium webkit

# Quantize a checkpoint to .enargeia (needs tools/.venv with torch + safetensors).
# --embed-dtype q8 is the shipping configuration; see the ablation in BENCH.md.
npm run quantize -- --input model.safetensors \
  --output public/models/qwen2.5-0.5b.enargeia --embed-dtype q8

npm run dev          # local, weights from public/models
npm test             # 261 kernel and unit tests, Vitest in browser mode
```

Tests run in a real browser through Playwright, not Node: Node has no WebGPU, so a kernel test
outside a browser would only be testing a mock.

| command | what it checks |
|---|---|
| `npm test` | 261 tests — kernels against float64 CPU references, tokenizer, loaders |
| `npm run parity` | every activation boundary against a PyTorch dump — **387 stages** |
| `npm run session` | 11 tests — prefill+decode reproduces the no-cache path, sampling, stop tokens |
| `npm run ablation` | one row per optimization, measured on its own |
| `npm run residency` | live GPU allocations, largest first |
| `npm run quality` | perplexity on held-out text |
| `npm run sweep` | five widths in WebKit, then wheel routing and the software-adapter warning |
| `npm run lighthouse` | median of three runs, because one is not a measurement |

The parity harness has two columns. **Isolated** feeds each stage reference-quality inputs and
gates the run — read it from the top, the first red row is the bug. **Accumulated** runs the
full chain and is reported, not asserted, because fp32 drift over 24 layers is expected.

## What this is not

**The model is not good.** Qwen2.5-0.5B-Instruct is roughly a thousandth the size of a frontier
model and behaves like it: it loses the thread over a few paragraphs, invents facts confidently,
and cannot do arithmetic. It also fails to emit a turn terminator about 20% of the time, which
is why replies are capped and the chat says when a cap was hit.

**This is not the right choice for production. Use [WebLLM](https://webllm.mlc.ai/).** It
supports many more models, is far better tested across devices, has real batching and
grammar-constrained output, and is maintained by people who do this full time. Enargeia exists
because writing the kernels is the interesting part, and because an engine small enough to read
end to end teaches things a compiled runtime does not.

**Quantization costs about 13.5% of perplexity** — 36.21 against 31.90 for the same weights in
fp32. That is the honest number over 1,256 held-out positions; an earlier evaluation on 95
positions read 35.22 against 30.25 and could not resolve differences that small.

**It is not the fastest browser engine, and speed was never the goal.** Some of the code is
slower than it could be on purpose, because the version that shows what it is doing is worth
more here than the version that hides it.

## Documents

- **[BENCH.md](BENCH.md)** — every number, with method, including the changes that made things
  slower and the predictions that were wrong.
- **[docs/DECISIONS.md](docs/DECISIONS.md)** — 87 entries: what was chosen, what was rejected,
  and what decided it. Including the ones later withdrawn.
- **[docs/tiled-matmul.md](docs/tiled-matmul.md)** — the matmul walkthrough, naive to 2D
  register blocking, for someone who has only seen the naive version.

## Licence

Model weights are Qwen2.5-0.5B-Instruct, Apache 2.0, quantized offline. The engine code has no
licence file yet — add one before anyone is invited to reuse it.
