# Benchmarks

Device: Apple M2 · shader-f16 yes · timestamp-query yes · maxStorageBufferBindingSize
4096 MiB (adapter max; the WebGPU default is 128 MiB)

Environment: **headless Chromium driven over CDP** — the same environment `npm test` runs
in. Every number in this file is measured there unless a row says otherwise, so kernels are
compared like for like rather than across browsers.

Method: `npm run bench`. Warmup iterations are discarded, then 10 timed iterations are
submitted back to back and divided by wall clock around `onSubmittedWorkDone()`. Where
`timestamp-query` exists, GPU pass duration is measured in a second, separately timed run so
the wall number stays free of query perturbation. All kernels in a session are benchmarked
against the same operand buffers, so a speedup is never two different sets of random numbers.

## M1 — matmul, 1024³ fp32

Baseline: **219.4 GFLOP/s**, `matmul_naive`, headless Chromium. Every M1 comparison is
against that number.

| variant | GFLOP/s | ms | vs baseline | vs stage 2 | predicted |
|---|---|---|---|---|---|
| `matmul_naive` — no shared memory | 222.4 | 9.66 | 1.01× | — | — |
| `matmul_tiled` — 16×16 tiles, stage 1 | 518.2 | 4.14 | 2.36× | — | 16× traffic |
| `matmul_tiled4` — 4×1 strip, stage 2 | 838.9 | 2.56 | 3.82× | — | 1.60× |
| `matmul_tiled8` — 8×1 strip, stage 3 | 870.8 | 2.47 | 3.97× | 1.04× | 1.11× |
| `matmul_block42` — **4×2 block**, 2D | **1141.7** | **1.88** | **5.20×** | **1.36×** | 1.67× |

Medians of three runs in one session at 100 iterations, all five kernels against the same
operand buffers. GPU-side medians from `timestamp-query`: naive 214.7 GFLOP/s, tiled 547.2,
tiled4 899.7, tiled8 928.8, **block42 1240.8**.

`block42` is by far the most reproducible of the five — its GPU-side figure across the three
runs was 1233.8, 1240.8, 1244.3, a spread under 1%, while the 1D kernels spanned 8–11% on
the same runs.

Submission overhead at 1024³ is under 0.2 ms/iteration. That does *not* hold at smaller
sizes — see the size sweep below, where it dominates the wall number entirely.

### What a foreground browser tab sees

The original probe in `public/probe.html` measured **186.7 GFLOP/s** for the same naive
kernel in a foreground Safari tab. That is 15% below the headless Chromium number for
identical arithmetic — same shader, same workgroup size, same dispatch geometry. A
foreground tab shares the GPU with the compositor; a headless one does not.

The figure is kept here because it is what a visitor to the site will actually experience,
which makes it the honest number for anything user-facing. It is *not* the number to
optimize against, because it moves with whatever else is on screen. Kernel comparisons use
the headless baseline; site copy should quote something closer to the foreground one.

### Correctness

**Pass.** `npm test` — 114 tests, 9 files, run before every benchmark in this file.

Each variant is checked against the float64 CPU reference at 1×1×1, 1×896×896 (the decode
shape), 64×64×64, 37×45×23, 17×129×96, 16×16×1, and 1×1×4864. `matmul_tiled4` adds 65×33×40
and 80×16×16, which straddle its 64-row footprint; `matmul_tiled8` adds 129×33×40, 144×16×16
and 113×17×33 for its 128-row one. `matmul_block42` adds eight more — 63×31×17, 64×32×16,
65×33×40, 65×32×16, 64×33×16, 80×48×33, 129×65×40 and 20×20×16 — because its 64×32 footprint
is ragged on *both* axes, so the row mask and the column mask have to be able to disagree.
20×20×16 is the extreme case: exactly one of a thread's eight accumulators is in range. Worst
error across all of them is 0.4% of the fp32 tolerance (`abs 1e-4 + rel 1e-4` per element).

Every variant survives 20 concurrent runs at 512³ with byte-identical results — the check
that would catch a missing second `workgroupBarrier()`. None writes past the last output
element.

All five kernels are **byte-identical to each other** at 129×65×200. They walk k in the same
ascending order, so no reassociation occurs. This is not required for correctness, but while
it holds, any mismatch is a tiling bug and nothing else, which is a much sharper signal than
a tolerance check. The bench page asserts it on every run.

### Arithmetic intensity vs measured speedup

| variant | footprint | global traffic | intensity | shared loads/MAC | shared mem | resident WGs |
|---|---|---|---|---|---|---|
| naive | 16×16 | 8590 MB | 0.25 | — | 0 | — |
| tiled | 16×16 | 537 MB | 4.00 | 2.000 | 2 KiB | 16 |
| tiled4 | 64×16 | 336 MB | 6.40 | 1.250 | 5 KiB | 6 |
| tiled8 | 128×16 | 302 MB | 7.11 | 1.125 | 9 KiB | 3 |
| block42 | 64×32 | 201 MB | 10.67 | **0.750** | 6 KiB | 5 |

"Resident WGs" is workgroups that fit concurrently on a 32 KiB threadgroup-memory budget.

**Stage 1: 16× less global traffic bought 2.41×.** The gap is real and explained in
`docs/tiled-matmul.md` — the naive kernel's redundant reads were already mostly served by
cache, so tiling converted cache hits into shared-memory reads rather than converting DRAM
traffic into nothing.

**Stage 2: 1.6× predicted, 1.599× measured.** Both global traffic and shared-memory loads
per multiply-add fall by exactly 1.6×, and throughput rises by exactly 1.6×. Unlike stage 1
there is no gap to explain, which is itself the evidence that stage 1 left the kernel
genuinely shared-memory-bound.

**Stage 3: 1.11× predicted, 1.04× measured — and a loss at small shapes.** This is where 1D
coarsening stops paying. Details below.

**2D blocking: 1.67× predicted, 1.36× measured.** `matmul_block42` reaches 1240.8 GFLOP/s
GPU-side, roughly 34% of an M2's fp32 peak against tiled8's 26%.

## M1 — where coarsening stops paying

Stage 3 doubles the strip to eight outputs per thread. Predicted 1.11× on both global traffic
and shared loads per multiply-add. It does not deliver that at any measured size, and at
n = 256 it is slower than stage 2.

GPU-side throughput, 200 iterations at the two smallest sizes so timestamp quantization and
submission overhead stop dominating:

| n | tiled4 | tiled8 | ratio | fraction of predicted gain | workgroups/core, tiled4 → tiled8 |
|---|---|---|---|---|---|
| 256 | 687.7 | 651.0 | **0.95×** | −48% | 6.4 → 3.2 |
| 512 | 870.4 | 874.2 | 1.00× | 4% | 25.6 → 12.8 |
| 1024 | 904.8 | 951.9 | 1.05× | 47% | 102.4 → 51.2 |
| 2048 | 866.9 | 890.3 | 1.03× | 24% | 409.6 → 204.8 |

Every size here is a multiple of 128, so none of this is ragged-footprint waste — the tall
tile divides evenly in all four cases.

**It is not register spilling.** Eight accumulators plus addressing is not obviously past what
the register file holds, but the decisive evidence is behavioural: spilling would cost the
same per thread at every problem size, so a spilled kernel could not beat stage 2 anywhere.
Stage 3 wins by 5% at n = 1024. Whatever is wrong is not in the inner loop.

**It is occupancy, at least at the small end.** The deficit tracks how many workgroups the
kernel can keep in flight, and nothing else in the table correlates as cleanly. Stage 3 halves
the workgroup count (a 128-row footprint instead of 64) and nearly doubles shared memory per
workgroup, 5 KiB → 9 KiB. On a 32 KiB threadgroup-memory budget that is 6 resident workgroups
down to 3. Two independent halvings of latency-hiding capacity, bought for a predicted 1.11×.
At n = 256 there are 3.2 workgroups per core and the machine cannot be filled; the trade nets
out negative. By n = 1024 there is enough parallelism that it nets out positive.

**What is capping the large-shape case is not established.** At n = 1024 there are 51
workgroups per core — ample — and stage 3 still captures under half the predicted gain, while
both coarsened kernels sit at the same ~900 GFLOP/s ceiling. The loads-per-MAC model assumes
shared-load bandwidth is the sole limit; the plateau says it is not any more. Candidates are
instruction issue rate, shared-memory latency not fully hidden at the lower occupancy, and
ALU/load co-issue limits. Ranking them needs a Metal GPU capture reporting achieved occupancy
and registers per thread, which WebGPU exposes by no API.

**The structural finding.** Coarsening only along rows amortizes tileB and never tileA, so
the ratio can only approach 1.0 loads per MAC — 8×1 gets to 1.125 and 16×1 would reach 1.06
with a 16 KiB tile. Coarsening in *both* directions amortizes both operands. That prediction
was then tested; see below.

## M1 — 2D blocking, and what the ~900 plateau actually was

`matmul_block42` gives each thread a 4×2 block: four rows, two columns, eight accumulators —
the same count as stage 3, arranged so both operands amortize. 4 tileA loads + 2 tileB loads
per 8 multiply-adds is **0.75 loads/MAC**, against 1.125 for an 8×1 strip. A 16×16 workgroup
then covers 64 rows × 32 columns, so tileA is 64×16 and tileB is 16×32 — 6 KiB, *less* than
stage 3's 9 KiB. Dispatch divides by 32 in x and 64 in y.

GPU-side throughput across the sweep, 400/400/100/30 iterations by size:

| n | naive | tiled | tiled4 | tiled8 | **block42** | block42/tiled4 (pred 1.67×) |
|---|---|---|---|---|---|---|
| 256 | 211.3 | 498.1 | 682.5 | 634.6 | **817.6** | 1.20× — 30% of predicted |
| 512 | 227.1 | 546.5 | 865.4 | 872.1 | **1162.1** | 1.34× — 51% |
| 1024 | 218.0 | 547.2 | 899.7 | 928.8 | **1233.8** | 1.37× — 56% |
| 2048 | 218.5 | 526.6 | 881.4 | 907.5 | **1204.9** | 1.37× — 55% |

### The hypothesis under test

The stage 3 section observed that tiled4 and tiled8 both sat near 900 GFLOP/s despite load
ratios of 1.25 and 1.125, and read that as evidence that shared-load ratio had stopped being
the binding constraint — with a co-limit somewhere else that could not be identified from
inside the browser.

**That reading was wrong, and block42 falsifies it.** A kernel with a better ratio reaches
1240 GFLOP/s on the same device at the same shape. There is no ceiling at 900.

What the plateau actually was: two opposing effects cancelling. Stage 3 improved the load
ratio by 1.11× and simultaneously halved occupancy — 9 KiB per workgroup against stage 2's
5 KiB, so 3 resident workgroups instead of 6, on top of half as many workgroups dispatched.
The net was 1.04×, and two kernels landing in the same band looked like a wall.

`block42` separates the two variables cleanly, which is what makes the comparison worth
having. It dispatches **exactly the same number of workgroups as tiled8** at every size —
both have a 64-row-equivalent footprint area — but needs 6 KiB rather than 9, so 5 workgroups
fit concurrently instead of 3. Same parallelism, better ratio, better occupancy: 1.33× over
tiled8, and it wins at n = 256 where tiled8 loses.

### What the model still does not explain

The ratio model predicted 1.67× over tiled4 and delivered 1.37×. So the load ratio clearly
still binds — improving it produced the largest single jump in this table — but throughput
responds sublinearly to it. The remaining gap is unexplained here, and the honest position is
the same one stage 3 reached: identifying it needs achieved occupancy and registers per
thread from a Metal GPU capture, which WebGPU exposes by no API.

What has changed is the direction of the conclusion. Stage 3 suggested the ratio had stopped
mattering; it had not. It suggested a hard ceiling; there was none.

---

# M2 — weight loading and tokenizer

## Load times

File: a synthetic safetensors with Qwen2.5-0.5B-Instruct's exact tensor inventory — 290
BF16 tensors, 988,065,536 bytes of data, 32 KB header — served from the Vite dev server over
loopback. Real geometry, synthetic bytes. `npm run loadbench`.

| | unclamped (4096 MiB bindings) | clamped (128 MiB bindings) |
|---|---|---|
| cold load | 2.51 s | 3.31 s |
| warm load (Cache API) | 1.47 s | 1.67 s |
| warm speedup | 1.71× | 1.98× |
| chunks | 73 fetched → 73 cached | 73 fetched → 73 cached |
| embedding parts | **1** | **5** |
| resident on GPU | 1884.6 MiB | 1884.6 MiB |

**Read the cold number carefully.** Over loopback the transfer is nearly free, so 2.51 s is
mostly BF16→F32 widening and `writeBuffer` upload, not download. It is the floor, not the
experience. A visitor on a 50 Mbit/s connection waits about 160 seconds for 988 MB; on
100 Mbit/s, 80 seconds. The warm path is what makes the second visit 1.5 seconds regardless,
and that gap — minutes to seconds — is the entire reason the Cache API work exists.

The clamped run is consistently slower (3.31 s vs 2.51 s cold). Five buffers instead of one
means five `writeBuffer` calls for the embedding and five allocations rather than one; at
519 MiB that is measurable. It is the correct trade — the alternative on a 128 MiB device is
not loading at all.

## The embedding table under both limits

| | bytes | bindings |
|---|---|---|
| BF16 on disk | 259.7 MiB | — |
| F32 resident | 519.3 MiB | — |
| at the 128 MiB WebGPU default | | **5** parts, 30,388 rows each, 103.9 MiB per part |
| at this adapter's 4096 MiB | | **1** part |

Five bindings fits inside `maxStorageBuffersPerShaderStage`, which is 10 here — but only
just, once the other operands of a matmul are bound. That is a real constraint on how the
LM-head projection can be written, not a hypothetical.

**All weights as F32 come to 1884.6 MiB resident.** For a 0.5B model. That number is the
argument for the whole int4 path: 988 MB on disk becomes 1.88 GB on the GPU purely because
WGSL has no bf16 and `shader-f16` is missing on roughly a third of devices, so f32 is the
only universally bindable representation.

## Tokenizer

**Pass.** 2,226 fixture cases from `tools/dump_tokenizer_cases.py`, running the Rust
`tokenizers` library over Qwen's real tokenizer.json, all matching exactly on both encode and
decode. Groups: emoji (22), CJK (9), other scripts (9), whitespace runs (19), numbers (13),
contractions (15), code (8), edge cases (21), special tokens (98), NFC pairs (8), and 2,000
random mixtures.

One real bug was caught by the fixtures and would not have been caught by inspection: a
leading U+FEFF decoded to the empty string, because `TextDecoder` strips a BOM unless
`ignoreBOM: true` is set. Any document beginning with a byte-order mark would have silently
lost its first character.

---

# M3 — fp32 forward pass

## End-to-end result

**Greedy output is identical to HuggingFace.** 20 of 20 tokens, on the fixed prompt in
`test/fixtures/prompt.txt`, and identical under both binding configurations:

```
[785, 6722, 315, 9625, 374, 12095, 13, 151645, 198, 151643,
 33975, 22977, 614, 1012, 5382, 304, 279, 3639, 4180, 369]
"The capital of France is Paris.<|im_end|>\n<|endoftext|>Human beings have been living in the United States for"
```

| | unclamped (4096 MiB bindings) | clamped (128 MiB bindings) |
|---|---|---|
| embedding parts | 1 | **5** |
| dispatches per forward pass | 412 | 420 |
| greedy tokens matching reference | **20 / 20** | **20 / 20** |
| ms per token | 129 | 129 |
| resident weights | 1976 MB | 1976 MB |

129 ms/token at 15→35 tokens, no KV cache, recomputing the full sequence every step. That is
the intended cost for M3; the number is here as a baseline for the cache to beat, not as a
result. Decode is byte-identical across runs.

The eight extra dispatches under clamping are the four additional embedding-gather dispatches
and four additional LM-head dispatches — one of each per extra part. Throughput is unchanged
because the work is the same; only its partitioning differs.

## Per-kernel correctness

**Pass.** 31 tests in `test/kernels/forward_ops.test.ts`, every kernel against its CPU
reference in `test/reference/ops.ts`, all within the fp32 tolerance. Includes the properties
that are wrong-but-plausible if missed:

- RMSNorm normalizes by the mean of squares, not the sum
- `matmul_bias` reads its weight transposed, as the checkpoint stores it `[out, in]`
- RoPE rotates halves (element `i` with `i + headDim/2`), not adjacent pairs
- attention groups the 14 query heads onto 2 kv heads **contiguously** — heads 0–6 on kv 0
- softmax survives inputs of 200+ that would overflow without max subtraction
- argmax breaks ties toward the lowest index, as `torch.argmax` does

A separate test asserts all ten shaders compile. That is not redundant:
`createComputePipeline` is synchronous and reports a bad shader through the uncaptured error
handler rather than by throwing, so a pipeline built from a broken shader is a valid object
that dispatches nothing. Three shaders failed to compile on first run — `active` and `shared`
are WGSL reserved keywords, and an `f32::MIN` literal rounded outside the representable range
— and every symptom looked like a numerically wrong kernel.

## Per-layer parity table

`npm run parity`. 387 stages, two columns.

| column | how it is produced | role |
|---|---|---|
| **ISOLATED** | every input to the stage written from the reference dump, then only that stage's dispatches encoded | **gates the run** |
| ACCUMULATED | the whole prefix from the prompt, as the engine actually runs it | reported, not asserted |

**ISOLATED: all 387 stages within tolerance.** Worst case `layer0.attn_weights` at 23.4% of
the fp32 threshold (`abs 1e-4 + rel 1e-4` per element). There is no real bug: with
reference-quality inputs every kernel reproduces the reference at every depth, and the first
red row genuinely would be the bug if there were one.

**ACCUMULATED: 22 of 387 over tolerance, from `layer1.q` onward.** Worst is `final_norm` at
635% (2.23e-3 absolute). This is the fp32 path's real end-to-end error and it is kept because
it is the baseline the int4 path gets compared against — a quantized run that lands near these
numbers is behaving as expected, and one that lands far above them is not.

The two columns side by side at layer 23:

| stage | isolated | accumulated |
|---|---|---|
| `post_rmsnorm` | 0.3% | 159.4% |
| `q` | 5.7% | 67.8% |
| `attn_weights` | 0.5% | 17.8% |
| `mlp_silu_mul` | 0.2% | 266.4% |
| `resid_mlp` | 0.0% | 125.9% |
| `final_norm` | 0.2% | 635.4% |

Accumulated error grows monotonically with depth while isolated error does not grow at all.
That is the signature of compounding rather than of a defect, and it is now legible from the
table instead of requiring three separate arguments to establish.

### Which stages are bit-exact, and why it is not suspicious

Several isolated rows read `0.00e+0` — bit-identical to PyTorch. The pattern is exact:

- **bias-free projections** (`o_proj`, `mlp_gate`, `mlp_up`, `mlp_down`) and the pure adds
  (`resid_attn`, `resid_mlp`) and `attn_out`: bit-exact
- **biased projections** (`q`, `k`, `v`): 3–10% of tolerance
- **transcendental kernels** (`rmsnorm`, `rope`, `softmax`, `silu_mul`): 0.1–23%

Addition of two reference tensors is exactly reproducible, so the residuals must be exact. The
bias-free GEMMs matching bit-for-bit says PyTorch's CPU f32 kernel happens to accumulate in the
same order this kernel does at these shapes; adding a bias diverges because `addmm` folds it in
differently. The transcendental kernels differ in the last ulp because GPU and CPU
implementations of `rsqrt`, `sin`, `cos` and `exp` are not required to agree. None of that is
load-bearing — it is recorded so a future change in the pattern is recognisable as a change
rather than as noise.

### Isolation is per stage, with two exceptions

`embeddings` has no activation inputs — its only input is the prompt — so isolated and
accumulated are the same measurement. `attn_weights` is isolated as two dispatches, scores then
softmax, fed from reference post-RoPE q and k: the pre-softmax scores are not a dumped
boundary, so the softmax cannot be fed a reference input directly. Every other stage is one
dispatch fed entirely from the dump.

---

# M4 — int4 block-wise quantization

## Resident VRAM

| | resident | vs fp32 |
|---|---|---|
| fp32 (M2 measured) | **1884.6 MiB** | — |
| int4 | **268.9 MiB** | **7.0× smaller** |

On disk: 942 MB bf16 safetensors → **282.1 MB** `.enargeia`, **4.57 bits/weight** over 494.0M
quantized parameters. Scales are f32 rather than f16, which costs 0.5 bits/weight and buys one
universal path on devices without `shader-f16`.

**The split-binding constraint disappears.** The fp32 embedding table needed five bindings
under the 128 MiB default; at int4 the whole 151,936 × 896 table is 68 MB of packed nibbles
and fits in one. The per-part machinery is kept because the constraint returns for a larger
vocabulary, but M1's most awkward architectural problem is not a problem at int4.

## Throughput

96-token context, 6 iterations after 2 warmup, same operand buffers.

| | fp32 | int4 | |
|---|---|---|---|
| step (96-token context) | 422.7 ms | **108.9 ms** | **3.88× faster** |
| prefill throughput | 227 tok/s | **883 tok/s** | 3.88× |
| 20-token greedy run | 129 ms/token | **60 ms/token** | 2.15× |

**int4 is faster at prefill shapes, which is not what was predicted.** The expectation was a
win at decode where bandwidth binds, and possibly a loss at prefill where the nibble unpacking
is arithmetic the fp32 kernel does not do. The unpack cost is real but it is not the binding
constraint here: at 1885 MiB the fp32 weights do not fit in any cache, so even the
reuse-heavy prefill shape is limited by weight traffic, and quarter-size weights help it too.

**The decode number above is not a decode measurement, and cannot be one yet.** Without a KV
cache every step recomputes the whole prefix, so a "decode step" at context 95 *is* a prefill
of 95 tokens — the two columns measure the same thing and come out equal (108.9 vs 108.7 ms).
The only genuinely decode-shaped dispatch in this graph is the tied LM head at m=1. Separating
the two properly needs the KV cache in M5; `matmul_q4_decode` exists, is tested, and is used
for the LM head, but its bandwidth-bound win is unmeasured.

## Quality

| | perplexity, 95 positions of held-out text |
|---|---|
| fp32 | **30.2520** |
| int4 | **38.8165** (+28.3%) |

Greedy decode on the fixed prompt still answers correctly — "The capital of France is Paris."
— and matches fp32 token-for-token for 11 of 20 tokens before diverging into a different but
coherent continuation. That is what correct int4 looks like; exact agreement would mean
quantization was lossless.

**+28% is a larger perplexity cost than a well-behaved int4 should pay, and the evidence
points at the tied LM head.** The embedding table is both the input lookup and the output
projection, and quantizing it is the single largest saving — 544 MB of the 1885 MB. It is also
where a small weight error turns directly into a shifted token distribution. Measured on the
CPU from the shipped file, quantizing that one tensor perturbs the final logits by rms 0.376
over a logit range of −15.4 to 20.5, which is the same order as the 0.249-nat increase in mean
NLL that the perplexity ratio implies.

**The next measurement, not yet run:** keep `model.embed_tokens.weight` in fp32 and re-measure.
That costs 544 MiB of the 1616 MiB saved and should recover most of the perplexity if the
diagnosis is right. It is left for a separate change rather than folded in here, because
"quantize everything" is the honest baseline and the exception needs its own number.

## Is the kernel correct?

The int4 thresholds (`abs 5e-2, rel 8e-2`) cannot answer this — that is the skill's own point,
and here they are not even loose enough: 165 of 387 stages exceed them. Three checks answer it
instead, and all three pass:

1. **Kernel against the CPU dequantized reference.** `matmul_q4_prefill` and
   `matmul_q4_decode` both reproduce `linearQ4` to fp32 tolerance, and the prefill kernel
   equals an fp32 matmul over the dequantized weights. The two q4 kernels also agree with each
   other at m=1 despite different tiling and different unpack placement.
2. **The shipped file decodes to the weights it was made from.** `node tools/check_quantization.ts`
   reads `.enargeia` with the shader's exact indexing and recovers `q_proj` at max abs 0.07214,
   rms 0.00665 — identical to what `quantize.py` reported when it wrote the file.
3. **The GPU's error equals the arithmetic minimum.** At `layer0.q` the GPU produced max abs
   error 0.586 against the fp32 reference; a CPU forward pass over the *dequantized shipped
   weights* produces 0.5864. Identical, so every bit of the divergence is quantization loss and
   none of it is the kernel.

### What the parity table means under int4

The isolated column gates fp32 and does not gate int4, and the harness now says so. Giving a
stage reference-quality *inputs* does not remove the quantization of its *weights* — the
weights are the thing that changed — so under int4 that column measures per-stage quantization
loss. It is a useful diagnostic (it is how the q and k projections were identified as the worst
quantizers, matching `quantize.py`'s own per-tensor report) but it cannot detect a bug, and
asserting on it would be asserting that quantization is lossless.

## M4 ablation — tied embedding precision

Every other tensor stays int4 in all four quantized rows. Same held-out passage, 95 positions,
same device.

| config | resident | vs fp32 | perplexity | Δ MiB | Δ PPL | **PPL per MiB** |
|---|---|---|---|---|---|---|
| fp32 everywhere | 1884.6 MiB | 1.00× | 30.2520 | +1615.7 | 8.56 | 0.0053 |
| int4, embed **q4** | **268.9 MiB** | **7.01×** | 38.8165 | — | — | — |
| int4, embed **q8** | 334.9 MiB | 5.63× | **35.2248** | +66.0 | 3.59 | **0.0544** |
| int4, embed **f16** | 454.5 MiB | 4.15× | 35.3002 | +185.6 | 3.52 | 0.0189 |
| int4, embed **f32** | 714.2 MiB | 2.64× | 35.3002 | +445.3 | 3.52 | 0.0079 |

Throughput is unchanged across all four — 106–109 ms/step, 881–888 tok/s prefill. The
embedding's precision moves memory and quality, not speed, because the LM head is one
dispatch of 412.

### The hypothesis was half right, and the half it got wrong matters more

**int8 recovers 102% of what a full f32 exemption recovers, for 15% of the memory.** So the
"disproportionately sensitive" part holds: the output projection does pay more per bit than
other tensors, and 66 MiB buys back everything the embedding was costing. On perplexity per
MiB, q8 is **2.9× better than f16 and 6.9× better than f32 exemption**, and 10× better than
just staying in fp32.

**But the embedding is not where most of the damage is.** Decomposing the 28.3% increase in
log space:

| | nats | share |
|---|---|---|
| total int4 degradation | 0.2493 | 100% |
| attributable to the tied embedding | 0.0950 | **38%** |
| attributable to the other 169 int4 tensors | 0.1543 | **62%** |

Exempting the embedding entirely still leaves perplexity at 35.30 against an fp32 baseline of
30.25 — a 16.7% gap that no amount of embedding precision touches. The hypothesis predicted
the output projection would dominate; it accounts for a bit over a third. The embedding is
27.5% of the parameters and causes 38% of the loss, so it is *mildly* disproportionate, not
the main story.

### f16 and f32 are indistinguishable, and q8 is inside the noise of both

f16 and f32 give **identical perplexity to four decimal places** (35.3002). Half precision on
this tensor has no measurable effect at all: 10 mantissa bits is already lossless for a logit
that sums 896 terms. q8 lands at 35.2248, marginally *better* than exact f32 — which cannot be
a real improvement over exact weights, so the honest reading is that q8, f16 and f32 are
indistinguishable at this sample size. The three span 0.2%; the gap to q4 is 10× that, so the
ordering that matters is safe even though the ordering among the top three is not.

95 positions of one passage is a small sample and is the main limitation on these numbers. It
is enough to separate q4 from everything else and not enough to rank q8 against f32.

### What to ship

**int4 with an int8 embedding: 334.9 MiB, 5.63×, perplexity 35.22.** It captures the entire
recoverable perplexity for 66 MiB, and the next 119.6 MiB (f16) buys nothing — the marginal
return is −0.075 PPL, which is to say negative within noise.

The remaining 16.7% gap to fp32 is in the layer projections, and closing it is a different
question than this ablation asked: it would mean a larger block size study, or higher
precision on specific layers, or a better rounding rule than round-to-nearest. Worth doing
before M5 only if decode quality turns out to matter more than decode speed.

---

# M5 — KV cache, prefill/decode split, GPU sampling

Shipping config, per the M4 ablation: **int4 everywhere, int8 tied embedding**.

## Decode, measured properly for the first time

Every previous "decode" number in this file was a prefill in disguise. Without a cache a step
at context 95 recomputed all 95 positions, so decode and prefill measured the same thing and
came out equal. With the cache a step is one matrix-by-vector pass over the weights plus
O(context) reads of cached K and V, and `matmul_q4_decode` finally runs in the regime it was
written for.

**Inter-token throughput**, median of 16 tokens after a warm-up generation:

| context | int4 | fp32 | int4 speedup |
|---|---|---|---|
| 128 | **43.9 tok/s** (22.8 ms) | 28.5 tok/s (35.1 ms) | 1.54× |
| 512 | **39.8 tok/s** (25.1 ms) | 26.9 tok/s (37.2 ms) | 1.48× |
| 1024 | **35.7 tok/s** (28.0 ms) | 24.1 tok/s (41.5 ms) | 1.48× |
| 2048 | **29.0 tok/s** (34.5 ms) | 21.6 tok/s (46.2 ms) | 1.34× |

**Against M4's no-cache path — 108.9 ms/step at context 95, i.e. 9.2 tok/s — the cache is a
4.8× speedup at context 128, and the gap widens with length** because the old path was
quadratic in context and this one is linear.

int4's decode advantage is 1.34–1.54× and shrinks as context grows, which is what the shapes
predict: the weight traffic that quantization shrinks is a fixed cost per token, while the KV
traffic it does not touch grows linearly. The cache is f32 — quantizing it is a separate change
with its own quality question.

## TTFT and prefill

> **CORRECTED IN M6.** Every number in this table was measured through a prefill graph that
> baked `maxSeq` dispatch geometry, launching 5.5M workgroups per prefill at any prompt length.
> The original values are kept below for the record; the corrected ones follow.

| context | int4 TTFT | int4 prefill | fp32 TTFT | fp32 prefill |
|---|---|---|---|---|
| 128 | ~~1342 ms~~ → **113 ms** | ~~84~~ → **999 tok/s** | ~~575 ms~~ → 500 ms | ~~195~~ → 225 tok/s |
| 512 | ~~1441 ms~~ → **438 ms** | ~~345~~ → **1136 tok/s** | ~~2213 ms~~ → 2172 ms | ~~224~~ → 228 tok/s |
| 1024 | ~~1633 ms~~ → **970 ms** | ~~618~~ → **1040 tok/s** | ~~4539 ms~~ → 4513 ms | ~~222~~ → 223 tok/s |
| 2048 | ~~2418 ms~~ → 2432 ms | ~~841~~ → 837 tok/s | ~~9538 ms~~ → 9567 ms | ~~213~~ → 212 tok/s |

int4 was affected severely and fp32 barely, which is why the comparison inverted. At 2048 both
are unchanged, because there the baked geometry was already correct.

> **CORRECTED IN M6: this paragraph is wrong.** There is no crossover. int4 prefill is 4.4×
> *faster* than fp32 at 128 tokens once dispatch geometry is sized to the actual sequence. The
> reasoning below was applied to a comparison with a much larger unexamined confound in it.

~~**int4 prefill is slower than fp32 at short prompts and 3.9× faster at long ones**, crossing
over somewhere between 112 and 496 tokens. This is M1's stage-3 finding recurring in a new
place: the int4 prefill kernel inherits `block42`'s 64×32 output footprint, so a 112-token
prompt at 896 columns dispatches only 2×28 = 56 workgroups — 5.6 per core on a 10-core M2 —
while fp32's 16×16 one-thread-per-output kernel dispatches 392 for the same work. The tall
footprint that wins on long prompts starves the machine on short ones.~~

fp32 prefill is flat at ~215 tok/s regardless of length; int4 climbs from 84 to 841. The flat
line is the one that should be suspicious: it means fp32 is bandwidth-bound at every length,
which at 1885 MiB of weights is exactly what it should be.

TTFT is dominated by prefill at every context — the sampling step and the first decode are
under 2 ms of it. Numbers are taken after a warm-up generation, because pipelines compile
lazily and charging one shader compilation to TTFT reports a figure no user experiences twice
(the unwarmed run measured 1506 ms at context 128 against 2434 ms at 2048, which is the
compile, not the prompt).

## Correctness

**Pass.** `npm run session` — 9 tests. Prefill-then-decode reproduces the no-cache path's
greedy output **24 tokens out of 24**, which is the check that catches a RoPE applied at the
wrong position, a cache row written at the wrong offset, or an attention window off by one.
None of those look wrong in the generated text.

Sampling runs entirely on the GPU. Verified: greedy is reproducible across sessions; the same
seed gives the same sampled sequence and a different seed does not; temperature 0.01 with
top-p 1.0 collapses to greedy; and the repetition penalty visibly changes the continuation.

## Memory

| | resident |
|---|---|
| weights (int4, int8 embedding) | 334.9 MiB |
| KV cache at 2048 context | 48.0 MiB |
| **total** | **382.9 MiB** |

The cache is 24 layers × 2 tensors × 2048 positions × 128 elements × 4 bytes. It is allocated
once at session start and never grows; `reset()` forgets the contents without freeing, because
a reset that reallocated 48 MiB would reintroduce the stall the preallocation exists to avoid.

## Perplexity at 13× the sample size

The M4 evaluation used 95 positions, which could not resolve differences smaller than a few
percent. The cache makes long-context evaluation cheap — one O(context) step per position
instead of an O(context²) recomputation — so this is 1,256 positions of fresh held-out prose.

| config | resident | perplexity | Δ MiB | Δ PPL | **PPL per MiB** |
|---|---|---|---|---|---|
| fp32 everywhere | 1884.6 MiB | 31.9022 | +1615.7 | 7.111 | 0.0044 |
| int4, embed q4 | 268.9 MiB | 39.0132 | — | — | — |
| int4, embed **q8** | 334.9 MiB | **36.2103** | +66.0 | 2.803 | **0.0425** |
| int4, embed f16 | 454.5 MiB | 36.2090 | +185.6 | 2.804 | 0.0151 |
| int4, embed f32 | 714.2 MiB | 36.2090 | +445.3 | 2.804 | 0.0063 |

**The M4 ordering survives, and the larger sample corrects a sign the smaller one got
backwards.** At 95 positions q8 measured 0.075 *better* than exact f32 — impossible, and
correctly flagged at the time as noise. At 1,256 positions q8 is 0.0013 **worse** than f32,
which is the right direction and vanishingly small. q8 recovers **99.95%** of what a full f32
exemption recovers, for 15% of the memory. The shipping choice is unchanged and now rests on a
measurement that can support it.

f16 and f32 remain identical to four decimal places, confirming half precision is lossless for
this tensor.

The decomposition also holds: of the 0.2012-nat total degradation, the tied embedding accounts
for **37%** and the other 169 int4 tensors for **63%** — against 38%/62% at the smaller sample.
The remaining gap is in the layer projections, exactly where M4 said it was.

---

# M6 — optimization (partial)

Each row toggles exactly one change, measured by `npm run ablation`. Parity is
`npm run session` — prefill-then-decode reproducing the no-cache path's greedy output.

| change | decode @512 | decode @2048 | prefill @128 | prefill @2048 | TTFT @32 | parity |
|---|---|---|---|---|---|---|
| M5 baseline | 40.3 tok/s | 29.0 tok/s | 95 tok/s | 838 tok/s | 1518 ms | pass |
| 1. precompile pipelines | 40.3 | 29.0 | 95 | 838 | **1544** | pass |
| **1b. right-size prefill dispatch** | 40.2 | 29.2 | **1077** | 840 | **246** | pass |
| 4. kernel selection by size | — | — | — | — | — | not built, premise refuted |
| 5. batched decode uniforms | 40.3 | 29.2 | 1109 | 842 | 170 | pass |
| 2. fused RMSNorm | — | — | — | — | — | not built, bounded at 0.15% |
| 3. f16 KV cache | 41.0 | 30.2 | 1102 | 802 | 172 | pass |
| **3b. parallel history reduction** | **45.5** | **43.1** | 1105 | 797 | 219 | **pass** |
| **shipping (all of the above)** | **45.5** | **43.1** | **1105** | **797** | **219** | **pass** |

Against the M5 baseline the shipping state is **+13% decode at 512, +49% decode at 2048,
11.6× prefill at 128**, and prefill at 2048 is 5% slower from the f16 pack dispatches.

The two largest wins — right-sizing dispatch and parallelising the history reduction — were
both absent from the original list. Of the five listed items, two measured null, one had its
premise refuted, one was bounded below the noise floor and skipped, and one (item 6) is still
unbuilt.

Two of the five requested changes measured as **null results**, one **refuted its own premise**,
and the largest win in the milestone was not on the list.

## 1. Precompiling pipelines: no effect (kept anyway)

The premise was that TTFT hid "over a second" of first-dispatch shader compilation. It does
not. **All 17 pipelines compile in 4–13 ms**, and first-generation TTFT measured 1544 ms with
precompilation against 1518 ms without — a difference inside run-to-run noise, in the wrong
direction.

Kept regardless, because it costs 4 ms of a 282 MB download and the same measurement on a
device with a slower shader compiler could easily go the other way. But it is recorded as
buying nothing measurable here, not as a win.

## 1b. Right-sizing prefill dispatch: 11.3× on short prompts

Not a requested item; found while measuring item 1. The 1518 ms TTFT on a **32-token** prompt
was almost entirely prefill, which made no sense until the dispatch geometry was checked.

The prefill graph is sized to the maximum context and was baking `maxSeq` dispatch geometry
into every step at build time. At a 2048 context that is:

```
attn_scores    229,376 workgroups per layer  ->  5,505,024 per prefill
softmax         28,672 workgroups per layer
```

launched for **every prompt, at every length**. A 32-token prompt paid the launch cost of a
2048-token one, with every out-of-range thread returning immediately from its bounds check.

Making dispatch geometry a function of the sequence length, evaluated at encode time:

| | before | after |
|---|---|---|
| prefill @128 | 95 tok/s | **1077 tok/s** (11.3×) |
| TTFT @32 | 1518 ms | **246 ms** (6.2×) |
| prefill @2048 | 838 tok/s | 840 tok/s (unchanged, as expected) |
| decode @512 | 40.3 tok/s | 40.2 tok/s (unchanged — decode was already right-sized) |

The 2048 row being flat is the check that this is the right explanation: at the maximum
context the old geometry was already correct, so there was nothing to win.

## 4. Kernel selection by problem size: premise refuted, not implemented

M5 concluded that int4 prefill lost to fp32 below ~500 tokens because `block42`'s 64×32
footprint dispatched too few workgroups per core, and `docs/tiled-matmul.md` recorded that as
the third appearance of the same effect. **That conclusion was wrong.** Both formats were
measured through the same over-dispatched attention, which dominated both and hid the matmul
difference entirely.

Re-measured after 1b:

| context | int4 prefill | fp32 prefill | |
|---|---|---|---|
| 128 | **999 tok/s** | 225 tok/s | int4 4.4× faster |
| 512 | **1136** | 228 | 5.0× |
| 1024 | **1040** | 223 | 4.7× |
| 2048 | **837** | 212 | 3.9× |

There is no crossover. int4 wins at every length, and the widest margin is at the *short*
prompts where M5 said it lost. Item 4 was not implemented, because the problem it solves does
not exist. `docs/tiled-matmul.md` has been corrected.

## 5. Batched decode uniforms: no effect (kept anyway)

Decode issued around 410 `writeBuffer` calls per token, one per dispatch, for 16–48 bytes
each. Consolidating them into a single 105 KB allocation with static per-step binding offsets
and one write per token: **decode @512 went 40.3 → 40.3 tok/s.**

Decode is GPU-bound, not CPU-bound, so removing CPU work buys nothing. Kept because it is
strictly less work and would matter on a slower host, but it is a null result here.

## 2. Fused RMSNorm: not implemented, bounded by arithmetic first

The saving is the RMSNorm output write, which the following matmul would otherwise read back
from VRAM. Bounding it before building it:

| | RMSNorm writes per pass | at 100 GB/s | share of the pass |
|---|---|---|---|
| decode (seq 1) | 0.2 MB | 0.002 ms | **0.01%** |
| prefill (seq 2048) | 359.7 MB | 3.60 ms | **0.15%** |

Prefill at 2048 runs 2.02 TFLOP in 2.433 s = 832 GFLOP/s, about 67% of `block42`'s measured
1240 GFLOP/s peak — it is compute-bound, which is exactly the regime where extra bandwidth is
hidden rather than paid for.

So the ceiling on this change is 0.15% of prefill and 0.01% of decode, and the implementation
is not free: the fused form needs `rowScale` and `normWeight` bindings on every matmul, taking
the bind count from 7 to 9 against a limit of 10. **Deferred, with the numbers, rather than
built and measured at zero.** If it should be built anyway to confirm the bound, say so.

## 3 and 6: not done

**f16 accumulation** and **prompt-lookup speculative decoding** are not implemented. Both are
substantial, and the milestone's own rule is that it ends when changes are measured and written
up — so they are listed as outstanding rather than rushed.

Note that the interesting f16 target may not be accumulation at all. Decode at 2048 reads
about 50 MB of KV cache per token against 335 MB of weights, so an **f16 KV cache** is worth
roughly 13% of decode traffic at long context, while f16 accumulation in matmuls whose weights
are already int4 changes far less. That is a measurement to make before choosing which f16 to
build.

---

# M6 audit — which numbers went through the over-dispatched graph

Every timing in this file was re-derived against the fix in entry 61. The result is not the
one expected: **M4's int4-vs-fp32 comparison was unaffected, and M5's was the one that was
wrong.**

| section | affected? | why |
|---|---|---|
| M1 matmul benchmarks | **no** | standalone bench page, never built a graph |
| M2 load times, embedding split | **no** | no dispatches |
| M3 forward pass, 129 ms/token | **no** | re-measured at 130 ms/token — the graph was built at `maxSeq` 35 and run at 15–35, so over-dispatch was under 2.3× and inside noise |
| M3 per-layer parity | **no** | correctness, not timing; re-run passes 387/387 |
| M4 int4 vs fp32 at 96 tokens | **no** | re-measured 843 vs 214 tok/s against the recorded 883 vs 227 — the graph was built at `maxSeq` 96 and run at 96, so its geometry was already exact |
| M4 perplexity, VRAM, ablation | **no** | correctness and memory, not dispatch |
| **M5 TTFT / prefill table** | **YES** | built at `maxContext` 2048 and run at 128–2048; corrected in place above |
| M5 decode table | **no** | re-measured within noise at every context; decode geometry was almost all genuinely fixed |

The expectation going in was that M4 was the problem. It was not, and the reason is worth
stating: M4's harness happened to size its graph to exactly the sequence it measured, so the
bug could not express itself. M5's harness sized to the maximum context because a session must
support any prompt length — which is the correct design and is precisely what exposed the bug.
**A latent bug of this kind only appears once the surrounding code becomes realistic.**

## A second instance, found by the audit

`attn_scores_decode` sized its dispatch from `maxContext` rather than the live history:
448 workgroups per layer per token at a 2048 context, when position 100 needs 28. Fixed the
same way.

**Measured effect: none.** Decode at 512 went 40.3 → 40.0 tok/s, inside noise at every context.
The over-dispatch was real — 16× at short contexts — and immaterial, because those workgroups
return immediately and decode is not launch-bound. Recorded because the bug was genuine even
though the fix bought nothing; a reader checking this class of bug should know both instances
were found and only one mattered.

No other dispatch in either graph takes a build-time constant. Everything else is genuinely
fixed-extent: one token, `hidden` wide, `intermediate` wide, or the full vocabulary.

---

# M6 item 3 — f16: which one, decided by measurement

Rather than implementing f16 accumulation as specified, decode time was decomposed against
context first, since that bounds both candidates without building either.

```
decode ms = 22.08 + 5.8725 µs/position     (least squares over 128, 512, 1024, 2048)
```

| context | total | KV-dependent | share |
|---|---|---|---|
| 128 | 23.1 ms | 0.8 ms | 3.3% |
| 512 | 24.8 ms | 3.0 ms | 12.1% |
| 1024 | 28.0 ms | 6.0 ms | 21.5% |
| 2048 | 34.2 ms | 12.0 ms | **35.2%** |

**An f16 KV cache halves the position-dependent term**, so its ceiling is:

| context | now | with f16 KV | gain |
|---|---|---|---|
| 128 | 43.3 tok/s | 44.5 | +2.9% |
| 512 | 40.3 | 42.4 | +5.2% |
| 1024 | 35.7 | 39.9 | +11.6% |
| 2048 | 29.2 | 35.6 | **+21.8%** |

**f16 accumulation targets the 22.08 ms fixed term instead, and that term is not
bandwidth-bound.** It moves 335 MB of int4 weights in 22.08 ms — an effective 15.2 GB/s
against an M2's ~100 GB/s. Decode is limited by something other than weight traffic, most
likely the reduction structure: `matmul_q4_decode` gives each output column one 64-thread
workgroup that does 14 loop iterations and then a 6-step barrier reduction, which is a lot of
synchronisation per unit of arithmetic. Narrowing the accumulator does not change that.

**Decision: build the f16 KV cache, not f16 accumulation.** The cache is the larger target at
every context above 512 and the only one whose mechanism is bandwidth. It also needs no
`shader-f16` extension — `pack2x16float` and `unpack2x16float` are core WGSL — so unlike f16
accumulation it does not need an fp32 sibling kernel to remain universal.

## Item 3 result: f16 KV cache built, and the prediction was wrong

| context | f32 cache | predicted with f16 | **measured with f16** | predicted gain | **actual gain** |
|---|---|---|---|---|---|
| 128 | 43.3 tok/s | 44.5 | **44.6** | +2.8% | **+3.0%** |
| 512 | 40.3 | 42.4 | **41.0** | +5.2% | **+1.7%** |
| 1024 | 35.7 | 39.9 | **36.5** | +11.8% | **+2.2%** |
| 2048 | 29.2 | 35.6 | **30.2** | +21.9% | **+3.4%** |

Prefill at 2048 went 837 → 802 tok/s, a 4% regression from the added pack dispatches. Cache
memory at a 2048 context halved, 48.0 → 24.0 MiB. Parity holds: greedy output still matches the
no-cache path 24/24.

**The decomposition was right about *how much* time is position-dependent and wrong about
*what* that time is.** It measured 12.0 ms of the 34.2 ms step scaling with context, and
assumed that meant bytes. Halving the bytes returned a sixth of the predicted gain, which
falsifies the assumption: the position-dependent cost is not the KV bandwidth, it is the
attention kernels walking the history one position per loop iteration per thread. Each thread
in `attn_apply_decode` serially accumulates over every cached position, so the cost tracks
*iteration count*, and the number of iterations does not change when each one loads two bytes
instead of four.

This is the same mistake in a different costume as the M5 footprint claim: a term was measured
correctly, a mechanism was assumed for it, and the assumption was not tested until something
was built on top of it. The check that would have caught it earlier is the one that caught it
now — change the suspected cause and see whether the effect moves.

**Shipped anyway, on memory rather than speed.** +3% decode against −4% prefill is a wash, but
the cache halves, and at long contexts on a phone the 24 MiB matters more than either. The
speed case for it did not materialise and this table says so.

The real target, now identified by measurement rather than assumption, is the serial history
walk in the decode attention kernels — one thread per output element looping over every
position. Parallelising the reduction across positions is a bigger change than any item in M6
and is where the next decode work should go.

---

# M6 — parallel history reduction

## The falsification check, run before the build this time

The claim to test: the position-dependent term in decode is iteration count, not bytes. The
f16 cache already varied bytes at constant iterations and returned +3.4% against a predicted
+21.9%, so bytes were ruled out. The remaining candidates both scale with context and are
indistinguishable in a plot of time against context:

- **A. iteration/latency** — each thread walks the history serially with a dependent accumulator
- **B. occupancy** — `attn_apply_decode` dispatches 896 output elements as **14 workgroups**, i.e.
  1.4 per core on a 10-core M2, whatever the context

The variation that separates them: unroll the loop with independent accumulators, holding bytes
and dispatch constant while cutting iterations.

| unroll | inter-token @2048 | speedup | if purely iteration-bound |
|---|---|---|---|
| 1 | 47.0 ms | 1.00× | 1× |
| 2 | 40.9 ms | 1.15× | 2× |
| 4 | 35.8 ms | 1.31× | 4× |
| 8 | 33.9 ms | **1.39×** | 8× |

**Both mechanisms are real.** An 8× iteration cut bought 1.39×, so latency is a genuine
component — and it saturated far short of 8×, so it is not the whole story. The residue is
occupancy, which no amount of doing less work per thread can fix.

Worth noting this probe would have been misleading if run alone and read carelessly: "unrolling
helps, therefore it's iterations" is true and incomplete, and a fix built only on it would have
recovered 1.39× instead of 1.5×+.

## The build

One workgroup per output element instead of one thread: 64 threads split the history and reduce
in shared memory. Dispatch goes 14 → 896 workgroups, and per-thread iterations go
`history` → `history / 64`. Both mechanisms at once.

| context | serial | **parallel** | gain |
|---|---|---|---|
| 128 | 35.1 tok/s | **42.7** | +21.8% |
| 512 | 37.9 | **42.7** | +12.8% |
| 1024 | 34.6 | **44.6** | +29.0% |
| 2048 | 28.7 | **43.1** | **+50.4%** |

Parity: greedy output still matches the no-cache path 24/24.

**Decode is now flat in context.** Fitting the same model as before:

```
serial     ms = 26.06 +  3.8908 µs/position
parallel   ms = 23.24 + -0.1541 µs/position
```

The position-dependent term is gone — −0.15 µs/position is indistinguishable from zero at this
sample size. Decode at 2048 now costs what decode at 128 costs, which is the property the KV
cache was supposed to deliver and did not until the kernel that reads it stopped walking the
history one position at a time.

This is a larger win than every M6 item as specified put together, and it came from taking the
mechanism question seriously rather than from a listed optimization.

## Retrospective on the f16 cache

With the serial kernel, f16 bought +3.4% at 2048. With the parallel kernel the position term is
gone entirely, so halving the bytes of a term that costs nothing buys nothing. **The f16 cache
is now purely a memory optimization**: 24 MiB instead of 48 at a 2048 context, at a 4% prefill
cost from the pack dispatches. Still worth it on a phone, still not a speed feature, and now not
a speed feature for a second and better-understood reason.

---

# M7 — the site

## A sampling bug the site found

Building the chat surface put the engine in front of realistic settings for the first time:
temperature 0.7, top-p 0.9, repetition penalty 1.1. Decode measured **85 ms/token** against
the harness's 22. Varying one sampling parameter at a time located it immediately.

| configuration | before | after |
|---|---|---|
| greedy | 26.7 ms | 26.7 |
| temperature only | 26.4 | 24.7 |
| top-p 0.9 only | 26.3 | 28.5 |
| **repetition penalty 1.1 only** | **71.4** | **24.4** |
| all three (the app's defaults) | **84.9** | **28.6** |

**The repetition penalty was 2.7× the cost of everything else in decoding put together.**
`sample.wgsl` applied it in a helper called at the point of reading a logit, and that helper
scanned the entire history for every read. The vocabulary is read about 36 times per token —
max, sum, 32 bisection steps, the draw — so the cost was `vocab × history × 36`. At a
500-token history that is 2.7 billion comparisons to penalize at most 500 tokens.

The fix applies the penalty once, in place, before anything reads a logit: threads split the
history, only the first occurrence of an id applies (so no two threads write the same address
and no atomics are needed), then a `storageBarrier()`. Cost goes from 36 passes over the
vocabulary to one pass over the history. **3.0× on the app's default sampling settings**, and
the penalty is now free — 24.4 ms with it against 26.7 greedy, inside noise.

Parity: `npm run session` 9/9, `npm run parity` 387/387 isolated, `npm test` 252/252. The
greedy path is arithmetically unchanged and reproduces the no-cache path 24/24.

Worth noting where this was found. Every harness in this file measures with `GREEDY`, because
greedy is what parity requires — and greedy skips the penalty entirely. **A cost that only
appears under settings no test uses is invisible to the whole test suite.** The site was the
first thing to run the engine the way a person would.

## What the demo measures against the harness

Same device, same weights, measured through the page rather than through vitest. Median of
three generations, each a full reply.

| configuration | ms/token | tok/s |
|---|---|---|
| as shipped — page motion, inspector profiling, sampled | 29.4 | 34.0 |
| inspector profiling off | 29.2 | 34.3 |
| page motion stopped as well | 30.2 | 33.1 |
| greedy, no motion — the harness's configuration | 29.6 | 33.8 |
| **the harness itself, `npm run ablation` at ctx 512** | **22.0** | **45.5** |

Two null results and one real gap:

- **The inspector costs nothing measurable.** Panels read a published snapshot at 30 Hz and
  per-kernel timing profiles one step in 16. 29.2 against 29.4 ms is noise.
- **Page motion costs nothing measurable either**, which contradicts an earlier reading in
  this session: before the sampling fix, stopping every `requestAnimationFrame` loop moved
  116 ms/token to 100. That gap does not reproduce now, so the earlier 14% was almost
  certainly variance on a much larger number rather than a real compositor tax. Recorded
  because the wrong inference was made first.
- **The page costs about 4 ms/token against the harness**: the inspector's own rolling
  inter-token readout says 26.3 ms while wall clock over a whole reply says 29.4, and the
  harness says 22.0. Per-token tokenizer decode, DOM updates and sharing a GPU with the
  compositor are all in that 4–7 ms. The site says so next to the demo rather than quoting
  the headless figure as if a visitor would see it.

## Per-kernel GPU time, one decode step

From `timestamp-query`, one profiled step in 16, at a short context. **460 dispatches,
23.8 ms of GPU time per token** — 460 rather than 412 because the f16 KV cache adds a pack
dispatch for K and V in each of the 24 layers.

| group | share |
|---|---|
| projection | 61.1% |
| embedding (tied LM head) | 24.7% |
| sample | 9.3% |
| attention | 2.4% |
| rmsnorm | 1.1% |
| mlp (SwiGLU + residual adds) | 0.9% |
| rope | 0.6% |

The tied LM head at 24.7% is the second-largest consumer in the model, from a single dispatch
of 460 — 896 × 151,936 against 896 × 896 for a projection. Attention at 2.4% is the parallel
history reduction doing its job; before that change it was the term that grew with context.

## Memory, live

| | |
|---|---|
| weights (int4, int8 embedding) | 334.9 MiB |
| KV cache at 2048, f16 | 24.0 MiB |
| activation scratch | 773.8 MiB |
| **resident total** | **1132.6 MiB** |

The scratch is the prefill graph, allocated once for the longest prompt it must accept. Two
14 × 2048 × 2048 attention buffers are 470 MiB of it on their own, and it scales with the
square of the context — which is why a mobile-class adapter gets a 1024-token context instead,
cutting those two buffers to 118 MiB. The inspector reports the whole ledger rather than only
the weights, because 335 MiB is the honest number for the weights and not for the process.

## Lighthouse and mobile

`npx lighthouse`, production build served by `vite preview`.

| | performance | accessibility | best practices | SEO |
|---|---|---|---|---|
| desktop | **100** | **100** | **100** | **100** |
| mobile (slow 4G, 4× CPU throttle) | **96** | **100** | **100** | **100** |

Mobile metrics: FCP 1.6 s, LCP 2.5 s, TBT 100 ms, CLS 0.

Three changes got there. The initial JavaScript bundle is **8.24 kB** (3.67 kB gzipped): the
engine, kernels, tokenizer and inspector are behind the load button, and Three.js — 517 kB, the
largest thing on the page after the model — is imported after first paint. Fonts are
self-hosted latin subsets with `preload`, rather than Google's render-blocking stylesheet on two
extra connections. And the contrast audit caught `--muted` (#63637C, 3.5:1 on the background)
being used for eyebrows, table headers and the footer — content, not decoration, which is
exactly the line the design system draws. Those moved to `--dim`.

Checked on an iPhone 14 Pro profile: no horizontal overflow at 393 px, and the pinned pipeline
section drops its pin below 760 px, because holding the viewport for seven screens of scroll on
a phone is hostile rather than controlled.

---

# M7 deploy — residency, and the first load numbers that were never real

## Prefill scratch was sized to the longest prompt it might ever see

`npm run residency` lists live allocations rather than inferring them. The claim to check was
that peak residency is dominated by the two `heads × maxSeq × maxSeq` attention buffers.

| buffer | requested | capacity |
|---|---|---|
| `scores` | 224.0 MiB | **256.0 MiB** |
| `attn_weights` | 224.0 MiB | **256.0 MiB** |
| `mlp_gate` / `mlp_up` / `mlp_silu_mul` | 38.0 MiB each | 64.0 each |
| eight `seq × hidden` buffers | 7.0 MiB each | 8.0 each |

**Confirmed: 512 MiB of 773.8 is the two attention buffers.** A second finding came free —
**151.1 MiB of the 773.8 is size-class rounding**, because the pool rounds to powers of two and
224 MiB rounds to 256.

This is the M6 over-dispatch bug in a different resource: sizing to a build-time maximum rather
than the extent actually in play. The fix is the same shape — prefill runs in chunks of 256
queries against the whole cached prefix, so the score tensors are `heads × 256 × context`.

| | scratch | resident total |
|---|---|---|
| one shot, sized to 2048 | 773.8 MiB | 1132.6 MiB |
| **chunked at 256** | **99.1 MiB** | **458.0 MiB** |

## The prediction about speed was wrong twice, and the answer is "no change"

Prefill time fits `ms = 0.87·N + 1.86e-4·N²`, which puts 30.4% of a 2048-token prefill in the
quadratic term. Chunking replaces the full square with a triangle, so the fit predicted a **13%
speedup**. The first measurement after building said **10% slower**. Both were wrong, and the
second was wrong for an embarrassing reason: it compared a number from this session against one
recorded in a previous session, and run-to-run variance on this measurement is about 8%.

Measured properly — one session, sweeping chunk size, with `chunk = 2048` as the control that
reproduces the old one-shot path exactly:

| chunk | prefill @2048 | vs one-shot | scratch |
|---|---|---|---|
| 128 | 2975 ms | +0.6% | 50.9 MiB |
| **256** | **2894 ms** | **−2.1%** | **99.1 MiB** |
| 512 | 2925 ms | −1.1% | 195.5 MiB |
| 1024 | 2935 ms | −0.8% | 388.3 MiB |
| 2048 (control) | 2957 ms | — | 773.8 MiB |

**Flat.** A 2.7% spread across a 16× range of chunk sizes, with no ordering — it is noise.
Chunking costs nothing and returns 87% of the scratch. The predicted 13% speedup did not appear
because the model over-credited the square-to-triangle saving: the masked half of the square
already skipped its dot product and paid only a store.

Correctness: greedy output is identical at chunk 256, 64 and 37 (a ragged last chunk), and the
attention kernels have unit tests asserting a strip at `queryBegin > 0` reproduces the matching
rows of the full square. Parity 387/387, `npm run session` 10/10.

**The mobile 1024-token context is withdrawn.** It existed because scratch was 773.8 MiB; at
99.1 MiB, halving the context would save about 58 MiB of a 458 MiB footprint and cost half the
usable conversation.

## Every load number in this file before now was measured over loopback

M2 said 2.51 s cold, and said in the same paragraph that this was the floor rather than the
experience. It was worse than that: loopback hid a design fault, because it made a round trip
free.

The `.enargeia` container stores three arrays per quantized tensor — packed values, scales,
zero points — so the model is **630 byte ranges**, and the loader fetched them **one at a time**.
It also never touched the Cache API on the quantized path, which the fp32 path had used since
M2. Measured against R2 through a custom domain:

| | requests | cold | warm |
|---|---|---|---|
| as first deployed | 630 serial | **78.3 s** | 75.7 s (no caching at all) |
| + 8 concurrent, + Cache API | 630 | 58.8 s | **1.5 s** |
| + concurrency 24 | 630 | 50.7 s | — |
| **+ coalesced into 16 MiB spans** | **17** | **14.6 s** | **1.4 s** |

Concurrency alone stalled at 7 MB/s against a link measured at 25–30 MB/s from the same browser,
which ruled out bandwidth, the Cache API (identical with `noCache`), and the browser. The
progress readout was the clue: 178 MB arrived in the first 10 seconds and the remaining 173 took
44 — a large-tensor phase followed by a long tail of small ones. Coalescing adjacent ranges into
16 MiB spans turns 630 requests into 17 and reaches **28.7 MB/s**, matching raw fetch throughput.

**Cold first load, R2 over Cloudflare's CDN: 14.6 s from clicking Load to a first token**, of
which 0.11 s is the model producing that token. Second visit: **1.4 s and two requests**, both
of them header reads.

A third bug fell out of the same measurement. The progress readout summed *per-request*
durations to compute a rate, which is correct for a serial loader and badly wrong for a
concurrent one — eight overlapping one-second requests sum to eight seconds of "network time".
A visitor was told **16 minutes remained on a 54-second download**. It now measures wall clock,
with a regression test.

## What the site says about memory now

The headline number was "335 MiB resident", and live residency was 1132.6. Both figures were
real and the sentence joining them was not: 334.9 MiB is the weights, and the fp32 figure it is
compared against is also weights-only. The page now states the weights, the live total, and the
itemisation:

| | |
|---|---|
| weights (int4, int8 embedding) | 334.9 MiB |
| KV cache at 2048, f16 | 24.0 MiB |
| activation scratch | 99.1 MiB |
| **live residency** | **458.0 MiB** |
| the same weights in fp32 | 1884.6 MiB |

## Deployed

`enargeia.dev`, Cloudflare Pages, weights from R2 at `models.enargeia.dev`. Measured against
the live site, not a preview server:

| | |
|---|---|
| cold first load → first token | **15.4 s** |
| second visit → first token | **1.5 s** |
| time to first token, engine warm | **0.20 s** |
| decode, sampled, foreground tab | **38.2 tok/s** |
| live residency | 458.0 MiB |

Per-kernel share on the live site, from `timestamp-query`: projection 60.8%, tied LM head
24.7%, sample 9.3%, attention 2.6%, rmsnorm 1.1%, mlp 1.0%, rope 0.6%.

Lighthouse against the live URL, mobile preset, **median of three runs** — `npm run lighthouse`:

| | median | runs |
|---|---|---|
| performance | **98** | 100, 98, 95 |
| accessibility | **100** | |
| best practices | **100** | |
| SEO | **100** | |
| FCP | 1717 ms | 823, 1717, 1735 |
| TBT | 14 ms | 14, 0, 14 |

The median matters here. Single runs of the same deployment scored **88 and 84**, and speed
index ranged from 1217 ms to 4208 across three runs — a live site over a real network is far
noisier than a preview server, and one run of it is not a measurement. That is the third time
in this project that a single sample nearly became a conclusion: the M6 f16 prediction, the
"chunking is 10% slower" reading earlier in this milestone, and this. The harness now takes
a median by default.

Two deployment faults worth recording, both invisible until the site was actually served:

- **Pages had no build step configured**, so it published the repository as-is and the browser
  refused `/src/main.ts` as `video/mp2t`. The page had been a blank `#app` on every deploy since
  the project was created; nobody noticed because until now it was a scaffold.
- **`vite build` copies `public/` wholesale**, so `dist/` carried every `.enargeia` file in
  `public/models/` — up to 2 GB, against a Pages limit of 25 MiB per file.

---

# M7 — cross-browser verification

## The pinned section, and what "broken in Safari" turned out to mean

Reported as a Safari bug: the stage heading clipped behind the sticky masthead, and about a
viewport of empty space below the content. Both reproduce exactly. **Neither is a Safari bug.**

Driving the same page in WebKit 26.5 and Chromium and reading the pinned element's box at four
scroll positions gives identical numbers in both engines:

| scroll | section top | `position` | active stage |
|---|---|---|---|
| pin start | 0 | static | embedding |
| +400 | −400 | static | embedding |
| +900 | −900 | static | rmsnorm |
| +1800 | −1800 | static | attention |

The element never becomes `fixed` and its top marches off the screen — the pin was never
holding, in any browser. The cause: `mountPipeline` pinned the element it was handed, which is
the inner `<div id="pipeline-stages">`, not the `<section>` around it. So the section heading
scrolled away while the stage grid stuck under a 53 px sticky masthead that covered its first
line, and because the grid is 284 px tall in an 847 px viewport, the rest was empty.

It looked correct in exactly one position — the first frame of the pin — **which is the frame
every screenshot in this project had been taken at**, because every check scrolled the section
into view and shot it immediately.

Fixed by pinning the section, starting at `top ${mastheadHeight()}px` rather than `top top`,
and giving the pinned section `min-height: calc(100svh - var(--masthead-h))` with its content
centred. After:

| scroll | section top | `position` | active stage |
|---|---|---|---|
| pin start | 52 | fixed | embedding |
| +400 | 52 | fixed | rmsnorm |
| +900 | 52 | fixed | rope |
| +1800 | 52 | fixed | sample |

Identical in WebKit and Chromium. The two no-pin paths — `prefers-reduced-motion` and widths
under 760 px — create no pin and no spacer, and lay all seven stages out as a plain list.

## What else the first non-Chromium sweep found

`npm run sweep` — every section, five widths, WebKit, with automatic checks for the failure
classes a screenshot review misses.

| | |
|---|---|
| **Both loading bars rendered full before any download** | `.bar i` is `display: block` with no width, which resolves to `auto` — the full width of the bar. Present in every browser since the loader was written, and invisible in every screenshot because they were all taken after clicking Load. |
| **Horizontal overflow at 320 px** | The masthead's five nav links plus the wordmark forced `scrollWidth` to 381. The masthead now wraps, and `--masthead-h` is re-measured on resize so the pin start follows it. |
| **Nav and footer links were 16–21 px tall** | Under the 24 px minimum in WCAG 2.5.8. Padded to clear it. The two remaining flagged links are inline in sentences, which that criterion exempts. |
| **"The panel on the right"** | False at narrow widths, where the inspector is below the chat. |
| **"Load the model — 351 MB" beside "334.9 MiB"** | Two numbers for one quantity on one screen. Both were right; only one is now shown. |
| **A duplicate heading** | The loader repeated the section's heading directly beneath it. |

Anchor links also landed their targets behind the sticky masthead — fixed with
`scroll-padding-top` on `html`, which is a different symptom of the same missing offset.

## A cache hazard, self-inflicted

The first sweep of the live site reported a module served as `text/html`, while `curl` got the
same URL as `application/javascript`. Both were true: the sweep had requested the asset in the
window between the deploy completing and the file propagating, Cloudflare cached the SPA
fallback against that URL, and afterwards the edge served two different responses for one path.
Redeploying does not clear it and the OAuth token cannot purge; escaping the URL by changing the
chunk's content hash does.

`npm run sweep` now preflights every module the document references with a plain fetch and
refuses to open a browser unless they are all served as JavaScript — so a stale asset is
reported as a deployment problem instead of being requested and cached.

## Scope of the verification

WebKit 26.5, which is Safari 26's engine, driven through Playwright. **Not Safari itself**:
`safaridriver` requires "Allow remote automation" in Safari's Developer settings, and
`screencapture` requires Screen Recording permission — both are user-granted and neither is
enabled. `tools/safari.mjs` is a WebDriver client ready to run against the real browser once
that box is ticked. The finding that mattered most does not depend on it: the reported bug was
never engine-specific and reproduces in Chromium.

## The favicon, and the last of the scaffold

`public/favicon.svg` was still Vite's lightning bolt, and `public/icons.svg` — a Bluesky icon
sprite from the same template, referenced by nothing — was next to it. Both are gone.

The replacement is the residual stream falling through layers: three bars, one stroke, two hues
from the ends of the kernel spectrum. Chosen by rasterizing eight candidates to real 16×16 PNGs
and comparing them in a mock browser tab against light and dark chrome, because **three designs
that were unambiguous at 512 px were unreadable at 16**, and the favicon is only ever seen at 16.

Shipped alongside: `apple-touch-icon.png` at 180 (iOS ignores the manifest for home-screen
icons and screenshots the page without it), `icon-192`, `icon-512`, a maskable 512 with the mark
inside the 80% safe circle, a 32px PNG for clients that ignore SVG favicons, and
`site.webmanifest`. All generated by `tools/make-icons.mjs` from one definition.

Lighthouse against the live site after the change: **median 97** over three runs (97, 96, 99),
unchanged within noise.

`/apple-touch-icon.png` briefly served `text/html` after the deploy — Safari probes that path on
its own without the page referencing it, so a request during propagation cached the SPA fallback
against it, exactly as had happened to a JavaScript chunk. It revalidated and healed on its own.
The sweep's preflight now checks every referenced file, plus the icons named in the manifest,
against the content type each should have.
