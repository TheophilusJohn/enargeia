---
name: enargeia-parity
description: The numerical correctness gate for the Enargeia inference engine — golden activation fixtures dumped from PyTorch, layer-by-layer comparison, error thresholds, and determinism checks. Use this before and after every optimization, whenever a kernel is added or changed, whenever generated text looks plausible but subtly wrong, whenever output differs between runs or devices, and whenever setting up or regenerating test fixtures. Also use when the task involves quantization quality, perplexity measurement, or deciding whether a performance change is safe to keep, even if parity is not mentioned by name.
---

# Parity harness

The failure mode this exists to catch: a kernel that is 99% correct. The model still
produces fluent, confident English. Nothing crashes. Perplexity drifts slightly and text
quality degrades in a way nobody notices for weeks, by which point a dozen commits sit on
top of the bug.

Numerical comparison against a reference is the only way to catch this. Everything else —
reading the output, checking it "looks fine" — does not work and should not be offered as
evidence that a change is safe.

## The fixtures

`tools/dump_reference.py` runs the model in PyTorch on a fixed prompt and writes every
intermediate activation to `test/fixtures/reference.bin` with a JSON sidecar giving name,
shape, dtype, and byte offset.

Dump at every boundary, not just the final logits:

- token embeddings
- per layer: post-RMSNorm, Q, K, V, post-RoPE Q and K, attention weights, attention
  output, post-projection, post-residual, MLP gate, MLP up, post-SiLU-mul, MLP down,
  post-residual
- final norm
- logits

The prompt is fixed and lives in `test/fixtures/prompt.txt`. Do not change it casually —
regenerating fixtures invalidates every historical comparison. If it must change, say so
explicitly and regenerate in a standalone commit that touches nothing else.

## Thresholds

Compare with both max absolute and max relative error. Relative error alone is misleading
near zero; absolute alone is misleading for large activations.

| Path | max abs | max rel |
|---|---|---|
| fp32 kernels | 1e-4 | 1e-4 |
| f16 accumulation | 5e-3 | 5e-3 |
| int4 weights | 5e-2 | 8e-2 |

The int4 thresholds are loose by necessity — quantization *is* lossy, and the harness
cannot distinguish expected loss from a bug at that tolerance. So int4 gets a second check
that fp32 does not need: perplexity on a held-out set, compared against the fp32 baseline.
A correct int4 implementation of this model costs a small perplexity increase. A broken one
costs much more. Record the number in `BENCH.md`; a sudden jump is the signal.

## Running it

```bash
npm run parity              # full layer-by-layer, prints an error table
npm run parity -- --layer 7 # isolate one layer once the table shows where it breaks
npm run parity -- --strict  # fail on the first threshold breach
```

Read the table from the top. The first layer that exceeds threshold is where the bug is —
everything after it is downstream contamination and tells you nothing. Do not investigate
layer 19 because its error is largest; investigate the first one that went red.

## When parity fails

1. Find the first failing stage. Ignore everything downstream.
2. Isolate that kernel and run its unit test against the CPU reference. If the unit test
   passes but parity fails, the bug is in dispatch geometry, buffer offsets, or the graph
   wiring — not the shader.
3. Shrink the input until you can read every number. A 4×4 matmul makes transpose and
   indexing errors obvious in a way a 896×896 one never will.
4. If the error appears only sometimes, it is a barrier. See `enargeia-kernels`.

## Determinism

Greedy decode on the fixed prompt must produce byte-identical token IDs across runs, and
across devices with the same feature set.

```bash
npm run determinism         # 20 runs, compares token ID sequences
```

Non-determinism is a correctness bug, not noise, and the cause is nearly always a missing
`workgroupBarrier()` or a race on shared memory. Do not retry until it passes and move on.
Do not add tolerance to make the check green. A race that shows up one run in twenty on
your machine will show up far more often on someone else's.

## The gate

**No optimization lands without parity passing before and after.**

Optimizations are exactly the changes most likely to introduce subtle numerical error —
fusing kernels, changing accumulation order, dropping to f16, restructuring tiles. They are
also the changes whose benefit is easiest to measure and hardest to give up, which makes
them the ones people are most tempted to keep despite a red test.

If a change improves throughput and breaks parity, it does not land. Report the speedup
and the error together and let the tradeoff be an explicit decision, not a quiet one. If a
threshold genuinely needs to move, that is a separate commit with a written justification —
never bundled into the change that made it necessary.

## Recording results

Every performance change appends a row to `BENCH.md`:

```
| change | tok/s | delta | parity | perplexity |
|---|---|---|---|---|
| f16 accumulation | 28.1 | +13% | pass (2.1e-3) | 11.84 |
```

Include changes that made things slower. An ablation table with only wins is a marketing
document; one that shows what failed is evidence of method, and it is what makes the
benchmark section of the site worth reading.
