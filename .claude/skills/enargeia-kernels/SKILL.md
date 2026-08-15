---
name: enargeia-kernels
description: Conventions and correctness rules for writing WGSL compute shaders in the Enargeia inference engine — matmul, quantized matmul, RMSNorm, RoPE, attention, softmax, SiLU, sampling. Use this whenever writing, editing, reviewing, optimizing, or debugging any .wgsl file or its TypeScript dispatch wrapper, whenever touching workgroup sizes, tiling, shared memory, or barriers, and whenever a kernel produces wrong or non-deterministic numbers. Also use when the task mentions GPU kernels, compute shaders, dequantization in the hot loop, or making inference faster, even if WGSL is not named explicitly.
---

# Writing kernels for Enargeia

Every kernel here is hand-written. There is no cuBLAS to fall back on and no framework
to hide a mistake. A kernel that is subtly wrong still produces fluent text, so
correctness has to be established numerically before performance work begins.

## Before writing any kernel

Write the CPU reference first, in TypeScript, in `test/reference/`. It should be the
slowest, most obviously correct implementation you can write — nested loops, no cleverness.
This is what the kernel gets compared against, and writing it first forces you to pin down
the semantics before the GPU makes them hard to inspect.

Then write the kernel. Then the unit test. Then, and only then, optimize.

## File layout

One kernel per pair of files:

```
src/kernels/matmul_q4.wgsl      the shader
src/kernels/matmul_q4.ts        typed params, bind group layout, dispatch fn
```

The `.ts` wrapper owns the bind group layout and dispatch geometry. Nothing outside
`src/kernels/` constructs a `GPUComputePipeline` — that is the wrapper's job, and it keeps
pipeline creation in one place where the cache can see it.

## WGSL conventions

**Bindings** in fixed order: inputs, then outputs, then uniforms. Read-only storage
buffers get `var<storage, read>`; never `read_write` on something you only read, because
the driver uses that to reason about aliasing.

```wgsl
@group(0) @binding(0) var<storage, read>       x:      array<f32>;
@group(0) @binding(1) var<storage, read>       w:      array<u32>;
@group(0) @binding(2) var<storage, read>       scale:  array<f16>;
@group(0) @binding(3) var<storage, read_write> out:    array<f32>;
@group(0) @binding(4) var<uniform>             dims:   Dims;
```

**Always bounds-check.** Dispatch geometry rounds up to whole workgroups, so the last
workgroup runs threads that are out of range. Every kernel begins with a guard:

```wgsl
if (row >= dims.m || col >= dims.n) { return; }
```

An early `return` before a `workgroupBarrier()` is undefined behaviour — barriers must be
in uniform control flow. When a kernel uses shared memory, do not return early; mask the
*write* instead and let every thread reach every barrier.

**Workgroup size** is `@workgroup_size(16, 16)` for 2D matmul kernels and
`@workgroup_size(256)` for 1D reductions, unless a benchmark says otherwise. Do not
introduce a new size without a measurement showing it wins.

**No f16 without a guard.** `enable f16;` requires the `shader-f16` feature. Every kernel
that uses it needs an fp32 sibling, selected at pipeline-creation time from the device's
reported features. Roughly a third of real devices lack it.

## Tiled matmul — the shape everything else follows

The naive version reads a full row and column from global memory per output element. The
tiled version stages sub-blocks into `var<workgroup>` shared memory so each loaded value
serves the whole workgroup.

```wgsl
var<workgroup> tileA: array<f32, 256>;   // 16x16
var<workgroup> tileB: array<f32, 256>;

for (var t = 0u; t < numTiles; t = t + 1u) {
    tileA[lid.y*16u + lid.x] = loadA(...);   // guarded, zero-fill out of range
    tileB[lid.y*16u + lid.x] = loadB(...);
    workgroupBarrier();                       // stage complete

    for (var k = 0u; k < 16u; k = k + 1u) {
        acc += tileA[lid.y*16u + k] * tileB[k*16u + lid.x];
    }
    workgroupBarrier();                       // consumed; safe to overwrite
}
```

Two barriers per tile, not one. The second is the one people forget — without it, a fast
thread starts overwriting shared memory for the next tile while a slow thread is still
reading the current one. This produces small, intermittent, device-dependent errors and is
the single most common bug in this codebase's history.

## Quantized matmul — the core kernel

Weights arrive as int4, eight nibbles packed per `u32`, with an fp16 scale and zero-point
per block of 64 along the reduction axis.

Dequantize inside the accumulation loop, in registers. Never write dequantized weights to
a buffer — the entire benefit is moving fewer bytes across the memory bus, and
materializing fp32 weights hands that benefit straight back.

```wgsl
let blk    = i / 64u;
let packed = w[(i >> 3u)];
let nib    = (packed >> ((i & 7u) * 4u)) & 0xFu;
let val    = (f32(nib) - f32(zero[blk])) * f32(scale[blk]);
acc += val * x[i];
```

Block-wise rather than per-tensor because weight distributions have outliers. A single
scale for a whole matrix is set by its largest magnitude, which crushes everything else
toward zero. Blocks of 64 keep each scale local to weights of similar magnitude. If asked
to "simplify" to per-tensor, refuse and explain — the perplexity damage is severe and
shows up as subtly worse text rather than an obvious failure.

## Prefill and decode need different kernels

Prefill is matrix-by-matrix: high arithmetic intensity, compute-bound, wants large tiles
and deep unrolling.

Decode is matrix-by-*vector*: every weight is read once and used once, arithmetic intensity
near 1, entirely memory-bound. Large tiles do nothing here. What matters is coalesced reads
and keeping the memory pipeline saturated.

A kernel tuned for one will underperform on the other. Keep them separate and name them
`_prefill` and `_decode` rather than adding a branch.

## Before optimizing: confirm the mechanism

Vary the suspected cause independently and check that the effect moves. If you think a cost is
bandwidth, halve the bytes and confirm the time drops. If you think it is iteration count,
unroll and confirm. If you think it is occupancy, change the dispatch and confirm. Do this
*before* building the optimization, not after it disappoints.

A term measured correctly does not tell you what the term is. Time that scales with context
could be bytes, iterations, or occupancy; time that scales with size could be arithmetic or
launch overhead. These look identical in the measurement that motivated the work and differ
completely in what fixes them.

**This check has caught three wrong mechanism assumptions in this codebase:**

- **the M5 footprint claim** — int4 prefill was reported slower than fp32 below ~500 tokens and
  blamed on the tile footprint. Both had been measured through the same over-dispatched
  attention. Sizing dispatch to the real sequence made int4 4.4× *faster* at the length where
  it supposedly lost.
- **the over-dispatch diagnosis** — geometry baked from `maxSeq` launched 5.5M workgroups per
  prefill at any prompt length. Found only because a 32-token prompt taking 1518 ms did not fit
  any story about the kernel, and the geometry was checked rather than assumed.
- **the f16 cache prediction** — decode time decomposed cleanly into a fixed term plus 5.87
  µs/position, and the position term was assumed to be KV bandwidth. Halving the bytes returned
  a sixth of the predicted gain. It was iterations and occupancy; the real fix was a workgroup
  reduction, worth +50% where f16 was worth +3%.

Each time the wrong mechanism was plausible, consistent with the data in hand, and would have
been caught in minutes by changing the suspected cause and watching the effect.

## Debugging

There is no printf. To inspect intermediate values, bind a scratch `read_write` buffer,
write the values you want, read them back once outside the loop. Delete the scratch buffer
before committing — a debug binding left in a hot kernel costs bandwidth.

Diagnostic order when a kernel is wrong:

1. **NaN or Inf** → uninitialized shared memory, or a division without an epsilon guard.
   Check that out-of-range tile loads write zero rather than leaving stale values.
2. **Correct for small inputs, wrong for large** → a bounds check is missing, or a `u32`
   index overflowed. `151936u * 896u` exceeds a `u32`; compute large offsets carefully.
3. **Slightly wrong, everywhere, consistently** → an indexing or transpose error. Compare
   against the CPU reference on a 4×4 input where you can read every number.
4. **Slightly wrong, intermittently, only on some devices** → a missing barrier. This is
   almost always the answer.
5. **Correct output, wrong performance** → check the dispatch geometry before touching the
   shader. Dispatching one workgroup per output element is a common and invisible mistake.

## Definition of done

A kernel is finished when:

- The CPU reference exists and the unit test compares against it on random inputs
  including edge shapes — sequence length 1, dimensions that are not multiples of the tile
  size, and the maximum supported context.
- The parity harness still passes end to end. See `enargeia-parity`.
- Running it twice on the same input gives byte-identical output.
- If it replaced an earlier version, the benchmark delta is recorded in `BENCH.md` —
  including when the change made things slower, because that is the useful half of an
  ablation table.
