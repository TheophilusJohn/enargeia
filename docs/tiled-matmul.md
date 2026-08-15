# Tiled matmul, stage 1

For someone who has read `matmul_naive.wgsl` and wants to know what `matmul_tiled.wgsl`
actually changed. Measured on an Apple M2 under headless Chrome: **219.4 → 517.5 GFLOP/s,
2.36×**, at 1024³ fp32.

## What did not change

Almost everything. Same bindings in the same order, same `@workgroup_size(16, 16)`, same
dispatch geometry, one thread per output element, same `acc` in a register, same
`c[row * n + col] = acc` at the end. Both kernels walk k in ascending order one term at a
time, so at 129×65×200 their outputs are **byte-identical** — not required, but true, and
worth keeping true because it means any future mismatch is a tiling bug rather than a
difference in floating-point association.

What changed is where the operands are read from.

## The reuse argument

The naive kernel computes one output element by streaming a full row of A and a full column
of B out of global memory:

```wgsl
for (var i = 0u; i < dims.k; i = i + 1u) {
    acc = acc + a[aBase + i] * b[i * dims.n + col];
}
```

Two global loads per multiply-add. Nothing is remembered between output elements, so the
thread computing `C[0][0]` and the thread computing `C[0][1]` both read the entire row
`A[0][*]`, independently. Across the whole matmul, every element of A is fetched n times
and every element of B is fetched m times.

The tiled kernel makes the workgroup cooperate. Its 256 threads compute a 16×16 block of C,
and that block needs a 16×k strip of A and a k×16 strip of B — the two strips that intersect
at the block. Instead of every thread fetching what it needs, the workgroup walks the strips
in 16-wide steps, staging one 16×16 block of each into shared memory at a time:

```wgsl
var<workgroup> tileA: array<f32, 256>;
var<workgroup> tileB: array<f32, 256>;
```

Each thread loads exactly one element of each tile — 256 threads, 256 slots — and then
reads 16 values out of each tile in the inner loop. The value a thread loaded is read by
itself and 15 others.

Concretely: `tileA[ty * 16 + i]` is read by every thread sharing that `ty`, which is 16
threads (`tx` = 0..15). `tileB[i * 16 + tx]` is read by every thread sharing that `tx`,
again 16. One global load, sixteen consumers. That is the entire trick.

## Why tileA indexes k with tx and tileB with ty

This is the part that looks arbitrary and is not.

```wgsl
let aCol = t * TILE + tx;   // A's k index comes from tx
let bRow = t * TILE + ty;   // B's k index comes from ty

tileA[ty * TILE + tx] = a[row * dims.k + aCol];   // A[row][k]
tileB[ty * TILE + tx] = b[bRow * dims.n + col];   // B[k][col]
```

Look at what the output element needs:

```
C[row][col] = Σ  A[row][i] · B[i][col]
              i
```

A is subscripted `[row, i]` and B is subscripted `[i, col]`. The reduction index `i` is
A's **second** subscript and B's **first**. So the 16×16 block of A the workgroup needs is
`rows × k`, and the 16×16 block of B is `k × cols`. The k axis runs horizontally through
one and vertically through the other.

The workgroup has one 16×16 grid of threads and has to cover both blocks. There is no
choice: whichever local axis is spent on k for A must be spent on the *other* thing for B.

```
        tileA (rows × k)              tileB (k × cols)

        k = t*16 + tx  →              col = wg.x*16 + tx  →
      ┌────────────────┐            ┌────────────────┐
row   │                │      k     │                │
 ↓    │   thread       │      ↓     │   thread       │
ty    │   (tx, ty)     │     ty     │   (tx, ty)     │
      └────────────────┘            └────────────────┘
      tx walks k                    ty walks k
      ty walks the output row       tx walks the output column
```

Both loads stay coalesced, which is the other reason this assignment and not some
transposed variant. `tx` is the fastest-varying thread index, and in both loads `tx` lands
on the fastest-varying memory axis: `a[row * k + (t*16 + tx)]` is 16 consecutive floats, and
`b[(t*16 + ty) * n + (wg.x*16 + tx)]` is likewise 16 consecutive floats. Swap the roles and
one of the two becomes a strided gather.

Then the inner loop reads each tile along k — row `ty` of tileA, column `tx` of tileB:

```wgsl
for (var i = 0u; i < TILE; i = i + 1u) {
    acc = acc + tileA[ty * TILE + i] * tileB[i * TILE + tx];
}
```

## Why both barriers are load-bearing

```wgsl
    // ... stage both tiles ...
    workgroupBarrier();          // 1

    for (var i = 0u; i < TILE; i = i + 1u) { ... }

    workgroupBarrier();          // 2
}
```

**Barrier 1 — the tile is complete.** Thread `(0, 0)` reads `tileA[0..15]`, but it only
wrote `tileA[0]`. The other 15 slots were written by 15 other threads. Without this
barrier a thread can reach the inner loop before its neighbours have stored their values,
and read a slot that still holds whatever was there before.

**Barrier 2 — the tile is finished with.** This is the one people leave out, because with
only barrier 1 the kernel is correct on most runs, on most hardware, most of the time. The
loop body ends and iteration `t+1` immediately overwrites `tileA` and `tileB`. A thread that
finishes its inner loop early will start staging tile `t+1` into slots that a slower thread
in the same workgroup is still reading for tile `t`. The slow thread then multiplies a value
from the wrong position in k.

The failure mode is a handful of elements off by a small amount, intermittently,
differently on different devices — the kind of bug that produces fluent text and a slightly
worse perplexity and survives for weeks. `test/helpers/determinism.ts` exists specifically
for it: 20 runs dispatched concurrently at 512³, every result compared against the first on
bit patterns. That is the test that catches a missing barrier 2. Nothing else in the suite
would.

Note also that neither barrier may sit inside a conditional. That is why the kernel has **no
early return** — out-of-range threads run the whole loop, reach both barriers every
iteration, and are masked only at the final write. Returning early instead would leave the
surviving threads waiting at a barrier that the departed threads will never arrive at, which
WGSL declares undefined behaviour rather than a hang you could debug.

## What breaks without the zero-fill

```wgsl
if (row < dims.m && aCol < dims.k) {
    tileA[ty * TILE + tx] = a[row * dims.k + aCol];
} else {
    tileA[ty * TILE + tx] = 0.0;      // ← this branch
}
```

The tempting simplification is to skip the store when out of range. It is wrong, and here is
exactly how.

WebGPU zero-initializes workgroup memory at the start of a dispatch, so the *first* tile
iteration would be fine. The problem is every iteration after it: shared memory is not
cleared between iterations of the `t` loop, so a skipped store leaves the **previous tile's
value** sitting in the slot, and the inner loop multiplies it in as though it were part of
this tile.

Take k = 40 with TILE = 16. There are three tile iterations: k 0–15, k 16–31, and k 32–39.
The last one is ragged — only 8 of the 16 columns are real. On that iteration, threads with
`tx` = 8..15 have `aCol` = 40..47, past the end of the reduction. With the zero-fill they
contribute `0 · something = 0` and the sum is right. Without it, they contribute
`A[row][24..31] · B[24..31][col]` — the tail of the *previous* tile, added a second time.
Every output whose k is not a multiple of 16 comes out wrong, and consistently so; it would
sail past a determinism check and be caught only by the CPU reference.

That is why `17×17×40` is in the test suite. It spans three tiles with a ragged last one,
which is the smallest shape that exercises this.

The `row < dims.m` and `col < dims.n` halves of those guards are a different matter. Trace
who reads what: a slot in row `ty` of tileA is only ever read by threads with that same
`ty`, and those threads all share the same `row` — so if the row is out of range, every
thread that could see the garbage is masked at the write anyway. The same argument holds for
tileB and `tx`/`col`. Those guards are defensive rather than load-bearing today. Keep them:
they cost nothing, they stop an out-of-bounds read from being silently clamped, and stages 2
and 3 change the thread-to-output mapping, at which point the argument above stops holding.

## The numbers

Per output element, at k = 1024:

| | global loads | shared loads | flops |
|---|---|---|---|
| naive | 2k = 2048 | 0 | 2k = 2048 |
| tiled | 2k/16 = 128 | 2k = 2048 | 2k = 2048 |

Whole matmul at 1024³, total flops 2.15 GFLOP either way:

| | bytes of global load traffic | arithmetic intensity |
|---|---|---|
| naive | 2·m·n·k·4 = **8.59 GB** | **0.25 flops/byte** |
| tiled | (m·n/256)·(k/16)·512·4 = **537 MB** | **4.0 flops/byte** |

A 16× reduction in bytes moved, for a 2.36× speedup. The gap is the interesting part.

**Why not 16×?** Because the naive kernel was never actually moving 8.59 GB from DRAM. At
9.60 ms per iteration, 8.59 GB would be 895 GB/s of sustained traffic — far above any M2
variant's memory bandwidth. The redundant reads were mostly hitting cache: the row of A a
thread streams is the same row its 15 neighbours just streamed, so it is hot. Tiling is not
converting DRAM traffic into nothing. It is converting cache hits into shared-memory reads,
and the win is the difference between those two, not the difference between DRAM and shared
memory.

Look at the table again with that in mind: the *load instruction count* barely moved. Naive
does 2048 global loads per output; tiled does 128 global plus 2048 shared, which is more
total loads. What improved is the class of memory each one touches. A 2.36× return on that
is a reasonable return, and it is the honest reason tiling helps here rather than the
textbook bandwidth story.

**What is still on the table.** 517 GFLOP/s is roughly 14% of an M2's fp32 peak. The kernel
is now limited by shared-memory bandwidth and by the fact that each thread does one
multiply-add per two shared loads — the same ratio the naive kernel had against global
memory, one level down the hierarchy. That is what thread coarsening addresses: give each
thread several outputs so the values it pulls from shared memory get reused in registers.
Stages 2 and 3.

## Reading list in this repo

- `src/kernels/matmul_tiled.wgsl` — the shader, commented at each of the points above
- `src/kernels/matmul_shared.ts` — bind group layout and shape validation, shared by every
  matmul variant so they cannot drift apart on what counts as a legal shape
- `test/kernels/matmul_tiled.test.ts` — CPU-reference comparison, edge shapes, bounds
- `test/helpers/determinism.ts` — the race hunter, and why it is built the way it is
- `BENCH.md` — measurements, including the ones that did not help

---

# Stage 2: four outputs per thread

`matmul_tiled4.wgsl`. Same tiling idea, one thread now owning a 4×1 column strip — four
output rows, one column, four accumulators in registers. Measured **543.7 → 869.4 GFLOP/s,
1.60×** over stage 1, and 3.96× over the naive baseline.

## Why coarsening helps

Look at what stage 1's inner loop costs per unit of arithmetic:

```wgsl
for (var i = 0u; i < TILE; i = i + 1u) {
    acc = acc + tileA[ty * TILE + i] * tileB[i * TILE + tx];
}
```

Two shared-memory loads, one multiply-add. That is a 2:1 load-to-arithmetic ratio — exactly
the ratio the naive kernel had against *global* memory. Stage 1 did not remove the
bottleneck, it moved it one level down the hierarchy. The kernel is no longer waiting on
global memory; it is waiting on shared memory instead.

Stage 2 attacks that ratio directly. A thread computes four outputs that share a column, so
they all need the same value of B and differ only in which row of A they use:

```wgsl
for (var i = 0u; i < TILE_K; i = i + 1u) {
    let bVal = tileB[i * TILE_N + tx];                       // one load
    acc0 = acc0 + tileA[(ty +  0u) * TILE_K + i] * bVal;
    acc1 = acc1 + tileA[(ty + 16u) * TILE_K + i] * bVal;
    acc2 = acc2 + tileA[(ty + 32u) * TILE_K + i] * bVal;
    acc3 = acc3 + tileA[(ty + 48u) * TILE_K + i] * bVal;
}
```

`bVal` is loaded once into a register and multiplied into all four accumulators. Five shared
loads, four multiply-adds: **1.25 loads per MAC instead of 2.0**, a 1.6× reduction.

The four rows are 16 apart rather than adjacent because the tile is loaded that way — thread
`(tx, ty)` stages tile rows `ty`, `ty+16`, `ty+32`, `ty+48`, which keeps `tx` on the
fastest-varying memory axis so all four loads stay coalesced. Adjacent rows would have meant
each thread reading a 4-row block with a stride, which is the same total traffic through a
worse access pattern.

The accumulators are four separate scalars, not `array<f32, 4>`. A dynamically indexed local
array is backed by memory rather than registers, which would hand straight back the traffic
this kernel exists to avoid.

The workgroup is still 16×16 threads, but it now covers **64 rows × 16 columns** of output.
That is why the dispatch y-dimension divides by 64: `matmulTiled4Workgroups({m: 32, ...})`
is `[_, 1, 1]` where stage 1 gave `[_, 2, 1]`. Getting it wrong dispatches four times the
workgroups, each recomputing rows another one already wrote — correct output, quarter speed,
no error anywhere. The geometry assertion in `matmul_tiled4.test.ts` pins it.

## Why the tile-load guards changed status

Stage 1's doc argued that the `row < m` and `col < n` halves of the tile-load guards were
*defensive* rather than load-bearing: a garbage slot in tileA row `ty` could only ever be
read by threads with that same `ty`, and those threads all shared the same output row, so if
the row was out of range every thread that could see the garbage was masked at the write
anyway.

Under coarsening the reachability argument still runs — tile row `r` is read only by threads
with `ty = r mod 16`, using accumulator `r div 16`, whose output row is `rowBase + r`, so
garbage still only reaches a masked accumulator. **The guards remain defensive by that
argument, and the note requesting this change overstated it.** They are kept, and should be:
they cost nothing, they stop an out-of-bounds read from being silently clamped, and stage 3
changes the mapping again.

What genuinely did become load-bearing is the **write mask**, and it is worth being precise
about which one, because the natural translation from stage 1 is wrong:

```wgsl
// WRONG — one mask for a thread that owns four rows
if (row0 < dims.m && col < dims.n) {
    c[row0 * dims.n + col] = acc0;
    c[row1 * dims.n + col] = acc1;   // row1 may be past the end
    ...
}
```

A thread's four rows are 16 apart, so near the bottom edge of C some are in range and some
are not. At m = 20 with `rowBase = 0` and `ty = 5`, the rows are 5, 21, 37 and 53 — only the
first exists. One combined guard either drops row 5 (if keyed on the last row) or writes
three rows past the end of the matrix (if keyed on the first). The kernel therefore masks
**each accumulator separately**.

This is what `65×33×40` and `80×16×16` are doing in the test suite: both leave a workgroup
straddling the bottom edge with most of its 64 rows out of range, so every accumulator mask
gets exercised. `17×17×40` covers the same ground more brutally — one workgroup produces
1024 candidate outputs for 289 real ones, 735 of them masked.

The k-direction zero-fill is unchanged and still load-bearing, for exactly the reason given
in the stage 1 section. Coarsening only widens the blast radius: a stale tileA value now
corrupts four accumulators instead of one.

## What intensity predicted, and what happened

| | global traffic | intensity | shared loads per MAC | GFLOP/s |
|---|---|---|---|---|
| naive | 8590 MB | 0.25 | — | 225.6 |
| tiled | 537 MB | 4.00 | 2.00 | 543.7 |
| tiled4 | 336 MB | 6.40 | 1.25 | 869.4 |

Global traffic falls 1.6× because tileB is now loaded once per 64 output rows instead of
once per 16 — the same 256 staged floats serve 1024 outputs rather than 256. Shared loads per
multiply-add fall 1.6× for the reason above. Both ratios are 1.6.

**Measured: 1.599×.** Predicted 1.600×.

This is the opposite of stage 1, where a 16× traffic reduction bought 2.41× and the doc had
to explain the gap. Here there is no gap, and that is the useful result: it confirms the
diagnosis that stage 1 left the kernel genuinely shared-memory-bound. When you remove the
thing that is actually limiting a kernel, the speedup matches the model. When you remove
something the cache was already hiding, it does not.

Two caveats on that clean agreement. The match is to three digits on the medians of three
runs, and run-to-run spread is about 9% — so the honest claim is "the ratios agree to within
measurement noise", not that the model is accurate to 0.1%. And it is one shape on one
device; a shape where the ragged edges dominate, or a device with a different shared-memory
architecture, has no obligation to reproduce it.

## What stage 2 did not fix

`matmul_tiled4` reaches roughly 24% of an M2's fp32 peak, up from 15%. Still 1.25 shared
loads per multiply-add, so shared memory is still the limit — just a less severe one.

The obvious next move is more outputs per thread: 8 accumulators takes the ratio to
9 loads / 8 MACs = 1.125, a further 1.11×. The returns are visibly diminishing, and the
register file is finite — at some width the accumulators and staged values stop fitting,
occupancy drops, and the curve turns over. Finding that cliff is what stage 3 is for, which
is why it is a separate kernel and a separate measurement rather than a parameter on this
one.

One thing deliberately left alone: `tileA` is indexed `[row * TILE_K + i]`, so the inner
loop reads it with a stride of 16 floats across threads varying in `ty`. That is a plausible
shared-memory bank conflict. Transposing the tile would fix the read and break the write.
It is not changed here because nothing has measured it — and a change made on that basis
belongs in an ablation row, not in this kernel.

---

# Stage 3: eight outputs per thread, and where this stops working

`matmul_tiled8.wgsl`. Same pattern again, doubled: an 8×1 column strip, eight accumulators,
a 128×16 tile of A. Predicted 1.11×. Measured **1.04× at 1024³, and 0.95× at 256³** — the
first change in this sequence that makes something worse.

That is the result. It is recorded rather than tuned away, because knowing where coarsening
stops paying is more useful than another win.

## The shape of the failure

GPU-side throughput, 200 iterations at the small sizes so timestamp quantization and
submission overhead stop dominating the measurement:

| n | tiled4 | tiled8 | ratio | fraction of the predicted 1.11× | workgroups/core |
|---|---|---|---|---|---|
| 256 | 687.7 | 651.0 | **0.95×** | −48% | 6.4 → 3.2 |
| 512 | 870.4 | 874.2 | 1.00× | 4% | 25.6 → 12.8 |
| 1024 | 904.8 | 951.9 | 1.05× | 47% | 102.4 → 51.2 |
| 2048 | 866.9 | 890.3 | 1.03× | 24% | 409.6 → 204.8 |

The gain is size-dependent, never reaches prediction, and goes negative at the small end.
Note that every size in the table is a multiple of 128, so none of this is the tall tile
wasting rows on a ragged edge — that hypothesis is ruled out by construction.

## It is not register spilling

This was the expected answer, and it is wrong.

Eight accumulators plus eight tileA addresses, `bVal`, and loop counters is perhaps twenty
live values, which is not obviously past what an Apple GPU thread holds. But the decisive
argument is behavioural rather than a register count: **spilling costs the same per thread
regardless of problem size.** A kernel whose inner loop spills would be slower than stage 2
at every n. Stage 3 is 5% *faster* at n = 1024. Whatever is going wrong is not in the inner
loop.

This is worth keeping in mind as a general debugging move — a per-thread explanation has to
produce a size-independent deficit, so a size sweep separates per-thread causes from
per-dispatch ones for free.

## It is occupancy, at the small end

Two things halve when the strip doubles, and neither appears in the loads-per-MAC model:

**Workgroup count.** A 128-row footprint means half as many workgroups as a 64-row one for
the same output. At n = 1024 that is 512 workgroups instead of 1024; at n = 256 it is 32
instead of 64 — 3.2 per core on a 10-core M2.

**Shared memory per workgroup.** tileA goes from 64×16 to 128×16, so the workgroup needs
9 KiB against stage 2's 5 KiB. On a 32 KiB threadgroup-memory budget that is three resident
workgroups where stage 2 fits six.

Both reduce how much latency the core can hide. Two halvings of latency-hiding capacity,
bought for a predicted 1.11× improvement in load ratio — and the table shows the trade
netting out negative wherever there is not enough work to fill the machine anyway. The
deficit tracks workgroups per core more cleanly than anything else measured.

## What is capping the large-shape case is not established

At n = 1024 there are 51 workgroups per core, which is ample, and stage 3 still captures
under half of the predicted gain. Meanwhile both coarsened kernels sit at roughly the same
900 GFLOP/s ceiling despite different load ratios — 1.25 for stage 2, 1.125 for stage 3.

That plateau is the interesting part. The loads-per-MAC model assumes shared-load bandwidth
is the only thing in the way, which is what made stage 2's prediction land to three digits.
It clearly is not the only thing any more. Candidates, unranked because nothing here
distinguishes them:

- instruction issue rate, which counts loads and multiply-adds alike
- shared-memory *latency* not fully hidden at the lower occupancy
- ALU and load co-issue limits on this architecture

Settling it needs achieved occupancy and registers-per-thread for both pipelines, which
means a Metal GPU capture from Xcode. WebGPU exposes neither by any API, and no arrangement
of this benchmark will produce them. That is the honest end of what can be concluded from
inside the browser.

## The structural finding

Coarsening along rows only ever amortizes tileB. Each thread still loads its own tileA value
per accumulator, so the ratio approaches 1.0 loads per MAC and stops:

| strip | loads per MAC | tileA | shared mem |
|---|---|---|---|
| 1×1 (stage 1) | 2.000 | 16×16 | 2 KiB |
| 4×1 (stage 2) | 1.250 | 64×16 | 5 KiB |
| 8×1 (stage 3) | 1.125 | 128×16 | 9 KiB |
| 16×1 | 1.063 | 256×16 | 17 KiB |

The remaining headroom in this direction is 12%, and buying it costs another doubling of
shared memory — the same trade that already went negative once.

Coarsening in **both** directions amortizes both operands. A 4×2 block is also eight outputs
per thread, but costs 4 tileA loads + 2 tileB loads per 8 multiply-adds:

**0.75 loads per MAC, in 6 KiB of shared memory.**

Better than anything reachable by making the strip taller, and in less shared memory than
stage 3 uses. It is strictly the better direction on both axes at once, which is the actual
lesson of this stage: the 1D sequence was the wrong sequence, and running it to the point
where it stopped paying is what made that legible.

Not written here, deliberately — it is a different kernel, and it deserves its own row in
the table rather than being folded in as a fix to this one.

---

# 2D blocking: a 4×2 register block

`matmul_block42.wgsl`. Eight accumulators, the same count as stage 3, arranged as four rows
by two columns instead of eight rows by one. Measured **1240.8 GFLOP/s GPU-side at 1024³** —
1.37× over stage 2, 1.33× over stage 3, and 5.2× over the naive baseline.

This section also corrects the stage 3 section above, which drew the wrong conclusion from a
plateau. See "What the ~900 plateau was" below.

## Working the dimensions out from the target

The target was 0.75 shared loads per multiply-add in about 6 KiB of workgroup memory.
Everything else follows.

A thread computing R rows × C columns loads R values of A and C values of B per k step, and
performs R·C multiply-adds:

```
loads per MAC = (R + C) / (R · C)
```

For eight outputs, 8×1 gives (8+1)/8 = 1.125 and 4×2 gives (4+2)/8 = **0.75**. The product is
fixed at eight; the sum is what you are minimising, and a square-ish block minimises it. That
is the whole reason 2D beats 1D, and it is why the 1D sequence was asymptotically stuck: a
strip of R×1 costs (R+1)/R, which approaches 1.0 and never goes below it.

With a 16×16 workgroup and a 4×2 block per thread, the workgroup covers:

- **64 rows** (16 threads in y × 4) — so tileA is 64×16
- **32 columns** (16 threads in x × 2) — so tileB is 16×32

which is 4 KiB + 2 KiB = **6 KiB**, hitting the target. Dispatch geometry divides by 32 in x
and 64 in y — the first kernel in this family whose x-dimension is not simply `n/16`, and the
easiest thing to get silently wrong when copying the wrapper from stage 3.

Tile loading falls out evenly: 1024 slots of tileA over 256 threads is 4 each (tile rows
`ty`, `ty+16`, `ty+32`, `ty+48` at column `tx`), and 512 slots of tileB is 2 each (row `ty`,
tile columns `tx` and `tx+16`). Both keep `tx` on the fastest-varying memory axis, so both
stay coalesced.

The inner loop pulls six values into registers, then spends them eight ways:

```wgsl
let a0 = tileA[(ty +  0u) * TILE_K + i];
let a1 = tileA[(ty + 16u) * TILE_K + i];
let a2 = tileA[(ty + 32u) * TILE_K + i];
let a3 = tileA[(ty + 48u) * TILE_K + i];
let b0 = tileB[i * COLS + tx];
let b1 = tileB[i * COLS + tx + 16u];

acc00 = acc00 + a0 * b0;  acc01 = acc01 + a0 * b1;
acc10 = acc10 + a1 * b0;  acc11 = acc11 + a1 * b1;
acc20 = acc20 + a2 * b0;  acc21 = acc21 + a2 * b1;
acc30 = acc30 + a3 * b0;  acc31 = acc31 + a3 * b1;
```

Each A value is used twice (once per column) and each B value four times (once per row).
In the 8×1 strip, every A value was used exactly once.

## Masking, now in two dimensions

Stage 2 established that a thread owning several outputs needs one mask per accumulator
rather than one mask per thread. With a 2D block that reasoning applies on both axes at once,
and a guard that combines either axis is wrong in a different way:

- combining **rows** drops real outputs — the shortest row still has outputs in it
- combining **columns** is worse: writing past the end of a row does not run off the buffer,
  it lands in the *next row* of C, silently corrupting a valid output somewhere else

At m = 20, n = 20 with `rowBase = colBase = 0` and thread `(5, 5)`, the rows are 5, 21, 37, 53
and the columns are 5 and 21. Exactly one of the eight outputs exists. So the kernel writes
eight independently guarded stores, each testing its own row *and* its own column.

The test suite carries `20×20×16` for precisely that case, plus `65×32×16` (ragged rows,
exact columns), `64×33×16` (exact rows, ragged columns) and `63×31×17` (ragged on both,
inside a single workgroup) — the combinations that catch a mask which is right on one axis
and wrong on the other.

## What the ~900 plateau was

The stage 3 section observed that tiled4 and tiled8 both landed near 900 GFLOP/s despite load
ratios of 1.25 and 1.125, and concluded that shared-load ratio had stopped being the binding
constraint, with some unidentifiable co-limit holding both down.

**That was wrong.** `block42` reaches 1240 GFLOP/s on the same device at the same shape. There
was no ceiling at 900.

The plateau was two opposing effects cancelling. Stage 3 improved the ratio by 1.11× and at
the same time cut occupancy — 9 KiB per workgroup against 5 KiB, so three resident workgroups
instead of six, on top of dispatching half as many. Net 1.04×. Two kernels arriving at the
same number for different reasons looked like a wall, and one data point either side of a
supposed limit is not enough to establish one.

`block42` is a clean control for this, which is the other reason it is worth having. It
dispatches **exactly the same number of workgroups as tiled8** at every size — 32 at n = 256,
512 at n = 1024 — because the footprint areas match. What differs is 6 KiB against 9 KiB, and
0.75 loads/MAC against 1.125. Holding parallelism fixed and improving both of the other two
gives 1.33×, and it wins at n = 256 where stage 3 loses outright.

## What the model still does not explain

| n | block42 / tiled4 | predicted 1.67× | captured |
|---|---|---|---|
| 256 | 1.20× | | 30% |
| 512 | 1.34× | | 51% |
| 1024 | 1.37× | | 56% |
| 2048 | 1.37× | | 55% |

The ratio still binds — improving it produced the largest single jump in the table — but
throughput responds sublinearly to it. Stage 2 remains the only step whose prediction landed
exactly, and it is now clear why that was flattering: it was the one step that changed the
ratio without changing shared memory per workgroup by much (5 KiB against 2 KiB, both leaving
plenty resident).

The residual gap is unexplained, and it is the same wall stage 3 hit: separating it needs
achieved occupancy and registers per thread, which means a Metal GPU capture from Xcode.
WebGPU exposes neither. What has changed is the direction of the conclusion — the ratio had
not stopped mattering, and there was no hard ceiling.

At 1240 GFLOP/s the kernel is at roughly 34% of an M2's fp32 peak, up from 25% for stage 2.

---

# int4: what unpacking does to the blocking

`matmul_q4_prefill.wgsl` is `matmul_block42` with the weight tile dequantized on the way in.
Same 16×16 workgroup, same 4×2 register block, same 64×32 output footprint, same 0.75 shared
loads per multiply-add. What is new is that each weight arrives as a nibble and has to be
turned into a float.

## Where the unpack goes, and why the blocking decides it

The kernels skill says to dequantize inside the accumulation loop and never write dequantized
weights to a buffer. The second half is about the memory bus: materializing fp32 weights in a
`GPUBuffer` means reading fp32 bytes from VRAM, which is the cost quantization exists to
remove. Unpacking into `var<workgroup>` memory does not do that — shared memory is on-chip,
and the bytes crossing the bus are still int4.

What the blocking decides is *how often the unpack runs*. In a 4×2 block each staged weight is
consumed by four accumulators:

| | unpacks per weight | shared loads per MAC |
|---|---|---|
| unpack while staging | 1 | 0.75 |
| unpack in the inner loop | 4 | 0.75 |

So the register blocking that M1 arrived at for bandwidth reasons turns out to also amortize
the unpack, by exactly its row factor. Coarsening 4 rows divides the unpack cost by 4. That
is a second, independent reason to prefer 2D blocking that the fp32 analysis never surfaced,
because fp32 has no per-element unpack to amortize.

The cost is 2 KiB more shared memory than holding `tileB` packed — and M1's stage 3 is the
warning that shared memory buys occupancy at a rate which can go negative. It is staged here
because prefill has the reuse; the number that would settle whether the packed-tile variant
wins is an ablation that has not been run.

## Decode is a different kernel, and the skill is right about why

`matmul_q4_decode.wgsl` does not tile at all. At m=1 every weight is read exactly once, so
there is no reuse to amortize anything against — staging a weight into shared memory to read
it once is pure overhead, and the unpack happens in the accumulation loop exactly as the skill
writes it. One workgroup per output column, 64 threads splitting the reduction, scale and
zero-point hoisted out of the innermost work because they change once per 64 elements.

One thing that only shows up at this scale: the tied LM head has 151,936 output columns, and
one workgroup per column exceeds `maxComputeWorkgroupsPerDimension` (65,535). That is not an
encode-time error. It invalidates the entire command buffer at submit, and the symptom is a
forward pass that silently produces zeros — perplexity came out as exactly the vocabulary
size, which is what a uniform distribution over 151,936 tokens gives. The grid is now folded
into two dimensions, and `ComputeKernel.encode` checks the limit so the next occurrence is a
named error rather than a silent one.

## What it measured

At 96-token prefill shapes int4 is **3.88× faster** than fp32, not slower. The unpack cost is
real but it is not the binding constraint: at 1885 MiB the fp32 weights miss every cache, so
even the reuse-heavy prefill shape is limited by weight traffic and quarter-size weights help
it too. The prediction that prefill might regress assumed the fp32 kernel was compute-bound at
these shapes. It is not — M1 measured `block42` at 1240 GFLOP/s on square matrices, about 34%
of peak, and the projections here are far more rectangular than that.

The decode-shaped win remains unmeasured, because without a KV cache there is no decode shape
to measure — every step recomputes the whole prefix. See BENCH.md.

---

# Postscript: the 64-row footprint at decode-length prompts
#
# CORRECTED IN M6 — the measurement below was confounded. See the correction at the end.

M5 measured prefill at several prompt lengths for the first time, and the int4 kernel — which
inherits `block42`'s 64×32 output footprint — is **slower than the fp32 16×16 kernel below
about 500 prompt tokens** and 3.9× faster above it.

The cause is the one stage 3 already documented: workgroups, not arithmetic. A 112-token
prompt projecting to 896 columns dispatches `ceil(112/64) × ceil(896/32)` = 2 × 28 = **56
workgroups**, which is 5.6 per core on a 10-core M2. The fp32 kernel's one-thread-per-output
mapping dispatches 392 for the same work. Below the crossover the tall footprint cannot fill
the machine, and above it the same footprint is what makes int4 win.

This is the third time the same measurement has explained a result — stage 3's loss at n=256,
stage 3's flat curve at n=2048, and now int4 prefill at short prompts. The general form: a
kernel's output footprint sets a minimum problem size below which it is the wrong kernel, and
that minimum is `workgroups ≥ several × cores`, not anything about flops.

The fix, if it turns out to matter, is a second int4 prefill kernel with a smaller footprint
selected on prompt length — which is the same prefill/decode split one level finer. It has not
been written, because a chat prompt is usually longer than the crossover and TTFT at 2048
tokens is where users actually wait.


## Correction (M6)

The section above is wrong, and the way it was wrong is worth keeping.

Both formats in that comparison were measured through a prefill graph that baked `maxSeq`
dispatch geometry into every step, launching 5.5 million workgroups per prefill regardless of
prompt length. That fixed cost dominated short prompts for *both* kernels and hid the matmul
difference entirely. Re-measured with dispatch sized to the actual sequence, int4 prefill is
**4.4× faster than fp32 at 128 tokens** — the length at which it was reported to lose — and
there is no crossover at any length.

The footprint argument itself is still sound; it explained stage 3 correctly, where the
comparison was between two kernels under identical dispatch conditions. What was wrong was
applying it to a comparison where a much larger confound was present and unexamined. Three
consistent appearances of an explanation made the fourth feel established, and the check that
would have caught it — does the effect vanish when the suspected cause is removed — was never
run because the explanation already fit.
