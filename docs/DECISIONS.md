# Decisions

Non-obvious choices, one entry each: what was chosen, what was rejected, and what decided
it. A decision with no measurement behind it says so — that is useful information about how
much to trust it.

Entries are append-only. When one is overturned, add a new entry that supersedes it rather
than editing history; the reason something changed is usually more interesting than the
change.

---

## M1 — GPU foundation and matmul

### 1. WebGPU flag constants declared locally, not `@webgpu/types`

**Chosen:** `src/gpu/webgpu-flags.d.ts`, declaring `GPUBufferUsage`, `GPUShaderStage` and
`GPUMapMode` by hand.
**Rejected:** adding `@webgpu/types` as a devDependency.
**Decided by:** TypeScript 6's `lib.dom` ships every WebGPU *interface* but omits the four
flag namespace objects, so `GPUBufferUsage.STORAGE` does not typecheck out of the box.
`@webgpu/types` redeclares the whole API and collides with `lib.dom`. Three `declare var`
blocks of spec constants is the smaller problem, and it keeps the engine's dependency count
at zero.

### 2. Adapter maximum limits requested at device creation

**Chosen:** build `requiredLimits` from `adapter.limits` for every limit the engine cares
about.
**Rejected:** accepting WebGPU's defaults.
**Decided by:** the default `maxStorageBufferBindingSize` is 128 MiB on hardware that allows
4096 MiB — measured on this M2. The fp32 embedding table is 519 MiB. Accepting the default
would fail at weight-load time on hardware that was never the problem.

### 3. A dev flag that clamps the binding size back down

**Chosen:** `?clampStorage` / `VITE_GPU_CLAMP_STORAGE`, forcing 128 MiB.
**Rejected:** trusting that the split-binding path works because it was written carefully.
**Decided by:** with the clamp on, the embedding table needs 5 bindings instead of 1 —
verified in `device.test.ts`. Without a way to reproduce a constrained device, that path
would only ever run on someone else's machine. `maxBufferSize` is deliberately left at the
adapter maximum: the table is one allocation read through several bindings, not several
allocations.

### 4. Buffer pool recycles by power-of-two size class, keyed on usage

**Chosen:** classes of 2ⁿ from 256 B, one free list per `(usage, class)`.
**Rejected:** exact-size matching; a single slab allocator.
**Decided by:** the pool holds transient scratch only — weights and KV cache have their own
owners — so up to 2× slack on activations is cheap next to a 384 MB model. Measured effect
in a bench session: 25 allocations serve 98 acquisitions. Exact-size matching would have
allocated far more; a slab would have needed its own defragmentation story for no gain at
this scale.

### 5. Pooled buffers bind their requested size, not their capacity

**Chosen:** `PooledBuffer.binding` carries `{buffer, offset: 0, size: requested}`.
**Rejected:** binding the whole allocation.
**Decided by:** `arrayLength()` in WGSL reports the *bound* range. A 40-byte tensor handed
its full 256-byte allocation would make a shader compute over 64 elements instead of 10.
This is a correctness issue disguised as an efficiency detail.

### 6. Pipeline cache keyed on full shader source

**Chosen:** the source string itself as part of the map key.
**Rejected:** hashing the source.
**Decided by:** no measurement — the shader corpus is a few dozen kilobytes total, so the
memory cost is irrelevant and a hash only introduces collision risk. Revisit if the corpus
ever grows by an order of magnitude.

### 7. Correctness gate uses a combined per-element tolerance

**Chosen:** an element passes when `abs ≤ atol + rtol·|expected|`; both max-abs and max-rel
are still reported as diagnostics.
**Rejected:** thresholding max-abs and max-rel separately at 1e-4, as first written.
**Decided by:** measurement. A correct kernel at 64×64×64 produces `maxAbs 6.6e-7` and
`maxRel 1.26e-2` — the relative figure comes from an output element near 5e-5, where
relative error is meaningless. A separate max-rel threshold failed a correct kernel, and a
gate that cries wolf gets ignored.

### 8. Kernel tests run in a browser, not in Node

**Chosen:** Vitest browser mode, Playwright, `channel: 'chromium'`.
**Rejected:** Node with a mocked GPU device.
**Decided by:** Node has no WebGPU, so a mocked device tests the mock. Playwright's default
headless build is the headless *shell*, which reports no adapter at all — `channel:
'chromium'` is required, and `device.test.ts` asserts the adapter is not SwiftShader so a
silent fall back to software cannot pass the suite.

### 9. Determinism harness runs iterations concurrently, into separate buffers

**Chosen:** N output buffers, N dispatches in one compute pass, one submit, readbacks at the
end.
**Rejected:** run, read back, compare, repeat.
**Decided by:** reading each result before launching the next serializes exactly the overlap
that makes a race visible. Separate outputs mean no resource hazard between dispatches, so
the driver is free to run them concurrently. The sequential version would have been a
determinism check that could not detect the bug it exists for.

### 10. Determinism compares bit patterns, not float values

**Chosen:** `Uint32Array` view comparison.
**Rejected:** `===` on floats.
**Decided by:** `NaN !== NaN` would mark every element of a uniformly-NaN output as
divergent and bury the real signal; `-0 === 0` would hide a sign flip. "Byte-identical" in
CLAUDE.md means bits, so the check should mean bits.

### 11. The race detector's reporting is tested on fabricated data

**Chosen:** `diffRuns` is a pure function with its own unit tests; the GPU path is proved
separately by a kernel that differs by construction.
**Rejected:** proving the detector works by writing a deliberately racy kernel.
**Decided by:** a racy kernel is not *reliably* racy across drivers, so that test would be
flaky in the direction that matters. The first version of the GPU-path test passed
vacuously — a `vec3<u32>` in the tag uniform forced 32-byte alignment against a 16-byte
binding, the dispatch failed validation, every output stayed zeroed, and zeros agree
perfectly. `DeterminismReport.baseline` now exists so callers can assert the kernel actually
ran.

### 12. Wall clock is the headline benchmark number

**Chosen:** wall clock around a batch of submits; GPU time from `timestamp-query` reported
alongside.
**Rejected:** GPU-side timing as the primary figure.
**Decided by:** wall clock keeps continuity with the recorded baseline and includes
submission overhead, which a decode loop pays 170 times per token. Measured: the GPU-side
figure is also the noisier of the two across runs (tiled4 spans 801–912 GFLOP/s GPU-side
against 813–895 wall).

### 13. `matmul_shared.ts`, and one bind group layout for all variants

**Chosen:** shape validation, Dims uniform and bind group layout shared by every matmul.
**Rejected:** each kernel pair fully self-contained, per the skill's one-kernel-per-file-pair
rule.
**Decided by:** four variants were coming, which meant four copies of the same validation
drifting apart on what counts as a legal shape. Sharing the layout also makes bindings
portable between variants, which is what allows the bench page to A/B them against the same
operand buffers — that turned out to matter more than the duplication did. Dispatch geometry
is deliberately *not* shared, because coarsening changes it.

### 14. Every matmul variant accumulates k in the same order

**Chosen:** ascending k, one term at a time, in all three kernels.
**Rejected:** reassociating the reduction for instruction-level parallelism.
**Decided by:** it makes the variants byte-identical to each other, which is a far sharper
correctness signal than a tolerance check — verified at 129×65×200 on every bench run. A
reassociated variant would still be correct, but a mismatch would then be ambiguous between
"different summation order" and "tiling bug". Worth revisiting only if a measurement shows
the ordering constraint costs real throughput.

### 15. Benchmark baseline moved from 186.7 to 219.4 GFLOP/s

**Chosen:** headless Chromium, the same environment as the test suite, as the reference for
every M1 comparison.
**Rejected:** keeping the original `public/probe.html` figure as the baseline.
**Decided by:** 186.7 came from a foreground Safari tab and 219.4 from headless Chromium,
for byte-identical arithmetic — a 15% environment difference that would have been silently
credited to whichever kernel was measured next. 186.7 is kept in `BENCH.md` as the
user-facing number, since a visitor to the site is in a foreground tab; it is just not the
number kernels are optimized against.

### 16. Thread coarsening as separate kernels, not a parameter

**Chosen:** `matmul_tiled`, `matmul_tiled4`, and a future stage 3, as distinct files.
**Rejected:** one kernel with an override constant for outputs-per-thread.
**Decided by:** the point of the exercise is to see where the register pressure cliff is, and
a parameterized kernel reports one number per configuration without making the cliff visible
in the repository. Separate files also keep each stage's ablation row honest — including the
stage that eventually makes things worse.

### 17. Stage 3 kept in the tree despite being a loss at small shapes

**Chosen:** ship `matmul_tiled8` and record 0.95× at n = 256 alongside 1.04× at n = 1024.
**Rejected:** tuning it until it won; deleting it as a failed experiment.
**Decided by:** the size sweep is what identified the cause, and it only exists because the
kernel does. An ablation table showing where coarsening stops paying is worth more than one
showing four wins in a row — and the losing row is what makes the 4×2 finding (entry 19)
visible. None of the four kernels is wired into a model yet, so carrying one that loses at
small shapes costs nothing but a file.

### 18. Register spilling ruled out behaviourally, not by counting registers

**Chosen:** conclude that stage 3's shortfall is not spilling, on the grounds that its
deficit is size-dependent.
**Rejected:** inferring it from an estimate of live values per thread.
**Decided by:** a per-thread cost has to produce a size-independent deficit. Stage 3 is 5%
faster than stage 2 at n = 1024 and 5% slower at n = 256, so whatever is wrong scales with
the dispatch, not with the thread. A size sweep separates per-thread causes from
per-dispatch ones without needing a profiler, which is useful because WebGPU exposes no
register or occupancy counters at all.

The residual question — what caps *both* coarsened kernels at ~900 GFLOP/s — is explicitly
left open in `docs/tiled-matmul.md`. Settling it needs a Metal GPU capture from Xcode. It is
recorded as unresolved rather than guessed at.

### 19. Next coarsening step is 2D, not a taller strip

**Chosen:** stop the 1D sequence at 8×1 and record that the next measurement should be a 4×2
block.
**Rejected:** 16×1, the obvious continuation.
**Decided by:** arithmetic, confirmed by where stage 3 landed. Coarsening along rows only
amortizes tileB, so the ratio approaches 1.0 loads per MAC and stalls — 16×1 reaches 1.063
and costs 17 KiB of shared memory, another doubling of the thing that already made stage 3
lose at small shapes. A 4×2 block is the same eight outputs per thread at **0.75 loads per
MAC in 6 KiB**, better on both axes simultaneously. Running the 1D sequence until it stopped
paying is what made that comparison legible.

### 20. Benchmark iteration count is a URL parameter

**Chosen:** `?iters=`, defaulting to 10.
**Rejected:** raising the default for everything; leaving it fixed at 10.
**Decided by:** at n = 512 a dispatch is ~0.3 ms, which is three of Chrome's 100 µs timestamp
quanta, and submission overhead is a sixth of the wall measurement. Neither resolves the 5%
difference stage 3 was being judged on. At 200 iterations the same comparison was stable and
reversed the apparent result at n = 256 — the first measurement said tiled8 was 12% faster
there on wall clock, which was pure submission-overhead noise; it is actually 5% slower.
Raising the default instead would have made every large-shape benchmark needlessly slow.

### 21. The "~900 GFLOP/s ceiling" from entry 18 was an artifact, not a limit

**Chosen:** retract the stage 3 reading that shared-load ratio had stopped being the binding
constraint.
**Rejected:** the earlier conclusion, which is left in place in `docs/tiled-matmul.md` with a
correction attached rather than edited away.
**Decided by:** `matmul_block42` reaches 1240 GFLOP/s GPU-side at 1024³ on the same device
where tiled4 and tiled8 both sat near 900. The plateau was two opposing effects cancelling —
stage 3 improved the ratio 1.11× while cutting resident workgroups from 6 to 3 — and two
kernels arriving at the same number for different reasons is not evidence of a wall. One data
point either side of a supposed limit does not establish one; it took a third kernel that
moved the ratio *without* paying for it in occupancy.

Worth generalising: a plateau across two configurations is weak evidence. The cheap test is a
third configuration that varies the suspected cause while holding the suspected confound
fixed, which is exactly what block42 does against tiled8.

### 22. 2D register blocking chosen over a taller 1D strip

**Chosen:** a 4×2 block per thread — 0.75 shared loads per MAC in 6 KiB.
**Rejected:** 16×1, the next rung of the 1D sequence — 1.063 loads per MAC in 17 KiB.
**Decided by:** the arithmetic, then the measurement. For a thread computing R×C outputs the
cost is (R + C) / (R·C); with the product fixed at eight, minimising the sum means a
square-ish block. A strip of R×1 costs (R+1)/R, which approaches 1.0 asymptotically and never
goes below it, so the entire 1D direction was capped at a 12% remaining gain — and buying it
meant another doubling of the shared memory that had already made stage 3 lose at n = 256.
Measured: block42 is 1.37× tiled4 and 1.33× tiled8, and unlike tiled8 it wins at every size
in the sweep.

### 23. Kernel comparisons take GPU-side time as primary at small shapes

**Chosen:** report both, but read the GPU-side figure when comparing kernels below n = 1024.
**Rejected:** wall clock everywhere, per entry 12.
**Decided by:** entry 12 still holds for *absolute* numbers — submission overhead is real and
a decode loop pays it 170 times per token. But at n = 256 the wall figure is up to 20% below
the GPU figure for the same kernel, and that overhead is a property of the harness rather than
of the kernel, so it contaminates a kernel-versus-kernel ratio. The sweep in `BENCH.md` is
GPU-side for this reason and says so. Both are recorded so neither reading is lost.

---

## M2 — weight loading and tokenizer

### 24. Weights are cached per chunk, not as one file

**Chosen:** each planned byte range is its own Cache API entry, keyed by model id, revision
and range.
**Rejected:** storing the whole 988 MB response under one key.
**Decided by:** the Cache API has no partial matching, so a whole-file entry is all-or-nothing.
A download interrupted at 70% would resume from zero — on the highest-abandonment moment of
the site — and a warm load would have to materialize a gigabyte in memory to serve ranges
from it. Chunk entries resume from where they stopped and are read one at a time. Measured:
73 chunks for the full model, warm load 1.47 s.

### 25. Chunk plan is derived from the header, not fixed-size

**Chosen:** group whole tensors into ~16 MiB contiguous ranges; a tensor larger than the
target gets its own chunk.
**Rejected:** one request per tensor; one request for the whole file; fixed-size chunks that
split tensors.
**Decided by:** 290 tensors means 290 round trips one way, most for tensors under a kilobyte;
one request the other way reports no progress and cannot resume. Chunking on tensor
boundaries means no tensor is ever reassembled across two reads, which keeps the upload path
free of seam handling. `planChunks` is pure, so the plan is asserted in a test and the ranges
double as cache keys.

### 26. bf16 is widened to f32 on upload, and the cost is accepted

**Chosen:** every weight becomes f32 in GPU memory.
**Rejected:** keeping bf16 packed and unpacking in the shader; using f16 storage.
**Decided by:** WGSL has no bf16 type at all, and `shader-f16` is missing on roughly a third
of devices, so f32 is the only universally bindable representation. Measured cost: 988 MB on
disk becomes **1884.6 MiB resident**. That is the number that justifies the int4 path — it is
not a rounding error, it is the model twice over. Widening itself is exact and cheap, since
bf16 is literally the top 16 bits of an f32.

### 27. Embedding parts are equal-sized and split on row boundaries

**Chosen:** `ceil(rows / partCount)` rows per part, stepped down until each part fits.
**Rejected:** filling each part to the byte limit and leaving a small remainder; splitting on
raw byte offsets.
**Decided by:** equal row-sized parts make `floor(row / rowsPerPart)` a complete index with no
lookup table and no seam handling. A byte-aligned split would put some rows across two
bindings, forcing every reader — input lookup and tied LM head both — to handle the boundary.
Measured at the 128 MiB default: 5 parts, 30,388 rows each, 103.9 MiB per part, and the
clamped device loads it.

### 28. Weight buffers carry COPY_SRC

**Chosen:** `STORAGE | COPY_DST | COPY_SRC` on every weight buffer.
**Rejected:** `STORAGE | COPY_DST`, the minimum that works.
**Decided by:** a test failure. Without COPY_SRC a readback is a validation error, and a
validation error surfaces asynchronously — so the staging buffer simply stayed zero and the
comparison failed against zeros rather than reporting the real cause. The parity harness will
need to read weights back; the usage flag costs nothing.

### 29. The split regex spells out case alternatives instead of using an inline modifier

**Chosen:** `'[sS]|'[tT]|'[rR][eE]|...` in place of Qwen's `(?i:'s|'t|...)`.
**Rejected:** using the pattern verbatim from tokenizer.json.
**Decided by:** inline regex modifiers are ES2025. V8 has them, so the pattern works
verbatim in Chrome and in the test browser — which is exactly why this would not have been
caught by the suite. Safari ships them later than the WebGPU support this project already
requires, so the verbatim pattern would throw at construction on a browser that can otherwise
run the engine. The rewrite is exactly equivalent; every alternative is ASCII with a two-case
mapping.

### 30. TextDecoder is constructed with ignoreBOM

**Chosen:** `new TextDecoder('utf-8', { fatal: false, ignoreBOM: true })` everywhere token
bytes are decoded.
**Rejected:** the default, which is what the code originally had.
**Decided by:** a fixture failure — 1 of 2,226. `TextDecoder` silently deletes a leading
U+FEFF, so the token for a byte-order mark decoded to the empty string and any document
starting with one lost its first character. Invisible in every hand-written test, and exactly
the class of bug the exact-agreement fixture exists to catch.

### 31. Decode concatenates bytes before UTF-8 decoding

**Chosen:** gather every token's bytes, then decode the whole buffer once.
**Rejected:** decoding each token to a string and joining.
**Decided by:** token boundaries fall between bytes, not between characters. A single emoji
spans several tokens, and per-token decoding turns each fragment into replacement characters.
There is a test that asserts the naive version differs, so the property cannot regress
silently.

### 32. Added tokens are matched before normalization, and can be disabled

**Chosen:** match added tokens against the raw text first; `allowSpecial: false` treats them
as literal characters.
**Rejected:** normalizing first; always honouring special tokens.
**Decided by:** matching first is what the reference does, and it means NFC can never alter a
special token's spelling. The opt-out exists because untrusted user input that can emit
`<|im_start|>` can forge a conversation role — so the runtime will encode user text with
`allowSpecial: false` and only the template with it on.

### 33. Load benchmarks use a synthetic file with real geometry

**Chosen:** generate a 988 MB safetensors with Qwen's exact 290-tensor inventory, serve it
from the dev server.
**Rejected:** downloading the real checkpoint from the CDN for benchmarking.
**Decided by:** the loader's cost is parse, widen and upload; a real CDN adds network variance
that swamps all three and is not reproducible run to run. The synthetic file exercises
identical code paths on identical shapes. The tradeoff is stated in `BENCH.md`: the cold
number is a floor, not the visitor experience, and the real-world figure is derived from
bandwidth arithmetic instead of being quietly implied.

---

## M3 — fp32 forward pass

### 34. The split embedding is never bound as five buffers at once

**Chosen:** one dispatch per embedding part, each writing a disjoint slice of the output.
Both consumers of the tied table decompose this way, so no bind group ever holds more than
one part.
**Rejected:** binding all five parts to a single dispatch (7 storage buffers before counting
anything else, against a limit of 10); a gather pass that copies the needed rows into one
contiguous buffer first (an extra 519 MiB and a full copy per token).
**Decided by:** the split axis is vocabulary rows, and vocabulary rows are the *index* axis of
the input gather and the *output* axis of the LM head. That is what makes both decompose:

- **Input gather** — dispatch `p` binds only part `p`, and writes an output row when the
  token's id falls in that part's range. Every token id belongs to exactly one part, so each
  output row is written exactly once across the five dispatches.
- **Tied LM head** — dispatch `p` computes `logits[p*rowsPerPart ... ]` from part `p` alone.
  The output slices are disjoint by construction and no reduction crosses a part boundary.

Peak storage bindings anywhere in the graph is therefore 3, not 7. Worth noting *why* it
works, because it is a property of this split and not a general one: if the table were split
along `hidden` instead of vocab, the LM head would need a reduction across every part and all
five would have to be resident in one bind group, or a second pass would have to sum partial
results. Splitting on rows is what buys the decomposition.

### 35. The parity harness was built before any kernel, and validated against PyTorch first

**Chosen:** dump the reference, then check the *TypeScript CPU reference* against it, then
write shaders against the CPU reference.
**Rejected:** writing kernels first and building the harness to check them.
**Decided by:** the CPU check found nothing wrong, which is the point — it confirmed the
model's semantics (RoPE half-rotation, contiguous GQA grouping, `[out, in]` weight layout,
mean-not-sum RMSNorm) on all 411 stages before a single line of WGSL existed. Had any of
those been misunderstood, every kernel would have been confidently wrong in the same way and
the parity table would have shown a uniform failure with no first-failing-stage to read.
`node tools/check_cpu_reference.ts` runs in Node with native type stripping and needs no GPU.

### 36. Reference hooks are detached before greedy decode

**Chosen:** remove the forward hooks and restore the patched RoPE immediately after the
single reference forward pass.
**Rejected:** leaving them attached for the whole script, which is what the first version did.
**Decided by:** a wrong-shaped fixture. Hooks overwrite on every call and greedy decode
re-runs the model on a growing sequence, so `reference.bin` contained the *last decode step's*
activations at sequence length 34 instead of the prompt's at 15. It failed loudly here only
because the lengths differed; had the decode been one token it would have produced a
plausible fixture of the wrong thing.

### 37. Attention is three kernels, not one fused one

**Chosen:** `attn_scores` → `softmax` → `attn_apply`, materializing the attention weights.
**Rejected:** a single fused attention kernel.
**Decided by:** the parity skill lists attention weights as a boundary to dump, and a fused
kernel never materializes them — there would be nothing to compare. Softmax also becomes
independently testable, including the overflow and fully-masked cases. Fusing is a
performance change to make later, against a correct baseline, and it will cost this
observability.

### 38. Masked attention scores use a finite sentinel, not -infinity

**Chosen:** `-1.0e30` for masked positions.
**Rejected:** `-inf`, which is what the CPU reference uses.
**Decided by:** the softmax subtracts the row maximum, and `-inf - -inf` is NaN if an entire
row is masked. A finite sentinel exponentiates to exactly zero and cannot poison a row. The
kernel test compares only the unmasked triangle against the reference and checks the mask
separately, since the two representations differ by construction.

### 39. Uniforms are rewritten every decode step, not written once

**Chosen:** `setSequenceLength` rewrites every dispatch's uniform when the length changes.
**Rejected:** writing uniforms once at build time with `maxSeq` and letting kernels mask.
**Decided by:** `seq` is a *stride*, not only a bound. Attention scores are indexed
`(h*seq + i)*seq + j`, so a stale `seq` reshapes the buffer rather than over-running it — the
kernel reads well-formed values from the wrong places and produces a plausible wrong answer.
Buffers are still sized for `maxSeq` and bind groups still outlive every token; only the
uniforms move.

### 40. The LM head offsets its output inside the shader, not through a binding offset

**Chosen:** `outStride` and `outOffset` fields in the matmul uniform.
**Rejected:** binding each part's logits slice at a byte offset.
**Decided by:** storage binding offsets must be 256-byte aligned, and the slice boundaries
land at vocabulary row 30,388 — 121,552 bytes, which is not a multiple of 256. No choice of
part count fixes this without also constraining `rowsPerPart` to a multiple of 64 and
changing the split. Offsetting inside the shader sidesteps the alignment rule entirely and
costs one add.

### 41. Only the last position is projected to logits

**Chosen:** copy row `seq-1` of the final norm into a fixed one-row buffer, then project that.
**Rejected:** projecting every position.
**Decided by:** greedy decode reads one row, and projecting all of them costs
`seq × 151,936` floats per step — 2 MB per step at seq 35, for 34 rows that are discarded.
The copy exists because the source row moves with the sequence and a bind group cannot; a
`copyBufferToBuffer` offset is chosen at encode time, so it can.

### 42. Parity is excluded from `npm test`

**Chosen:** `npm test` runs everything except `test/parity`; `npm run parity` runs it.
**Rejected:** one suite.
**Decided by:** parity needs `model.safetensors` (942 MB) and `reference.bin` (44 MB), both
gitignored and both regenerated by scripts in `tools/`. A default suite that cannot run on a
fresh clone is worse than an explicit command. Everything that *can* run on a fresh clone —
including all 31 per-kernel comparisons — stays in `npm test`.

### 43. Parity reports two columns; only the isolated one gates

**Chosen:** every stage is measured twice — once with all its inputs written from the
reference dump and only its own dispatches encoded, once with the whole prefix run from the
prompt. The isolated column asserts; the accumulated column is reported.
**Rejected:** accumulated-only, which is what the harness did first; isolated-only, which
would have discarded real information.
**Decided by:** the accumulated-only harness reported 21 failures whose cause could not be
read off the table — establishing they were drift and not bugs took three separate arguments
(per-kernel unit tests, a float64 CPU reference, and the exact greedy match). With reference
inputs injected, **all 387 stages pass**, worst case 23.4% of tolerance, and the parity
skill's rule — read from the top, the first red row is the bug — is true as written rather
than true only if you already know the answer.

The accumulated column is kept rather than dropped because it measures something real: the
fp32 path's end-to-end error, growing monotonically to 635% of tolerance at `final_norm`.
That is the baseline the int4 path will be compared against. A quantized run landing near
those numbers is behaving as expected; one landing far above them is not, and without this
column there would be nothing to say which.

This is deliberately *not* a threshold change. The threshold is unchanged at
`abs 1e-4 + rel 1e-4`; what changed is that the harness now measures the quantity the
threshold was written for.

### 44. `encode` takes a step range, not just an end

**Chosen:** `ForwardGraph.encode(encoder, seq, upToStep, fromStep)`.
**Rejected:** rebuilding a single-stage graph for isolation; a separate isolation-only code
path.
**Decided by:** isolation has to run *the same bind groups and the same uniforms* as the real
forward pass, or it tests a different program than the one that ships. A step range reuses the
graph exactly as built; anything that reconstructs the stage risks passing while the real path
fails. It also keeps the isolation machinery to one extra parameter and a loop bound.

---

## M4 — int4 quantization

### 45. Scales are f32, zero-points are packed 4-bit

**Chosen:** one f32 scale per block of 64, zero-points packed eight per u32.
**Rejected:** f16 scales, as the kernels skill's example shows; f32 zero-points.
**Decided by:** `shader-f16` is missing on roughly a third of devices and there is one
universal path here, so an f16 scale would need an fp32 sibling kernel and a tested fallback
before it could ship. The cost is 0.5 bits/weight — 4.57 measured instead of ~4.07 — which is
27 MB on a 282 MB file. Packing the zero-points recovers most of what a naive f32 zero-point
would have wasted, for one extra shift and mask that the kernel is already doing for the
weights.

### 46. A zero-range block uses |c|/15 as its scale, not 1

**Chosen:** when a block's values are all equal, `scale = |c|/15` (or 1 when c is 0).
**Rejected:** `scale = 1`, which is what both implementations did first.
**Decided by:** a round-trip test. With scale 1 the constant 0.375 quantizes to nibble 0 and
dequantizes to **0** — the value is lost entirely, silently, only in blocks that happen to be
constant. `|c|/15` puts the constant on nibble 15 (nibble 0 when negative) and the existing
zero-point formula recovers it exactly. Found by `test/reference/quant.ts`'s round-trip tests
before the quantizer ever ran on real weights.

### 47. The unpack is amortized by the register blocking, not moved into the loop

**Chosen:** dequantize while staging the weight tile into shared memory, in the prefill kernel.
**Rejected:** unpacking in the inner loop, which is what the skill's example shows.
**Decided by:** the skill's rule is about the memory bus — never write fp32 weights to a
`GPUBuffer` — and shared memory is on-chip, so the bytes crossing the bus are still int4. What
the placement changes is frequency: in a 4×2 block each staged weight feeds four accumulators,
so staging unpacks once where the loop would unpack four times. The 2D blocking M1 arrived at
for bandwidth reasons amortizes the unpack by exactly its row factor, which is a second reason
to prefer it that the fp32 analysis never surfaced. The decode kernel, which has no reuse,
unpacks in the loop exactly as the skill describes.

### 48. Dispatch dimensions are checked at encode time

**Chosen:** `ComputeKernel.encode` throws when a workgroup count exceeds
`maxComputeWorkgroupsPerDimension`.
**Rejected:** relying on the driver to report it.
**Decided by:** it does not report it. The tied LM head at one workgroup per output column is
151,936 workgroups against a 65,535 limit; that invalidates the whole command buffer at submit
with no error surfaced anywhere, and the forward pass silently produced zeros. Perplexity came
out as exactly 151,936 — a uniform distribution over the vocabulary — which is the only reason
it was noticed at all. The decode kernel now folds its grid into two dimensions, and the check
turns the next occurrence into a named failure.

### 49. The isolated parity column gates fp32 and does not gate int4

**Chosen:** under `--q4` the per-stage table is printed and not asserted; the int4 gate is
per-kernel agreement with the CPU dequantized reference, plus perplexity.
**Rejected:** asserting the skill's int4 thresholds on the isolated column, which fails 165 of
387 stages; quietly loosening those thresholds.
**Decided by:** the isolated column means something different under int4. Feeding a stage
reference-quality *inputs* does not remove the quantization of its *weights* — the weights are
what changed — so the column measures per-stage quantization loss rather than kernel error. It
is a good diagnostic (it identified q and k projections as the worst quantizers, matching
`quantize.py`'s own per-tensor report) and it cannot detect a bug. Asserting on it would be
asserting that quantization is lossless. Correctness is established instead by three checks
that do not depend on a tolerance: the kernels reproduce `linearQ4` exactly, the shipped file
decodes to the weights it was made from, and the GPU's stage error equals what a CPU forward
pass over the dequantized shipped weights produces to four digits.

### 50. Everything 2-D is quantized, including the tied embedding

**Chosen:** quantize all 169 two-dimensional weights; keep norms and biases in f32.
**Rejected:** exempting the embedding/LM head, which is a common choice and would likely
improve perplexity.
**Decided by:** "quantize everything" is the honest baseline and the measurement it produces is
the one that says whether an exception is worth its cost. It is: perplexity rises 28.3%
(30.25 → 38.82), and the tied LM head is the leading suspect — quantizing that one tensor
perturbs final logits by rms 0.376 over a range of −15.4 to 20.5, the same order as the
0.249-nat NLL increase the perplexity ratio implies. Exempting it would cost 544 MiB of the
1616 MiB saved. That is a real trade with a real number on each side, and it belongs in its own
change with its own row rather than folded into the baseline.

### 51. The tied embedding ships at int8, not int4 and not exempt

**Chosen:** int4 everywhere except the tied embedding / LM head, which is int8. 334.9 MiB,
5.63× smaller than fp32, perplexity 35.22.
**Rejected:** int4 throughout (268.9 MiB, 38.82 — 3.59 PPL worse for 66 MiB saved); f16 or f32
exemption (454.5 / 714.2 MiB for no measurable gain over int8).
**Decided by:** a four-way ablation at fixed everything-else. int8 recovers **102%** of what a
full f32 exemption recovers, for **15%** of the memory. On perplexity per MiB it is 2.9× better
than f16, 6.9× better than f32 exemption and 10× better than staying in fp32. The marginal
step from int8 to f16 costs 119.6 MiB and returns −0.075 PPL — negative, within noise.

### 52. The "output projection dominates" hypothesis is 38% right, and recorded as such

**Chosen:** record that the tied embedding accounts for 38% of int4's perplexity cost and the
other 169 tensors for 62%.
**Rejected:** reporting only that int8 fixed the embedding, which was the actionable half and
would have left the wrong mental model in place.
**Decided by:** decomposing the 0.2493-nat degradation. Exempting the embedding entirely still
leaves perplexity at 35.30 against 30.25, a 16.7% gap that no embedding precision touches. The
prediction was that the output projection would dominate because it maps straight to logits
with no downstream layer to absorb the error; it is mildly disproportionate — 27.5% of
parameters, 38% of loss — and not the main cause. Believing otherwise would send the next
quantization change at the wrong tensor.

### 53. f16 weights use `unpack2x16float`, not the `shader-f16` extension

**Chosen:** store halves two per u32 and unpack to f32 in registers with `unpack2x16float`.
**Rejected:** `enable f16` with native `f16` storage.
**Decided by:** `unpack2x16float` is core WGSL and works everywhere, while `shader-f16` is
missing on roughly a third of devices and would need an fp32 sibling kernel plus a tested
fallback before it could ship. The values are half in memory and f32 in registers either way,
so the residency saving is identical and the universal path is preserved. This is what made a
four-way ablation cheap: all four dtypes are the same kernel with a different dequant body.

---

## M5 — KV cache, prefill/decode split, GPU sampling

### 54. Top-p is a threshold bisection, not a sort

**Chosen:** bisect on a probability threshold — 32 iterations, each one reduction over the
vocabulary — and keep every token above it.
**Rejected:** a bitonic sort of 151,936 logits; a partial top-k reduction.
**Decided by:** top-p asks for a cut point, not a total order. A bitonic sort is
`log2(n)²/2 ≈ 145` global passes to compute an ordering that gets thrown away. The mass above
a threshold is monotone in the threshold, so bisection converges to the same cut in 32 passes
with no scratch storage and no ordering network. Partial top-k was the other candidate and is
better when k is small and known; top-p's k is data-dependent and can reach the thousands on a
flat distribution, at which point the candidate set stops fitting in shared memory.

The cost is that tokens tied at exactly the threshold are all kept or all dropped, so the
retained mass can overshoot p by at most the tied group. That is harmless for sampling and,
unlike the approximations that would fix it, deterministic.

### 55. The sampler walks contiguous chunks, not strided ones

**Chosen:** thread `t` owns `[t·C, (t+1)·C)` of the vocabulary.
**Rejected:** strided assignment, which coalesces better.
**Decided by:** the final draw is a cumulative scan, and which token a given random number
selects depends on the order the scan visits ids in. Contiguous chunks concatenate into
ascending id order, so the result matches what a sequential scan would pick and does not depend
on the workgroup size. The coalescing loss is irrelevant at one dispatch per token.

### 56. K is cached after RoPE

**Chosen:** rotate the key on the way into the cache.
**Rejected:** caching pre-rotation keys and rotating on read.
**Decided by:** a key's rotation depends only on its own absolute position and never changes,
so rotating on read would re-rotate the entire history every step — which is the cost the
cache exists to remove. Storing post-RoPE also means the decode attention kernel reads the
cache directly with no preprocessing.

### 57. Projections write into the cache; there is no append kernel

**Chosen:** the V projection and the K rotation target the cache buffer at
`position × kvHeads × headDim` using the `outOffset` those kernels already had.
**Rejected:** a separate `kv_append` kernel; a `copyBufferToBuffer` per layer per token.
**Decided by:** the offset is already a uniform field on both kernels, added for the tied LM
head in M4. Reusing it makes the append free — 48 fewer dispatches or copies per token — and
the alternative would have been 48 command-encoder operations per token for data that was
about to be written anyway.

### 58. Prefill and decode are separate files, not a mode flag

**Chosen:** `graph.ts` for prefill, `graph_decode.ts` for decode.
**Rejected:** one graph with a branch.
**Decided by:** the kernels skill says a kernel tuned for one regime underperforms on the
other and that they should be separate. The graphs differ in more than a flag anyway — every
projection changes kernel, attention changes kernel, the softmax needs a stride distinct from
its row length, and RoPE needs an output offset. Measured vindication: int4 decode is
1.34–1.54× fp32 while int4 prefill is *slower* than fp32 below ~500 tokens. One kernel could
not have been right for both.

### 59. Softmax takes a stride separate from its row length

**Chosen:** `softmaxDims(rows, cols, stride)`.
**Rejected:** deriving the stride from `cols`, which is what it did through M4.
**Decided by:** decode's score buffer is preallocated to the maximum context while the live row
grows by one every step. Conflating the two reads the wrong row — silently, with well-formed
values from the wrong positions. Prefill passes `stride === cols` and is unaffected.

### 60. TTFT is measured after a warm-up generation

**Chosen:** run one short generation, reset, then time.
**Rejected:** timing the first call.
**Decided by:** pipelines compile lazily on first dispatch, and the unwarmed run reported
1506 ms at context 128 against 2434 ms at 2048 — a number that goes *down* per token as the
prompt grows, which is the compile, not the prompt. Charging one shader compilation to TTFT
reports a figure no user experiences more than once per session. The compile cost is real and
belongs in a load-time budget, not in a per-request latency.

---

## M6 — optimization

### 61. Prefill dispatch geometry is a function of sequence length, not a build-time constant

**Chosen:** `workgroupsFor(seq)`, evaluated at encode time.
**Rejected:** baking `maxSeq` geometry at graph-build time, which is what the graph did from M3
through M5.
**Decided by:** a 32-token prompt was taking 1518 ms. The graph is sized to the maximum context
so its buffers can be allocated once, and the dispatch geometry had silently inherited that —
launching 5,505,024 attention workgroups per prefill at every prompt length, each one returning
immediately from its bounds check. Sizing dispatches to the actual sequence: **prefill at 128
went 95 → 1077 tok/s, TTFT at 32 tokens went 1518 → 246 ms.** Prefill at 2048 was unchanged,
which is the control that confirms the diagnosis — at the maximum context the old geometry was
already right.

Bounds checks made this correct but slow, which is the worst combination: nothing failed, no
test went red, and the cost was invisible for three milestones.

### 62. Precompiling pipelines is kept despite measuring nothing

**Chosen:** compile all pipelines up front during the weight download.
**Rejected:** removing it after it measured null.
**Decided by:** the premise — that TTFT hid over a second of shader compilation — is false on
this device. All 17 pipelines compile in 4–13 ms, and TTFT measured 1544 ms with precompilation
against 1518 ms without. It is kept because it costs 4 ms against a 282 MB download and a device
with a slower shader compiler could plausibly differ, but BENCH.md records it as a null result
rather than a win. A change kept on judgement rather than measurement should say so.

### 63. Kernel selection by problem size was not implemented, because its premise was wrong

**Chosen:** delete the plan, correct the documentation.
**Rejected:** implementing a small-footprint int4 prefill kernel and selecting on a crossover.
**Decided by:** the crossover does not exist. M5 measured int4 prefill losing to fp32 below ~500
tokens and attributed it to `block42`'s 64×32 footprint dispatching too few workgroups — the
third appearance of an explanation that had been correct twice before. Both formats had been
measured through the same over-dispatched attention, which dominated both. After entry 61, int4
prefill is 4.4× *faster* than fp32 at 128 tokens, with the widest margin at exactly the lengths
where it was reported to lose.

Worth stating plainly because the failure was one of method, not arithmetic: an explanation that
had worked three times was applied a fourth without running the check that would have falsified
it. `docs/tiled-matmul.md` carries the correction next to the original claim rather than in
place of it.

### 64. Fused RMSNorm deferred against a computed bound rather than built and measured

**Chosen:** bound the payoff first — 0.15% of prefill, 0.01% of decode — and defer.
**Rejected:** implementing it and adding a 0.0% row to the ablation table.
**Decided by:** the saving is the RMSNorm output write, 359.7 MB per prefill at 2048 and 0.2 MB
per decode step. At 100 GB/s that is 3.6 ms of a 2433 ms pass. Prefill at 2048 runs at 832
GFLOP/s against `block42`'s measured 1240 GFLOP/s peak, so it is compute-bound and that
bandwidth is hidden rather than paid. The implementation is not free either: the fused form
needs `rowScale` and `normWeight` bindings on every matmul, 7 storage bindings becoming 9
against a limit of 10.

Deferred with the numbers rather than skipped silently, so the call can be reversed on evidence
rather than on preference.

### 65. Batched decode uniforms kept as a null result

**Chosen:** one uniform allocation with static per-step binding offsets, one write per token,
replacing ~410 `writeBuffer` calls.
**Decided by:** decode @512 went 40.3 → 40.3 tok/s. Decode is GPU-bound, so removing CPU work
buys nothing here. Kept because it is strictly less work and the same change would matter on a
slower host, but recorded as measuring nothing.

### 66. The BENCH.md audit found the opposite of what was expected

**Chosen:** re-derive every timing against the dispatch fix, and mark corrected rows in place
rather than replacing them.
**Decided by:** the expectation was that M4's int4-vs-fp32 prefill comparison was the main
casualty. It was not affected at all — M4's harness sized its graph to exactly the sequence it
measured, so the bug could not express itself. **M5's was the wrong one**, because its harness
sized to `maxContext` as a real session must. A latent bug of this class only appears once the
surrounding code becomes realistic, which means "it was fine when we measured it before" is not
evidence that a measurement was sound.

Corrected rows keep the original value struck through beside the new one. A silently replaced
number is indistinguishable from a number that was always right.

### 67. f16 KV cache shipped on memory, not on the speed case that motivated it

**Chosen:** f16 KV cache as the default, with `pack2x16float` / `unpack2x16float` (core WGSL,
no `shader-f16` and no fp32 sibling needed).
**Rejected:** f16 accumulation, which the decomposition showed targets a term that is not
bandwidth-bound at all — 335 MB of weights moving in 22.08 ms is 15.2 GB/s against ~100 GB/s
available.
**Decided by:** measurement, which then contradicted the measurement that chose it. The
decomposition predicted +21.9% decode at a 2048 context; the build delivered **+3.4%**, against
a 4% prefill regression, and halved the cache from 48 to 24 MiB.

The decomposition was right that 12.0 ms of a 34.2 ms step scales with context and wrong that
the term is bandwidth. It is iteration count: `attn_apply_decode` walks the history serially,
one position per loop iteration per thread, and halving the bytes per iteration does not remove
iterations. Shipped for the memory, with the speed claim retracted in BENCH.md rather than
quietly dropped.

Recorded because it is the same error as entry 63 in different clothing — a term measured
correctly, a mechanism assumed, and the assumption left untested until something was built on
it. The general fix is cheap and was skipped twice: change the suspected cause and check that
the effect moves.

### 68. The mechanism check runs before the build, and is now in the kernels skill

**Chosen:** vary the suspected cause independently and confirm the effect moves, before
optimizing against it. Added to `.claude/skills/enargeia-kernels`.
**Decided by:** three wrong mechanism assumptions in six milestones — the M5 footprint claim,
the over-dispatch diagnosis, and the f16 cache prediction. Each was plausible, consistent with
the data in hand, and would have been caught in minutes by the check.

Run properly this time, the probe was also more informative than expected. Unrolling
`attn_apply_decode` 8× at constant bytes bought 1.39×, which says iteration latency is real —
and saturating far short of 8× says it is not the whole cost. A fix built on the unroll result
alone would have recovered 1.39× instead of the 1.5× the full parallelisation gave. The check
does not just falsify a wrong mechanism; it sizes the right ones.

### 69. Decode attention reduces across the history in a workgroup

**Chosen:** one workgroup per output element, 64 threads splitting the history, shared-memory
reduction.
**Rejected:** one thread per output element walking the history serially, which is what M5
shipped; unrolling alone.
**Decided by:** the serial form dispatched 896 output elements as 14 workgroups — 1.4 per core
on a 10-core GPU — with each thread iterating once per cached position. The workgroup form is
896 workgroups with `history / 64` iterations each, addressing occupancy and iteration count
together. **Decode at 2048 went 28.7 → 43.1 tok/s (+50.4%), and the position-dependent term
went from 3.89 µs/position to −0.15 — flat in context.**

The sum runs in a different order than the serial kernel's, so it is a different floating-point
association. Deterministic, and not bit-identical to the old kernel; greedy output against the
no-cache reference is unchanged at 24/24.

### 70. f16 KV cache reclassified as memory-only

**Chosen:** keep it, and retract the remaining speed justification.
**Decided by:** it bought +3.4% against the serial attention kernel. Against the parallel one
the position-dependent term is gone, so halving the bytes of a term that costs nothing buys
nothing. It remains worth 24 MiB against 48 at a 2048 context, at a 4% prefill cost. Recorded
because a change whose justification has been invalidated twice should say so rather than keep
inheriting the original claim.

### 71. Repetition penalty applied once in place, not at every read

**Chosen:** a first phase in `sample.wgsl` that writes penalized logits back into the logits
buffer before anything reads one, with `logits` bound `read_write`.
**Rejected:** the original per-read helper; a separate dispatch; a workgroup-resident bitset of
the vocabulary (151,936 bits is 19 KB and does not fit in workgroup storage).
**Decided by:** the helper scanned the whole history on every logit read, and the vocabulary is
read about 36 times per token, so the cost was `vocab × history × 36`. **Measured through the
app: 71.4 ms/token with the penalty alone against 26.7 greedy; 24.4 after the fix.** Full
sampling went 84.9 → 28.6 ms/token, 3.0×.

Only the first occurrence of an id applies the penalty, which is what the old code did (it
broke at the first match) and what the reference does — it penalizes a set, not a multiset.
That is also what makes the in-place write safe without atomics: no two threads target the same
address.

The cost was invisible to every harness in the repository because they all measure with
`GREEDY`, and greedy skips the penalty. A cost that only appears under settings no test uses is
not caught by adding more tests of the same kind — it was caught by running the engine the way
a person would.

### 72. The site owns one seam to the engine, and no panel crosses it

**Chosen:** `src/ui/engine.ts` is the only file under `src/ui` that imports from `src/gpu`,
`src/kernels`, `src/model` or `src/runtime`. Everything else renders a `Telemetry` snapshot the
runtime publishes.
**Rejected:** panels reading GPU buffers directly; a shared store both sides mutate.
**Decided by:** the boundary is what lets the inspector be closed, deleted, or rebuilt without
changing a dispatch — and it is what makes "the inspector costs nothing measurable" checkable
rather than asserted (29.2 ms/token with profiling off against 29.4 with it on). The runtime
throttles publication to 30 Hz; no panel schedules a frame.

### 73. Attention sampling and per-kernel profiling are opt-in and duty-cycled

**Chosen:** timestamp-query profiling on one decode step in 16; attention weights read back
only while the heatmap panel is open, on the same cycle.
**Rejected:** profiling every step; sampling attention every token.
**Decided by:** the decode budget is one readback per token and the attention heatmap needs a
second one. Rather than quietly breaking that rule, the panel is off by default and says in its
own copy what turning it on costs. Profiling a step also records ~460 separate compute passes
instead of one, which is not free even though it measured inside noise here.

### 74. A mobile adapter gets a 1024-token context

**Chosen:** `contextFor(profile)` returns 1024 when the adapter classifies as mobile.
**Rejected:** one context length everywhere; asking the visitor.
**Decided by:** prefill activations dominate resident memory and scale with the square of the
context — the two attention buffers alone are 470 MiB at 2048 and 118 at 1024. On a phone that
is the difference between running and failing to allocate. The device panel names the choice
and the reason rather than silently shipping a different product.

### 75. Three.js, GSAP and Lenis are loaded after first paint

**Chosen:** a 8.24 kB entry bundle; the engine behind the load button, the hero behind a
dynamic import.
**Rejected:** one bundle, on the grounds that 773 kB is 0.2% of the model download.
**Decided by:** that argument is right about bandwidth and wrong about the first three seconds.
A visitor who reads the page and never runs the demo should not download an inference engine,
and a hero that arrives a beat late costs nothing. Lighthouse: desktop 100/100/100/100, mobile
96/100/100/100 with FCP 1.6 s under 4× CPU throttling.

### 76. Prefill runs in chunks of 256 queries

**Chosen:** size the prefill graph to a 256-query chunk against the full cached prefix, and run
a long prompt through it in several passes.
**Rejected:** sizing to `maxContext` (773.8 MiB of scratch); a smaller context on mobile, which
was a workaround for this and is now withdrawn.
**Decided by:** enumerating live allocations rather than reasoning about them — the two
`heads × 2048 × 2048` attention buffers were 512 MiB of 773.8 MiB of capacity. Chunked:
**99.1 MiB of scratch, 458.0 MiB resident against 1132.6.** A chunk-size sweep with `chunk =
2048` as the control showed prefill time flat within 2.7% across a 16× range, so the memory is
free.

Both predictions about speed were wrong — a fit said +13%, the first measurement said −10% —
and the second was wrong because it compared across sessions where run-to-run variance is 8%.
The control row exists so that cannot happen again.

### 77. Weights are fetched as coalesced 16 MiB spans, concurrently, through the Cache API

**Chosen:** sort every byte range, merge into spans of at most 16 MiB, fetch eight at a time
through `CachedChunkReader`, and slice each tensor out of its span.
**Rejected:** one request per range (what shipped), which is 630 serial round trips; raising
concurrency alone, which saturated at 7 MB/s against a 25–30 MB/s link.
**Decided by:** **78.3 s → 14.6 s cold, and 75.7 s → 1.4 s warm.** The quantized loader had
never used the Cache API at all, so every visit re-downloaded 335 MB. None of this was visible
over loopback, where a round trip is free and a cache miss costs nothing — which is why the M2
load numbers were honest measurements of the wrong thing.

### 78. Progress is rated against wall clock

**Chosen:** bytes divided by elapsed time since the load began.
**Rejected:** bytes divided by the sum of per-request durations, which is what it did.
**Decided by:** the sum counts overlapping requests several times over. The loader told a
visitor **16 minutes remained on a 54-second download**. A displayed number that is wrong at the
worst moment of the experience is worse than no number.

### 79. Weights are hosted on R2 behind a zone domain, not the r2.dev subdomain

**Chosen:** `models.enargeia.dev` as a custom domain on the bucket, with a CORS policy listing
the site's origins and exposing `Content-Range`; the URL comes from `VITE_MODEL_URL`, so local
development and production differ in that value and nothing else.
**Rejected:** the `pub-*.r2.dev` URL, which Cloudflare documents as development-only and rate
limits; committing 335 MB to the Pages build, which caps files at 25 MiB.
**Decided by:** the loader is built on range requests, so the host has to support them and has
to expose `Content-Range` cross-origin. R2 does both; a zone domain adds the CDN cache in front.

### 80. The pipeline section pins the section, offset by the masthead

**Chosen:** `pin: section`, `start: () => 'top ' + mastheadHeight() + 'px'`,
`invalidateOnRefresh: true`, and a `--masthead-h` custom property re-measured on resize.
**Rejected:** pinning the inner container with `start: 'top top'`, which is what shipped; a
`position: sticky` rewrite.
**Decided by:** the inner container was the wrong element and `top top` was the wrong offset,
and together they produced a section that never actually pinned and whose first line sat under
a sticky header. Sticky was rejected because it would have had the identical header-overlap
bug — the fault was the missing offset, not the mechanism.

The reported symptom was "broken in Safari". It reproduces identically in Chromium; the box
geometry at four scroll positions matches to the pixel in both engines. It survived every
previous check because every check screenshotted the section immediately after scrolling it
into view, which is the one position where a failed pin looks correct.

### 81. Cross-browser verification is a command, not a habit

**Chosen:** `npm run sweep` — every section at five widths in WebKit, with a module preflight
and automatic checks for overflow, undersized tap targets and console errors.
**Rejected:** relying on remembering to open a second browser.
**Decided by:** the site was built and shipped after being looked at only in headless Chromium,
and the first person to open it elsewhere found a broken section in seconds. Two of the six
faults the first sweep found — bars that rendered full when empty, and a masthead that
overflowed at 320 px — were browser-independent and had simply never been looked at.

### 82. The favicon is the residual stream through layers, in two hues

**Chosen:** three horizontal bars for layer boundaries, a vertical stroke through them for the
residual stream, `--k0` above and `--k6` below, on a `--bg` tile. Drawn on a 16-unit grid so
every edge lands on a device pixel at 16 CSS px.
**Rejected:** the causal-mask staircase, which reads beautifully at 16px but is attention's
image rather than the project's, and in attention's hue would have looked like a chart icon;
a segmented stream, which reads as a battery; a plain two-tone bar, same; anything with more
than two hues, which turns to noise at 16px.
**Decided by:** rasterizing every candidate to a real 16×16 PNG and comparing them in a mock tab
strip against light and dark chrome. Three of the eight first-round designs were perfectly legible
at 512 px and unreadable at 16. Judging at the size the thing is used is the whole exercise.

The two hues are the two ends of the kernel spectrum, used for what they mean: `--k0` is the
embedding, where a token enters, and `--k6` is the sampler, where it leaves. The hue changes
behind the middle bar, because that is where the stream is inside a layer. That keeps the
colour encoding intact — the rule is that a kernel hue must mean that kernel, not that hues
may never appear outside a chart.

`tools/make-icons.mjs` generates the SVG and every raster, so the apple-touch-icon and the
manifest icons cannot drift from the favicon.

### 83. Nested scroll containers carry `data-lenis-prevent`

**Chosen:** the attribute on the inspector, the chat transcript, and the composer textarea,
plus a check that finds any scrollable element lacking it.
**Rejected:** dropping Lenis; reconfiguring it to ignore events inside scrollable ancestors,
which is what the attribute already does.
**Decided by:** Lenis takes wheel events on `window`, so a nested container only scrolls if it
is excluded. Verified by removing the attribute and watching the reported bug reappear —
container 0 → 0, page 835 → 1074 — rather than by assuming the fix worked.

The pinned section needs the opposite guarantee and is asserted separately: wheeling over it
must move the page, because the page's scroll position is the scrub.

### 84. Behavioural checks, not only visual ones

**Chosen:** `tools/scroll-check.mjs`, run at the end of `npm run sweep`.
**Decided by:** the wheel bug survived a five-width, two-engine visual sweep, because a
container that can only be dragged looks identical to one that scrolls. Screenshots cannot
falsify a claim about event routing. The check runs with trusted input and starts from an
injected fault, so it has been shown to fail before being trusted to pass.

### 85. A degraded adapter is announced, in three places, with three states

**Chosen:** `DeviceProfile.software` as `no | suspected | yes`, surfaced above the chat, in the
device panel, and on the throughput reading itself.
**Rejected:** a boolean; warning only in the inspector; warning only once telemetry arrives.
**Decided by:** the number is the product. A reading two orders of magnitude below every
published figure, with nothing attached to it, is indistinguishable from a result — and the
person best placed to misread it is exactly the visitor the site is for. The throughput
annotation is set from the profile at construction for the same reason: a panel that only
becomes honest after the first generation is dishonest for the whole of the first impression.

Three states because the evidence has two strengths. A named rasterizer is certain. A blank
architecture plus absent `shader-f16` is the shape a redacted fallback takes, and also the shape
of a perfectly good Intel GPU behind a privacy-conscious browser, so it says "may be" and the
unit tests pin down every negative case.

Injected with `?forceSoftware`, following `?clampStorage`. Both directions are verified end to
end because Chromium and WebKit disagreed on this machine — Chromium on SwiftShader, WebKit on
the Apple GPU — which gave the check a genuine negative case and, incidentally, identified the
fallback as Chromium's rather than the system's.

### 86. A reply that hit the cap says so; a conversation that outgrows the context is trimmed

**Chosen:** engine-voiced notices in the transcript for both, and dropping the oldest exchanges
rather than refusing.
**Rejected:** lowering the token cap, which trades one arbitrary truncation for another;
a degeneracy detector, which is inventing policy about what the model meant.
**Decided by:** measurement. 5 of 24 generations ran to the 512-token cap, because a 0.5B model
does not reliably emit `<|im_end|>` — no engine change fixes that, so the surface has to be
honest about it instead. And refusing on a full context dead-ends the chat permanently after
about four verbose turns, which is a worse failure than losing the oldest exchange.

The reported diagnosis — missing stop-token handling — was not the cause; stop handling works
and 19 of 24 generations used it. Worth recording because the fix that followed from measuring
is a different fix from the one the symptom suggested.
