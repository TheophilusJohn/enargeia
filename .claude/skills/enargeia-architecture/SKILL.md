---
name: enargeia-architecture
description: Module boundaries, GPU resource lifetime, capability detection, and dependency rules for the Enargeia inference engine. Use this when adding a module or file, wiring the forward-pass graph, allocating or destroying GPUBuffers, managing the KV cache, handling device limits and missing features like shader-f16, loading weights, or considering adding any dependency. Also use when a change would cross the ui/runtime/model/kernels/gpu layering, when deciding where new code belongs, and when the task mentions memory, buffers, device support, or browser compatibility.
---

# Architecture

The engine is four layers with one-way imports. The constraint exists so kernels stay
testable in isolation and so the GPU layer never grows transformer-specific assumptions
that make it impossible to reuse or reason about.

```
ui  →  runtime  →  model  →  kernels  →  gpu
```

Nothing imports leftward. `src/gpu` must compile and pass its tests with no knowledge that
transformers exist. `src/ui` never touches a `GPUBuffer` — it reads snapshots the runtime
publishes.

If a change seems to require an upward import, the abstraction is wrong. Say so rather
than adding the import; the usual fix is that the lower layer needs a callback or the
value belongs in a struct the caller already owns.

## Dependencies

The engine takes none. No ONNX Runtime, no transformers.js, no WebLLM, no tfjs, no
wgpu-matrix, no linear algebra helper. Every line that touches the GPU is authored here.
This is the project's entire claim; a single convenience import invalidates it.

The *site* may use libraries — Lenis, GSAP, Three.js — because it is a separate concern.
They live under `src/ui` and never cross into the engine. See `enargeia-web`.

Before adding any dependency, state what it does, what it would replace, and why writing
it is not reasonable. Most of the time the answer is that writing it is the point.

## GPU resource lifetime

WebGPU has no garbage collection for GPU memory. A buffer not explicitly destroyed leaks
until the tab closes, and on a 384 MB model that is one reload from an out-of-memory error.

**Every buffer has an owner.** Ownership lives in one of three places:

- `BufferPool` — transient scratch, reused across dispatches, recycled by size class
- `WeightStore` — model weights, allocated once at load, destroyed on model unload
- `KVCache` — preallocated at session start, sized to max context, destroyed on reset

Nothing calls `device.createBuffer` outside these three. A bare `createBuffer` in a kernel
wrapper or a UI component is a bug even when it works, because it has no destruction path.

**Preallocate the KV cache.** Growing it per token means reallocation and copy mid-decode,
which stalls the pipeline and fragments memory. Size it once from max context and treat
running out as a session-level event, not a per-token concern.

**Pipelines are cached by content.** `PipelineCache` keys on shader source plus
specialization constants. Creating a pipeline is expensive and happens once per unique
kernel variant, never per dispatch.

## Capability detection

Device variation is a first-class concern, not an edge case. Detect once at init, store the
result in a `DeviceProfile`, and branch on the profile — never re-query features at call
sites.

```ts
interface DeviceProfile {
  f16: boolean;              // shader-f16 — roughly a third of devices lack it
  timestampQuery: boolean;   // per-kernel GPU timing for the inspector
  subgroups: boolean;
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
  tier: 'discrete' | 'integrated' | 'mobile' | 'unknown';
}
```

Request the adapter's maximum limits at device creation. Defaults are conservative —
`maxStorageBufferBindingSize` commonly defaults to 128 MiB when the hardware allows far
more, and silently accepting the default will fail on the embedding table.

**Every capability needs a fallback path that is tested, not assumed.** An f16 kernel
without an fp32 sibling means the page is blank on a third of visitors' machines. The
device panel in the UI reports which path engaged, so failures are visible rather than
mysterious.

## The embedding table

151,936 × 896 ≈ 136M parameters. It is the largest single tensor and will exceed
`maxStorageBufferBindingSize` on many devices. Split it across bindings from the start
rather than discovering the limit at load time on someone else's hardware.

The table is tied to the LM head, so both the input lookup and the output projection read
the same split representation. Keep the split as a property of `WeightStore` so callers do
not each reimplement the indexing.

## Weight loading

Two formats. `.safetensors` for the fp32 reference path used by the parity harness;
`.enargeia` for the quantized production path.

`.enargeia` layout: magic, version, JSON header with per-tensor dtype, shape, block size,
and byte offsets, then packed data. Quantization happens offline in `tools/quantize.py`
and is never done in the browser — it would require downloading fp32 weights, which is the
thing quantization exists to avoid.

Load with HTTP range requests so progress is reportable, and persist to the Cache API so a
second visit is instant. First load is the highest-abandonment moment on the whole site;
treat its responsiveness as a feature, not as loading-screen decoration.

## The graph

`model/graph.ts` builds an ordered list of dispatch descriptors once, at load. The decode
loop walks that list. It does not construct bind groups, look up pipelines, or branch on
device capability per token — all of that is resolved at build time.

This matters because 170 dispatches per token times any per-dispatch JavaScript work is
measurable overhead, and because a decode loop that only walks a list is a decode loop you
can read.

## Where new code goes

| Kind of change | Home |
|---|---|
| New shader | `src/kernels/<name>.wgsl` + `.ts` wrapper |
| Buffer allocation strategy | `src/gpu/pool.ts` |
| New model architecture support | `src/model/config.ts` + `graph.ts` |
| Sampling strategy | `src/kernels/sample_*.wgsl`, orchestrated from `src/runtime` |
| Anything the user sees | `src/ui` |
| Python tooling | `tools/` — never shipped to the browser |

When a change does not obviously fit, that is usually a signal the layering is being
violated. Raise it rather than picking the nearest directory.
