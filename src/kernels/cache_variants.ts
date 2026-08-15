/**
 * The four attention kernels, generated per KV cache dtype.
 *
 * The cache is the largest bandwidth term in decode above a 512 context — 35% of the step at
 * 2048 — and halving it is the one f16 change whose mechanism is actually bandwidth. See the
 * decomposition in BENCH.md.
 *
 * f16 here means `pack2x16float` / `unpack2x16float`, which are **core WGSL**, not the
 * `shader-f16` extension. The cache holds halves; every value is f32 the moment it is read.
 * That is what lets this ship without an fp32 sibling kernel and a device-capability branch —
 * the same trick the f16 embedding uses.
 *
 * Writes go through a small pack kernel rather than being fused into the projections. Two
 * adjacent halves share a u32, and the threads that compute them are different threads, so a
 * fused write would need either a thread remapping or an atomic. One extra dispatch per layer
 * per token, over 128 elements, is cheaper than either.
 */

import { coverage, type KernelSpec } from './kernel.ts';

export type CacheDType = 'f32' | 'f16';

/** Bytes per cached element. */
export function cacheElementBytes(dtype: CacheDType): number {
  return dtype === 'f16' ? 2 : 4;
}

/** Reading one cached element as f32. */
function readCache(dtype: CacheDType, array: string): string {
  return dtype === 'f16'
    ? `
fn ${array}At(i: u32) -> f32 {
    let pair = unpack2x16float(${array}[i >> 1u]);
    return select(pair.x, pair.y, (i & 1u) == 1u);
}`
    : `
fn ${array}At(i: u32) -> f32 {
    return bitcast<f32>(${array}[i]);
}`;
}

const DECODE_DIMS = `
struct Dims {
    position: u32,
    heads: u32,
    kvHeads: u32,
    headDim: u32,
    scoreStride: u32,
    cacheStride: u32,
    _pad0: u32,
    _pad1: u32,
};
`;

/**
 * Prefill attention over a chunk of queries against the whole cached prefix.
 *
 * `queries` is the chunk length and `keys` is everything cached so far including it, so the
 * score tensor is a `queries × keys` strip rather than a `context × context` square. That is
 * what lets a 2048-token prompt run without allocating 14 × 2048 × 2048 twice — see the
 * residency measurement in BENCH.md. When the whole prompt fits in one chunk the two are equal
 * and the shapes are exactly what they were before.
 *
 * `queryBegin` is the absolute position of the chunk's first query. The causal mask compares
 * against `queryBegin + i`, because a query in chunk 3 may attend to every key in chunks 0-2.
 */
const PREFILL_DIMS = `
struct Dims {
    queries: u32,
    keys: u32,
    queryBegin: u32,
    heads: u32,
    kvHeads: u32,
    headDim: u32,
    _pad0: u32,
    _pad1: u32,
};
`;

/** Prefill scores: q x cached K over the full causal triangle. */
export function attnScoresCode(dtype: CacheDType): string {
  return `// Generated for KV cache dtype: ${dtype}
const NEG_LARGE: f32 = -1.0e30;
${PREFILL_DIMS}
@group(0) @binding(0) var<storage, read>       q:    array<f32>;
@group(0) @binding(1) var<storage, read>       k:    array<u32>;
@group(0) @binding(2) var<storage, read_write> out:  array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;
${readCache(dtype, 'k')}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let i = gid.y;
    let h = gid.z;
    if (j >= dims.keys || i >= dims.queries || h >= dims.heads) {
        return;
    }
    let slot = (h * dims.queries + i) * dims.keys + j;
    if (j > i + dims.queryBegin) {
        out[slot] = NEG_LARGE;
        return;
    }
    let kvHead = h / (dims.heads / dims.kvHeads);
    let qBase = (i * dims.heads + h) * dims.headDim;
    let kBase = (j * dims.kvHeads + kvHead) * dims.headDim;
    var acc = 0.0;
    for (var d = 0u; d < dims.headDim; d = d + 1u) {
        acc = acc + q[qBase + d] * kAt(kBase + d);
    }
    out[slot] = acc * inverseSqrt(f32(dims.headDim));
}
`;
}

/** Prefill apply: attention weights x cached V. */
export function attnApplyCode(dtype: CacheDType): string {
  return `// Generated for KV cache dtype: ${dtype}
${PREFILL_DIMS}
@group(0) @binding(0) var<storage, read>       weights: array<f32>;
@group(0) @binding(1) var<storage, read>       v:       array<u32>;
@group(0) @binding(2) var<storage, read_write> out:     array<f32>;
@group(0) @binding(3) var<uniform>             dims:    Dims;
${readCache(dtype, 'v')}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let perRow = dims.heads * dims.headDim;
    if (gid.x >= perRow || gid.y >= dims.queries) {
        return;
    }
    let i = gid.y;
    let h = gid.x / dims.headDim;
    let d = gid.x % dims.headDim;
    let kvHead = h / (dims.heads / dims.kvHeads);
    let wBase = (h * dims.queries + i) * dims.keys;
    var acc = 0.0;
    for (var j = 0u; j <= i + dims.queryBegin; j = j + 1u) {
        acc = acc + weights[wBase + j] * vAt((j * dims.kvHeads + kvHead) * dims.headDim + d);
    }
    out[i * perRow + gid.x] = acc;
}
`;
}

/** Decode scores: one query against the whole cached history. */
export function attnScoresDecodeCode(dtype: CacheDType): string {
  return `// Generated for KV cache dtype: ${dtype}
const NEG_LARGE: f32 = -1.0e30;
${DECODE_DIMS}
@group(0) @binding(0) var<storage, read>       q:      array<f32>;
@group(0) @binding(1) var<storage, read>       cacheK: array<u32>;
@group(0) @binding(2) var<storage, read_write> out:    array<f32>;
@group(0) @binding(3) var<uniform>             dims:   Dims;
${readCache(dtype, 'cacheK')}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let h = gid.y;
    if (h >= dims.heads || j >= dims.scoreStride) {
        return;
    }
    let slot = h * dims.scoreStride + j;
    if (j > dims.position) {
        out[slot] = NEG_LARGE;
        return;
    }
    let kvHead = h / (dims.heads / dims.kvHeads);
    let qBase = h * dims.headDim;
    let kBase = j * dims.cacheStride + kvHead * dims.headDim;
    var acc = 0.0;
    for (var d = 0u; d < dims.headDim; d = d + 1u) {
        acc = acc + q[qBase + d] * cacheKAt(kBase + d);
    }
    out[slot] = acc * inverseSqrt(f32(dims.headDim));
}
`;
}

/**
 * Decode apply: weights x cached V for one query position.
 *
 * `unroll` is a diagnostic knob, not a tuning knob. It processes several positions per loop
 * iteration with independent accumulators, which holds the bytes moved constant while cutting
 * iteration count and breaking the serial accumulator dependency. That is the variation which
 * separates "cost tracks iterations" from "cost tracks occupancy" — both of which look
 * identical in a plot of time against context.
 */
export function attnApplyDecodeCode(dtype: CacheDType, unroll = 1): string {
  if (unroll === 1) {
    return attnApplyDecodeSerial(dtype);
  }
  const accs = Array.from({ length: unroll }, (_, u) => `    var acc${u} = 0.0;`).join('\n');
  const body = Array.from({ length: unroll }, (_, u) =>
    `        acc${u} = acc${u} + weights[wBase + j + ${u}u] * ` +
    `cacheVAt((j + ${u}u) * dims.cacheStride + kvHead * dims.headDim + d);`).join('\n');
  const sum = Array.from({ length: unroll }, (_, u) => `acc${u}`).join(' + ');
  return `// Generated for KV cache dtype: ${dtype}, unroll ${unroll}
${DECODE_DIMS}
@group(0) @binding(0) var<storage, read>       weights: array<f32>;
@group(0) @binding(1) var<storage, read>       cacheV:  array<u32>;
@group(0) @binding(2) var<storage, read_write> out:     array<f32>;
@group(0) @binding(3) var<uniform>             dims:    Dims;
${readCache(dtype, 'cacheV')}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = dims.heads * dims.headDim;
    if (gid.x >= total) {
        return;
    }
    let h = gid.x / dims.headDim;
    let d = gid.x % dims.headDim;
    let kvHead = h / (dims.heads / dims.kvHeads);
    let wBase = h * dims.scoreStride;
    let n = dims.position + 1u;
    let limit = (n / ${unroll}u) * ${unroll}u;

${accs}
    var j = 0u;
    for (; j < limit; j = j + ${unroll}u) {
${body}
    }
    var acc = ${sum};
    for (; j < n; j = j + 1u) {
        acc = acc + weights[wBase + j] * cacheVAt(j * dims.cacheStride + kvHead * dims.headDim + d);
    }
    out[gid.x] = acc;
}
`;
}

function attnApplyDecodeSerial(dtype: CacheDType): string {
  return `// Generated for KV cache dtype: ${dtype}
${DECODE_DIMS}
@group(0) @binding(0) var<storage, read>       weights: array<f32>;
@group(0) @binding(1) var<storage, read>       cacheV:  array<u32>;
@group(0) @binding(2) var<storage, read_write> out:     array<f32>;
@group(0) @binding(3) var<uniform>             dims:    Dims;
${readCache(dtype, 'cacheV')}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = dims.heads * dims.headDim;
    if (gid.x >= total) {
        return;
    }
    let h = gid.x / dims.headDim;
    let d = gid.x % dims.headDim;
    let kvHead = h / (dims.heads / dims.kvHeads);
    let wBase = h * dims.scoreStride;
    var acc = 0.0;
    for (var j = 0u; j <= dims.position; j = j + 1u) {
        acc = acc + weights[wBase + j] * cacheVAt(j * dims.cacheStride + kvHead * dims.headDim + d);
    }
    out[gid.x] = acc;
}
`;
}

/**
 * Pack f32 into the cache.
 *
 * `count` elements from `src`, written at element offset `dstOffset` in the cache. For an f32
 * cache this is a copy; for f16 each thread packs the pair it owns, which is why the offset
 * must be even — guaranteed because a cache row is `kvHeads * headDim` = 128 elements.
 */
export function cachePackCode(dtype: CacheDType): string {
  const body =
    dtype === 'f16'
      ? `
    // One thread per output word, packing the two halves it owns. Reading both sources here
    // rather than having two threads write one word is what avoids an atomic.
    let base = (dims.dstOffset + gid.x * 2u);
    let a = src[gid.x * 2u];
    let b = select(0.0, src[gid.x * 2u + 1u], gid.x * 2u + 1u < dims.count);
    dst[base >> 1u] = pack2x16float(vec2<f32>(a, b));`
      : `
    dst[dims.dstOffset + gid.x] = bitcast<u32>(src[gid.x]);`;
  const guard = dtype === 'f16' ? '(dims.count + 1u) / 2u' : 'dims.count';
  return `// Generated for KV cache dtype: ${dtype}
struct Dims {
    count: u32,
    dstOffset: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       src:  array<f32>;
@group(0) @binding(1) var<storage, read_write> dst:  array<u32>;
@group(0) @binding(2) var<uniform>             dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= ${guard}) {
        return;
    }${body}
}
`;
}

function spec(name: string, code: string, bindings: readonly ('read' | 'read_write' | 'uniform')[], uniformBytes: number, wg: number): KernelSpec {
  return { name, code, bindings, workgroupSize: [wg, wg === 16 ? 16 : 1, 1], uniformBytes };
}

export function attnScoresSpec(dtype: CacheDType): KernelSpec {
  return spec(`attn_scores_${dtype}`, attnScoresCode(dtype), ['read', 'read', 'read_write', 'uniform'], 32, 16);
}
export function attnApplySpec(dtype: CacheDType): KernelSpec {
  return spec(`attn_apply_${dtype}`, attnApplyCode(dtype), ['read', 'read', 'read_write', 'uniform'], 32, 64);
}
export function attnScoresDecodeSpec(dtype: CacheDType): KernelSpec {
  return spec(`attn_scores_decode_${dtype}`, attnScoresDecodeCode(dtype), ['read', 'read', 'read_write', 'uniform'], 32, 64);
}
export function attnApplyDecodeSpec(dtype: CacheDType, unroll = 1): KernelSpec {
  return spec(
    `attn_apply_decode_${dtype}_u${unroll}`,
    attnApplyDecodeCode(dtype, unroll),
    ['read', 'read', 'read_write', 'uniform'],
    32,
    64,
  );
}
export function cachePackSpec(dtype: CacheDType): KernelSpec {
  return spec(`cache_pack_${dtype}`, cachePackCode(dtype), ['read', 'read_write', 'uniform'], 16, 64);
}

export function cachePackDims(count: number, dstOffset: number): ArrayBuffer {
  return new Uint32Array([count, dstOffset, 0, 0]).buffer;
}

export function cachePackWorkgroups(dtype: CacheDType, count: number): [number, number, number] {
  const words = dtype === 'f16' ? Math.ceil(count / 2) : count;
  return coverage(cachePackSpec(dtype), [words, 1, 1]);
}


/**
 * Decode apply, parallelised across the history.
 *
 * One workgroup per output element instead of one thread. The 64 threads split the history
 * between them and reduce in shared memory, so cost tracks `history / 64` rather than
 * `history`, and the dispatch goes from 14 workgroups to 896.
 *
 * Both numbers matter, and a probe established that before this was written. Holding bytes
 * constant while cutting iterations 8x (an unrolled variant) bought 1.39x — real, so iteration
 * latency is part of it, but far short of 8x, so it is not all of it. The residue is occupancy:
 * 14 workgroups on a 10-core GPU leaves most of the machine idle, and no amount of doing less
 * work per thread fixes that. This kernel addresses both at once — fewer iterations per thread
 * *and* 64x the workgroups.
 */
export function attnApplyDecodeParallelCode(dtype: CacheDType): string {
  return `// Generated for KV cache dtype: ${dtype}, parallel reduction
const WG: u32 = 64u;
${DECODE_DIMS}
@group(0) @binding(0) var<storage, read>       weights: array<f32>;
@group(0) @binding(1) var<storage, read>       cacheV:  array<u32>;
@group(0) @binding(2) var<storage, read_write> out:     array<f32>;
@group(0) @binding(3) var<uniform>             dims:    Dims;
${readCache(dtype, 'cacheV')}

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id)        wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let outIndex = wid.x;
    let tid = lid.x;
    let h = outIndex / dims.headDim;
    let d = outIndex % dims.headDim;
    let kvHead = h / (dims.heads / dims.kvHeads);
    let wBase = h * dims.scoreStride;

    // Strided so the 64 threads cover the history evenly whatever its length. The sum is over
    // a different order than the serial kernel used, which is a different floating-point
    // association — deterministic, but not bit-identical to the old kernel.
    var acc = 0.0;
    for (var j = tid; j <= dims.position; j = j + WG) {
        acc = acc + weights[wBase + j] * cacheVAt(j * dims.cacheStride + kvHead * dims.headDim + d);
    }

    partial[tid] = acc;
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            partial[tid] = partial[tid] + partial[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        out[outIndex] = partial[0];
    }
}
`;
}

export function attnApplyDecodeParallelSpec(dtype: CacheDType): KernelSpec {
  return spec(
    `attn_apply_decode_parallel_${dtype}`,
    attnApplyDecodeParallelCode(dtype),
    ['read', 'read', 'read_write', 'uniform'],
    32,
    64,
  );
}

/** One workgroup per output element. */
export function attnApplyDecodeParallelWorkgroups(heads: number, headDim: number): [number, number, number] {
  return [heads * headDim, 1, 1];
}
