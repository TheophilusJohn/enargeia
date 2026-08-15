/**
 * The embedding table's two consumers, generated per storage dtype.
 *
 * The tied table is the one tensor whose precision is worth varying: it is both the input
 * lookup and the output projection, it is 72% of the int4 model's resident bytes, and the
 * output projection maps straight to logits with no downstream layer to absorb an error. So
 * the gather and the LM head are templated on the dtype, and every other weight stays int4.
 *
 * Four formats, all read as f32 in the shader:
 *
 *   q4    eight 4-bit values per u32, f32 scale and 4-bit zero-point per block of 64
 *   q8    four 8-bit values per u32,  f32 scale and 8-bit zero-point per block of 64
 *   f16   two halves per u32, unpacked with `unpack2x16float`
 *   f32   plain floats
 *
 * f16 uses `unpack2x16float`, which is core WGSL and *not* the `shader-f16` extension. That
 * matters: it keeps the f16 path on the same universal code path as the others rather than
 * needing an fp32 sibling and a tested fallback for the third of devices without the
 * extension. The values are half-precision in memory and f32 in registers.
 *
 * Each dtype gets its own pipeline rather than one kernel branching on a uniform, so the
 * inner loop stays branch-free. The pipeline cache keys on source, so templating is free.
 */

import { coverage, type KernelSpec } from './kernel.ts';

export type EmbeddingDType = 'q4' | 'q8' | 'f16' | 'f32';

/** Storage bytes the embedding occupies at each dtype, for a rows x cols table. */
export function embeddingBytes(dtype: EmbeddingDType, rows: number, cols: number, blockSize = 64): number {
  const elements = rows * cols;
  const blocks = elements / blockSize;
  switch (dtype) {
    case 'q4':
      return (elements / 8) * 4 + blocks * 4 + Math.ceil(blocks / 8) * 4;
    case 'q8':
      return (elements / 4) * 4 + blocks * 4 + Math.ceil(blocks / 4) * 4;
    case 'f16':
      return elements * 2;
    case 'f32':
      return elements * 4;
  }
}

/** The body of `weightAt(row, col) -> f32`, per dtype. */
function dequantBody(dtype: EmbeddingDType): string {
  switch (dtype) {
    case 'q4':
      return `
    let flat = row * cols + col;
    let word = packed[flat >> 3u];
    let q = (word >> ((flat & 7u) * 4u)) & 0xFu;
    let block = row * blocksPerRow + col / blockSize;
    let z = (zeros[block >> 3u] >> ((block & 7u) * 4u)) & 0xFu;
    return (f32(q) - f32(z)) * scales[block];`;
    case 'q8':
      return `
    let flat = row * cols + col;
    let word = packed[flat >> 2u];
    let q = (word >> ((flat & 3u) * 8u)) & 0xFFu;
    let block = row * blocksPerRow + col / blockSize;
    let z = (zeros[block >> 2u] >> ((block & 3u) * 8u)) & 0xFFu;
    return (f32(q) - f32(z)) * scales[block];`;
    case 'f16':
      return `
    let flat = row * cols + col;
    // unpack2x16float is core WGSL, not the shader-f16 extension: half in memory, f32 here.
    let pair = unpack2x16float(packed[flat >> 1u]);
    return select(pair.x, pair.y, (flat & 1u) == 1u);`;
    case 'f32':
      return `
    return bitcast<f32>(packed[row * cols + col]);`;
  }
}

const HEADER = (dtype: EmbeddingDType) => `// Generated for embedding dtype: ${dtype}
struct Dims {
    seq: u32,
    hidden: u32,
    rowBegin: u32,
    rowCount: u32,
    blockSize: u32,
    blocksPerRow: u32,
    outStride: u32,
    outOffset: u32,
    gridWidth: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
`;

/**
 * Gather: out[s, :] = table[ids[s], :]
 *
 * One dispatch per part, a thread writing its row only when the token id falls in this part's
 * range — the same structure as the fp32 gather, because the split is a property of the table
 * and not of its precision.
 */
export function embeddingGatherCode(dtype: EmbeddingDType): string {
  return `${HEADER(dtype)}
@group(0) @binding(0) var<storage, read>       ids:    array<u32>;
@group(0) @binding(1) var<storage, read>       packed: array<u32>;
@group(0) @binding(2) var<storage, read>       scales: array<f32>;
@group(0) @binding(3) var<storage, read>       zeros:  array<u32>;
@group(0) @binding(4) var<storage, read_write> out:    array<f32>;
@group(0) @binding(5) var<uniform>             dims:   Dims;

fn weightAt(row: u32, col: u32, cols: u32, blockSize: u32, blocksPerRow: u32) -> f32 {${dequantBody(dtype)}
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= dims.hidden || gid.y >= dims.seq) {
        return;
    }
    let id = ids[gid.y];
    if (id < dims.rowBegin || id >= dims.rowBegin + dims.rowCount) {
        return;
    }
    let localRow = id - dims.rowBegin;
    out[gid.y * dims.hidden + gid.x] =
        weightAt(localRow, gid.x, dims.hidden, dims.blockSize, dims.blocksPerRow);
}
`;
}

/**
 * Tied LM head: out[n] = sum_k x[k] * table[n, k]
 *
 * Decode-shaped — m is always 1 here — so every weight is read once and there is no reuse to
 * amortize an unpack against. The grid folds into two dimensions because 151,936 workgroups
 * exceeds `maxComputeWorkgroupsPerDimension`.
 */
export function embeddingHeadCode(dtype: EmbeddingDType): string {
  return `${HEADER(dtype)}
const WG: u32 = 64u;

@group(0) @binding(0) var<storage, read>       x:      array<f32>;
@group(0) @binding(1) var<storage, read>       packed: array<u32>;
@group(0) @binding(2) var<storage, read>       scales: array<f32>;
@group(0) @binding(3) var<storage, read>       zeros:  array<u32>;
@group(0) @binding(4) var<storage, read_write> out:    array<f32>;
@group(0) @binding(5) var<uniform>             dims:   Dims;

var<workgroup> partial: array<f32, 64>;

fn weightAt(row: u32, col: u32, cols: u32, blockSize: u32, blocksPerRow: u32) -> f32 {${dequantBody(dtype)}
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id)        wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let col = wid.y * dims.gridWidth + wid.x;
    let tid = lid.x;
    // No early return: the reduction has barriers and they must stay in uniform control flow.
    let colActive = col < dims.rowCount;

    var acc = 0.0;
    if (colActive) {
        for (var i = tid; i < dims.hidden; i = i + WG) {
            acc = acc + weightAt(col, i, dims.hidden, dims.blockSize, dims.blocksPerRow) * x[i];
        }
    }
    partial[tid] = acc;
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            partial[tid] = partial[tid] + partial[tid + stride];
        }
        workgroupBarrier();
    }
    if (tid == 0u && colActive) {
        out[col + dims.outOffset] = partial[0];
    }
}
`;
}

const BINDINGS = ['read', 'read', 'read', 'read', 'read_write', 'uniform'] as const;
const UNIFORM_BYTES = 48;

export function embeddingGatherSpec(dtype: EmbeddingDType): KernelSpec {
  return {
    name: `embed_gather_${dtype}`,
    code: embeddingGatherCode(dtype),
    bindings: BINDINGS,
    workgroupSize: [64, 1, 1],
    uniformBytes: UNIFORM_BYTES,
  };
}

export function embeddingHeadSpec(dtype: EmbeddingDType): KernelSpec {
  return {
    name: `lm_head_${dtype}`,
    code: embeddingHeadCode(dtype),
    bindings: BINDINGS,
    workgroupSize: [64, 1, 1],
    uniformBytes: UNIFORM_BYTES,
  };
}

export const MAX_WORKGROUPS_PER_DIMENSION = 65535;

export function embeddingDims(
  seq: number,
  hidden: number,
  rowBegin: number,
  rowCount: number,
  blockSize: number,
  outStride = 0,
  outOffset = 0,
): ArrayBuffer {
  return new Uint32Array([
    seq, hidden, rowBegin, rowCount,
    blockSize, hidden / blockSize, outStride, outOffset,
    Math.min(Math.max(rowCount, 1), MAX_WORKGROUPS_PER_DIMENSION), 0, 0, 0,
  ]).buffer;
}

export function embeddingGatherWorkgroups(
  spec: KernelSpec,
  seq: number,
  hidden: number,
): [number, number, number] {
  const [x] = coverage(spec, [hidden, 1, 1]);
  return [x, seq, 1];
}

/** One workgroup per vocabulary entry, folded into two dimensions. */
export function embeddingHeadWorkgroups(rows: number): [number, number, number] {
  const width = Math.min(rows, MAX_WORKGROUPS_PER_DIMENSION);
  return [width, Math.ceil(rows / width), 1];
}
