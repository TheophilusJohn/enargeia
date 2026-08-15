/**
 * Dispatch wrappers for the int4 projections.
 *
 * Two kernels, not one with a branch. Prefill is matrix-by-matrix and has reuse to amortize
 * the nibble unpacking against; decode is matrix-by-vector, reads every weight exactly once,
 * and is bound entirely by how fast the packed bytes arrive. The kernels skill is explicit
 * that a kernel tuned for one underperforms on the other.
 *
 * Both share a bind group layout — x, packed, scales, zeros, bias, out, dims — so the graph
 * can pick between them per dispatch without rebuilding anything.
 */

import prefillCode from './matmul_q4_prefill.wgsl?raw';
import decodeCode from './matmul_q4_decode.wgsl?raw';
import type { KernelSpec } from './kernel.ts';

const BINDINGS = ['read', 'read', 'read', 'read', 'read', 'read_write', 'uniform'] as const;
const UNIFORM_BYTES = 48;

/** 4 rows x 2 columns per thread, 64x32 output footprint. */
export const MATMUL_Q4_PREFILL: KernelSpec = {
  name: 'matmul_q4_prefill',
  code: prefillCode,
  bindings: BINDINGS,
  workgroupSize: [16, 16, 1],
  uniformBytes: UNIFORM_BYTES,
};

/** One workgroup per output element, 64 threads splitting the reduction. */
export const MATMUL_Q4_DECODE: KernelSpec = {
  name: 'matmul_q4_decode',
  code: decodeCode,
  bindings: BINDINGS,
  workgroupSize: [64, 1, 1],
  uniformBytes: UNIFORM_BYTES,
};

export const MATMUL_Q4_PREFILL_TILE = { rows: 64, cols: 32 } as const;

/** Shared memory the prefill kernel reserves: a 64x16 activation tile and a 16x32 weight tile. */
export const MATMUL_Q4_PREFILL_WORKGROUP_BYTES = (64 * 16 + 16 * 32) * 4;

/**
 * Largest workgroup count along one dimension that WebGPU guarantees. The real limit is a
 * device property and is usually exactly this; folding at the guaranteed value keeps the
 * dispatch valid everywhere rather than only where the device is generous.
 */
export const MAX_WORKGROUPS_PER_DIMENSION = 65535;

export function matmulQ4Dims(
  m: number,
  n: number,
  k: number,
  useBias: boolean,
  blockSize: number,
  outStride = n,
  outOffset = 0,
): ArrayBuffer {
  return new Uint32Array([
    m, n, k, useBias ? 1 : 0,
    blockSize, k / blockSize, outStride, outOffset,
    decodeGridWidth(n), 0, 0, 0,
  ]).buffer;
}

/** Workgroups along x for the decode kernel; the rest fold into y. */
export function decodeGridWidth(n: number): number {
  return Math.min(n, MAX_WORKGROUPS_PER_DIMENSION);
}

/** One workgroup per 64 rows by 32 columns of output. */
export function matmulQ4PrefillWorkgroups(m: number, n: number): [number, number, number] {
  return [
    Math.max(1, Math.ceil(n / MATMUL_Q4_PREFILL_TILE.cols)),
    Math.max(1, Math.ceil(m / MATMUL_Q4_PREFILL_TILE.rows)),
    1,
  ];
}

/**
 * One workgroup per output column, folded into two dimensions.
 *
 * The tied LM head has 151,936 columns, well past the 65,535 per-dimension limit. A dispatch
 * over the limit is not an encode-time error — it invalidates the command buffer, and the
 * forward pass silently produces zeros.
 */
export function matmulQ4DecodeWorkgroups(n: number): [number, number, number] {
  const width = decodeGridWidth(n);
  return [width, Math.ceil(n / width), 1];
}
