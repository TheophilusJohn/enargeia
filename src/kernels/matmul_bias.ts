import code from './matmul_bias.wgsl?raw';
import { coverage, type KernelSpec } from './kernel.ts';

/**
 * x, w, bias, out, dims. `w` is [n, k] — checkpoint layout — so this reads its second
 * operand transposed. Bias is always bound; `useBias` selects whether it is added.
 */
export const MATMUL_BIAS: KernelSpec = {
  name: 'matmul_bias',
  code,
  bindings: ['read', 'read', 'read', 'read_write', 'uniform'],
  workgroupSize: [16, 16, 1],
  uniformBytes: 32,
};

export function matmulBiasDims(
  m: number,
  n: number,
  k: number,
  useBias: boolean,
  outStride = n,
  outOffset = 0,
): ArrayBuffer {
  return new Uint32Array([m, n, k, useBias ? 1 : 0, outStride, outOffset, 0, 0]).buffer;
}

export function matmulBiasWorkgroups(m: number, n: number): [number, number, number] {
  return coverage(MATMUL_BIAS, [n, m, 1]);
}
