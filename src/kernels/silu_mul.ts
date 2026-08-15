import code from './silu_mul.wgsl?raw';
import { coverage, type KernelSpec } from './kernel.ts';

/** gate, up, out, dims. Elementwise over the MLP intermediate. */
export const SILU_MUL: KernelSpec = {
  name: 'silu_mul',
  code,
  bindings: ['read', 'read', 'read_write', 'uniform'],
  workgroupSize: [256, 1, 1],
  uniformBytes: 16,
};

export function countDims(count: number): ArrayBuffer {
  return new Uint32Array([count, 0, 0, 0]).buffer;
}

export function elementwiseWorkgroups(count: number): [number, number, number] {
  return coverage(SILU_MUL, [count, 1, 1]);
}
