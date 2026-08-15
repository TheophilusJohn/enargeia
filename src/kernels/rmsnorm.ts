import code from './rmsnorm.wgsl?raw';
import { coverage, type KernelSpec } from './kernel.ts';

/** x, weight, out, dims. One workgroup per row — the reduction is over `features`. */
export const RMSNORM: KernelSpec = {
  name: 'rmsnorm',
  code,
  bindings: ['read', 'read', 'read_write', 'uniform'],
  workgroupSize: [256, 1, 1],
  uniformBytes: 16,
};

export function rmsnormDims(seq: number, features: number, eps: number): ArrayBuffer {
  const buffer = new ArrayBuffer(16);
  new Uint32Array(buffer, 0, 2).set([seq, features]);
  new Float32Array(buffer, 8, 1)[0] = eps;
  return buffer;
}

/** One workgroup per row, not one per 256 elements. */
export function rmsnormWorkgroups(seq: number): [number, number, number] {
  return [seq, 1, 1];
}
export { coverage };
