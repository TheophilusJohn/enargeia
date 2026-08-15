import code from './argmax.wgsl?raw';
import type { KernelSpec } from './kernel.ts';

/**
 * logits, out, dims. One workgroup total — the whole vocabulary is reduced by 256 threads.
 * Writes [index, valueBits]; only those eight bytes ever cross back to JavaScript.
 */
export const ARGMAX: KernelSpec = {
  name: 'argmax',
  code,
  bindings: ['read', 'read_write', 'uniform'],
  workgroupSize: [256, 1, 1],
  uniformBytes: 16,
};

export function argmaxDims(count: number, offset: number): ArrayBuffer {
  return new Uint32Array([count, offset, 0, 0]).buffer;
}

export const ARGMAX_WORKGROUPS: [number, number, number] = [1, 1, 1];
