import code from './softmax.wgsl?raw';
import type { KernelSpec } from './kernel.ts';

/** x, out, dims. One workgroup per row; the reduction is over `cols`. */
export const SOFTMAX: KernelSpec = {
  name: 'softmax',
  code,
  bindings: ['read', 'read_write', 'uniform'],
  workgroupSize: [256, 1, 1],
  uniformBytes: 16,
};

export function softmaxDims(rows: number, cols: number, stride = cols): ArrayBuffer {
  return new Uint32Array([rows, cols, stride, 0]).buffer;
}

export function softmaxWorkgroups(rows: number): [number, number, number] {
  return [rows, 1, 1];
}
