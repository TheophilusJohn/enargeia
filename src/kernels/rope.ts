import code from './rope.wgsl?raw';
import { coverage, type KernelSpec } from './kernel.ts';

/** x, out, dims. One thread per rotated pair: seq * heads * headDim/2 of them. */
export const ROPE: KernelSpec = {
  name: 'rope',
  code,
  bindings: ['read', 'read_write', 'uniform'],
  workgroupSize: [64, 1, 1],
  uniformBytes: 32,
};

export function ropeDims(
  seq: number,
  heads: number,
  headDim: number,
  positionOffset: number,
  theta: number,
  outOffset = 0,
): ArrayBuffer {
  const buffer = new ArrayBuffer(32);
  new Uint32Array(buffer, 0, 4).set([seq, heads, headDim, positionOffset]);
  new Float32Array(buffer, 16, 1)[0] = theta;
  new Uint32Array(buffer, 20, 1)[0] = outOffset;
  return buffer;
}

export function ropeWorkgroups(seq: number, heads: number, headDim: number): [number, number, number] {
  return coverage(ROPE, [seq * heads * (headDim / 2), 1, 1]);
}
