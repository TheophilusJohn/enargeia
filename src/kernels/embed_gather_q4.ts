import code from './embed_gather_q4.wgsl?raw';
import { coverage, type KernelSpec } from './kernel.ts';

/** ids, packed, scales, zeros, out, dims. Dispatched once per embedding part. */
export const EMBED_GATHER_Q4: KernelSpec = {
  name: 'embed_gather_q4',
  code,
  bindings: ['read', 'read', 'read', 'read', 'read_write', 'uniform'],
  workgroupSize: [64, 1, 1],
  uniformBytes: 32,
};

export function embedQ4Dims(
  seq: number,
  hidden: number,
  rowBegin: number,
  rowCount: number,
  blockSize: number,
): ArrayBuffer {
  return new Uint32Array([
    seq, hidden, rowBegin, rowCount, blockSize, hidden / blockSize, 0, 0,
  ]).buffer;
}

export function embedQ4Workgroups(seq: number, hidden: number): [number, number, number] {
  const [x] = coverage(EMBED_GATHER_Q4, [hidden, 1, 1]);
  return [x, seq, 1];
}
