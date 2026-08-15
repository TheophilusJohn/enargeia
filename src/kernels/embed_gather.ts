import code from './embed_gather.wgsl?raw';
import { coverage, type KernelSpec } from './kernel.ts';

/**
 * ids, part, out, dims. Dispatched once per embedding part — never with all parts bound at
 * once, which would need seven storage buffers against a limit of ten.
 */
export const EMBED_GATHER: KernelSpec = {
  name: 'embed_gather',
  code,
  bindings: ['read', 'read', 'read_write', 'uniform'],
  workgroupSize: [64, 1, 1],
  uniformBytes: 16,
};

export function embedDims(
  seq: number,
  hidden: number,
  rowBegin: number,
  rowCount: number,
): ArrayBuffer {
  return new Uint32Array([seq, hidden, rowBegin, rowCount]).buffer;
}

export function embedWorkgroups(seq: number, hidden: number): [number, number, number] {
  const [x] = coverage(EMBED_GATHER, [hidden, 1, 1]);
  return [x, seq, 1];
}
