import code from './attn_scores.wgsl?raw';
import { coverage, type KernelSpec } from './kernel.ts';

/** q, k, out, dims. Dispatched over (key position, query position, head). */
export const ATTN_SCORES: KernelSpec = {
  name: 'attn_scores',
  code,
  bindings: ['read', 'read', 'read_write', 'uniform'],
  workgroupSize: [16, 16, 1],
  uniformBytes: 16,
};

export function attnDims(seq: number, heads: number, kvHeads: number, headDim: number): ArrayBuffer {
  return new Uint32Array([seq, heads, kvHeads, headDim]).buffer;
}

export function attnScoresWorkgroups(seq: number, heads: number): [number, number, number] {
  const [x, y] = coverage(ATTN_SCORES, [seq, seq, 1]);
  return [x, y, heads];
}
