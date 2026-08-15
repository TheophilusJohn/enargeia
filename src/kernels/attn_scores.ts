import code from './attn_scores.wgsl?raw';
import { coverage, type KernelSpec } from './kernel.ts';

/** q, k, out, dims. Dispatched over (key position, query position, head). */
export const ATTN_SCORES: KernelSpec = {
  name: 'attn_scores',
  code,
  bindings: ['read', 'read', 'read_write', 'uniform'],
  workgroupSize: [16, 16, 1],
  uniformBytes: 32,
};

/**
 * `queries` rows against `keys` columns, with the chunk starting at absolute `queryBegin`.
 * A prompt that fits in one chunk passes (n, n, 0), which is the pre-chunking shape.
 */
export function attnDims(
  queries: number,
  keys: number,
  queryBegin: number,
  heads: number,
  kvHeads: number,
  headDim: number,
): ArrayBuffer {
  const buffer = new ArrayBuffer(32);
  new Uint32Array(buffer, 0, 6).set([queries, keys, queryBegin, heads, kvHeads, headDim]);
  return buffer;
}

export function attnScoresWorkgroups(
  queries: number,
  keys: number,
  heads: number,
): [number, number, number] {
  const [x, y] = coverage(ATTN_SCORES, [keys, queries, 1]);
  return [x, y, heads];
}
