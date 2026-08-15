import scoresCode from './attn_decode.wgsl?raw';
import applyCode from './attn_apply_decode.wgsl?raw';
import { coverage, type KernelSpec } from './kernel.ts';

/** q, cacheK, out, dims. Dispatched over (key position, head). */
export const ATTN_SCORES_DECODE: KernelSpec = {
  name: 'attn_scores_decode',
  code: scoresCode,
  bindings: ['read', 'read', 'read_write', 'uniform'],
  workgroupSize: [64, 1, 1],
  uniformBytes: 32,
};

/** weights, cacheV, out, dims. One thread per output element. */
export const ATTN_APPLY_DECODE: KernelSpec = {
  name: 'attn_apply_decode',
  code: applyCode,
  bindings: ['read', 'read', 'read_write', 'uniform'],
  workgroupSize: [64, 1, 1],
  uniformBytes: 32,
};

export function attnDecodeDims(
  position: number,
  heads: number,
  kvHeads: number,
  headDim: number,
  scoreStride: number,
  cacheStride: number,
): ArrayBuffer {
  return new Uint32Array([position, heads, kvHeads, headDim, scoreStride, cacheStride, 0, 0]).buffer;
}

export function attnScoresDecodeWorkgroups(maxContext: number, heads: number): [number, number, number] {
  const [x] = coverage(ATTN_SCORES_DECODE, [maxContext, 1, 1]);
  return [x, heads, 1];
}

export function attnApplyDecodeWorkgroups(heads: number, headDim: number): [number, number, number] {
  return coverage(ATTN_APPLY_DECODE, [heads * headDim, 1, 1]);
}
