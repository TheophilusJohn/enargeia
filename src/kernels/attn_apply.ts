import code from './attn_apply.wgsl?raw';
import { coverage, type KernelSpec } from './kernel.ts';

/** weights, v, out, dims. Dispatched over (head*headDim, query position). */
export const ATTN_APPLY: KernelSpec = {
  name: 'attn_apply',
  code,
  bindings: ['read', 'read', 'read_write', 'uniform'],
  workgroupSize: [64, 1, 1],
  uniformBytes: 16,
};

export function attnApplyWorkgroups(
  seq: number,
  heads: number,
  headDim: number,
): [number, number, number] {
  const [x] = coverage(ATTN_APPLY, [heads * headDim, 1, 1]);
  return [x, seq, 1];
}
