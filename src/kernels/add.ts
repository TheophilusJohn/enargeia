import code from './add.wgsl?raw';
import type { KernelSpec } from './kernel.ts';

/** a, b, out, dims. Residual add. */
export const ADD: KernelSpec = {
  name: 'add',
  code,
  bindings: ['read', 'read', 'read_write', 'uniform'],
  workgroupSize: [256, 1, 1],
  uniformBytes: 16,
};
