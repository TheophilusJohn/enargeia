/**
 * Optimization toggles, so each M6 change can be measured on its own.
 *
 * A flag per change rather than a git revert per row: an ablation table is only honest if the
 * rows differ in exactly one thing, and reading the difference between two runs that changed
 * several things is how a table ends up attributing a win to the wrong change.
 *
 * These are not user-facing settings. They exist to be turned off in the harness and should
 * all be on in anything that ships.
 */
export interface Optimizations {
  /** Compile every pipeline during the weight download instead of on first dispatch. */
  precompile: boolean;
  /** Fold RMSNorm's scale into the following matmul so the normalized row stays off VRAM. */
  fusedNorm: boolean;
  /** Use f16 accumulation where `shader-f16` exists. */
  f16: boolean;
  /** Pick the prefill kernel from the measured crossover rather than always the coarsest. */
  kernelSelect: boolean;
  /** One uniform buffer per graph, written once per token, instead of one per dispatch. */
  batchedUniforms: boolean;
  /** Workgroup reduction over the KV history instead of a serial per-thread walk. */
  parallelAttention: boolean;
  /** Prompt-lookup speculative decoding. */
  speculation: boolean;
}

export const OPTIMIZATIONS: Optimizations = {
  precompile: true,
  fusedNorm: true,
  f16: true,
  kernelSelect: true,
  batchedUniforms: true,
  parallelAttention: true,
  speculation: true,
};

export function setOptimizations(patch: Partial<Optimizations>): void {
  Object.assign(OPTIMIZATIONS, patch);
}

export function describeOptimizations(): string {
  const off = Object.entries(OPTIMIZATIONS).filter(([, v]) => !v).map(([k]) => k);
  return off.length === 0 ? 'all optimizations on' : `off: ${off.join(', ')}`;
}
