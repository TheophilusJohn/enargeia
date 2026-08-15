/**
 * Compile every pipeline the session will need, up front.
 *
 * WebGPU compiles a shader on its first dispatch unless asked otherwise, so the cost lands on
 * the first generation — the single worst moment, because it is also the moment the user is
 * watching a blank response. There is a 282 MB download happening anyway, and compilation is
 * CPU work on a different resource, so it is free if it happens there.
 *
 * `createComputePipelineAsync` is what makes this possible: the synchronous form returns
 * immediately and defers the real work, so calling it early moves nothing.
 */

import type { PipelineCache } from '../gpu/pipeline.ts';
import type { KernelSpec } from '../kernels/kernel.ts';
import { ADD } from '../kernels/add.ts';
import { ARGMAX } from '../kernels/argmax.ts';
import { ATTN_SCORES } from '../kernels/attn_scores.ts';
import { ATTN_APPLY } from '../kernels/attn_apply.ts';
import { ATTN_APPLY_DECODE, ATTN_SCORES_DECODE } from '../kernels/attn_decode.ts';
import { EMBED_GATHER } from '../kernels/embed_gather.ts';
import { MATMUL_BIAS } from '../kernels/matmul_bias.ts';
import { MATMUL_Q4_DECODE, MATMUL_Q4_PREFILL } from '../kernels/matmul_q4.ts';
import { RMSNORM } from '../kernels/rmsnorm.ts';
import { ROPE } from '../kernels/rope.ts';
import { SILU_MUL } from '../kernels/silu_mul.ts';
import { SOFTMAX } from '../kernels/softmax.ts';
import { SAMPLE } from '../kernels/sample.ts';
import {
  embeddingGatherSpec,
  embeddingHeadSpec,
  type EmbeddingDType,
} from '../kernels/embedding_variants.ts';
import type { ModelConfig } from '../model/graph.ts';
import type { WeightStore } from '../model/weights.ts';

/** Every kernel a session built from these weights can reach. */
export function allKernelSpecs(weights: WeightStore, _config: ModelConfig): KernelSpec[] {
  const specs: KernelSpec[] = [
    ADD, ARGMAX, SAMPLE, RMSNORM, ROPE, SILU_MUL, SOFTMAX,
    ATTN_SCORES, ATTN_APPLY, ATTN_SCORES_DECODE, ATTN_APPLY_DECODE,
    MATMUL_BIAS, EMBED_GATHER,
  ];
  if (weights.isQuantized) {
    specs.push(MATMUL_Q4_PREFILL, MATMUL_Q4_DECODE);
  }
  const bits = weights.embedding.quantParts?.[0].bits;
  const dtype: EmbeddingDType | null =
    bits === 4 ? 'q4' : bits === 8 ? 'q8' : bits === 16 ? 'f16' : bits === 32 ? 'f32' : null;
  if (dtype) {
    specs.push(embeddingGatherSpec(dtype), embeddingHeadSpec(dtype));
  }
  return specs;
}

export interface PrecompileProgress {
  compiled: number;
  total: number;
}

/**
 * Compile all of them concurrently.
 *
 * Concurrent rather than sequential because each `createComputePipelineAsync` is mostly a wait
 * on a background compiler thread; serializing them would take as long as the sum instead of
 * as long as the longest.
 */
export async function precompile(
  _device: GPUDevice,
  cache: PipelineCache,
  specs: readonly KernelSpec[],
  onProgress?: (progress: PrecompileProgress) => void,
): Promise<void> {
  let compiled = 0;
  await Promise.all(
    specs.map(async (spec) => {
      await cache.pipelineAsync({
        code: spec.code,
        entryPoint: 'main',
        label: spec.name,
        layout: cache.pipelineLayout(`kernel:${spec.name}`, {
          bindGroupLayouts: [
            cache.bindGroupLayout(`kernel:${spec.name}`, {
              entries: spec.bindings.map((kind, binding) => ({
                binding,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                  type:
                    kind === 'uniform'
                      ? ('uniform' as const)
                      : kind === 'read'
                        ? ('read-only-storage' as const)
                        : ('storage' as const),
                  ...(kind === 'uniform' ? { minBindingSize: spec.uniformBytes } : {}),
                },
              })),
            }),
          ],
        }),
      });
      compiled++;
      onProgress?.({ compiled, total: specs.length });
    }),
  );
}
