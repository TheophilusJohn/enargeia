/**
 * Every shader in the forward pass must actually compile.
 *
 * `createComputePipeline` is synchronous and reports a bad shader through the uncaptured
 * error handler rather than by throwing, so a pipeline built from a broken shader is a valid
 * object that dispatches nothing. That failure mode is silent and looks exactly like a
 * numerically wrong kernel, which is why it gets its own test ahead of any parity run.
 */
import { afterAll, describe, expect, it } from 'vitest';
import { PipelineCache } from '../../src/gpu/index.ts';
import { gpu, teardownGPU } from '../helpers/gpu.ts';
import { ADD } from '../../src/kernels/add.ts';
import { ARGMAX } from '../../src/kernels/argmax.ts';
import { ATTN_APPLY } from '../../src/kernels/attn_apply.ts';
import { ATTN_SCORES } from '../../src/kernels/attn_scores.ts';
import { EMBED_GATHER } from '../../src/kernels/embed_gather.ts';
import { MATMUL_BIAS } from '../../src/kernels/matmul_bias.ts';
import { RMSNORM } from '../../src/kernels/rmsnorm.ts';
import { ROPE } from '../../src/kernels/rope.ts';
import { SILU_MUL } from '../../src/kernels/silu_mul.ts';
import { SOFTMAX } from '../../src/kernels/softmax.ts';
import { MATMUL_Q4_DECODE, MATMUL_Q4_PREFILL } from '../../src/kernels/matmul_q4.ts';
import { EMBED_GATHER_Q4 } from '../../src/kernels/embed_gather_q4.ts';

afterAll(teardownGPU);

const SPECS = [ADD, ARGMAX, ATTN_APPLY, ATTN_SCORES, EMBED_GATHER, EMBED_GATHER_Q4, MATMUL_BIAS,
  MATMUL_Q4_DECODE, MATMUL_Q4_PREFILL, RMSNORM, ROPE, SILU_MUL, SOFTMAX];

describe('shader compilation', () => {
  for (const spec of SPECS) {
    it(`${spec.name} compiles without errors`, async () => {
      const { ctx } = await gpu();
      const cache = new PipelineCache(ctx.device);
      const module = cache.module(spec.code, spec.name);
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === 'error');
      const text = errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n');
      expect(text, `${spec.name} failed to compile`).toBe('');
    });
  }
});
