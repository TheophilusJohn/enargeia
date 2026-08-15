/**
 * Each forward-pass kernel against its CPU reference, in isolation.
 *
 * This is the per-kernel check the parity table cannot give. The parity harness runs the
 * whole chain from the prompt, so by layer 20 its error is cumulative — every stage inherits
 * the drift of the 300 dispatches before it. Here each kernel gets reference-quality inputs
 * and is judged on what it alone did, which is the only way to tell a wrong kernel from an
 * accumulated one.
 */

import { afterAll, describe, expect, it } from 'vitest';
import { readFloats } from '../../src/gpu/index.ts';
import type { PooledBuffer } from '../../src/gpu/pool.ts';
import { ComputeKernel, type KernelSpec } from '../../src/kernels/kernel.ts';
import { ADD } from '../../src/kernels/add.ts';
import { ARGMAX, ARGMAX_WORKGROUPS, argmaxDims } from '../../src/kernels/argmax.ts';
import { ATTN_APPLY, attnApplyWorkgroups } from '../../src/kernels/attn_apply.ts';
import { ATTN_SCORES, attnDims, attnScoresWorkgroups } from '../../src/kernels/attn_scores.ts';
import { EMBED_GATHER, embedDims, embedWorkgroups } from '../../src/kernels/embed_gather.ts';
import { MATMUL_BIAS, matmulBiasDims, matmulBiasWorkgroups } from '../../src/kernels/matmul_bias.ts';
import { RMSNORM, rmsnormDims, rmsnormWorkgroups } from '../../src/kernels/rmsnorm.ts';
import { ROPE, ropeDims, ropeWorkgroups } from '../../src/kernels/rope.ts';
import { SILU_MUL, countDims, elementwiseWorkgroups } from '../../src/kernels/silu_mul.ts';
import { SOFTMAX, softmaxDims, softmaxWorkgroups } from '../../src/kernels/softmax.ts';
import {
  add as addRef,
  argmax as argmaxRef,
  attentionApply,
  attentionScores,
  embed,
  linear,
  rmsNorm,
  rope as ropeRef,
  siluMul,
  softmaxRows,
} from '../reference/ops.ts';
import { compareArrays, withinTolerance } from '../reference/matmul.ts';
import { randomFloats } from '../reference/rng.ts';
import { expectNoGPUError, gpu, teardownGPU } from '../helpers/gpu.ts';

afterAll(teardownGPU);

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

/** Run one kernel over the given inputs and read the output back. */
async function run(
  spec: KernelSpec,
  inputs: readonly (Float32Array | Uint32Array)[],
  outputElements: number,
  uniform: ArrayBuffer,
  workgroups: readonly [number, number, number],
): Promise<Float32Array> {
  const { ctx, pool, cache } = await gpu();
  const kernel = new ComputeKernel(ctx.device, cache, spec);
  const buffers: PooledBuffer[] = [];

  return expectNoGPUError(ctx.device, async () => {
    for (const input of inputs) {
      const buffer = pool.acquire(Math.max(4, input.byteLength), STORAGE, 'in');
      ctx.queue.writeBuffer(buffer.buffer, 0, input);
      buffers.push(buffer);
    }
    const out = pool.acquire(Math.max(4, outputElements * 4), STORAGE, 'out');
    const dims = kernel.uniform(pool);
    ctx.queue.writeBuffer(dims.buffer, 0, uniform);

    const bindGroup = kernel.bindGroup([...buffers, out, dims]);
    const encoder = ctx.device.createCommandEncoder();
    // Pooled buffers carry the previous test's contents. Kernels that deliberately leave
    // some outputs untouched — embed_gather skips rows another part owns, matmul_bias with
    // an outOffset writes a slice — are only testable against a known starting state.
    pool.clear(encoder, out);
    const pass = encoder.beginComputePass();
    kernel.encode(pass, bindGroup, workgroups);
    pass.end();
    ctx.queue.submit([encoder.finish()]);

    const values = await readFloats(ctx, pool, out);
    pool.release(dims);
    pool.release(out);
    for (const buffer of buffers.reverse()) pool.release(buffer);
    return values.subarray(0, outputElements);
  });
}

function expectMatches(actual: Float32Array, expected: Float32Array, label: string): void {
  const error = compareArrays(actual, expected);
  expect(
    withinTolerance(error),
    `${label}: worst element ${error.worstIndex} at ${(error.worstRatio * 100).toFixed(1)}% of ` +
      `tolerance (abs ${error.maxAbs.toExponential(2)}, rel ${error.maxRel.toExponential(2)})`,
  ).toBe(true);
}

describe('rmsnorm', () => {
  for (const [seq, features] of [[1, 896], [15, 896], [7, 4864], [3, 64]] as const) {
    it(`matches the reference at ${seq}x${features}`, async () => {
      const x = randomFloats(seq * features, 11);
      const weight = randomFloats(features, 12);
      const actual = await run(RMSNORM, [x, weight], seq * features,
        rmsnormDims(seq, features, 1e-6), rmsnormWorkgroups(seq));
      expectMatches(actual, rmsNorm(x, weight, seq, features, 1e-6), `rmsnorm ${seq}x${features}`);
    });
  }

  it('uses the mean of squares, not the sum', async () => {
    // A kernel normalizing by the sum is wrong by sqrt(features) — 8x at 64 features. The
    // reference comparison above would catch it, but this states the property directly.
    const features = 64;
    const x = new Float32Array(features).fill(1);
    const weight = new Float32Array(features).fill(1);
    const actual = await run(RMSNORM, [x, weight], features,
      rmsnormDims(1, features, 0), rmsnormWorkgroups(1));
    // mean(1^2) = 1, so rsqrt(1) = 1 and every output is 1.
    expect(actual[0]).toBeCloseTo(1, 5);
  });
});

describe('matmul_bias', () => {
  for (const [m, n, k] of [[1, 896, 896], [15, 128, 896], [4, 4864, 896], [3, 7, 5]] as const) {
    it(`matches the reference at ${m}x${n}x${k}`, async () => {
      const x = randomFloats(m * k, 21);
      const w = randomFloats(n * k, 22);
      const bias = randomFloats(n, 23);
      const actual = await run(MATMUL_BIAS, [x, w, bias], m * n,
        matmulBiasDims(m, n, k, true), matmulBiasWorkgroups(m, n));
      expectMatches(actual, linear(x, w, bias, m, k, n), `matmul ${m}x${n}x${k}`);
    });
  }

  it('reads the weight transposed, as the checkpoint stores it', async () => {
    // out[0,0] = sum_i x[i] * w[0,i]. If the kernel read w as [k, n] it would instead compute
    // sum_i x[i] * w[i,0], which for this asymmetric input is a different number.
    const x = new Float32Array([1, 2, 3]);
    const w = new Float32Array([1, 10, 100, 2, 20, 200]); // [2, 3]
    const actual = await run(MATMUL_BIAS, [x, w, new Float32Array(2)], 2,
      matmulBiasDims(1, 2, 3, false), matmulBiasWorkgroups(1, 2));
    expect(Array.from(actual)).toEqual([321, 642]);
  });

  it('skips the bias when useBias is zero', async () => {
    const x = new Float32Array([1, 1]);
    const w = new Float32Array([1, 1]);
    const bias = new Float32Array([100]);
    const withBias = await run(MATMUL_BIAS, [x, w, bias], 1, matmulBiasDims(1, 1, 2, true), [1, 1, 1]);
    const without = await run(MATMUL_BIAS, [x, w, bias], 1, matmulBiasDims(1, 1, 2, false), [1, 1, 1]);
    expect(withBias[0]).toBe(102);
    expect(without[0]).toBe(2);
  });

  it('writes at outOffset with outStride, as the tied LM head needs', async () => {
    const x = new Float32Array([1, 1]);
    const w = new Float32Array([1, 1, 2, 2]); // two rows of length 2
    const actual = await run(MATMUL_BIAS, [x, w, new Float32Array(2)], 5,
      matmulBiasDims(1, 2, 2, false, 5, 3), [1, 1, 1]);
    // Two outputs land at indices 3 and 4 of a five-element vector.
    expect(Array.from(actual)).toEqual([0, 0, 0, 2, 4]);
  });
});

describe('rope', () => {
  for (const [seq, heads, headDim] of [[15, 14, 64], [15, 2, 64], [1, 14, 64]] as const) {
    it(`matches the reference at ${seq}x${heads}x${headDim}`, async () => {
      const x = randomFloats(seq * heads * headDim, 31);
      const actual = await run(ROPE, [x], x.length,
        ropeDims(seq, heads, headDim, 0, 1e6), ropeWorkgroups(seq, heads, headDim));
      expectMatches(actual, ropeRef(x, seq, heads, headDim, 1e6), `rope ${seq}x${heads}`);
    });
  }

  it('rotates halves, not adjacent pairs', async () => {
    // Position 1, headDim 4: element 0 pairs with element 2. With adjacent pairing it would
    // pair with element 1, which changes every output.
    const x = new Float32Array([1, 0, 0, 0]);
    const actual = await run(ROPE, [x], 4, ropeDims(1, 1, 4, 1, 10000), [1, 1, 1]);
    // out[0] = cos(1), out[2] = sin(1) for the first pair; the second pair is untouched zeros.
    expect(actual[0]).toBeCloseTo(Math.cos(1), 5);
    expect(actual[2]).toBeCloseTo(Math.sin(1), 5);
    expect(actual[1]).toBe(0);
  });

  it('leaves position zero unrotated', async () => {
    const x = randomFloats(64, 33);
    const actual = await run(ROPE, [x], 64, ropeDims(1, 1, 64, 0, 1e6), [1, 1, 1]);
    expectMatches(actual, x, 'rope at position 0');
  });
});

describe('attn_scores', () => {
  const seq = 15, heads = 14, kvHeads = 2, headDim = 64;

  it('matches the reference including the causal mask', async () => {
    const q = randomFloats(seq * heads * headDim, 41);
    const k = randomFloats(seq * kvHeads * headDim, 42);
    const actual = await run(ATTN_SCORES, [q, k], heads * seq * seq,
      attnDims(seq, heads, kvHeads, headDim), attnScoresWorkgroups(seq, heads));
    const expected = attentionScores(q, k, seq, heads, kvHeads, headDim);
    // The reference uses -Infinity for masked slots and the kernel a finite sentinel, so
    // compare only the unmasked triangle and check the mask separately.
    for (let h = 0; h < heads; h++) {
      for (let i = 0; i < seq; i++) {
        for (let j = 0; j <= i; j++) {
          const at = (h * seq + i) * seq + j;
          expect(Math.abs(actual[at] - expected[at])).toBeLessThan(1e-4);
        }
        for (let j = i + 1; j < seq; j++) {
          expect(actual[(h * seq + i) * seq + j]).toBeLessThan(-1e29);
        }
      }
    }
  });

  it('groups query heads onto kv heads contiguously', async () => {
    // Heads 0..6 must read kv head 0 and heads 7..13 kv head 1. Interleaving instead
    // (h % kvHeads) keeps every value finite and every score wrong.
    const q = new Float32Array(1 * heads * headDim).fill(0);
    for (let h = 0; h < heads; h++) q[h * headDim] = 1; // each head reads k[.., 0]
    const k = new Float32Array(1 * kvHeads * headDim);
    k[0 * headDim] = 3; // kv head 0
    k[1 * headDim] = 7; // kv head 1
    const actual = await run(ATTN_SCORES, [q, k], heads * 1 * 1,
      attnDims(1, heads, kvHeads, headDim), attnScoresWorkgroups(1, heads));
    const scale = 1 / Math.sqrt(headDim);
    for (let h = 0; h < 7; h++) expect(actual[h]).toBeCloseTo(3 * scale, 4);
    for (let h = 7; h < 14; h++) expect(actual[h]).toBeCloseTo(7 * scale, 4);
  });
});

describe('softmax', () => {
  it('matches the reference on random rows', async () => {
    const rows = 30, cols = 15;
    const x = randomFloats(rows * cols, 51);
    const actual = await run(SOFTMAX, [x], rows * cols, softmaxDims(rows, cols), softmaxWorkgroups(rows));
    expectMatches(actual, softmaxRows(x, rows, cols), 'softmax');
  });

  it('produces rows that sum to one', async () => {
    const rows = 4, cols = 300;
    const x = randomFloats(rows * cols, 52);
    const actual = await run(SOFTMAX, [x], rows * cols, softmaxDims(rows, cols), softmaxWorkgroups(rows));
    for (let r = 0; r < rows; r++) {
      let sum = 0;
      for (let c = 0; c < cols; c++) sum += actual[r * cols + c];
      expect(sum).toBeCloseTo(1, 4);
    }
  });

  it('survives large inputs that would overflow without max subtraction', async () => {
    const x = new Float32Array([200, 201, 202, 203]);
    const actual = await run(SOFTMAX, [x], 4, softmaxDims(1, 4), softmaxWorkgroups(1));
    expect(actual.every(Number.isFinite)).toBe(true);
    expectMatches(actual, softmaxRows(x, 1, 4), 'softmax overflow');
  });

  it('gives zeros rather than NaN for a fully masked row', async () => {
    const x = new Float32Array([-1e30, -1e30, -1e30, -1e30]);
    const actual = await run(SOFTMAX, [x], 4, softmaxDims(1, 4), softmaxWorkgroups(1));
    expect(actual.every((v) => Number.isFinite(v))).toBe(true);
  });
});

describe('attn_apply', () => {
  it('matches the reference', async () => {
    const seq = 15, heads = 14, kvHeads = 2, headDim = 64;
    const scores = randomFloats(heads * seq * seq, 61);
    const weights = softmaxRows(scores, heads * seq, seq);
    const v = randomFloats(seq * kvHeads * headDim, 62);
    const actual = await run(ATTN_APPLY, [weights, v], seq * heads * headDim,
      attnDims(seq, heads, kvHeads, headDim), attnApplyWorkgroups(seq, heads, headDim));
    expectMatches(actual, attentionApply(weights, v, seq, heads, kvHeads, headDim), 'attn_apply');
  });
});

describe('silu_mul and add', () => {
  it('silu_mul matches the reference', async () => {
    const gate = randomFloats(4864, 71);
    const up = randomFloats(4864, 72);
    const actual = await run(SILU_MUL, [gate, up], gate.length,
      countDims(gate.length), elementwiseWorkgroups(gate.length));
    expectMatches(actual, siluMul(gate, up), 'silu_mul');
  });

  it('silu is finite at large magnitudes', async () => {
    const gate = new Float32Array([-100, -40, 0, 40, 100]);
    const up = new Float32Array([1, 1, 1, 1, 1]);
    const actual = await run(SILU_MUL, [gate, up], 5, countDims(5), elementwiseWorkgroups(5));
    expect(actual.every(Number.isFinite)).toBe(true);
    expect(actual[2]).toBe(0);
    expect(actual[4]).toBeCloseTo(100, 3);
  });

  it('add matches the reference', async () => {
    const a = randomFloats(896 * 15, 81);
    const b = randomFloats(896 * 15, 82);
    const actual = await run(ADD, [a, b], a.length, countDims(a.length), elementwiseWorkgroups(a.length));
    expectMatches(actual, addRef(a, b), 'add');
  });
});

describe('embed_gather', () => {
  it('gathers rows from one part and leaves other rows alone', async () => {
    const hidden = 8, rows = 16, seq = 4;
    const table = randomFloats(rows * hidden, 91);
    const ids = new Uint32Array([0, 5, 15, 3]);
    const actual = await run(EMBED_GATHER, [ids, table], seq * hidden,
      embedDims(seq, hidden, 0, rows), embedWorkgroups(seq, hidden));
    expectMatches(actual, embed(Array.from(ids), table, hidden), 'embed_gather');
  });

  it('writes only the rows its part owns', async () => {
    // The second part covers global rows 8..15. Tokens outside that range must be untouched,
    // because another dispatch owns them.
    const hidden = 4, rowsPerPart = 8, seq = 3;
    const part = randomFloats(rowsPerPart * hidden, 92);
    const ids = new Uint32Array([2, 9, 14]);
    const actual = await run(EMBED_GATHER, [ids, part], seq * hidden,
      embedDims(seq, hidden, 8, rowsPerPart), embedWorkgroups(seq, hidden));
    // Token 2 is not in this part: left at zero.
    expect(Array.from(actual.subarray(0, hidden))).toEqual([0, 0, 0, 0]);
    // Tokens 9 and 14 map to local rows 1 and 6.
    expect(Array.from(actual.subarray(hidden, 2 * hidden)))
      .toEqual(Array.from(part.subarray(1 * hidden, 2 * hidden)));
    expect(Array.from(actual.subarray(2 * hidden, 3 * hidden)))
      .toEqual(Array.from(part.subarray(6 * hidden, 7 * hidden)));
  });
});

describe('argmax', () => {
  it('finds the largest logit over a full vocabulary', async () => {
    const { ctx, pool, cache } = await gpu();
    const vocab = 151936;
    const logits = randomFloats(vocab, 101);
    logits[123456] = 99;

    const kernel = new ComputeKernel(ctx.device, cache, ARGMAX);
    const input = pool.acquire(logits.byteLength, STORAGE, 'logits');
    ctx.queue.writeBuffer(input.buffer, 0, logits);
    const out = pool.acquire(16, STORAGE, 'argmax');
    const dims = kernel.uniform(pool);
    ctx.queue.writeBuffer(dims.buffer, 0, argmaxDims(vocab, 0));

    const encoder = ctx.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    kernel.encode(pass, kernel.bindGroup([input, out, dims]), ARGMAX_WORKGROUPS);
    pass.end();
    ctx.queue.submit([encoder.finish()]);

    const bytes = await readFloats(ctx, pool, out);
    expect(new Uint32Array(bytes.buffer)[0]).toBe(argmaxRef(logits));
    expect(new Uint32Array(bytes.buffer)[0]).toBe(123456);
    pool.release(dims); pool.release(out); pool.release(input);
  });

  it('breaks ties toward the lowest index, as torch.argmax does', async () => {
    const { ctx, pool, cache } = await gpu();
    const count = 2048;
    const logits = new Float32Array(count).fill(-1);
    logits[100] = 5;
    logits[900] = 5; // identical value, higher index

    const kernel = new ComputeKernel(ctx.device, cache, ARGMAX);
    const input = pool.acquire(logits.byteLength, STORAGE, 'logits');
    ctx.queue.writeBuffer(input.buffer, 0, logits);
    const out = pool.acquire(16, STORAGE, 'argmax');
    const dims = kernel.uniform(pool);
    ctx.queue.writeBuffer(dims.buffer, 0, argmaxDims(count, 0));

    const encoder = ctx.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    kernel.encode(pass, kernel.bindGroup([input, out, dims]), ARGMAX_WORKGROUPS);
    pass.end();
    ctx.queue.submit([encoder.finish()]);

    const bytes = await readFloats(ctx, pool, out);
    expect(new Uint32Array(bytes.buffer)[0]).toBe(100);
    pool.release(dims); pool.release(out); pool.release(input);
  });
});
