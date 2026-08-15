/**
 * The int4 kernels against the CPU reference.
 *
 * The thing most worth testing here is agreement on *packing order* — which nibble of which
 * u32 holds which weight, and which block a weight belongs to. Getting it backwards produces
 * finite, plausibly-scaled garbage that no amount of staring at output text would reveal.
 * `tools/quantize.py` and `test/reference/quant.ts` implement the same scheme independently,
 * and a test that quantizes here and dequantizes on the GPU is what pins them together.
 */

import { afterAll, describe, expect, it } from 'vitest';
import { readFloats } from '../../src/gpu/index.ts';
import type { PooledBuffer } from '../../src/gpu/pool.ts';
import { ComputeKernel, type KernelSpec } from '../../src/kernels/kernel.ts';
import {
  MATMUL_Q4_DECODE,
  MATMUL_Q4_PREFILL,
  matmulQ4DecodeWorkgroups,
  matmulQ4Dims,
  matmulQ4PrefillWorkgroups,
} from '../../src/kernels/matmul_q4.ts';
import { EMBED_GATHER_Q4, embedQ4Dims, embedQ4Workgroups } from '../../src/kernels/embed_gather_q4.ts';
import { dequantizeMatrix, linearQ4, quantizeMatrix, type QuantizedMatrix } from '../reference/quant.ts';
import { linear } from '../reference/ops.ts';
import { compareArrays, withinTolerance } from '../reference/matmul.ts';
import { randomFloats } from '../reference/rng.ts';
import { expectNoGPUError, gpu, teardownGPU } from '../helpers/gpu.ts';

afterAll(teardownGPU);

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

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

    const encoder = ctx.device.createCommandEncoder();
    pool.clear(encoder, out);
    const pass = encoder.beginComputePass();
    kernel.encode(pass, kernel.bindGroup([...buffers, out, dims]), workgroups);
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

/** Run the prefill kernel against a quantized weight and its own dequantization. */
async function q4Prefill(
  x: Float32Array,
  w: QuantizedMatrix,
  bias: Float32Array | null,
  m: number,
  k: number,
  n: number,
): Promise<Float32Array> {
  return run(
    MATMUL_Q4_PREFILL,
    [x, w.packed, w.scales, w.zeros, bias ?? new Float32Array(Math.max(1, n))],
    m * n,
    matmulQ4Dims(m, n, k, bias !== null, w.blockSize),
    matmulQ4PrefillWorkgroups(m, n),
  );
}

describe('quantization round trip', () => {
  it('recovers a constant block exactly', () => {
    const values = new Float32Array(64).fill(0.375);
    const q = quantizeMatrix(values, 1, 64);
    const back = dequantizeMatrix(q);
    for (const v of back) expect(v).toBeCloseTo(0.375, 6);
  });

  it('recovers the block endpoints exactly', () => {
    // The asymmetric scheme maps the block minimum to nibble 0 and the maximum to nibble 15,
    // so both endpoints come back without error regardless of what is in between.
    const values = randomFloats(64, 5);
    values[3] = -1.25;
    values[40] = 2.5;
    const back = dequantizeMatrix(quantizeMatrix(values, 1, 64));
    expect(back[3]).toBeCloseTo(-1.25, 5);
    expect(back[40]).toBeCloseTo(2.5, 5);
  });

  it('keeps error inside one quantization step', () => {
    const values = randomFloats(64 * 20, 6);
    const q = quantizeMatrix(values, 20, 64);
    const back = dequantizeMatrix(q);
    for (let b = 0; b < 20; b++) {
      const step = q.scales[b];
      for (let i = 0; i < 64; i++) {
        const at = b * 64 + i;
        expect(Math.abs(back[at] - values[at])).toBeLessThanOrEqual(step / 2 + 1e-6);
      }
    }
  });

  it('is why blocks beat a single per-tensor scale', () => {
    // One outlier sets a per-tensor scale and crushes everything else. This is the failure
    // the skill says to refuse a "simplification" toward.
    const values = randomFloats(128, 7);
    for (let i = 0; i < 128; i++) values[i] *= 0.01;
    values[0] = 100; // outlier in block 0 only

    const blocked = dequantizeMatrix(quantizeMatrix(values, 2, 64, 64));
    // One scale for the whole 128 values, which is what "per-tensor" means here.
    const perTensor = dequantizeMatrix(quantizeMatrix(values, 1, 128, 128));

    const rms = (a: Float32Array) => {
      let sum = 0;
      for (let i = 64; i < 128; i++) sum += (a[i] - values[i]) ** 2;
      return Math.sqrt(sum / 64);
    };
    // The clean block is unaffected by the outlier when blocks are separate.
    expect(rms(blocked)).toBeLessThan(rms(perTensor) / 10);
  });
});

describe('matmul_q4_prefill', () => {
  for (const [m, n, k] of [[15, 896, 896], [15, 128, 896], [4, 4864, 896], [64, 32, 64]] as const) {
    it(`matches the dequantized reference at ${m}x${n}x${k}`, async () => {
      const x = randomFloats(m * k, 11);
      const weights = randomFloats(n * k, 12);
      const q = quantizeMatrix(weights, n, k);
      const bias = randomFloats(n, 13);

      const actual = await q4Prefill(x, q, bias, m, k, n);
      expectMatches(actual, linearQ4(x, q, bias, m, k, n), `q4 prefill ${m}x${n}x${k}`);
    });
  }

  it('agrees with the fp32 kernel on the dequantized weights', async () => {
    // The quantized kernel must equal an fp32 matmul over exactly the weights the
    // quantization produced. Any difference is a packing or indexing bug, not a
    // quantization error, so this is the check with no tolerance slack in it.
    const m = 8, n = 64, k = 128;
    const x = randomFloats(m * k, 21);
    const q = quantizeMatrix(randomFloats(n * k, 22), n, k);
    const actual = await q4Prefill(x, q, null, m, k, n);
    expectMatches(actual, linear(x, dequantizeMatrix(q), null, m, k, n), 'q4 vs dequantized fp32');
  });

  it('handles shapes that do not fill the 64x32 footprint', async () => {
    const m = 3, n = 5, k = 64;
    const x = randomFloats(m * k, 31);
    const q = quantizeMatrix(randomFloats(n * k, 32), n, k);
    const actual = await q4Prefill(x, q, null, m, k, n);
    expectMatches(actual, linearQ4(x, q, null, m, k, n), 'ragged q4');
  });

  it('writes at outOffset with outStride, as the tied LM head needs', async () => {
    const k = 64;
    const x = new Float32Array(k).fill(1);
    const q = quantizeMatrix(new Float32Array(2 * k).fill(0.5), 2, k);
    const actual = await run(
      MATMUL_Q4_PREFILL,
      [x, q.packed, q.scales, q.zeros, new Float32Array(2)],
      6,
      matmulQ4Dims(1, 2, k, false, q.blockSize, 6, 4),
      matmulQ4PrefillWorkgroups(1, 2),
    );
    expect(actual[0]).toBe(0);
    expect(actual[4]).toBeCloseTo(32, 3);
    expect(actual[5]).toBeCloseTo(32, 3);
  });
});

describe('matmul_q4_decode', () => {
  for (const n of [896, 128, 4864] as const) {
    it(`matches the reference for a single row at n=${n}`, async () => {
      const k = 896;
      const x = randomFloats(k, 41);
      const q = quantizeMatrix(randomFloats(n * k, 42), n, k);
      const bias = randomFloats(n, 43);
      const actual = await run(
        MATMUL_Q4_DECODE,
        [x, q.packed, q.scales, q.zeros, bias],
        n,
        matmulQ4Dims(1, n, k, true, q.blockSize),
        matmulQ4DecodeWorkgroups(n),
      );
      expectMatches(actual, linearQ4(x, q, bias, 1, k, n), `q4 decode n=${n}`);
    });
  }

  it('agrees with the prefill kernel at m=1', async () => {
    // The two kernels differ in tiling and in where the unpacking happens, so they are
    // independent implementations of the same arithmetic. Agreement between them catches a
    // bug that a shared helper would hide from both.
    const n = 256, k = 192;
    const x = randomFloats(k, 51);
    const q = quantizeMatrix(randomFloats(n * k, 52), n, k);

    const decode = await run(
      MATMUL_Q4_DECODE, [x, q.packed, q.scales, q.zeros, new Float32Array(n)], n,
      matmulQ4Dims(1, n, k, false, q.blockSize), matmulQ4DecodeWorkgroups(n),
    );
    const prefill = await q4Prefill(x, q, null, 1, k, n);
    expectMatches(decode, prefill, 'decode vs prefill');
  });
});

describe('embed_gather_q4', () => {
  it('gathers and dequantizes rows from one part', async () => {
    const hidden = 128, rows = 16, seq = 4;
    const table = randomFloats(rows * hidden, 61);
    const q = quantizeMatrix(table, rows, hidden);
    const ids = new Uint32Array([0, 5, 15, 3]);

    const actual = await run(
      EMBED_GATHER_Q4,
      [ids, q.packed, q.scales, q.zeros],
      seq * hidden,
      embedQ4Dims(seq, hidden, 0, rows, q.blockSize),
      embedQ4Workgroups(seq, hidden),
    );

    const dequantized = dequantizeMatrix(q);
    const expected = new Float32Array(seq * hidden);
    for (let s = 0; s < seq; s++) {
      expected.set(dequantized.subarray(ids[s] * hidden, (ids[s] + 1) * hidden), s * hidden);
    }
    expectMatches(actual, expected, 'q4 gather');
  });

  it('writes only the rows its part owns', async () => {
    const hidden = 64, rowsPerPart = 8, seq = 3;
    const part = quantizeMatrix(randomFloats(rowsPerPart * hidden, 71), rowsPerPart, hidden);
    const ids = new Uint32Array([2, 9, 14]);
    const actual = await run(
      EMBED_GATHER_Q4,
      [ids, part.packed, part.scales, part.zeros],
      seq * hidden,
      embedQ4Dims(seq, hidden, 8, rowsPerPart, part.blockSize),
      embedQ4Workgroups(seq, hidden),
    );
    const dequantized = dequantizeMatrix(part);
    // Token 2 belongs to another part and is left untouched.
    expect(Array.from(actual.subarray(0, hidden))).toEqual(new Array(hidden).fill(0));
    expect(Array.from(actual.subarray(hidden, 2 * hidden)))
      .toEqual(Array.from(dequantized.subarray(1 * hidden, 2 * hidden)));
  });
});
