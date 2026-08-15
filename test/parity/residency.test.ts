/**
 * Where the memory goes, and whether prefill has to allocate for the longest prompt it might
 * ever see.
 *
 * The site reports a live residency of 1132.6 MiB against a headline of 334.9 MiB of weights,
 * and the gap is prefill scratch sized to `maxContext`. That is the same shape as the
 * over-dispatch bug corrected in M6 — sizing to a build-time maximum rather than the actual
 * extent — so it gets the same treatment: measure the mechanism before building anything.
 *
 * Two questions, both answered here and neither assumed:
 *
 *   1. Is peak residency actually dominated by the two `heads × maxSeq × maxSeq` attention
 *      buffers? Listed per buffer rather than inferred from arithmetic.
 *   2. Would chunking prefill cost throughput? Prefill time is fitted against prompt length as
 *      `a·N + b·N²`. The quadratic coefficient is the attention term; chunking with a KV cache
 *      replaces the full N² square with a triangle, so the fit predicts the chunked cost
 *      without building it.
 *
 *   npm run residency
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { BufferPool, PipelineCache, initGPU, type GPUContext } from '../../src/gpu/index.ts';
import { HttpRangeSource } from '../../src/model/safetensors.ts';
import { WeightStore } from '../../src/model/weights.ts';
import type { ModelConfig } from '../../src/model/graph.ts';
import { Session } from '../../src/runtime/session.ts';
import { GREEDY } from '../../src/kernels/sample.ts';
import { Tokenizer, type TokenizerJSON } from '../../src/tokenizer/tokenizer.ts';
import tokenizerJson from '../fixtures/tokenizer.json';

const MODEL_URL = '/models/qwen2.5-0.5b.enargeia';
const MAX_CONTEXT = 2048;

let ctx: GPUContext;
let pool: BufferPool;
let pipelines: PipelineCache;
let weights: WeightStore;
let config: ModelConfig;
let corpus: number[];

beforeAll(async () => {
  ctx = await initGPU({ label: 'residency' });
  pool = new BufferPool(ctx.device, { label: 'residency', maxIdleBytes: 256 * 1024 * 1024 });
  pipelines = new PipelineCache(ctx.device);

  const sidecar = await (await fetch('/test/fixtures/reference.json')).json();
  config = sidecar.config as ModelConfig;

  const text = await (await fetch('/test/fixtures/heldout_large.txt')).text();
  const tokenizer = Tokenizer.fromJSON(tokenizerJson as unknown as TokenizerJSON);
  const once = tokenizer.encode(text);
  corpus = [];
  while (corpus.length < MAX_CONTEXT + 64) corpus.push(...once);

  weights = await WeightStore.loadQuantized(ctx.device, ctx.profile, {
    ref: { modelId: 'Qwen/Qwen2.5-0.5B-Instruct', revision: 'residency', file: 'model' },
    source: new HttpRangeSource(MODEL_URL),
    noCache: true,
  });
}, 900_000);

afterAll(() => {
  weights?.destroy();
  pool?.destroy();
  ctx?.device.destroy();
});

const mib = (bytes: number) => (bytes / 1048576).toFixed(1).padStart(7);

function median(xs: number[]): number {
  const sorted = [...xs].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

describe('question 1 — where the scratch actually is', () => {
  it('lists every live allocation, largest first', () => {
    const session = new Session(ctx.device, ctx.queue, pool, pipelines, weights, config, {
      maxContext: MAX_CONTEXT,
      sampling: GREEDY,
    });
    try {
      const rows = pool.describeLive();
      let total = 0;
      let waste = 0;
      console.log('[residency] live scratch, largest first:');
      for (const row of rows.slice(0, 14)) {
        total += row.capacity;
        waste += row.capacity - row.requested;
        console.log(
          `[residency]   ${row.label.padEnd(18)} requested ${mib(row.requested)} MiB  ` +
            `capacity ${mib(row.capacity)} MiB`,
        );
      }
      const all = rows.reduce((n, r) => n + r.capacity, 0);
      const allWaste = rows.reduce((n, r) => n + (r.capacity - r.requested), 0);
      console.log(
        `[residency] ${rows.length} buffers, ${mib(all)} MiB of capacity, ` +
          `${mib(allWaste)} MiB of it size-class rounding`,
      );
      console.log(`[residency] top 14 are ${mib(total)} MiB (${((total / all) * 100).toFixed(1)}%)`);
      console.log(
        `[residency] weights ${mib(weights.stats.gpuBytes)} MiB · ` +
          `KV cache ${mib(session.cache.totalBytes)} MiB · scratch ${mib(all)} MiB · ` +
          `total ${mib(weights.stats.gpuBytes + session.cache.totalBytes + all)} MiB`,
      );
      expect(rows.length).toBeGreaterThan(0);
    } finally {
      session.destroy();
    }
  });
});

describe('question 3 — the measured chunk-size tradeoff', () => {
  // Question 2 predicted chunking would be *faster*. It is not. This measures the curve
  // directly instead of extrapolating from a fit, because the prediction was wrong and the
  // fit is what was wrong about it. A chunk equal to the context is the control: it runs the
  // prompt in exactly one pass, which is what the graph did before chunking existed.
  it('prefill at 2048 and resident scratch, against chunk size', async () => {
    const rows: Array<[number, number, number]> = [];
    for (const chunk of [128, 256, 512, 1024, 2048]) {
      const session = new Session(ctx.device, ctx.queue, pool, pipelines, weights, config, {
        maxContext: MAX_CONTEXT,
        sampling: GREEDY,
        prefillChunk: chunk,
      });
      try {
        const scratch = pool.describeLive().reduce((n, r) => n + r.capacity, 0);
        await session.prime(corpus.slice(0, 8));
        session.reset();
        const runs: number[] = [];
        for (let i = 0; i < 3; i++) {
          session.reset();
          const started = performance.now();
          await session.prime(corpus.slice(0, 2048));
          runs.push(performance.now() - started);
        }
        const ms = median(runs);
        rows.push([chunk, ms, scratch]);
        console.log(
          `[residency] chunk ${String(chunk).padStart(4)}: prefill@2048 ${ms.toFixed(0).padStart(5)} ms ` +
            `(${(2048 / (ms / 1000)).toFixed(0).padStart(4)} tok/s) · scratch ${mib(scratch)} MiB`,
        );
      } finally {
        session.destroy();
      }
    }
    const control = rows[rows.length - 1];
    for (const [chunk, ms, scratch] of rows) {
      console.log(
        `[residency] chunk ${String(chunk).padStart(4)}: ` +
          `${((ms / control[1] - 1) * 100).toFixed(1).padStart(6)}% time · ` +
          `${(scratch / control[2]).toFixed(3)}x scratch vs one-shot`,
      );
    }
    expect(rows.length).toBe(5);
  }, 900_000);
});

describe('question 2 — is prefill quadratic enough for chunking to pay', () => {
  it('fits prefill time against prompt length', async () => {
    const session = new Session(ctx.device, ctx.queue, pool, pipelines, weights, config, {
      maxContext: MAX_CONTEXT,
      sampling: GREEDY,
    });
    try {
      await session.prime(corpus.slice(0, 8));
      session.reset();

      const lengths = [64, 128, 256, 512, 1024, 1536, 2048];
      const points: Array<[number, number]> = [];
      for (const n of lengths) {
        const runs: number[] = [];
        for (let i = 0; i < 3; i++) {
          session.reset();
          const started = performance.now();
          await session.prime(corpus.slice(0, n));
          runs.push(performance.now() - started);
        }
        const ms = median(runs);
        points.push([n, ms]);
        console.log(
          `[residency] prefill ${String(n).padStart(4)}: ${ms.toFixed(1).padStart(7)} ms  ` +
            `${(n / (ms / 1000)).toFixed(0).padStart(5)} tok/s`,
        );
      }

      // Least squares on ms/N = a + b·N, which is the same fit as ms = a·N + b·N² but weights
      // the short prompts properly instead of letting 2048 dominate.
      const xs = points.map(([n]) => n);
      const ys = points.map(([n, ms]) => ms / n);
      const meanX = xs.reduce((p, q) => p + q, 0) / xs.length;
      const meanY = ys.reduce((p, q) => p + q, 0) / ys.length;
      let num = 0;
      let den = 0;
      for (let i = 0; i < xs.length; i++) {
        num += (xs[i] - meanX) * (ys[i] - meanY);
        den += (xs[i] - meanX) ** 2;
      }
      const b = num / den;
      const a = meanY - b * meanX;
      console.log(`[residency] fit: ms = ${a.toFixed(4)}·N + ${b.toExponential(3)}·N²`);

      const N = 2048;
      const linear = a * N;
      const quadratic = b * N * N;
      console.log(
        `[residency] at N=2048: linear ${linear.toFixed(0)} ms, quadratic ${quadratic.toFixed(0)} ms ` +
          `(${((quadratic / (linear + quadratic)) * 100).toFixed(1)}% of the total)`,
      );

      // A chunked prefill of n chunks of C computes, per chunk, C queries against the prefix so
      // far — a triangle instead of a square. Total quadratic work becomes
      // b·C²·(1+2+...+n) = b·N²·(n+1)/(2n), against b·N² for one shot.
      for (const C of [128, 256, 512]) {
        const chunks = N / C;
        const predicted = linear + quadratic * ((chunks + 1) / (2 * chunks));
        console.log(
          `[residency] predicted chunked prefill at C=${C}: ${predicted.toFixed(0)} ms ` +
            `(${(N / (predicted / 1000)).toFixed(0)} tok/s) vs ${(linear + quadratic).toFixed(0)} ms one-shot`,
        );
        // Peak scratch: the square attention buffers become C × maxContext.
        const square = 2 * config.heads * N * N * 4;
        const strip = 2 * config.heads * C * N * 4;
        console.log(
          `[residency]   attention buffers ${mib(square)} MiB → ${mib(strip)} MiB`,
        );
      }
      expect(points.length).toBe(lengths.length);
    } finally {
      session.destroy();
    }
  }, 900_000);
});
