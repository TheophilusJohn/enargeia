/**
 * The M6 ablation harness.
 *
 * One number per row of the table in BENCH.md, measured identically every time so the rows are
 * comparable: decode tokens/sec at context 512, prefill tokens/sec at 128 and 2048, and cold
 * pipeline compilation time. Each optimization is measured on its own by toggling it, not by
 * reading the difference between two runs that changed several things.
 *
 *   npm run ablation
 *   npm run ablation -- --off fusedNorm,kernelSelect
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { BufferPool, PipelineCache, initGPU, type GPUContext } from '../../src/gpu/index.ts';
import { HttpRangeSource } from '../../src/model/safetensors.ts';
import { WeightStore } from '../../src/model/weights.ts';
import type { ModelConfig } from '../../src/model/graph.ts';
import { Session } from '../../src/runtime/session.ts';
import { GREEDY } from '../../src/kernels/sample.ts';
import { OPTIMIZATIONS, setOptimizations, describeOptimizations } from '../../src/runtime/options.ts';
import { precompile, allKernelSpecs } from '../../src/runtime/precompile.ts';
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

const disabled = String(import.meta.env?.VITE_ABLATION_OFF ?? '')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean);

beforeAll(async () => {
  for (const key of disabled) {
    if (!(key in OPTIMIZATIONS)) throw new Error(`unknown optimization "${key}"`);
    setOptimizations({ [key]: false } as Partial<typeof OPTIMIZATIONS>);
  }

  ctx = await initGPU({ label: 'ablation' });
  pool = new BufferPool(ctx.device, { label: 'ablation', maxIdleBytes: 256 * 1024 * 1024 });
  pipelines = new PipelineCache(ctx.device);

  const sidecar = await (await fetch('/test/fixtures/reference.json')).json();
  config = sidecar.config as ModelConfig;

  const text = await (await fetch('/test/fixtures/heldout_large.txt')).text();
  const tokenizer = Tokenizer.fromJSON(tokenizerJson as unknown as TokenizerJSON);
  const once = tokenizer.encode(text);
  corpus = [];
  while (corpus.length < MAX_CONTEXT + 64) corpus.push(...once);

  weights = await WeightStore.loadQuantized(ctx.device, ctx.profile, {
    ref: { modelId: 'Qwen/Qwen2.5-0.5B-Instruct', revision: 'ablation', file: 'model' },
    source: new HttpRangeSource(MODEL_URL),
    noCache: true,
  });
}, 900_000);

afterAll(() => {
  weights?.destroy();
  pool?.destroy();
  ctx?.device.destroy();
});

function median(xs: number[]): number {
  const sorted = [...xs].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

describe('ablation', () => {
  it('reports the configuration', () => {
    console.log(`[ablation] ${describeOptimizations()}`);
    console.log(`[ablation] f16 available: ${ctx.profile.f16}`);
  });

  it('first-generation TTFT, with and without precompilation', async () => {
    // What item 1 is actually supposed to buy: not the compile time in isolation, but how much
    // of it lands on the user's first token. Measured on a fresh PipelineCache so nothing is
    // already warm, and on a short prompt so prefill does not swamp the effect.
    const cold = new PipelineCache(ctx.device);
    const specs = allKernelSpecs(weights, config);

    let compileMs = 0;
    if (OPTIMIZATIONS.precompile) {
      const started = performance.now();
      let done = 0;
      await precompile(ctx.device, cold, specs, () => {
        done++;
      });
      compileMs = performance.now() - started;
      expect(done).toBe(specs.length);
    }

    const session = new Session(ctx.device, ctx.queue, pool, cold, weights, config, {
      maxContext: MAX_CONTEXT,
      sampling: GREEDY,
    });
    try {
      const prompt = corpus.slice(0, 32);
      const started = performance.now();
      const result = await session.generate({ prompt, maxTokens: 1 });
      const ttftMs = performance.now() - started;
      console.log(
        `[ablation] precompile ${OPTIMIZATIONS.precompile ? 'on ' : 'off'}: ` +
          `${specs.length} pipelines in ${compileMs.toFixed(0)} ms · ` +
          `first-generation TTFT ${ttftMs.toFixed(0)} ms ` +
          `(prefill ${(result.timing.prefillSeconds * 1000).toFixed(0)} ms)`,
      );
      expect(result.tokens).toHaveLength(1);
    } finally {
      session.destroy();
    }
  }, 300_000);

  for (const context of [512, 2048]) {
    it(`decode at context ${context}`, async () => {
      const session = new Session(ctx.device, ctx.queue, pool, pipelines, weights, config, {
        maxContext: MAX_CONTEXT,
        sampling: GREEDY,
      });
      try {
        const prompt = corpus.slice(0, context - 16);
        await session.generate({ prompt: prompt.slice(0, 8), maxTokens: 2 });
        session.reset();
        const result = await session.generate({ prompt, maxTokens: 16 });
        const ms = median(result.timing.interTokenSeconds) * 1000;
        console.log(
          `[ablation] decode ctx ${String(context).padStart(4)}: ` +
            `${(1000 / ms).toFixed(1)} tok/s (${ms.toFixed(1)} ms)`,
        );
        expect(ms).toBeGreaterThan(0);
      } finally {
        session.destroy();
      }
    }, 900_000);
  }

  for (const context of [128, 2048]) {
    it(`prefill at ${context}`, async () => {
      const session = new Session(ctx.device, ctx.queue, pool, pipelines, weights, config, {
        maxContext: MAX_CONTEXT,
        sampling: GREEDY,
      });
      try {
        const prompt = corpus.slice(0, context);
        await session.generate({ prompt: prompt.slice(0, 8), maxTokens: 1 });
        session.reset();

        const runs: number[] = [];
        for (let i = 0; i < 3; i++) {
          const started = performance.now();
          await session.prime(prompt);
          runs.push((performance.now() - started) / 1000);
          session.reset();
        }
        const seconds = median(runs);
        console.log(
          `[ablation] prefill ${String(context).padStart(4)}: ` +
            `${(context / seconds).toFixed(0)} tok/s (${(seconds * 1000).toFixed(0)} ms)`,
        );
        expect(seconds).toBeGreaterThan(0);
      } finally {
        session.destroy();
      }
    }, 900_000);
  }
});
