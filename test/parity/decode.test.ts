/**
 * Decode throughput, measured properly for the first time.
 *
 * Every previous "decode" number in BENCH.md was a prefill in disguise: without a cache, a
 * step at context 95 recomputed all 95 positions. With the cache a step is O(context) reads of
 * cached K and V plus one matrix-by-vector pass over the weights, which is the shape the
 * project's premise rests on and has never been measured.
 *
 * TTFT and inter-token latency are reported separately because they are different costs with
 * different fixes: TTFT is dominated by prefill over the prompt, inter-token by weight and
 * cache bandwidth.
 *
 *   npm run decode
 *   npm run decode -- --fp32
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { BufferPool, PipelineCache, initGPU, type GPUContext } from '../../src/gpu/index.ts';
import { HttpRangeSource } from '../../src/model/safetensors.ts';
import { WeightStore } from '../../src/model/weights.ts';
import type { ModelConfig } from '../../src/model/graph.ts';
import { Session } from '../../src/runtime/session.ts';
import { GREEDY } from '../../src/kernels/sample.ts';
import { OPTIMIZATIONS, setOptimizations } from '../../src/runtime/options.ts';
import { Tokenizer, type TokenizerJSON } from '../../src/tokenizer/tokenizer.ts';
import tokenizerJson from '../fixtures/tokenizer.json';

const Q4_URL = '/models/qwen2.5-0.5b.enargeia';
const FP32_URL = '/test/fixtures/model.safetensors';

const useFp32 = Boolean(import.meta.env?.VITE_DECODE_FP32);
const label = useFp32 ? 'fp32' : 'int4/q8-embed';
const MAX_CONTEXT = 2048;

let ctx: GPUContext;
let pool: BufferPool;
let pipelines: PipelineCache;
let weights: WeightStore;
let config: ModelConfig;
let corpus: number[];

const disabled = String(import.meta.env?.VITE_ABLATION_OFF ?? '')
  .split(',').map((s) => s.trim()).filter(Boolean);

beforeAll(async () => {
  for (const key of disabled) setOptimizations({ [key]: false } as Partial<typeof OPTIMIZATIONS>);
  ctx = await initGPU({ label: 'decode' });
  pool = new BufferPool(ctx.device, { label: 'decode', maxIdleBytes: 256 * 1024 * 1024 });
  pipelines = new PipelineCache(ctx.device);

  const sidecar = await (await fetch('/test/fixtures/reference.json')).json();
  config = sidecar.config as ModelConfig;

  const text = await (await fetch('/test/fixtures/heldout.txt')).text();
  const tokenizer = Tokenizer.fromJSON(tokenizerJson as unknown as TokenizerJSON);
  const once = tokenizer.encode(text);
  corpus = [];
  while (corpus.length < MAX_CONTEXT + 64) corpus.push(...once);

  const load = useFp32 ? WeightStore.load : WeightStore.loadQuantized;
  weights = await load(ctx.device, ctx.profile, {
    ref: { modelId: 'Qwen/Qwen2.5-0.5B-Instruct', revision: 'decode', file: 'model' },
    source: new HttpRangeSource(useFp32 ? FP32_URL : Q4_URL),
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

describe(`decode throughput — ${label}`, () => {
  it('reports resident memory', () => {
    console.log(
      `[decode] ${label} · ${(weights.stats.gpuBytes / 1048576).toFixed(1)} MiB weights · ` +
        `KV cache at ${MAX_CONTEXT} context adds ` +
        `${(config.layers * 2 * MAX_CONTEXT * config.kvHeads * config.headDim * (OPTIMIZATIONS.f16 ? 2 : 4) / 1048576).toFixed(1)} MiB ` +
        `(${OPTIMIZATIONS.f16 ? 'f16' : 'f32'})`,
    );
    expect(weights.stats.gpuBytes).toBeGreaterThan(0);
  });

  for (const context of [128, 512, 1024, 2048]) {
    it(`context ${context}`, async () => {
      const session = new Session(ctx.device, ctx.queue, pool, pipelines, weights, config, {
        maxContext: MAX_CONTEXT,
        sampling: GREEDY,
      });
      try {
        // Prime to just under the target so the decoded tokens land at the context of interest.
        const prompt = corpus.slice(0, context - 16);

        // Warm up first. Pipelines compile lazily on their first dispatch, and charging one
        // shader compilation to TTFT reports a number no user ever experiences twice — the
        // first run measured 1506 ms at context 128 against 2434 ms at 2048, which is the
        // compile, not the prompt.
        await session.generate({ prompt: prompt.slice(0, 8), maxTokens: 2 });
        session.reset();

        const result = await session.generate({ prompt, maxTokens: 16 });
        const inter = result.timing.interTokenSeconds;

        console.log(
          `[decode] ${label} ctx ${String(context).padStart(4)} · ` +
            `TTFT ${(result.timing.ttftSeconds * 1000).toFixed(0)} ms ` +
            `(prefill ${(result.timing.prefillSeconds * 1000).toFixed(0)} ms, ` +
            `${(prompt.length / result.timing.prefillSeconds).toFixed(0)} tok/s) · ` +
            `inter-token ${(median(inter) * 1000).toFixed(1)} ms ` +
            `= ${(1 / median(inter)).toFixed(1)} tok/s`,
        );
        expect(result.tokens).toHaveLength(16);
        expect(median(inter)).toBeGreaterThan(0);
      } finally {
        session.destroy();
      }
    }, 900_000);
  }
});
