/**
 * Perplexity and throughput, fp32 against int4.
 *
 * Perplexity is the check the loose int4 thresholds cannot replace. At `abs 5e-2, rel 8e-2`
 * the parity table cannot tell expected quantization loss from a bug — both look like "a bit
 * off" — so the question becomes whether the model still predicts text as well. A correct
 * int4 implementation costs a small perplexity increase. A broken one costs much more.
 *
 * Decode and prefill are timed separately because they bottleneck differently. Decode is
 * matrix-by-vector, every weight read once, entirely memory-bound — which is where int4 is
 * expected to win, because it moves a quarter of the bytes. Prefill is matrix-by-matrix with
 * real reuse, and the nibble unpacking is arithmetic the fp32 kernel simply does not do, so
 * a loss there would not be surprising.
 *
 *   npm run quality
 *   npm run quality -- --q4
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { BufferPool, PipelineCache, initGPU, type GPUContext } from '../../src/gpu/index.ts';
import { HttpRangeSource } from '../../src/model/safetensors.ts';
import { WeightStore } from '../../src/model/weights.ts';
import { ForwardGraph, type ModelConfig } from '../../src/model/graph.ts';
import { Generator } from '../../src/runtime/generate.ts';
import { Session } from '../../src/runtime/session.ts';
import { Tokenizer, type TokenizerJSON } from '../../src/tokenizer/tokenizer.ts';
import tokenizerJson from '../fixtures/tokenizer.json';

const FP32_URL = '/test/fixtures/model.safetensors';
const Q4_URL = '/models/qwen2.5-0.5b-q4.enargeia';

const useQ4 = Boolean(import.meta.env?.VITE_QUALITY_Q4);
/** q4 | q8 | f16 | f32 — the tied embedding's precision; every other tensor stays int4. */
const embedDType = (import.meta.env?.VITE_QUALITY_EMBED ?? 'q4') as string;
const VARIANT_URL: Record<string, string> = {
  q4: '/models/qwen2.5-0.5b-q4.enargeia',
  q8: '/models/qwen2.5-0.5b-embed-q8.enargeia',
  f16: '/models/qwen2.5-0.5b-embed-f16.enargeia',
  f32: '/models/qwen2.5-0.5b-embed-f32.enargeia',
};
const contextTokens = Number(import.meta.env?.VITE_QUALITY_TOKENS ?? 96);
const useCache = Boolean(import.meta.env?.VITE_QUALITY_CACHE);

let ctx: GPUContext;
let pool: BufferPool;
let weights: WeightStore;
let graph: ForwardGraph;
let generator: Generator;
let pipelines: PipelineCache;
let tokens: number[];
let config: ModelConfig;

beforeAll(async () => {
  ctx = await initGPU({ label: 'quality' });
  pool = new BufferPool(ctx.device, { label: 'quality', maxIdleBytes: 128 * 1024 * 1024 });
  const cache = new PipelineCache(ctx.device);

  const sidecar = await (await fetch('/test/fixtures/reference.json')).json();
  config = sidecar.config as ModelConfig;

  // The larger passage exists because 95 positions could not resolve the M4 ablation's top
  // three variants — they spanned 0.2%, well inside the sampling noise of a single short text.
  // With the cache, scoring a thousand positions costs one O(context) step each instead of an
  // O(context^2) recomputation, so the sample size stopped being the limit.
  const file = Boolean(import.meta.env?.VITE_QUALITY_LARGE)
    ? '/test/fixtures/heldout_large.txt'
    : '/test/fixtures/heldout.txt';
  const text = await (await fetch(file)).text();
  const tokenizer = Tokenizer.fromJSON(tokenizerJson as unknown as TokenizerJSON);
  tokens = tokenizer.encode(text).slice(0, contextTokens);

  const load = useQ4 ? WeightStore.loadQuantized : WeightStore.load;
  weights = await load(ctx.device, ctx.profile, {
    ref: { modelId: 'Qwen/Qwen2.5-0.5B-Instruct', revision: 'quality', file: 'model' },
    source: new HttpRangeSource(useQ4 ? (VARIANT_URL[embedDType] ?? Q4_URL) : FP32_URL),
    noCache: true,
  });
  pipelines = cache;
  if (!useCache) {
    graph = new ForwardGraph(ctx.device, pool, cache, weights, config, tokens.length);
    generator = new Generator(ctx, pool, graph);
  }
}, 900_000);

afterAll(() => {
  graph?.destroy();
  weights?.destroy();
  pool?.destroy();
  ctx?.device.destroy();
});

/** Mean negative log-likelihood of each true next token, in nats. */
function nll(logits: Float32Array, target: number): number {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - max);
  return -(logits[target] - max - Math.log(sum));
}

describe(useQ4 ? `int4/embed-${embedDType}` : 'fp32', () => {
  it('reports resident weights', () => {
    console.log(
      `[quality] ${useQ4 ? `int4/embed-${embedDType}` : 'fp32'} · ${weights.stats.tensorCount} tensors · ` +
        `${(weights.stats.gpuBytes / 1048576).toFixed(1)} MiB resident · ` +
        `${graph ? `${graph.embeddingParts} embedding part(s) · ${graph.dispatchCount} dispatches` : 'cached decode path'}`,
    );
    expect(weights.stats.gpuBytes).toBeGreaterThan(0);
  });

  it('computes perplexity on the held-out passage', async () => {
    let total = 0;
    let counted = 0;

    if (useCache) {
      // One O(context) decode step per position, against the cache. The no-cache path below
      // recomputes the whole prefix every time, which is what limited the M4 evaluation to
      // 95 positions.
      const session = new Session(ctx.device, ctx.queue, pool, pipelines, weights, config, {
        maxContext: tokens.length + 8,
      });
      try {
        await session.prime(tokens.slice(0, 1));
        for (let i = 1; i < tokens.length - 1; i++) {
          const logits = await session.logitsAfter(tokens[i]);
          total += nll(logits, tokens[i + 1]);
          counted++;
        }
      } finally {
        session.destroy();
      }
    } else {
      for (let i = 1; i < tokens.length; i++) {
        const logits = await generator.logits(tokens.slice(0, i));
        total += nll(logits, tokens[i]);
        counted++;
      }
    }
    const perplexity = Math.exp(total / counted);
    console.log(
      `[quality] perplexity ${perplexity.toFixed(4)} over ${counted} positions ` +
        `(${useQ4 ? `int4/embed-${embedDType}` : 'fp32'})`,
    );
    expect(Number.isFinite(perplexity)).toBe(true);
    // A broken quantization does not land near the fp32 baseline; it lands in the hundreds or
    // produces NaN. The comparison against the recorded fp32 number is the real check and is
    // done in BENCH.md, since both runs cannot share a process at 2 GB resident.
    expect(perplexity).toBeLessThan(1000);
  }, 1_800_000);

  it.skipIf(useCache)('times decode and prefill separately', async () => {
    const warm = 2;
    const iters = 6;

    // Decode: one token appended to a full context, which is the shape every step after the
    // first has. m = 1 for every projection.
    const decodeContext = tokens.slice(0, tokens.length - 1);
    for (let i = 0; i < warm; i++) await generator.step(decodeContext);
    const decodeStart = performance.now();
    for (let i = 0; i < iters; i++) await generator.step(decodeContext);
    const decodeMs = (performance.now() - decodeStart) / iters;

    // Prefill: the whole context at once, m = seq for every projection.
    const prefillTokens = tokens.length;
    for (let i = 0; i < warm; i++) await generator.step(tokens);
    const prefillStart = performance.now();
    for (let i = 0; i < iters; i++) await generator.step(tokens);
    const prefillMs = (performance.now() - prefillStart) / iters;

    console.log(
      `[quality] ${useQ4 ? `int4/embed-${embedDType}` : 'fp32'} decode ${decodeMs.toFixed(1)} ms/step ` +
        `(context ${decodeContext.length}) · prefill ${prefillMs.toFixed(1)} ms ` +
        `(${prefillTokens} tokens, ${(prefillTokens / (prefillMs / 1000)).toFixed(0)} tok/s)`,
    );
    expect(decodeMs).toBeGreaterThan(0);
    expect(prefillMs).toBeGreaterThan(0);
  }, 900_000);
});
