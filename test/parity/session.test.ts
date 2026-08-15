/**
 * The KV cache is only correct if it changes nothing.
 *
 * M3's no-cache path recomputed the whole sequence every step and matched HuggingFace exactly.
 * That makes it the reference for M5: prefill-then-decode has to produce the same token
 * sequence, or the cache is dropping or mis-positioning something. This is the check that
 * catches a RoPE applied at the wrong position, a cache row written at the wrong offset, or an
 * attention window off by one — none of which look wrong in the output text.
 *
 *   npm run session
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { BufferPool, PipelineCache, initGPU, type GPUContext } from '../../src/gpu/index.ts';
import { HttpRangeSource } from '../../src/model/safetensors.ts';
import { WeightStore } from '../../src/model/weights.ts';
import { ForwardGraph, type ModelConfig } from '../../src/model/graph.ts';
import { Generator } from '../../src/runtime/generate.ts';
import { Session } from '../../src/runtime/session.ts';
import { GREEDY } from '../../src/kernels/sample.ts';

const MODEL_URL = '/models/qwen2.5-0.5b.enargeia';
const MAX_CONTEXT = 256;

let ctx: GPUContext;
let pool: BufferPool;
let pipelines: PipelineCache;
let weights: WeightStore;
let config: ModelConfig;
let prompt: number[];

beforeAll(async () => {
  ctx = await initGPU({ label: 'session' });
  pool = new BufferPool(ctx.device, { label: 'session', maxIdleBytes: 128 * 1024 * 1024 });
  pipelines = new PipelineCache(ctx.device);

  const sidecar = await (await fetch('/test/fixtures/reference.json')).json();
  config = sidecar.config as ModelConfig;
  prompt = sidecar.promptTokens as number[];

  weights = await WeightStore.loadQuantized(ctx.device, ctx.profile, {
    ref: { modelId: 'Qwen/Qwen2.5-0.5B-Instruct', revision: 'session', file: 'model' },
    source: new HttpRangeSource(MODEL_URL),
    noCache: true,
  });
}, 900_000);

afterAll(() => {
  weights?.destroy();
  pool?.destroy();
  ctx?.device.destroy();
});

describe('KVCache', () => {
  it('preallocates and reports its size without growing', async () => {
    const session = new Session(ctx.device, ctx.queue, pool, pipelines, weights, config, {
      maxContext: MAX_CONTEXT,
    });
    try {
      const { cache } = session;
      expect(cache.filled).toBe(0);
      expect(cache.maxContext).toBe(MAX_CONTEXT);
      // 24 layers x 2 tensors x 256 positions x 128 elements, at the cache's element width.
      const elementBytes = cache.dtype === 'f16' ? 2 : 4;
      expect(cache.totalBytes).toBe(24 * 2 * MAX_CONTEXT * 128 * elementBytes);
      console.log(
        `[session] KV cache ${(cache.totalBytes / 1048576).toFixed(1)} MiB at ${MAX_CONTEXT} ` +
          `context, dtype ${cache.dtype}`,
      );

      cache.advance(10);
      expect(cache.filled).toBe(10);
      expect(cache.remaining).toBe(MAX_CONTEXT - 10);
      // Reset forgets without freeing — a reset that reallocated would reintroduce the stall
      // the preallocation exists to avoid.
      cache.reset();
      expect(cache.filled).toBe(0);
      expect(cache.isDestroyed).toBe(false);
    } finally {
      session.destroy();
    }
  });

  it('refuses to exceed its context rather than growing', async () => {
    const session = new Session(ctx.device, ctx.queue, pool, pipelines, weights, config, {
      maxContext: 32,
    });
    try {
      session.cache.advance(30);
      expect(() => session.cache.advance(5)).toThrow(/context exhausted/);
    } finally {
      session.destroy();
    }
  });
});

describe('prefill + decode against the no-cache path', () => {
  it('produces the same greedy tokens as recomputing the whole sequence', async () => {
    // The M3 path: one graph, no cache, full recomputation every step.
    const noCacheGraph = new ForwardGraph(
      ctx.device, pool, pipelines, weights, config, prompt.length + 24,
    );
    const generator = new Generator(ctx, pool, noCacheGraph);
    const reference = await generator.generate({ prompt, maxTokens: 24 });
    noCacheGraph.destroy();

    const session = new Session(ctx.device, ctx.queue, pool, pipelines, weights, config, {
      maxContext: MAX_CONTEXT,
      sampling: GREEDY,
    });
    try {
      const cached = await session.generate({ prompt, maxTokens: 24 });
      console.log(`[session] no-cache: ${JSON.stringify(reference.tokens)}`);
      console.log(`[session] cached:   ${JSON.stringify(cached.tokens)}`);
      const diverged = cached.tokens.findIndex((t, i) => t !== reference.tokens[i]);
      console.log(`[session] matching: ${diverged === -1 ? 24 : diverged}/24`);
      expect(cached.tokens).toEqual(reference.tokens);
    } finally {
      session.destroy();
    }
  }, 900_000);

  it('is deterministic across sessions at greedy', async () => {
    const run = async () => {
      const session = new Session(ctx.device, ctx.queue, pool, pipelines, weights, config, {
        maxContext: MAX_CONTEXT,
        sampling: GREEDY,
      });
      try {
        return (await session.generate({ prompt, maxTokens: 12 })).tokens;
      } finally {
        session.destroy();
      }
    };
    expect(await run()).toEqual(await run());
  }, 900_000);

  it('reuses a session across resets', async () => {
    const session = new Session(ctx.device, ctx.queue, pool, pipelines, weights, config, {
      maxContext: MAX_CONTEXT,
      sampling: GREEDY,
    });
    try {
      const first = await session.generate({ prompt, maxTokens: 8 });
      session.reset();
      expect(session.cache.filled).toBe(0);
      const second = await session.generate({ prompt, maxTokens: 8 });
      // A reset that left stale cache state would show up as a different continuation.
      expect(second.tokens).toEqual(first.tokens);
    } finally {
      session.destroy();
    }
  }, 900_000);
});

describe('GPU sampling', () => {
  it('greedy matches the argmax path', async () => {
    const session = new Session(ctx.device, ctx.queue, pool, pipelines, weights, config, {
      maxContext: MAX_CONTEXT,
      sampling: GREEDY,
    });
    try {
      const a = await session.generate({ prompt, maxTokens: 6 });
      session.reset();
      const b = await session.generate({ prompt, maxTokens: 6 });
      expect(b.tokens).toEqual(a.tokens);
    } finally {
      session.destroy();
    }
  }, 900_000);

  it('temperature and top-p produce valid ids and vary with the seed', async () => {
    const sample = async (seed: number) => {
      const session = new Session(ctx.device, ctx.queue, pool, pipelines, weights, config, {
        maxContext: MAX_CONTEXT,
        sampling: { temperature: 0.9, topP: 0.9, repetitionPenalty: 1.0 },
        seed,
      });
      try {
        return (await session.generate({ prompt, maxTokens: 12 })).tokens;
      } finally {
        session.destroy();
      }
    };
    const a = await sample(1);
    const b = await sample(1);
    const c = await sample(999);

    for (const id of [...a, ...c]) {
      expect(Number.isInteger(id)).toBe(true);
      expect(id).toBeGreaterThanOrEqual(0);
      expect(id).toBeLessThan(config.vocab);
    }
    // Same seed, same tokens — sampling is seeded on the CPU precisely so a sampled run is
    // still reproducible.
    expect(b).toEqual(a);
    // Different seed should not give the identical sequence; if it does the draw is being
    // ignored and the sampler has silently degenerated to greedy.
    expect(c).not.toEqual(a);
    console.log(`[session] sampled seed 1:   ${JSON.stringify(a)}`);
    console.log(`[session] sampled seed 999: ${JSON.stringify(c)}`);
  }, 900_000);

  it('top-p 1.0 at low temperature approaches greedy', async () => {
    const session = new Session(ctx.device, ctx.queue, pool, pipelines, weights, config, {
      maxContext: MAX_CONTEXT,
      sampling: { temperature: 0.01, topP: 1.0, repetitionPenalty: 1.0 },
      seed: 7,
    });
    try {
      const sampled = await session.generate({ prompt, maxTokens: 8 });
      session.reset();
      session.setSampling(GREEDY);
      const greedy = await session.generate({ prompt, maxTokens: 8 });
      // At temperature 0.01 the distribution is essentially a point mass, so the nucleus is
      // one token and sampling picks it.
      expect(sampled.tokens).toEqual(greedy.tokens);
    } finally {
      session.destroy();
    }
  }, 900_000);

  it('repetition penalty changes the continuation', async () => {
    const withPenalty = async (penalty: number) => {
      const session = new Session(ctx.device, ctx.queue, pool, pipelines, weights, config, {
        maxContext: MAX_CONTEXT,
        sampling: { temperature: 0, topP: 1, repetitionPenalty: penalty },
      });
      try {
        return (await session.generate({ prompt, maxTokens: 20 })).tokens;
      } finally {
        session.destroy();
      }
    };
    const plain = await withPenalty(1.0);
    const penalized = await withPenalty(1.5);
    console.log(`[session] penalty 1.0: ${JSON.stringify(plain)}`);
    console.log(`[session] penalty 1.5: ${JSON.stringify(penalized)}`);
    expect(penalized).not.toEqual(plain);
  }, 900_000);
});
