/**
 * Estimating prompt-lookup speculative decoding before building it.
 *
 * Speculation multiplies whatever the per-step cost already is, and that cost changed: decode
 * was 3.89 µs/position and scaling with context when this was scoped, and is now flat at
 * ~23 ms/step. The gain is worth recomputing against the new numbers.
 *
 * Two quantities decide it, and both are measured here rather than assumed:
 *
 *   1. accept rate — how often the n-gram draft matches what the model actually produces.
 *      Measured by generating greedily and replaying prompt-lookup over the real sequence, so
 *      it is this model on this text rather than a number from a paper about a different one.
 *   2. verify cost — what a batched forward over d tokens costs against a single decode step.
 *      Measured with the prefill graph at small batch sizes.
 *
 *   npm run speculate:estimate
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
let tokenizer: Tokenizer;
let passage: number[];

beforeAll(async () => {
  ctx = await initGPU({ label: 'speculate' });
  pool = new BufferPool(ctx.device, { label: 'speculate', maxIdleBytes: 256 * 1024 * 1024 });
  pipelines = new PipelineCache(ctx.device);

  const sidecar = await (await fetch('/test/fixtures/reference.json')).json();
  config = sidecar.config as ModelConfig;

  tokenizer = Tokenizer.fromJSON(tokenizerJson as unknown as TokenizerJSON);
  const text = await (await fetch('/test/fixtures/heldout_large.txt')).text();
  passage = tokenizer.encode(text);
}, 900_000);

afterAll(() => {
  weights?.destroy();
  pool?.destroy();
  ctx?.device.destroy();
});

/**
 * Prompt lookup: find the most recent earlier occurrence of the trailing n-gram and take the
 * `depth` tokens that followed it as the draft.
 *
 * Searching backwards means the most recent match wins, which is the standard choice — a
 * repeated phrase is most likely to continue the way it continued last time.
 */
function draft(context: readonly number[], ngram: number, depth: number): number[] {
  if (context.length < ngram + 1) return [];
  const tail = context.slice(context.length - ngram);
  for (let start = context.length - ngram - 1; start >= 0; start--) {
    let match = true;
    for (let i = 0; i < ngram; i++) {
      if (context[start + i] !== tail[i]) {
        match = false;
        break;
      }
    }
    if (match) {
      return context.slice(start + ngram, start + ngram + depth);
    }
  }
  return [];
}

/** Tokens accepted at each step, replaying prompt-lookup over an already-generated sequence. */
function replay(
  prompt: readonly number[],
  generated: readonly number[],
  ngram: number,
  depth: number,
): { steps: number; accepted: number[]; hitRate: number } {
  const context = [...prompt];
  const accepted: number[] = [];
  let i = 0;
  let steps = 0;
  let hits = 0;

  while (i < generated.length) {
    const proposal = draft(context, ngram, depth);
    if (proposal.length > 0) hits++;
    // A verify pass confirms the drafted tokens and always yields at least one correct token,
    // so accepted-per-step is 1 + (matching prefix length).
    let matched = 0;
    while (
      matched < proposal.length &&
      i + matched < generated.length &&
      proposal[matched] === generated[i + matched]
    ) {
      matched++;
    }
    const advance = Math.min(matched + 1, generated.length - i);
    accepted.push(advance);
    for (let k = 0; k < advance; k++) context.push(generated[i + k]);
    i += advance;
    steps++;
  }
  return { steps, accepted, hitRate: hits / steps };
}

async function loadWeights(): Promise<void> {
  if (weights) return;
  weights = await WeightStore.loadQuantized(ctx.device, ctx.profile, {
    ref: { modelId: 'Qwen/Qwen2.5-0.5B-Instruct', revision: 'speculate', file: 'model' },
    source: new HttpRangeSource(MODEL_URL),
    noCache: true,
  });
}

describe('accept rate on real continuations', () => {
  const scenarios: Array<{ name: string; prompt: () => number[]; tokens: number }> = [
    {
      name: 'open generation',
      prompt: () => passage.slice(0, 256),
      tokens: 96,
    },
    {
      name: 'quote-heavy (asked to repeat the passage)',
      prompt: () =>
        tokenizer.encode(
          '<|im_start|>user\nRepeat the following text exactly:\n\n' +
            tokenizer.decode(passage.slice(0, 200)) +
            '<|im_end|>\n<|im_start|>assistant\n',
        ),
      tokens: 96,
    },
    {
      name: 'chat answer (no context to copy from)',
      prompt: () =>
        tokenizer.encode(
          '<|im_start|>user\nExplain why bridges have expansion joints.<|im_end|>\n' +
            '<|im_start|>assistant\n',
        ),
      tokens: 96,
    },
  ];

  for (const scenario of scenarios) {
    it(`${scenario.name}`, async () => {
      await loadWeights();
      const session = new Session(ctx.device, ctx.queue, pool, pipelines, weights, config, {
        maxContext: MAX_CONTEXT,
        sampling: GREEDY,
      });
      try {
        const prompt = scenario.prompt();
        const result = await session.generate({ prompt, maxTokens: scenario.tokens });
        const rows: string[] = [];
        for (const depth of [2, 4, 8]) {
          for (const ngram of [2, 3]) {
            const r = replay(prompt, result.tokens, ngram, depth);
            const mean = r.accepted.reduce((a, b) => a + b, 0) / r.steps;
            rows.push(
              `n=${ngram} d=${depth}: ${mean.toFixed(2)} tok/step ` +
                `(${r.steps} steps for ${result.tokens.length} tokens, draft found ${(r.hitRate * 100).toFixed(0)}%)`,
            );
          }
        }
        console.log(`[spec] ${scenario.name} — prompt ${prompt.length} tok`);
        for (const row of rows) console.log(`[spec]   ${row}`);
        expect(result.tokens.length).toBeGreaterThan(0);
      } finally {
        session.destroy();
      }
    }, 900_000);
  }
});

describe('verify cost', () => {
  it('batched forward at small depths against one decode step', async () => {
    await loadWeights();
    const session = new Session(ctx.device, ctx.queue, pool, pipelines, weights, config, {
      maxContext: MAX_CONTEXT,
      sampling: GREEDY,
    });
    try {
      // A single decode step, warmed.
      const base = passage.slice(0, 512);
      await session.generate({ prompt: base, maxTokens: 4 });
      session.reset();
      const warm = await session.generate({ prompt: base, maxTokens: 12 });
      const stepMs =
        (warm.timing.interTokenSeconds.reduce((a, b) => a + b, 0) /
          warm.timing.interTokenSeconds.length) *
        1000;

      // A batched forward over d tokens, approximated by prefill at depth d. This understates
      // the real verify cost, which also attends over the existing cache — stated as a floor.
      const rows: string[] = [];
      for (const depth of [1, 2, 4, 8]) {
        const runs: number[] = [];
        for (let i = 0; i < 5; i++) {
          session.reset();
          const started = performance.now();
          await session.prime(passage.slice(0, depth));
          runs.push(performance.now() - started);
        }
        runs.sort((a, b) => a - b);
        rows.push(`depth ${depth}: ${runs[2].toFixed(1)} ms`);
      }
      console.log(`[spec] single decode step: ${stepMs.toFixed(1)} ms`);
      for (const row of rows) console.log(`[spec] batched forward ${row}`);
      expect(stepMs).toBeGreaterThan(0);
    } finally {
      session.destroy();
    }
  }, 900_000);
});
