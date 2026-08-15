/**
 * A generation session: prefill once, then decode a token at a time against the cache.
 *
 * Owns the KV cache and both graphs. The split is the point of M5 — prefill is one wide
 * dispatch set over the whole prompt, decode is a narrow one that appends a single position,
 * and until now every "decode" step was silently re-running prefill over the entire prefix.
 *
 * One readback per token, four bytes. Sampling happens on the GPU, so the logits never cross
 * the bus.
 */

import { readBuffer } from '../gpu/readback.ts';
import type { BufferPool } from '../gpu/pool.ts';
import type { PipelineCache } from '../gpu/pipeline.ts';
import type { WeightStore } from '../model/weights.ts';
import { ForwardGraph, type ModelConfig } from '../model/graph.ts';
import { DecodeGraph } from '../model/graph_decode.ts';
import { GREEDY, type SamplingParams } from '../kernels/sample.ts';
import { KVCache } from './kvcache.ts';
import { OPTIMIZATIONS } from './options.ts';

export interface SessionOptions {
  maxContext: number;
  /** KV cache precision. f16 halves it; see the decomposition in BENCH.md. */
  cacheDType?: 'f32' | 'f16';
  sampling?: SamplingParams;
  /** Seed for the sampling draws. Fixed by default so runs reproduce. */
  seed?: number;
}

export interface GenerateOptions {
  prompt: readonly number[];
  maxTokens: number;
  stopTokens?: readonly number[];
  onToken?: (id: number, index: number) => void;
}

export interface GenerateTiming {
  /** Seconds from the call to the first token — prompt processing plus one decode step. */
  ttftSeconds: number;
  /** Seconds for the prefill dispatch alone. */
  prefillSeconds: number;
  /** Per-token seconds, excluding the first. */
  interTokenSeconds: number[];
  tokensPerSecond: number;
}

export interface GenerateResult {
  tokens: number[];
  timing: GenerateTiming;
}

/** Seeded so a sampled run is reproducible; see the determinism rule in CLAUDE.md. */
function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Session {
  readonly cache: KVCache;
  readonly prefill: ForwardGraph;
  readonly decode: DecodeGraph;

  private readonly device: GPUDevice;
  private readonly queue: GPUQueue;
  private readonly pool: BufferPool;
  private readonly config: ModelConfig;
  private random: () => number;
  private sampling: SamplingParams;
  private tokens: number[] = [];

  constructor(
    device: GPUDevice,
    queue: GPUQueue,
    pool: BufferPool,
    pipelines: PipelineCache,
    weights: WeightStore,
    config: ModelConfig,
    options: SessionOptions,
  ) {
    this.device = device;
    this.queue = queue;
    this.pool = pool;
    this.config = config;
    this.sampling = options.sampling ?? GREEDY;
    this.random = mulberry32(options.seed ?? 0x5eed);

    this.cache = new KVCache(device, {
      layers: config.layers,
      kvHeads: config.kvHeads,
      headDim: config.headDim,
      maxContext: options.maxContext,
      dtype: options.cacheDType ?? (OPTIMIZATIONS.f16 ? 'f16' : 'f32'),
    });

    // The prefill graph is sized to the maximum context because a prompt can be any length up
    // to it; the decode graph is sized to one token and never changes shape.
    this.prefill = new ForwardGraph(
      device, pool, pipelines, weights, config, options.maxContext, this.cache,
    );
    this.decode = new DecodeGraph(
      device, pool, pipelines, weights, config, this.cache, this.sampling,
    );
  }

  setSampling(params: SamplingParams): void {
    this.sampling = params;
    this.decode.setSampling(params);
  }

  /** Drop the conversation without freeing anything. */
  reset(seed?: number): void {
    this.cache.reset();
    this.tokens = [];
    if (seed !== undefined) this.random = mulberry32(seed);
  }

  /**
   * Run the prompt through the prefill graph, filling the cache and producing the first
   * token's logits. Returns the seconds it took.
   */
  private async runPrefill(prompt: readonly number[]): Promise<number> {
    if (prompt.length > this.cache.remaining) {
      throw new RangeError(
        `prompt of ${prompt.length} exceeds the remaining context of ${this.cache.remaining}`,
      );
    }
    const started = performance.now();
    this.queue.writeBuffer(this.prefill.tokenIds.buffer, 0, new Uint32Array(prompt));
    this.prefill.setSequenceLength(prompt.length);

    const encoder = this.device.createCommandEncoder({ label: 'prefill' });
    this.prefill.encode(encoder, prompt.length);
    this.queue.submit([encoder.finish()]);
    await this.queue.onSubmittedWorkDone();

    this.cache.advance(prompt.length);
    this.tokens = [...prompt];
    return (performance.now() - started) / 1000;
  }

  /** One decode step: append `token` at the next position and sample what follows. */
  private async decodeStep(token: number): Promise<number> {
    const position = this.cache.filled;
    this.queue.writeBuffer(this.decode.tokenId.buffer, 0, new Uint32Array([token]));
    this.queue.writeBuffer(this.decode.history.buffer, 0, new Uint32Array(this.tokens));
    this.decode.setPosition(position, this.tokens.length, this.random());

    const encoder = this.device.createCommandEncoder({ label: 'decode' });
    this.decode.encode(encoder, position);
    this.queue.submit([encoder.finish()]);

    const bytes = await readBuffer(
      { device: this.device, queue: this.queue }, this.pool, this.decode.result,
    );
    this.cache.advance(1);
    return new Uint32Array(bytes)[0];
  }

  /**
   * Append `token` and return the full next-token logits.
   *
   * For perplexity only. The decode loop never does this — 608 KB per token is exactly the
   * readback the GPU sampler exists to avoid — but with a cache each position now costs one
   * O(context) step instead of an O(context^2) recomputation, which is what makes evaluating
   * a thousand positions cheap enough to be worth doing.
   */
  async logitsAfter(token: number): Promise<Float32Array> {
    const position = this.cache.filled;
    this.queue.writeBuffer(this.decode.tokenId.buffer, 0, new Uint32Array([token]));
    this.queue.writeBuffer(this.decode.history.buffer, 0, new Uint32Array(this.tokens));
    this.decode.setPosition(position, this.tokens.length, this.random());

    const encoder = this.device.createCommandEncoder({ label: 'decode-logits' });
    this.decode.encode(encoder, position);
    this.queue.submit([encoder.finish()]);
    this.cache.advance(1);
    this.tokens.push(token);

    const bytes = await readBuffer(
      { device: this.device, queue: this.queue }, this.pool, this.decode.logits,
    );
    return new Float32Array(bytes);
  }

  /** Prefill without generating, for evaluation harnesses. */
  async prime(prompt: readonly number[]): Promise<number> {
    return this.runPrefill(prompt);
  }

  get contextLength(): number {
    return this.cache.filled;
  }

  async generate(options: GenerateOptions): Promise<GenerateResult> {
    const stop = new Set(options.stopTokens ?? []);
    const produced: number[] = [];
    const interToken: number[] = [];

    const overallStart = performance.now();
    const prefillSeconds = await this.runPrefill(options.prompt);

    // The prefill graph produced logits for the last prompt position; the first generated
    // token comes from sampling those. Running one decode step over the final prompt token
    // would double-count it, so the first token is taken from prefill's logits via the same
    // sampling kernel the decode graph uses.
    let next = await this.sampleFromPrefill();
    let ttft = (performance.now() - overallStart) / 1000;

    produced.push(next);
    options.onToken?.(next, 0);
    this.tokens.push(next);

    for (let i = 1; i < options.maxTokens && !stop.has(next); i++) {
      if (this.cache.remaining <= 0) break;
      const started = performance.now();
      next = await this.decodeStep(next);
      interToken.push((performance.now() - started) / 1000);
      produced.push(next);
      options.onToken?.(next, i);
      this.tokens.push(next);
    }

    const mean = interToken.length
      ? interToken.reduce((a, b) => a + b, 0) / interToken.length
      : ttft;
    return {
      tokens: produced,
      timing: {
        ttftSeconds: ttft,
        prefillSeconds,
        interTokenSeconds: interToken,
        tokensPerSecond: 1 / mean,
      },
    };
  }

  /** Sample from the logits the prefill pass already produced. */
  private async sampleFromPrefill(): Promise<number> {
    const tap = this.prefill.taps.get('logits');
    if (!tap) throw new Error('prefill graph has no logits tap');
    const encoder = this.device.createCommandEncoder({ label: 'prefill-sample' });
    encoder.copyBufferToBuffer(
      tap.buffer.buffer, 0, this.decode.logits.buffer, 0, this.config.vocab * 4,
    );
    this.queue.submit([encoder.finish()]);

    this.queue.writeBuffer(this.decode.history.buffer, 0, new Uint32Array(this.tokens));
    this.decode.setPosition(this.cache.filled - 1, this.tokens.length, this.random());

    const sample = this.decode.steps[this.decode.steps.length - 1];
    const sampleEncoder = this.device.createCommandEncoder({ label: 'sample' });
    const pass = sampleEncoder.beginComputePass();
    sample.kernel.encode(pass, sample.bindGroup, sample.workgroupsFor(this.cache.filled - 1));
    pass.end();
    this.queue.submit([sampleEncoder.finish()]);

    const bytes = await readBuffer(
      { device: this.device, queue: this.queue }, this.pool, this.decode.result,
    );
    return new Uint32Array(bytes)[0];
  }

  destroy(): void {
    this.decode.destroy();
    this.prefill.destroy();
    this.cache.destroy();
  }
}
