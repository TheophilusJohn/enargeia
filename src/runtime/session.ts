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
import { Rolling, kernelGroup, type KernelTiming, type Telemetry } from './telemetry.ts';
import { readBuffer as readRaw } from '../gpu/readback.ts';
import type { DeviceProfile } from '../gpu/device.ts';

/**
 * Tokens per prefill pass. 256 from the residency measurement: it takes the attention scratch
 * from 448 MiB to 56 while predicting a *faster* prefill than one shot, and going lower buys
 * little more memory while adding submits.
 */
export const DEFAULT_PREFILL_CHUNK = 256;

export interface SessionOptions {
  maxContext: number;
  /** Queries per prefill pass. Defaults to {@link DEFAULT_PREFILL_CHUNK}. */
  prefillChunk?: number;
  /** Reported in telemetry so the UI can name the fallback path that engaged. */
  profile?: DeviceProfile;
  /**
   * Publish a telemetry snapshot. Called at most 30 times a second regardless of decode rate:
   * faster is invisible and steals GPU time from the thing being measured.
   */
  onTelemetry?: (telemetry: Telemetry) => void;
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
  /** Checked between steps. A generation the user has abandoned should stop costing GPU time. */
  shouldStop?: () => boolean;
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

  private readonly onTelemetry: ((telemetry: Telemetry) => void) | undefined;
  private readonly deviceProfile: DeviceProfile | undefined;
  private readonly rollingMs = new Rolling(16);
  private lastPublish = 0;
  readonly prefillChunk: number = DEFAULT_PREFILL_CHUNK;
  private kernelTimings: KernelTiming[] = [];
  private gpuMsPerToken: number | null = null;
  private querySet: GPUQuerySet | null = null;
  private profileEvery = 0;
  private sinceProfile = 0;
  private generated = 0;
  private phase: Telemetry['phase'] = 'idle';

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
    this.onTelemetry = options.onTelemetry;
    this.deviceProfile = options.profile;

    this.cache = new KVCache(device, {
      layers: config.layers,
      kvHeads: config.kvHeads,
      headDim: config.headDim,
      maxContext: options.maxContext,
      dtype: options.cacheDType ?? (OPTIMIZATIONS.f16 ? 'f16' : 'f32'),
    });

    // The prefill graph is sized to a *chunk*, not to the maximum context, and a long prompt
    // runs through it in several passes.
    //
    // Sizing it to `maxContext` allocated the attention score tensors as
    // heads x maxContext x maxContext — 224 MiB each at a 2048 context, 512 MiB of capacity for
    // the pair, two thirds of all scratch. That is the same mistake as the M6 over-dispatch:
    // sizing to a build-time maximum rather than to the extent actually in play. Chunking makes
    // them heads x chunk x maxContext instead, and it is also slightly *faster*, because a
    // chunk attends over the prefix it needs rather than a full square that is half masked.
    this.prefillChunk = Math.min(options.prefillChunk ?? DEFAULT_PREFILL_CHUNK, options.maxContext);
    this.prefill = new ForwardGraph(
      device, pool, pipelines, weights, config, this.prefillChunk, this.cache,
      { keyCapacity: options.maxContext },
    );
    this.decode = new DecodeGraph(
      device, pool, pipelines, weights, config, this.cache, this.sampling,
    );
  }

  /**
   * Turn on per-kernel GPU timing, profiling one step in every `everyNTokens`.
   *
   * Not free: a profiled step records ~410 separate compute passes instead of one, plus a
   * readback. The duty cycle keeps the cost bounded and the inspector reports that it is on.
   */
  setInstrumented(enabled: boolean, everyNTokens = 16): void {
    this.profileEvery = enabled ? everyNTokens : 0;
    // Profile the next step rather than the sixteenth: an inspector that shows nothing for the
    // first half of a short reply reads as broken.
    this.sinceProfile = everyNTokens;
    if (!enabled) {
      this.querySet?.destroy();
      this.querySet = null;
      this.kernelTimings = [];
      this.gpuMsPerToken = null;
    } else if (!this.querySet && this.deviceProfile?.timestampQuery) {
      this.querySet = this.device.createQuerySet({
        type: 'timestamp',
        count: Math.min(this.decode.maxQueries, this.decode.steps.length * 2),
      });
    }
  }

  get instrumented(): boolean {
    return this.profileEvery > 0 && this.querySet !== null;
  }

  private publish(force = false): void {
    if (!this.onTelemetry) return;
    const now = performance.now();
    // At most 30 times a second. The panels cannot show more and the readbacks are not free.
    if (!force && now - this.lastPublish < 33) return;
    this.lastPublish = now;

    const kvBytes = this.cache.totalBytes;
    const scratch = this.pool.stats().liveBytes;
    this.onTelemetry({
      phase: this.phase,
      tokensPerSecond: this.rollingMs.mean > 0 ? 1000 / this.rollingMs.mean : 0,
      interTokenMs: this.rollingMs.last,
      tokensGenerated: this.generated,
      contextUsed: this.cache.filled,
      contextMax: this.cache.maxContext,
      memory: {
        weightsBytes: this.weightBytes,
        kvCacheBytes: kvBytes,
        scratchBytes: scratch,
        totalBytes: this.weightBytes + kvBytes + scratch,
      },
      kernels: this.kernelTimings,
      gpuMsPerToken: this.gpuMsPerToken,
      attention: this.attentionSample,
      device: this.deviceProfile!,
      instrumented: this.instrumented,
    });
  }

  private attentionSample: Telemetry['attention'] = null;
  private weightBytes = 0;

  /** Told once at construction so the ledger does not have to reach back into the store. */
  setWeightBytes(bytes: number): void {
    this.weightBytes = bytes;
  }

  /**
   * Read one layer's attention weights for the heatmap.
   *
   * This is an extra readback that the decode loop's budget does not include, which is why it
   * happens only while the panel is open and only on the profiling duty cycle. The inspector
   * says so rather than hiding it.
   */
  private async sampleAttention(layer: number): Promise<void> {
    const heads = this.config.heads;
    const positions = Math.min(this.cache.filled, 128);
    if (positions <= 0) return;
    const step = this.decode.steps.find((s) => s.stage === 'attn_weights');
    if (!step) return;
    const source = this.decode.attentionWeights;
    if (!source) return;

    const staging = this.pool.acquire(
      heads * this.cache.maxContext * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      'inspect.attn',
    );
    const encoder = this.device.createCommandEncoder({ label: 'inspect' });
    encoder.copyBufferToBuffer(source.buffer, 0, staging.buffer, 0, staging.size);
    this.queue.submit([encoder.finish()]);
    const bytes = await readRaw({ device: this.device, queue: this.queue }, this.pool, staging);
    this.pool.release(staging);

    const all = new Float32Array(bytes);
    const trimmed = new Float32Array(heads * positions);
    for (let h = 0; h < heads; h++) {
      for (let p = 0; p < positions; p++) {
        trimmed[h * positions + p] = all[h * this.cache.maxContext + p];
      }
    }
    this.attentionSample = { layer, heads, positions, weights: trimmed };
  }

  private async readKernelTimings(count: number): Promise<void> {
    if (!this.querySet) return;
    const bytes = count * 2 * 8;
    const resolve = this.pool.acquire(
      bytes,
      GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      'profile.resolve',
    );
    const readback = this.pool.acquire(
      bytes,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      'profile.read',
    );
    const encoder = this.device.createCommandEncoder({ label: 'profile-resolve' });
    encoder.resolveQuerySet(this.querySet, 0, count * 2, resolve.buffer, 0);
    encoder.copyBufferToBuffer(resolve.buffer, 0, readback.buffer, 0, bytes);
    this.queue.submit([encoder.finish()]);

    await readback.buffer.mapAsync(GPUMapMode.READ);
    const stamps = new BigInt64Array(readback.buffer.getMappedRange().slice(0));
    readback.buffer.unmap();
    this.pool.release(readback);
    this.pool.release(resolve);

    const names = this.decode.stepKernels;
    const totals = new Map<string, { ms: number; colour: string; dispatches: number }>();
    let total = 0;
    for (let i = 0; i < count; i++) {
      const ns = stamps[i * 2 + 1] - stamps[i * 2];
      if (ns <= 0n) continue;
      const ms = Number(ns) / 1e6;
      const { group, colour } = kernelGroup(names[i] ?? 'other');
      const entry = totals.get(group) ?? { ms: 0, colour, dispatches: 0 };
      entry.ms += ms;
      entry.dispatches++;
      totals.set(group, entry);
      total += ms;
    }
    if (total <= 0) return;
    this.gpuMsPerToken = total;
    this.kernelTimings = [...totals.entries()]
      .map(([group, entry]) => ({
        group,
        colour: entry.colour,
        ms: entry.ms,
        share: entry.ms / total,
        dispatches: entry.dispatches,
      }))
      .sort((a, b) => b.ms - a.ms);
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
    this.prefill.encodeChunks(this.device, this.queue, prompt.length, (begin, count) => {
      this.queue.writeBuffer(
        this.prefill.tokenIds.buffer, 0,
        new Uint32Array(prompt.slice(begin, begin + count)),
      );
    });
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
    const shouldProfile =
      this.profileEvery > 0 && this.querySet !== null && this.sinceProfile >= this.profileEvery;
    let profiledSteps = 0;
    if (shouldProfile) {
      profiledSteps = this.decode.encodeProfiled(encoder, position, this.querySet!);
      this.sinceProfile = 0;
    } else {
      this.decode.encode(encoder, position);
      this.sinceProfile++;
    }
    this.queue.submit([encoder.finish()]);

    const bytes = await readBuffer(
      { device: this.device, queue: this.queue }, this.pool, this.decode.result,
    );
    this.cache.advance(1);
    if (shouldProfile) {
      await this.readKernelTimings(profiledSteps);
      if (this.inspectAttention) await this.sampleAttention(this.inspectLayer);
    }
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

  /** Whether the heatmap panel is open, and which layer it is showing. */
  inspectAttention = false;
  inspectLayer = 0;

  async generate(options: GenerateOptions): Promise<GenerateResult> {
    const stop = new Set(options.stopTokens ?? []);
    const produced: number[] = [];
    const interToken: number[] = [];

    const overallStart = performance.now();
    this.phase = 'prefill';
    this.rollingMs.clear();
    this.generated = 0;
    this.publish(true);
    const prefillSeconds = await this.runPrefill(options.prompt);
    this.phase = 'decoding';

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
      if (options.shouldStop?.()) break;
      const started = performance.now();
      next = await this.decodeStep(next);
      const ms = performance.now() - started;
      interToken.push(ms / 1000);
      this.rollingMs.push(ms);
      this.generated = produced.length + 1;
      produced.push(next);
      options.onToken?.(next, i);
      this.tokens.push(next);
      this.publish();
    }
    this.phase = 'idle';
    this.publish(true);

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
    this.querySet?.destroy();
    this.querySet = null;
    this.decode.destroy();
    this.prefill.destroy();
    this.cache.destroy();
  }
}
