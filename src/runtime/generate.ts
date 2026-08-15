/**
 * Greedy decode.
 *
 * No KV cache: every step recomputes the full sequence. That is quadratic and intentional
 * for M3 — the deliverable is a forward pass that agrees with the reference, and a cache is a
 * change worth making against something already known to be correct.
 *
 * One readback per token, and it is 8 bytes: the argmax kernel picks the token on the GPU and
 * only the chosen id crosses back. Reading 151,936 logits per step instead would be 608 KB
 * per token and a `mapAsync` round trip on the critical path.
 */

import { readBuffer, readFloats } from '../gpu/readback.ts';
import type { BufferPool } from '../gpu/pool.ts';
import type { ForwardGraph } from '../model/graph.ts';

export interface GenerateOptions {
  /** Prompt token ids. */
  prompt: readonly number[];
  maxTokens: number;
  /** Stop when one of these is produced. The token is still returned. */
  stopTokens?: readonly number[];
  onToken?: (id: number, index: number) => void;
}

export interface GenerateResult {
  tokens: number[];
  /** Seconds per decode step, in order. */
  stepSeconds: number[];
}

export interface RuntimeContext {
  device: GPUDevice;
  queue: GPUQueue;
}

export class Generator {
  private readonly ctx: RuntimeContext;
  private readonly pool: BufferPool;
  private readonly graph: ForwardGraph;

  constructor(ctx: RuntimeContext, pool: BufferPool, graph: ForwardGraph) {
    this.ctx = ctx;
    this.pool = pool;
    this.graph = graph;
  }

  /** Run one forward pass over `tokens` and return the greedy next id. */
  async step(tokens: readonly number[]): Promise<number> {
    const seq = tokens.length;
    if (seq > this.graph.maxSeq) {
      throw new RangeError(`sequence ${seq} exceeds the graph's maximum of ${this.graph.maxSeq}`);
    }
    this.ctx.queue.writeBuffer(
      this.graph.tokenIds.buffer,
      0,
      new Uint32Array(tokens),
    );
    this.graph.setSequenceLength(seq);

    const encoder = this.ctx.device.createCommandEncoder({ label: 'forward' });
    this.graph.encode(encoder, seq);
    this.ctx.queue.submit([encoder.finish()]);

    const bytes = await readBuffer(this.ctx, this.pool, this.graph.result);
    return new Uint32Array(bytes)[0];
  }

  /**
   * Full logits for the last position, rather than just the argmax.
   *
   * Only perplexity needs this — decode reads the 8 bytes the argmax kernel writes, because
   * 608 KB per token across the bus is exactly the readback the decode loop exists to avoid.
   */
  async logits(tokens: readonly number[]): Promise<Float32Array> {
    const seq = tokens.length;
    this.ctx.queue.writeBuffer(this.graph.tokenIds.buffer, 0, new Uint32Array(tokens));
    this.graph.setSequenceLength(seq);

    const tap = this.graph.taps.get('logits');
    if (!tap) throw new Error('graph has no logits tap');
    const encoder = this.ctx.device.createCommandEncoder({ label: 'logits' });
    this.graph.encode(encoder, seq, tap.afterStep);
    this.ctx.queue.submit([encoder.finish()]);

    const elements = tap.elements(seq);
    const staging = this.pool.acquire(
      elements * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      'logits.read',
    );
    const copy = this.ctx.device.createCommandEncoder();
    copy.copyBufferToBuffer(tap.buffer.buffer, 0, staging.buffer, 0, elements * 4);
    this.ctx.queue.submit([copy.finish()]);
    const values = await readFloats(this.ctx, this.pool, staging);
    this.pool.release(staging);
    return values;
  }

  async generate(options: GenerateOptions): Promise<GenerateResult> {
    const stop = new Set(options.stopTokens ?? []);
    const tokens = [...options.prompt];
    const produced: number[] = [];
    const stepSeconds: number[] = [];

    for (let i = 0; i < options.maxTokens; i++) {
      const started = performance.now();
      const next = await this.step(tokens);
      stepSeconds.push((performance.now() - started) / 1000);

      tokens.push(next);
      produced.push(next);
      options.onToken?.(next, i);
      if (stop.has(next)) break;
    }

    return { tokens: produced, stepSeconds };
  }
}
