/**
 * The single seam between the site and the engine.
 *
 * Everything else under `src/ui` talks to this file and to the telemetry snapshot it forwards.
 * No panel imports from `src/gpu`, `src/kernels`, `src/model` or `src/runtime`, and no panel
 * has ever seen a GPUBuffer. That boundary is what lets the inspector be closed, resized or
 * deleted without changing a single dispatch.
 */

import { BufferPool, PipelineCache, initGPU, GPUUnavailableError } from '../gpu/index.ts';
import type { DeviceProfile } from '../gpu/device.ts';
import { HttpRangeSource } from '../model/safetensors.ts';
import { WeightStore } from '../model/weights.ts';
import type { LoadProgress } from '../model/cache.ts';
import type { ModelConfig } from '../model/graph.ts';
import { Session } from '../runtime/session.ts';
import { precompile, allKernelSpecs } from '../runtime/precompile.ts';
import type { Telemetry } from '../runtime/telemetry.ts';
import { Tokenizer } from '../tokenizer/tokenizer.ts';

/**
 * Where the weights come from. `.env` for local development, `.env.production` for the
 * deployed site — the two differ in this value and nothing else.
 *
 * Production points at an R2 bucket on a zone subdomain. The loader issues HTTP range requests
 * to fetch and cache the file in chunks, so the bucket needs `Accept-Ranges` (R2 gives it) and
 * a CORS policy that exposes `Content-Range` (set on the bucket, not here).
 */
const MODEL_URL = import.meta.env.VITE_MODEL_URL ?? '/models/qwen2.5-0.5b.enargeia';
const TOKENIZER_URL = '/tokenizer.json';

/** Qwen2.5-0.5B-Instruct. Fixed because exactly one model ships. */
export const CONFIG: ModelConfig = {
  layers: 24,
  hidden: 896,
  heads: 14,
  kvHeads: 2,
  headDim: 64,
  intermediate: 4864,
  vocab: 151936,
  rmsNormEps: 1e-6,
  ropeTheta: 1000000,
};

/**
 * Context length. The same on every device, which it was not for one afternoon.
 *
 * A mobile adapter briefly got 1024 instead, because prefill activations were sized to the
 * whole context and the two attention buffers alone were 448 MiB at 2048. Chunking prefill
 * removed that: the buffers are now sized to a 256-query chunk and total scratch is 99 MiB
 * regardless of context. Halving the context would save about 58 MiB of a 458 MiB footprint
 * and cost half the usable conversation, which is no longer a trade worth making.
 */
export const MAX_CONTEXT = 2048;

/**
 * Longest single reply.
 *
 * A 0.5B model does not reliably emit `<|im_end|>`: measured over 24 generations on the chat
 * template, 19 stopped on it and 5 ran to the cap. The cap is what bounds a runaway, and when it
 * is what ended a reply the chat says so rather than presenting a truncated answer as finished.
 *
 * 256 rather than 512, because the cap is really a bound on how long the failure lasts. At the
 * measured 30 tok/s a runaway spent about 17 seconds visibly rambling, which reads as broken
 * whatever the notice underneath it says; 256 bounds that at about 8. Useful replies from this
 * model are well under 100 tokens, so the ceiling almost never binds on a good answer — and
 * shorter replies mean more turns before the context has to start evicting the oldest ones.
 */
export const MAX_REPLY_TOKENS = 256;

/**
 * `?maxReply=8` to force replies to hit the cap, and `?maxTurns=2` to force the trim.
 *
 * Both surfaces are otherwise only reachable by chance — ten turns through the real UI produced
 * neither — and a notice nobody has seen render is a notice that has not been shown to work.
 * Same family as `?clampStorage` and `?forceSoftware`.
 */
function devOverride(name: string, fallback: number): number {
  if (typeof location === 'undefined') return fallback;
  const value = Number(new URLSearchParams(location.search).get(name));
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

/**
 * Download and compilation are reported separately because they are different waits with
 * different failure modes, and collapsing them into one bar makes the second phase look like
 * a hang. Compilation is short — 17 pipelines in 4–13 ms, measured — but the phase label is
 * what tells a stalled-looking page apart from a stalled page.
 */
export interface LoadState {
  phase: 'idle' | 'device' | 'tokenizer' | 'download' | 'compile' | 'ready' | 'error';
  /** Bytes of weights transferred, and the total once the header is read. */
  loaded: number;
  total: number | null;
  bytesPerSecond: number | null;
  etaSeconds: number | null;
  /** True while every chunk is coming from the Cache API rather than the network. */
  warm: boolean;
  compiled: number;
  pipelines: number;
  message: string;
  error: string | null;
}

export interface EngineEvents {
  onLoad?: (state: LoadState) => void;
  onTelemetry?: (telemetry: Telemetry) => void;
}

export interface ChatTurn {
  role: 'user' | 'assistant';
  content: string;
}

export interface GenerateHandle {
  onToken: (text: string, tokenCount: number) => void;
  onDone: (stopped: 'eos' | 'limit' | 'cancelled' | 'error', message?: string) => void;
  /** Oldest exchanges were dropped to make the conversation fit the context. */
  onTrimmed?: (dropped: number) => void;
}

export class UnsupportedError extends Error {
  readonly detail: string;
  constructor(message: string, detail: string) {
    super(message);
    this.name = 'UnsupportedError';
    this.detail = detail;
  }
}

export class Engine {
  private pool: BufferPool | null = null;
  private pipelines: PipelineCache | null = null;
  private weights: WeightStore | null = null;
  private session: Session | null = null;
  private tokenizer: Tokenizer | null = null;
  private cancelled = false;

  profile: DeviceProfile | null = null;
  /** Resolved from the device at load time. */
  maxContext = MAX_CONTEXT;
  /** Longest reply, after any dev override — the chat quotes it in the truncation notice. */
  get replyCap(): number {
    return devOverride('maxReply', MAX_REPLY_TOKENS);
  }
  /** Bytes of weights resident on the GPU, for the memory ledger. */
  weightBytes = 0;
  /** Compile time and load time, reported on the ready panel rather than thrown away. */
  downloadSeconds = 0;
  compileSeconds = 0;
  warmLoad = false;

  private state: LoadState = {
    phase: 'idle',
    loaded: 0,
    total: null,
    bytesPerSecond: null,
    etaSeconds: null,
    warm: false,
    compiled: 0,
    pipelines: 0,
    message: '',
    error: null,
  };

  private readonly events: EngineEvents;

  constructor(events: EngineEvents = {}) {
    this.events = events;
  }

  get ready(): boolean {
    return this.session !== null;
  }

  private emit(patch: Partial<LoadState>): void {
    this.state = { ...this.state, ...patch };
    this.events.onLoad?.(this.state);
  }

  /** Whether this browser can run the engine at all, checked before anything is downloaded. */
  static supported(): boolean {
    return typeof navigator !== 'undefined' && 'gpu' in navigator;
  }

  async load(): Promise<void> {
    if (!Engine.supported()) {
      throw new UnsupportedError(
        'This browser does not expose WebGPU.',
        'navigator.gpu is undefined.',
      );
    }

    this.emit({ phase: 'device', message: 'Requesting a GPU adapter' });
    let ctx;
    try {
      ctx = await initGPU({ label: 'enargeia' });
    } catch (error) {
      if (error instanceof GPUUnavailableError) {
        throw new UnsupportedError(
          'WebGPU is present but no adapter was available.',
          error.message,
        );
      }
      throw error;
    }
    this.profile = ctx.profile;
    this.ctx = ctx;
    this.pool = new BufferPool(ctx.device, { label: 'app', maxIdleBytes: 128 * 1024 * 1024 });
    this.pipelines = new PipelineCache(ctx.device);

    this.emit({ phase: 'tokenizer', message: 'Loading the tokenizer' });
    this.tokenizer = await Tokenizer.fromURL(TOKENIZER_URL);

    this.emit({ phase: 'download', message: 'Downloading weights' });
    const downloadStarted = performance.now();
    this.weights = await WeightStore.loadQuantized(ctx.device, ctx.profile, {
      ref: { modelId: 'Qwen/Qwen2.5-0.5B-Instruct', revision: 'q4-embed-q8', file: 'model' },
      source: new HttpRangeSource(MODEL_URL),
      // Overridable for measurement: the right value depends on round-trip latency to whatever
      // is serving the weights, which is not knowable from here.
      concurrency: Number(new URLSearchParams(location.search).get('concurrency')) || undefined,
      // Diagnostic only: skips the Cache API entirely, which is how the cost of populating it
      // was separated from the cost of the download.
      noCache: new URLSearchParams(location.search).has('nocache'),
      onProgress: (progress: LoadProgress) => {
        this.emit({
          loaded: progress.loaded,
          total: progress.total,
          bytesPerSecond: progress.bytesPerSecond,
          etaSeconds: progress.etaSeconds,
          warm: progress.warm,
          message: progress.warm ? 'Reading weights from cache' : 'Downloading weights',
        });
      },
    });
    this.downloadSeconds = (performance.now() - downloadStarted) / 1000;
    this.weightBytes = this.weights.stats.gpuBytes;
    this.warmLoad = this.weights.stats.warm;

    // Compilation is its own phase and its own bar. It runs after the download rather than
    // during it only because the specs depend on which embedding dtype the file turned out to
    // carry; the cost it removes still lands before the first token instead of on it.
    const specs = allKernelSpecs(this.weights, CONFIG);
    this.emit({ phase: 'compile', compiled: 0, pipelines: specs.length,
      message: 'Compiling compute pipelines' });
    const compileStarted = performance.now();
    let compiled = 0;
    await precompile(ctx.device, this.pipelines, specs, () => {
      compiled++;
      this.emit({ compiled });
    });
    this.compileSeconds = (performance.now() - compileStarted) / 1000;

    this.maxContext = MAX_CONTEXT;
    this.session = new Session(
      ctx.device, ctx.queue, this.pool, this.pipelines, this.weights, CONFIG,
      {
        maxContext: this.maxContext,
        profile: ctx.profile,
        sampling: { temperature: 0.7, topP: 0.9, repetitionPenalty: 1.1 },
        seed: Date.now() & 0xffff,
        onTelemetry: (telemetry) => this.events.onTelemetry?.(telemetry),
      },
    );
    this.session.setWeightBytes(this.weightBytes);
    this.emit({ phase: 'ready', message: 'Ready' });
  }

  private ctx: Awaited<ReturnType<typeof initGPU>> | null = null;

  /** Turn per-kernel GPU timing on or off. Costs extra passes, so the inspector says it is on. */
  setInstrumented(enabled: boolean): void {
    this.session?.setInstrumented(enabled);
  }

  setAttentionSampling(enabled: boolean, layer = CONFIG.layers - 1): void {
    if (!this.session) return;
    this.session.inspectAttention = enabled;
    this.session.inspectLayer = layer;
  }

  /** How many bindings the embedding table needed on this device. 1 at int8 on most. */
  get embeddingParts(): number {
    const embedding = this.weights?.embedding;
    if (!embedding) return 1;
    return embedding.quantParts?.length ?? embedding.parts.length;
  }

  get timestampQuery(): boolean {
    return this.profile?.timestampQuery ?? false;
  }

  cancel(): void {
    this.cancelled = true;
  }

  /**
   * Qwen's chat template, applied here rather than in the tokenizer because it is a property
   * of the instruct fine-tune, not of the BPE.
   *
   * User text is encoded with `allowSpecial: false` so a visitor typing "<|im_start|>system"
   * gets those characters as literal bytes instead of a forged role.
   */
  private renderPrompt(turns: readonly ChatTurn[]): number[] {
    const tok = this.tokenizer!;
    const ids: number[] = [];
    const literal = (text: string) => ids.push(...tok.encode(text, { allowSpecial: false }));
    const control = (text: string) => ids.push(...tok.encode(text, { allowSpecial: true }));

    control('<|im_start|>system\n');
    literal('You are a helpful assistant.');
    control('<|im_end|>\n');
    for (const turn of turns) {
      control(`<|im_start|>${turn.role}\n`);
      literal(turn.content);
      control('<|im_end|>\n');
    }
    control('<|im_start|>assistant\n');
    return ids;
  }

  /**
   * Generate a reply to the conversation so far.
   *
   * The cache is reset and the whole conversation re-prefilled each turn rather than appended
   * to. Prefill runs at ~1100 tok/s, so a 600-token conversation costs about half a second and
   * the alternative — keeping cache state consistent with an edited transcript — is a class of
   * bug that produces plausible text while being wrong.
   */
  async chat(turns: readonly ChatTurn[], handle: GenerateHandle): Promise<void> {
    const session = this.session;
    const tok = this.tokenizer;
    if (!session || !tok) throw new Error('engine not loaded');

    this.cancelled = false;

    // Drop the oldest exchanges until the prompt fits, keeping room to answer.
    //
    // Refusing instead — which is what this did — dead-ends the conversation permanently: once
    // a few verbose replies fill 2048 tokens, every later message fails and there is nothing
    // the visitor can do about it. Measured: four turns of a rambling model was enough.
    // `?maxTurns` shrinks the window rather than the context, which forces the trim path
    // without pretending the device has less memory than it does.
    const windowLimit = devOverride('maxTurns', Number.POSITIVE_INFINITY);
    const RESERVE = 256;
    let kept = [...turns];
    let prompt = this.renderPrompt(kept);
    let dropped = 0;
    while ((prompt.length + RESERVE > this.maxContext || kept.length > windowLimit) && kept.length > 1) {
      // Two at a time so a user turn never keeps an orphaned assistant reply before it.
      kept = kept.slice(kept.length >= 3 ? 2 : 1);
      dropped++;
      prompt = this.renderPrompt(kept);
    }
    if (prompt.length + 8 > this.maxContext) {
      handle.onDone('error', `That message alone exceeds the ${this.maxContext}-token context.`);
      return;
    }
    if (dropped > 0) handle.onTrimmed?.(dropped);

    const budget = this.maxContext - prompt.length - 8;

    const imEnd = tok.idForToken('<|im_end|>');
    const endOfText = tok.idForToken('<|endoftext|>');
    const stop = [imEnd, endOfText].filter((id): id is number => id !== undefined);

    session.reset();
    const produced: number[] = [];
    let emitted = '';
    let stopped: 'eos' | 'limit' | 'cancelled' = 'limit';

    try {
      const result = await session.generate({
        prompt,
        maxTokens: Math.min(budget, devOverride('maxReply', MAX_REPLY_TOKENS)),
        stopTokens: stop,
        shouldStop: () => this.cancelled,
        onToken: (id) => {
          if (stop.includes(id)) {
            stopped = 'eos';
            return;
          }
          produced.push(id);
          // Decode the whole run each time and emit the difference. A token is not a string —
          // a multi-byte character can span two of them — so decoding incrementally per id
          // would print a replacement character mid-emoji.
          const text = tok.decode(produced, { skipSpecialTokens: true });
          if (text.length > emitted.length) {
            handle.onToken(text.slice(emitted.length), produced.length);
            emitted = text;
          }
        },
      });
      if (this.cancelled) stopped = 'cancelled';
      else if (result.tokens.some((id) => stop.includes(id))) stopped = 'eos';
      handle.onDone(stopped);
    } catch (error) {
      handle.onDone('error', error instanceof Error ? error.message : String(error));
    }
  }

  destroy(): void {
    this.session?.destroy();
    this.weights?.destroy();
    this.pool?.destroy();
    this.ctx?.device.destroy();
  }
}
