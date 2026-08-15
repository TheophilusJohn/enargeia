/**
 * The forward pass as an ordered list of steps, built once at load.
 *
 * The decode loop walks this list. It does not construct bind groups, look up pipelines, or
 * branch on device capability per token — all of that is resolved here, at build time. With
 * roughly 400 dispatches per forward pass, per-dispatch JavaScript is measurable, and a
 * decode loop that only walks a list is a decode loop you can read.
 *
 * Buffers are sized for `maxSeq` once and bind groups outlive every token. What changes per
 * step is the uniforms: each step rewrites its `seq`, because several kernels use it as a
 * *stride* and not only as a bound — attention scores are indexed `(h*seq + i)*seq + j`, so a
 * stale seq silently reshapes the buffer rather than merely over-running it.
 *
 * This is the no-KV-cache path: every step recomputes the whole sequence. Quadratic, and
 * deliberately so — a cache is a change worth making once there is something correct to
 * compare it against.
 */

import type { PipelineCache } from '../gpu/pipeline.ts';
import type { BufferPool, PooledBuffer } from '../gpu/pool.ts';
import { ComputeKernel, type KernelSpec } from '../kernels/kernel.ts';
import { ADD } from '../kernels/add.ts';
import { ARGMAX, ARGMAX_WORKGROUPS, argmaxDims } from '../kernels/argmax.ts';
import { attnApplyWorkgroups } from '../kernels/attn_apply.ts';
import { attnDims, attnScoresWorkgroups } from '../kernels/attn_scores.ts';
import {
  attnApplySpec,
  attnScoresSpec,
  cachePackDims,
  cachePackSpec,
  cachePackWorkgroups,
} from '../kernels/cache_variants.ts';
import { EMBED_GATHER, embedDims, embedWorkgroups } from '../kernels/embed_gather.ts';
import {
  embeddingDims,
  embeddingGatherSpec,
  embeddingGatherWorkgroups,
  embeddingHeadSpec,
  embeddingHeadWorkgroups,
  type EmbeddingDType,
} from '../kernels/embedding_variants.ts';
import {
  MATMUL_Q4_DECODE,
  MATMUL_Q4_PREFILL,
  matmulQ4DecodeWorkgroups,
  matmulQ4Dims,
  matmulQ4PrefillWorkgroups,
} from '../kernels/matmul_q4.ts';
import { MATMUL_BIAS, matmulBiasDims, matmulBiasWorkgroups } from '../kernels/matmul_bias.ts';
import { RMSNORM, rmsnormDims, rmsnormWorkgroups } from '../kernels/rmsnorm.ts';
import { ROPE, ropeDims, ropeWorkgroups } from '../kernels/rope.ts';
import { SILU_MUL, countDims, elementwiseWorkgroups } from '../kernels/silu_mul.ts';
import { SOFTMAX, softmaxDims, softmaxWorkgroups } from '../kernels/softmax.ts';
import type { WeightStore } from './weights.ts';
import type { KVCache } from '../runtime/kvcache.ts';

export interface ModelConfig {
  layers: number;
  hidden: number;
  heads: number;
  kvHeads: number;
  headDim: number;
  intermediate: number;
  vocab: number;
  rmsNormEps: number;
  ropeTheta: number;
}

export interface ForwardGraphOptions {
  /**
   * Longest key extent any chunk will attend over. Defaults to `maxSeq`, which is right for a
   * graph that runs whole prompts; a chunked prefill passes the full context.
   */
  keyCapacity?: number;
}

export interface DispatchStep {
  kind: 'dispatch';
  /** The reference's name for the activation this produces. */
  stage: string;
  layer: number | null;
  kernel: ComputeKernel;
  bindGroup: GPUBindGroup;
  uniform: PooledBuffer;
  /**
   * Uniform contents for a chunk: `seq` queries starting at absolute position `begin`.
   *
   * Most steps ignore `begin` — an RMSNorm over 256 rows is the same work wherever those rows
   * sit. The ones that do not are attention, RoPE, and the cache writes, which is exactly the
   * set that has to know where in the sequence it is.
   */
  uniformFor: (seq: number, begin: number) => ArrayBuffer;
  /**
   * Dispatch geometry for a given sequence length, evaluated at encode time.
   *
   * Not baked at build time. The graph is sized to the maximum context, and baking maxSeq
   * geometry meant a 32-token prompt launched the same 5.5 million workgroups a 2048-token one
   * did — every one of them returning immediately from its bounds check, and the launch
   * overhead alone dominating short prompts.
   */
  workgroupsFor: (seq: number, begin: number) => readonly [number, number, number];
}

/** A buffer-to-buffer copy whose offset depends on the sequence length. */
export interface CopyStep {
  kind: 'copy';
  stage: string;
  layer: null;
  from: PooledBuffer;
  to: PooledBuffer;
  /** Source offset in bytes, for a given sequence length. */
  sourceOffset: (seq: number) => number;
  byteLength: number;
}

export type Step = DispatchStep | CopyStep;

/**
 * One input a stage needs in order to run in isolation, and the reference tensor that
 * supplies it.
 *
 * `layout` marks a reference whose axes differ from the engine's. The dump stores q and k
 * after RoPE as [heads, seq, headDim]; the engine uses [seq, heads, headDim], so the values
 * have to be permuted before they are written into a GPU buffer.
 */
export interface StageInput {
  buffer: PooledBuffer;
  /** Name in the reference sidecar. */
  reference: string;
  layout?: 'head-major';
  heads?: number;
}

/**
 * Where an activation lives at the moment it is produced, and what it takes to reproduce it
 * from reference inputs alone.
 *
 * `firstStep..afterStep` is the run of steps that constitutes this stage. It is usually one
 * dispatch; the embedding gather and the tied LM head are five each, and `attn_weights` is
 * two because the scores it softmaxes are not a dumped boundary.
 */
export interface Tap {
  stage: string;
  layer: number | null;
  buffer: PooledBuffer;
  /** First step of this stage. Encoding [firstStep, afterStep] runs the stage alone. */
  firstStep: number;
  /** Index of the step after which the value is valid. */
  afterStep: number;
  elements: (seq: number) => number;
  /** Preload these to run the stage isolated from everything before it. */
  inputs: StageInput[];
}

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

export function tapKey(stage: string, layer: number | null): string {
  return layer === null ? stage : `layer${layer}.${stage}`;
}

export class ForwardGraph {
  readonly config: ModelConfig;
  readonly maxSeq: number;
  readonly keyCapacity: number;
  readonly steps: Step[] = [];
  readonly taps = new Map<string, Tap>();
  readonly tokenIds: PooledBuffer;
  readonly result: PooledBuffer;
  readonly embeddingParts: number;
  /** Which precision the tied table is stored at. Reported in benchmarks. */
  embeddingDType = 'unknown';

  private readonly device: GPUDevice;
  private readonly pool: BufferPool;
  private readonly owned: PooledBuffer[] = [];
  private readonly kernelCache = new Map<string, ComputeKernel>();
  private currentSeq = -1;

  constructor(
    device: GPUDevice,
    pool: BufferPool,
    cache: PipelineCache,
    weights: WeightStore,
    config: ModelConfig,
    maxSeq: number,
    /**
     * Where K and V land. Prefill fills positions 0..seq-1, which is exactly the whole cache
     * at that moment, so the projections write into it directly and attention reads it back —
     * no separate copy, and decode inherits a populated cache with nothing to migrate.
     */
    kv?: KVCache,
    options: ForwardGraphOptions = {},
  ) {
    this.device = device;
    this.pool = pool;
    this.config = config;
    this.maxSeq = maxSeq;
    // How many keys a chunk may have to attend over. Equal to `maxSeq` when the graph runs a
    // whole prompt in one pass, and the full context when it is chunked — the score tensor is
    // `heads x maxSeq x keyCapacity`, so this is the term that decides prefill's memory.
    this.keyCapacity = Math.max(maxSeq, options.keyCapacity ?? maxSeq);
    this.embeddingParts = (weights.embedding.quantParts ?? weights.embedding.parts).length;

    const C = config;
    const kvDim = C.kvHeads * C.headDim;

    const kernelFor = (spec: KernelSpec): ComputeKernel => {
      let existing = this.kernelCache.get(spec.name);
      if (!existing) {
        existing = new ComputeKernel(device, cache, spec);
        this.kernelCache.set(spec.name, existing);
      }
      return existing;
    };

    const scratch = (elements: number, label: string) => {
      const buffer = pool.acquire(Math.max(4, elements * 4), STORAGE, label);
      this.owned.push(buffer);
      return buffer;
    };

    this.tokenIds = scratch(maxSeq, 'token_ids');
    this.result = scratch(4, 'argmax');

    const hiddenA = scratch(maxSeq * C.hidden, 'hidden_a');
    const hiddenB = scratch(maxSeq * C.hidden, 'hidden_b');
    const normed = scratch(maxSeq * C.hidden, 'normed');
    const q = scratch(maxSeq * C.hidden, 'q');
    const k = scratch(maxSeq * kvDim, 'k');
    const v = scratch(maxSeq * kvDim, 'v');
    const qRope = scratch(maxSeq * C.hidden, 'q_rope');
    const kRope = scratch(maxSeq * kvDim, 'k_rope');
    const scores = scratch(C.heads * maxSeq * this.keyCapacity, 'scores');
    const attnWeights = scratch(C.heads * maxSeq * this.keyCapacity, 'attn_weights');
    const attnOut = scratch(maxSeq * C.hidden, 'attn_out');
    const projected = scratch(maxSeq * C.hidden, 'o_proj');
    const gate = scratch(maxSeq * C.intermediate, 'mlp_gate');
    const up = scratch(maxSeq * C.intermediate, 'mlp_up');
    const gated = scratch(maxSeq * C.intermediate, 'mlp_silu_mul');
    const down = scratch(maxSeq * C.hidden, 'mlp_down');
    const lastHidden = scratch(C.hidden, 'last_hidden');
    const logits = scratch(C.vocab, 'logits');
    const zeroBias = scratch(1, 'zero_bias');

    const dispatch = (
      stage: string,
      layer: number | null,
      spec: KernelSpec,
      inputs: readonly (PooledBuffer | GPUBufferBinding)[],
      output: PooledBuffer | GPUBufferBinding,
      uniformFor: (seq: number, begin: number) => ArrayBuffer,
      workgroupsFor: (seq: number, begin: number) => readonly [number, number, number],
    ): void => {
      const kernel = kernelFor(spec);
      const uniform = kernel.uniform(pool);
      this.owned.push(uniform);
      this.steps.push({
        kind: 'dispatch',
        stage,
        layer,
        kernel,
        bindGroup: kernel.bindGroup([...inputs, output, uniform]),
        uniform,
        uniformFor,
        workgroupsFor,
      });
    };

    /**
     * A linear projection, quantized or not.
     *
     * The graph is built once at load, so the fp32/int4 choice and the prefill/decode choice
     * are both resolved here rather than branched on per token. `decodeShaped` selects the
     * matrix-by-vector kernel: it is the right one exactly when m is 1, which for a graph
     * without a KV cache is only the LM head.
     */
    const project = (
      stage: string,
      layer: number | null,
      input: PooledBuffer,
      weightName: string,
      biasName: string | null,
      output: PooledBuffer | GPUBufferBinding,
      n: number,
      k: number,
      mFor: (seq: number) => number,
      workgroupsFor: (m: number, n: number) => readonly [number, number, number],
      decodeShaped = false,
      outStride?: number,
      /** Element offset of the output, per chunk — non-zero only when writing the KV cache. */
      outOffset: (begin: number) => number = () => 0,
    ): void => {
      const bias = biasName ? weights.get(biasName) : zeroBias;
      if (weights.isQuantized) {
        const q = weights.getQuant(weightName);
        const spec = decodeShaped ? MATMUL_Q4_DECODE : MATMUL_Q4_PREFILL;
        dispatch(stage, layer, spec,
          [input, q.packed.binding, q.scales.binding, q.zeros.binding, bias], output,
          (seq, begin) =>
            matmulQ4Dims(mFor(seq), n, k, biasName !== null, q.blockSize, outStride ?? n, outOffset(begin)),
          decodeShaped
            ? () => matmulQ4DecodeWorkgroups(n)
            : (seq) => matmulQ4PrefillWorkgroups(mFor(seq), n));
      } else {
        dispatch(stage, layer, MATMUL_BIAS, [input, weights.get(weightName), bias], output,
          (seq, begin) =>
            matmulBiasDims(mFor(seq), n, k, biasName !== null, outStride ?? n, outOffset(begin)),
          (seq) => workgroupsFor(mFor(seq), n));
      }
    };

    const tap = (
      stage: string,
      layer: number | null,
      buffer: PooledBuffer,
      elements: (seq: number) => number,
      firstStep: number,
      inputs: StageInput[] = [],
    ): void => {
      this.taps.set(tapKey(stage, layer), {
        stage,
        layer,
        buffer,
        firstStep,
        afterStep: this.steps.length - 1,
        elements,
        inputs,
      });
    };

    /** Index the next dispatch will occupy, captured before a stage emits its dispatches. */
    const mark = () => this.steps.length;

    // ---- embeddings: one dispatch per part, disjoint output rows -------
    const embeddingStart = mark();
    const embedParts = weights.embedding.quantParts ?? weights.embedding.parts;
    /** q4 | q8 | f16 | f32, from the loaded tensor's bit width. */
    const embedDType: EmbeddingDType | null =
      weights.embedding.quantParts
        ? ((): EmbeddingDType => {
            switch (weights.embedding.quantParts[0].bits) {
              case 4: return 'q4';
              case 8: return 'q8';
              case 16: return 'f16';
              default: return 'f32';
            }
          })()
        : null;
    this.embeddingDType = embedDType ?? 'f32-unquantized';

    for (const [index, part] of embedParts.entries()) {
      const rowBegin = index * weights.embedding.rowsPerPart;
      const rowCount = part.shape[0];
      if (embedDType && 'packed' in part) {
        const spec = embeddingGatherSpec(embedDType);
        dispatch('embeddings', null, spec,
          [this.tokenIds, part.packed.binding, part.scales.binding, part.zeros.binding], hiddenA,
          (seq) => embeddingDims(seq, C.hidden, rowBegin, rowCount, part.blockSize),
          (seq) => embeddingGatherWorkgroups(spec, seq, C.hidden));
      } else {
        dispatch('embeddings', null, EMBED_GATHER,
          [this.tokenIds, (part as { binding: GPUBufferBinding }).binding], hiddenA,
          (seq) => embedDims(seq, C.hidden, rowBegin, rowCount),
          (seq) => embedWorkgroups(seq, C.hidden));
      }
    }
    // No activation inputs: the gather's only input is the prompt, which the caller writes.
    tap('embeddings', null, hiddenA, (s) => s * C.hidden, embeddingStart);

    let current = hiddenA;
    let spare = hiddenB;

    for (let layer = 0; layer < C.layers; layer++) {
      const w = (suffix: string) => weights.get(`model.layers.${layer}.${suffix}`);
      const wn = (suffix: string) => `model.layers.${layer}.${suffix}`;

      let at = mark();
      dispatch('post_rmsnorm', layer, RMSNORM, [current, w('input_layernorm.weight')], normed,
        (seq) => rmsnormDims(seq, C.hidden, C.rmsNormEps), (seq) => rmsnormWorkgroups(seq));
      tap('post_rmsnorm', layer, normed, (s) => s * C.hidden, at,
        [{ buffer: current, reference: `layer${layer}.input` }]);

      at = mark();
      project('q', layer, normed, wn('self_attn.q_proj.weight'), wn('self_attn.q_proj.bias'),
        q, C.hidden, C.hidden, (seq) => seq, matmulBiasWorkgroups);
      tap('q', layer, q, (s) => s * C.hidden, at,
        [{ buffer: normed, reference: `layer${layer}.post_rmsnorm` }]);

      at = mark();
      project('k', layer, normed, wn('self_attn.k_proj.weight'), wn('self_attn.k_proj.bias'),
        k, kvDim, C.hidden, (seq) => seq, matmulBiasWorkgroups);
      tap('k', layer, k, (s) => s * kvDim, at,
        [{ buffer: normed, reference: `layer${layer}.post_rmsnorm` }]);

      const layerCache = kv?.get(layer);

      at = mark();
      const packSpec = cachePackSpec(kv?.dtype ?? 'f32');
      const f16Cache = kv?.dtype === 'f16';
      project('v', layer, normed, wn('self_attn.v_proj.weight'), wn('self_attn.v_proj.bias'),
        layerCache && !f16Cache ? layerCache.vBinding : v, kvDim, C.hidden,
        (seq) => seq, matmulBiasWorkgroups, false, undefined,
        layerCache && !f16Cache ? (begin) => begin * kvDim : undefined);
      tap('v', layer, v, (s) => s * kvDim, at,
        [{ buffer: normed, reference: `layer${layer}.post_rmsnorm` }]);
      if (layerCache && f16Cache) {
        // Halves share a u32, so the projection cannot write the cache directly.
        dispatch('v_pack', layer, packSpec, [v], layerCache.vBinding,
          (seq, begin) => cachePackDims(seq * kvDim, begin * kvDim),
          (seq) => cachePackWorkgroups('f16', seq * kvDim));
      }

      at = mark();
      dispatch('q_rope', layer, ROPE, [q], qRope,
        (seq, begin) => ropeDims(seq, C.heads, C.headDim, begin, C.ropeTheta),
        (seq) => ropeWorkgroups(seq, C.heads, C.headDim));
      tap('q_rope', layer, qRope, (s) => s * C.hidden, at,
        [{ buffer: q, reference: `layer${layer}.q` }]);

      at = mark();
      dispatch('k_rope', layer, ROPE, [k],
        layerCache && !f16Cache ? layerCache.kBinding : kRope,
        (seq, begin) =>
          ropeDims(seq, C.kvHeads, C.headDim, begin, C.ropeTheta,
            layerCache && !f16Cache ? begin * kvDim : 0),
        (seq) => ropeWorkgroups(seq, C.kvHeads, C.headDim));
      tap('k_rope', layer, kRope, (s) => s * kvDim, at,
        [{ buffer: k, reference: `layer${layer}.k` }]);
      if (layerCache && f16Cache) {
        dispatch('k_pack', layer, packSpec, [kRope], layerCache.kBinding,
          (seq, begin) => cachePackDims(seq * kvDim, begin * kvDim),
          (seq) => cachePackWorkgroups('f16', seq * kvDim));
      }

      at = mark();
      dispatch('scores', layer, attnScoresSpec(kv?.dtype ?? 'f32'),
        [qRope, layerCache ? layerCache.kBinding : kRope], scores,
        (seq, begin) => attnDims(seq, begin + seq, begin, C.heads, C.kvHeads, C.headDim),
        (seq, begin) => attnScoresWorkgroups(seq, begin + seq, C.heads));

      dispatch('attn_weights', layer, SOFTMAX, [scores], attnWeights,
        (seq, begin) => softmaxDims(C.heads * seq, begin + seq),
        (seq) => softmaxWorkgroups(C.heads * seq));
      // Two dispatches: the pre-softmax scores are not a dumped boundary, so isolating the
      // softmax means recomputing them from reference q and k.
      tap('attn_weights', layer, attnWeights, (s) => C.heads * s * s, at, [
        { buffer: qRope, reference: `layer${layer}.q_rope`, layout: 'head-major', heads: C.heads },
        { buffer: kRope, reference: `layer${layer}.k_rope`, layout: 'head-major', heads: C.kvHeads },
      ]);

      at = mark();
      dispatch('attn_out', layer, attnApplySpec(kv?.dtype ?? 'f32'),
        [attnWeights, layerCache ? layerCache.vBinding : v], attnOut,
        (seq, begin) => attnDims(seq, begin + seq, begin, C.heads, C.kvHeads, C.headDim),
        (seq) => attnApplyWorkgroups(seq, C.heads, C.headDim));
      tap('attn_out', layer, attnOut, (s) => s * C.hidden, at, [
        { buffer: attnWeights, reference: `layer${layer}.attn_weights` },
        { buffer: v, reference: `layer${layer}.v` },
      ]);

      at = mark();
      project('o_proj', layer, attnOut, wn('self_attn.o_proj.weight'), null,
        projected, C.hidden, C.hidden, (seq) => seq, matmulBiasWorkgroups);
      tap('o_proj', layer, projected, (s) => s * C.hidden, at,
        [{ buffer: attnOut, reference: `layer${layer}.attn_out` }]);

      at = mark();
      dispatch('resid_attn', layer, ADD, [current, projected], spare,
        (seq) => countDims(seq * C.hidden), (seq) => elementwiseWorkgroups(seq * C.hidden));
      tap('resid_attn', layer, spare, (s) => s * C.hidden, at, [
        { buffer: current, reference: `layer${layer}.input` },
        { buffer: projected, reference: `layer${layer}.o_proj` },
      ]);
      [current, spare] = [spare, current];

      at = mark();
      dispatch('post_attn_rmsnorm', layer, RMSNORM,
        [current, w('post_attention_layernorm.weight')], normed,
        (seq) => rmsnormDims(seq, C.hidden, C.rmsNormEps), (seq) => rmsnormWorkgroups(seq));
      tap('post_attn_rmsnorm', layer, normed, (s) => s * C.hidden, at,
        [{ buffer: current, reference: `layer${layer}.resid_attn` }]);

      at = mark();
      project('mlp_gate', layer, normed, wn('mlp.gate_proj.weight'), null,
        gate, C.intermediate, C.hidden, (seq) => seq, matmulBiasWorkgroups);
      tap('mlp_gate', layer, gate, (s) => s * C.intermediate, at,
        [{ buffer: normed, reference: `layer${layer}.post_attn_rmsnorm` }]);

      at = mark();
      project('mlp_up', layer, normed, wn('mlp.up_proj.weight'), null,
        up, C.intermediate, C.hidden, (seq) => seq, matmulBiasWorkgroups);
      tap('mlp_up', layer, up, (s) => s * C.intermediate, at,
        [{ buffer: normed, reference: `layer${layer}.post_attn_rmsnorm` }]);

      at = mark();
      dispatch('mlp_silu_mul', layer, SILU_MUL, [gate, up], gated,
        (seq) => countDims(seq * C.intermediate),
        (seq) => elementwiseWorkgroups(seq * C.intermediate));
      tap('mlp_silu_mul', layer, gated, (s) => s * C.intermediate, at, [
        { buffer: gate, reference: `layer${layer}.mlp_gate` },
        { buffer: up, reference: `layer${layer}.mlp_up` },
      ]);

      at = mark();
      project('mlp_down', layer, gated, wn('mlp.down_proj.weight'), null,
        down, C.hidden, C.intermediate, (seq) => seq, matmulBiasWorkgroups);
      tap('mlp_down', layer, down, (s) => s * C.hidden, at,
        [{ buffer: gated, reference: `layer${layer}.mlp_silu_mul` }]);

      at = mark();
      dispatch('resid_mlp', layer, ADD, [current, down], spare,
        (seq) => countDims(seq * C.hidden), (seq) => elementwiseWorkgroups(seq * C.hidden));
      tap('resid_mlp', layer, spare, (s) => s * C.hidden, at, [
        { buffer: current, reference: `layer${layer}.resid_attn` },
        { buffer: down, reference: `layer${layer}.mlp_down` },
      ]);
      [current, spare] = [spare, current];
    }

    const finalNormAt = mark();
    this.tailBegin = finalNormAt;
    dispatch('final_norm', null, RMSNORM, [current, weights.get('model.norm.weight')], normed,
      (seq) => rmsnormDims(seq, C.hidden, C.rmsNormEps), (seq) => rmsnormWorkgroups(seq));
    tap('final_norm', null, normed, (s) => s * C.hidden, finalNormAt,
      [{ buffer: current, reference: `layer${C.layers - 1}.resid_mlp` }]);

    // Only the last position is projected to logits — greedy decode reads one row, and
    // producing all of them would cost seq x 151,936 floats per step. The row index moves
    // with the sequence, and a bind group cannot, so it is copied into a fixed buffer whose
    // source offset is chosen at encode time.
    const logitsAt = mark();
    this.steps.push({
      kind: 'copy',
      stage: 'last_hidden',
      layer: null,
      from: normed,
      to: lastHidden,
      sourceOffset: (seq) => (seq - 1) * C.hidden * 4,
      byteLength: C.hidden * 4,
    });

    // ---- tied LM head: one dispatch per part, disjoint output slices ----
    for (const [index, part] of embedParts.entries()) {
      const rowBegin = index * weights.embedding.rowsPerPart;
      const rows = part.shape[0];
      if (embedDType && 'packed' in part) {
        // Decode-shaped: the LM head is always m=1 here, every weight read once, no reuse to
        // amortize an unpack against. This is the kernel the skill describes literally.
        const spec = embeddingHeadSpec(embedDType);
        dispatch('logits', null, spec,
          [lastHidden, part.packed.binding, part.scales.binding, part.zeros.binding], logits,
          () => embeddingDims(1, C.hidden, rowBegin, rows, part.blockSize, C.vocab, rowBegin),
          () => embeddingHeadWorkgroups(rows));
      } else {
        dispatch('logits', null, MATMUL_BIAS,
          [lastHidden, (part as { binding: GPUBufferBinding }).binding, zeroBias], logits,
          () => matmulBiasDims(1, rows, C.hidden, false, C.vocab, rowBegin),
          () => matmulBiasWorkgroups(1, rows));
      }
    }
    // Isolation starts at the copy step: the reference final norm is written into `normed`
    // and the copy selects row seq-1 from it, exactly as the real path does.
    tap('logits', null, logits, () => C.vocab, logitsAt,
      [{ buffer: normed, reference: 'final_norm' }]);

    dispatch('argmax', null, ARGMAX, [logits], this.result,
      () => argmaxDims(C.vocab, 0), () => ARGMAX_WORKGROUPS);

    if (embedParts.length === 0) {
      throw new Error('weight store has no embedding parts');
    }
  }

  /**
   * Rewrite every uniform for a new sequence length.
   *
   * Not just a bounds update: `seq` is the stride of the attention score buffer, so a step
   * that ran at the wrong length reads a differently-shaped tensor rather than an
   * out-of-range one.
   */
  setSequenceLength(seq: number, begin = 0): void {
    if (seq === this.currentSeq && begin === this.currentBegin) return;
    if (seq < 1 || seq > this.maxSeq) {
      throw new RangeError(`sequence length ${seq} outside [1, ${this.maxSeq}]`);
    }
    for (const step of this.steps) {
      if (step.kind !== 'dispatch') continue;
      this.device.queue.writeBuffer(step.uniform.buffer, 0, step.uniformFor(seq, begin));
    }
    this.currentSeq = seq;
    this.currentBegin = begin;
  }

  private currentBegin = -1;

  /** Index of the first step of the tail — final norm, LM head, argmax. */
  private tailBegin = 0;

  /**
   * Run a prompt of any length through the graph, in chunks of at most `maxSeq`.
   *
   * One submit per chunk, because uniforms are written through the queue rather than recorded
   * into the encoder: a single encoder covering every chunk would see only the last chunk's
   * uniforms. Prefill is hundreds of milliseconds, so a handful of extra submits is free.
   *
   * The tail — final norm, LM head, argmax — runs only on the last chunk. It is 25% of the
   * per-token GPU cost at decode shapes and produces logits nobody reads for an intermediate
   * chunk.
   */
  encodeChunks(device: GPUDevice, queue: GPUQueue, length: number, writeTokens: (begin: number, count: number) => void): void {
    if (length < 1) throw new RangeError(`prompt length ${length} must be at least 1`);
    for (let begin = 0; begin < length; begin += this.maxSeq) {
      const count = Math.min(this.maxSeq, length - begin);
      const last = begin + count >= length;
      writeTokens(begin, count);
      this.setSequenceLength(count, begin);
      const encoder = device.createCommandEncoder({ label: `prefill/chunk@${begin}` });
      this.encode(encoder, count, last ? this.steps.length - 1 : this.tailBegin - 1, 0, begin);
      queue.submit([encoder.finish()]);
    }
  }

  /**
   * Record the forward pass, or a slice of it.
   *
   * `fromStep` exists for the parity harness: running a single stage against reference inputs
   * is the only way to attribute an error to the kernel that produced it rather than to the
   * three hundred dispatches before it.
   */
  encode(
    encoder: GPUCommandEncoder,
    seq: number,
    upToStep = this.steps.length - 1,
    fromStep = 0,
    begin = 0,
  ): void {
    if (this.currentSeq !== seq || this.currentBegin !== begin) {
      throw new Error('setSequenceLength must be called before encode');
    }
    let pass: GPUComputePassEncoder | null = null;
    const endPass = () => {
      if (pass) {
        pass.end();
        pass = null;
      }
    };

    for (let i = fromStep; i <= upToStep && i < this.steps.length; i++) {
      const step = this.steps[i];
      if (step.kind === 'copy') {
        // A copy cannot be recorded inside a compute pass.
        endPass();
        encoder.copyBufferToBuffer(
          step.from.buffer,
          step.sourceOffset(seq),
          step.to.buffer,
          0,
          step.byteLength,
        );
        continue;
      }
      if (!pass) pass = encoder.beginComputePass({ label: 'forward' });
      step.kernel.encode(pass, step.bindGroup, step.workgroupsFor(seq, begin));
    }
    endPass();
  }

  get dispatchCount(): number {
    return this.steps.filter((s) => s.kind === 'dispatch').length;
  }

  destroy(): void {
    for (const buffer of this.owned) this.pool.release(buffer);
    this.owned.length = 0;
    this.steps.length = 0;
    this.taps.clear();
  }
}
