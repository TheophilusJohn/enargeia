/**
 * The decode graph: one token, appended to a populated KV cache.
 *
 * A separate file from the prefill graph rather than a mode flag on it, because the kernels
 * skill is explicit that prefill and decode want different kernels and that branching between
 * them inside one is how a kernel ends up tuned for neither. Almost every dispatch here is
 * m=1, which is the regime `matmul_q4_decode` was written for and has never actually run in.
 *
 * What differs from prefill, concretely:
 *
 *   - every projection is matrix-by-vector, so the decode matmul replaces the tiled one
 *   - K and V for the new token are written *into the cache* by the projection itself, using
 *     the kernel's existing `outOffset`; there is no append kernel and no copy
 *   - RoPE runs at `positionOffset = position`, and its output offset puts the rotated key
 *     straight into the cache slot
 *   - attention reads the whole cached history for one query row
 *   - the score buffer is preallocated to the maximum context, so its row stride and its live
 *     row length are different numbers and the softmax takes both
 *
 * The uniforms that carry `position` are rewritten every step. That is the only per-token
 * JavaScript: bind groups, pipelines and buffers outlive the whole generation.
 */

import type { PipelineCache } from '../gpu/pipeline.ts';
import type { BufferPool, PooledBuffer } from '../gpu/pool.ts';
import { ComputeKernel, type KernelSpec } from '../kernels/kernel.ts';
import { ADD } from '../kernels/add.ts';
import {
  attnApplyDecodeWorkgroups,
  attnDecodeDims,
  attnScoresDecodeWorkgroups,
} from '../kernels/attn_decode.ts';
import {
  attnApplyDecodeParallelSpec,
  attnApplyDecodeParallelWorkgroups,
  attnApplyDecodeSpec,
  attnScoresDecodeSpec,
  cachePackDims,
  cachePackSpec,
  cachePackWorkgroups,
} from '../kernels/cache_variants.ts';
import { MATMUL_Q4_DECODE, matmulQ4DecodeWorkgroups, matmulQ4Dims } from '../kernels/matmul_q4.ts';
import { MATMUL_BIAS, matmulBiasDims, matmulBiasWorkgroups } from '../kernels/matmul_bias.ts';
import { EMBED_GATHER, embedDims, embedWorkgroups } from '../kernels/embed_gather.ts';
import { RMSNORM, rmsnormDims, rmsnormWorkgroups } from '../kernels/rmsnorm.ts';
import { ROPE, ropeDims, ropeWorkgroups } from '../kernels/rope.ts';
import { SILU_MUL, countDims, elementwiseWorkgroups } from '../kernels/silu_mul.ts';
import { SOFTMAX, softmaxDims, softmaxWorkgroups } from '../kernels/softmax.ts';
import { SAMPLE, SAMPLE_WORKGROUPS, sampleDims, type SamplingParams } from '../kernels/sample.ts';
import {
  embeddingDims,
  embeddingGatherSpec,
  embeddingGatherWorkgroups,
  embeddingHeadSpec,
  embeddingHeadWorkgroups,
  type EmbeddingDType,
} from '../kernels/embedding_variants.ts';
import type { ModelConfig } from './graph.ts';
import type { WeightStore } from './weights.ts';
import type { KVCache } from '../runtime/kvcache.ts';
import { OPTIMIZATIONS } from '../runtime/options.ts';

interface Step {
  stage: string;
  kernel: ComputeKernel;
  bindGroup: GPUBindGroup;
  /** Byte offset of this step's uniform inside the shared block. */
  uniformOffset: number;
  /** Uniform contents for a given position and history length. */
  uniformFor: (position: number, historyLength: number, random: number) => ArrayBuffer;
  /**
   * Dispatch geometry for a given position, evaluated at encode time.
   *
   * Almost every decode dispatch is genuinely fixed — one token, `hidden` wide — but the
   * attention scores span the history, which grows. Sizing that from `maxContext` launched
   * 448 workgroups per layer at every position when 28 were needed at position 100.
   */
  workgroupsFor: (position: number) => readonly [number, number, number];
}

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

/**
 * Diagnostic only. 1 is the shipping path; higher values hold bytes constant while cutting
 * iteration count, to test whether the position-dependent cost is iterations or occupancy.
 */
const PROBE_UNROLL = Number((import.meta as { env?: Record<string, string> }).env?.VITE_PROBE_UNROLL ?? 1);

export class DecodeGraph {
  readonly steps: Step[] = [];
  /** One token id, written before each step. */
  readonly tokenId: PooledBuffer;
  /** Every token so far, for the repetition penalty. */
  readonly history: PooledBuffer;
  /** The sampled id. Four bytes, and the only thing read back per token. */
  readonly result: PooledBuffer;
  readonly logits: PooledBuffer;
  /** Exposed so the inspector can sample it; nothing in the decode path reads it back. */
  attentionWeights: PooledBuffer | null = null;

  private readonly device: GPUDevice;
  private readonly pool: BufferPool;
  private readonly owned: PooledBuffer[] = [];
  private readonly kernelCache = new Map<string, ComputeKernel>();
  private params: SamplingParams;
  /**
   * One buffer holding every step's uniform, at 256-byte stride.
   *
   * Decode used to issue ~410 `writeBuffer` calls per token, one per dispatch, for a few bytes
   * each. Uniform bindings can carry a static offset, so the whole set lives in one allocation
   * and one write replaces all of them — the bind groups never change, only the bytes.
   */
  private readonly uniformBlock: PooledBuffer;
  private readonly uniformStaging: ArrayBuffer;
  private readonly uniformStride: number;

  constructor(
    device: GPUDevice,
    pool: BufferPool,
    cache: PipelineCache,
    weights: WeightStore,
    config: ModelConfig,
    kv: KVCache,
    params: SamplingParams,
  ) {
    this.device = device;
    this.pool = pool;
    this.params = params;

    const C = config;
    const kvDim = C.kvHeads * C.headDim;
    const maxContext = kv.maxContext;

    const kernelFor = (spec: KernelSpec): ComputeKernel => {
      let existing = this.kernelCache.get(spec.name);
      if (!existing) {
        existing = new ComputeKernel(device, cache, spec);
        this.kernelCache.set(spec.name, existing);
      }
      return existing;
    };

    const scratch = (elements: number, label: string) => {
      const buffer = pool.acquire(Math.max(4, elements * 4), STORAGE, `decode/${label}`);
      this.owned.push(buffer);
      return buffer;
    };

    this.tokenId = scratch(1, 'token_id');
    this.history = scratch(maxContext, 'history');
    this.result = scratch(4, 'result');
    this.logits = scratch(C.vocab, 'logits');

    const hiddenA = scratch(C.hidden, 'hidden_a');
    const hiddenB = scratch(C.hidden, 'hidden_b');
    const normed = scratch(C.hidden, 'normed');
    const q = scratch(C.hidden, 'q');
    const kScratch = scratch(kvDim, 'k');
    const kRopeScratch = scratch(kvDim, 'k_rope');
    const vScratch = scratch(kvDim, 'v');
    const qRope = scratch(C.hidden, 'q_rope');
    const scores = scratch(C.heads * maxContext, 'scores');
    const attnWeights = scratch(C.heads * maxContext, 'attn_weights');
    this.attentionWeights = attnWeights;
    const attnOut = scratch(C.hidden, 'attn_out');
    const projected = scratch(C.hidden, 'o_proj');
    const gate = scratch(C.intermediate, 'mlp_gate');
    const up = scratch(C.intermediate, 'mlp_up');
    const gated = scratch(C.intermediate, 'mlp_silu_mul');
    const down = scratch(C.hidden, 'mlp_down');
    const zeroBias = scratch(1, 'zero_bias');

    // Bind group creation needs the uniform block to exist first, but its size depends on the
    // number of steps — which is known from the config without building them.
    this.uniformStride = Math.max(256, device.limits.minUniformBufferOffsetAlignment);
    const stepBudget = 64 + C.layers * 22;
    this.uniformBlock = pool.acquire(
      stepBudget * this.uniformStride,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      'decode/uniforms',
    );
    this.owned.push(this.uniformBlock);
    this.uniformStaging = new ArrayBuffer(stepBudget * this.uniformStride);

    const dispatch = (
      stage: string,
      spec: KernelSpec,
      inputs: readonly (PooledBuffer | GPUBufferBinding)[],
      output: PooledBuffer | GPUBufferBinding,
      uniformFor: Step['uniformFor'],
      workgroupsFor: Step['workgroupsFor'] | readonly [number, number, number],
    ): void => {
      const kernel = kernelFor(spec);
      const uniformOffset = this.steps.length * this.uniformStride;
      if (uniformOffset + this.uniformStride > this.uniformBlock.size) {
        throw new Error(`decode graph exceeded its uniform budget of ${stepBudget} steps`);
      }
      this.steps.push({
        stage,
        kernel,
        bindGroup: kernel.bindGroup([
          ...inputs,
          output,
          { buffer: this.uniformBlock.buffer, offset: uniformOffset, size: spec.uniformBytes },
        ]),
        uniformOffset,
        uniformFor,
        workgroupsFor:
          typeof workgroupsFor === 'function' ? workgroupsFor : () => workgroupsFor,
      });
    };

    /** A matrix-by-vector projection: the decode regime for every weight in the model. */
    const project = (
      stage: string,
      input: PooledBuffer,
      weightName: string,
      biasName: string | null,
      output: PooledBuffer | GPUBufferBinding,
      n: number,
      k: number,
      outOffsetFor: (position: number) => number = () => 0,
    ): void => {
      const bias = biasName ? weights.get(biasName) : zeroBias;
      if (weights.isQuantized) {
        const w = weights.getQuant(weightName);
        dispatch(stage, MATMUL_Q4_DECODE,
          [input, w.packed.binding, w.scales.binding, w.zeros.binding, bias], output,
          (position) => matmulQ4Dims(1, n, k, biasName !== null, w.blockSize, n, outOffsetFor(position)),
          matmulQ4DecodeWorkgroups(n));
      } else {
        dispatch(stage, MATMUL_BIAS, [input, weights.get(weightName), bias], output,
          (position) => matmulBiasDims(1, n, k, biasName !== null, n, outOffsetFor(position)),
          matmulBiasWorkgroups(1, n));
      }
    };

    // ---- embedding, one token ------------------------------------------
    const embedParts = weights.embedding.quantParts ?? weights.embedding.parts;
    const embedDType: EmbeddingDType | null = weights.embedding.quantParts
      ? (({ 4: 'q4', 8: 'q8', 16: 'f16' } as Record<number, EmbeddingDType>)[
          weights.embedding.quantParts[0].bits
        ] ?? 'f32')
      : null;

    for (const [index, part] of embedParts.entries()) {
      const rowBegin = index * weights.embedding.rowsPerPart;
      const rowCount = part.shape[0];
      if (embedDType && 'packed' in part) {
        const spec = embeddingGatherSpec(embedDType);
        dispatch('embeddings', spec,
          [this.tokenId, part.packed.binding, part.scales.binding, part.zeros.binding], hiddenA,
          () => embeddingDims(1, C.hidden, rowBegin, rowCount, part.blockSize),
          embeddingGatherWorkgroups(spec, 1, C.hidden));
      } else {
        // The unquantized path, so decode can be measured against fp32 rather than only
        // against itself.
        dispatch('embeddings', EMBED_GATHER,
          [this.tokenId, (part as { binding: GPUBufferBinding }).binding], hiddenA,
          () => embedDims(1, C.hidden, rowBegin, rowCount),
          embedWorkgroups(1, C.hidden));
      }
    }

    let current = hiddenA;
    let spare = hiddenB;

    for (let layer = 0; layer < C.layers; layer++) {
      const wn = (suffix: string) => `model.layers.${layer}.${suffix}`;
      const layerCache = kv.get(layer);
      const packSpec = cachePackSpec(kv.dtype);

      dispatch('post_rmsnorm', RMSNORM, [current, weights.get(wn('input_layernorm.weight'))], normed,
        () => rmsnormDims(1, C.hidden, C.rmsNormEps), rmsnormWorkgroups(1));

      project('q', normed, wn('self_attn.q_proj.weight'), wn('self_attn.q_proj.bias'), q, C.hidden, C.hidden);
      project('k', normed, wn('self_attn.k_proj.weight'), wn('self_attn.k_proj.bias'), kScratch, kvDim, C.hidden);
      // V needs no rotation. With an f32 cache the projection writes its slot directly; with
      // f16 it goes via the pack kernel, because two adjacent halves share a u32 and the
      // threads that compute them are different threads.
      if (kv.dtype === 'f32') {
        project('v', normed, wn('self_attn.v_proj.weight'), wn('self_attn.v_proj.bias'),
          layerCache.vBinding, kvDim, C.hidden, (position) => position * kv.stride);
      } else {
        project('v', normed, wn('self_attn.v_proj.weight'), wn('self_attn.v_proj.bias'),
          vScratch, kvDim, C.hidden);
        dispatch('v_pack', packSpec, [vScratch], layerCache.vBinding,
          (position) => cachePackDims(kvDim, position * kv.stride),
          cachePackWorkgroups(kv.dtype, kvDim));
      }

      dispatch('q_rope', ROPE, [q], qRope,
        (position) => ropeDims(1, C.heads, C.headDim, position, C.ropeTheta),
        ropeWorkgroups(1, C.heads, C.headDim));

      // K is cached *after* rotation: a key's rotation depends only on its own position and
      // never changes, so rotating once on the way in is the whole saving.
      if (kv.dtype === 'f32') {
        dispatch('k_rope', ROPE, [kScratch], layerCache.kBinding,
          (position) => ropeDims(1, C.kvHeads, C.headDim, position, C.ropeTheta, position * kv.stride),
          ropeWorkgroups(1, C.kvHeads, C.headDim));
      } else {
        dispatch('k_rope', ROPE, [kScratch], kRopeScratch,
          () => ropeDims(1, C.kvHeads, C.headDim, 0, C.ropeTheta),
          ropeWorkgroups(1, C.kvHeads, C.headDim));
        // The rotation still depends on the absolute position; only the destination moved.
        this.steps[this.steps.length - 1].uniformFor = (position) =>
          ropeDims(1, C.kvHeads, C.headDim, position, C.ropeTheta);
        dispatch('k_pack', packSpec, [kRopeScratch], layerCache.kBinding,
          (position) => cachePackDims(kvDim, position * kv.stride),
          cachePackWorkgroups(kv.dtype, kvDim));
      }

      dispatch('scores', attnScoresDecodeSpec(kv.dtype), [qRope, layerCache.kBinding], scores,
        (position) => attnDecodeDims(position, C.heads, C.kvHeads, C.headDim, maxContext, kv.stride),
        // The live history is position+1 keys, not maxContext.
        (position) => attnScoresDecodeWorkgroups(position + 1, C.heads));

      // Row length grows each step while the buffer stays at maxContext, so the softmax needs
      // the live length and the stride as separate numbers.
      dispatch('attn_weights', SOFTMAX, [scores], attnWeights,
        (position) => softmaxDims(C.heads, position + 1, maxContext),
        softmaxWorkgroups(C.heads));

      if (OPTIMIZATIONS.parallelAttention) {
        dispatch('attn_out', attnApplyDecodeParallelSpec(kv.dtype),
          [attnWeights, layerCache.vBinding], attnOut,
          (position) => attnDecodeDims(position, C.heads, C.kvHeads, C.headDim, maxContext, kv.stride),
          attnApplyDecodeParallelWorkgroups(C.heads, C.headDim));
      } else {
        dispatch('attn_out', attnApplyDecodeSpec(kv.dtype, PROBE_UNROLL),
          [attnWeights, layerCache.vBinding], attnOut,
          (position) => attnDecodeDims(position, C.heads, C.kvHeads, C.headDim, maxContext, kv.stride),
          attnApplyDecodeWorkgroups(C.heads, C.headDim));
      }

      project('o_proj', attnOut, wn('self_attn.o_proj.weight'), null, projected, C.hidden, C.hidden);

      dispatch('resid_attn', ADD, [current, projected], spare,
        () => countDims(C.hidden), elementwiseWorkgroups(C.hidden));
      [current, spare] = [spare, current];

      dispatch('post_attn_rmsnorm', RMSNORM,
        [current, weights.get(wn('post_attention_layernorm.weight'))], normed,
        () => rmsnormDims(1, C.hidden, C.rmsNormEps), rmsnormWorkgroups(1));

      project('mlp_gate', normed, wn('mlp.gate_proj.weight'), null, gate, C.intermediate, C.hidden);
      project('mlp_up', normed, wn('mlp.up_proj.weight'), null, up, C.intermediate, C.hidden);

      dispatch('mlp_silu_mul', SILU_MUL, [gate, up], gated,
        () => countDims(C.intermediate), elementwiseWorkgroups(C.intermediate));

      project('mlp_down', gated, wn('mlp.down_proj.weight'), null, down, C.hidden, C.intermediate);

      dispatch('resid_mlp', ADD, [current, down], spare,
        () => countDims(C.hidden), elementwiseWorkgroups(C.hidden));
      [current, spare] = [spare, current];
    }

    dispatch('final_norm', RMSNORM, [current, weights.get('model.norm.weight')], normed,
      () => rmsnormDims(1, C.hidden, C.rmsNormEps), rmsnormWorkgroups(1));

    for (const [index, part] of embedParts.entries()) {
      const rowBegin = index * weights.embedding.rowsPerPart;
      const rows = part.shape[0];
      if (embedDType && 'packed' in part) {
        const spec = embeddingHeadSpec(embedDType);
        dispatch('logits', spec,
          [normed, part.packed.binding, part.scales.binding, part.zeros.binding], this.logits,
          () => embeddingDims(1, C.hidden, rowBegin, rows, part.blockSize, C.vocab, rowBegin),
          embeddingHeadWorkgroups(rows));
      } else {
        dispatch('logits', MATMUL_BIAS,
          [normed, (part as { binding: GPUBufferBinding }).binding, zeroBias], this.logits,
          () => matmulBiasDims(1, rows, C.hidden, false, C.vocab, rowBegin),
          matmulBiasWorkgroups(1, rows));
      }
    }

    // Temperature, top-p, repetition penalty and the draw, all on the GPU. Four bytes come
    // back; the 608 KB of logits never leave the device.
    dispatch('sample', SAMPLE, [this.logits, this.history], this.result,
      (_position, historyLength, random) =>
        sampleDims(C.vocab, historyLength, this.params, random),
      SAMPLE_WORKGROUPS);
  }

  setSampling(params: SamplingParams): void {
    this.params = params;
  }

  /**
   * Rewrite every uniform for a new position. The only per-token JavaScript.
   *
   * Assembled into one staging array and written once. The previous version issued one
   * `writeBuffer` per dispatch — around 410 calls per token, each for 16 to 48 bytes.
   */
  setPosition(position: number, historyLength: number, random: number): void {
    const staging = new Uint8Array(this.uniformStaging);
    for (const step of this.steps) {
      const bytes = new Uint8Array(step.uniformFor(position, historyLength, random));
      staging.set(bytes, step.uniformOffset);
    }
    const used = this.steps.length * this.uniformStride;
    this.device.queue.writeBuffer(this.uniformBlock.buffer, 0, this.uniformStaging, 0, used);
  }

  /**
   * Record every step in its own compute pass with timestamps, for the inspector's per-kernel
   * breakdown.
   *
   * ~410 passes instead of one, so this is materially more expensive than a normal step and is
   * run on a duty cycle rather than every token. Step order is preserved exactly — grouping by
   * kernel name would reorder dependent dispatches and produce a fast, wrong answer.
   */
  encodeProfiled(encoder: GPUCommandEncoder, position: number, querySet: GPUQuerySet): number {
    let query = 0;
    for (const step of this.steps) {
      if (query + 2 > this.maxQueries) break;
      const pass = encoder.beginComputePass({
        label: `profile/${step.stage}`,
        timestampWrites: {
          querySet,
          beginningOfPassWriteIndex: query,
          endOfPassWriteIndex: query + 1,
        },
      });
      step.kernel.encode(pass, step.bindGroup, step.workgroupsFor(position));
      pass.end();
      query += 2;
    }
    return query / 2;
  }

  /** Kernel name of each step, in dispatch order, for attributing timestamps. */
  get stepKernels(): string[] {
    return this.steps.map((step) => step.kernel.spec.name);
  }

  readonly maxQueries = 4096;

  encode(encoder: GPUCommandEncoder, position: number): void {
    const pass = encoder.beginComputePass({ label: 'decode' });
    for (const step of this.steps) {
      step.kernel.encode(pass, step.bindGroup, step.workgroupsFor(position));
    }
    pass.end();
  }

  get dispatchCount(): number {
    return this.steps.length;
  }

  destroy(): void {
    for (const buffer of this.owned) this.pool.release(buffer);
    this.owned.length = 0;
    this.steps.length = 0;
  }
}
