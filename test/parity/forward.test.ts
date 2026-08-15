/**
 * Layer-by-layer parity against the PyTorch dump.
 *
 * Every activation boundary in the graph has a tap. For each one, the graph is encoded up to
 * that step and the buffer read back — scratch buffers are reused across layers, so a single
 * forward pass followed by 411 reads would return the last layer's values 411 times. One
 * partial encode per tap is quadratic and cheap enough at 15 tokens.
 *
 * Read the table from the top. The first stage over threshold is where the bug is; everything
 * below it is downstream contamination.
 *
 *   npm run parity
 *   npm run parity -- --layer 7
 *   npm run parity -- --strict
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { BufferPool, PipelineCache, initGPU, readFloats, type GPUContext } from '../../src/gpu/index.ts';
import { HttpRangeSource } from '../../src/model/safetensors.ts';
import { WeightStore } from '../../src/model/weights.ts';
import { ForwardGraph, tapKey, type ModelConfig } from '../../src/model/graph.ts';
import { Generator } from '../../src/runtime/generate.ts';
import { FP32_TOLERANCE, Q4_TOLERANCE, compareArrays, withinTolerance, type ErrorReport } from '../reference/matmul.ts';

const FP32_WEIGHTS_URL = '/test/fixtures/model.safetensors';
const Q4_WEIGHTS_URL = '/models/qwen2.5-0.5b-q4.enargeia';
const REFERENCE_URL = '/test/fixtures/reference.bin';
const SIDECAR_URL = '/test/fixtures/reference.json';

interface RefTensor {
  name: string;
  shape: number[];
  offset: number;
  byteLength: number;
  layer: number | null;
  stage: string;
}
interface Sidecar {
  seq: number;
  promptTokens: number[];
  greedy: number[];
  greedyText: string;
  config: ModelConfig;
  tensors: RefTensor[];
}

const onlyLayer = readEnvNumber('VITE_PARITY_LAYER');
const strict = readEnvFlag('VITE_PARITY_STRICT');
const decodeTokens = readEnvNumber('VITE_PARITY_TOKENS') ?? 20;
/**
 * Force the 128 MiB binding limit so the embedding table splits into five parts and the
 * per-part gather and per-part LM head are the code paths under test. Without this the run
 * uses this adapter's 4096 MiB and never exercises the split at all.
 */
const clampBytes = readEnvFlag('VITE_PARITY_CLAMP') ? 128 * 1024 * 1024 : undefined;
/** `npm run parity -- --q4` loads the int4 container and applies the int4 thresholds. */
const useQ4 = readEnvFlag('VITE_PARITY_Q4');
const TOLERANCE = useQ4 ? Q4_TOLERANCE : FP32_TOLERANCE;
const WEIGHTS_URL = useQ4 ? Q4_WEIGHTS_URL : FP32_WEIGHTS_URL;

function readEnvNumber(name: string): number | null {
  const value = (import.meta.env?.[name] ?? null) as string | null;
  return value === null || value === '' ? null : Number(value);
}
function readEnvFlag(name: string): boolean {
  return Boolean(import.meta.env?.[name]);
}

let ctx: GPUContext;
let pool: BufferPool;
let cache: PipelineCache;
let weights: WeightStore;
let graph: ForwardGraph;
let sidecar: Sidecar;
let referenceBytes: ArrayBuffer;
interface Row {
  stage: string;
  /** Stage run from reference inputs — its own error, and the column that gates. */
  isolated: { abs: number; rel: number; ratio: number; ok: boolean };
  /** Stage run from the prompt through every preceding dispatch. Reported, not asserted. */
  accumulated: { abs: number; rel: number; ratio: number };
}
const rows: Row[] = [];

beforeAll(async () => {
  ctx = await initGPU({ label: 'parity' });
  pool = new BufferPool(ctx.device, { label: 'parity', maxIdleBytes: 64 * 1024 * 1024 });
  cache = new PipelineCache(ctx.device);

  sidecar = (await (await fetch(SIDECAR_URL)).json()) as Sidecar;
  referenceBytes = await (await fetch(REFERENCE_URL)).arrayBuffer();

  const load = useQ4 ? WeightStore.loadQuantized : WeightStore.load;
  weights = await load(ctx.device, ctx.profile, {
    ref: { modelId: 'Qwen/Qwen2.5-0.5B-Instruct', revision: 'parity', file: 'model.safetensors' },
    source: new HttpRangeSource(WEIGHTS_URL),
    noCache: true,
    maxBindingBytes: clampBytes,
  });

  graph = new ForwardGraph(
    ctx.device, pool, cache, weights, sidecar.config,
    sidecar.seq + decodeTokens,
  );
}, 600_000);

afterAll(() => {
  if (rows.length > 0) printTable();
  graph?.destroy();
  weights?.destroy();
  pool?.destroy();
  ctx?.device.destroy();
});

function reference(name: string): Float32Array {
  const info = sidecar.tensors.find((t) => t.name === name);
  if (!info) throw new Error(`reference tensor "${name}" not found`);
  return new Float32Array(referenceBytes.slice(info.offset, info.offset + info.byteLength));
}

/**
 * The reference stores q/k after RoPE as [heads, seq, headDim]; the engine uses
 * [seq, heads, headDim]. Reshaping the reference is safer than reshaping the engine output,
 * because the engine's layout is the one the next kernel actually consumes.
 */
function toSeqMajor(x: Float32Array, seq: number, heads: number, dim: number): Float32Array {
  const out = new Float32Array(x.length);
  for (let h = 0; h < heads; h++) {
    for (let s = 0; s < seq; s++) {
      for (let d = 0; d < dim; d++) {
        out[(s * heads + h) * dim + d] = x[(h * seq + s) * dim + d];
      }
    }
  }
  return out;
}

/**
 * Read one stage's output.
 *
 * `isolated` runs only that stage, after writing the reference values for each of its inputs
 * into the buffers it reads. That is the number that says whether *this* kernel is right.
 * Without it the harness reports the drift of every dispatch before the stage, which by layer
 * 20 is 300 of them, and "the first red row is the bug" stops being true.
 *
 * `accumulated` runs the whole prefix from the prompt, which is what the engine actually does
 * and therefore what the fp32 path's real error looks like.
 */
async function readTap(
  stage: string,
  layer: number | null,
  seq: number,
  mode: 'isolated' | 'accumulated',
): Promise<Float32Array> {
  const tap = graph.taps.get(tapKey(stage, layer));
  if (!tap) throw new Error(`no tap for ${tapKey(stage, layer)}`);
  // The prompt has to be in place before anything is encoded: the gather reads it, and a
  // zeroed id buffer produces a forward pass over token 0 repeated, which is a perfectly
  // valid forward pass of the wrong thing.
  ctx.queue.writeBuffer(graph.tokenIds.buffer, 0, new Uint32Array(sidecar.promptTokens));
  graph.setSequenceLength(seq);

  if (mode === 'isolated') {
    for (const input of tap.inputs) {
      let values = reference(input.reference);
      if (input.layout === 'head-major') {
        values = toSeqMajor(values, seq, input.heads!, sidecar.config.headDim);
      }
      ctx.queue.writeBuffer(input.buffer.buffer, 0, values);
    }
  }

  const encoder = ctx.device.createCommandEncoder({ label: `parity:${stage}` });
  graph.encode(encoder, seq, tap.afterStep, mode === 'isolated' ? tap.firstStep : 0);
  ctx.queue.submit([encoder.finish()]);

  const elements = tap.elements(seq);
  const staging = pool.acquire(
    elements * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    'parity.stage',
  );
  const copy = ctx.device.createCommandEncoder();
  copy.copyBufferToBuffer(tap.buffer.buffer, 0, staging.buffer, 0, elements * 4);
  ctx.queue.submit([copy.finish()]);
  const values = await readFloats(ctx, pool, staging);
  pool.release(staging);
  return values;
}

function record(stage: string, isolated: ErrorReport, accumulated: ErrorReport): boolean {
  const ok = withinTolerance(isolated);
  rows.push({
    stage,
    isolated: { abs: isolated.maxAbs, rel: isolated.maxRel, ratio: isolated.worstRatio, ok },
    accumulated: {
      abs: accumulated.maxAbs,
      rel: accumulated.maxRel,
      ratio: accumulated.worstRatio,
    },
  });
  return ok;
}

/** Both columns for one stage, with any layout fix applied to the reference. */
async function compareStage(
  stage: string,
  layer: number | null,
  seq: number,
  reshape: (x: Float32Array) => Float32Array = (x) => x,
  referenceName = tapKey(stage, layer),
): Promise<boolean> {
  const expected = reshape(reference(referenceName));
  const isolated = compareArrays(await readTap(stage, layer, seq, 'isolated'), expected, TOLERANCE);
  const accumulated = compareArrays(await readTap(stage, layer, seq, 'accumulated'), expected, TOLERANCE);
  return record(tapKey(stage, layer), isolated, accumulated);
}

function printTable(): void {
  const lines = [
    '',
    ' '.repeat(32) + '   ISOLATED (gates)        ACCUMULATED (reported)',
    ' stage'.padEnd(32) + 'max abs'.padStart(11) + '  of tol' + 'max abs'.padStart(13) + '  of tol',
    '-'.repeat(78),
  ];
  for (const row of rows) {
    lines.push(
      ` ${row.stage}`.padEnd(32) +
        row.isolated.abs.toExponential(2).padStart(11) +
        `${(row.isolated.ratio * 100).toFixed(1)}%`.padStart(8) +
        (row.isolated.ok ? '   ' : ' ! ') +
        row.accumulated.abs.toExponential(2).padStart(11) +
        `${(row.accumulated.ratio * 100).toFixed(1)}%`.padStart(9),
    );
  }
  const failed = rows.filter((r) => !r.isolated.ok);
  const drifted = rows.filter((r) => r.accumulated.ratio > 1);
  lines.push('-'.repeat(78));
  if (useQ4) {
    // Under int4 the isolated column measures something different, and saying otherwise
    // would be the harness lying. Feeding a stage reference-quality *inputs* does not remove
    // the quantization of its *weights* — the weights are the thing that changed. So this
    // column is per-stage quantization loss, which is a useful diagnostic and not a bug
    // detector. What gates int4 is that each kernel reproduces the CPU dequantized reference
    // exactly (test/kernels/matmul_q4.test.ts) and that perplexity holds (npm run quality).
    lines.push(
      ` ISOLATED: ${failed.length} of ${rows.length} over the int4 threshold` +
        (failed.length ? ` from ${failed[0].stage} onward — per-stage quantization loss` : ''),
    );
  } else {
    lines.push(
      failed.length === 0
        ? ` ISOLATED: all ${rows.length} stages within tolerance`
        : ` ISOLATED: first failing stage ${failed[0].stage} (${failed.length} of ${rows.length}) <- REAL BUG`,
    );
  }
  lines.push(
    ` ACCUMULATED: ${drifted.length} of ${rows.length} over tolerance` +
      (drifted.length ? ` from ${drifted[0].stage} onward — fp32 drift, reported not asserted` : ''),
  );
  console.log(lines.join('\n'));
}

describe('forward pass parity', () => {
  it('loads the reference and the weights', () => {
    expect(sidecar.seq).toBeGreaterThan(0);
    expect(sidecar.tensors.length).toBeGreaterThan(400);
    expect(weights.stats.tensorCount).toBeGreaterThan(280);
    console.log(
      `[parity] ${sidecar.seq} tokens, ${sidecar.config.layers} layers, ` +
        `${graph.dispatchCount} dispatches, ${graph.embeddingParts} embedding part(s), ` +
        `${(weights.stats.gpuBytes / 1e6).toFixed(0)} MB resident`,
    );
  });

  it('matches the reference at the embedding boundary', async () => {
    const ok = await compareStage('embeddings', null, sidecar.seq);
    if (!useQ4) {
      expect(ok, 'embeddings diverged — the gather or the split indexing is wrong').toBe(true);
    }
  }, 300_000);

  const layers = Array.from({ length: 24 }, (_, i) => i).filter(
    (i) => onlyLayer === null || i === onlyLayer,
  );

  for (const layer of layers) {
    it(`layer ${layer}`, async () => {
      const seq = sidecar.seq;
      const C = sidecar.config;
      const stages: Array<[string, (x: Float32Array) => Float32Array]> = [
        ['post_rmsnorm', (x) => x],
        ['q', (x) => x],
        ['k', (x) => x],
        ['v', (x) => x],
        ['q_rope', (x) => toSeqMajor(x, seq, C.heads, C.headDim)],
        ['k_rope', (x) => toSeqMajor(x, seq, C.kvHeads, C.headDim)],
        ['attn_weights', (x) => x],
        ['attn_out', (x) => x],
        ['o_proj', (x) => x],
        ['resid_attn', (x) => x],
        ['post_attn_rmsnorm', (x) => x],
        ['mlp_gate', (x) => x],
        ['mlp_up', (x) => x],
        ['mlp_silu_mul', (x) => x],
        ['mlp_down', (x) => x],
        ['resid_mlp', (x) => x],
      ];

      let firstFailure: string | null = null;
      for (const [stage, reshape] of stages) {
        const ok = await compareStage(stage, layer, seq, reshape);
        if (!ok && firstFailure === null) firstFailure = stage;
        if (strict && !ok) break;
      }
      if (useQ4) {
        // Reported, not asserted — see printTable for why the isolated column cannot gate
        // under int4. The per-kernel gate is in test/kernels/matmul_q4.test.ts.
        expect(stages.length).toBeGreaterThan(0);
      } else {
        expect(
          firstFailure,
          `layer ${layer} diverged at ${firstFailure} with reference inputs — this is a real bug`,
        ).toBeNull();
      }
    }, 600_000);
  }

  it('matches the reference at the final norm and logits', async () => {
    const seq = sidecar.seq;
    const normOk = await compareStage('final_norm', null, seq);
    if (!useQ4) expect(normOk).toBe(true);

    // Only the last position is projected, so compare against the reference's last row.
    const lastRow = (x: Float32Array) =>
      Float32Array.from(x.subarray((seq - 1) * sidecar.config.vocab));
    const logitsOk = await compareStage('logits', null, seq, lastRow);
    if (!useQ4) expect(logitsOk).toBe(true);
  }, 300_000);
});

describe('greedy decode', () => {
  it('reproduces the reference token sequence exactly', async () => {
    const generator = new Generator(ctx, pool, graph);
    const result = await generator.generate({
      prompt: sidecar.promptTokens,
      maxTokens: decodeTokens,
    });
    console.log(
      `[parity] generated ${result.tokens.length} tokens, ` +
        `${(result.stepSeconds.reduce((a, b) => a + b, 0) / result.stepSeconds.length * 1000).toFixed(0)} ms/token`,
    );
    console.log(`[parity] engine:    ${JSON.stringify(result.tokens)}`);
    console.log(`[parity] reference: ${JSON.stringify(sidecar.greedy.slice(0, decodeTokens))}`);
    const matching = result.tokens.findIndex((t, i) => t !== sidecar.greedy[i]);
    console.log(`[parity] tokens matching the fp32 reference: ${matching === -1 ? decodeTokens : matching}/${decodeTokens}`);

    if (useQ4) {
      // int4 is not expected to reproduce fp32's greedy sequence — quantization changes the
      // logits and therefore the argmax. Divergence here is information, not a failure; the
      // check that gates quantization quality is perplexity. Asserting exact agreement would
      // be asserting that quantization is lossless.
      expect(result.tokens).toHaveLength(decodeTokens);
    } else {
      expect(result.tokens).toEqual(sidecar.greedy.slice(0, decodeTokens));
    }
  }, 900_000);

  it('is deterministic across runs', async () => {
    const generator = new Generator(ctx, pool, graph);
    const first = await generator.generate({ prompt: sidecar.promptTokens, maxTokens: 5 });
    const second = await generator.generate({ prompt: sidecar.promptTokens, maxTokens: 5 });
    expect(second.tokens).toEqual(first.tokens);
  }, 600_000);
});
