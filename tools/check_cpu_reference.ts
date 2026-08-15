/**
 * Run the TypeScript CPU reference against the PyTorch dump, stage by stage.
 *
 * This runs before any shader exists, and it checks a different thing than the GPU parity
 * harness does. The GPU harness asks "does the shader match the CPU reference"; this asks
 * "does the CPU reference match the model" — layouts, transposes, the RoPE pairing, the GQA
 * head grouping. If this is wrong, every kernel written against it is confidently wrong in
 * the same way.
 *
 *     node tools/check_cpu_reference.ts [--layer N] [--strict]
 */

import { readFileSync } from 'node:fs';
import {
  add,
  attentionApply,
  attentionScores,
  embed,
  linear,
  rmsNorm,
  rope,
  siluMul,
  softmaxRows,
  argmax,
} from '../test/reference/ops.ts';
import { compareArrays, withinTolerance } from '../test/reference/matmul.ts';
import { parseHeader, toFloat32 } from '../src/model/safetensors.ts';

interface RefTensor {
  name: string;
  shape: number[];
  offset: number;
  byteLength: number;
}
interface Sidecar {
  seq: number;
  promptTokens: number[];
  greedy: number[];
  config: {
    layers: number; hidden: number; heads: number; kvHeads: number; headDim: number;
    intermediate: number; vocab: number; rmsNormEps: number; ropeTheta: number;
  };
  tensors: RefTensor[];
}

const argv = process.argv.slice(2);
const onlyLayer = argv.includes('--layer') ? Number(argv[argv.indexOf('--layer') + 1]) : null;
const strict = argv.includes('--strict');

const sidecar = JSON.parse(readFileSync('test/fixtures/reference.json', 'utf8')) as Sidecar;
const refBytes = readFileSync('test/fixtures/reference.bin');
const byName = new Map(sidecar.tensors.map((t) => [t.name, t]));

function ref(name: string): Float32Array {
  const info = byName.get(name);
  if (!info) throw new Error(`reference tensor "${name}" not found`);
  return new Float32Array(
    refBytes.buffer.slice(
      refBytes.byteOffset + info.offset,
      refBytes.byteOffset + info.offset + info.byteLength,
    ),
  );
}

const wBytes = readFileSync('test/fixtures/model.safetensors');
const wHeader = parseHeader(
  wBytes.buffer.slice(wBytes.byteOffset, wBytes.byteOffset + 8 + 1024 * 1024) as ArrayBuffer,
);
const weightCache = new Map<string, Float32Array>();
function weight(name: string): Float32Array {
  const cached = weightCache.get(name);
  if (cached) return cached;
  const info = wHeader.tensors.get(name);
  if (!info) throw new Error(`weight "${name}" not found`);
  const begin = wBytes.byteOffset + wHeader.dataOffset + info.begin;
  const values = toFloat32(info.dtype, wBytes.buffer.slice(begin, begin + info.byteLength));
  weightCache.set(name, values);
  return values;
}

const C = sidecar.config;
const S = sidecar.seq;
const rows: Array<[string, number, number, number, boolean]> = [];
let firstFailure: string | null = null;

function check(name: string, actual: Float32Array): Float32Array {
  const expected = ref(name);
  if (actual.length !== expected.length) {
    throw new Error(`${name}: length ${actual.length} vs reference ${expected.length}`);
  }
  const error = compareArrays(actual, expected);
  const ok = withinTolerance(error);
  rows.push([name, error.maxAbs, error.maxRel, error.worstRatio, ok]);
  if (!ok && firstFailure === null) firstFailure = name;
  if (strict && !ok) {
    printTable();
    console.error(`\nFAIL at ${name}: worst element ${error.worstIndex}, ` +
      `${(error.worstRatio * 100).toFixed(1)}% of tolerance`);
    process.exit(1);
  }
  // Continue from the reference, not from our own output: otherwise one early error
  // contaminates every stage after it and the table stops telling you where the bug is.
  return expected;
}

function printTable(): void {
  console.log('\n stage'.padEnd(34) + 'max abs'.padStart(12) + 'max rel'.padStart(12) + '  of tol   ');
  console.log('-'.repeat(72));
  for (const [name, abs, rel, ratio, ok] of rows) {
    console.log(
      ` ${name}`.padEnd(34) +
        abs.toExponential(2).padStart(12) +
        rel.toExponential(2).padStart(12) +
        `${(ratio * 100).toFixed(1)}%`.padStart(9) +
        (ok ? '  ok' : '  FAIL'),
    );
  }
}

// ---------------------------------------------------------------------------

const ids = sidecar.promptTokens;
let hidden = check('embeddings', embed(ids, weight('model.embed_tokens.weight'), C.hidden));

for (let layer = 0; layer < C.layers; layer++) {
  const p = `layer${layer}`;
  const w = (suffix: string) => weight(`model.layers.${layer}.${suffix}`);
  const skip = onlyLayer !== null && layer !== onlyLayer;

  const normed = rmsNorm(hidden, w('input_layernorm.weight'), S, C.hidden, C.rmsNormEps);
  if (!skip) check(`${p}.post_rmsnorm`, normed);

  const kvDim = C.kvHeads * C.headDim;
  const q = linear(normed, w('self_attn.q_proj.weight'), w('self_attn.q_proj.bias'), S, C.hidden, C.hidden);
  const k = linear(normed, w('self_attn.k_proj.weight'), w('self_attn.k_proj.bias'), S, C.hidden, kvDim);
  const v = linear(normed, w('self_attn.v_proj.weight'), w('self_attn.v_proj.bias'), S, C.hidden, kvDim);
  if (!skip) {
    check(`${p}.q`, q);
    check(`${p}.k`, k);
    check(`${p}.v`, v);
  }

  const qRope = rope(q, S, C.heads, C.headDim, C.ropeTheta);
  const kRope = rope(k, S, C.kvHeads, C.headDim, C.ropeTheta);
  if (!skip) {
    // The reference stores q/k as [batch, heads, seq, headDim]; ours is [seq, heads, headDim].
    check(`${p}.q_rope`, transposeHeadsSeq(qRope, S, C.heads, C.headDim));
    check(`${p}.k_rope`, transposeHeadsSeq(kRope, S, C.kvHeads, C.headDim));
  }

  const scores = attentionScores(qRope, kRope, S, C.heads, C.kvHeads, C.headDim);
  const weights = softmaxRows(scores, C.heads * S, S);
  if (!skip) check(`${p}.attn_weights`, weights);

  const attnOut = attentionApply(weights, v.length === S * kvDim ? ropeIdentity(v) : v, S, C.heads, C.kvHeads, C.headDim);
  if (!skip) check(`${p}.attn_out`, attnOut);

  const projected = linear(attnOut, w('self_attn.o_proj.weight'), null, S, C.hidden, C.hidden);
  if (!skip) check(`${p}.o_proj`, projected);

  const resid1 = add(hidden, projected);
  if (!skip) check(`${p}.resid_attn`, resid1);

  const normed2 = rmsNorm(resid1, w('post_attention_layernorm.weight'), S, C.hidden, C.rmsNormEps);
  if (!skip) check(`${p}.post_attn_rmsnorm`, normed2);

  const gate = linear(normed2, w('mlp.gate_proj.weight'), null, S, C.hidden, C.intermediate);
  const up = linear(normed2, w('mlp.up_proj.weight'), null, S, C.hidden, C.intermediate);
  if (!skip) {
    check(`${p}.mlp_gate`, gate);
    check(`${p}.mlp_up`, up);
  }

  const gated = siluMul(gate, up);
  if (!skip) check(`${p}.mlp_silu_mul`, gated);

  const down = linear(gated, w('mlp.down_proj.weight'), null, S, C.intermediate, C.hidden);
  if (!skip) check(`${p}.mlp_down`, down);

  hidden = add(resid1, down);
  if (!skip) hidden = check(`${p}.resid_mlp`, hidden);
  else hidden = ref(`${p}.resid_mlp`);
}

const finalNorm = rmsNorm(hidden, weight('model.norm.weight'), S, C.hidden, C.rmsNormEps);
check('final_norm', finalNorm);

// Tied LM head: the embedding table is the output projection.
const logits = linear(finalNorm, weight('model.embed_tokens.weight'), null, S, C.hidden, C.vocab);
check('logits', logits);

const lastLogits = logits.subarray((S - 1) * C.vocab, S * C.vocab);
const predicted = argmax(lastLogits);
printTable();
console.log(`\nfirst greedy token: ${predicted} (reference ${sidecar.greedy[0]}) ` +
  (predicted === sidecar.greedy[0] ? 'ok' : 'MISMATCH'));
console.log(firstFailure ? `\nfirst failing stage: ${firstFailure}` : '\nall stages within tolerance');
process.exit(firstFailure || predicted !== sidecar.greedy[0] ? 1 : 0);

/** [seq, heads, dim] -> [heads, seq, dim], which is how the reference stores q/k. */
function transposeHeadsSeq(x: Float32Array, seq: number, heads: number, dim: number): Float32Array {
  const out = new Float32Array(x.length);
  for (let s = 0; s < seq; s++) {
    for (let h = 0; h < heads; h++) {
      for (let d = 0; d < dim; d++) {
        out[(h * seq + s) * dim + d] = x[(s * heads + h) * dim + d];
      }
    }
  }
  return out;
}

function ropeIdentity(v: Float32Array): Float32Array {
  return v;
}
