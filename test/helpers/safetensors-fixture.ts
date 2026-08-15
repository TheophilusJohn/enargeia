/**
 * Synthetic safetensors files.
 *
 * The real Qwen checkpoint is 988 MB, which is not something a test should download. These
 * builders produce byte-exact files in the same format, and `qwenLikeHeader` reproduces the
 * real checkpoint's tensor inventory — 290 BF16 tensors, the same names, shapes and
 * offsets — so chunk planning and the embedding split are exercised against the real
 * geometry without the bytes.
 */

import {
  parseHeader,
  type SafetensorsDType,
  type SafetensorsHeader,
} from '../../src/model/safetensors.ts';

export interface TensorSpec {
  name: string;
  dtype: SafetensorsDType;
  shape: number[];
  /** Bytes to write. Defaults to a deterministic pattern derived from the name. */
  data?: Uint8Array;
}

const DTYPE_BYTES: Record<string, number> = {
  F64: 8, F32: 4, F16: 2, BF16: 2, I64: 8, I32: 4, I16: 2, I8: 1, U8: 1, BOOL: 1,
};

export interface BuiltFile {
  bytes: Uint8Array;
  headerLength: number;
  dataOffset: number;
}

export function buildSafetensors(
  specs: TensorSpec[],
  metadata?: Record<string, string>,
): BuiltFile {
  const entries: Record<string, unknown> = {};
  if (metadata) entries.__metadata__ = metadata;

  let offset = 0;
  const blobs: Array<{ at: number; data: Uint8Array }> = [];
  for (const spec of specs) {
    const elements = spec.shape.reduce((a, b) => a * b, 1);
    const byteLength = elements * DTYPE_BYTES[spec.dtype];
    const data = spec.data ?? patternBytes(spec.name, byteLength);
    if (data.byteLength !== byteLength) {
      throw new Error(`${spec.name}: supplied ${data.byteLength} bytes, shape needs ${byteLength}`);
    }
    entries[spec.name] = { dtype: spec.dtype, shape: spec.shape, data_offsets: [offset, offset + byteLength] };
    blobs.push({ at: offset, data });
    offset += byteLength;
  }

  const json = new TextEncoder().encode(JSON.stringify(entries));
  const bytes = new Uint8Array(8 + json.length + offset);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(json.length), true);
  bytes.set(json, 8);
  for (const blob of blobs) {
    bytes.set(blob.data, 8 + json.length + blob.at);
  }
  return { bytes, headerLength: json.length, dataOffset: 8 + json.length };
}

/** Deterministic, name-dependent bytes so a misrouted tensor is visible rather than plausible. */
export function patternBytes(seed: string, byteLength: number): Uint8Array {
  let state = 0x811c9dc5;
  for (let i = 0; i < seed.length; i++) {
    state = (Math.imul(state ^ seed.charCodeAt(i), 0x01000193) >>> 0);
  }
  const out = new Uint8Array(byteLength);
  for (let i = 0; i < byteLength; i++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    out[i] = (state >>> 24) & 0xff;
  }
  return out;
}

/** Qwen2.5-0.5B-Instruct's real tensor inventory. */
export const QWEN_CONFIG = {
  layers: 24,
  hidden: 896,
  heads: 14,
  kvHeads: 2,
  headDim: 64,
  intermediate: 4864,
  vocab: 151936,
} as const;

export function qwenTensorSpecs(): TensorSpec[] {
  const { layers, hidden, kvHeads, headDim, intermediate, vocab } = QWEN_CONFIG;
  const kv = kvHeads * headDim;
  const specs: TensorSpec[] = [
    { name: 'model.embed_tokens.weight', dtype: 'BF16', shape: [vocab, hidden] },
  ];
  for (let layer = 0; layer < layers; layer++) {
    const p = `model.layers.${layer}`;
    specs.push(
      { name: `${p}.input_layernorm.weight`, dtype: 'BF16', shape: [hidden] },
      { name: `${p}.self_attn.k_proj.bias`, dtype: 'BF16', shape: [kv] },
      { name: `${p}.self_attn.k_proj.weight`, dtype: 'BF16', shape: [kv, hidden] },
      { name: `${p}.self_attn.o_proj.weight`, dtype: 'BF16', shape: [hidden, hidden] },
      { name: `${p}.self_attn.q_proj.bias`, dtype: 'BF16', shape: [hidden] },
      { name: `${p}.self_attn.q_proj.weight`, dtype: 'BF16', shape: [hidden, hidden] },
      { name: `${p}.self_attn.v_proj.bias`, dtype: 'BF16', shape: [kv] },
      { name: `${p}.self_attn.v_proj.weight`, dtype: 'BF16', shape: [kv, hidden] },
      { name: `${p}.post_attention_layernorm.weight`, dtype: 'BF16', shape: [hidden] },
      { name: `${p}.mlp.down_proj.weight`, dtype: 'BF16', shape: [hidden, intermediate] },
      { name: `${p}.mlp.gate_proj.weight`, dtype: 'BF16', shape: [intermediate, hidden] },
      { name: `${p}.mlp.up_proj.weight`, dtype: 'BF16', shape: [intermediate, hidden] },
    );
  }
  specs.push({ name: 'model.norm.weight', dtype: 'BF16', shape: [hidden] });
  return specs;
}

/**
 * A header with the real inventory but no blob, for testing planning and splitting. Building
 * the actual 988 MB of data is neither necessary nor kind to CI.
 */
export function qwenLikeHeader(): SafetensorsHeader {
  const specs = qwenTensorSpecs();
  const entries: Record<string, unknown> = { __metadata__: { format: 'pt' } };
  let offset = 0;
  for (const spec of specs) {
    const byteLength = spec.shape.reduce((a, b) => a * b, 1) * DTYPE_BYTES[spec.dtype];
    entries[spec.name] = { dtype: spec.dtype, shape: spec.shape, data_offsets: [offset, offset + byteLength] };
    offset += byteLength;
  }
  const json = new TextEncoder().encode(JSON.stringify(entries));
  const prefix = new Uint8Array(8 + json.length);
  new DataView(prefix.buffer).setBigUint64(0, BigInt(json.length), true);
  prefix.set(json, 8);
  return parseHeader(prefix.buffer);
}

/** A small model with Qwen's structure, for end-to-end load tests that need real bytes. */
export function tinyModelSpecs(hidden = 8, vocab = 64, layers = 2): TensorSpec[] {
  const specs: TensorSpec[] = [
    { name: 'model.embed_tokens.weight', dtype: 'BF16', shape: [vocab, hidden] },
  ];
  for (let layer = 0; layer < layers; layer++) {
    specs.push(
      { name: `model.layers.${layer}.input_layernorm.weight`, dtype: 'BF16', shape: [hidden] },
      { name: `model.layers.${layer}.self_attn.q_proj.weight`, dtype: 'BF16', shape: [hidden, hidden] },
      { name: `model.layers.${layer}.self_attn.q_proj.bias`, dtype: 'BF16', shape: [hidden] },
    );
  }
  specs.push({ name: 'model.norm.weight', dtype: 'BF16', shape: [hidden] });
  return specs;
}
