#!/usr/bin/env node
/**
 * Generate a safetensors file with Qwen2.5-0.5B-Instruct's exact tensor inventory.
 *
 * Real geometry, synthetic bytes: 290 BF16 tensors with the real names, shapes and offsets,
 * totalling 988,065,536 bytes of data. Used by the M2 load benchmark so its cold/warm numbers
 * are reproducible without a 988 MB download, and so the loader is exercised on the shapes it
 * will actually see.
 *
 *   node tools/gen_synthetic_safetensors.mjs public/models/qwen2.5-0.5b-synthetic.safetensors
 */
import { createWriteStream } from 'node:fs';
const C = { layers: 24, hidden: 896, kvHeads: 2, headDim: 64, intermediate: 4864, vocab: 151936 };
const kv = C.kvHeads * C.headDim;
const specs = [['model.embed_tokens.weight', [C.vocab, C.hidden]]];
for (let l = 0; l < C.layers; l++) {
  const p = `model.layers.${l}`;
  specs.push([`${p}.input_layernorm.weight`, [C.hidden]],
    [`${p}.self_attn.k_proj.bias`, [kv]], [`${p}.self_attn.k_proj.weight`, [kv, C.hidden]],
    [`${p}.self_attn.o_proj.weight`, [C.hidden, C.hidden]],
    [`${p}.self_attn.q_proj.bias`, [C.hidden]], [`${p}.self_attn.q_proj.weight`, [C.hidden, C.hidden]],
    [`${p}.self_attn.v_proj.bias`, [kv]], [`${p}.self_attn.v_proj.weight`, [kv, C.hidden]],
    [`${p}.post_attention_layernorm.weight`, [C.hidden]],
    [`${p}.mlp.down_proj.weight`, [C.hidden, C.intermediate]],
    [`${p}.mlp.gate_proj.weight`, [C.intermediate, C.hidden]],
    [`${p}.mlp.up_proj.weight`, [C.intermediate, C.hidden]]);
}
specs.push(['model.norm.weight', [C.hidden]]);

const entries = { __metadata__: { format: 'pt' } };
let off = 0;
for (const [name, shape] of specs) {
  const bytes = shape.reduce((a, b) => a * b, 1) * 2;
  entries[name] = { dtype: 'BF16', shape, data_offsets: [off, off + bytes] };
  off += bytes;
}
const json = Buffer.from(JSON.stringify(entries), 'utf8');
const out = createWriteStream(process.argv[2]);
const len = Buffer.alloc(8); len.writeBigUInt64LE(BigInt(json.length));
out.write(len); out.write(json);
// bf16 values in a sane range so widening produces finite floats
const CH = 1 << 22; const buf = Buffer.alloc(CH);
for (let i = 0; i < CH; i += 2) buf.writeUInt16LE(0x3f00 | ((i >> 1) & 0x7f), i);
let written = 0;
while (written < off) {
  const n = Math.min(CH, off - written);
  if (!out.write(buf.subarray(0, n))) await new Promise(r => out.once('drain', r));
  written += n;
}
out.end();
await new Promise(r => out.once('finish', r));
console.log(`tensors=${specs.length} header=${json.length} data=${off} total=${8 + json.length + off}`);
