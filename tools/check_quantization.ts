// Does the shipped .enargeia file decode to the weights it was made from?
import { readFileSync } from 'node:fs';
import { parseEnargeiaHeader } from '../src/model/enargeia.ts';
import { parseHeader, toFloat32 } from '../src/model/safetensors.ts';

const q4 = readFileSync('public/models/qwen2.5-0.5b-q4.enargeia');
const head = parseEnargeiaHeader(q4.buffer.slice(q4.byteOffset, q4.byteOffset + 1_000_000) as ArrayBuffer);
const st = readFileSync('test/fixtures/model.safetensors');
const stHead = parseHeader(st.buffer.slice(st.byteOffset, st.byteOffset + 2_000_000) as ArrayBuffer);

for (const name of ['model.layers.0.self_attn.q_proj.weight', 'model.layers.0.self_attn.v_proj.weight', 'model.embed_tokens.weight']) {
  const info = head.tensors.get(name)!;
  if (info.dtype !== 'Q4') throw new Error('not q4');
  const base = q4.byteOffset + head.dataOffset;
  const packed = new Uint32Array(q4.buffer.slice(base + info.packed[0], base + info.packed[1]));
  const scales = new Float32Array(q4.buffer.slice(base + info.scales[0], base + info.scales[1]));
  const zeros  = new Uint32Array(q4.buffer.slice(base + info.zeros[0], base + info.zeros[1]));

  const orig = stHead.tensors.get(name)!;
  const ob = st.byteOffset + stHead.dataOffset + orig.begin;
  const ref = toFloat32(orig.dtype, st.buffer.slice(ob, ob + orig.byteLength));

  const [rows, cols] = info.shape;
  const bpr = info.blocksPerRow;
  let maxAbs = 0, sum = 0;
  const limit = Math.min(rows, 2000);
  for (let r = 0; r < limit; r++) {
    for (let c = 0; c < cols; c++) {
      const flat = r * cols + c;
      const nib = (packed[flat >>> 3] >>> ((flat & 7) * 4)) & 0xf;
      const blk = r * bpr + ((c / info.blockSize) | 0);
      const z = (zeros[blk >>> 3] >>> ((blk & 7) * 4)) & 0xf;
      const v = (nib - z) * scales[blk];
      const d = Math.abs(v - ref[flat]);
      if (d > maxAbs) maxAbs = d;
      sum += d * d;
    }
  }
  console.log(`${name}  maxAbs=${maxAbs.toFixed(5)}  rms=${Math.sqrt(sum / (limit * cols)).toFixed(5)}`);
}

// ---------------------------------------------------------------------------
// What error should a *correct* int4 implementation produce at a stage boundary?
//
// If the GPU's error matches this, every bit of the divergence is quantization loss and none
// of it is a bug. Computed on the CPU from the shipped file, so it exercises the same bytes
// the shader reads.
const ref = JSON.parse(readFileSync('test/fixtures/reference.json', 'utf8'));
const refBin = readFileSync('test/fixtures/reference.bin');
const tensor = (n: string) => {
  const t = ref.tensors.find((x: any) => x.name === n);
  return new Float32Array(refBin.buffer.slice(refBin.byteOffset + t.offset, refBin.byteOffset + t.offset + t.byteLength));
};

function dequantTensor(name: string): { values: Float32Array; rows: number; cols: number } {
  const info = head.tensors.get(name)!;
  if (info.dtype !== 'Q4') throw new Error('not q4');
  const base = q4.byteOffset + head.dataOffset;
  const packed = new Uint32Array(q4.buffer.slice(base + info.packed[0], base + info.packed[1]));
  const scales = new Float32Array(q4.buffer.slice(base + info.scales[0], base + info.scales[1]));
  const zeros = new Uint32Array(q4.buffer.slice(base + info.zeros[0], base + info.zeros[1]));
  const [rows, cols] = info.shape;
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const flat = r * cols + c;
      const nib = (packed[flat >>> 3] >>> ((flat & 7) * 4)) & 0xf;
      const blk = r * info.blocksPerRow + ((c / info.blockSize) | 0);
      const z = (zeros[blk >>> 3] >>> ((blk & 7) * 4)) & 0xf;
      out[flat] = (nib - z) * scales[blk];
    }
  }
  return { values: out, rows, cols };
}

console.log('\nexpected stage error from quantization alone (CPU, shipped weights):');
for (const [stage, weightName, biasName, n] of [
  ['layer0.q', 'model.layers.0.self_attn.q_proj.weight', 'model.layers.0.self_attn.q_proj.bias', 896],
  ['layer0.v', 'model.layers.0.self_attn.v_proj.weight', 'model.layers.0.self_attn.v_proj.bias', 128],
] as const) {
  const w = dequantTensor(weightName);
  const bInfo = head.tensors.get(biasName)!;
  const bBase = q4.byteOffset + head.dataOffset;
  const bias = new Float32Array(q4.buffer.slice(bBase + (bInfo as any).data[0], bBase + (bInfo as any).data[1]));
  const x = tensor('layer0.post_rmsnorm');
  const seq = ref.seq, k = 896;
  const expected = tensor(stage);
  let maxAbs = 0;
  for (let row = 0; row < seq; row++) {
    for (let col = 0; col < n; col++) {
      let acc = bias[col];
      for (let i = 0; i < k; i++) acc += x[row * k + i] * w.values[col * k + i];
      maxAbs = Math.max(maxAbs, Math.abs(acc - expected[row * n + col]));
    }
  }
  console.log(`  ${stage}: max abs error ${maxAbs.toFixed(4)} (int4 threshold is 0.05 + 0.08*|value|)`);
}

// ---------------------------------------------------------------------------
// How much of the perplexity cost comes from the tied LM head?
//
// The embedding table is both the input lookup and the output projection. Quantizing it is
// the largest single saving — 544 MB of the 1885 MB fp32 residency — but the output
// projection is also where a small weight error turns directly into a shifted token
// distribution. This measures the logit error attributable to that one tensor.
const fn = tensor('final_norm');
const seqLen = ref.seq, hidden = 896;
const embQ = dequantTensor('model.embed_tokens.weight');
const embOrig = (() => {
  const o = stHead.tensors.get('model.embed_tokens.weight')!;
  const b = st.byteOffset + stHead.dataOffset + o.begin;
  return toFloat32(o.dtype, st.buffer.slice(b, b + o.byteLength));
})();

const row = fn.subarray((seqLen - 1) * hidden, seqLen * hidden);
const sample = 20000; // first 20k vocabulary entries is plenty to characterise the error
let maxAbs = 0, sum = 0, spread = 0;
const exact = new Float64Array(sample);
for (let v = 0; v < sample; v++) {
  let a = 0, b = 0;
  for (let i = 0; i < hidden; i++) {
    a += row[i] * embOrig[v * hidden + i];
    b += row[i] * embQ.values[v * hidden + i];
  }
  exact[v] = a;
  const d = Math.abs(a - b);
  if (d > maxAbs) maxAbs = d;
  sum += d * d;
}
let lo = Infinity, hi = -Infinity;
for (const v of exact) { if (v < lo) lo = v; if (v > hi) hi = v; }
console.log(`\nlm head logit error from quantizing the tied embedding (first ${sample} of vocab):`);
console.log(`  max abs ${maxAbs.toFixed(4)}  rms ${Math.sqrt(sum / sample).toFixed(4)}  logit range ${lo.toFixed(2)}..${hi.toFixed(2)}`);
console.log(`  rms error is ${(Math.sqrt(sum / sample) / (hi - lo) * 100).toFixed(2)}% of the logit range`);
