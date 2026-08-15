/**
 * CPU reference for int4 block-wise quantization.
 *
 * Written before the shaders, as the kernels skill requires, and deliberately the slowest
 * obviously-correct version. It also serves as the encoder for tests: quantizing a small
 * matrix here and dequantizing it on the GPU is what proves the two agree on the packing
 * order, which is the part that is easy to get backwards and impossible to see.
 */

export const BLOCK = 64;
export const NIBBLES_PER_WORD = 8;
export const ZEROS_PER_WORD = 8;

export interface QuantizedMatrix {
  rows: number;
  cols: number;
  blockSize: number;
  packed: Uint32Array;
  scales: Float32Array;
  zeros: Uint32Array;
}

/** Asymmetric int4 over the last axis, matching tools/quantize.py exactly. */
export function quantizeMatrix(values: Float32Array, rows: number, cols: number, blockSize = BLOCK): QuantizedMatrix {
  if (cols % blockSize !== 0) throw new Error(`reduction axis ${cols} is not a multiple of ${blockSize}`);
  const blocks = (rows * cols) / blockSize;
  const scales = new Float32Array(blocks);
  const zeroValues = new Uint32Array(blocks);
  const quantized = new Uint32Array(rows * cols);

  for (let b = 0; b < blocks; b++) {
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < blockSize; i++) {
      const v = values[b * blockSize + i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    let scale = (hi - lo) / 15;
    // A constant block has zero range, and an arbitrary scale does not reproduce it: at
    // scale 1 the constant 0.375 quantizes to nibble 0 and dequantizes to 0. |c|/15 puts the
    // constant on nibble 15, or on nibble 0 when it is negative, and the zero-point formula
    // below recovers it exactly.
    if (!(scale > 0)) scale = hi !== 0 ? Math.abs(hi) / 15 : 1;
    const zero = Math.min(15, Math.max(0, Math.round(-lo / scale)));
    scales[b] = scale;
    zeroValues[b] = zero;
    for (let i = 0; i < blockSize; i++) {
      const q = Math.round(values[b * blockSize + i] / scale + zero);
      quantized[b * blockSize + i] = Math.min(15, Math.max(0, q));
    }
  }

  const packed = new Uint32Array((rows * cols) / NIBBLES_PER_WORD);
  for (let w = 0; w < packed.length; w++) {
    let word = 0;
    for (let n = 0; n < NIBBLES_PER_WORD; n++) {
      word |= quantized[w * NIBBLES_PER_WORD + n] << (n * 4);
    }
    packed[w] = word >>> 0;
  }

  const zeros = new Uint32Array(Math.ceil(blocks / ZEROS_PER_WORD));
  for (let w = 0; w < zeros.length; w++) {
    let word = 0;
    for (let n = 0; n < ZEROS_PER_WORD; n++) {
      const b = w * ZEROS_PER_WORD + n;
      if (b < blocks) word |= zeroValues[b] << (n * 4);
    }
    zeros[w] = word >>> 0;
  }

  return { rows, cols, blockSize, packed, scales, zeros };
}

/** One weight, by the same indexing the shaders use. */
export function dequantElement(q: QuantizedMatrix, row: number, col: number): number {
  const flat = row * q.cols + col;
  const nib = (q.packed[flat >>> 3] >>> ((flat & 7) * 4)) & 0xf;
  const blocksPerRow = q.cols / q.blockSize;
  const block = row * blocksPerRow + Math.floor(col / q.blockSize);
  const zero = (q.zeros[block >>> 3] >>> ((block & 7) * 4)) & 0xf;
  return (nib - zero) * q.scales[block];
}

export function dequantizeMatrix(q: QuantizedMatrix): Float32Array {
  const out = new Float32Array(q.rows * q.cols);
  for (let r = 0; r < q.rows; r++) {
    for (let c = 0; c < q.cols; c++) out[r * q.cols + c] = dequantElement(q, r, c);
  }
  return out;
}

/** out[m, n] = sum_k x[m, k] * dequant(W)[n, k] + bias[n] */
export function linearQ4(
  x: Float32Array,
  w: QuantizedMatrix,
  bias: Float32Array | null,
  m: number,
  k: number,
  n: number,
): Float32Array {
  const out = new Float32Array(m * n);
  for (let row = 0; row < m; row++) {
    for (let col = 0; col < n; col++) {
      let acc = bias ? bias[col] : 0;
      for (let i = 0; i < k; i++) acc += x[row * k + i] * dequantElement(w, col, i);
      out[row * n + col] = acc;
    }
  }
  return out;
}
