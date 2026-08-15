/**
 * CPU reference implementations, written before the shaders.
 *
 * Deliberately the slowest, most obviously correct versions: nested loops, no blocking, no
 * cleverness. These pin down the semantics — layouts, transposes, which axis is reduced —
 * before the GPU makes any of it hard to inspect, and they are what every kernel is compared
 * against.
 *
 * Layout conventions, matching the checkpoint:
 *   activations   [seq, features], row-major
 *   linear weight [outFeatures, inFeatures], row-major — so a projection reads W transposed
 *   q             [seq, heads, headDim]
 *   k, v          [seq, kvHeads, headDim]
 *   scores        [heads, seq, seq]
 */

export interface ModelShape {
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

/** y[s, f] = x[s, f] / sqrt(mean(x[s, :]^2) + eps) * weight[f] */
export function rmsNorm(
  x: Float32Array,
  weight: Float32Array,
  seq: number,
  features: number,
  eps: number,
): Float32Array {
  const out = new Float32Array(seq * features);
  for (let s = 0; s < seq; s++) {
    let sum = 0;
    for (let f = 0; f < features; f++) {
      const v = x[s * features + f];
      sum += v * v;
    }
    // Qwen2 computes the reciprocal square root of the *mean*, not the sum.
    const scale = 1 / Math.sqrt(sum / features + eps);
    for (let f = 0; f < features; f++) {
      out[s * features + f] = x[s * features + f] * scale * weight[f];
    }
  }
  return out;
}

/**
 * y[m, n] = sum_k x[m, k] * w[n, k] + bias[n]
 *
 * Note the transpose: checkpoint linear weights are [out, in], so the reduction runs along
 * each weight *row*, not down a column. Getting this backwards produces plausible garbage.
 */
export function linear(
  x: Float32Array,
  w: Float32Array,
  bias: Float32Array | null,
  m: number,
  k: number,
  n: number,
): Float32Array {
  const out = new Float32Array(m * n);
  for (let row = 0; row < m; row++) {
    for (let col = 0; col < n; col++) {
      let acc = bias ? bias[col] : 0;
      for (let i = 0; i < k; i++) {
        acc += x[row * k + i] * w[col * k + i];
      }
      out[row * n + col] = acc;
    }
  }
  return out;
}

/**
 * Rotary position embedding, applied in place over a copy.
 *
 * Qwen uses the "half rotation" layout: element i pairs with element i + headDim/2, not with
 * its immediate neighbour. Pairing adjacently is the single most common way to get RoPE
 * subtly wrong — it produces correct-looking magnitudes and wrong directions.
 */
export function rope(
  x: Float32Array,
  seq: number,
  numHeads: number,
  headDim: number,
  theta: number,
  positionOffset = 0,
): Float32Array {
  const out = Float32Array.from(x);
  const half = headDim / 2;
  for (let s = 0; s < seq; s++) {
    const position = s + positionOffset;
    for (let h = 0; h < numHeads; h++) {
      const base = (s * numHeads + h) * headDim;
      for (let i = 0; i < half; i++) {
        const freq = 1 / theta ** ((2 * i) / headDim);
        const angle = position * freq;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const a = x[base + i];
        const b = x[base + i + half];
        out[base + i] = a * cos - b * sin;
        out[base + i + half] = b * cos + a * sin;
      }
    }
  }
  return out;
}

/**
 * scores[h, i, j] = dot(q[i, h], k[j, kv(h)]) / sqrt(headDim), with j > i set to -inf.
 *
 * Grouped-query attention: 14 query heads share 2 key/value heads, so query head h reads
 * kv head floor(h / (heads / kvHeads)). The grouping is contiguous — heads 0..6 use kv head
 * 0 and 7..13 use kv head 1 — which matches how the projections are laid out.
 */
export function attentionScores(
  q: Float32Array,
  k: Float32Array,
  seq: number,
  heads: number,
  kvHeads: number,
  headDim: number,
): Float32Array {
  const group = heads / kvHeads;
  const scale = 1 / Math.sqrt(headDim);
  const out = new Float32Array(heads * seq * seq);
  for (let h = 0; h < heads; h++) {
    const kvHead = Math.floor(h / group);
    for (let i = 0; i < seq; i++) {
      for (let j = 0; j < seq; j++) {
        if (j > i) {
          out[(h * seq + i) * seq + j] = -Infinity;
          continue;
        }
        let acc = 0;
        for (let d = 0; d < headDim; d++) {
          acc += q[(i * heads + h) * headDim + d] * k[(j * kvHeads + kvHead) * headDim + d];
        }
        out[(h * seq + i) * seq + j] = acc * scale;
      }
    }
  }
  return out;
}

/**
 * Row-wise softmax, max-subtracted.
 *
 * Subtracting the row max before exponentiating is not an optimization — attention scores
 * reach magnitudes where exp overflows to Infinity, and Infinity/Infinity is NaN. A row that
 * is entirely -inf (impossible under a causal mask, but reachable with an empty window)
 * yields zeros rather than NaN.
 */
export function softmaxRows(x: Float32Array, rows: number, cols: number): Float32Array {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    let max = -Infinity;
    for (let c = 0; c < cols; c++) max = Math.max(max, x[r * cols + c]);
    if (!Number.isFinite(max)) {
      continue; // all -inf: leave the row at zero
    }
    let sum = 0;
    for (let c = 0; c < cols; c++) {
      const e = Math.exp(x[r * cols + c] - max);
      out[r * cols + c] = e;
      sum += e;
    }
    for (let c = 0; c < cols; c++) out[r * cols + c] /= sum;
  }
  return out;
}

/** out[i, h, d] = sum_j weights[h, i, j] * v[j, kv(h), d], flattened to [seq, heads*headDim]. */
export function attentionApply(
  weights: Float32Array,
  v: Float32Array,
  seq: number,
  heads: number,
  kvHeads: number,
  headDim: number,
): Float32Array {
  const group = heads / kvHeads;
  const out = new Float32Array(seq * heads * headDim);
  for (let h = 0; h < heads; h++) {
    const kvHead = Math.floor(h / group);
    for (let i = 0; i < seq; i++) {
      for (let d = 0; d < headDim; d++) {
        let acc = 0;
        for (let j = 0; j <= i; j++) {
          acc += weights[(h * seq + i) * seq + j] * v[(j * kvHeads + kvHead) * headDim + d];
        }
        out[(i * heads + h) * headDim + d] = acc;
      }
    }
  }
  return out;
}

/** SiLU(x) = x * sigmoid(x), also called swish. */
export function silu(x: number): number {
  return x / (1 + Math.exp(-x));
}

/** out[i] = silu(gate[i]) * up[i] — the SwiGLU gate. */
export function siluMul(gate: Float32Array, up: Float32Array): Float32Array {
  const out = new Float32Array(gate.length);
  for (let i = 0; i < gate.length; i++) out[i] = silu(gate[i]) * up[i];
  return out;
}

export function add(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

/** out[s, :] = table[ids[s], :] */
export function embed(
  ids: ArrayLike<number>,
  table: Float32Array,
  hidden: number,
): Float32Array {
  const out = new Float32Array(ids.length * hidden);
  for (let s = 0; s < ids.length; s++) {
    out.set(table.subarray(ids[s] * hidden, (ids[s] + 1) * hidden), s * hidden);
  }
  return out;
}

/** Index of the largest value, lowest index winning ties — matching torch.argmax. */
export function argmax(x: Float32Array): number {
  let best = 0;
  for (let i = 1; i < x.length; i++) {
    if (x[i] > x[best]) best = i;
  }
  return best;
}
