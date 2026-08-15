/**
 * CPU reference for matmul. Deliberately the slowest, most obviously correct version:
 * three nested loops, no blocking, no accumulator tricks. Every GPU matmul is compared
 * against this before it is compared against a clock.
 */

export interface MatmulShape {
  m: number;
  n: number;
  k: number;
}

/** C[m,n] = A[m,k] * B[k,n], all row-major. */
export function matmulRef(a: Float32Array, b: Float32Array, shape: MatmulShape): Float32Array {
  const { m, n, k } = shape;
  const c = new Float32Array(m * n);
  for (let row = 0; row < m; row++) {
    for (let col = 0; col < n; col++) {
      let acc = 0;
      for (let i = 0; i < k; i++) {
        acc += a[row * k + i] * b[i * n + col];
      }
      c[row * n + col] = acc;
    }
  }
  return c;
}

export interface ErrorReport {
  maxAbs: number;
  maxRel: number;
  /**
   * Worst per-element violation of `abs <= atol + rtol * |expected|`, normalized so 1.0
   * is exactly at tolerance. This is the number that decides pass or fail.
   */
  worstRatio: number;
  /** Index of the worst violation, for shrinking the case down to one number. */
  worstIndex: number;
  count: number;
}

export interface Tolerance {
  abs: number;
  rel: number;
}

/**
 * Both error measures are reported, because either alone lies: relative error explodes
 * near zero, absolute error hides real damage in large activations.
 *
 * The verdict combines them per element rather than thresholding each separately. An
 * output of 5e-5 that is off by 6e-7 has a 1.2% relative error and is fine; thresholding
 * max-rel on its own would fail every correct fp32 matmul that happens to produce a value
 * near zero, and a gate that cries wolf gets ignored.
 */
export function compareArrays(
  actual: Float32Array,
  expected: Float32Array,
  tolerance: Tolerance = FP32_TOLERANCE,
): ErrorReport {
  if (actual.length !== expected.length) {
    throw new Error(`length mismatch: ${actual.length} vs ${expected.length}`);
  }
  let maxAbs = 0;
  let maxRel = 0;
  let worstRatio = 0;
  let worstIndex = -1;
  for (let i = 0; i < actual.length; i++) {
    const abs = Math.abs(actual[i] - expected[i]);
    const magnitude = Math.abs(expected[i]);
    maxAbs = Math.max(maxAbs, abs);
    if (magnitude > 1e-6) maxRel = Math.max(maxRel, abs / magnitude);

    const ratio = abs / (tolerance.abs + tolerance.rel * magnitude);
    if (ratio > worstRatio) {
      worstRatio = ratio;
      worstIndex = i;
    }
  }
  return { maxAbs, maxRel, worstRatio, worstIndex, count: actual.length };
}

export function withinTolerance(report: ErrorReport): boolean {
  return report.worstRatio <= 1;
}

/** fp32 kernel thresholds, per the parity harness. */
export const FP32_TOLERANCE: Tolerance = { abs: 1e-4, rel: 1e-4 };

/**
 * int4 thresholds, per the parity harness.
 *
 * Loose by necessity — quantization *is* lossy, and at this tolerance the harness cannot
 * distinguish expected loss from a bug. That is why int4 gets a second check that fp32 does
 * not need: perplexity against the fp32 baseline. See `npm run perplexity`.
 */
export const Q4_TOLERANCE: Tolerance = { abs: 5e-2, rel: 8e-2 };
