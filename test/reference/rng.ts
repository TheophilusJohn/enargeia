/** Seeded PRNG so a failing kernel comparison can be reproduced exactly. */
export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Values in [-0.5, 0.5), the range activations and weights actually live in. */
export function randomFloats(count: number, seed = 1): Float32Array {
  const rand = mulberry32(seed);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = rand() - 0.5;
  return out;
}
