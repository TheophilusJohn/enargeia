import { afterAll, describe, expect, it } from 'vitest';
import { readFloats } from '../../src/gpu/index.ts';
import { MatmulNaive, matmulNaiveWorkgroups } from '../../src/kernels/matmul_naive.ts';
import { matmulFlops, type MatmulShape } from '../../src/kernels/matmul_shared.ts';
import { compareArrays, matmulRef, withinTolerance } from '../reference/matmul.ts';
import { randomFloats } from '../reference/rng.ts';
import { bytesEqual, expectNoGPUError, gpu, teardownGPU } from '../helpers/gpu.ts';
import { checkDeterminism } from '../helpers/determinism.ts';
import { RACE_HUNT_SHAPE, matmulSubject } from '../helpers/matmul-subject.ts';

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

afterAll(teardownGPU);

async function run(shape: MatmulShape, a: Float32Array, b: Float32Array): Promise<Float32Array> {
  const { ctx, pool, cache } = await gpu();
  const matmul = new MatmulNaive(ctx.device, cache);

  return expectNoGPUError(ctx.device, async () => {
    const bufA = pool.acquire(a.byteLength, STORAGE, 'a');
    const bufB = pool.acquire(b.byteLength, STORAGE, 'b');
    const bufC = pool.acquire(shape.m * shape.n * 4, STORAGE, 'c');
    ctx.queue.writeBuffer(bufA.buffer, 0, a);
    ctx.queue.writeBuffer(bufB.buffer, 0, b);

    const binding = matmul.bind(pool, { a: bufA, b: bufB, out: bufC, shape });
    const encoder = ctx.device.createCommandEncoder();
    matmul.dispatch(encoder, binding);
    ctx.queue.submit([encoder.finish()]);

    const out = await readFloats(ctx, pool, bufC);
    matmul.release(pool, binding);
    for (const buf of [bufC, bufB, bufA]) pool.release(buf);
    return out;
  });
}

describe('matmul_naive', () => {
  /**
   * Edge shapes first. The workgroup is 16x16, so anything not a multiple of 16 runs
   * invocations past the end of the output and exercises the bounds check; a kernel
   * missing that guard passes on 64x64x64 and corrupts memory on 17x129x96.
   */
  const shapes: Array<[string, MatmulShape]> = [
    ['1x1x1 — single element', { m: 1, n: 1, k: 1 }],
    ['1x896x896 — decode shape, matrix by vector', { m: 1, n: 896, k: 896 }],
    ['64x64x64 — exact workgroup multiple', { m: 64, n: 64, k: 64 }],
    ['37x45x23 — no dimension a multiple of 16', { m: 37, n: 45, k: 23 }],
    ['17x129x96 — one past a tile boundary', { m: 17, n: 129, k: 96 }],
    ['16x16x1 — reduction of length one', { m: 16, n: 16, k: 1 }],
    ['1x1x4864 — deep reduction, MLP intermediate', { m: 1, n: 1, k: 4864 }],
  ];

  for (const [name, shape] of shapes) {
    it(`matches the CPU reference: ${name}`, async () => {
      const a = randomFloats(shape.m * shape.k, 7);
      const b = randomFloats(shape.k * shape.n, 13);

      const actual = await run(shape, a, b);
      const error = compareArrays(actual, matmulRef(a, b, shape));

      expect(
        withinTolerance(error),
        `worst element ${error.worstIndex} at ${(error.worstRatio * 100).toFixed(1)}% of tolerance ` +
          `(abs ${error.maxAbs.toExponential(2)}, rel ${error.maxRel.toExponential(2)})`,
      ).toBe(true);
    });
  }

  it('produces byte-identical output across runs', async () => {
    const shape: MatmulShape = { m: 37, n: 45, k: 96 };
    const a = randomFloats(shape.m * shape.k, 21);
    const b = randomFloats(shape.k * shape.n, 22);

    const runs = [await run(shape, a, b), await run(shape, a, b), await run(shape, a, b)];
    expect(bytesEqual(runs[0], runs[1])).toBe(true);
    expect(bytesEqual(runs[1], runs[2])).toBe(true);
  });

  /**
   * This kernel has no shared memory and therefore no barrier to omit, so it cannot race.
   * The check is here anyway: it keeps the harness exercised against a known-good kernel,
   * and it is the baseline any tiled version has to match.
   */
  it('survives 20 concurrent runs at a race-hunting shape', async () => {
    const { ctx, pool, cache } = await gpu();
    const matmul = new MatmulNaive(ctx.device, cache);
    const handle = matmulSubject(ctx.queue, pool, matmul, RACE_HUNT_SHAPE);

    try {
      const report = await checkDeterminism(ctx, pool, handle.subject, {
        label: 'matmul_naive',
      });
      expect(report.deterministic, report.summary).toBe(true);
      expect(report.iterations).toBe(20);
      expect(handle.subject.workgroupsPerRun).toBe(1024);
    } finally {
      handle.dispose();
    }
  });

  it('does not write past the last output element', async () => {
    // 17x17 dispatches 2x2 workgroups = 32x32 invocations, so 735 of the 1024 threads are
    // out of range. The output buffer is bound large enough that a missing guard would
    // land inside it and be visible, rather than being swallowed by WGSL's clamping of
    // out-of-bounds writes — which is what makes this test worth having.
    const { ctx, pool, cache } = await gpu();
    const matmul = new MatmulNaive(ctx.device, cache);
    const shape: MatmulShape = { m: 17, n: 17, k: 8 };
    const used = shape.m * shape.n;

    const a = randomFloats(shape.m * shape.k, 5);
    const b = randomFloats(shape.k * shape.n, 6);
    const bufA = pool.acquire(a.byteLength, STORAGE, 'a');
    const bufB = pool.acquire(b.byteLength, STORAGE, 'b');
    const bufC = pool.acquire(32 * 32 * 4, STORAGE, 'c');
    ctx.queue.writeBuffer(bufA.buffer, 0, a);
    ctx.queue.writeBuffer(bufB.buffer, 0, b);
    ctx.queue.writeBuffer(bufC.buffer, 0, new Float32Array(32 * 32).fill(-999));

    const binding = matmul.bind(pool, { a: bufA, b: bufB, out: bufC, shape });
    const encoder = ctx.device.createCommandEncoder();
    matmul.dispatch(encoder, binding);
    ctx.queue.submit([encoder.finish()]);

    const all = await readFloats(ctx, pool, bufC);
    const error = compareArrays(all.slice(0, used), matmulRef(a, b, shape));

    expect(withinTolerance(error)).toBe(true);
    expect(all.subarray(used).every((v) => v === -999)).toBe(true);

    matmul.release(pool, binding);
    for (const buf of [bufC, bufB, bufA]) pool.release(buf);
  });

  it('reuses one pipeline across instances', async () => {
    const { ctx, cache } = await gpu();
    const first = new MatmulNaive(ctx.device, cache);
    const second = new MatmulNaive(ctx.device, cache);
    expect(second.pipeline).toBe(first.pipeline);
    expect(second.bindGroupLayout).toBe(first.bindGroupLayout);
  });

  it('rejects shapes it cannot index correctly', async () => {
    const { ctx, pool, cache } = await gpu();
    const matmul = new MatmulNaive(ctx.device, cache);
    const buf = pool.acquire(1024, STORAGE, 'x');

    const bad = { a: buf, b: buf, out: buf };
    expect(() => matmul.bind(pool, { ...bad, shape: { m: 0, n: 4, k: 4 } })).toThrow(RangeError);
    expect(() => matmul.bind(pool, { ...bad, shape: { m: 4, n: -1, k: 4 } })).toThrow(RangeError);
    expect(() => matmul.bind(pool, { ...bad, shape: { m: 1.5, n: 4, k: 4 } })).toThrow(RangeError);
    // 128 KiB of buffer cannot hold a 4096x4096 operand; caught before the GPU sees it.
    expect(() => matmul.bind(pool, { ...bad, shape: { m: 4096, n: 4096, k: 4096 } })).toThrow(RangeError);
    pool.release(buf);
  });

  it('derives dispatch geometry that covers the output exactly once', () => {
    expect(matmulNaiveWorkgroups({ m: 32, n: 32, k: 1 })).toEqual([2, 2, 1]);
    expect(matmulNaiveWorkgroups({ m: 17, n: 129, k: 1 })).toEqual([9, 2, 1]);
    expect(matmulNaiveWorkgroups({ m: 1, n: 1, k: 1 })).toEqual([1, 1, 1]);
    expect(matmulFlops({ m: 2, n: 3, k: 5 })).toBe(60);
  });
});
