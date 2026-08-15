import { afterAll, describe, expect, it } from 'vitest';
import { readFloats } from '../../src/gpu/index.ts';
import { MatmulBlock42, matmulBlock42Workgroups } from '../../src/kernels/matmul_block42.ts';
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
  const matmul = new MatmulBlock42(ctx.device, cache);

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

describe('matmul_block42', () => {
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
    // Specific to the 64x32 footprint, which is ragged on *both* axes. A thread owns four
    // rows 16 apart and two columns 16 apart, so these leave workgroups where the row mask
    // and the column mask disagree — the case a guard combining either axis gets wrong.
    ['63x31x17 — inside one tile, ragged on both axes', { m: 63, n: 31, k: 17 }],
    ['64x32x16 — exactly one footprint, no masking', { m: 64, n: 32, k: 16 }],
    ['65x33x40 — one row and one column past the tile', { m: 65, n: 33, k: 40 }],
    ['65x32x16 — ragged rows, exact columns', { m: 65, n: 32, k: 16 }],
    ['64x33x16 — exact rows, ragged columns', { m: 64, n: 33, k: 16 }],
    ['80x48x33 — straddles both edges mid-footprint', { m: 80, n: 48, k: 33 }],
    ['129x65x40 — two workgroups on each axis, both ragged', { m: 129, n: 65, k: 40 }],
    ['20x20x16 — one of eight accumulators in range', { m: 20, n: 20, k: 16 }],
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
   * The test this file exists for. 20 runs dispatched concurrently at 512x512x512 — 1024
   * workgroups each, 32 tile iterations per output — comparing every result against the
   * first on bit patterns. A missing second barrier shows up here and essentially nowhere
   * else in the suite.
   */
  it('survives 20 concurrent runs at a race-hunting shape', async () => {
    const { ctx, pool, cache } = await gpu();
    const matmul = new MatmulBlock42(ctx.device, cache);
    const handle = matmulSubject(ctx.queue, pool, matmul, RACE_HUNT_SHAPE);

    try {
      const report = await checkDeterminism(ctx, pool, handle.subject, {
        label: 'matmul_block42',
      });
      expect(report.deterministic, report.summary).toBe(true);
    } finally {
      handle.dispose();
    }
  });

  it('does not write past the last output element', async () => {
    // 17x17 dispatches a single workgroup whose footprint is 64x32, so it produces 2048
    // candidate outputs for 289 real ones — 1759 masked writes, and because the footprint is
    // ragged on both axes the row and column masks disagree for most of them. The output buffer is bound large enough that a missing guard would land
    // inside it and be visible, rather than being swallowed by WGSL's clamping of
    // out-of-bounds writes.
    //
    // k=40 spans three tiles of 16 with a ragged last one of 8. That is the case a tiled
    // kernel gets wrong: out-of-range tile loads must write zero rather than leave the
    // previous tile's values in shared memory, and the out-of-range threads must reach
    // every barrier rather than returning early.
    const { ctx, pool, cache } = await gpu();
    const matmul = new MatmulBlock42(ctx.device, cache);
    const shape: MatmulShape = { m: 17, n: 17, k: 40 };
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
    const first = new MatmulBlock42(ctx.device, cache);
    const second = new MatmulBlock42(ctx.device, cache);
    expect(second.pipeline).toBe(first.pipeline);
    expect(second.bindGroupLayout).toBe(first.bindGroupLayout);
  });

  it('rejects shapes it cannot index correctly', async () => {
    const { ctx, pool, cache } = await gpu();
    const matmul = new MatmulBlock42(ctx.device, cache);
    const buf = pool.acquire(1024, STORAGE, 'x');

    const bad = { a: buf, b: buf, out: buf };
    expect(() => matmul.bind(pool, { ...bad, shape: { m: 0, n: 4, k: 4 } })).toThrow(RangeError);
    expect(() => matmul.bind(pool, { ...bad, shape: { m: 4, n: -1, k: 4 } })).toThrow(RangeError);
    expect(() => matmul.bind(pool, { ...bad, shape: { m: 1.5, n: 4, k: 4 } })).toThrow(RangeError);
    // 128 KiB of buffer cannot hold a 4096x4096 operand; caught before the GPU sees it.
    expect(() => matmul.bind(pool, { ...bad, shape: { m: 4096, n: 4096, k: 4096 } })).toThrow(RangeError);
    pool.release(buf);
  });

  /**
   * Deliberately different from stage 1. One workgroup still has 16x16 threads but now
   * produces 64 rows, so the y dimension divides by 64 — 32 rows and 17 rows both fit in a
   * single workgroup row where stage 1 needed two. Getting this wrong dispatches four times
   * the workgroups and silently recomputes the same rows.
   */
  /**
   * Divides by 128 in y, where stage 2 divided by 64 and stage 1 by 16. The interesting
   * cases are 65 and 128 rows: both fit in a single workgroup row here and needed two under
   * stage 2. At m = 65 that workgroup leaves 63 of its 128 rows idle, which is the cost of a
   * tall footprint on short matrices and one of the things stage 3 is being measured for.
   */
  /**
   * The first variant whose x-dimension is not n/16. A workgroup covers 64 rows x 32 columns,
   * so x divides by 32 and y by 64 — every earlier kernel divided x by 16, and carrying that
   * assumption over silently dispatches twice the workgroups in x, each recomputing columns
   * another already wrote.
   */
  it('derives dispatch geometry that covers the output exactly once', () => {
    expect(matmulBlock42Workgroups({ m: 64, n: 32, k: 1 })).toEqual([1, 1, 1]);
    expect(matmulBlock42Workgroups({ m: 32, n: 32, k: 1 })).toEqual([1, 1, 1]);
    expect(matmulBlock42Workgroups({ m: 65, n: 33, k: 1 })).toEqual([2, 2, 1]);
    expect(matmulBlock42Workgroups({ m: 17, n: 129, k: 1 })).toEqual([5, 1, 1]);
    expect(matmulBlock42Workgroups({ m: 1, n: 1, k: 1 })).toEqual([1, 1, 1]);
    expect(matmulBlock42Workgroups({ m: 256, n: 64, k: 1 })).toEqual([2, 4, 1]);
    expect(matmulBlock42Workgroups({ m: 1024, n: 1024, k: 1 })).toEqual([32, 16, 1]);
    expect(matmulFlops({ m: 2, n: 3, k: 5 })).toBe(60);
  });
});
