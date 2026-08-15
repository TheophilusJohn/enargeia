/**
 * Adapts a matmul kernel wrapper to the generic determinism harness.
 *
 * The harness knows nothing about matmul; this is the thin piece that binds a shape and a
 * pair of operands to whichever output buffer the harness hands over. Any kernel gets its
 * own version of this — the shared part is the concurrency and the diffing, not the shape.
 */

import type { BufferPool, PooledBuffer } from '../../src/gpu/index.ts';
import type { MatmulShape } from '../../src/kernels/matmul_shared.ts';
import { randomFloats } from '../reference/rng.ts';
import type { DeterminismSubject } from './determinism.ts';

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

/**
 * Big enough that a race has somewhere to happen. A 512x512 output at a 16x16 workgroup is
 * 1024 workgroups per run, so dozens are resident at once on any real GPU, and k=512 gives
 * a tiled kernel 32 tile iterations per output — 32 chances to overwrite shared memory a
 * slow thread is still reading.
 */
export const RACE_HUNT_SHAPE: MatmulShape = { m: 512, n: 512, k: 512 };

/**
 * The minimal surface the harness needs from a matmul wrapper, generic over the wrapper's
 * own binding type so tiled and naive both satisfy it without casts.
 */
export interface MatmulLike<Binding> {
  bind(
    pool: BufferPool,
    inputs: { a: PooledBuffer; b: PooledBuffer; out: PooledBuffer; shape: MatmulShape },
  ): Binding & { workgroups: readonly [number, number, number] };
  encode(pass: GPUComputePassEncoder, binding: Binding): void;
  release(pool: BufferPool, binding: Binding): void;
}

export interface MatmulSubjectHandle {
  subject: DeterminismSubject;
  /** Releases the operands. Call after the harness returns. */
  dispose(): void;
}

export function matmulSubject<Binding>(
  queue: GPUQueue,
  pool: BufferPool,
  matmul: MatmulLike<Binding>,
  shape: MatmulShape,
  seed = 101,
): MatmulSubjectHandle {
  const a = randomFloats(shape.m * shape.k, seed);
  const b = randomFloats(shape.k * shape.n, seed + 1);
  const bufA = pool.acquire(a.byteLength, STORAGE, 'race.a');
  const bufB = pool.acquire(b.byteLength, STORAGE, 'race.b');
  queue.writeBuffer(bufA.buffer, 0, a);
  queue.writeBuffer(bufB.buffer, 0, b);

  // Filled in by the first bind. The harness reads it when it builds the report, which is
  // after every run has been bound, so there is nothing to probe for up front.
  const subject: DeterminismSubject = {
    outputBytes: shape.m * shape.n * 4,
    bind(out: PooledBuffer) {
      const binding = matmul.bind(pool, { a: bufA, b: bufB, out, shape });
      subject.workgroupsPerRun = binding.workgroups[0] * binding.workgroups[1] * binding.workgroups[2];
      return {
        encode: (pass: GPUComputePassEncoder) => matmul.encode(pass, binding),
        release: () => matmul.release(pool, binding),
      };
    },
  };

  return {
    subject,
    dispose() {
      pool.release(bufB);
      pool.release(bufA);
    },
  };
}
