/**
 * Copying GPU memory back to the CPU.
 *
 * `mapAsync` costs a round trip of roughly a millisecond, which is why the decode loop
 * gets exactly one readback per token — the sampled token ID — and nothing else. This
 * module exists for tests, benchmarks, and that one token. Anything that calls it in a
 * loop is a bug in the caller, not a slow function.
 */

import type { BufferPool, PooledBuffer } from './pool.ts';

export interface ReadbackTarget {
  device: GPUDevice;
  queue: GPUQueue;
}

/** Raw bytes of a pooled buffer's bound range. */
export async function readBuffer(
  ctx: ReadbackTarget,
  pool: BufferPool,
  source: PooledBuffer,
): Promise<ArrayBuffer> {
  const staging = pool.acquire(
    source.size,
    GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    'readback',
  );
  const encoder = ctx.device.createCommandEncoder({ label: 'readback' });
  encoder.copyBufferToBuffer(source.buffer, 0, staging.buffer, 0, source.size);
  ctx.queue.submit([encoder.finish()]);

  await staging.buffer.mapAsync(GPUMapMode.READ, 0, source.size);
  // slice() before unmap: the mapped range is detached the moment the buffer is unmapped.
  const bytes = staging.buffer.getMappedRange(0, source.size).slice(0);
  staging.buffer.unmap();
  pool.release(staging);
  return bytes;
}

export async function readFloats(
  ctx: ReadbackTarget,
  pool: BufferPool,
  source: PooledBuffer,
): Promise<Float32Array> {
  return new Float32Array(await readBuffer(ctx, pool, source));
}
