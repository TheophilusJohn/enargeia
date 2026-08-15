/**
 * One device per test file. Device creation costs tens of milliseconds and every buffer,
 * pipeline, and bind group is tied to the device that made it, so tests share one rather
 * than each paying for their own.
 */

import { BufferPool, PipelineCache, initGPU, type GPUContext } from '../../src/gpu/index.ts';

export interface TestHarness {
  ctx: GPUContext;
  pool: BufferPool;
  cache: PipelineCache;
}

let harness: TestHarness | null = null;

export async function gpu(): Promise<TestHarness> {
  if (!harness) {
    const ctx = await initGPU({ label: 'test' });
    harness = {
      ctx,
      pool: new BufferPool(ctx.device, { label: 'test' }),
      cache: new PipelineCache(ctx.device),
    };
  }
  return harness;
}

export async function teardownGPU(): Promise<void> {
  if (!harness) return;
  harness.pool.destroy();
  harness.cache.clear();
  harness.ctx.device.destroy();
  harness = null;
}

/**
 * Fail a test on any validation error raised inside `fn`, rather than letting it surface
 * as an unrelated console message after the assertion has already passed.
 */
export async function expectNoGPUError<T>(device: GPUDevice, fn: () => Promise<T> | T): Promise<T> {
  device.pushErrorScope('validation');
  device.pushErrorScope('out-of-memory');
  const result = await fn();
  const oom = await device.popErrorScope();
  const validation = await device.popErrorScope();
  if (validation) throw new Error(`validation error: ${validation.message}`);
  if (oom) throw new Error(`out-of-memory: ${oom.message}`);
  return result;
}

export function bytesEqual(a: Float32Array, b: Float32Array): boolean {
  if (a.length !== b.length) return false;
  const x = new Uint32Array(a.buffer, a.byteOffset, a.length);
  const y = new Uint32Array(b.buffer, b.byteOffset, b.length);
  for (let i = 0; i < x.length; i++) {
    if (x[i] !== y[i]) return false;
  }
  return true;
}
