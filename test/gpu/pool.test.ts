import { afterAll, describe, expect, it } from 'vitest';
import { BufferPool, readFloats, toBinding } from '../../src/gpu/index.ts';
import { gpu, teardownGPU } from '../helpers/gpu.ts';

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

afterAll(teardownGPU);

async function freshPool(): Promise<BufferPool> {
  const { ctx } = await gpu();
  return new BufferPool(ctx.device, { label: 'unit' });
}

describe('BufferPool', () => {
  it('recycles the same allocation for a repeated request', async () => {
    const pool = await freshPool();
    const first = pool.acquire(1024, STORAGE);
    const handle = first.buffer;
    pool.release(first);

    const second = pool.acquire(1024, STORAGE);
    expect(second.buffer).toBe(handle);
    expect(pool.stats().created).toBe(1);
    expect(pool.stats().reused).toBe(1);
    pool.release(second);
    pool.destroy();
  });

  it('recycles across sizes inside one class but not across classes', async () => {
    const pool = await freshPool();
    const big = pool.acquire(4096, STORAGE);
    pool.release(big);

    // 4096 and 3000 both round up to the 4 KiB class.
    const sameClass = pool.acquire(3000, STORAGE);
    expect(sameClass.buffer).toBe(big.buffer);
    expect(sameClass.capacity).toBe(4096);
    expect(sameClass.size).toBe(3000);
    pool.release(sameClass);

    const nextClass = pool.acquire(4097, STORAGE);
    expect(nextClass.buffer).not.toBe(big.buffer);
    expect(nextClass.capacity).toBe(8192);
    expect(pool.stats().created).toBe(2);
    pool.release(nextClass);
    pool.destroy();
  });

  it('keeps buffers of different usage in different classes', async () => {
    const pool = await freshPool();
    const storage = pool.acquire(256, STORAGE);
    pool.release(storage);

    const uniform = pool.acquire(256, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    expect(uniform.buffer).not.toBe(storage.buffer);
    expect(pool.stats().created).toBe(2);
    pool.release(uniform);
    pool.destroy();
  });

  it('rounds sizes up to 4 bytes and refuses non-positive ones', async () => {
    const pool = await freshPool();
    const one = pool.acquire(1, STORAGE);
    const thirteen = pool.acquire(13, STORAGE);
    expect(one.size).toBe(4);
    expect(thirteen.size).toBe(16);
    expect(() => pool.acquire(0, STORAGE)).toThrow(RangeError);
    expect(() => pool.acquire(-8, STORAGE)).toThrow(RangeError);
    pool.release(thirteen);
    pool.release(one);
    pool.destroy();
  });

  it('binds the requested range, not the whole allocation', async () => {
    // The distinction matters: arrayLength() in WGSL reports the bound range, so a shader
    // handed the full 4 KiB allocation for a 40-byte tensor computes a different answer.
    const pool = await freshPool();
    const buf = pool.acquire(40, STORAGE);
    expect(buf.capacity).toBe(256);
    expect(buf.binding.size).toBe(40);
    expect(toBinding(buf)).toBe(buf.binding);
    pool.release(buf);
    pool.destroy();
  });

  it('catches double release', async () => {
    const pool = await freshPool();
    const buf = pool.acquire(256, STORAGE);
    pool.release(buf);
    expect(() => pool.release(buf)).toThrow(/double release/);
    pool.destroy();
  });

  it('releases everything acquired inside a scope, and nests', async () => {
    const pool = await freshPool();
    const outer = pool.acquire(256, STORAGE);

    pool.beginScope();
    pool.acquire(256, STORAGE);
    pool.beginScope();
    pool.acquire(512, STORAGE);
    pool.endScope();
    expect(pool.stats().liveCount).toBe(2); // outer plus the one in the outer scope
    pool.endScope();

    expect(pool.stats().liveCount).toBe(1);
    pool.release(outer);
    expect(pool.stats().liveCount).toBe(0);
    expect(() => pool.endScope()).toThrow(/without beginScope/);
    pool.destroy();
  });

  it('destroys idle buffers on trim and keeps live ones', async () => {
    const pool = await freshPool();
    const live = pool.acquire(256, STORAGE);
    pool.release(pool.acquire(256, STORAGE));
    expect(pool.stats().idleCount).toBe(1);

    pool.trim();
    expect(pool.stats().idleCount).toBe(0);
    expect(pool.stats().liveCount).toBe(1);
    pool.release(live);
    pool.destroy();
  });

  it('stops retaining once idle bytes pass the cap', async () => {
    const { ctx } = await gpu();
    const pool = new BufferPool(ctx.device, { label: 'capped', maxIdleBytes: 1024 });
    pool.release(pool.acquire(1024, STORAGE));
    expect(pool.stats().idleBytes).toBe(1024);

    // The next release would exceed the cap, so that buffer is destroyed instead.
    pool.release(pool.acquire(512, STORAGE));
    expect(pool.stats().idleBytes).toBe(1024);
    expect(pool.stats().destroyed).toBe(1);
    pool.destroy();
  });

  it('hands back a usable buffer after a readback mapped and unmapped it', async () => {
    const { ctx } = await gpu();
    const pool = await freshPool();
    const source = pool.acquire(64, STORAGE);
    ctx.queue.writeBuffer(source.buffer, 0, new Float32Array(16).fill(2.5));

    const first = await readFloats(ctx, pool, source);
    expect(first[0]).toBe(2.5);

    // The staging buffer is back in the pool and must be re-mappable.
    ctx.queue.writeBuffer(source.buffer, 0, new Float32Array(16).fill(-1));
    const second = await readFloats(ctx, pool, source);
    expect(second[0]).toBe(-1);
    expect(pool.stats().created).toBe(2); // source plus one staging buffer, reused
    pool.release(source);
    pool.destroy();
  });
});
