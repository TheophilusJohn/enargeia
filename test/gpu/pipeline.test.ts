import { afterAll, describe, expect, it } from 'vitest';
import { PipelineCache } from '../../src/gpu/index.ts';
import { gpu, teardownGPU } from '../helpers/gpu.ts';

afterAll(teardownGPU);

const TRIVIAL = `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() { out[0] = 1.0; }
`;

const WITH_OVERRIDE = `
override scale: f32 = 1.0;
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() { out[0] = scale; }
@compute @workgroup_size(1)
fn other() { out[0] = scale * 2.0; }
`;

async function freshCache(): Promise<PipelineCache> {
  const { ctx } = await gpu();
  return new PipelineCache(ctx.device);
}

describe('PipelineCache', () => {
  it('returns one pipeline and one module for the same source', async () => {
    const cache = await freshCache();
    const first = cache.pipeline({ code: TRIVIAL });
    const second = cache.pipeline({ code: TRIVIAL });

    expect(second).toBe(first);
    expect(cache.stats().pipelines).toBe(1);
    expect(cache.stats().modules).toBe(1);
    expect(cache.stats().hits).toBe(1);
  });

  it('keys on source, so a changed shader gets a new pipeline', async () => {
    const cache = await freshCache();
    const first = cache.pipeline({ code: TRIVIAL });
    const edited = cache.pipeline({ code: TRIVIAL.replace('1.0', '2.0') });

    expect(edited).not.toBe(first);
    expect(cache.stats().pipelines).toBe(2);
  });

  it('keys on entry point and on override constants', async () => {
    const cache = await freshCache();
    const main = cache.pipeline({ code: WITH_OVERRIDE, entryPoint: 'main' });
    const other = cache.pipeline({ code: WITH_OVERRIDE, entryPoint: 'other' });
    const scaled = cache.pipeline({
      code: WITH_OVERRIDE,
      entryPoint: 'main',
      constants: { scale: 4 },
    });

    expect(other).not.toBe(main);
    expect(scaled).not.toBe(main);
    expect(cache.stats().pipelines).toBe(3);
    // All three share one compiled module.
    expect(cache.stats().modules).toBe(1);
  });

  it('treats constants as unordered', async () => {
    const cache = await freshCache();
    const a = cache.pipeline({ code: WITH_OVERRIDE, constants: { scale: 2 } });
    const b = cache.pipeline({ code: WITH_OVERRIDE, constants: { scale: 2 } });
    expect(b).toBe(a);
    expect(cache.stats().pipelines).toBe(1);
  });

  it('distinguishes explicit layouts from auto and from each other', async () => {
    const { ctx } = await gpu();
    const cache = new PipelineCache(ctx.device);
    const bgl = cache.bindGroupLayout('trivial', {
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const layout = cache.pipelineLayout('trivial', { bindGroupLayouts: [bgl] });

    const auto = cache.pipeline({ code: TRIVIAL });
    const explicit = cache.pipeline({ code: TRIVIAL, layout });
    const explicitAgain = cache.pipeline({ code: TRIVIAL, layout });

    expect(explicit).not.toBe(auto);
    expect(explicitAgain).toBe(explicit);
    expect(cache.bindGroupLayout('trivial', { entries: [] })).toBe(bgl); // memoized by key
    expect(cache.stats().pipelines).toBe(2);
  });

  it('reports WGSL diagnostics instead of an opaque failure', async () => {
    const cache = await freshCache();
    const broken = `
      @compute @workgroup_size(1)
      fn main() { let x: f32 = "not a number"; }
    `;
    await expect(cache.pipelineAsync({ code: broken, label: 'broken' })).rejects.toThrow(
      /broken.*failed to compile/s,
    );
  });

  it('compiles asynchronously into the same cache the sync path reads', async () => {
    const cache = await freshCache();
    const asyncPipeline = await cache.pipelineAsync({ code: TRIVIAL });
    expect(cache.pipeline({ code: TRIVIAL })).toBe(asyncPipeline);
  });

  it('drops everything on clear', async () => {
    const cache = await freshCache();
    cache.pipeline({ code: TRIVIAL });
    cache.clear();
    expect(cache.stats().pipelines).toBe(0);
    expect(cache.stats().modules).toBe(0);
  });
});
