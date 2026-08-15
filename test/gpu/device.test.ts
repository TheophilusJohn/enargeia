import { describe, expect, it } from 'vitest';
import { DEFAULT_STORAGE_BINDING_SIZE, initGPU } from '../../src/gpu/index.ts';
import { gpu } from '../helpers/gpu.ts';

describe('DeviceProfile', () => {
  it('reports the limits the device actually got, not the defaults', async () => {
    const { ctx } = await gpu();
    const { profile, adapter } = ctx;

    expect(profile.maxBufferSize).toBe(ctx.device.limits.maxBufferSize);
    expect(profile.maxStorageBufferBindingSize).toBe(adapter.limits.maxStorageBufferBindingSize);
    expect(profile.maxComputeInvocationsPerWorkgroup).toBeGreaterThanOrEqual(256);
    expect(profile.storageBindingClamped).toBe(false);
  });

  it('agrees with the device about which features are enabled', async () => {
    const { ctx } = await gpu();
    expect(ctx.profile.f16).toBe(ctx.device.features.has('shader-f16'));
    expect(ctx.profile.timestampQuery).toBe(ctx.device.features.has('timestamp-query'));
    expect(ctx.profile.subgroups).toBe(ctx.device.features.has('subgroups'));
  });

  it('classifies the adapter into a known tier', async () => {
    const { ctx } = await gpu();
    expect(['discrete', 'integrated', 'mobile', 'unknown']).toContain(ctx.profile.tier);
  });

  /**
   * The constrained path is the one that breaks on other people's hardware, so it gets a
   * test rather than a comment. With the clamp on, the fp32 embedding table needs more
   * than one binding on any device.
   */
  it('clamps the storage binding size on request', async () => {
    const clamped = await initGPU({ clampStorageBindingSize: true, label: 'clamped' });
    try {
      expect(clamped.profile.maxStorageBufferBindingSize).toBe(DEFAULT_STORAGE_BINDING_SIZE);
      expect(clamped.profile.storageBindingClamped).toBe(true);
      expect(clamped.profile.adapterStorageBufferBindingSize).toBeGreaterThanOrEqual(
        DEFAULT_STORAGE_BINDING_SIZE,
      );

      const embeddingBytes = 151936 * 896 * 4;
      const bindings = Math.ceil(embeddingBytes / clamped.profile.maxStorageBufferBindingSize);
      expect(bindings).toBeGreaterThan(1);
    } finally {
      clamped.device.destroy();
    }
  });

  it('still allocates a buffer larger than the clamped binding size', async () => {
    // maxBufferSize is left at the adapter maximum on purpose: the embedding table is one
    // allocation read through several bindings, not several allocations.
    const clamped = await initGPU({ clampStorageBindingSize: true, label: 'clamped-alloc' });
    try {
      expect(clamped.profile.maxBufferSize).toBeGreaterThan(
        clamped.profile.maxStorageBufferBindingSize,
      );
    } finally {
      clamped.device.destroy();
    }
  });
});

describe('test environment', () => {
  it('is running on a hardware adapter, not a software fallback', async () => {
    const { ctx } = await gpu();
    console.log(
      `[test] adapter: ${ctx.profile.vendor}/${ctx.profile.architecture} · tier ${ctx.profile.tier} · ` +
        `f16 ${ctx.profile.f16} · timestamps ${ctx.profile.timestampQuery}`,
    );
    // SwiftShader and llvmpipe are correct but ~100x slower; a suite that silently runs on
    // one proves the shader compiles, not that it runs on a GPU.
    expect(`${ctx.profile.vendor} ${ctx.profile.architecture} ${ctx.profile.description}`)
      .not.toMatch(/swiftshader|llvmpipe|warp/i);
  });
});
