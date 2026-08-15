/**
 * Which adapters are called software, and — the part that matters — which are not.
 *
 * The warning tells a visitor their machine is misconfigured. Firing it on a real GPU is worse
 * than never firing at all, and the false-positive risk is concentrated in one place: browsers
 * redact adapter strings for fingerprinting reasons, and roughly a third of real GPUs lack
 * `shader-f16`. Either fact alone must not be enough.
 *
 * The end-to-end check (`tools/software-check.mjs`) can only exercise the positive path, because
 * it forces the flag on. The negative cases live here, where no GPU is needed to state them.
 */

import { describe, expect, it } from 'vitest';
import { classifySoftware } from '../../src/gpu/device.ts';

const info = (architecture: string, extra: Partial<GPUAdapterInfo> = {}) =>
  ({ vendor: '', architecture, device: '', description: '', ...extra }) as Partial<GPUAdapterInfo>;

describe('classifySoftware', () => {
  it('names the known software rasterizers', () => {
    for (const architecture of ['swiftshader', 'llvmpipe', 'lavapipe', 'warp']) {
      expect(classifySoftware(info(architecture), false)).toBe('yes');
      // Still software when the browser reports f16 support for it, which SwiftShader does.
      expect(classifySoftware(info(architecture), true)).toBe('yes');
    }
  });

  it('finds them in any of the adapter strings, not only architecture', () => {
    expect(classifySoftware(info('', { vendor: 'google', description: 'SwiftShader Device' }), true))
      .toBe('yes');
    expect(classifySoftware(info('', { device: 'llvmpipe (LLVM 15.0.7, 256 bits)' }), true))
      .toBe('yes');
  });

  it('suspects a blank adapter that also lacks f16', () => {
    expect(classifySoftware(info(''), false)).toBe('suspected');
  });

  it('does not warn on a real GPU that merely lacks f16', () => {
    // The common case this must not fire on: about a third of real devices have no shader-f16.
    expect(classifySoftware(info('gen-12lp', { vendor: 'intel' }), false)).toBe('no');
    expect(classifySoftware(info('adreno-7xx', { vendor: 'qualcomm' }), false)).toBe('no');
    expect(classifySoftware(info('rdna-3', { vendor: 'amd' }), false)).toBe('no');
  });

  it('does not warn on a redacted adapter that has f16', () => {
    // Chrome withholds adapter strings in some configurations; f16 is enough to acquit it.
    expect(classifySoftware(info(''), true)).toBe('no');
  });

  it('reports software when forced, which is how the warning is exercised', () => {
    expect(classifySoftware(info('apple-m2', { vendor: 'apple' }), true, true)).toBe('yes');
    expect(classifySoftware(info('apple-m2', { vendor: 'apple' }), true, false)).toBe('no');
  });
});
