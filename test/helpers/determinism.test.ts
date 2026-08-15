/**
 * Tests for the race detector itself.
 *
 * A detector that cannot detect is worse than none, and the GPU path cannot prove this: a
 * correct kernel produces no divergence to find, and a deliberately racy one is not
 * reliably racy on every driver. So the diffing runs on fabricated data where the answer
 * is known.
 */

import { describe, expect, it } from 'vitest';
import type { BufferPool, PipelineCache } from '../../src/gpu/index.ts';
import { checkDeterminism, diffRuns, type DeterminismSubject } from './determinism.ts';
import { gpu } from './gpu.ts';

const run = (...values: number[]) => Float32Array.from(values);

describe('diffRuns', () => {
  it('passes identical runs', () => {
    const report = diffRuns([run(1, 2, 3), run(1, 2, 3), run(1, 2, 3)]);
    expect(report.deterministic).toBe(true);
    expect(report.divergentIndexCount).toBe(0);
    expect(report.firstDivergentIteration).toBeNull();
    expect(report.summary).toMatch(/3 concurrent runs byte-identical/);
  });

  it('names the diverging index and the iteration it first appeared', () => {
    const report = diffRuns([
      run(1, 2, 3, 4),
      run(1, 2, 3, 4),
      run(1, 2, 99, 4), // index 2 diverges at iteration 2
    ]);

    expect(report.deterministic).toBe(false);
    expect(report.divergentIndexCount).toBe(1);
    expect(report.firstDivergentIteration).toBe(2);
    expect(report.divergences).toEqual([{ index: 2, iteration: 2, baseline: 3, observed: 99 }]);
    expect(report.perIteration).toEqual([0, 0, 1]);
  });

  it('reports the earliest iteration per index, not the last', () => {
    const report = diffRuns([run(0, 0), run(5, 0), run(7, 9)]);
    const byIndex = new Map(report.divergences.map((d) => [d.index, d.iteration]));
    expect(byIndex.get(0)).toBe(1);
    expect(byIndex.get(1)).toBe(2);
    expect(report.perIteration).toEqual([0, 1, 2]);
  });

  it('caps the detail list but keeps the full count', () => {
    const baseline = new Float32Array(100);
    const drifted = new Float32Array(100).fill(1);
    const report = diffRuns([baseline, drifted]);

    expect(report.divergentIndexCount).toBe(100);
    expect(report.divergences).toHaveLength(16);
    expect(report.divergences[0].index).toBe(0);
    expect(report.summary).toMatch(/and 84 more/);
  });

  /**
   * Bit comparison, not float equality. NaN !== NaN would otherwise mark every element of
   * a uniformly NaN output as divergent and bury the real signal.
   */
  it('treats identical NaN runs as deterministic', () => {
    const report = diffRuns([run(NaN, 1), run(NaN, 1)]);
    expect(report.deterministic).toBe(true);
  });

  /** -0 and +0 compare equal as floats and differ as bits. Byte-identical means bits. */
  it('catches a sign flip on zero', () => {
    const report = diffRuns([run(0), run(-0)]);
    expect(report.deterministic).toBe(false);
    expect(report.divergences[0].index).toBe(0);
  });

  it('explains what a divergence usually means', () => {
    const report = diffRuns([run(1), run(2)], { label: 'matmul_tiled' });
    expect(report.summary).toMatch(/matmul_tiled: NOT DETERMINISTIC/);
    expect(report.summary).toMatch(/workgroupBarrier/);
  });

  it('warns when the shape was too small to prove anything', () => {
    const clean = [run(1, 2), run(1, 2)];
    expect(diffRuns(clean, { workgroupsPerRun: 4 }).summary).toMatch(/too little concurrency/);
    expect(diffRuns(clean, { workgroupsPerRun: 1024 }).summary).not.toMatch(/warning/);
  });

  it('refuses inputs it cannot compare', () => {
    expect(() => diffRuns([run(1)])).toThrow(RangeError);
    expect(() => diffRuns([run(1, 2), run(1)])).toThrow(/lengths differ/);
  });
});

/**
 * The pure tests above prove the diffing. These prove the GPU path feeding it: that each
 * iteration really gets its own output buffer and its own dispatch.
 *
 * Without this, a plumbing bug — every run reading buffer 0, say — would report
 * "deterministic" forever and the harness would be decorative.
 */
const TAGGED = `
// vec4<u32> rather than a struct with vec3 padding: a vec3 member forces the struct to 32
// bytes, and binding only 16 fails validation silently enough that the output stays zeroed
// and every run matches — which is exactly the vacuous pass this file exists to prevent.
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> tag: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&out)) { return; }
    out[gid.x] = f32(tag.x) + f32(gid.x);
}
`;

/** A subject whose runs differ by construction when `varyPerRun` is set. */
function taggedSubject(
  ctx: { device: GPUDevice; queue: GPUQueue },
  pool: BufferPool,
  cache: PipelineCache,
  elements: number,
  varyPerRun: boolean,
): DeterminismSubject {
  const layout = cache.bindGroupLayout('tagged', {
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  const pipeline = cache.pipeline({
    code: TAGGED,
    layout: cache.pipelineLayout('tagged', { bindGroupLayouts: [layout] }),
  });

  let run = 0;
  return {
    outputBytes: elements * 4,
    workgroupsPerRun: Math.ceil(elements / 64),
    bind(out) {
      const tagValue = varyPerRun ? run++ : 0;
      const tag = pool.acquire(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'tag');
      ctx.queue.writeBuffer(tag.buffer, 0, new Uint32Array([tagValue, 0, 0, 0]));
      const bindGroup = ctx.device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: out.binding },
          { binding: 1, resource: tag.binding },
        ],
      });
      return {
        encode(pass) {
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(Math.ceil(elements / 64));
        },
        release: () => pool.release(tag),
      };
    },
  };
}

describe('checkDeterminism on real dispatches', () => {
  it('reports identical runs of a kernel that cannot vary', async () => {
    const { ctx, pool, cache } = await gpu();
    const report = await checkDeterminism(
      ctx,
      pool,
      taggedSubject(ctx, pool, cache, 4096, false),
      { iterations: 20, label: 'tagged/constant' },
    );
    expect(report.deterministic, report.summary).toBe(true);
    expect(report.iterations).toBe(20);
    expect(report.elements).toBe(4096);
    // Non-vacuous: the kernel really ran. An all-zero output would agree across runs
    // whether or not the dispatch did anything at all.
    expect(Array.from(report.baseline.slice(0, 4))).toEqual([0, 1, 2, 3]);
  });

  it('catches divergence, proving each run gets its own buffer and dispatch', async () => {
    const { ctx, pool, cache } = await gpu();
    const report = await checkDeterminism(
      ctx,
      pool,
      taggedSubject(ctx, pool, cache, 4096, true),
      { iterations: 5, label: 'tagged/varying' },
    );

    expect(report.deterministic).toBe(false);
    // Run i writes i + index, so every element of every run after the first differs.
    expect(report.divergentIndexCount).toBe(4096);
    expect(report.firstDivergentIteration).toBe(1);
    expect(report.perIteration).toEqual([0, 4096, 4096, 4096, 4096]);
    expect(report.divergences[0]).toEqual({
      index: 0,
      iteration: 1,
      baseline: 0,
      observed: 1,
    });
  });
});
