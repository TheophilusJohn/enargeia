/**
 * Dispatch wrapper for the naive fp32 matmul.
 *
 * The wrapper owns the bind group layout and the dispatch geometry; nothing outside
 * `src/kernels` builds a compute pipeline. Bind groups are built once, by `bind()`, and
 * reused across dispatches — the decode loop walks a list of prepared bindings rather
 * than assembling them per token.
 */

import code from './matmul_naive.wgsl?raw';
import type { PipelineCache } from '../gpu/pipeline.ts';
import type { BufferPool } from '../gpu/pool.ts';
import {
  bindMatmul,
  matmulBindGroupLayout,
  matmulPipelineLayout,
  matmulWorkgroups,
  releaseMatmul,
  type MatmulBinding,
  type MatmulInputs,
  type MatmulShape,
} from './matmul_shared.ts';

const KERNEL = 'matmul_naive';

/** Must match `@workgroup_size` in the shader. */
export const MATMUL_NAIVE_WORKGROUP = { x: 16, y: 16 } as const;

export function matmulNaiveWorkgroups(shape: MatmulShape): [number, number, number] {
  return matmulWorkgroups(shape, MATMUL_NAIVE_WORKGROUP.x, MATMUL_NAIVE_WORKGROUP.y);
}

export class MatmulNaive {
  private readonly device: GPUDevice;
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;

  constructor(device: GPUDevice, cache: PipelineCache) {
    this.device = device;
    this.bindGroupLayout = matmulBindGroupLayout(cache);
    this.pipeline = cache.pipeline({
      code,
      entryPoint: 'main',
      label: KERNEL,
      layout: matmulPipelineLayout(cache),
    });
  }

  bind(pool: BufferPool, inputs: MatmulInputs): MatmulBinding {
    return bindMatmul(
      KERNEL,
      this.device,
      pool,
      this.bindGroupLayout,
      inputs,
      matmulNaiveWorkgroups(inputs.shape),
    );
  }

  release(pool: BufferPool, binding: MatmulBinding): void {
    releaseMatmul(pool, binding);
  }

  /** Record into an existing pass. Preferred — passes are worth sharing across kernels. */
  encode(pass: GPUComputePassEncoder, binding: MatmulBinding): void {
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, binding.bindGroup);
    pass.dispatchWorkgroups(...binding.workgroups);
  }

  /** Record into its own pass. For one-off work such as tests and benchmarks. */
  dispatch(encoder: GPUCommandEncoder, binding: MatmulBinding): void {
    const pass = encoder.beginComputePass({ label: KERNEL });
    this.encode(pass, binding);
    pass.end();
  }
}
