/**
 * Dispatch wrapper for the tiled fp32 matmul, stage 1.
 *
 * One thread per output element, 16x16 workgroup, no coarsening — so the dispatch geometry
 * is identical to the naive kernel and the two are directly comparable. Stages 2 and 3
 * change that and will compute their own geometry.
 */

import code from './matmul_tiled.wgsl?raw';
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

const KERNEL = 'matmul_tiled';

/** Must match `@workgroup_size` and the `TILE` constant in the shader. */
export const MATMUL_TILED_WORKGROUP = { x: 16, y: 16 } as const;

/** Shared memory the kernel reserves: two 16x16 fp32 tiles. */
export const MATMUL_TILED_WORKGROUP_BYTES = 2 * 16 * 16 * 4;

export function matmulTiledWorkgroups(shape: MatmulShape): [number, number, number] {
  return matmulWorkgroups(shape, MATMUL_TILED_WORKGROUP.x, MATMUL_TILED_WORKGROUP.y);
}

export class MatmulTiled {
  private readonly device: GPUDevice;
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;

  constructor(device: GPUDevice, cache: PipelineCache) {
    if (device.limits.maxComputeWorkgroupStorageSize < MATMUL_TILED_WORKGROUP_BYTES) {
      throw new Error(
        `${KERNEL}: needs ${MATMUL_TILED_WORKGROUP_BYTES} bytes of workgroup storage, ` +
          `device allows ${device.limits.maxComputeWorkgroupStorageSize}`,
      );
    }
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
      matmulTiledWorkgroups(inputs.shape),
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
