/**
 * Dispatch wrapper for the 2D register-blocked fp32 matmul: 4 rows x 2 columns per thread.
 *
 * Workgroup output footprint is 64 rows by 32 columns, so the dispatch divides by 32 in x
 * and 64 in y — the first variant in this family whose x-dimension is not simply n/16.
 * Shared memory is 6 KiB, between stage 2's 5 KiB and stage 3's 9 KiB.
 */

import code from './matmul_block42.wgsl?raw';
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

const KERNEL = 'matmul_block42';

/** Threads per workgroup. Must match `@workgroup_size` in the shader. */
export const MATMUL_BLOCK42_WORKGROUP = { x: 16, y: 16 } as const;

/** Outputs each thread accumulates, as a 2D block. */
export const MATMUL_BLOCK42_BLOCK = { rows: 4, cols: 2 } as const;

/** Output footprint of one workgroup: 64 rows by 32 columns. */
export const MATMUL_BLOCK42_TILE = {
  rows: MATMUL_BLOCK42_WORKGROUP.y * MATMUL_BLOCK42_BLOCK.rows,
  cols: MATMUL_BLOCK42_WORKGROUP.x * MATMUL_BLOCK42_BLOCK.cols,
} as const;

/** Shared memory the kernel reserves: a 64x16 tile of A and a 16x32 tile of B. */
export const MATMUL_BLOCK42_WORKGROUP_BYTES = (64 * 16 + 16 * 32) * 4;

/** Shared-memory loads per multiply-add in the inner loop. Recorded for the ablation table. */
export const MATMUL_BLOCK42_LOADS_PER_MAC =
  (MATMUL_BLOCK42_BLOCK.rows + MATMUL_BLOCK42_BLOCK.cols) /
  (MATMUL_BLOCK42_BLOCK.rows * MATMUL_BLOCK42_BLOCK.cols);

export function matmulBlock42Workgroups(shape: MatmulShape): [number, number, number] {
  return matmulWorkgroups(shape, MATMUL_BLOCK42_TILE.cols, MATMUL_BLOCK42_TILE.rows);
}

export class MatmulBlock42 {
  private readonly device: GPUDevice;
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;

  constructor(device: GPUDevice, cache: PipelineCache) {
    if (device.limits.maxComputeWorkgroupStorageSize < MATMUL_BLOCK42_WORKGROUP_BYTES) {
      throw new Error(
        `${KERNEL}: needs ${MATMUL_BLOCK42_WORKGROUP_BYTES} bytes of workgroup storage, ` +
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
      matmulBlock42Workgroups(inputs.shape),
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
