/**
 * Dispatch wrapper for the tiled fp32 matmul, stage 3: eight outputs per thread.
 *
 * Workgroup output footprint is 128 rows by 16 columns, so the dispatch y-dimension divides
 * by 128. Shared memory per workgroup is 9 KiB, which is the number to watch — it is what
 * limits how many workgroups a core can hold at once.
 */

import code from './matmul_tiled8.wgsl?raw';
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

const KERNEL = 'matmul_tiled8';

/** Threads per workgroup. Must match `@workgroup_size` in the shader. */
export const MATMUL_TILED8_WORKGROUP = { x: 16, y: 16 } as const;

/** Outputs each thread accumulates, stacked in the row direction. */
export const MATMUL_TILED8_ROWS_PER_THREAD = 8;

/** Output footprint of one workgroup: 128 rows by 16 columns. */
export const MATMUL_TILED8_TILE = {
  rows: MATMUL_TILED8_WORKGROUP.y * MATMUL_TILED8_ROWS_PER_THREAD,
  cols: MATMUL_TILED8_WORKGROUP.x,
} as const;

/** Shared memory the kernel reserves: a 128x16 tile of A and a 16x16 tile of B. */
export const MATMUL_TILED8_WORKGROUP_BYTES = (128 * 16 + 16 * 16) * 4;

export function matmulTiled8Workgroups(shape: MatmulShape): [number, number, number] {
  return matmulWorkgroups(shape, MATMUL_TILED8_TILE.cols, MATMUL_TILED8_TILE.rows);
}

export class MatmulTiled8 {
  private readonly device: GPUDevice;
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;

  constructor(device: GPUDevice, cache: PipelineCache) {
    if (device.limits.maxComputeWorkgroupStorageSize < MATMUL_TILED8_WORKGROUP_BYTES) {
      throw new Error(
        `${KERNEL}: needs ${MATMUL_TILED8_WORKGROUP_BYTES} bytes of workgroup storage, ` +
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
      matmulTiled8Workgroups(inputs.shape),
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
