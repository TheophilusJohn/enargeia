/**
 * Shared plumbing for the simple forward-pass kernels.
 *
 * Every kernel in the fp32 forward pass has the same shape: some read-only storage buffers,
 * one read_write output, one uniform, and a dispatch geometry derived from the dims. Writing
 * that out nine times would be nine chances for the bind group layout and the shader's
 * binding order to drift apart. This keeps pipeline creation in one place where the cache can
 * see it, exactly as a per-kernel wrapper would, and each kernel file stays a declaration of
 * what makes it different.
 *
 * Same reasoning as `matmul_shared.ts`, and the same limit: dispatch geometry is per kernel,
 * because that is the part that is genuinely kernel-specific.
 */

import type { PipelineCache } from '../gpu/pipeline.ts';
import type { BufferPool, PooledBuffer } from '../gpu/pool.ts';

export type BindingKind = 'read' | 'read_write' | 'uniform';

export interface KernelSpec {
  name: string;
  code: string;
  /** In binding order: inputs, then outputs, then the uniform. */
  bindings: readonly BindingKind[];
  workgroupSize: readonly [number, number, number];
  /** Bytes of the uniform struct. Must be a multiple of 16. */
  uniformBytes: number;
}

export type BufferLike = GPUBufferBinding | PooledBuffer;

function resolve(ref: BufferLike): GPUBufferBinding {
  return 'binding' in ref ? ref.binding : ref;
}

export class ComputeKernel {
  readonly spec: KernelSpec;
  readonly bindGroupLayout: GPUBindGroupLayout;
  readonly pipeline: GPUComputePipeline;
  private readonly device: GPUDevice;

  constructor(device: GPUDevice, cache: PipelineCache, spec: KernelSpec) {
    this.device = device;
    this.spec = spec;
    this.bindGroupLayout = cache.bindGroupLayout(`kernel:${spec.name}`, {
      entries: spec.bindings.map((kind, binding) => ({
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type:
            kind === 'uniform'
              ? ('uniform' as const)
              : kind === 'read'
                ? ('read-only-storage' as const)
                : ('storage' as const),
          ...(kind === 'uniform' ? { minBindingSize: spec.uniformBytes } : {}),
        },
      })),
    });
    this.pipeline = cache.pipeline({
      code: spec.code,
      entryPoint: 'main',
      label: spec.name,
      layout: cache.pipelineLayout(`kernel:${spec.name}`, {
        bindGroupLayouts: [this.bindGroupLayout],
      }),
    });
  }

  bindGroup(resources: readonly BufferLike[]): GPUBindGroup {
    if (resources.length !== this.spec.bindings.length) {
      throw new Error(
        `${this.spec.name}: expected ${this.spec.bindings.length} bindings, got ${resources.length}`,
      );
    }
    return this.device.createBindGroup({
      label: this.spec.name,
      layout: this.bindGroupLayout,
      entries: resources.map((resource, binding) => ({ binding, resource: resolve(resource) })),
    });
  }

  /** Allocate this kernel's uniform buffer from the pool. */
  uniform(pool: BufferPool): PooledBuffer {
    return pool.acquire(
      this.spec.uniformBytes,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      `${this.spec.name}.dims`,
    );
  }

  encode(
    pass: GPUComputePassEncoder,
    bindGroup: GPUBindGroup,
    workgroups: readonly [number, number, number],
  ): void {
    // Exceeding maxComputeWorkgroupsPerDimension does not throw here — it invalidates the
    // whole command buffer at submit, and the symptom is a forward pass that produces zeros
    // with no error anywhere. Checking at encode time turns that into a named failure.
    const limit = this.device.limits.maxComputeWorkgroupsPerDimension;
    for (const [axis, count] of workgroups.entries()) {
      if (count > limit) {
        throw new RangeError(
          `${this.spec.name}: ${count} workgroups on axis ${axis} exceeds the device limit of ${limit}`,
        );
      }
    }
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups[0], workgroups[1], workgroups[2]);
  }
}

/** Workgroups needed to cover `counts` items at this kernel's workgroup size. */
export function coverage(
  spec: KernelSpec,
  counts: readonly [number, number, number],
): [number, number, number] {
  return [
    Math.max(1, Math.ceil(counts[0] / spec.workgroupSize[0])),
    Math.max(1, Math.ceil(counts[1] / spec.workgroupSize[1])),
    Math.max(1, Math.ceil(counts[2] / spec.workgroupSize[2])),
  ];
}
