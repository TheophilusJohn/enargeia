/**
 * Plumbing common to every matmul variant.
 *
 * The shaders differ — that is the whole point of having several — but the operand
 * contract does not: three row-major fp32 buffers and an `m, n, k` uniform, in that order.
 * Keeping the shape validation, the Dims uniform, and the bind group layout here means a
 * new variant is a shader plus a dispatch geometry, and means the four variants cannot
 * drift apart on what counts as a legal shape.
 *
 * All variants share one bind group layout, so a binding built for one can be dispatched
 * through another. That is what makes an honest A/B benchmark possible: same buffers, same
 * data, same bind group, only the pipeline changes. Dispatch geometry is *not* shared —
 * each wrapper computes its own, because thread coarsening changes it.
 */

import type { PipelineCache } from '../gpu/pipeline.ts';
import type { BufferPool, BufferRef, PooledBuffer } from '../gpu/pool.ts';
import { toBinding } from '../gpu/pool.ts';

/** Four u32: m, n, k, and padding to the 16-byte uniform alignment. */
export const MATMUL_DIMS_BYTES = 16;

/** Shared by every variant, so bind groups are portable between them. */
const LAYOUT_KEY = 'matmul';

export interface MatmulShape {
  /** rows of A and C */
  m: number;
  /** columns of B and C */
  n: number;
  /** reduction extent: columns of A, rows of B */
  k: number;
}

export interface MatmulInputs {
  /** A, row-major, m x k */
  a: BufferRef;
  /** B, row-major, k x n */
  b: BufferRef;
  /** C, row-major, m x n */
  out: BufferRef;
  shape: MatmulShape;
}

export interface MatmulBinding {
  readonly shape: MatmulShape;
  readonly bindGroup: GPUBindGroup;
  readonly workgroups: readonly [number, number, number];
  /** The Dims uniform. Owned by the binding, returned to the pool on release. */
  readonly dims: PooledBuffer;
}

/** Multiply-add counted as two flops, the convention every matmul number is quoted in. */
export function matmulFlops(shape: MatmulShape): number {
  return 2 * shape.m * shape.n * shape.k;
}

/**
 * Workgroups needed to cover the output when each one produces `tileN` columns by `tileM`
 * rows. For a kernel with one thread per output element these are the workgroup
 * dimensions; for a coarsened kernel they are the workgroup's output footprint, which is
 * larger.
 */
export function matmulWorkgroups(
  shape: MatmulShape,
  tileN: number,
  tileM: number,
): [number, number, number] {
  return [Math.ceil(shape.n / tileN), Math.ceil(shape.m / tileM), 1];
}

export function matmulBindGroupLayout(cache: PipelineCache): GPUBindGroupLayout {
  return cache.bindGroupLayout(LAYOUT_KEY, {
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform', minBindingSize: MATMUL_DIMS_BYTES },
      },
    ],
  });
}

export function matmulPipelineLayout(cache: PipelineCache): GPUPipelineLayout {
  return cache.pipelineLayout(LAYOUT_KEY, {
    bindGroupLayouts: [matmulBindGroupLayout(cache)],
  });
}

/**
 * Validate, write the Dims uniform, build the bind group. The shape is fixed for the life
 * of the binding — a new shape means a new binding, which is what lets the decode loop
 * walk a prepared list instead of assembling one per token.
 */
export function bindMatmul(
  kernel: string,
  device: GPUDevice,
  pool: BufferPool,
  layout: GPUBindGroupLayout,
  inputs: MatmulInputs,
  workgroups: readonly [number, number, number],
): MatmulBinding {
  const { shape } = inputs;
  assertMatmulShape(kernel, shape);

  const a = toBinding(inputs.a);
  const b = toBinding(inputs.b);
  const out = toBinding(inputs.out);
  assertExtent(kernel, 'a', a, shape.m * shape.k);
  assertExtent(kernel, 'b', b, shape.k * shape.n);
  assertExtent(kernel, 'out', out, shape.m * shape.n);

  const dims = pool.acquire(
    MATMUL_DIMS_BYTES,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    `${kernel}.dims`,
  );
  device.queue.writeBuffer(dims.buffer, 0, new Uint32Array([shape.m, shape.n, shape.k, 0]));

  const bindGroup = device.createBindGroup({
    label: kernel,
    layout,
    entries: [
      { binding: 0, resource: a },
      { binding: 1, resource: b },
      { binding: 2, resource: out },
      { binding: 3, resource: dims.binding },
    ],
  });

  return { shape, bindGroup, workgroups, dims };
}

export function releaseMatmul(pool: BufferPool, binding: MatmulBinding): void {
  pool.release(binding.dims);
}

export function assertMatmulShape(kernel: string, shape: MatmulShape): void {
  for (const axis of ['m', 'n', 'k'] as const) {
    const value = shape[axis];
    if (!Number.isInteger(value) || value <= 0) {
      throw new RangeError(`${kernel}: shape.${axis} must be a positive integer, got ${value}`);
    }
  }
  // u32 indices in the shader: row * k + i must stay representable.
  const largest = Math.max(shape.m * shape.k, shape.k * shape.n, shape.m * shape.n);
  if (largest > 0xffffffff) {
    throw new RangeError(`${kernel}: ${shape.m}x${shape.n}x${shape.k} overflows u32 indexing`);
  }
}

/** Bindings usually carry an explicit size; when they do, catch the shape mismatch here. */
function assertExtent(
  kernel: string,
  name: string,
  binding: GPUBufferBinding,
  elements: number,
): void {
  const size = binding.size ?? binding.buffer.size - (binding.offset ?? 0);
  const needed = elements * 4;
  if (size < needed) {
    throw new RangeError(`${kernel}: ${name} binds ${size} bytes, needs ${needed}`);
  }
}
