/**
 * Kernel benchmark runner.
 *
 * Takes anything that can record itself into a compute pass and reports throughput:
 * warmup passes that are thrown away, then a timed run of N iterations.
 *
 * The headline number is wall clock around submit plus `onSubmittedWorkDone()`, measured
 * over a batch of iterations — the same method that produced the baseline in BENCH.md, so
 * the numbers stay comparable. It includes per-dispatch JavaScript and queue overhead,
 * which is honest for a decode loop that pays that overhead 170 times per token.
 *
 * When `timestamp-query` is available, a second pass measures GPU-side time per
 * dispatch. That number excludes submission overhead and is the one to use when asking
 * whether a shader change helped; the gap between the two is the overhead itself.
 */

import type { GPUContext } from '../gpu/device.ts';
import type { BufferPool } from '../gpu/pool.ts';

export interface BenchCase {
  name: string;
  /** Floating-point operations in one iteration. For matmul, 2*m*n*k. */
  flopsPerIter: number;
  /** Bytes moved in one iteration, if known. Reported as GB/s. */
  bytesPerIter?: number;
  /** Record one iteration's work. Called once per iteration, into a fresh pass. */
  encode(pass: GPUComputePassEncoder): void;
}

export interface BenchOptions {
  /** Iterations run and discarded before timing. Default 2. */
  warmup?: number;
  /** Timed iterations. Default 10. */
  iters?: number;
  /** Also measure GPU-side time. Defaults to whether the device supports it. */
  timestamps?: boolean;
}

export interface Throughput {
  meanMs: number;
  gflops: number;
  gbPerSec?: number;
}

export interface GPUThroughput extends Throughput {
  minMs: number;
  medianMs: number;
  /** Per-iteration GPU times, ms. */
  samples: number[];
}

export interface BenchResult {
  name: string;
  iters: number;
  warmup: number;
  flopsPerIter: number;
  bytesPerIter?: number;
  /** Wall clock, including submission overhead. The headline number. */
  wall: Throughput;
  /** GPU-side pass duration. Present only with `timestamp-query`. */
  gpu?: GPUThroughput;
}

export async function benchmark(
  ctx: GPUContext,
  pool: BufferPool,
  bench: BenchCase,
  options: BenchOptions = {},
): Promise<BenchResult> {
  const warmup = options.warmup ?? 2;
  const iters = options.iters ?? 10;
  if (iters < 1) throw new RangeError('benchmark: iters must be at least 1');

  const { device, queue } = ctx;

  for (let i = 0; i < warmup; i++) submitIteration(device, bench);
  await queue.onSubmittedWorkDone();

  const start = performance.now();
  for (let i = 0; i < iters; i++) submitIteration(device, bench);
  await queue.onSubmittedWorkDone();
  const wallMs = (performance.now() - start) / iters;

  const result: BenchResult = {
    name: bench.name,
    iters,
    warmup,
    flopsPerIter: bench.flopsPerIter,
    bytesPerIter: bench.bytesPerIter,
    wall: throughput(wallMs, bench),
  };

  const useTimestamps = options.timestamps ?? ctx.profile.timestampQuery;
  if (useTimestamps && ctx.profile.timestampQuery) {
    // Timing the timestamped run separately keeps the wall number free of any
    // perturbation from the query writes.
    const samples = await measureGPUTime(ctx, pool, bench, iters);
    if (samples) {
      const median = percentile(samples, 0.5);
      result.gpu = {
        ...throughput(median, bench),
        meanMs: mean(samples),
        minMs: Math.min(...samples),
        medianMs: median,
        samples,
      };
    }
  }

  return result;
}

function submitIteration(device: GPUDevice, bench: BenchCase): void {
  const encoder = device.createCommandEncoder({ label: bench.name });
  const pass = encoder.beginComputePass({ label: bench.name });
  bench.encode(pass);
  pass.end();
  device.queue.submit([encoder.finish()]);
}

/**
 * Per-iteration GPU duration in ms, or null if the device returned unusable timestamps.
 * Chrome quantizes these to 100 microseconds by default, which is fine for kernels in the
 * millisecond range and useless below about 10 microseconds.
 */
async function measureGPUTime(
  ctx: GPUContext,
  pool: BufferPool,
  bench: BenchCase,
  iters: number,
): Promise<number[] | null> {
  const { device, queue } = ctx;
  const queryCount = iters * 2;
  const byteCount = queryCount * 8;

  const querySet = device.createQuerySet({ type: 'timestamp', count: queryCount });
  const resolve = pool.acquire(
    byteCount,
    GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    'bench.resolve',
  );
  const readback = pool.acquire(
    byteCount,
    GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    'bench.readback',
  );

  try {
    for (let i = 0; i < iters; i++) {
      const encoder = device.createCommandEncoder({ label: `${bench.name}/timed` });
      const pass = encoder.beginComputePass({
        label: `${bench.name}/timed`,
        timestampWrites: {
          querySet,
          beginningOfPassWriteIndex: i * 2,
          endOfPassWriteIndex: i * 2 + 1,
        },
      });
      bench.encode(pass);
      pass.end();
      queue.submit([encoder.finish()]);
    }

    const encoder = device.createCommandEncoder({ label: `${bench.name}/resolve` });
    encoder.resolveQuerySet(querySet, 0, queryCount, resolve.buffer, 0);
    encoder.copyBufferToBuffer(resolve.buffer, 0, readback.buffer, 0, byteCount);
    queue.submit([encoder.finish()]);

    // One readback, outside any decode loop. This is exactly the cost the decode loop
    // exists to avoid, which is why timing lives here and not in the runtime.
    await readback.buffer.mapAsync(GPUMapMode.READ);
    const stamps = new BigInt64Array(readback.buffer.getMappedRange().slice(0));
    readback.buffer.unmap();

    const samples: number[] = [];
    for (let i = 0; i < iters; i++) {
      const ns = stamps[i * 2 + 1] - stamps[i * 2];
      if (ns <= 0n) return null; // unsupported or disabled at runtime
      samples.push(Number(ns) / 1e6);
    }
    return samples;
  } catch (error) {
    console.warn(`[bench] timestamp query failed, wall clock only: ${String(error)}`);
    return null;
  } finally {
    querySet.destroy();
    pool.release(readback);
    pool.release(resolve);
  }
}

function throughput(ms: number, bench: BenchCase): Throughput {
  const seconds = ms / 1000;
  return {
    meanMs: ms,
    gflops: bench.flopsPerIter / seconds / 1e9,
    gbPerSec: bench.bytesPerIter ? bench.bytesPerIter / seconds / 1e9 : undefined,
  };
}

function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function percentile(xs: number[], p: number): number {
  const sorted = [...xs].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
}

export function formatBenchResult(r: BenchResult): string {
  const lines = [
    `${r.name}  ${r.wall.gflops.toFixed(1)} GFLOP/s  (${r.wall.meanMs.toFixed(2)} ms/iter, wall, ${r.iters} iters after ${r.warmup} warmup)`,
  ];
  if (r.wall.gbPerSec !== undefined) {
    lines.push(`  wall bandwidth ${r.wall.gbPerSec.toFixed(1)} GB/s`);
  }
  if (r.gpu) {
    lines.push(
      `  gpu ${r.gpu.gflops.toFixed(1)} GFLOP/s  (median ${r.gpu.medianMs.toFixed(3)} ms, min ${r.gpu.minMs.toFixed(3)} ms)`,
    );
  }
  return lines.join('\n');
}
