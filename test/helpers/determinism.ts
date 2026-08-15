/**
 * Race hunting.
 *
 * Greedy decode on a fixed prompt must produce byte-identical output across runs. When it
 * does not, the cause is almost always a missing `workgroupBarrier()` — a fast thread
 * overwriting shared memory while a slow one is still reading the previous tile. That bug
 * is intermittent, device-dependent, and invisible to a single-run comparison against a
 * CPU reference, because any one run is usually correct.
 *
 * Two things make a race show up, and this harness does both:
 *
 * 1. **Contention.** Every iteration writes to its own output buffer, so no two dispatches
 *    have a resource hazard and the driver is free to run them concurrently. They go into
 *    one compute pass in one submit, with no CPU synchronization in between — reading each
 *    result before launching the next would serialize the very overlap being tested.
 * 2. **Repetition.** A race that shows up one run in twenty on this machine shows up far
 *    more often on someone else's, so twenty runs is the floor, not a thorough search.
 *
 * Comparison is on bit patterns, not float equality: that is what "byte-identical" means,
 * and it keeps NaN from reading as a divergence in every run and -0 from silently matching
 * +0.
 */

import { readFloats, type BufferPool, type PooledBuffer } from '../../src/gpu/index.ts';
import type { ReadbackTarget } from '../../src/gpu/readback.ts';

export const DEFAULT_ITERATIONS = 20;

/** Most divergent elements described individually before the report starts counting only. */
const MAX_DETAIL = 16;

/** Refuse to allocate more than this across all iterations. */
const MAX_TOTAL_BYTES = 512 * 1024 * 1024;

const OUTPUT_USAGE =
  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

/** One bound, encodable execution of the kernel under test. */
export interface DeterminismRun {
  encode(pass: GPUComputePassEncoder): void;
  /** Return anything the binding acquired — a uniform buffer, usually. */
  release?(): void;
}

/**
 * A kernel under test, described in the only two terms the harness needs: how big its
 * output is, and how to point it at a given output buffer. Everything kernel-specific —
 * shape, inputs, workgroup geometry — lives inside the closure.
 */
export interface DeterminismSubject {
  /** Bytes of output one run produces. */
  outputBytes: number;
  bind(out: PooledBuffer): DeterminismRun;
  /**
   * Workgroups one run dispatches, if known. Used only to warn when a shape is too small
   * to produce the concurrency a race needs — it does not affect the verdict.
   */
  workgroupsPerRun?: number;
}

export interface DeterminismOptions {
  /** Default 20. */
  iterations?: number;
  /** Name used in the report. */
  label?: string;
  /**
   * Workgroups below which the shape is called out as too small to hunt races. Default 64
   * — enough to occupy several compute units at once on any real GPU.
   */
  minWorkgroups?: number;
}

export interface Divergence {
  /** Element index into the output. */
  index: number;
  /** First iteration at which this element differed from the baseline run. */
  iteration: number;
  baseline: number;
  observed: number;
}

export interface DeterminismReport {
  deterministic: boolean;
  label: string;
  iterations: number;
  elements: number;
  /** Distinct element indices that differed in at least one iteration. */
  divergentIndexCount: number;
  /** Earliest iteration that differed from run 0, or null when all runs agree. */
  firstDivergentIteration: number | null;
  /** Up to 16 examples, lowest index first. */
  divergences: Divergence[];
  /** Divergent element count per iteration; entry 0 is the baseline and always 0. */
  perIteration: number[];
  /**
   * The first run's output. Check it: a kernel that never dispatched leaves a cleared
   * buffer, and runs of zeros agree with each other perfectly.
   */
  baseline: Float32Array;
  /** Ready to pass as a test assertion message. */
  summary: string;
}

/**
 * Run the kernel `iterations` times concurrently and compare every result against the
 * first.
 */
export async function checkDeterminism(
  ctx: ReadbackTarget,
  pool: BufferPool,
  subject: DeterminismSubject,
  options: DeterminismOptions = {},
): Promise<DeterminismReport> {
  const iterations = options.iterations ?? DEFAULT_ITERATIONS;
  const label = options.label ?? 'kernel';
  if (iterations < 2) {
    throw new RangeError('checkDeterminism: at least 2 iterations are needed to compare');
  }
  if (subject.outputBytes * iterations > MAX_TOTAL_BYTES) {
    throw new RangeError(
      `checkDeterminism: ${iterations} x ${subject.outputBytes} bytes exceeds the ` +
        `${MAX_TOTAL_BYTES} byte budget — lower the iteration count or shrink the shape`,
    );
  }

  const outputs: PooledBuffer[] = [];
  const runs: DeterminismRun[] = [];
  try {
    for (let i = 0; i < iterations; i++) {
      const out = pool.acquire(subject.outputBytes, OUTPUT_USAGE, `determinism.out${i}`);
      outputs.push(out);
      runs.push(subject.bind(out));
    }

    const encoder = ctx.device.createCommandEncoder({ label: `determinism/${label}` });
    // Clear first: a recycled buffer holds the previous run's results, and a kernel that
    // fails to write an element would otherwise match by accident.
    for (const out of outputs) encoder.clearBuffer(out.buffer, 0, out.size);

    const pass = encoder.beginComputePass({ label: `determinism/${label}` });
    for (const run of runs) run.encode(pass);
    pass.end();
    ctx.queue.submit([encoder.finish()]);

    const results: Float32Array[] = [];
    for (const out of outputs) results.push(await readFloats(ctx, pool, out));

    const report = diffRuns(results, {
      label,
      workgroupsPerRun: subject.workgroupsPerRun,
      minWorkgroups: options.minWorkgroups,
    });
    return report;
  } finally {
    for (const run of runs) run.release?.();
    for (const out of outputs.reverse()) pool.release(out);
  }
}

export interface DiffOptions {
  label?: string;
  workgroupsPerRun?: number;
  minWorkgroups?: number;
}

/**
 * Compare every run against the first, on bit patterns. Pure — separated from the GPU path
 * so the reporting can be tested without needing a kernel that is actually racy.
 */
export function diffRuns(runs: Float32Array[], options: DiffOptions = {}): DeterminismReport {
  const label = options.label ?? 'kernel';
  if (runs.length < 2) {
    throw new RangeError('diffRuns: at least 2 runs are needed to compare');
  }
  const baseline = runs[0];
  for (const run of runs) {
    if (run.length !== baseline.length) {
      throw new RangeError(`diffRuns: run lengths differ (${run.length} vs ${baseline.length})`);
    }
  }

  const baselineBits = bits(baseline);
  const perIteration = new Array<number>(runs.length).fill(0);
  /** index → first iteration it diverged at */
  const firstSeen = new Map<number, number>();

  for (let iteration = 1; iteration < runs.length; iteration++) {
    const runBits = bits(runs[iteration]);
    for (let index = 0; index < baselineBits.length; index++) {
      if (runBits[index] === baselineBits[index]) continue;
      perIteration[iteration]++;
      if (!firstSeen.has(index)) firstSeen.set(index, iteration);
    }
  }

  const divergences: Divergence[] = [...firstSeen.entries()]
    .sort((a, b) => a[0] - b[0])
    .slice(0, MAX_DETAIL)
    .map(([index, iteration]) => ({
      index,
      iteration,
      baseline: baseline[index],
      observed: runs[iteration][index],
    }));

  let firstDivergentIteration: number | null = null;
  for (let i = 1; i < perIteration.length; i++) {
    if (perIteration[i] > 0) {
      firstDivergentIteration = i;
      break;
    }
  }

  const report: DeterminismReport = {
    deterministic: firstSeen.size === 0,
    label,
    iterations: runs.length,
    elements: baseline.length,
    divergentIndexCount: firstSeen.size,
    firstDivergentIteration,
    divergences,
    perIteration,
    baseline,
    summary: '',
  };
  report.summary = formatDeterminismReport(report, options);
  return report;
}

export function formatDeterminismReport(
  report: DeterminismReport,
  options: DiffOptions = {},
): string {
  const workgroups = options.workgroupsPerRun;
  const minWorkgroups = options.minWorkgroups ?? 64;

  if (report.deterministic) {
    const lines = [
      `${report.label}: ${report.iterations} concurrent runs byte-identical across ` +
        `${report.elements} elements`,
    ];
    if (workgroups !== undefined && workgroups < minWorkgroups) {
      lines.push(
        `  warning: ${workgroups} workgroups per run is below ${minWorkgroups} — too little ` +
          `concurrency for this to be evidence against a race. Use a larger shape.`,
      );
    }
    return lines.join('\n');
  }

  const lines = [
    `${report.label}: NOT DETERMINISTIC — ${report.divergentIndexCount} of ${report.elements} ` +
      `elements differed across ${report.iterations} runs, first at iteration ` +
      `${report.firstDivergentIteration}.`,
    '  This is a race, not noise. The usual cause is a missing workgroupBarrier() after a',
    '  shared-memory tile is consumed, before the next tile overwrites it.',
    `  divergent elements per iteration: ${report.perIteration.join(', ')}`,
  ];
  for (const d of report.divergences) {
    lines.push(
      `  [${d.index}] run 0 = ${d.baseline} · run ${d.iteration} = ${d.observed} ` +
        `(delta ${Math.abs(d.observed - d.baseline).toExponential(2)})`,
    );
  }
  if (report.divergentIndexCount > report.divergences.length) {
    lines.push(`  … and ${report.divergentIndexCount - report.divergences.length} more`);
  }
  return lines.join('\n');
}

function bits(values: Float32Array): Uint32Array {
  return new Uint32Array(values.buffer, values.byteOffset, values.length);
}
