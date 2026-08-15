/**
 * Bench page. Dev-only: it is not part of the production build, and it is the only place
 * that imports the CPU references out of `test/`.
 *
 * Order matters. Every kernel is compared against the CPU reference before it is compared
 * against a clock, and the result is shown either way — a fast kernel that fails the
 * comparison is not a result, it is a bug with a number attached.
 */

import { BufferPool, PipelineCache, initGPU, readFloats, type GPUContext } from '../gpu/index.ts';
import { MatmulNaive } from '../kernels/matmul_naive.ts';
import { MatmulTiled } from '../kernels/matmul_tiled.ts';
import { MatmulTiled4 } from '../kernels/matmul_tiled4.ts';
import { MatmulTiled8 } from '../kernels/matmul_tiled8.ts';
import { MatmulBlock42 } from '../kernels/matmul_block42.ts';
import {
  matmulFlops,
  type MatmulBinding,
  type MatmulInputs,
  type MatmulShape,
} from '../kernels/matmul_shared.ts';
import { benchmark, type BenchResult } from './runner.ts';
import {
  FP32_TOLERANCE,
  compareArrays,
  matmulRef,
  withinTolerance,
  type ErrorReport,
} from '../../test/reference/matmul.ts';
import { randomFloats } from '../../test/reference/rng.ts';

/**
 * The naive number BENCH.md records, and the size it was measured at. Headless Chromium —
 * the same environment the test suite runs in, so every M1 comparison is like for like.
 */
const BASELINE_GFLOPS = 219.4;
const BASELINE_N = 1024;

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

/** Structural surface shared by every matmul variant. */
interface MatmulKernel {
  bind(pool: BufferPool, inputs: MatmulInputs): MatmulBinding;
  encode(pass: GPUComputePassEncoder, binding: MatmulBinding): void;
  release(pool: BufferPool, binding: MatmulBinding): void;
  dispatch(encoder: GPUCommandEncoder, binding: MatmulBinding): void;
}

interface NamedKernel {
  name: string;
  kernel: MatmulKernel;
}

const $ = (id: string) => document.getElementById(id)!;
const mib = (bytes: number) => `${(bytes / 1048576).toFixed(0)} MiB`;

function table(rows: Array<[string, string, string?]>): string {
  return `<table>${rows
    .map(([k, v, cls]) => `<tr><td>${k}</td><td class="${cls ?? ''}">${v}</td></tr>`)
    .join('')}</table>`;
}

function fill(id: string, heading: string, html: string): void {
  $(id).innerHTML = `<h2>${heading}</h2>${html}`;
}

async function main(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const n = Number(params.get('n') ?? 1024);
  // Small shapes need more iterations: at n=512 a single dispatch is ~0.3 ms, which is only
  // three of Chrome's 100 us timestamp quanta, and submission overhead is a sixth of the
  // wall measurement. Neither resolves a 5% difference at the default 10.
  const iters = Number(params.get('iters') ?? 10);

  const ctx = await initGPU({
    label: 'bench',
    onUncapturedError: (error) => console.error('[gpu] uncaptured error', error),
  });
  const pool = new BufferPool(ctx.device, { label: 'bench' });
  const cache = new PipelineCache(ctx.device);
  const kernels: NamedKernel[] = [
    { name: 'matmul_naive', kernel: new MatmulNaive(ctx.device, cache) },
    { name: 'matmul_tiled', kernel: new MatmulTiled(ctx.device, cache) },
    { name: 'matmul_tiled4', kernel: new MatmulTiled4(ctx.device, cache) },
    { name: 'matmul_tiled8', kernel: new MatmulTiled8(ctx.device, cache) },
    { name: 'matmul_block42', kernel: new MatmulBlock42(ctx.device, cache) },
  ];

  reportDevice(ctx);
  await reportCorrectness(ctx, pool, kernels);

  const button = $('run') as HTMLButtonElement;
  button.disabled = false;
  button.addEventListener('click', () => {
    button.disabled = true;
    void runThroughput(ctx, pool, kernels, n, iters)
      .catch((error: unknown) => {
        $('throughput-out').innerHTML = `<p class="fail">benchmark failed: ${String(error)}</p>`;
      })
      .finally(() => {
        button.disabled = false;
        reportResources(pool, cache);
      });
  });

  reportResources(pool, cache);
}

function reportDevice(ctx: GPUContext): void {
  const p = ctx.profile;
  const yes = (v: boolean) => [v ? 'available' : 'missing', v ? 'pass' : 'fail'] as const;
  const [f16, f16cls] = yes(p.f16);
  const [ts, tscls] = yes(p.timestampQuery);
  const [sg, sgcls] = yes(p.subgroups);

  fill(
    'device',
    'Device',
    table([
      ['vendor / architecture', `${p.vendor || '—'} / ${p.architecture || '—'}`],
      ['tier', p.tier],
      ['shader-f16', f16, f16cls],
      ['timestamp-query', ts, tscls],
      ['subgroups', sg, sgcls],
      ['maxBufferSize', mib(p.maxBufferSize)],
      [
        'maxStorageBufferBindingSize',
        p.storageBindingClamped
          ? `${mib(p.maxStorageBufferBindingSize)} (clamped from ${mib(p.adapterStorageBufferBindingSize)})`
          : mib(p.maxStorageBufferBindingSize),
        p.storageBindingClamped ? 'note' : '',
      ],
      [
        'maxComputeWorkgroupStorageSize',
        `${(p.maxComputeWorkgroupStorageSize / 1024).toFixed(0)} KiB — tiled 2, tiled4 5, tiled8 9, block42 6 KiB`,
      ],
      ['maxComputeInvocationsPerWorkgroup', String(p.maxComputeInvocationsPerWorkgroup)],
      ['maxStorageBuffersPerShaderStage', String(p.maxStorageBuffersPerShaderStage)],
      [
        'embedding table (151936x896 fp32)',
        `${mib(151936 * 896 * 4)} — ${Math.ceil((151936 * 896 * 4) / p.maxStorageBufferBindingSize)} binding(s)`,
        'note',
      ],
    ]),
  );
}

/**
 * Small shapes on purpose. The reference accumulates in float64 and the kernels in f32, so
 * a deep reduction diverges by more than the fp32 parity threshold for reasons that are
 * not bugs. Shapes that are not multiples of the tile size are the ones that catch missing
 * bounds checks and missing zero-fill.
 */
async function reportCorrectness(
  ctx: GPUContext,
  pool: BufferPool,
  kernels: NamedKernel[],
): Promise<void> {
  const shapes: MatmulShape[] = [
    { m: 1, n: 1, k: 1 },
    { m: 37, n: 45, k: 23 },
    { m: 64, n: 64, k: 64 },
    { m: 17, n: 129, k: 96 },
    { m: 17, n: 17, k: 40 },
    { m: 65, n: 33, k: 40 },
  ];

  const rows: Array<[string, string, string?]> = [];
  let allPassed = true;

  for (const { name, kernel } of kernels) {
    for (const shape of shapes) {
      const a = randomFloats(shape.m * shape.k, 7);
      const b = randomFloats(shape.k * shape.n, 13);
      const first = await runOnce(ctx, pool, kernel, shape, a, b);
      const second = await runOnce(ctx, pool, kernel, shape, a, b);

      const error: ErrorReport = compareArrays(first, matmulRef(a, b, shape));
      const identical = bytesEqual(first, second);
      const ok = withinTolerance(error) && identical;
      allPassed &&= ok;

      rows.push([
        `${name} ${shape.m}x${shape.n}x${shape.k}`,
        `${ok ? 'pass' : 'FAIL'} · abs ${error.maxAbs.toExponential(2)} · rel ${error.maxRel.toExponential(2)} · ${(error.worstRatio * 100).toFixed(1)}% of tolerance${identical ? '' : ' · NON-DETERMINISTIC'}`,
        ok ? 'pass' : 'fail',
      ]);
    }
  }

  // Every variant accumulates k in the same ascending order, so they should all agree with
  // the naive kernel bit for bit. They are not required to — a different association would
  // still be correct — but while they do, any mismatch is a tiling bug and nothing else.
  if (kernels.length > 1) {
    const shape: MatmulShape = { m: 129, n: 65, k: 200 };
    const a = randomFloats(shape.m * shape.k, 31);
    const b = randomFloats(shape.k * shape.n, 37);
    const reference = await runOnce(ctx, pool, kernels[0].kernel, shape, a, b);
    for (const { name, kernel } of kernels.slice(1)) {
      const other = await runOnce(ctx, pool, kernel, shape, a, b);
      const identical = bytesEqual(reference, other);
      allPassed &&= identical;
      rows.push([
        `${name} vs ${kernels[0].name} @ ${shape.m}x${shape.n}x${shape.k}`,
        identical ? 'byte-identical' : 'DIFFERS',
        identical ? 'pass' : 'fail',
      ]);
    }
  }

  rows.push([
    'tolerance',
    `abs ${FP32_TOLERANCE.abs} + rel ${FP32_TOLERANCE.rel} per element · byte-identical reruns`,
    'note',
  ]);
  fill('correctness', 'Correctness', table(rows));
  console.log(`[bench] correctness ${allPassed ? 'PASS' : 'FAIL'}`);
}

async function runOnce(
  ctx: GPUContext,
  pool: BufferPool,
  kernel: MatmulKernel,
  shape: MatmulShape,
  a: Float32Array,
  b: Float32Array,
): Promise<Float32Array> {
  const bufA = pool.acquire(a.byteLength, STORAGE, 'a');
  const bufB = pool.acquire(b.byteLength, STORAGE, 'b');
  const bufC = pool.acquire(shape.m * shape.n * 4, STORAGE, 'c');
  ctx.queue.writeBuffer(bufA.buffer, 0, a);
  ctx.queue.writeBuffer(bufB.buffer, 0, b);

  const binding = kernel.bind(pool, { a: bufA, b: bufB, out: bufC, shape });
  const encoder = ctx.device.createCommandEncoder();
  kernel.dispatch(encoder, binding);
  ctx.queue.submit([encoder.finish()]);

  const out = await readFloats(ctx, pool, bufC);
  kernel.release(pool, binding);
  for (const buf of [bufC, bufB, bufA]) pool.release(buf);
  return out;
}

async function runThroughput(
  ctx: GPUContext,
  pool: BufferPool,
  kernels: NamedKernel[],
  n: number,
  iters: number,
): Promise<void> {
  $('throughput-out').innerHTML = '<p class="note">running…</p>';

  const shape: MatmulShape = { m: n, n, k: n };
  const elements = n * n;
  // One set of operands for every kernel, so the comparison is not measuring two different
  // sets of random numbers.
  const bufA = pool.acquire(elements * 4, STORAGE, 'a');
  const bufB = pool.acquire(elements * 4, STORAGE, 'b');
  const bufC = pool.acquire(elements * 4, STORAGE, 'c');
  const values = randomFloats(elements, 3);
  ctx.queue.writeBuffer(bufA.buffer, 0, values);
  ctx.queue.writeBuffer(bufB.buffer, 0, values);

  const results: Array<{ name: string; result: BenchResult; deterministic: boolean }> = [];

  for (const { name, kernel } of kernels) {
    const binding = kernel.bind(pool, { a: bufA, b: bufB, out: bufC, shape });
    const result = await benchmark(
      ctx,
      pool,
      {
        name: `${name} ${n}x${n}x${n} fp32`,
        flopsPerIter: matmulFlops(shape),
        encode: (pass) => kernel.encode(pass, binding),
      },
      { warmup: 2, iters },
    );

    const first = await readFloats(ctx, pool, bufC);
    const encoder = ctx.device.createCommandEncoder();
    kernel.dispatch(encoder, binding);
    ctx.queue.submit([encoder.finish()]);
    const second = await readFloats(ctx, pool, bufC);

    results.push({ name, result, deterministic: bytesEqual(first, second) });
    kernel.release(pool, binding);
  }

  const rows: Array<[string, string, string?]> = [];
  const baseline = results[0];
  for (const { name, result, deterministic } of results) {
    const speedup = result.wall.gflops / baseline.result.wall.gflops;
    rows.push([
      name,
      `${result.wall.gflops.toFixed(1)} GFLOP/s wall · ${result.wall.meanMs.toFixed(2)} ms/iter` +
        (result.gpu ? ` · ${result.gpu.gflops.toFixed(1)} GFLOP/s gpu` : '') +
        (name === baseline.name ? '' : ` · ${speedup.toFixed(2)}x`),
      deterministic ? '' : 'fail',
    ]);
    if (!deterministic) {
      rows.push([`${name} determinism`, 'DIFFERS ACROSS RUNS', 'fail']);
    }
  }

  rows.push(['iterations', `${baseline.result.iters} timed, ${baseline.result.warmup} warmup`]);
  rows.push([
    'problem',
    `${n}x${n}x${n} fp32, ${(matmulFlops(shape) / 1e9).toFixed(2)} GFLOP/iter`,
  ]);
  if (n === BASELINE_N) {
    const delta = ((baseline.result.wall.gflops / BASELINE_GFLOPS - 1) * 100).toFixed(1);
    rows.push([`naive vs BENCH.md baseline (${BASELINE_GFLOPS} GFLOP/s)`, `${delta}%`, 'note']);
  }

  // Headline is the fastest kernel, not the last one — a stage that loses should not be
  // quietly presented as the result.
  const best = results.reduce((a, b) => (b.result.wall.gflops > a.result.wall.gflops ? b : a));
  $('throughput-out').innerHTML =
    `<p class="headline">${best.result.wall.gflops.toFixed(1)} GFLOP/s <span style="font-size:14px;color:var(--dim)">${best.name}</span></p>` +
    table(rows);
  console.log(
    `[bench] ${results
      .map((r) => `${r.name} wall ${r.result.wall.gflops.toFixed(1)} gpu ${r.result.gpu?.gflops.toFixed(1) ?? '—'}`)
      .join(' · ')}`,
  );

  for (const buf of [bufC, bufB, bufA]) pool.release(buf);
}

function bytesEqual(a: Float32Array, b: Float32Array): boolean {
  if (a.length !== b.length) return false;
  const x = new Uint32Array(a.buffer, a.byteOffset, a.length);
  const y = new Uint32Array(b.buffer, b.byteOffset, b.length);
  for (let i = 0; i < x.length; i++) {
    if (x[i] !== y[i]) return false;
  }
  return true;
}

function reportResources(pool: BufferPool, cache: PipelineCache): void {
  const p = pool.stats();
  const c = cache.stats();
  fill(
    'resources',
    'Resources',
    table([
      ['pool buffers', `${p.liveCount} live (${mib(p.liveBytes)}) · ${p.idleCount} idle (${mib(p.idleBytes)})`],
      ['pool allocations', `${p.created} created · ${p.reused} reused · ${p.destroyed} destroyed`],
      ['pipeline cache', `${c.pipelines} pipeline(s) · ${c.modules} module(s) · ${c.hits} hit / ${c.misses} miss`],
    ]),
  );
}

main().catch((error: unknown) => {
  fill('device', 'Device', `<p class="fail">${String(error)}</p>`);
});
