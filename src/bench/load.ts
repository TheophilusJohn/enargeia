/**
 * Weight-loading benchmark. Dev-only.
 *
 * Measures the thing a visitor actually waits for: cold load over HTTP with range requests,
 * then warm load from the Cache API. The file is a synthetic safetensors with Qwen2.5-0.5B's
 * exact tensor inventory — 290 BF16 tensors, 988,065,536 bytes — served from the dev server,
 * so the numbers isolate parse, transfer and GPU upload from CDN variance.
 */

import { DEFAULT_STORAGE_BINDING_SIZE, initGPU, type GPUContext } from '../gpu/index.ts';
import { HttpRangeSource, readHeader } from '../model/safetensors.ts';
import { WeightCache, type LoadProgress, type ModelRef } from '../model/cache.ts';
import { EMBEDDING_TENSOR, WeightStore, planEmbeddingSplit } from '../model/weights.ts';

const MODEL_URL = '/models/qwen2.5-0.5b-synthetic.safetensors';
const REF: ModelRef = {
  modelId: 'Qwen/Qwen2.5-0.5B-Instruct',
  revision: 'synthetic-1',
  file: 'model.safetensors',
};

const $ = (id: string) => document.getElementById(id)!;
const mib = (bytes: number) => `${(bytes / 1048576).toFixed(1)} MiB`;
const mb = (bytes: number) => `${(bytes / 1e6).toFixed(1)} MB`;

function table(rows: Array<[string, string, string?]>): string {
  return `<table>${rows
    .map(([k, v, cls]) => `<tr><td>${k}</td><td class="${cls ?? ''}">${v}</td></tr>`)
    .join('')}</table>`;
}

async function main(): Promise<void> {
  const ctx = await initGPU({ label: 'loadbench' });
  $('device').innerHTML =
    `<h2>Device</h2>` +
    table([
      ['adapter', `${ctx.profile.vendor} / ${ctx.profile.architecture}`],
      ['maxStorageBufferBindingSize', mib(ctx.profile.maxStorageBufferBindingSize)],
      ['maxBufferSize', mib(ctx.profile.maxBufferSize)],
      ['maxStorageBuffersPerShaderStage', String(ctx.profile.maxStorageBuffersPerShaderStage)],
      ['Cache API', typeof caches !== 'undefined' ? 'available' : 'MISSING', typeof caches !== 'undefined' ? 'pass' : 'fail'],
    ]);

  await reportSplit(ctx);

  ($('run') as HTMLButtonElement).disabled = false;
  $('run').addEventListener('click', () => {
    ($('run') as HTMLButtonElement).disabled = true;
    void runLoads(ctx).finally(() => {
      ($('run') as HTMLButtonElement).disabled = false;
    });
  });
}

/**
 * The binding count under both limits. This is the number that decides whether the model
 * loads at all on a given device, so it is reported before anything is downloaded.
 */
async function reportSplit(ctx: GPUContext): Promise<void> {
  const source = new HttpRangeSource(MODEL_URL);
  const header = await readHeader(source);
  const info = header.tensors.get(EMBEDDING_TENSOR)!;

  const clamped = planEmbeddingSplit(info, DEFAULT_STORAGE_BINDING_SIZE);
  const actual = planEmbeddingSplit(info, ctx.profile.maxStorageBufferBindingSize);
  const totalF32 = [...header.tensors.values()].reduce((sum, t) => sum + t.elementCount * 4, 0);

  $('split').innerHTML =
    `<h2>Embedding split</h2>` +
    table([
      ['tensors in file', `${header.tensors.size}, all BF16`],
      ['file size', mb(header.fileSize)],
      ['embedding shape', `${info.shape[0]} x ${info.shape[1]}`],
      ['embedding, BF16 on disk', mib(info.byteLength)],
      ['embedding, F32 resident', mib(clamped.totalBytes)],
      [
        `bindings at the 128 MiB default`,
        `${clamped.partRows.length} parts, ${clamped.rowsPerPart} rows each, ${mib(clamped.rowsPerPart * clamped.rowBytes)} per part`,
        'note',
      ],
      [
        `bindings at this adapter's ${mib(ctx.profile.maxStorageBufferBindingSize)}`,
        `${actual.partRows.length} part${actual.partRows.length === 1 ? '' : 's'}`,
        'note',
      ],
      ['all weights, F32 resident', mib(totalF32), totalF32 > ctx.profile.maxBufferSize ? 'fail' : ''],
    ]);
}

async function runLoads(ctx: GPUContext): Promise<void> {
  const out = $('loads');
  const render = (html: string) => {
    out.innerHTML = html;
  };
  render('<p class="note">clearing cache…</p>');
  await new WeightCache(REF).clear();

  const rows: Array<[string, string, string?]> = [];

  try {
    const cold = await timedLoad(ctx, 'cold', (p) =>
      render(
        `<p class="note">cold: ${(p.loaded / 1e6).toFixed(0)} MB` +
          (p.total ? ` / ${(p.total / 1e6).toFixed(0)} MB` : '') +
          (p.bytesPerSecond ? ` · ${(p.bytesPerSecond / 1e6).toFixed(0)} MB/s` : '') +
          (p.etaSeconds !== null ? ` · ${p.etaSeconds.toFixed(0)}s left` : '') +
          '</p>',
      ),
    );
    rows.push([
      'cold load',
      `${cold.seconds.toFixed(2)} s · ${(cold.store.stats.gpuBytes / 1e6 / cold.seconds).toFixed(0)} MB/s effective`,
    ]);
    rows.push(['  chunks', `${cold.store.stats.fetchedChunks} fetched, ${cold.store.stats.cachedChunks} cached`]);
    rows.push(['  resident on GPU', mib(cold.store.stats.gpuBytes)]);
    rows.push(['  embedding parts', String(cold.store.stats.embeddingParts)]);
    cold.store.destroy();

    render('<p class="note">warm…</p>');
    const warm = await timedLoad(ctx, 'warm', (p) =>
      render(`<p class="note">warm: ${(p.loaded / 1e6).toFixed(0)} MB</p>`),
    );
    rows.push([
      'warm load',
      `${warm.seconds.toFixed(2)} s · ${(cold.seconds / warm.seconds).toFixed(2)}x faster`,
      'pass',
    ]);
    rows.push(['  chunks', `${warm.store.stats.fetchedChunks} fetched, ${warm.store.stats.cachedChunks} cached`]);
    rows.push(['  fully warm', String(warm.store.stats.warm), warm.store.stats.warm ? 'pass' : 'fail']);
    warm.store.destroy();

    console.log(`[loadbench] cold ${cold.seconds.toFixed(2)}s warm ${warm.seconds.toFixed(2)}s parts ${cold.store.stats.embeddingParts}`);
  } catch (error) {
    rows.push(['failed', String(error), 'fail']);
    console.error('[loadbench] failed', error);
  }

  render(table(rows));
}

async function timedLoad(
  ctx: GPUContext,
  label: string,
  onProgress: (p: LoadProgress) => void,
): Promise<{ seconds: number; store: WeightStore }> {
  const started = performance.now();
  const store = await WeightStore.load(ctx.device, ctx.profile, {
    ref: REF,
    source: new HttpRangeSource(MODEL_URL),
    onProgress,
  });
  const seconds = (performance.now() - started) / 1000;
  console.log(`[loadbench] ${label} ${seconds.toFixed(2)}s`);
  return { seconds, store };
}

main().catch((error: unknown) => {
  $('device').innerHTML = `<h2>Device</h2><p class="fail">${String(error)}</p>`;
  console.error('[loadbench] fatal', error);
});
