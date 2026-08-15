/**
 * WeightStore, with the embedding split exercised under the 128 MiB default rather than
 * this machine's 4096 MiB. The split path is the one that breaks on other people's
 * hardware, so it is tested on a device that has been clamped to reproduce theirs.
 */

import { afterAll, describe, expect, it } from 'vitest';
import { DEFAULT_STORAGE_BINDING_SIZE, initGPU, readFloats, BufferPool } from '../../src/gpu/index.ts';
import { BufferSource, bf16ToF32, parseHeader } from '../../src/model/safetensors.ts';
import { EMBEDDING_TENSOR, WeightStore, planEmbeddingSplit } from '../../src/model/weights.ts';
import { buildSafetensors, qwenLikeHeader, tinyModelSpecs, QWEN_CONFIG } from '../helpers/safetensors-fixture.ts';
import { expectNoGPUError, gpu, teardownGPU } from '../helpers/gpu.ts';

afterAll(teardownGPU);

const MIB = 1024 * 1024;

describe('planEmbeddingSplit', () => {
  const embedding = qwenLikeHeader().tensors.get(EMBEDDING_TENSOR)!;

  it('describes Qwen2.5-0.5B correctly', () => {
    expect(embedding.shape).toEqual([QWEN_CONFIG.vocab, QWEN_CONFIG.hidden]);
    // The file is BF16, but WebGPU storage is f32, so the resident size doubles.
    expect(embedding.byteLength).toBe(151936 * 896 * 2);
  });

  it('needs five bindings under the 128 MiB spec default', () => {
    const layout = planEmbeddingSplit(embedding, DEFAULT_STORAGE_BINDING_SIZE);
    expect(layout.totalBytes).toBe(151936 * 896 * 4);
    expect(layout.partRows).toHaveLength(5);
    for (const rows of layout.partRows) {
      expect(rows * layout.rowBytes).toBeLessThanOrEqual(DEFAULT_STORAGE_BINDING_SIZE);
    }
    expect(layout.partRows.reduce((a, b) => a + b, 0)).toBe(151936);
  });

  it('needs one binding when the adapter allows 4096 MiB', () => {
    const layout = planEmbeddingSplit(embedding, 4096 * MIB);
    expect(layout.partRows).toHaveLength(1);
    expect(layout.partRows[0]).toBe(151936);
  });

  it('keeps every part under the limit even when rows do not divide evenly', () => {
    for (const limit of [16 * MIB, 33 * MIB, 100 * MIB, 128 * MIB, 200 * MIB, 519 * MIB]) {
      const layout = planEmbeddingSplit(embedding, limit);
      expect(layout.partRows.reduce((a, b) => a + b, 0)).toBe(151936);
      for (const rows of layout.partRows) {
        expect(rows * layout.rowBytes, `limit ${limit}`).toBeLessThanOrEqual(limit);
      }
    }
  });

  it('splits on row boundaries so indexing needs no lookup table', () => {
    const layout = planEmbeddingSplit(embedding, DEFAULT_STORAGE_BINDING_SIZE);
    const rowsPerPart = layout.rowsPerPart;
    // Row r is at part floor(r / rowsPerPart), local row r % rowsPerPart. Check the seams.
    for (const row of [0, rowsPerPart - 1, rowsPerPart, rowsPerPart * 2 - 1, 151935]) {
      const part = Math.floor(row / rowsPerPart);
      expect(part).toBeLessThan(layout.partRows.length);
      expect(row % rowsPerPart).toBeLessThan(layout.partRows[part]);
    }
  });

  it('refuses a limit too small to hold a single row', () => {
    expect(() => planEmbeddingSplit(embedding, 1024)).toThrow(/single embedding row/);
  });
});

describe('WeightStore', () => {
  const specs = tinyModelSpecs();
  const file = buildSafetensors(specs, { format: 'pt' });
  const ref = { modelId: 'test/tiny', revision: 'r1', file: 'model.safetensors' };

  it('loads every tensor and widens bf16 to f32 on the way to the GPU', async () => {
    const { ctx } = await gpu();
    const store = await WeightStore.load(ctx.device, ctx.profile, {
      ref,
      source: new BufferSource(file.bytes),
      noCache: true,
    });
    try {
      const header = parseHeader(file.bytes.buffer as ArrayBuffer);
      // Every non-embedding tensor, plus one entry per embedding part.
      expect(store.has('model.norm.weight')).toBe(true);
      expect(store.get('model.layers.0.self_attn.q_proj.weight').shape).toEqual([8, 8]);

      // Compare a tensor's GPU contents against the file, widened the same way.
      const pool = new BufferPool(ctx.device, { label: 'weights-test' });
      const info = header.tensors.get('model.layers.0.self_attn.q_proj.weight')!;
      const expected = bf16ToF32(
        new Uint16Array(
          file.bytes.buffer.slice(
            file.dataOffset + info.begin,
            file.dataOffset + info.end,
          ),
        ),
      );
      const tensor = store.get('model.layers.0.self_attn.q_proj.weight');
      const actual = await expectNoGPUError(ctx.device, async () => {
      const staging = pool.acquire(tensor.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, 'cmp');
      const encoder = ctx.device.createCommandEncoder();
      encoder.copyBufferToBuffer(tensor.buffer, 0, staging.buffer, 0, tensor.byteLength);
      ctx.queue.submit([encoder.finish()]);
      const values = await readFloats(ctx, pool, staging);
      pool.release(staging);
      return values;
      });
      expect(Array.from(actual)).toEqual(Array.from(expected));
      pool.destroy();
    } finally {
      store.destroy();
    }
  });

  it('splits the embedding when the binding limit demands it', async () => {
    const { ctx } = await gpu();
    // 64 rows x 8 cols x 4 bytes = 2048 bytes total; a 512-byte limit forces four parts.
    const store = await WeightStore.load(ctx.device, ctx.profile, {
      ref,
      source: new BufferSource(file.bytes),
      noCache: true,
      maxBindingBytes: 512,
    });
    try {
      expect(store.embedding.parts).toHaveLength(4);
      expect(store.embedding.rowsPerPart).toBe(16);
      expect(store.embedding.locate(0)).toEqual({ part: 0, localRow: 0 });
      expect(store.embedding.locate(15)).toEqual({ part: 0, localRow: 15 });
      expect(store.embedding.locate(16)).toEqual({ part: 1, localRow: 0 });
      expect(store.embedding.locate(63)).toEqual({ part: 3, localRow: 15 });
      for (const part of store.embedding.parts) {
        expect(part.byteLength).toBeLessThanOrEqual(512);
      }
    } finally {
      store.destroy();
    }
  });

  it('puts each embedding row in the part its index says it is in', async () => {
    const { ctx } = await gpu();
    const pool = new BufferPool(ctx.device, { label: 'embed-test' });
    const store = await WeightStore.load(ctx.device, ctx.profile, {
      ref,
      source: new BufferSource(file.bytes),
      noCache: true,
      maxBindingBytes: 512,
    });
    try {
      const header = parseHeader(file.bytes.buffer as ArrayBuffer);
      const info = header.tensors.get(EMBEDDING_TENSOR)!;
      const all = bf16ToF32(
        new Uint16Array(file.bytes.buffer.slice(file.dataOffset + info.begin, file.dataOffset + info.end)),
      );
      const cols = store.embedding.cols;

      // Check a row in every part, including both sides of each seam.
      for (const row of [0, 15, 16, 31, 32, 47, 48, 63]) {
        const { part, localRow } = store.embedding.locate(row);
        const tensor = store.embedding.parts[part];
        const staging = pool.acquire(
          tensor.byteLength,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
          'part',
        );
        const encoder = ctx.device.createCommandEncoder();
        encoder.copyBufferToBuffer(tensor.buffer, 0, staging.buffer, 0, tensor.byteLength);
        ctx.queue.submit([encoder.finish()]);
        const values = await readFloats(ctx, pool, staging);
        const actual = Array.from(values.subarray(localRow * cols, (localRow + 1) * cols));
        const expected = Array.from(all.subarray(row * cols, (row + 1) * cols));
        expect(actual, `row ${row} -> part ${part}[${localRow}]`).toEqual(expected);
        pool.release(staging);
      }
    } finally {
      store.destroy();
      pool.destroy();
    }
  });

  it('reports progress that ends at 100% of the planned bytes', async () => {
    const { ctx } = await gpu();
    const updates: number[] = [];
    let sawTotal = false;
    const store = await WeightStore.load(ctx.device, ctx.profile, {
      ref,
      source: new BufferSource(file.bytes),
      noCache: true,
      chunkBytes: 256,
      onProgress: (p) => {
        updates.push(p.loaded);
        if (p.total !== null) sawTotal = true;
      },
    });
    try {
      expect(sawTotal).toBe(true);
      expect(updates.length).toBeGreaterThan(2);
      // Monotonic, and finishing at the total.
      for (let i = 1; i < updates.length; i++) {
        expect(updates[i]).toBeGreaterThanOrEqual(updates[i - 1]);
      }
      expect(store.stats.fetchedChunks).toBeGreaterThan(1);
    } finally {
      store.destroy();
    }
  });

  it('destroys every buffer exactly once and refuses use afterwards', async () => {
    const { ctx } = await gpu();
    const store = await WeightStore.load(ctx.device, ctx.profile, {
      ref,
      source: new BufferSource(file.bytes),
      noCache: true,
    });
    expect(store.isDestroyed).toBe(false);
    store.destroy();
    expect(store.isDestroyed).toBe(true);
    store.destroy(); // idempotent
    expect(() => store.get('model.norm.weight')).toThrow(/not found/);
  });

  it('rejects a checkpoint with no embedding table', async () => {
    const { ctx } = await gpu();
    const orphan = buildSafetensors([{ name: 'model.norm.weight', dtype: 'BF16', shape: [8] }]);
    await expect(
      WeightStore.load(ctx.device, ctx.profile, {
        ref,
        source: new BufferSource(orphan.bytes),
        noCache: true,
      }),
    ).rejects.toThrow(/embed_tokens/);
  });
});

describe('the clamped device', () => {
  it('reports a binding limit that forces the embedding split', async () => {
    // The same clamp the bench page uses, so the constrained path is exercised on hardware
    // that would otherwise hide it entirely.
    const clamped = await initGPU({ clampStorageBindingSize: true, label: 'weights-clamped' });
    try {
      expect(clamped.profile.maxStorageBufferBindingSize).toBe(DEFAULT_STORAGE_BINDING_SIZE);
      const embedding = qwenLikeHeader().tensors.get(EMBEDDING_TENSOR)!;
      const layout = planEmbeddingSplit(embedding, clamped.profile.maxStorageBufferBindingSize);
      expect(layout.partRows.length).toBe(5);
      // And the parts have to fit in the bindings a shader stage can hold at once.
      expect(layout.partRows.length).toBeLessThanOrEqual(
        clamped.profile.maxStorageBuffersPerShaderStage,
      );
    } finally {
      clamped.device.destroy();
    }
  });
});
