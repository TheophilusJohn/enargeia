/**
 * Cache API persistence. Runs in a real browser, so this is the real Cache API rather than
 * a mock — which matters, because the behaviours worth testing (partial matching, quota,
 * key normalization) are exactly the ones a mock would get wrong.
 */

import { afterEach, describe, expect, it } from 'vitest';
import { BufferSource, parseHeader, planChunks } from '../../src/model/safetensors.ts';
import {
  CachedChunkReader,
  ProgressTracker,
  WeightCache,
  cacheAvailable,
  type LoadProgress,
  type ModelRef,
} from '../../src/model/cache.ts';
import { buildSafetensors, tinyModelSpecs } from '../helpers/safetensors-fixture.ts';

const ref: ModelRef = { modelId: 'test/cache', revision: 'rev-a', file: 'model.safetensors' };
const other: ModelRef = { ...ref, revision: 'rev-b' };

const created: WeightCache[] = [];
function makeCache(r: ModelRef): WeightCache {
  const cache = new WeightCache(r);
  created.push(cache);
  return cache;
}

afterEach(async () => {
  for (const cache of created.splice(0)) await cache.clear();
});

describe('environment', () => {
  it('has the Cache API, so warm loads are actually testable', () => {
    expect(cacheAvailable()).toBe(true);
  });
});

describe('WeightCache', () => {
  const chunk = { begin: 100, end: 132 };
  const bytes = () => new Uint8Array(32).map((_, i) => i).buffer;

  it('misses before a write and hits after', async () => {
    const cache = makeCache(ref);
    expect(await cache.readChunk(chunk)).toBeNull();
    await cache.writeChunk(chunk, bytes());
    const hit = await cache.readChunk(chunk);
    expect(hit).not.toBeNull();
    expect(new Uint8Array(hit!)).toEqual(new Uint8Array(bytes()));
  });

  it('keys on byte range, so a different range is a different entry', async () => {
    const cache = makeCache(ref);
    await cache.writeChunk({ begin: 0, end: 32 }, bytes());
    expect(await cache.readChunk({ begin: 32, end: 64 })).toBeNull();
  });

  it('keys on revision, so a new revision does not read the old one', async () => {
    const a = makeCache(ref);
    const b = makeCache(other);
    await a.writeChunk(chunk, bytes());
    expect(await b.readChunk(chunk)).toBeNull();
  });

  it('treats a truncated entry as a miss and deletes it', async () => {
    const cache = makeCache(ref);
    // A short body under the right key: worse than a miss, because it would otherwise be
    // uploaded to the GPU as weights.
    await cache.writeChunk({ begin: 0, end: 32 }, new Uint8Array(16).buffer);
    expect(await cache.readChunk({ begin: 0, end: 32 })).toBeNull();
    // And it is gone rather than left to fail again.
    expect(await cache.countCached([{ begin: 0, end: 32 }])).toBe(0);
  });

  it('counts cached chunks without reading their bodies', async () => {
    const cache = makeCache(ref);
    const chunks = [
      { begin: 0, end: 32 },
      { begin: 32, end: 64 },
      { begin: 64, end: 96 },
    ];
    await cache.writeChunk(chunks[0], bytes());
    await cache.writeChunk(chunks[2], bytes());
    expect(await cache.countCached(chunks)).toBe(2);
  });

  it('evicts other revisions but keeps the current one', async () => {
    const a = makeCache(ref);
    const b = makeCache(other);
    await a.writeChunk(chunk, bytes());
    await b.writeChunk(chunk, bytes());

    await WeightCache.evictOtherRevisions(other);

    // The storage for the old revision is gone.
    const names = await caches.keys();
    expect(names.some((n) => n.includes('rev-b'))).toBe(true);
    expect(names.some((n) => n.includes('rev-a'))).toBe(false);

    // The current revision still reads.
    expect(await b.readChunk(chunk)).not.toBeNull();

    // A fresh handle for the evicted revision misses. An *existing* handle does not: an
    // already-open Cache object keeps serving reads after caches.delete(), which is why
    // eviction runs after a successful load of the new revision, when nothing is still
    // holding a handle to the old one.
    expect(await new WeightCache(ref).readChunk(chunk)).toBeNull();
  });
});

describe('CachedChunkReader', () => {
  const file = buildSafetensors(tinyModelSpecs());
  const header = parseHeader(file.bytes.buffer as ArrayBuffer);
  const chunks = planChunks(header, 256);

  it('fetches on a cold pass and serves from cache on a warm one', async () => {
    const cache = makeCache(ref);
    let networkReads = 0;
    const source = new BufferSource(file.bytes);
    const counting = {
      byteLength: () => source.byteLength(),
      read: (begin: number, end: number) => {
        networkReads++;
        return source.read(begin, end);
      },
    };

    const coldUpdates: LoadProgress[] = [];
    const cold = new ProgressTracker((p) => coldUpdates.push(p));
    const coldReader = new CachedChunkReader(counting, cache, cold);
    for (const chunk of chunks) await coldReader.read(chunk);

    expect(networkReads).toBe(chunks.length);
    expect(cold.snapshot.warm).toBe(false);
    expect(cold.snapshot.fetchedChunks).toBe(chunks.length);
    expect(cold.snapshot.cachedChunks).toBe(0);

    // Writes are fire-and-forget; give them a turn to settle before reading back.
    await new Promise((resolve) => setTimeout(resolve, 50));

    networkReads = 0;
    const warm = new ProgressTracker();
    const warmReader = new CachedChunkReader(counting, cache, warm);
    for (const chunk of chunks) await warmReader.read(chunk);

    expect(networkReads).toBe(0);
    expect(warm.snapshot.warm).toBe(true);
    expect(warm.snapshot.cachedChunks).toBe(chunks.length);
  });

  it('resumes a partial cache rather than starting over', async () => {
    const cache = makeCache(ref);
    const source = new BufferSource(file.bytes);
    // Simulate an interrupted download: the first half made it to disk.
    const half = Math.floor(chunks.length / 2);
    for (const chunk of chunks.slice(0, half)) {
      await cache.writeChunk(chunk, await source.read(chunk.begin, chunk.end));
    }

    let networkReads = 0;
    const counting = {
      byteLength: () => source.byteLength(),
      read: (begin: number, end: number) => {
        networkReads++;
        return source.read(begin, end);
      },
    };
    const tracker = new ProgressTracker();
    const reader = new CachedChunkReader(counting, cache, tracker);
    for (const chunk of chunks) await reader.read(chunk);

    expect(networkReads).toBe(chunks.length - half);
    expect(tracker.snapshot.cachedChunks).toBe(half);
    expect(tracker.snapshot.fetchedChunks).toBe(chunks.length - half);
  });

  it('returns the same bytes whether the chunk was cached or fetched', async () => {
    const cache = makeCache(ref);
    const source = new BufferSource(file.bytes);
    const tracker = new ProgressTracker();
    const reader = new CachedChunkReader(source, cache, tracker);

    const fetched = new Uint8Array(await reader.read(chunks[0]));
    await new Promise((resolve) => setTimeout(resolve, 50));
    const cached = new Uint8Array(await reader.read(chunks[0]));
    expect(cached).toEqual(fetched);
  });
});

describe('ProgressTracker', () => {
  it('reports monotonic progress and flips warm on the first network read', () => {
    const seen: LoadProgress[] = [];
    const tracker = new ProgressTracker((p) => seen.push(p));
    tracker.total(1000);
    tracker.chunkFromCache(400);
    expect(tracker.snapshot.warm).toBe(true);
    tracker.chunkFromNetwork(600, 100);
    expect(tracker.snapshot.warm).toBe(false);
    expect(tracker.snapshot.loaded).toBe(1000);

    for (let i = 1; i < seen.length; i++) {
      expect(seen[i].loaded).toBeGreaterThanOrEqual(seen[i - 1].loaded);
    }
  });

  it('measures rate over network bytes only', () => {
    const tracker = new ProgressTracker();
    tracker.total(2000);
    // A cached chunk must not inflate the rate to something meaningless.
    tracker.chunkFromCache(1000);
    expect(tracker.snapshot.bytesPerSecond).toBeNull();
    tracker.chunkFromNetwork(1000, 1000);
    expect(tracker.snapshot.bytesPerSecond).toBeCloseTo(1000, 5);
    expect(tracker.snapshot.etaSeconds).toBe(0);
  });

  it('estimates time remaining from the observed rate', () => {
    const tracker = new ProgressTracker();
    tracker.total(10_000);
    tracker.chunkFromNetwork(2_000, 1_000); // 2000 B/s, 8000 B left
    expect(tracker.snapshot.etaSeconds).toBeCloseTo(4, 5);
  });
});
