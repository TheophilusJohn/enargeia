/**
 * Persisting downloaded weights to the Cache API.
 *
 * First load is the highest-abandonment moment on the site, and a second visit that
 * re-downloads 988 MB is worse than one that never loaded at all. So chunks are stored
 * individually as they arrive: a cold load that is interrupted at 70% resumes from 70%
 * rather than from zero, and a warm load reads one chunk at a time instead of
 * materializing the whole file.
 *
 * Storing the file as a single Response would be simpler and is wrong on both counts —
 * the Cache API has no partial match, so a whole-file entry means all-or-nothing on
 * resume, and reading it back means holding a gigabyte in memory to serve ranges from.
 *
 * Cache keys carry the model id and revision. A revision change is a different model, and
 * the old entries are evicted rather than left to occupy storage forever.
 */

import type { ByteSource, Chunk } from './safetensors.ts';

/** Bump when the key layout or stored payload shape changes. */
const CACHE_VERSION = 'v1';

export interface ModelRef {
  /** e.g. "Qwen/Qwen2.5-0.5B-Instruct" */
  modelId: string;
  /** Commit sha or tag. "main" is accepted but pins nothing — prefer a sha. */
  revision: string;
  /** File within the repo, e.g. "model.safetensors". */
  file: string;
}

export type LoadPhase = 'probing' | 'header' | 'downloading' | 'reading-cache' | 'done';

export interface LoadProgress {
  phase: LoadPhase;
  /** Bytes of tensor data transferred so far. */
  loaded: number;
  /** Total bytes of tensor data, or null before the header is parsed. */
  total: number | null;
  /** Chunks served from the cache this session. */
  cachedChunks: number;
  /** Chunks fetched over the network this session. */
  fetchedChunks: number;
  /** True when every chunk so far came from cache. Flips false on the first network read. */
  warm: boolean;
  /** Bytes per second over the network, null while nothing has been fetched. */
  bytesPerSecond: number | null;
  /** Seconds remaining at the current rate, null when unknown or warm. */
  etaSeconds: number | null;
}

export type ProgressCallback = (progress: LoadProgress) => void;

function cacheName(ref: ModelRef): string {
  return `enargeia-${CACHE_VERSION}-${ref.modelId}@${ref.revision}`;
}

/**
 * Cache keys must be URLs. The origin is irrelevant — the Cache API never fetches these —
 * but it has to parse, so a stable synthetic origin is used and the real identity lives in
 * the path and query.
 */
function chunkKey(ref: ModelRef, chunk: { begin: number; end: number }): string {
  return `https://enargeia.invalid/${encodeURIComponent(ref.modelId)}/${encodeURIComponent(
    ref.revision,
  )}/${encodeURIComponent(ref.file)}?bytes=${chunk.begin}-${chunk.end}`;
}

function headerKey(ref: ModelRef): string {
  return `https://enargeia.invalid/${encodeURIComponent(ref.modelId)}/${encodeURIComponent(
    ref.revision,
  )}/${encodeURIComponent(ref.file)}?header`;
}

export function cacheAvailable(): boolean {
  return typeof caches !== 'undefined';
}

/**
 * Chunk-granular persistence for one model revision.
 *
 * Every method degrades to a no-op when the Cache API is unavailable (older Safari, some
 * private-browsing modes, non-secure contexts). Loading still works; it is just never warm.
 */
export class WeightCache {
  readonly ref: ModelRef;
  private readonly name: string;
  private cache: Cache | null = null;
  private opened = false;

  constructor(ref: ModelRef) {
    this.ref = ref;
    this.name = cacheName(ref);
  }

  private async open(): Promise<Cache | null> {
    if (this.opened) return this.cache;
    this.opened = true;
    if (!cacheAvailable()) return null;
    try {
      this.cache = await caches.open(this.name);
    } catch (error) {
      console.warn(`[cache] unavailable, loading uncached: ${String(error)}`);
      this.cache = null;
    }
    return this.cache;
  }

  async readChunk(chunk: { begin: number; end: number }): Promise<ArrayBuffer | null> {
    const cache = await this.open();
    if (!cache) return null;
    try {
      const hit = await cache.match(chunkKey(this.ref, chunk));
      if (!hit) return null;
      const bytes = await hit.arrayBuffer();
      // A truncated entry is worse than a miss, because it would be uploaded as weights.
      if (bytes.byteLength !== chunk.end - chunk.begin) {
        await cache.delete(chunkKey(this.ref, chunk));
        return null;
      }
      return bytes;
    } catch {
      return null;
    }
  }

  async writeChunk(chunk: { begin: number; end: number }, bytes: ArrayBuffer): Promise<void> {
    const cache = await this.open();
    if (!cache) return;
    try {
      await cache.put(
        chunkKey(this.ref, chunk),
        new Response(bytes, {
          headers: {
            'Content-Type': 'application/octet-stream',
            'Content-Length': String(bytes.byteLength),
          },
        }),
      );
    } catch (error) {
      // Quota exceeded is the expected failure and is not fatal: the model is already in
      // memory, this visit works, and the next one pays the download again.
      console.warn(`[cache] could not store chunk ${chunk.begin}-${chunk.end}: ${String(error)}`);
    }
  }

  async readHeaderBytes(): Promise<ArrayBuffer | null> {
    const cache = await this.open();
    if (!cache) return null;
    try {
      const hit = await cache.match(headerKey(this.ref));
      return hit ? await hit.arrayBuffer() : null;
    } catch {
      return null;
    }
  }

  async writeHeaderBytes(bytes: ArrayBuffer): Promise<void> {
    const cache = await this.open();
    if (!cache) return;
    try {
      await cache.put(headerKey(this.ref), new Response(bytes));
    } catch (error) {
      console.warn(`[cache] could not store header: ${String(error)}`);
    }
  }

  /** How many of these chunks are already stored. Cheap — reads keys, not bodies. */
  async countCached(chunks: Array<{ begin: number; end: number }>): Promise<number> {
    const cache = await this.open();
    if (!cache) return 0;
    try {
      const keys = new Set((await cache.keys()).map((request) => request.url));
      return chunks.filter((chunk) => keys.has(chunkKey(this.ref, chunk))).length;
    } catch {
      return 0;
    }
  }

  async clear(): Promise<void> {
    if (!cacheAvailable()) return;
    try {
      await caches.delete(this.name);
      this.cache = null;
      this.opened = false;
    } catch (error) {
      console.warn(`[cache] could not clear: ${String(error)}`);
    }
  }

  /**
   * Delete cached data for other revisions of the same model. Called after a successful
   * load, so a failed upgrade does not throw away a working copy.
   *
   * Note that a `WeightCache` instance which has already opened its `Cache` keeps serving
   * reads after the storage is deleted — the open handle outlives the entry. That is fine
   * given when this runs, but it means an instance for an evicted revision is stale and
   * should be discarded rather than reused.
   */
  static async evictOtherRevisions(ref: ModelRef): Promise<number> {
    if (!cacheAvailable()) return 0;
    const keep = cacheName(ref);
    const prefix = `enargeia-${CACHE_VERSION}-${ref.modelId}@`;
    let removed = 0;
    try {
      for (const name of await caches.keys()) {
        if (name.startsWith(prefix) && name !== keep) {
          await caches.delete(name);
          removed++;
        }
      }
    } catch (error) {
      console.warn(`[cache] eviction failed: ${String(error)}`);
    }
    return removed;
  }
}

/**
 * Tracks progress across a load and hands the caller a fresh snapshot per update.
 *
 * Rate is measured over network bytes only. Mixing cached chunks in would report a
 * meaningless "10 GB/s" on a warm load and make the ETA useless on a partial one.
 */
export class ProgressTracker {
  private readonly onProgress: ProgressCallback | undefined;
  private readonly startedAt: number;
  private networkBytes = 0;
  private networkMillis = 0;
  private state: LoadProgress = {
    phase: 'probing',
    loaded: 0,
    total: null,
    cachedChunks: 0,
    fetchedChunks: 0,
    warm: true,
    bytesPerSecond: null,
    etaSeconds: null,
  };

  constructor(onProgress?: ProgressCallback, now = performance.now()) {
    this.onProgress = onProgress;
    this.startedAt = now;
  }

  get snapshot(): LoadProgress {
    return { ...this.state };
  }

  get elapsedSeconds(): number {
    return (performance.now() - this.startedAt) / 1000;
  }

  phase(phase: LoadPhase): void {
    this.state.phase = phase;
    this.emit();
  }

  total(bytes: number): void {
    this.state.total = bytes;
    this.emit();
  }

  chunkFromCache(byteLength: number): void {
    this.state.cachedChunks++;
    this.state.loaded += byteLength;
    this.emit();
  }

  chunkFromNetwork(byteLength: number, millis: number): void {
    this.state.fetchedChunks++;
    this.state.warm = false;
    this.state.loaded += byteLength;
    this.networkBytes += byteLength;
    this.networkMillis += millis;
    if (this.networkMillis > 0) {
      this.state.bytesPerSecond = (this.networkBytes / this.networkMillis) * 1000;
      const remaining = (this.state.total ?? 0) - this.state.loaded;
      this.state.etaSeconds = remaining > 0 ? remaining / this.state.bytesPerSecond : 0;
    }
    this.emit();
  }

  done(): void {
    this.state.phase = 'done';
    this.state.etaSeconds = 0;
    this.emit();
  }

  private emit(): void {
    this.onProgress?.(this.snapshot);
  }
}

/**
 * A {@link ByteSource} that reads planned chunks through the cache, fetching only what is
 * missing and storing what it fetches.
 *
 * Reads must be chunk-aligned: this is a loader, not a general random-access source. That
 * restriction is what lets a cache entry be addressed by its byte range without the Cache
 * API needing partial matching.
 */
export class CachedChunkReader {
  private readonly source: ByteSource;
  private readonly cache: WeightCache;
  private readonly tracker: ProgressTracker;

  constructor(source: ByteSource, cache: WeightCache, tracker: ProgressTracker) {
    this.source = source;
    this.cache = cache;
    this.tracker = tracker;
  }

  async read(chunk: Chunk): Promise<ArrayBuffer> {
    const cached = await this.cache.readChunk(chunk);
    if (cached) {
      this.tracker.chunkFromCache(cached.byteLength);
      return cached;
    }
    const started = performance.now();
    const bytes = await this.source.read(chunk.begin, chunk.end);
    this.tracker.chunkFromNetwork(bytes.byteLength, performance.now() - started);
    // Store a copy: the caller may transfer or neuter the buffer during GPU upload.
    void this.cache.writeChunk(chunk, bytes.slice(0));
    return bytes;
  }
}
