/**
 * The KV cache: one of the three buffer owners, alongside `BufferPool` and `WeightStore`.
 *
 * Preallocated at session start, sized to the maximum context, destroyed on reset. Growing it
 * per token would mean a reallocation and a copy in the middle of decode, which stalls the
 * pipeline and fragments memory — so running out of context is a session-level event that the
 * caller handles, never a per-token concern.
 *
 * Layout, per layer, two buffers:
 *
 *     k[position][kvHead][dim]   maxContext x kvHeads x headDim, f32
 *     v[position][kvHead][dim]   same
 *
 * Position-major so appending a token is a write to a contiguous range at
 * `position * kvHeads * headDim`, which the projection kernels reach with their existing
 * `outOffset` — no separate append kernel, and no copy.
 *
 * K is stored *after* RoPE. Rotation depends only on a key's own absolute position, so a
 * cached key never needs re-rotating; storing pre-RoPE keys would mean rotating the whole
 * history again on every step, which is the cost the cache exists to remove.
 */

export interface KVCacheConfig {
  layers: number;
  kvHeads: number;
  headDim: number;
  /** Maximum positions the session can hold. */
  maxContext: number;
  /**
   * Element precision. f16 halves the cache and, above a 512 context, is the largest single
   * bandwidth term in decode — 35% of the step at 2048. Stored as packed halves read with
   * `unpack2x16float`, which is core WGSL and needs no device feature.
   */
  dtype?: 'f32' | 'f16';
}

export interface LayerCache {
  layer: number;
  k: GPUBuffer;
  v: GPUBuffer;
  kBinding: GPUBufferBinding;
  vBinding: GPUBufferBinding;
}

const USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

export class KVCache {
  readonly config: KVCacheConfig;
  /** Elements per position, per layer: kvHeads * headDim. */
  readonly stride: number;
  readonly dtype: 'f32' | 'f16';
  readonly bytesPerLayer: number;
  readonly totalBytes: number;

  private readonly layers: LayerCache[] = [];
  private destroyed = false;
  private length = 0;

  constructor(device: GPUDevice, config: KVCacheConfig) {
    this.config = config;
    this.stride = config.kvHeads * config.headDim;

    this.dtype = config.dtype ?? 'f32';
    const elementBytes = this.dtype === 'f16' ? 2 : 4;
    const bytes = config.maxContext * this.stride * elementBytes;
    const limit = device.limits.maxStorageBufferBindingSize;
    if (bytes > limit) {
      throw new RangeError(
        `KV cache needs ${bytes} bytes per tensor at a context of ${config.maxContext}, ` +
          `over the ${limit} byte binding limit — reduce maxContext`,
      );
    }

    for (let layer = 0; layer < config.layers; layer++) {
      const k = device.createBuffer({ label: `kv/${layer}/k`, size: bytes, usage: USAGE });
      const v = device.createBuffer({ label: `kv/${layer}/v`, size: bytes, usage: USAGE });
      this.layers.push({
        layer,
        k,
        v,
        kBinding: { buffer: k, offset: 0, size: bytes },
        vBinding: { buffer: v, offset: 0, size: bytes },
      });
    }

    this.bytesPerLayer = bytes * 2;
    this.totalBytes = this.bytesPerLayer * config.layers;
  }

  get(layer: number): LayerCache {
    const entry = this.layers[layer];
    if (!entry) throw new RangeError(`layer ${layer} outside a ${this.layers.length}-layer cache`);
    return entry;
  }

  /** Positions currently populated. */
  get filled(): number {
    return this.length;
  }

  get maxContext(): number {
    return this.config.maxContext;
  }

  get remaining(): number {
    return this.config.maxContext - this.length;
  }

  /**
   * Element offset of a position within a layer's K or V buffer. This is what the projection
   * kernels write at, so an appended token lands in place with no copy.
   */
  offsetFor(position: number): number {
    return position * this.stride;
  }

  /** Record that `count` positions are now populated. */
  advance(count: number): void {
    if (this.length + count > this.config.maxContext) {
      throw new RangeError(
        `context exhausted: ${this.length} + ${count} exceeds ${this.config.maxContext}`,
      );
    }
    this.length += count;
  }

  /**
   * Forget the contents without freeing anything.
   *
   * The buffers stay allocated: a new conversation reuses them, and a reset that freed and
   * reallocated 50 MB would reintroduce exactly the stall this class exists to avoid. Stale
   * bytes past `filled` are unreachable because every kernel bounds its reads by the current
   * length.
   */
  reset(): void {
    this.length = 0;
  }

  /** Destroy every buffer. Called on session reset, and nowhere else. */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const entry of this.layers) {
      entry.k.destroy();
      entry.v.destroy();
    }
    this.layers.length = 0;
    this.length = 0;
  }

  get isDestroyed(): boolean {
    return this.destroyed;
  }
}
