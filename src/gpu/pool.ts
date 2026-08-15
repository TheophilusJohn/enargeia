/**
 * Transient buffer allocation.
 *
 * WebGPU does not garbage-collect GPU memory: a buffer that is dropped without
 * `destroy()` stays resident until the tab closes. So every buffer in the engine has
 * exactly one owner — this pool for scratch, WeightStore for weights, KVCache for the
 * cache — and nothing calls `device.createBuffer` outside those three.
 *
 * Recycling is by size class, not exact size. A decode step asks for the same handful of
 * shapes 170 times; rounding up to a power of two turns that into a handful of
 * allocations reused forever, at the cost of up to 2x slack on scratch that is small
 * relative to the weights.
 */

/** Smallest size class. Below this, rounding up is free. */
const MIN_CLASS_LOG2 = 8; // 256 B

/** Idle bytes retained before released buffers start being destroyed instead of kept. */
const DEFAULT_MAX_IDLE_BYTES = 256 * 1024 * 1024;

export interface PooledBuffer {
  readonly buffer: GPUBuffer;
  /** Bytes the caller asked for, rounded up to 4. Bind with this, not with capacity. */
  readonly size: number;
  /** Bytes actually allocated. Always >= size. */
  readonly capacity: number;
  readonly usage: GPUBufferUsageFlags;
  /**
   * Explicit binding range. Always bind through this: a pooled buffer is usually larger
   * than requested, and `arrayLength()` in WGSL reports the bound range, so binding the
   * whole buffer changes what a shader computes.
   */
  readonly binding: GPUBufferBinding;
}

export interface PoolStats {
  liveCount: number;
  liveBytes: number;
  idleCount: number;
  idleBytes: number;
  created: number;
  reused: number;
  destroyed: number;
}

export interface PoolOptions {
  maxIdleBytes?: number;
  label?: string;
}

interface PoolEntry extends PooledBuffer {
  classLog2: number;
}

export class BufferPool {
  private readonly device: GPUDevice;
  private readonly maxIdleBytes: number;
  private readonly label: string;
  /** key = `${usage}:${classLog2}` */
  private readonly idle = new Map<string, GPUBuffer[]>();
  private readonly live = new Set<PoolEntry>();
  private readonly scopes: PoolEntry[][] = [];

  private idleBytes = 0;
  private liveBytes = 0;
  private idleCount = 0;
  private created = 0;
  private reused = 0;
  private destroyed = 0;

  constructor(device: GPUDevice, options: PoolOptions = {}) {
    this.device = device;
    this.maxIdleBytes = options.maxIdleBytes ?? DEFAULT_MAX_IDLE_BYTES;
    this.label = options.label ?? 'pool';
  }

  /**
   * Take a buffer of at least `size` bytes with exactly `usage`.
   *
   * Contents are undefined — a recycled buffer holds whatever the last user left. Zero it
   * with {@link clear} if the kernel reads before it writes.
   */
  acquire(size: number, usage: GPUBufferUsageFlags, label?: string): PooledBuffer {
    if (!Number.isFinite(size) || size <= 0) {
      throw new RangeError(`[${this.label}] acquire(${size}): size must be positive`);
    }
    const bound = align4(size);
    const classLog2 = classOf(bound);
    const capacity = 2 ** classLog2;
    const key = `${usage}:${classLog2}`;

    const free = this.idle.get(key);
    let buffer = free?.pop();
    if (buffer) {
      this.reused++;
      this.idleCount--;
      this.idleBytes -= capacity;
    } else {
      buffer = this.device.createBuffer({
        size: capacity,
        usage,
        label: `${this.label}/${label ?? 'scratch'}:${capacity}`,
      });
      this.created++;
    }

    const entry: PoolEntry = {
      buffer,
      size: bound,
      capacity,
      usage,
      binding: { buffer, offset: 0, size: bound },
      classLog2,
    };
    this.live.add(entry);
    this.liveBytes += capacity;
    this.scopes[this.scopes.length - 1]?.push(entry);
    return entry;
  }

  /** Return a buffer for reuse. The caller must not touch it afterwards. */
  release(pooled: PooledBuffer): void {
    const entry = pooled as PoolEntry;
    if (!this.live.delete(entry)) {
      throw new Error(`[${this.label}] release() on a buffer this pool does not own, or a double release`);
    }
    this.liveBytes -= entry.capacity;

    // A mapped buffer cannot be handed out again; unmapping is cheap and idempotent
    // enough here, and a readback buffer is the common case.
    if (entry.buffer.mapState === 'mapped') entry.buffer.unmap();

    if (this.idleBytes + entry.capacity > this.maxIdleBytes) {
      entry.buffer.destroy();
      this.destroyed++;
      return;
    }
    const key = `${entry.usage}:${entry.classLog2}`;
    const free = this.idle.get(key);
    if (free) free.push(entry.buffer);
    else this.idle.set(key, [entry.buffer]);
    this.idleCount++;
    this.idleBytes += entry.capacity;
  }

  /**
   * Scope acquisitions so a whole dispatch's scratch is released together. Scopes nest;
   * a buffer that outlives its scope must be acquired outside it.
   */
  beginScope(): void {
    this.scopes.push([]);
  }

  endScope(): void {
    const frame = this.scopes.pop();
    if (!frame) throw new Error(`[${this.label}] endScope() without beginScope()`);
    for (const entry of frame) {
      if (this.live.has(entry)) this.release(entry);
    }
  }

  async withScope<T>(fn: () => Promise<T> | T): Promise<T> {
    this.beginScope();
    try {
      return await fn();
    } finally {
      this.endScope();
    }
  }

  /** Zero a pooled buffer's bound range. Requires `COPY_DST` usage. */
  clear(encoder: GPUCommandEncoder, pooled: PooledBuffer): void {
    encoder.clearBuffer(pooled.buffer, 0, pooled.size);
  }

  stats(): PoolStats {
    return {
      liveCount: this.live.size,
      liveBytes: this.liveBytes,
      idleCount: this.idleCount,
      idleBytes: this.idleBytes,
      created: this.created,
      reused: this.reused,
      destroyed: this.destroyed,
    };
  }

  /** Destroy idle buffers, keeping live ones. Useful after a phase change such as prefill. */
  trim(): void {
    for (const free of this.idle.values()) {
      for (const buffer of free) {
        buffer.destroy();
        this.destroyed++;
      }
    }
    this.idle.clear();
    this.idleCount = 0;
    this.idleBytes = 0;
  }

  /** Destroy everything. Outstanding buffers are a leak in the caller, so they are named. */
  destroy(): void {
    this.trim();
    if (this.live.size > 0) {
      console.warn(`[${this.label}] destroy() with ${this.live.size} buffer(s) still acquired`);
    }
    for (const entry of this.live) {
      entry.buffer.destroy();
      this.destroyed++;
    }
    this.live.clear();
    this.liveBytes = 0;
    this.scopes.length = 0;
  }
}

/** Accept a pooled buffer, a raw buffer, or an explicit range wherever a binding is wanted. */
export type BufferRef = PooledBuffer | GPUBuffer | GPUBufferBinding;

export function toBinding(ref: BufferRef): GPUBufferBinding {
  if ('binding' in ref) return ref.binding;
  if ('buffer' in ref) return ref;
  return { buffer: ref };
}

function align4(bytes: number): number {
  return (bytes + 3) & ~3;
}

function classOf(bytes: number): number {
  let log2 = MIN_CLASS_LOG2;
  while (2 ** log2 < bytes) log2++;
  return log2;
}
