/**
 * WeightStore — the owner of every model weight buffer.
 *
 * Allocated once at load, destroyed on unload, and nothing else calls `createBuffer` for
 * weights. WebGPU does not garbage-collect GPU memory, so on a model this size a leaked
 * store is one reload away from an out-of-memory error.
 *
 * The embedding table is split across bindings from the start rather than when someone
 * else's device rejects it. Qwen2.5-0.5B's table is 151,936 x 896; widened to f32 for
 * WebGPU that is 519.4 MiB, which fits in one binding only where the adapter allows far
 * more than the 128 MiB spec default. Under the default it needs five. The split is a
 * property of the store so callers do not each reimplement the row indexing, and because
 * the table is tied to the LM head, both the input lookup and the output projection read
 * the same split.
 */

import type { DeviceProfile } from '../gpu/device.ts';
import {
  DEFAULT_CHUNK_BYTES,
  planChunks,
  readHeader,
  toFloat32,
  type ByteSource,
  type SafetensorsHeader,
  type TensorInfo,
} from './safetensors.ts';
import {
  alignRowsPerPart,
  isQuantized,
  quantizedByteLength,
  ranges,
  readEnargeiaHeader,
  type EnargeiaTensorInfo,
} from './enargeia.ts';
import {
  CachedChunkReader,
  ProgressTracker,
  WeightCache,
  type ModelRef,
  type ProgressCallback,
} from './cache.ts';

/** Qwen2.5's embedding tensor. Named here because the split is specific to it. */
export const EMBEDDING_TENSOR = 'model.embed_tokens.weight';

/**
 * A quantized weight: three buffers, not one.
 *
 * Kept separate rather than interleaved so each is a contiguous, correctly-aligned binding
 * the shader reads directly. `blocksPerRow` is what the kernel needs to turn a (row, k) pair
 * into a block index without a division it cannot fold.
 */
export interface QuantTensor {
  name: string;
  /** 4 or 8 for integer formats, 16 for half, 32 for float. */
  bits: 4 | 8 | 16 | 32;
  shape: [number, number];
  blockSize: number;
  blocksPerRow: number;
  packed: WeightTensor;
  scales: WeightTensor;
  zeros: WeightTensor;
  byteLength: number;
}

export interface WeightTensor {
  name: string;
  shape: number[];
  buffer: GPUBuffer;
  /** Bind through this — the buffer may be larger than the tensor. */
  binding: GPUBufferBinding;
  byteLength: number;
}

/**
 * The embedding table as several buffers, each holding a contiguous band of rows.
 *
 * Splitting on row boundaries rather than raw bytes is what keeps indexing trivial: row r
 * lives in part `floor(r / rowsPerPart)` at row `r % rowsPerPart`. A byte-aligned split
 * would put some rows across two bindings and force every reader to handle the seam.
 */
export interface SplitEmbedding {
  rows: number;
  cols: number;
  rowsPerPart: number;
  parts: WeightTensor[];
  /** Present when the table is quantized; `parts` is then empty. */
  quantParts?: QuantTensor[];
  /** Bytes per row, f32. */
  rowBytes: number;
  totalBytes: number;
  /** Which part and local row a global row index maps to. */
  locate(row: number): { part: number; localRow: number };
}

export interface WeightStoreStats {
  tensorCount: number;
  /** Bytes resident on the GPU. */
  gpuBytes: number;
  bufferCount: number;
  embeddingParts: number;
  maxStorageBufferBindingSize: number;
  /** Seconds from load start to every buffer written. */
  loadSeconds: number;
  warm: boolean;
  cachedChunks: number;
  fetchedChunks: number;
}

export interface LoadOptions {
  ref: ModelRef;
  source: ByteSource;
  onProgress?: ProgressCallback;
  chunkBytes?: number;
  /** Skip the Cache API entirely. For benchmarking cold loads repeatedly. */
  noCache?: boolean;
  /**
   * Override the binding limit used to size the embedding split. Defaults to the device
   * profile's value; set it lower to exercise the split path on generous hardware.
   */
  maxBindingBytes?: number;
}

export class WeightStore {
  private readonly tensors = new Map<string, WeightTensor>();
  private readonly quant = new Map<string, QuantTensor>();
  private readonly buffers: GPUBuffer[] = [];
  private destroyed = false;

  readonly header: SafetensorsHeader | null;
  readonly embedding: SplitEmbedding;
  readonly stats: WeightStoreStats;

  private constructor(
    header: SafetensorsHeader | null,
    tensors: Map<string, WeightTensor>,
    embedding: SplitEmbedding,
    stats: WeightStoreStats,
    quant?: Map<string, QuantTensor>,
  ) {
    if (quant) {
      for (const [name, tensor] of quant) this.quant.set(name, tensor);
    }
    this.header = header;
    this.embedding = embedding;
    this.stats = stats;
    for (const [name, tensor] of tensors) {
      this.tensors.set(name, tensor);
    }
    const seen = new Set<GPUBuffer>();
    for (const q of this.quant.values()) {
      for (const part of [q.packed, q.scales, q.zeros]) {
        if (!seen.has(part.buffer)) {
          seen.add(part.buffer);
          this.buffers.push(part.buffer);
        }
      }
    }
    for (const tensor of tensors.values()) {
      if (!seen.has(tensor.buffer)) {
        seen.add(tensor.buffer);
        this.buffers.push(tensor.buffer);
      }
    }
  }

  static async load(
    device: GPUDevice,
    profile: DeviceProfile,
    options: LoadOptions,
  ): Promise<WeightStore> {
    const started = performance.now();
    const tracker = new ProgressTracker(options.onProgress);
    const maxBinding = options.maxBindingBytes ?? profile.maxStorageBufferBindingSize;

    tracker.phase('header');
    const header = await readHeader(options.source);
    const chunks = planChunks(header, options.chunkBytes ?? DEFAULT_CHUNK_BYTES);
    tracker.total(chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0));

    const cache = new WeightCache(options.ref);
    if (!options.noCache) {
      const cachedCount = await cache.countCached(chunks);
      tracker.phase(cachedCount === chunks.length ? 'reading-cache' : 'downloading');
    } else {
      tracker.phase('downloading');
    }

    const reader = options.noCache
      ? { read: async (chunk: (typeof chunks)[number]) => {
          const at = performance.now();
          const bytes = await options.source.read(chunk.begin, chunk.end);
          tracker.chunkFromNetwork(bytes.byteLength, performance.now() - at);
          return bytes;
        } }
      : new CachedChunkReader(options.source, cache, tracker);

    const embeddingInfo = header.tensors.get(EMBEDDING_TENSOR);
    if (!embeddingInfo) {
      throw new Error(`${EMBEDDING_TENSOR} is missing; this does not look like a Qwen2 checkpoint`);
    }
    const layout = planEmbeddingSplit(embeddingInfo, maxBinding);

    const tensors = new Map<string, WeightTensor>();
    const embeddingParts: WeightTensor[] = layout.partRows.map((rows, index) =>
      createTensor(device, `${EMBEDDING_TENSOR}#${index}`, [rows, layout.cols], rows * layout.rowBytes),
    );

    for (const chunk of chunks) {
      const bytes = await reader.read(chunk);
      for (const info of chunk.tensors) {
        const offset = info.begin - (chunk.begin - header.dataOffset);
        const values = toFloat32(info.dtype, bytes, offset, info.byteLength);
        if (info.name === EMBEDDING_TENSOR) {
          writeEmbedding(device, embeddingParts, layout, values);
        } else {
          const tensor = createTensor(device, info.name, info.shape, values.byteLength);
          device.queue.writeBuffer(tensor.buffer, 0, values);
          tensors.set(info.name, tensor);
        }
      }
    }

    await device.queue.onSubmittedWorkDone();

    const embedding: SplitEmbedding = {
      rows: layout.rows,
      cols: layout.cols,
      rowsPerPart: layout.rowsPerPart,
      parts: embeddingParts,
      rowBytes: layout.rowBytes,
      totalBytes: layout.totalBytes,
      locate(row: number) {
        return { part: Math.floor(row / layout.rowsPerPart), localRow: row % layout.rowsPerPart };
      },
    };
    for (const [index, part] of embeddingParts.entries()) {
      tensors.set(`${EMBEDDING_TENSOR}#${index}`, part);
    }

    tracker.done();
    const progress = tracker.snapshot;
    const gpuBytes = [...tensors.values()].reduce((sum, tensor) => sum + tensor.byteLength, 0);

    const store = new WeightStore(header, tensors, embedding, {
      tensorCount: tensors.size,
      gpuBytes,
      bufferCount: tensors.size,
      embeddingParts: embeddingParts.length,
      maxStorageBufferBindingSize: maxBinding,
      loadSeconds: (performance.now() - started) / 1000,
      warm: progress.warm,
      cachedChunks: progress.cachedChunks,
      fetchedChunks: progress.fetchedChunks,
    });

    if (!options.noCache) {
      void WeightCache.evictOtherRevisions(options.ref);
    }
    return store;
  }

  /** A quantized weight's three buffers. Throws if the tensor is not quantized. */
  getQuant(name: string): QuantTensor {
    const tensor = this.quant.get(name);
    if (!tensor) {
      throw new Error(`quantized weight "${name}" not found; the store has ${this.quant.size}`);
    }
    return tensor;
  }

  hasQuant(name: string): boolean {
    return this.quant.has(name);
  }

  /** True when this store was loaded from a .enargeia container. */
  get isQuantized(): boolean {
    return this.quant.size > 0;
  }

  /**
   * Load a `.enargeia` container.
   *
   * Simpler than the safetensors path in one important way: nothing is converted. Packed
   * nibbles, scales and zero-points go to the GPU as the bytes they already are, so the only
   * work between the network and VRAM is a copy. The fp32 path has to widen every bf16 value
   * on the way, which is why its resident size is four times its download and this one's is
   * roughly equal to it.
   */
  static async loadQuantized(
    device: GPUDevice,
    profile: DeviceProfile,
    options: LoadOptions,
  ): Promise<WeightStore> {
    const started = performance.now();
    const tracker = new ProgressTracker(options.onProgress);
    const maxBinding = options.maxBindingBytes ?? profile.maxStorageBufferBindingSize;

    tracker.phase('header');
    const header = await readEnargeiaHeader(options.source);
    const total = [...header.tensors.values()]
      .flatMap((info) => ranges(info))
      .reduce((sum, [begin, end]) => sum + (end - begin), 0);
    tracker.total(total);
    tracker.phase('downloading');

    const tensors = new Map<string, WeightTensor>();
    const quant = new Map<string, QuantTensor>();
    let fetched = 0;

    /** Read one byte range and upload it verbatim. */
    const upload = async (
      name: string,
      shape: number[],
      [begin, end]: readonly [number, number],
    ): Promise<WeightTensor> => {
      const at = performance.now();
      const bytes = await options.source.read(header.dataOffset + begin, header.dataOffset + end);
      tracker.chunkFromNetwork(bytes.byteLength, performance.now() - at);
      fetched++;
      const tensor = createTensor(device, name, shape, bytes.byteLength);
      device.queue.writeBuffer(tensor.buffer, 0, bytes);
      return tensor;
    };

    const embeddingInfo: EnargeiaTensorInfo | undefined = header.tensors.get(EMBEDDING_TENSOR);
    if (!embeddingInfo) {
      throw new Error(`${EMBEDDING_TENSOR} is missing`);
    }

    // A 1x1 placeholder for formats with no scales or zero-points. Binding one layout for
    // every dtype keeps the kernel templating to the dequant body alone.
    const empty = createTensor(device, 'unused', [1], 4);

    for (const [name, info] of header.tensors) {
      if (isQuantized(info)) {
        const perWord = 32 / info.bits;
        const packed = await upload(`${name}.packed`, [info.shape[0], info.shape[1] / perWord], info.packed);
        const scales = await upload(`${name}.scales`, [info.totalBlocks], info.scales);
        const zeros = await upload(`${name}.zeros`, [Math.ceil(info.totalBlocks / perWord)], info.zeros);
        quant.set(name, {
          name,
          bits: info.bits,
          shape: info.shape,
          blockSize: info.blockSize,
          blocksPerRow: info.blocksPerRow,
          packed,
          scales,
          zeros,
          byteLength: quantizedByteLength(info),
        });
        continue;
      }

      // 1-D f32 tensors are norms and biases and belong in the plain map.
      if (info.dtype === 'F32' && info.shape.length === 1) {
        tensors.set(name, await upload(name, info.shape, info.data));
        continue;
      }

      // 2-D f16 or f32: the embedding at an exempted precision. Presented as a QuantTensor
      // with no scales so the graph selects a kernel on `bits` alone.
      const data = await upload(`${name}.data`, info.shape, info.data);
      quant.set(name, {
        name,
        bits: info.dtype === 'F16' ? 16 : 32,
        shape: info.shape as [number, number],
        blockSize: header.blockSize,
        blocksPerRow: (info.shape[1] as number) / header.blockSize,
        packed: data,
        scales: empty,
        zeros: empty,
        byteLength: data.byteLength,
      });
    }

    await device.queue.onSubmittedWorkDone();

    const embedding = planQuantizedEmbedding(quant.get(EMBEDDING_TENSOR)!, maxBinding);
    tracker.done();
    const progress = tracker.snapshot;

    const gpuBytes =
      [...tensors.values()].reduce((sum, t) => sum + t.byteLength, 0) +
      [...quant.values()].reduce((sum, t) => sum + t.byteLength, 0);

    return new WeightStore(
      null,
      tensors,
      embedding,
      {
        tensorCount: tensors.size + quant.size,
        gpuBytes,
        bufferCount: tensors.size + quant.size * 3,
        embeddingParts: embedding.quantParts?.length ?? 1,
        maxStorageBufferBindingSize: maxBinding,
        loadSeconds: (performance.now() - started) / 1000,
        warm: progress.warm,
        cachedChunks: progress.cachedChunks,
        fetchedChunks: fetched,
      },
      quant,
    );
  }

  get(name: string): WeightTensor {
    const tensor = this.tensors.get(name);
    if (!tensor) {
      throw new Error(`weight "${name}" not found; the store has ${this.tensors.size} tensors`);
    }
    return tensor;
  }

  has(name: string): boolean {
    return this.tensors.has(name);
  }

  names(): string[] {
    return [...this.tensors.keys()];
  }

  /** Every buffer is destroyed here and nowhere else. */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const buffer of this.buffers) {
      buffer.destroy();
    }
    this.tensors.clear();
    this.buffers.length = 0;
  }

  get isDestroyed(): boolean {
    return this.destroyed;
  }
}

export interface EmbeddingLayout {
  rows: number;
  cols: number;
  rowBytes: number;
  rowsPerPart: number;
  partRows: number[];
  totalBytes: number;
  maxBindingBytes: number;
}

/**
 * Choose the row split for the embedding table.
 *
 * Parts are equal-sized except the last, which holds the remainder. Equal sizing means
 * `floor(row / rowsPerPart)` locates a row with no lookup table, and it keeps every part
 * comfortably under the limit rather than filling four to the brim and leaving a stub.
 */
export function planEmbeddingSplit(info: TensorInfo, maxBindingBytes: number): EmbeddingLayout {
  if (info.shape.length !== 2) {
    throw new Error(`${info.name} should be 2-D, got shape [${info.shape.join(', ')}]`);
  }
  const [rows, cols] = info.shape;
  const rowBytes = cols * 4; // f32 on the GPU regardless of the file's dtype
  const totalBytes = rows * rowBytes;

  if (rowBytes > maxBindingBytes) {
    throw new Error(
      `a single embedding row is ${rowBytes} bytes, larger than the ${maxBindingBytes} byte binding limit`,
    );
  }

  const partCount = Math.max(1, Math.ceil(totalBytes / maxBindingBytes));
  let rowsPerPart = Math.ceil(rows / partCount);
  // Rounding rows up can push a part back over the limit; step down until it fits.
  while (rowsPerPart * rowBytes > maxBindingBytes) {
    rowsPerPart--;
  }

  const partRows: number[] = [];
  for (let start = 0; start < rows; start += rowsPerPart) {
    partRows.push(Math.min(rowsPerPart, rows - start));
  }

  return { rows, cols, rowBytes, rowsPerPart, partRows, totalBytes, maxBindingBytes };
}

function createTensor(
  device: GPUDevice,
  name: string,
  shape: number[],
  byteLength: number,
): WeightTensor {
  // WebGPU requires buffer sizes to be a multiple of 4; every f32 tensor already is, but
  // rounding here keeps the invariant local rather than assumed.
  const size = (byteLength + 3) & ~3;
  const buffer = device.createBuffer({
    label: `weights/${name}`,
    size,
    // COPY_SRC costs nothing and is what lets the parity harness read a weight back to
    // compare it against the reference. Without it a readback is a validation error, which
    // surfaces as a buffer full of zeros rather than as a failure.
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  return { name, shape, buffer, binding: { buffer, offset: 0, size: byteLength }, byteLength };
}

function writeEmbedding(
  device: GPUDevice,
  parts: WeightTensor[],
  layout: EmbeddingLayout,
  values: Float32Array,
): void {
  const perPart = layout.rowsPerPart * layout.cols;
  for (const [index, part] of parts.entries()) {
    const begin = index * perPart;
    const end = Math.min(begin + perPart, values.length);
    device.queue.writeBuffer(part.buffer, 0, values.subarray(begin, end));
  }
}


/**
 * Split a quantized embedding table across bindings.
 *
 * Usually a no-op, and that is the finding: at int4 the whole 151,936 x 896 table is 68 MB of
 * packed nibbles, so it fits in a single 128 MiB binding where the fp32 version needed five.
 * Quantization removes the constraint that shaped the fp32 graph.
 *
 * The split logic is kept because the constraint returns for a larger vocabulary, and because
 * a path that only runs on someone else's hardware is a path that does not work. A part
 * boundary has to fall where all three arrays divide cleanly — zero-points pack eight blocks
 * per u32 — which is what `alignRowsPerPart` enforces.
 */
export function planQuantizedEmbedding(
  tensor: QuantTensor,
  maxBindingBytes: number,
): SplitEmbedding {
  const [rows, cols] = tensor.shape;
  const packedBytesPerRow = (cols * tensor.bits) / 8;
  const partCount = Math.max(1, Math.ceil((rows * packedBytesPerRow) / maxBindingBytes));
  const rowsPerPart = partCount === 1 ? rows : alignRowsPerPart(Math.ceil(rows / partCount));

  const quantParts: QuantTensor[] = [];
  for (let begin = 0; begin < rows; begin += rowsPerPart) {
    const count = Math.min(rowsPerPart, rows - begin);
    if (partCount === 1) {
      quantParts.push(tensor);
      break;
    }
    // Sub-ranges of the same buffers: a binding offset, not a copy.
    quantParts.push({
      name: `${tensor.name}#${quantParts.length}`,
      bits: tensor.bits,
      shape: [count, cols],
      blockSize: tensor.blockSize,
      blocksPerRow: tensor.blocksPerRow,
      packed: sliceTensor(tensor.packed, (begin * cols) / (32 / tensor.bits), (count * cols) / (32 / tensor.bits)),
      scales: sliceTensor(tensor.scales, begin * tensor.blocksPerRow, count * tensor.blocksPerRow),
      zeros: sliceTensor(tensor.zeros, (begin * tensor.blocksPerRow) / 8, (count * tensor.blocksPerRow) / 8),
      byteLength: 0,
    });
  }

  return {
    rows,
    cols,
    rowsPerPart: partCount === 1 ? rows : rowsPerPart,
    parts: [],
    quantParts,
    rowBytes: packedBytesPerRow,
    totalBytes: rows * packedBytesPerRow,
    locate(row: number) {
      const per = partCount === 1 ? rows : rowsPerPart;
      return { part: Math.floor(row / per), localRow: row % per };
    },
  };
}

/** A view of part of a buffer. Storage binding offsets must be 256-byte aligned. */
function sliceTensor(tensor: WeightTensor, elementOffset: number, elementCount: number): WeightTensor {
  const offset = elementOffset * 4;
  if (offset % 256 !== 0) {
    throw new Error(`${tensor.name}: slice offset ${offset} is not 256-byte aligned`);
  }
  return {
    name: tensor.name,
    shape: [elementCount],
    buffer: tensor.buffer,
    binding: { buffer: tensor.buffer, offset, size: elementCount * 4 },
    byteLength: elementCount * 4,
  };
}
