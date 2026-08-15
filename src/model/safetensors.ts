/**
 * safetensors parsing and range-request streaming.
 *
 * Layout: 8 bytes of little-endian u64 header length, then that many bytes of JSON giving
 * each tensor's dtype, shape and byte range, then the raw blob. Tensor offsets are relative
 * to the start of the blob, which begins at 8 + headerLength.
 *
 * Qwen2.5-0.5B-Instruct ships 290 tensors, every one of them BF16, in 988,097,824 bytes.
 * `lm_head.weight` is absent because the embedding table is tied to the output projection.
 * The other dtypes are implemented because the format allows them and the parity path may
 * hand us an F32 file, not because this model uses them.
 */

/** Bytes of little-endian u64 preceding the JSON header. */
export const HEADER_LENGTH_BYTES = 8;

/** Refuse headers larger than this — a corrupt length field should not allocate a gigabyte. */
const MAX_HEADER_BYTES = 100 * 1024 * 1024;

export type SafetensorsDType = 'F64' | 'F32' | 'F16' | 'BF16' | 'I64' | 'I32' | 'I16' | 'I8' | 'U8' | 'BOOL';

const DTYPE_BYTES: Record<SafetensorsDType, number> = {
  F64: 8,
  F32: 4,
  F16: 2,
  BF16: 2,
  I64: 8,
  I32: 4,
  I16: 2,
  I8: 1,
  U8: 1,
  BOOL: 1,
};

export interface TensorInfo {
  name: string;
  dtype: SafetensorsDType;
  shape: number[];
  /** Byte offset of the first byte, relative to the start of the data blob. */
  begin: number;
  /** Byte offset one past the last byte, relative to the start of the data blob. */
  end: number;
  byteLength: number;
  elementCount: number;
}

export interface SafetensorsHeader {
  tensors: Map<string, TensorInfo>;
  metadata: Record<string, string>;
  /** Absolute offset of the data blob within the file: 8 + headerLength. */
  dataOffset: number;
  headerLength: number;
  /** Absolute size of the whole file implied by the header. */
  fileSize: number;
}

export class SafetensorsError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SafetensorsError';
  }
}

/** How many bytes of the file are needed before {@link parseHeader} can succeed. */
export function headerByteLength(prefix: ArrayBuffer): number {
  if (prefix.byteLength < HEADER_LENGTH_BYTES) {
    throw new SafetensorsError(`need at least ${HEADER_LENGTH_BYTES} bytes to read the header length`);
  }
  const length = new DataView(prefix).getBigUint64(0, true);
  if (length > BigInt(MAX_HEADER_BYTES)) {
    throw new SafetensorsError(`header length ${length} exceeds the ${MAX_HEADER_BYTES} byte limit`);
  }
  return HEADER_LENGTH_BYTES + Number(length);
}

/**
 * Parse the header from a prefix of the file. The prefix must be at least
 * {@link headerByteLength} bytes; anything beyond that is ignored.
 */
export function parseHeader(prefix: ArrayBuffer): SafetensorsHeader {
  const total = headerByteLength(prefix);
  if (prefix.byteLength < total) {
    throw new SafetensorsError(`header needs ${total} bytes, got ${prefix.byteLength}`);
  }
  const json = new TextDecoder().decode(new Uint8Array(prefix, HEADER_LENGTH_BYTES, total - HEADER_LENGTH_BYTES));

  let raw: unknown;
  try {
    raw = JSON.parse(json);
  } catch (error) {
    throw new SafetensorsError(`header is not valid JSON: ${String(error)}`);
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    throw new SafetensorsError('header must be a JSON object');
  }

  const entries = raw as Record<string, unknown>;
  const metadata = (entries.__metadata__ ?? {}) as Record<string, string>;
  const tensors = new Map<string, TensorInfo>();
  let dataEnd = 0;

  for (const [name, value] of Object.entries(entries)) {
    if (name === '__metadata__') continue;
    tensors.set(name, parseTensor(name, value));
    dataEnd = Math.max(dataEnd, tensors.get(name)!.end);
  }

  return {
    tensors,
    metadata,
    dataOffset: total,
    headerLength: total - HEADER_LENGTH_BYTES,
    fileSize: total + dataEnd,
  };
}

function parseTensor(name: string, value: unknown): TensorInfo {
  if (typeof value !== 'object' || value === null) {
    throw new SafetensorsError(`tensor "${name}" is not an object`);
  }
  const entry = value as { dtype?: unknown; shape?: unknown; data_offsets?: unknown };

  const dtype = entry.dtype;
  if (typeof dtype !== 'string' || !(dtype in DTYPE_BYTES)) {
    throw new SafetensorsError(`tensor "${name}" has unsupported dtype ${JSON.stringify(dtype)}`);
  }
  if (!Array.isArray(entry.shape) || !entry.shape.every((d) => Number.isInteger(d) && d >= 0)) {
    throw new SafetensorsError(`tensor "${name}" has a malformed shape`);
  }
  const offsets = entry.data_offsets;
  if (!Array.isArray(offsets) || offsets.length !== 2 || !offsets.every((o) => Number.isInteger(o) && o >= 0)) {
    throw new SafetensorsError(`tensor "${name}" has malformed data_offsets`);
  }

  const shape = entry.shape as number[];
  const [begin, end] = offsets as [number, number];
  if (end < begin) {
    throw new SafetensorsError(`tensor "${name}" has data_offsets [${begin}, ${end}] running backwards`);
  }

  const elementCount = shape.reduce((a, b) => a * b, 1);
  const expected = elementCount * DTYPE_BYTES[dtype as SafetensorsDType];
  if (end - begin !== expected) {
    throw new SafetensorsError(
      `tensor "${name}" spans ${end - begin} bytes but ${shape.join('x')} ${dtype} needs ${expected}`,
    );
  }

  return {
    name,
    dtype: dtype as SafetensorsDType,
    shape,
    begin,
    end,
    byteLength: end - begin,
    elementCount,
  };
}

export function dtypeByteLength(dtype: SafetensorsDType): number {
  return DTYPE_BYTES[dtype];
}

// ---------------------------------------------------------------------------
// Byte sources
// ---------------------------------------------------------------------------

/** Somewhere bytes come from, addressed by absolute offset. */
export interface ByteSource {
  /** Total bytes, or null when the source cannot report it without reading. */
  byteLength(): Promise<number | null>;
  /** Read [begin, end). Implementations must return exactly that many bytes. */
  read(begin: number, end: number): Promise<ArrayBuffer>;
}

/** For tests and for files already in memory. */
export class BufferSource implements ByteSource {
  private readonly bytes: Uint8Array;

  constructor(bytes: ArrayBuffer | Uint8Array) {
    this.bytes = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  }

  async byteLength(): Promise<number> {
    return this.bytes.byteLength;
  }

  async read(begin: number, end: number): Promise<ArrayBuffer> {
    if (begin < 0 || end > this.bytes.byteLength || end < begin) {
      throw new SafetensorsError(`read [${begin}, ${end}) is outside a ${this.bytes.byteLength} byte source`);
    }
    return this.bytes.slice(begin, end).buffer as ArrayBuffer;
  }
}

export interface HttpSourceOptions {
  fetchImpl?: typeof fetch;
  signal?: AbortSignal;
  /** Extra headers, e.g. an authorization token for a gated repo. */
  headers?: Record<string, string>;
}

/**
 * HTTP range requests.
 *
 * Every read is a separate `Range` request, which is what makes progress reportable: the
 * caller decides the chunk size and gets a callback per chunk rather than one opaque wait.
 * A server that ignores `Range` and returns 200 with the whole body is treated as an error
 * rather than silently buffering a gigabyte.
 */
export class HttpRangeSource implements ByteSource {
  readonly url: string;
  private readonly options: HttpSourceOptions;
  private cachedLength: number | null = null;

  constructor(url: string, options: HttpSourceOptions = {}) {
    this.url = url;
    this.options = options;
  }

  async byteLength(): Promise<number | null> {
    if (this.cachedLength !== null) return this.cachedLength;
    const fetchImpl = this.options.fetchImpl ?? fetch;
    // A HEAD is cheaper, but some CDNs answer HEAD differently from GET; a one-byte range
    // request returns Content-Range with the true total and costs the same round trip.
    const response = await fetchImpl(this.url, {
      headers: { ...this.options.headers, Range: 'bytes=0-0' },
      signal: this.options.signal,
    });
    if (!response.ok) {
      throw new SafetensorsError(`${this.url}: HTTP ${response.status} probing length`);
    }
    const total = parseContentRangeTotal(response.headers.get('Content-Range'));
    await response.arrayBuffer();
    if (total === null) {
      const length = response.headers.get('Content-Length');
      this.cachedLength = length === null ? null : Number(length);
    } else {
      this.cachedLength = total;
    }
    return this.cachedLength;
  }

  async read(begin: number, end: number): Promise<ArrayBuffer> {
    if (end <= begin) return new ArrayBuffer(0);
    const fetchImpl = this.options.fetchImpl ?? fetch;
    const response = await fetchImpl(this.url, {
      headers: { ...this.options.headers, Range: `bytes=${begin}-${end - 1}` },
      signal: this.options.signal,
    });
    if (!response.ok) {
      throw new SafetensorsError(`${this.url}: HTTP ${response.status} reading [${begin}, ${end})`);
    }
    if (response.status !== 206) {
      throw new SafetensorsError(
        `${this.url}: server ignored Range and returned ${response.status}; ` +
          'range support is required so progress can be reported',
      );
    }
    const bytes = await response.arrayBuffer();
    if (bytes.byteLength !== end - begin) {
      throw new SafetensorsError(
        `${this.url}: asked for ${end - begin} bytes at ${begin}, received ${bytes.byteLength}`,
      );
    }
    return bytes;
  }
}

function parseContentRangeTotal(header: string | null): number | null {
  if (!header) return null;
  const match = /\/(\d+)\s*$/.exec(header);
  return match ? Number(match[1]) : null;
}

/** Read the header from any source, in two reads: the length, then the JSON. */
export async function readHeader(source: ByteSource): Promise<SafetensorsHeader> {
  const lengthPrefix = await source.read(0, HEADER_LENGTH_BYTES);
  const total = headerByteLength(lengthPrefix);
  const full = await source.read(0, total);
  return parseHeader(full);
}

// ---------------------------------------------------------------------------
// Chunk planning
// ---------------------------------------------------------------------------

export interface Chunk {
  index: number;
  /** Absolute file offsets. */
  begin: number;
  end: number;
  byteLength: number;
  tensors: TensorInfo[];
}

/** Default request size. Large enough to amortize round trips, small enough to show progress. */
export const DEFAULT_CHUNK_BYTES = 16 * 1024 * 1024;

/**
 * Group tensors into contiguous range requests of roughly `targetBytes`.
 *
 * One request per tensor would mean 290 round trips for Qwen2.5-0.5B, most of them for
 * tensors under a kilobyte. One request for everything reports no progress and cannot be
 * resumed. Chunking is the middle, and it is pure so the plan can be asserted in a test and
 * reused as cache keys.
 *
 * A tensor larger than `targetBytes` gets its own chunk rather than being split, so a chunk
 * always contains whole tensors. The embedding table is 260 MB, so that chunk is large — it
 * is one request either way, and keeping tensors whole means no reassembly across chunks.
 */
export function planChunks(header: SafetensorsHeader, targetBytes = DEFAULT_CHUNK_BYTES): Chunk[] {
  const ordered = [...header.tensors.values()].sort((a, b) => a.begin - b.begin);
  const chunks: Chunk[] = [];
  let current: TensorInfo[] = [];
  let begin = 0;
  let end = 0;

  const flush = () => {
    if (current.length === 0) return;
    chunks.push({
      index: chunks.length,
      begin: header.dataOffset + begin,
      end: header.dataOffset + end,
      byteLength: end - begin,
      tensors: current,
    });
    current = [];
  };

  for (const tensor of ordered) {
    if (current.length === 0) {
      current = [tensor];
      begin = tensor.begin;
      end = tensor.end;
      continue;
    }
    // Only extend while the span stays under target. Gaps between tensors are rare but
    // possible; including them costs bytes, so a gap that would blow the budget flushes.
    if (tensor.end - begin > targetBytes) {
      flush();
      current = [tensor];
      begin = tensor.begin;
      end = tensor.end;
    } else {
      current.push(tensor);
      end = tensor.end;
    }
  }
  flush();
  return chunks;
}

// ---------------------------------------------------------------------------
// dtype conversion
// ---------------------------------------------------------------------------

/**
 * bfloat16 to float32. bf16 is literally the top 16 bits of an IEEE f32, so widening is a
 * shift and is exact — no rounding, no special cases for NaN or infinity.
 */
export function bf16ToF32(source: Uint16Array, out?: Float32Array): Float32Array {
  const result = out ?? new Float32Array(source.length);
  const bits = new Uint32Array(result.buffer, result.byteOffset, source.length);
  for (let i = 0; i < source.length; i++) {
    bits[i] = source[i] << 16;
  }
  return result;
}

/** IEEE half to float32. Handles subnormals, infinity and NaN. */
export function f16ToF32(source: Uint16Array, out?: Float32Array): Float32Array {
  const result = out ?? new Float32Array(source.length);
  for (let i = 0; i < source.length; i++) {
    const h = source[i];
    const sign = (h & 0x8000) >> 15;
    const exponent = (h & 0x7c00) >> 10;
    const fraction = h & 0x03ff;
    let value: number;
    if (exponent === 0) {
      value = fraction === 0 ? 0 : fraction * 2 ** -24;
    } else if (exponent === 0x1f) {
      value = fraction === 0 ? Infinity : NaN;
    } else {
      value = (1 + fraction / 1024) * 2 ** (exponent - 15);
    }
    result[i] = sign ? -value : value;
  }
  return result;
}

/**
 * Widen any supported float dtype to f32, which is what WebGPU storage buffers hold. There
 * is no bf16 in WGSL, and f16 needs the `shader-f16` feature that a third of devices lack,
 * so f32 is the only universally bindable representation.
 */
export function toFloat32(dtype: SafetensorsDType, bytes: ArrayBuffer, byteOffset = 0, byteLength?: number): Float32Array {
  const length = byteLength ?? bytes.byteLength - byteOffset;
  switch (dtype) {
    case 'F32':
      return new Float32Array(bytes.slice(byteOffset, byteOffset + length));
    case 'BF16':
      return bf16ToF32(new Uint16Array(bytes.slice(byteOffset, byteOffset + length)));
    case 'F16':
      return f16ToF32(new Uint16Array(bytes.slice(byteOffset, byteOffset + length)));
    default:
      throw new SafetensorsError(`cannot widen ${dtype} to f32`);
  }
}
