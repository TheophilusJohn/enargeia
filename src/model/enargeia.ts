/**
 * The `.enargeia` container: int4 block-quantized weights.
 *
 *     magic     "ENARGEIA"          8 bytes
 *     version   u32 little-endian   4 bytes
 *     headerLen u32 little-endian   4 bytes
 *     header    JSON                headerLen bytes
 *     padding   zeros               to a 256-byte boundary
 *     blob      tensor data
 *
 * Every byte range in the header is 256-byte aligned, because that is what a storage buffer
 * binding offset requires — a format that ignores it forces a copy at load time for every
 * tensor.
 *
 * A quantized tensor is three ranges, not one: packed nibbles, per-block scales, per-block
 * zero-points. Keeping them separate means each is a contiguous, correctly-aligned buffer
 * that a shader binds directly, rather than an interleaved layout the kernel has to stride
 * through.
 */

import type { ByteSource } from './safetensors.ts';

export const ENARGEIA_MAGIC = 'ENARGEIA';
export const ENARGEIA_VERSION = 1;
/** Weights per block along the reduction axis. */
export const ENARGEIA_BLOCK = 64;
/** Nibbles per packed u32. */
export const NIBBLES_PER_WORD = 8;
/** Zero-points per packed u32. */
export const ZEROS_PER_WORD = 8;

const PREFIX_BYTES = 16; // magic + version + headerLength
const MAX_HEADER_BYTES = 64 * 1024 * 1024;

export type ByteRange = readonly [number, number];

/** Integer block-quantized: Q4 packs eight per u32, Q8 four. */
export interface QuantTensorInfo {
  name: string;
  dtype: 'Q4' | 'Q8';
  bits: 4 | 8;
  shape: [number, number];
  blockSize: number;
  packed: ByteRange;
  scales: ByteRange;
  zeros: ByteRange;
  /** Blocks along the reduction axis per output row. */
  blocksPerRow: number;
  totalBlocks: number;
}

export interface F32TensorInfo {
  name: string;
  dtype: 'F32';
  shape: number[];
  data: ByteRange;
}

/** Half precision, two values per u32, read with `unpack2x16float`. */
export interface F16TensorInfo {
  name: string;
  dtype: 'F16';
  shape: number[];
  data: ByteRange;
}

export type EnargeiaTensorInfo = QuantTensorInfo | F32TensorInfo | F16TensorInfo;

/** Kept as an alias so existing call sites keep meaning what they meant. */
export type Q4TensorInfo = QuantTensorInfo;

export interface EnargeiaHeader {
  version: number;
  blockSize: number;
  quantization: string;
  tensors: Map<string, EnargeiaTensorInfo>;
  /** Absolute offset of the blob. */
  dataOffset: number;
  headerLength: number;
  fileSize: number;
}

export class EnargeiaError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'EnargeiaError';
  }
}

/** Bytes needed before {@link parseEnargeiaHeader} can succeed. */
export function enargeiaHeaderLength(prefix: ArrayBuffer): number {
  if (prefix.byteLength < PREFIX_BYTES) {
    throw new EnargeiaError(`need at least ${PREFIX_BYTES} bytes to read the header length`);
  }
  const bytes = new Uint8Array(prefix, 0, 8);
  const magic = String.fromCharCode(...bytes);
  if (magic !== ENARGEIA_MAGIC) {
    throw new EnargeiaError(`not a .enargeia file: magic is ${JSON.stringify(magic)}`);
  }
  const view = new DataView(prefix);
  const version = view.getUint32(8, true);
  if (version !== ENARGEIA_VERSION) {
    throw new EnargeiaError(`unsupported .enargeia version ${version}, expected ${ENARGEIA_VERSION}`);
  }
  const headerLength = view.getUint32(12, true);
  if (headerLength > MAX_HEADER_BYTES) {
    throw new EnargeiaError(`header length ${headerLength} exceeds the ${MAX_HEADER_BYTES} byte limit`);
  }
  return PREFIX_BYTES + headerLength;
}

export function parseEnargeiaHeader(prefix: ArrayBuffer): EnargeiaHeader {
  const total = enargeiaHeaderLength(prefix);
  if (prefix.byteLength < total) {
    throw new EnargeiaError(`header needs ${total} bytes, got ${prefix.byteLength}`);
  }
  const view = new DataView(prefix);
  const headerLength = view.getUint32(12, true);
  const json = new TextDecoder().decode(new Uint8Array(prefix, PREFIX_BYTES, headerLength));

  let raw: { blockSize?: number; quantization?: string; tensors?: Record<string, unknown> };
  try {
    raw = JSON.parse(json);
  } catch (error) {
    throw new EnargeiaError(`header is not valid JSON: ${String(error)}`);
  }
  if (!raw.tensors || typeof raw.tensors !== 'object') {
    throw new EnargeiaError('header has no "tensors" object');
  }

  const blockSize = raw.blockSize ?? ENARGEIA_BLOCK;
  const tensors = new Map<string, EnargeiaTensorInfo>();
  let end = 0;

  for (const [name, value] of Object.entries(raw.tensors)) {
    const info = parseTensor(name, value, blockSize);
    tensors.set(name, info);
    for (const range of ranges(info)) end = Math.max(end, range[1]);
  }

  // The blob starts at the next 256-byte boundary after the header.
  const dataOffset = total + ((-total % 256) + 256) % 256;
  return {
    version: ENARGEIA_VERSION,
    blockSize,
    quantization: raw.quantization ?? 'unknown',
    tensors,
    dataOffset,
    headerLength,
    fileSize: dataOffset + end,
  };
}

export function ranges(info: EnargeiaTensorInfo): ByteRange[] {
  return isQuantized(info) ? [info.packed, info.scales, info.zeros] : [info.data];
}

export function isQuantized(info: EnargeiaTensorInfo): info is QuantTensorInfo {
  return info.dtype === 'Q4' || info.dtype === 'Q8';
}

function parseTensor(name: string, value: unknown, defaultBlock: number): EnargeiaTensorInfo {
  if (typeof value !== 'object' || value === null) {
    throw new EnargeiaError(`tensor "${name}" is not an object`);
  }
  const entry = value as {
    dtype?: unknown;
    shape?: unknown;
    blockSize?: unknown;
    offsets?: Record<string, unknown>;
  };
  if (!Array.isArray(entry.shape) || !entry.shape.every((d) => Number.isInteger(d) && d > 0)) {
    throw new EnargeiaError(`tensor "${name}" has a malformed shape`);
  }
  const shape = entry.shape as number[];
  const offsets = entry.offsets;
  if (!offsets || typeof offsets !== 'object') {
    throw new EnargeiaError(`tensor "${name}" has no offsets`);
  }

  if (entry.dtype === 'F32' || entry.dtype === 'F16') {
    const bytesPer = entry.dtype === 'F32' ? 4 : 2;
    const data = range(name, 'data', offsets.data);
    const expected = shape.reduce((a, b) => a * b, 1) * bytesPer;
    if (data[1] - data[0] !== expected) {
      throw new EnargeiaError(
        `${name}: ${entry.dtype} range is ${data[1] - data[0]} bytes, shape needs ${expected}`,
      );
    }
    return { name, dtype: entry.dtype, shape, data } as EnargeiaTensorInfo;
  }

  if (entry.dtype !== 'Q4' && entry.dtype !== 'Q8') {
    throw new EnargeiaError(`tensor "${name}" has unsupported dtype ${JSON.stringify(entry.dtype)}`);
  }
  const bits = entry.dtype === 'Q4' ? 4 : 8;
  const perWord = 32 / bits;
  if (shape.length !== 2) {
    throw new EnargeiaError(`${name}: Q4 tensors must be 2-D, got [${shape.join(', ')}]`);
  }

  const blockSize = typeof entry.blockSize === 'number' ? entry.blockSize : defaultBlock;
  const [rows, cols] = shape;
  if (cols % blockSize !== 0) {
    throw new EnargeiaError(`${name}: reduction axis ${cols} is not a multiple of block size ${blockSize}`);
  }

  const packed = range(name, 'packed', offsets.packed);
  const scales = range(name, 'scales', offsets.scales);
  const zeros = range(name, 'zeros', offsets.zeros);

  const elements = rows * cols;
  const totalBlocks = elements / blockSize;
  const expectPacked = (elements / perWord) * 4;
  const expectScales = totalBlocks * 4;
  const expectZeros = Math.ceil(totalBlocks / perWord) * 4;

  if (packed[1] - packed[0] !== expectPacked) {
    throw new EnargeiaError(`${name}: packed is ${packed[1] - packed[0]} bytes, expected ${expectPacked}`);
  }
  if (scales[1] - scales[0] !== expectScales) {
    throw new EnargeiaError(`${name}: scales is ${scales[1] - scales[0]} bytes, expected ${expectScales}`);
  }
  if (zeros[1] - zeros[0] !== expectZeros) {
    throw new EnargeiaError(`${name}: zeros is ${zeros[1] - zeros[0]} bytes, expected ${expectZeros}`);
  }

  return {
    name,
    dtype: entry.dtype,
    bits: bits as 4 | 8,
    shape: [rows, cols],
    blockSize,
    packed,
    scales,
    zeros,
    blocksPerRow: cols / blockSize,
    totalBlocks,
  };
}

function range(tensor: string, field: string, value: unknown): ByteRange {
  if (!Array.isArray(value) || value.length !== 2 || !value.every((v) => Number.isInteger(v) && v >= 0)) {
    throw new EnargeiaError(`${tensor}: offsets.${field} is malformed`);
  }
  const [begin, end] = value as [number, number];
  if (end < begin) {
    throw new EnargeiaError(`${tensor}: offsets.${field} runs backwards`);
  }
  return [begin, end];
}

export async function readEnargeiaHeader(source: ByteSource): Promise<EnargeiaHeader> {
  const prefix = await source.read(0, PREFIX_BYTES);
  const total = enargeiaHeaderLength(prefix);
  return parseEnargeiaHeader(await source.read(0, total));
}

/** Detect the container from its first bytes, so one loader entry point serves both formats. */
export async function detectFormat(source: ByteSource): Promise<'enargeia' | 'safetensors'> {
  const prefix = await source.read(0, 8);
  const magic = String.fromCharCode(...new Uint8Array(prefix));
  return magic === ENARGEIA_MAGIC ? 'enargeia' : 'safetensors';
}

/** Bytes a quantized tensor occupies, all three ranges together. */
export function quantizedByteLength(info: QuantTensorInfo): number {
  return ranges(info).reduce((sum, [begin, end]) => sum + (end - begin), 0);
}

/**
 * Rows per part when a quantized tensor is split across bindings.
 *
 * A split has to land where all three arrays divide cleanly. Packed and scales are fine at
 * any row boundary, but zero-points pack eight blocks per u32, so a part boundary must fall
 * on a multiple of `8 * blockSize` elements. With 896 columns that means the row count must
 * be a multiple of 4; rounding to 8 covers every shape this model has without a per-tensor
 * calculation.
 */
export const Q4_ROW_GRANULARITY = 8;

export function alignRowsPerPart(rowsPerPart: number): number {
  return Math.max(Q4_ROW_GRANULARITY, Math.floor(rowsPerPart / Q4_ROW_GRANULARITY) * Q4_ROW_GRANULARITY);
}
