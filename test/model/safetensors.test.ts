import { describe, expect, it } from 'vitest';
import {
  BufferSource,
  DEFAULT_CHUNK_BYTES,
  HttpRangeSource,
  SafetensorsError,
  bf16ToF32,
  f16ToF32,
  headerByteLength,
  parseHeader,
  planChunks,
  readHeader,
  toFloat32,
  type SafetensorsDType,
} from '../../src/model/safetensors.ts';
import { buildSafetensors, qwenLikeHeader } from '../helpers/safetensors-fixture.ts';

describe('parseHeader', () => {
  it('reads the length prefix, JSON header and data offset', () => {
    const file = buildSafetensors([
      { name: 'a', dtype: 'F32', shape: [2, 3] },
      { name: 'b', dtype: 'BF16', shape: [4] },
    ]);
    const header = parseHeader(file.bytes.buffer as ArrayBuffer);

    expect(header.tensors.size).toBe(2);
    expect(header.dataOffset).toBe(8 + header.headerLength);
    expect(header.fileSize).toBe(file.bytes.byteLength);

    const a = header.tensors.get('a')!;
    expect(a.dtype).toBe('F32');
    expect(a.shape).toEqual([2, 3]);
    expect(a.elementCount).toBe(6);
    expect(a.byteLength).toBe(24);
    expect(a.begin).toBe(0);

    const b = header.tensors.get('b')!;
    expect(b.begin).toBe(24);
    expect(b.byteLength).toBe(8);
  });

  it('exposes __metadata__ separately from tensors', () => {
    const file = buildSafetensors([{ name: 'x', dtype: 'F32', shape: [1] }], { format: 'pt' });
    const header = parseHeader(file.bytes.buffer as ArrayBuffer);
    expect(header.metadata).toEqual({ format: 'pt' });
    expect(header.tensors.has('__metadata__')).toBe(false);
  });

  it('computes the required prefix length from the first 8 bytes', () => {
    const file = buildSafetensors([{ name: 'x', dtype: 'F32', shape: [1] }]);
    const prefix = file.bytes.slice(0, 8).buffer as ArrayBuffer;
    expect(headerByteLength(prefix)).toBe(8 + file.headerLength);
  });

  it('rejects a truncated prefix rather than parsing garbage', () => {
    const file = buildSafetensors([{ name: 'x', dtype: 'F32', shape: [1] }]);
    const short = file.bytes.slice(0, 8 + file.headerLength - 1).buffer as ArrayBuffer;
    expect(() => parseHeader(short)).toThrow(SafetensorsError);
    expect(() => headerByteLength(new ArrayBuffer(4))).toThrow(SafetensorsError);
  });

  it('rejects an absurd header length instead of allocating', () => {
    const bytes = new Uint8Array(16);
    new DataView(bytes.buffer).setBigUint64(0, 1n << 40n, true);
    expect(() => headerByteLength(bytes.buffer)).toThrow(/exceeds/);
  });

  it('rejects a tensor whose byte span disagrees with its shape and dtype', () => {
    const header = JSON.stringify({ bad: { dtype: 'F32', shape: [4], data_offsets: [0, 8] } });
    const json = new TextEncoder().encode(header);
    const bytes = new Uint8Array(8 + json.length + 8);
    new DataView(bytes.buffer).setBigUint64(0, BigInt(json.length), true);
    bytes.set(json, 8);
    expect(() => parseHeader(bytes.buffer)).toThrow(/spans 8 bytes but 4 F32 needs 16/);
  });

  it('rejects unknown dtypes and malformed offsets', () => {
    const make = (entry: unknown) => {
      const json = new TextEncoder().encode(JSON.stringify({ t: entry }));
      const bytes = new Uint8Array(8 + json.length);
      new DataView(bytes.buffer).setBigUint64(0, BigInt(json.length), true);
      bytes.set(json, 8);
      return bytes.buffer;
    };
    expect(() => parseHeader(make({ dtype: 'F8', shape: [1], data_offsets: [0, 1] }))).toThrow(/dtype/);
    expect(() => parseHeader(make({ dtype: 'U8', shape: [1], data_offsets: [4, 0] }))).toThrow(/backwards/);
    expect(() => parseHeader(make({ dtype: 'U8', shape: 'x', data_offsets: [0, 1] }))).toThrow(/shape/);
  });
});

describe('readHeader over a ByteSource', () => {
  it('reads the header in two requests', async () => {
    const file = buildSafetensors([{ name: 'x', dtype: 'BF16', shape: [8] }]);
    const reads: Array<[number, number]> = [];
    const source = new BufferSource(file.bytes);
    const spy = {
      byteLength: () => source.byteLength(),
      read: (begin: number, end: number) => {
        reads.push([begin, end]);
        return source.read(begin, end);
      },
    };
    const header = await readHeader(spy);
    expect(header.tensors.size).toBe(1);
    expect(reads).toEqual([
      [0, 8],
      [0, 8 + file.headerLength],
    ]);
  });
});

describe('planChunks', () => {
  it('groups small tensors together and keeps large ones whole', () => {
    const file = buildSafetensors([
      { name: 'small1', dtype: 'F32', shape: [16] },
      { name: 'small2', dtype: 'F32', shape: [16] },
      { name: 'big', dtype: 'F32', shape: [1024] },
      { name: 'small3', dtype: 'F32', shape: [16] },
    ]);
    const header = parseHeader(file.bytes.buffer as ArrayBuffer);
    const chunks = planChunks(header, 512);

    expect(chunks.length).toBeGreaterThan(1);
    // Every tensor appears exactly once, in offset order.
    const names = chunks.flatMap((c) => c.tensors.map((t) => t.name));
    expect(names).toEqual(['small1', 'small2', 'big', 'small3']);
    // The oversized tensor is alone in its chunk rather than split.
    const bigChunk = chunks.find((c) => c.tensors.some((t) => t.name === 'big'))!;
    expect(bigChunk.tensors).toHaveLength(1);
  });

  it('produces absolute offsets that tile the data blob without gaps or overlap', () => {
    const header = qwenLikeHeader();
    const chunks = planChunks(header, DEFAULT_CHUNK_BYTES);
    let previousEnd = header.dataOffset;
    for (const chunk of chunks) {
      expect(chunk.begin).toBe(previousEnd);
      expect(chunk.end).toBeGreaterThan(chunk.begin);
      expect(chunk.byteLength).toBe(chunk.end - chunk.begin);
      previousEnd = chunk.end;
    }
    expect(previousEnd).toBe(header.fileSize);
  });

  it('turns 290 tensors into a manageable number of requests', () => {
    const header = qwenLikeHeader();
    expect(header.tensors.size).toBe(290);
    const chunks = planChunks(header, DEFAULT_CHUNK_BYTES);
    // One request per tensor would be 290 round trips; one for everything reports no
    // progress. The plan should land well inside both.
    expect(chunks.length).toBeLessThan(80);
    expect(chunks.length).toBeGreaterThan(10);
  });

  it('is deterministic, so chunk ranges are stable cache keys', () => {
    const header = qwenLikeHeader();
    expect(planChunks(header)).toEqual(planChunks(header));
  });
});

describe('dtype widening', () => {
  it('widens bf16 exactly, since bf16 is the top half of an f32', () => {
    const values = new Float32Array([1, -2, 0.5, 0, Infinity, -Infinity]);
    const truncated = new Uint16Array(values.length);
    const bits = new Uint32Array(values.buffer);
    for (let i = 0; i < values.length; i++) truncated[i] = bits[i] >>> 16;
    expect(Array.from(bf16ToF32(truncated))).toEqual(Array.from(values));
  });

  it('preserves NaN through bf16 widening', () => {
    expect(Number.isNaN(bf16ToF32(new Uint16Array([0x7fc0]))[0])).toBe(true);
  });

  it('widens f16 including subnormals, infinity and NaN', () => {
    expect(f16ToF32(new Uint16Array([0x3c00]))[0]).toBe(1);
    expect(f16ToF32(new Uint16Array([0xc000]))[0]).toBe(-2);
    expect(f16ToF32(new Uint16Array([0x0001]))[0]).toBe(2 ** -24); // smallest subnormal, exact
    expect(f16ToF32(new Uint16Array([0x7c00]))[0]).toBe(Infinity);
    expect(Number.isNaN(f16ToF32(new Uint16Array([0x7e00]))[0])).toBe(true);
    expect(Object.is(f16ToF32(new Uint16Array([0x8000]))[0], -0)).toBe(true);
  });

  it('refuses to widen integer dtypes', () => {
    for (const dtype of ['I64', 'U8', 'BOOL'] as SafetensorsDType[]) {
      expect(() => toFloat32(dtype, new ArrayBuffer(8))).toThrow(/cannot widen/);
    }
  });
});

describe('HttpRangeSource', () => {
  const body = new Uint8Array(1000).map((_, i) => i % 251);

  const fakeFetch = (behaviour: 'range' | 'ignores-range' | 'error'): typeof fetch =>
    (async (_url: string | URL | Request, init?: RequestInit) => {
      if (behaviour === 'error') return new Response('nope', { status: 500 });
      const header = (init?.headers as Record<string, string> | undefined)?.Range;
      if (behaviour === 'ignores-range' || !header) {
        return new Response(body, { status: 200, headers: { 'Content-Length': String(body.length) } });
      }
      const [begin, end] = /bytes=(\d+)-(\d+)/.exec(header)!.slice(1).map(Number);
      return new Response(body.slice(begin, end + 1), {
        status: 206,
        headers: { 'Content-Range': `bytes ${begin}-${end}/${body.length}` },
      });
    }) as typeof fetch;

  it('reads exact ranges', async () => {
    const source = new HttpRangeSource('https://example.invalid/w', { fetchImpl: fakeFetch('range') });
    const bytes = new Uint8Array(await source.read(10, 20));
    expect(bytes).toEqual(body.slice(10, 20));
  });

  it('learns the total size from Content-Range', async () => {
    const source = new HttpRangeSource('https://example.invalid/w', { fetchImpl: fakeFetch('range') });
    expect(await source.byteLength()).toBe(1000);
  });

  it('fails loudly when the server ignores Range', async () => {
    // Silently buffering a gigabyte because a CDN does not do ranges is the failure mode
    // this guard exists for — it would look like a hang, not an error.
    const source = new HttpRangeSource('https://example.invalid/w', { fetchImpl: fakeFetch('ignores-range') });
    await expect(source.read(0, 10)).rejects.toThrow(/ignored Range/);
  });

  it('surfaces HTTP errors', async () => {
    const source = new HttpRangeSource('https://example.invalid/w', { fetchImpl: fakeFetch('error') });
    await expect(source.read(0, 10)).rejects.toThrow(/HTTP 500/);
  });

  it('reads nothing for an empty range without a request', async () => {
    let calls = 0;
    const counting: typeof fetch = (async () => {
      calls++;
      return new Response(new Uint8Array(0), { status: 206 });
    }) as typeof fetch;
    const source = new HttpRangeSource('https://example.invalid/w', { fetchImpl: counting });
    expect((await source.read(5, 5)).byteLength).toBe(0);
    expect(calls).toBe(0);
  });
});
