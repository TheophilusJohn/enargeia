#!/usr/bin/env python3
"""Quantize a safetensors checkpoint to int4 and write a .enargeia file.

Offline, and never in the browser. Quantizing client-side would mean downloading the fp32
weights first, which is the thing quantization exists to avoid.

Scheme: asymmetric int4, blocks of 64 along the reduction axis, one scale and one zero-point
per block. Eight 4-bit values pack into each u32.

Block-wise rather than per-tensor because weight distributions have outliers. A single scale
for a whole matrix is set by its largest magnitude and crushes everything else toward zero;
blocks of 64 keep each scale local to weights of similar magnitude. The damage from getting
this wrong is a small perplexity increase and subtly worse text, not an obvious failure,
which is why `npm run perplexity` exists as a second check.

Layout per quantized tensor, three separate byte ranges:

    packed   u32[elements / 8]   eight nibbles each, reduction axis contiguous
    scales   f32[blocks]         one per block
    zeros    u32[ceil(blocks/8)] eight 4-bit zero-points each

Scales are f32 rather than f16 on purpose: `shader-f16` is missing on roughly a third of
devices and there is one universal path here. That costs 0.5 bits per weight against an f16
scale — 4.56 bits/weight instead of 4.06 — and is recorded in DECISIONS.md as a size
optimization gated on a tested fallback.

Usage:
    tools/.venv/bin/python tools/quantize.py \
        --input test/fixtures/model.safetensors \
        --output public/models/qwen2.5-0.5b-q4.enargeia
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    sys.exit("missing dependency: tools/.venv/bin/pip install numpy")

EMBEDDING = "model.embed_tokens.weight"
MAGIC = b"ENARGEIA"
VERSION = 1
BLOCK = 64
#: Data starts here so every tensor range is 256-byte aligned, which is what a storage
#: buffer binding offset requires.
ALIGN = 256


def read_safetensors(path: Path) -> tuple[dict, memoryview, int]:
    raw = memoryview(path.read_bytes())
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(bytes(raw[8 : 8 + header_len]))
    return header, raw, 8 + header_len


def to_f32(dtype: str, buf: memoryview) -> np.ndarray:
    if dtype == "F32":
        return np.frombuffer(buf, dtype=np.float32)
    if dtype == "F16":
        return np.frombuffer(buf, dtype=np.float16).astype(np.float32)
    if dtype == "BF16":
        # bf16 is the top 16 bits of an f32, so widening is a shift and is exact.
        u16 = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32)
        return (u16 << 16).view(np.float32)
    raise SystemExit(f"unsupported dtype {dtype}")


def quantize_blocks_n(values: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Asymmetric integer quantization at `bits` bits, blocks of BLOCK along the last axis.

    int4 and int8 differ only in the level count and the packing density, so one function
    covers both and the two cannot drift apart on the scale/zero-point convention.
    """
    levels = (1 << bits) - 1
    per_word = 32 // bits
    rows, cols = values.shape
    assert cols % BLOCK == 0, f"reduction axis {cols} is not a multiple of {BLOCK}"
    blocks = values.reshape(rows * cols // BLOCK, BLOCK).astype(np.float32)

    lo = blocks.min(axis=1)
    hi = blocks.max(axis=1)
    scale = (hi - lo) / levels
    degenerate = scale <= 0
    scale = np.where(degenerate, np.where(hi != 0, np.abs(hi) / levels, 1.0), scale)
    zero = np.clip(np.rint(-lo / scale), 0, levels).astype(np.int64)

    q = np.clip(np.rint(blocks / scale[:, None] + zero[:, None]), 0, levels).astype(np.uint32)

    shifts = (np.arange(per_word, dtype=np.uint32) * bits)
    packed = np.bitwise_or.reduce(q.reshape(-1, per_word) << shifts, axis=1).astype(np.uint32)

    n_blocks = zero.shape[0]
    padded = np.zeros(((n_blocks + per_word - 1) // per_word) * per_word, dtype=np.uint32)
    padded[:n_blocks] = zero.astype(np.uint32)
    zeros_packed = np.bitwise_or.reduce(padded.reshape(-1, per_word) << shifts, axis=1).astype(np.uint32)

    return packed, scale.astype(np.float32), zeros_packed


def quantize_blocks(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Asymmetric int4 over the last axis, in blocks of BLOCK.

    Returns (packed u32, scales f32, zeros u32-packed). `values` must be 2-D with its second
    axis a multiple of BLOCK — that axis is the reduction axis, which is what makes a block's
    weights the ones a single dot product touches consecutively.
    """
    # A block of identical values has zero range, and an arbitrary scale does NOT reproduce
    # it: with scale 1 the constant 0.375 quantizes to nibble 0 and dequantizes to 0. The
    # shared implementation handles that; see quantize_blocks_n.
    return quantize_blocks_n(values, 4)


def dequantize(packed: np.ndarray, scale: np.ndarray, zeros: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Inverse, used only to report the error this quantization introduces."""
    n = rows * cols
    nib = np.empty(n, dtype=np.uint32)
    expanded = packed.repeat(8)
    shifts = np.tile(np.arange(8, dtype=np.uint32) * 4, packed.shape[0])
    nib[:] = (expanded >> shifts) & 0xF
    blocks = n // BLOCK
    z = np.empty(blocks, dtype=np.uint32)
    zexp = zeros.repeat(8)[:blocks]
    zshift = np.tile(np.arange(8, dtype=np.uint32) * 4, zeros.shape[0])[:blocks]
    z[:] = (zexp >> zshift) & 0xF
    out = (nib.reshape(blocks, BLOCK).astype(np.float32) - z[:, None]) * scale[:, None]
    return out.reshape(rows, cols)


def write_embedding(matrix: np.ndarray, dtype: str, append) -> dict:
    """The embedding at a precision other than int4."""
    rows, cols = matrix.shape
    if dtype == "q8":
        packed, scale, zeros = quantize_blocks_n(matrix, 8)
        return {
            "dtype": "Q8",
            "shape": [rows, cols],
            "blockSize": BLOCK,
            "offsets": {
                "packed": append(packed.tobytes()),
                "scales": append(scale.tobytes()),
                "zeros": append(zeros.tobytes()),
            },
        }
    if dtype == "f16":
        # Two halves per u32, so the shader reads it with unpack2x16float — core WGSL, not
        # the shader-f16 extension, which keeps this on the same universal path as int4.
        halves = matrix.astype(np.float16).reshape(-1)
        return {
            "dtype": "F16",
            "shape": [rows, cols],
            "offsets": {"data": append(halves.tobytes())},
        }
    return {
        "dtype": "F32",
        "shape": [rows, cols],
        "offsets": {"data": append(matrix.astype(np.float32).tobytes())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("test/fixtures/model.safetensors"))
    parser.add_argument("--output", type=Path, default=Path("public/models/qwen2.5-0.5b-q4.enargeia"))
    parser.add_argument("--report", type=Path, default=None, help="write per-tensor error JSON")
    parser.add_argument(
        "--embed-dtype",
        choices=["q4", "q8", "f16", "f32"],
        default="q4",
        help="precision for the tied embedding / LM head; every other tensor stays int4",
    )
    args = parser.parse_args()

    header, raw, data_offset = read_safetensors(args.input)
    header.pop("__metadata__", None)

    tensors: dict[str, dict] = {}
    chunks: list[bytes] = []
    cursor = 0
    errors: list[dict] = []

    def append(payload: bytes) -> list[int]:
        nonlocal cursor
        pad = (-cursor) % ALIGN
        if pad:
            chunks.append(b"\0" * pad)
            cursor += pad
        begin = cursor
        chunks.append(payload)
        cursor += len(payload)
        return [begin, cursor]

    for name in sorted(header):
        entry = header[name]
        shape = entry["shape"]
        begin, end = entry["data_offsets"]
        values = to_f32(entry["dtype"], raw[data_offset + begin : data_offset + end])

        # Only 2-D weights are quantized. Norm weights and biases are a few hundred floats
        # each, contribute nothing to the download, and are exactly the tensors whose small
        # errors propagate through every position.
        if len(shape) != 2 or shape[1] % BLOCK != 0:
            tensors[name] = {
                "dtype": "F32",
                "shape": shape,
                "offsets": {"data": append(values.astype(np.float32).tobytes())},
            }
            continue

        rows, cols = shape
        matrix = values.reshape(rows, cols)

        # The tied embedding is the one tensor whose precision is a knob: it is both the input
        # lookup and the output projection, and the projection maps straight to logits.
        if name == EMBEDDING and args.embed_dtype != "q4":
            tensors[name] = write_embedding(matrix, args.embed_dtype, append)
            continue

        packed, scale, zeros = quantize_blocks(matrix)

        recovered = dequantize(packed, scale, zeros, rows, cols)
        abs_err = float(np.abs(recovered - matrix).max())
        rms = float(np.sqrt(np.mean((recovered - matrix) ** 2)))
        errors.append({"name": name, "shape": shape, "maxAbs": abs_err, "rms": rms})

        tensors[name] = {
            "dtype": "Q4",
            "shape": shape,
            "blockSize": BLOCK,
            "offsets": {
                "packed": append(packed.tobytes()),
                "scales": append(scale.tobytes()),
                "zeros": append(zeros.tobytes()),
            },
        }

    blob = b"".join(chunks)
    header_json = json.dumps(
        {
            "generator": "tools/quantize.py",
            "blockSize": BLOCK,
            "quantization": "asymmetric-int4-blockwise",
            "tensors": tensors,
        },
        separators=(",", ":"),
    ).encode("utf-8")

    # magic + version + headerLength, then the header, then padding to ALIGN, then the blob.
    prefix_len = len(MAGIC) + 4 + 4 + len(header_json)
    pad = (-prefix_len) % ALIGN
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<I", VERSION))
        handle.write(struct.pack("<I", len(header_json)))
        handle.write(header_json)
        handle.write(b"\0" * pad)
        handle.write(blob)

    total = args.output.stat().st_size
    quantized = [t for t in tensors.values() if t["dtype"] == "Q4"]
    params = sum(t["shape"][0] * t["shape"][1] for t in quantized)
    print(f"wrote {args.output} — {total / 1e6:.1f} MB  (embedding: {args.embed_dtype})")
    print(f"  {len(tensors)} tensors, {len(quantized)} quantized, {params / 1e6:.1f}M params at int4")
    print(f"  effective bits/weight: {total * 8 / params:.2f}")
    worst = sorted(errors, key=lambda e: -e["rms"])[:3]
    for e in worst:
        print(f"  worst rms: {e['name']} {e['shape']} rms={e['rms']:.5f} maxabs={e['maxAbs']:.5f}")

    if args.report:
        args.report.write_text(json.dumps({"blockSize": BLOCK, "tensors": errors}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
