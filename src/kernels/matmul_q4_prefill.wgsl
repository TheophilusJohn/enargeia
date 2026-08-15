// int4 block-quantized projection, prefill shape: out[m, n] = sum_k x[m, k] * W[n, k] + bias
//
// W arrives as nibbles — eight per u32 — with one f32 scale and one 4-bit zero-point per
// block of 64 along k. Dequantization is (nibble - zero) * scale.
//
// Built on the 4x2 register blocking from matmul_block42, because M1 established that 1D
// coarsening is asymptotically stuck at 1.0 shared loads per multiply-add while 2D reaches
// 0.75. Same 16x16 workgroup, same 64-row by 32-column output footprint.
//
// WHERE THE DEQUANTIZATION HAPPENS, AND WHY IT IS NOT IN THE INNER LOOP
//
// The kernels skill says to dequantize inside the accumulation loop and never write
// dequantized weights to a buffer. The second half of that rule is about the *memory bus*:
// materializing fp32 weights in a GPUBuffer would mean reading fp32 bytes from VRAM, which
// is exactly the cost quantization exists to remove.
//
// Here the nibbles are unpacked while staging into `var<workgroup>` shared memory, which is
// on-chip. The bytes crossing the bus are still int4; nothing fp32 is written to a buffer.
// What changes is how often the unpack runs. In this blocking each staged weight is consumed
// by four different accumulators, so unpacking during staging does the work once per value
// instead of four times:
//
//   staged   1 unpack per weight, 6 shared loads per 8 multiply-adds  (0.75 loads/MAC)
//   in-loop  4 unpacks per weight, 6 shared loads per 8 multiply-adds
//
// The trade is real: staging costs 2 KiB more shared memory than holding tileB packed, and
// M1's stage 3 showed shared memory buying occupancy at a rate that can go negative. It is
// staged here because prefill has the reuse to amortize it. The decode kernel, where every
// weight is read exactly once and there is no reuse to amortize, unpacks in the loop as the
// skill describes — see matmul_q4_decode.wgsl.

const TILE_K: u32 = 16u;   // reduction depth staged per iteration
const ROWS:   u32 = 64u;   // rows of C per workgroup (16 threads x 4)
const COLS:   u32 = 32u;   // columns of C per workgroup (16 threads x 2)
const RSTEP:  u32 = 16u;
const CSTEP:  u32 = 16u;

struct Dims {
    m: u32,
    n: u32,
    k: u32,
    useBias: u32,
    blockSize: u32,
    blocksPerRow: u32,
    outStride: u32,
    outOffset: u32,
    gridWidth: u32,   // unused here; the struct is shared with the decode kernel
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       x:      array<f32>;
@group(0) @binding(1) var<storage, read>       packed: array<u32>;
@group(0) @binding(2) var<storage, read>       scales: array<f32>;
@group(0) @binding(3) var<storage, read>       zeros:  array<u32>;
@group(0) @binding(4) var<storage, read>       bias:   array<f32>;
@group(0) @binding(5) var<storage, read_write> out:    array<f32>;
@group(0) @binding(6) var<uniform>             dims:   Dims;

var<workgroup> tileA: array<f32, 1024>; // 64 rows x 16 k — activations, already f32
var<workgroup> tileB: array<f32, 512>;  // 16 k x 32 cols — weights, dequantized on the way in

/** One weight of W, at output row `n` and reduction index `i`. */
fn dequant(n: u32, i: u32) -> f32 {
    let flat = n * dims.k + i;
    let word = packed[flat >> 3u];
    let nib = (word >> ((flat & 7u) * 4u)) & 0xFu;

    let block = n * dims.blocksPerRow + i / dims.blockSize;
    let zeroWord = zeros[block >> 3u];
    let zero = (zeroWord >> ((block & 7u) * 4u)) & 0xFu;

    return (f32(nib) - f32(zero)) * scales[block];
}

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id)        wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tx = lid.x;
    let ty = lid.y;
    let rowBase = wid.y * ROWS;
    let colBase = wid.x * COLS;

    let numTiles = (dims.k + TILE_K - 1u) / TILE_K;

    var acc00 = 0.0; var acc01 = 0.0;
    var acc10 = 0.0; var acc11 = 0.0;
    var acc20 = 0.0; var acc21 = 0.0;
    var acc30 = 0.0; var acc31 = 0.0;

    for (var t = 0u; t < numTiles; t = t + 1u) {
        let kIndex = t * TILE_K + tx;
        let kForB = t * TILE_K + ty;

        // Activations: four rows per thread, tx walking k, so the load is coalesced.
        for (var s = 0u; s < 4u; s = s + 1u) {
            let r = ty + s * RSTEP;
            let row = rowBase + r;
            if (row < dims.m && kIndex < dims.k) {
                tileA[r * TILE_K + tx] = x[row * dims.k + kIndex];
            } else {
                tileA[r * TILE_K + tx] = 0.0;
            }
        }

        // Weights: two columns per thread, unpacked here rather than in the inner loop.
        // Zero-fill out of range for the same reason as every other tiled kernel — shared
        // memory is not cleared between tile iterations, so a skipped store leaves the
        // previous tile's weight in place and it is multiplied in.
        for (var s = 0u; s < 2u; s = s + 1u) {
            let cLocal = tx + s * CSTEP;
            let col = colBase + cLocal;
            if (col < dims.n && kForB < dims.k) {
                tileB[ty * COLS + cLocal] = dequant(col, kForB);
            } else {
                tileB[ty * COLS + cLocal] = 0.0;
            }
        }

        workgroupBarrier();

        for (var i = 0u; i < TILE_K; i = i + 1u) {
            let a0 = tileA[(ty + 0u * RSTEP) * TILE_K + i];
            let a1 = tileA[(ty + 1u * RSTEP) * TILE_K + i];
            let a2 = tileA[(ty + 2u * RSTEP) * TILE_K + i];
            let a3 = tileA[(ty + 3u * RSTEP) * TILE_K + i];
            let b0 = tileB[i * COLS + tx + 0u * CSTEP];
            let b1 = tileB[i * COLS + tx + 1u * CSTEP];

            acc00 = acc00 + a0 * b0; acc01 = acc01 + a0 * b1;
            acc10 = acc10 + a1 * b0; acc11 = acc11 + a1 * b1;
            acc20 = acc20 + a2 * b0; acc21 = acc21 + a2 * b1;
            acc30 = acc30 + a3 * b0; acc31 = acc31 + a3 * b1;
        }

        workgroupBarrier();
    }

    // One mask per accumulator, right on both axes — a thread owns four rows 16 apart and two
    // columns 16 apart, so at the corner of C any of the eight can independently fall outside.
    let row0 = rowBase + ty + 0u * RSTEP;
    let row1 = rowBase + ty + 1u * RSTEP;
    let row2 = rowBase + ty + 2u * RSTEP;
    let row3 = rowBase + ty + 3u * RSTEP;
    let col0 = colBase + tx + 0u * CSTEP;
    let col1 = colBase + tx + 1u * CSTEP;

    let b0v = select(0.0, bias[col0], dims.useBias != 0u && col0 < dims.n);
    let b1v = select(0.0, bias[col1], dims.useBias != 0u && col1 < dims.n);

    if (row0 < dims.m && col0 < dims.n) { out[row0 * dims.outStride + col0 + dims.outOffset] = acc00 + b0v; }
    if (row0 < dims.m && col1 < dims.n) { out[row0 * dims.outStride + col1 + dims.outOffset] = acc01 + b1v; }
    if (row1 < dims.m && col0 < dims.n) { out[row1 * dims.outStride + col0 + dims.outOffset] = acc10 + b0v; }
    if (row1 < dims.m && col1 < dims.n) { out[row1 * dims.outStride + col1 + dims.outOffset] = acc11 + b1v; }
    if (row2 < dims.m && col0 < dims.n) { out[row2 * dims.outStride + col0 + dims.outOffset] = acc20 + b0v; }
    if (row2 < dims.m && col1 < dims.n) { out[row2 * dims.outStride + col1 + dims.outOffset] = acc21 + b1v; }
    if (row3 < dims.m && col0 < dims.n) { out[row3 * dims.outStride + col0 + dims.outOffset] = acc30 + b0v; }
    if (row3 < dims.m && col1 < dims.n) { out[row3 * dims.outStride + col1 + dims.outOffset] = acc31 + b1v; }
}
