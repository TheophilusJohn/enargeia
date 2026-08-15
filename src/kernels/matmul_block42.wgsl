// Tiled fp32 matmul, 2D register blocking: each thread computes a 4x2 block of C.
//
// Eight accumulators, the same count as matmul_tiled8, arranged differently. Stage 3 stacked
// them in one column, which amortizes tileB across the strip but leaves every accumulator
// needing its own tileA value — 8 A loads + 1 B load per 8 multiply-adds, 1.125 loads/MAC,
// and the sequence stalls near 1.0 no matter how tall the strip gets.
//
// A 4x2 block amortizes both operands. The four rows share two column values and the two
// columns share four row values, so one iteration costs 4 A loads + 2 B loads for 8
// multiply-adds:
//
//     0.75 shared loads per MAC, against 1.125 for an 8x1 strip.
//
// Tile dimensions follow from that. 16x16 threads each owning 4 rows x 2 columns means the
// workgroup covers 64 rows x 32 columns of C, so tileA is 64x16 (4 KiB) and tileB is 16x32
// (2 KiB) — 6 KiB total, less than stage 3's 9 KiB. Dispatch divides by 32 in x and 64 in y.
//
// Workgroup count is identical to stage 3's at every size, which is deliberate: the two
// kernels differ in load ratio and shared-memory footprint while holding the amount of
// available parallelism fixed.
//
// Accumulation order is still k ascending, matching every other variant term for term.

const TILE_K: u32 = 16u;  // reduction depth staged per iteration
const ROWS:   u32 = 64u;  // rows of C per workgroup (16 threads x 4)
const COLS:   u32 = 32u;  // columns of C per workgroup (16 threads x 2)
const RSTEP:  u32 = 16u;  // gap between a thread's four output rows
const CSTEP:  u32 = 16u;  // gap between a thread's two output columns

struct Dims {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>       a:    array<f32>;
@group(0) @binding(1) var<storage, read>       b:    array<f32>;
@group(0) @binding(2) var<storage, read_write> c:    array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

var<workgroup> tileA: array<f32, 1024>; // 64 rows x 16 k, row-major — 4 KiB
var<workgroup> tileB: array<f32, 512>;  // 16 k x 32 cols, row-major — 2 KiB

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

    // Eight accumulators, named by their position in the 4x2 block. Separate scalars rather
    // than array<f32, 8>: a dynamically indexed local array is backed by memory, not
    // registers.
    var acc00 = 0.0; var acc01 = 0.0;
    var acc10 = 0.0; var acc11 = 0.0;
    var acc20 = 0.0; var acc21 = 0.0;
    var acc30 = 0.0; var acc31 = 0.0;

    for (var t = 0u; t < numTiles; t = t + 1u) {
        let aCol = t * TILE_K + tx;
        let bRow = t * TILE_K + ty;

        // tileA: four rows per thread at tile rows ty, ty+16, ty+32, ty+48, column tx.
        // tx stays on the fastest-varying memory axis, so each load is 16 consecutive floats.
        //
        // The k guard is load-bearing exactly as in every earlier stage: a skipped store
        // leaves the previous tile's value in the slot and it is multiplied in. The row and
        // column guards remain defensive under the reachability argument in
        // docs/tiled-matmul.md, and are kept for the same reasons given there.
        for (var s = 0u; s < 4u; s = s + 1u) {
            let r = ty + s * RSTEP;
            let row = rowBase + r;
            if (row < dims.m && aCol < dims.k) {
                tileA[r * TILE_K + tx] = a[row * dims.k + aCol];
            } else {
                tileA[r * TILE_K + tx] = 0.0;
            }
        }

        // tileB: two columns per thread at tile columns tx and tx+16, row ty. 512 slots over
        // 256 threads. Also coalesced — consecutive tx reads consecutive columns of B.
        for (var s = 0u; s < 2u; s = s + 1u) {
            let cLocal = tx + s * CSTEP;
            let col = colBase + cLocal;
            if (bRow < dims.k && col < dims.n) {
                tileB[ty * COLS + cLocal] = b[bRow * dims.n + col];
            } else {
                tileB[ty * COLS + cLocal] = 0.0;
            }
        }

        // Barrier 1: both tiles fully staged before anyone reads them.
        workgroupBarrier();

        for (var i = 0u; i < TILE_K; i = i + 1u) {
            // Six loads into registers, then eight multiply-adds against them. This is the
            // whole point: every loaded value is used twice (the A values across two columns,
            // the B values across four rows) instead of once.
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

        // Barrier 2: tiles fully consumed before iteration t+1 overwrites them.
        workgroupBarrier();
    }

    // One mask per accumulator, and now the mask has to be right on both axes. A thread owns
    // four rows 16 apart and two columns 16 apart, so at the corner of C any of the eight can
    // independently fall outside: at m = 20, n = 20, rowBase = colBase = 0 and (tx, ty) =
    // (5, 5), the rows are 5, 21, 37, 53 and the columns are 5 and 21 — exactly one of the
    // eight outputs exists. A guard combining rows would drop it; a guard combining columns
    // would write past the end of a row into the next one, which is the silent kind of wrong.
    let row0 = rowBase + ty + 0u * RSTEP;
    let row1 = rowBase + ty + 1u * RSTEP;
    let row2 = rowBase + ty + 2u * RSTEP;
    let row3 = rowBase + ty + 3u * RSTEP;
    let col0 = colBase + tx + 0u * CSTEP;
    let col1 = colBase + tx + 1u * CSTEP;

    if (row0 < dims.m && col0 < dims.n) { c[row0 * dims.n + col0] = acc00; }
    if (row0 < dims.m && col1 < dims.n) { c[row0 * dims.n + col1] = acc01; }
    if (row1 < dims.m && col0 < dims.n) { c[row1 * dims.n + col0] = acc10; }
    if (row1 < dims.m && col1 < dims.n) { c[row1 * dims.n + col1] = acc11; }
    if (row2 < dims.m && col0 < dims.n) { c[row2 * dims.n + col0] = acc20; }
    if (row2 < dims.m && col1 < dims.n) { c[row2 * dims.n + col1] = acc21; }
    if (row3 < dims.m && col0 < dims.n) { c[row3 * dims.n + col0] = acc30; }
    if (row3 < dims.m && col1 < dims.n) { c[row3 * dims.n + col1] = acc31; }
}
