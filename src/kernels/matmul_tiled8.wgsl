// Tiled fp32 matmul, stage 3: thread coarsening, 8 outputs per thread.
//
// Same pattern as matmul_tiled4, doubled. Each thread owns an 8x1 column strip — eight
// output rows, one column, eight accumulators. One tileB load feeds eight multiply-adds, so
// the shared-load ratio falls from 5 loads / 4 MACs = 1.25 to 9 loads / 8 MACs = 1.125.
//
// That is a 1.11x predicted improvement, against 1.60x for stage 2. This kernel exists to
// find out whether the returns keep arriving at all, and what stops them. Two things get
// worse as the strip widens and neither shows up in the load-ratio model:
//
//   - tileA grows to 128x16 = 8 KiB, so with tileB the workgroup needs 9 KiB of shared
//     memory against stage 2's 5 KiB. Fewer workgroups fit on a core at once.
//   - eight accumulators plus addressing live in the register file, and occupancy is
//     register-limited too.
//
// The workgroup covers 128 rows x 16 columns, so the dispatch y-dimension divides by 128.
//
// Accumulation order is still k ascending, matching every other variant term for term.

const TILE_K: u32 = 16u;   // reduction depth staged per iteration
const TILE_N: u32 = 16u;   // columns of C per workgroup
const ROWS:   u32 = 128u;  // rows of C per workgroup
const STRIDE: u32 = 16u;   // gap between a thread's eight output rows

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

var<workgroup> tileA: array<f32, 2048>; // 128 rows x 16 k, row-major — 8 KiB
var<workgroup> tileB: array<f32, 256>;  // 16 k x 16 cols, row-major  — 1 KiB

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id)        wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tx = lid.x;
    let ty = lid.y;

    let rowBase = wid.y * ROWS;
    let col = wid.x * TILE_N + tx;

    let numTiles = (dims.k + TILE_K - 1u) / TILE_K;

    // Eight separate scalars rather than array<f32, 8>: a dynamically indexed local array is
    // backed by memory, not registers. Whether eight of these plus addressing actually stay
    // in registers is the open question this kernel is here to answer.
    var acc0 = 0.0;
    var acc1 = 0.0;
    var acc2 = 0.0;
    var acc3 = 0.0;
    var acc4 = 0.0;
    var acc5 = 0.0;
    var acc6 = 0.0;
    var acc7 = 0.0;

    for (var t = 0u; t < numTiles; t = t + 1u) {
        let aCol = t * TILE_K + tx;

        // Eight staged rows per thread, at tile rows ty, ty+16, ... ty+112. tx stays on the
        // fastest-varying memory axis so every load is 16 consecutive floats.
        //
        // The k guard (aCol < k) is load-bearing exactly as in stages 1 and 2: a skipped
        // store leaves the previous tile's value in the slot and it is multiplied in. The
        // row guard remains defensive under the reachability argument in
        // docs/tiled-matmul.md — tile row r is read only by threads with ty = r mod 16 using
        // accumulator r div 16, whose output row is rowBase + r, so garbage reaches only a
        // masked accumulator. Kept regardless: it costs nothing and stops an out-of-bounds
        // read from being silently clamped.
        for (var s = 0u; s < 8u; s = s + 1u) {
            let r = ty + s * STRIDE;
            let row = rowBase + r;
            if (row < dims.m && aCol < dims.k) {
                tileA[r * TILE_K + tx] = a[row * dims.k + aCol];
            } else {
                tileA[r * TILE_K + tx] = 0.0;
            }
        }

        let bRow = t * TILE_K + ty;
        if (bRow < dims.k && col < dims.n) {
            tileB[ty * TILE_N + tx] = b[bRow * dims.n + col];
        } else {
            tileB[ty * TILE_N + tx] = 0.0;
        }

        // Barrier 1: both tiles fully staged before anyone reads them.
        workgroupBarrier();

        for (var i = 0u; i < TILE_K; i = i + 1u) {
            let bVal = tileB[i * TILE_N + tx];
            acc0 = acc0 + tileA[(ty + 0u * STRIDE) * TILE_K + i] * bVal;
            acc1 = acc1 + tileA[(ty + 1u * STRIDE) * TILE_K + i] * bVal;
            acc2 = acc2 + tileA[(ty + 2u * STRIDE) * TILE_K + i] * bVal;
            acc3 = acc3 + tileA[(ty + 3u * STRIDE) * TILE_K + i] * bVal;
            acc4 = acc4 + tileA[(ty + 4u * STRIDE) * TILE_K + i] * bVal;
            acc5 = acc5 + tileA[(ty + 5u * STRIDE) * TILE_K + i] * bVal;
            acc6 = acc6 + tileA[(ty + 6u * STRIDE) * TILE_K + i] * bVal;
            acc7 = acc7 + tileA[(ty + 7u * STRIDE) * TILE_K + i] * bVal;
        }

        // Barrier 2: tiles fully consumed before iteration t+1 overwrites them.
        workgroupBarrier();
    }

    // One mask per accumulator. A thread's eight rows are 16 apart and span 113 rows of C, so
    // near the bottom edge almost any mix of in-range and out-of-range is possible — at
    // m = 20 with rowBase = 0 and ty = 5 the rows are 5, 21, 37, 53, 69, 85, 101 and 117, and
    // only the first exists. A single combined guard would drop seven real rows or write
    // seven past the end.
    if (col < dims.n) {
        let row0 = rowBase + ty + 0u * STRIDE;
        let row1 = rowBase + ty + 1u * STRIDE;
        let row2 = rowBase + ty + 2u * STRIDE;
        let row3 = rowBase + ty + 3u * STRIDE;
        let row4 = rowBase + ty + 4u * STRIDE;
        let row5 = rowBase + ty + 5u * STRIDE;
        let row6 = rowBase + ty + 6u * STRIDE;
        let row7 = rowBase + ty + 7u * STRIDE;
        if (row0 < dims.m) { c[row0 * dims.n + col] = acc0; }
        if (row1 < dims.m) { c[row1 * dims.n + col] = acc1; }
        if (row2 < dims.m) { c[row2 * dims.n + col] = acc2; }
        if (row3 < dims.m) { c[row3 * dims.n + col] = acc3; }
        if (row4 < dims.m) { c[row4 * dims.n + col] = acc4; }
        if (row5 < dims.m) { c[row5 * dims.n + col] = acc5; }
        if (row6 < dims.m) { c[row6 * dims.n + col] = acc6; }
        if (row7 < dims.m) { c[row7 * dims.n + col] = acc7; }
    }
}
