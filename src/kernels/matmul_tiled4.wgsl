// Tiled fp32 matmul, stage 2: thread coarsening, 4 outputs per thread.
//
// Stage 1 gave every thread one output and read two shared-memory values per multiply-add.
// That is the same 2:1 load-to-arithmetic ratio the naive kernel had against global memory,
// moved one level down the hierarchy — so shared memory became the new bottleneck.
//
// Here each thread owns a 4x1 column strip: four output rows, one column, four accumulators
// living in registers. The inner loop loads a tileB value once and multiplies it into all
// four, so the ratio drops from 2 shared loads per MAC to 5 loads per 4 MACs = 1.25.
//
// The workgroup is still 16x16 threads but now covers 64 rows x 16 columns of output, so the
// dispatch y-dimension is ceil(m / 64) rather than ceil(m / 16).
//
// Accumulation order is still k ascending, matching matmul_naive and matmul_tiled term for
// term, so all three agree bit for bit and any mismatch is a bug rather than reassociation.

const TILE_K: u32 = 16u;  // reduction depth staged per iteration
const TILE_N: u32 = 16u;  // columns of C per workgroup
const ROWS:   u32 = 64u;  // rows of C per workgroup
const STRIDE: u32 = 16u;  // gap between a thread's four output rows

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

var<workgroup> tileA: array<f32, 1024>; // 64 rows x 16 k, row-major
var<workgroup> tileB: array<f32, 256>;  // 16 k x 16 cols, row-major

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id)        wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tx = lid.x;
    let ty = lid.y;

    // The workgroup's output footprint. rowBase cannot come from global_invocation_id any
    // more: there are 16 threads in y covering 64 rows, so the two no longer coincide.
    let rowBase = wid.y * ROWS;
    let col = wid.x * TILE_N + tx;

    let numTiles = (dims.k + TILE_K - 1u) / TILE_K;

    // Four accumulators as separate scalars, not array<f32, 4>. A dynamically indexed local
    // array is backed by memory rather than registers, which would hand back exactly the
    // traffic this kernel exists to avoid.
    var acc0 = 0.0;
    var acc1 = 0.0;
    var acc2 = 0.0;
    var acc3 = 0.0;

    for (var t = 0u; t < numTiles; t = t + 1u) {
        let aCol = t * TILE_K + tx;

        // Each thread stages four rows of tileA — its own four output rows — and one element
        // of tileB. tx walks k for A and the column for B, so both loads are 16 consecutive
        // floats across the thread's row of the workgroup.
        //
        // Both halves of every guard matter here. The k guard (aCol < k) is load-bearing for
        // the same reason as stage 1: a skipped store leaves the previous tile's value in the
        // slot and it gets multiplied in. The row guard is what keeps a partially filled
        // 64-row footprint from reading garbage into a tile row, and with four rows per
        // thread a workgroup straddles the bottom edge of C far more often than a 16-row one
        // did — any m not a multiple of 64 leaves some of these rows past the end.
        for (var s = 0u; s < 4u; s = s + 1u) {
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
            // One shared load of B, four multiply-adds. This is the whole point of stage 2:
            // the value is in a register and every accumulator that needs it takes it from
            // there rather than going back to shared memory.
            let bVal = tileB[i * TILE_N + tx];
            acc0 = acc0 + tileA[(ty + 0u  * STRIDE) * TILE_K + i] * bVal;
            acc1 = acc1 + tileA[(ty + 1u  * STRIDE) * TILE_K + i] * bVal;
            acc2 = acc2 + tileA[(ty + 2u  * STRIDE) * TILE_K + i] * bVal;
            acc3 = acc3 + tileA[(ty + 3u  * STRIDE) * TILE_K + i] * bVal;
        }

        // Barrier 2: tiles fully consumed before iteration t+1 overwrites them. Without it a
        // fast thread stages tile t+1 over slots a slow thread is still reading for tile t —
        // intermittent, device-dependent, and the single most common bug in this codebase.
        workgroupBarrier();
    }

    // One mask per accumulator, not one mask for the thread. A thread's four rows are 16
    // apart, so near the bottom edge of C some are in range and some are not: at m = 20 with
    // rowBase = 0 and ty = 5 the rows are 5, 21, 37, 53 and only the first survives. A single
    // combined guard would either drop row 5 or write three rows past the end of the matrix.
    if (col < dims.n) {
        let row0 = rowBase + ty + 0u * STRIDE;
        let row1 = rowBase + ty + 1u * STRIDE;
        let row2 = rowBase + ty + 2u * STRIDE;
        let row3 = rowBase + ty + 3u * STRIDE;
        if (row0 < dims.m) { c[row0 * dims.n + col] = acc0; }
        if (row1 < dims.m) { c[row1 * dims.n + col] = acc1; }
        if (row2 < dims.m) { c[row2 * dims.n + col] = acc2; }
        if (row3 < dims.m) { c[row3 * dims.n + col] = acc3; }
    }
}
