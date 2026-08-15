// Tiled fp32 matmul, stage 1: C[m,n] = A[m,k] * B[k,n], both row-major.
//
// Same result as matmul_naive, same one-thread-per-output-element mapping. The difference
// is where the operands come from. The naive kernel reads a full row of A and a full column
// of B from global memory per output element, so every value of A is fetched n times and
// every value of B m times. Here the workgroup cooperatively stages a 16x16 block of each
// operand into shared memory, and every staged value is then read by 16 threads.
//
// Arithmetic intensity per output element goes from 2 global loads per multiply-add to
// 2/16 — one eighth of the traffic for identical arithmetic.
//
// Accumulation order is unchanged from the naive kernel: k ascending, one term at a time.
// That is deliberate. It makes the two kernels bit-comparable, so a mismatch is a bug in
// the tiling rather than a difference in floating-point association.

const TILE: u32 = 16u;

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

var<workgroup> tileA: array<f32, 256>; // 16x16 block of A
var<workgroup> tileB: array<f32, 256>; // 16x16 block of B

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    let ty = lid.y;
    let tx = lid.x;

    // No early return. Threads outside the output still load tiles, still reach both
    // barriers, and are masked only at the final write. Returning here instead would put a
    // barrier in non-uniform control flow, which is undefined behaviour: the threads that
    // stayed would wait at a barrier the departed threads never reach.
    //
    // numTiles depends only on the uniform, so every thread in the workgroup runs the loop
    // the same number of times and the barriers stay uniform.
    let numTiles = (dims.k + TILE - 1u) / TILE;
    var acc = 0.0;

    for (var t = 0u; t < numTiles; t = t + 1u) {
        // Thread (tx, ty) stages one element of each tile. For A the thread's x index walks
        // k and its y index walks the output row; for B it is the other way round, because
        // the two operands are contracted along k from opposite sides. Both loads are
        // consecutive in x across the workgroup, so both are coalesced.
        let aCol = t * TILE + tx;
        let bRow = t * TILE + ty;

        // Zero-fill out of range rather than skipping the write. Shared memory is not
        // cleared between tile iterations, so a skipped write leaves the previous tile's
        // value in place and it gets multiplied into this tile's accumulation. Zero is the
        // identity for the sum, so padding beyond the edge of the matrix contributes
        // nothing.
        if (row < dims.m && aCol < dims.k) {
            tileA[ty * TILE + tx] = a[row * dims.k + aCol];
        } else {
            tileA[ty * TILE + tx] = 0.0;
        }

        if (bRow < dims.k && col < dims.n) {
            tileB[ty * TILE + tx] = b[bRow * dims.n + col];
        } else {
            tileB[ty * TILE + tx] = 0.0;
        }

        // Barrier 1: the tile is fully staged. Without it a thread could read a slot its
        // neighbour has not written yet.
        workgroupBarrier();

        for (var i = 0u; i < TILE; i = i + 1u) {
            acc = acc + tileA[ty * TILE + i] * tileB[i * TILE + tx];
        }

        // Barrier 2: the tile is fully consumed, so the next iteration may overwrite it.
        // Without it a fast thread races ahead and stages tile t+1 into slots a slow thread
        // is still reading for tile t. The result is small, intermittent, device-dependent
        // error — correct on most runs, which is what makes it expensive to find.
        workgroupBarrier();
    }

    if (row < dims.m && col < dims.n) {
        c[row * dims.n + col] = acc;
    }
}
