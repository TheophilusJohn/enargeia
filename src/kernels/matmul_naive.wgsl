// Naive fp32 matmul: C[m,n] = A[m,k] * B[k,n], both row-major.
//
// One invocation per output element, reading a full row of A and a full column of B from
// global memory. Every value of A is re-read n times and every value of B m times, so this
// is the arithmetic-intensity floor: it exists as the baseline the tiled kernel is measured
// against, and as the obviously-correct thing to compare against when tiling goes wrong.

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

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    // Dispatch geometry rounds up to whole workgroups, so the last one runs out of range.
    // No shared memory here, so an early return is safe — it would not be if this kernel
    // had a barrier.
    if (row >= dims.m || col >= dims.n) {
        return;
    }

    // Row and column bases are hoisted so the loop carries one multiply-add each.
    let aBase = row * dims.k;
    var acc = 0.0;
    for (var i = 0u; i < dims.k; i = i + 1u) {
        acc = acc + a[aBase + i] * b[i * dims.n + col];
    }
    c[row * dims.n + col] = acc;
}
