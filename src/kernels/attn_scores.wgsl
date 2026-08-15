// Attention scores with a causal mask: scores[h, i, j] = dot(q[i,h], k[j,kv(h)]) / sqrt(d)
//
// Grouped-query attention. 14 query heads share 2 key/value heads, so query head h reads kv
// head h / (heads / kvHeads) — heads 0..6 use kv head 0, heads 7..13 use kv head 1. The
// grouping is contiguous, matching how the projections are laid out; interleaving it instead
// (h % kvHeads) is a silent corruption that keeps every value finite.
//
// Masked positions are written as a large negative rather than -inf. The softmax that follows
// subtracts the row max, and -inf minus -inf is NaN if an entire row is masked; a finite
// sentinel exponentiates to exactly zero and cannot poison the row.

const NEG_LARGE: f32 = -1.0e30;

struct Dims {
    seq: u32,
    heads: u32,
    kvHeads: u32,
    headDim: u32,
};

@group(0) @binding(0) var<storage, read>       q:    array<f32>;
@group(0) @binding(1) var<storage, read>       k:    array<f32>;
@group(0) @binding(2) var<storage, read_write> out:  array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let i = gid.y;
    let h = gid.z;
    if (j >= dims.seq || i >= dims.seq || h >= dims.heads) {
        return;
    }

    let slot = (h * dims.seq + i) * dims.seq + j;
    if (j > i) {
        out[slot] = NEG_LARGE;
        return;
    }

    let kvHead = h / (dims.heads / dims.kvHeads);
    let qBase = (i * dims.heads + h) * dims.headDim;
    let kBase = (j * dims.kvHeads + kvHead) * dims.headDim;

    var acc = 0.0;
    for (var d = 0u; d < dims.headDim; d = d + 1u) {
        acc = acc + q[qBase + d] * k[kBase + d];
    }
    out[slot] = acc * inverseSqrt(f32(dims.headDim));
}
