// Attention scores for one query against the whole cached history.
//
//     scores[h, j] = dot(q[h], cacheK[j, kv(h)]) / sqrt(headDim),  j <= position
//
// The decode shape. One query row, `position + 1` keys, and the keys are already in the cache
// with RoPE applied — that is the whole point of storing K post-rotation, since a key's
// rotation depends only on its own absolute position and never changes again.
//
// Scores are written with a stride of maxContext rather than position+1, because the buffer is
// preallocated to the maximum and the row length grows every step. The softmax that follows
// takes the stride separately from the length for the same reason.

const NEG_LARGE: f32 = -1.0e30;

struct Dims {
    /** Index of the token being generated; keys 0..position inclusive are valid. */
    position: u32,
    heads: u32,
    kvHeads: u32,
    headDim: u32,
    /** Row stride of the scores buffer, in elements. */
    scoreStride: u32,
    /** Row stride of the cache, in elements: kvHeads * headDim. */
    cacheStride: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       q:      array<f32>;
@group(0) @binding(1) var<storage, read>       cacheK: array<f32>;
@group(0) @binding(2) var<storage, read_write> out:    array<f32>;
@group(0) @binding(3) var<uniform>             dims:   Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let h = gid.y;
    if (h >= dims.heads || j >= dims.scoreStride) {
        return;
    }

    let slot = h * dims.scoreStride + j;
    if (j > dims.position) {
        // Past the end of the history. Written rather than skipped so the softmax below sees a
        // defined value in every slot of its row — the buffer holds the previous step's scores
        // otherwise, and those are finite and plausible.
        out[slot] = NEG_LARGE;
        return;
    }

    let kvHead = h / (dims.heads / dims.kvHeads);
    let qBase = h * dims.headDim;
    let kBase = j * dims.cacheStride + kvHead * dims.headDim;

    var acc = 0.0;
    for (var d = 0u; d < dims.headDim; d = d + 1u) {
        acc = acc + q[qBase + d] * cacheK[kBase + d];
    }
    out[slot] = acc * inverseSqrt(f32(dims.headDim));
}
