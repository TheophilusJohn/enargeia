// Weighted sum of the cached values for one query position.
//
//     out[h, d] = sum_{j <= position} weights[h, j] * cacheV[j, kv(h), d]
//
// One thread per (head, dim) output element, each walking the history. At a context of 1024
// that is 1024 iterations per thread over 896 threads, which is the decode shape: no reuse,
// bound by how fast the cache streams.

struct Dims {
    position: u32,
    heads: u32,
    kvHeads: u32,
    headDim: u32,
    scoreStride: u32,
    cacheStride: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       weights: array<f32>;
@group(0) @binding(1) var<storage, read>       cacheV:  array<f32>;
@group(0) @binding(2) var<storage, read_write> out:     array<f32>;
@group(0) @binding(3) var<uniform>             dims:    Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = dims.heads * dims.headDim;
    if (gid.x >= total) {
        return;
    }
    let h = gid.x / dims.headDim;
    let d = gid.x % dims.headDim;
    let kvHead = h / (dims.heads / dims.kvHeads);
    let wBase = h * dims.scoreStride;

    var acc = 0.0;
    for (var j = 0u; j <= dims.position; j = j + 1u) {
        acc = acc + weights[wBase + j] * cacheV[j * dims.cacheStride + kvHead * dims.headDim + d];
    }
    out[gid.x] = acc;
}
