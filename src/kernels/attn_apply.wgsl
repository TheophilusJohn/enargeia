// out[i, h, d] = sum_j weights[h, i, j] * v[j, kv(h), d]
//
// The sum runs only to j <= i. Masked weights are already zero after softmax, so summing the
// whole row would give the same answer — but the causal bound is kept explicit because it is
// the property being relied on, and a softmax bug that leaves a masked weight non-zero should
// surface as wrong attention output rather than being quietly absorbed here.

struct Dims {
    seq: u32,
    heads: u32,
    kvHeads: u32,
    headDim: u32,
};

@group(0) @binding(0) var<storage, read>       weights: array<f32>;
@group(0) @binding(1) var<storage, read>       v:       array<f32>;
@group(0) @binding(2) var<storage, read_write> out:     array<f32>;
@group(0) @binding(3) var<uniform>             dims:    Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let perRow = dims.heads * dims.headDim;
    if (gid.x >= perRow || gid.y >= dims.seq) {
        return;
    }
    let i = gid.y;
    let h = gid.x / dims.headDim;
    let d = gid.x % dims.headDim;

    let kvHead = h / (dims.heads / dims.kvHeads);
    let wBase = (h * dims.seq + i) * dims.seq;

    var acc = 0.0;
    for (var j = 0u; j <= i; j = j + 1u) {
        acc = acc + weights[wBase + j] * v[(j * dims.kvHeads + kvHead) * dims.headDim + d];
    }
    out[i * perRow + gid.x] = acc;
}
