// Linear projection: out[m, n] = sum_k x[m, k] * w[n, k] + bias[n]
//
// Note that w is indexed [n, k], not [k, n]. Checkpoint linear weights are stored
// [outFeatures, inFeatures] row-major, so the reduction runs along a weight *row* and the
// matmul reads its second operand transposed. Using a plain A@B kernel here produces
// plausible-looking garbage — every value finite, every value wrong.
//
// Bias is always bound, even when the layer has none, so one bind group layout serves every
// projection. `useBias` selects it rather than a second pipeline.

struct Dims {
    m: u32,
    n: u32,
    k: u32,
    useBias: u32,
    // Where this dispatch's output lands: out[row * outStride + col + outOffset].
    //
    // The tied LM head needs this. Its five dispatches each write a disjoint slice of one
    // logits vector, and the slice boundaries land at row 30,388 — which is not a multiple
    // of 64, so a 256-byte-aligned *binding* offset cannot express it. Offsetting inside the
    // shader sidesteps the alignment rule entirely. Everywhere else outStride is n and
    // outOffset is 0.
    outStride: u32,
    outOffset: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       x:    array<f32>;
@group(0) @binding(1) var<storage, read>       w:    array<f32>;
@group(0) @binding(2) var<storage, read>       bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out:  array<f32>;
@group(0) @binding(4) var<uniform>             dims: Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= dims.m || col >= dims.n) {
        return;
    }

    let xBase = row * dims.k;
    let wBase = col * dims.k;
    var acc = 0.0;
    for (var i = 0u; i < dims.k; i = i + 1u) {
        acc = acc + x[xBase + i] * w[wBase + i];
    }
    if (dims.useBias != 0u) {
        acc = acc + bias[col];
    }
    out[row * dims.outStride + col + dims.outOffset] = acc;
}
