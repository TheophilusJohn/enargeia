// RMSNorm: y[s, f] = x[s, f] * rsqrt(mean(x[s, :]^2) + eps) * weight[f]
//
// One workgroup per row. The reduction is over `features` (896 for Qwen2.5-0.5B), which is
// more than one workgroup of 256 can hold one-per-thread, so each thread accumulates a
// strided slice first and the workgroup then reduces 256 partials in shared memory.
//
// Note it is the *mean* of squares, not the sum. Using the sum gives a norm that is off by
// sqrt(features) — a factor of 30 here — which is obvious. Using sum/features vs
// sum*(1/features) is not, and both are correct; the reference uses the former.

const WG: u32 = 256u;

struct Dims {
    seq: u32,
    features: u32,
    eps: f32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>       x:      array<f32>;
@group(0) @binding(1) var<storage, read>       weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out:    array<f32>;
@group(0) @binding(3) var<uniform>             dims:   Dims;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id)        wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.x;
    let tid = lid.x;
    // Rows past the current sequence length still run: the dispatch is sized for the maximum
    // sequence, and returning early here would put the barriers below in non-uniform control
    // flow across the workgroup. The row is masked at the write instead.
    let rowActive = row < dims.seq;
    let base = row * dims.features;

    var sum = 0.0;
    if (rowActive) {
        for (var f = tid; f < dims.features; f = f + WG) {
            let v = x[base + f];
            sum = sum + v * v;
        }
    }
    partial[tid] = sum;
    workgroupBarrier();

    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            partial[tid] = partial[tid] + partial[tid + stride];
        }
        workgroupBarrier();
    }

    let scale = inverseSqrt(partial[0] / f32(dims.features) + dims.eps);
    if (rowActive) {
        for (var f = tid; f < dims.features; f = f + WG) {
            out[base + f] = x[base + f] * scale * weight[f];
        }
    }
}
