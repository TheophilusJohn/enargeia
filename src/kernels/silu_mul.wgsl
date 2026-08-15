// SwiGLU gate: out[i] = silu(gate[i]) * up[i], where silu(x) = x * sigmoid(x).
//
// Computed as x / (1 + exp(-x)). For large negative x, exp(-x) overflows to infinity and the
// quotient is 0 rather than NaN, which is the correct limit — so no clamping is needed. For
// large positive x, exp(-x) underflows to 0 and the result is x, also correct.

struct Dims {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       gate: array<f32>;
@group(0) @binding(1) var<storage, read>       up:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out:  array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= dims.count) {
        return;
    }
    let g = gate[gid.x];
    out[gid.x] = (g / (1.0 + exp(-g))) * up[gid.x];
}
