// Residual add: out[i] = a[i] + b[i].

struct Dims {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       a:    array<f32>;
@group(0) @binding(1) var<storage, read>       b:    array<f32>;
@group(0) @binding(2) var<storage, read_write> out:  array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= dims.count) {
        return;
    }
    out[gid.x] = a[gid.x] + b[gid.x];
}
