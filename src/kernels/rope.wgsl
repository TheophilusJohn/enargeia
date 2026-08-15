// Rotary position embedding over [seq, heads, headDim].
//
// Qwen rotates *halves*: element i pairs with element i + headDim/2. Pairing adjacent
// elements (i with i+1) is the classic RoPE bug — magnitudes stay plausible, directions are
// wrong, and the model produces fluent text with degraded attention. The CPU reference and
// the PyTorch dump both use the half layout, and layer0.q_rope catches the difference.

struct Dims {
    seq: u32,
    heads: u32,
    headDim: u32,
    positionOffset: u32,
    theta: f32,
    /**
     * Element offset of the output. Decode rotates one token's K straight into the cache at
     * `position * kvHeads * headDim`, so the projection, the rotation and the append stay the
     * three dispatches they already were — no copy, no separate append kernel.
     */
    outOffset: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       x:    array<f32>;
@group(0) @binding(1) var<storage, read_write> out:  array<f32>;
@group(0) @binding(2) var<uniform>             dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half = dims.headDim / 2u;
    let perRow = dims.heads * half;
    let total = dims.seq * perRow;
    if (gid.x >= total) {
        return;
    }

    let s = gid.x / perRow;
    let rem = gid.x % perRow;
    let h = rem / half;
    let i = rem % half;

    let position = f32(s + dims.positionOffset);
    let freq = pow(dims.theta, -2.0 * f32(i) / f32(dims.headDim));
    let angle = position * freq;
    let c = cos(angle);
    let sn = sin(angle);

    let base = (s * dims.heads + h) * dims.headDim;
    let a = x[base + i];
    let b = x[base + i + half];
    out[dims.outOffset + base + i] = a * c - b * sn;
    out[dims.outOffset + base + i + half] = b * c + a * sn;
}
