// Row-wise softmax over the last dimension.
//
// One workgroup per row, two reductions: the maximum, then the sum of exponentials. The max
// subtraction is not an optimization — attention scores reach magnitudes where exp() saturates
// to infinity, and infinity divided by infinity is NaN. Every row here has at least one
// unmasked element (the diagonal), but a fully masked row still yields zeros rather than NaN.

const WG: u32 = 256u;
const NEG_LARGE: f32 = -1.0e30;

struct Dims {
    rows: u32,
    /** Elements per row that are live. */
    cols: u32,
    /**
     * Elements between row starts. Equal to `cols` during prefill; during decode the scores
     * buffer is preallocated to the maximum context while the live row grows each step, so the
     * two differ and conflating them reads the wrong row.
     */
    stride: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read>       x:    array<f32>;
@group(0) @binding(1) var<storage, read_write> out:  array<f32>;
@group(0) @binding(2) var<uniform>             dims: Dims;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id)        wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.x;
    let tid = lid.x;
    let rowActive = row < dims.rows;
    let base = row * dims.stride;

    var localMax = NEG_LARGE;
    if (rowActive) {
        for (var c = tid; c < dims.cols; c = c + WG) {
            localMax = max(localMax, x[base + c]);
        }
    }
    scratch[tid] = localMax;
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        workgroupBarrier();
    }
    let rowMax = scratch[0];
    workgroupBarrier();

    var localSum = 0.0;
    if (rowActive) {
        for (var c = tid; c < dims.cols; c = c + WG) {
            let e = exp(x[base + c] - rowMax);
            out[base + c] = e;
            localSum = localSum + e;
        }
    }
    scratch[tid] = localSum;
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            scratch[tid] = scratch[tid] + scratch[tid + stride];
        }
        workgroupBarrier();
    }
    let total = scratch[0];

    if (rowActive && total > 0.0) {
        let inv = 1.0 / total;
        for (var c = tid; c < dims.cols; c = c + WG) {
            out[base + c] = out[base + c] * inv;
        }
    }
}
