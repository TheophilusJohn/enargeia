// int4 block-quantized projection, decode shape: out[n] = sum_k x[k] * W[n, k] + bias[n]
//
// Matrix by vector. Every weight is read once and used once, so arithmetic intensity is
// near 1 and the kernel is entirely memory-bound. Tiling buys nothing here: there is no reuse
// to amortize, and staging a weight into shared memory only to read it once is pure overhead.
//
// So this is the shape the kernels skill describes literally — dequantize inside the
// accumulation loop, in registers, never touching a buffer:
//
//     let word = packed[flat >> 3u];
//     let nib  = (word >> ((flat & 7u) * 4u)) & 0xFu;
//     acc += (f32(nib) - f32(zero)) * scale * x[i];
//
// One workgroup per output element, 64 threads splitting the reduction. That mapping is what
// keeps the reads coalesced: adjacent threads read adjacent nibbles of the same weight row,
// so a 64-thread group covers 8 consecutive u32 words rather than gathering.
//
// The scale and zero-point are hoisted out of the innermost work: they change once every 64
// elements, and reloading them per element would triple the loads for a value that has not
// changed.

const WG: u32 = 64u;

struct Dims {
    m: u32,
    n: u32,
    k: u32,
    useBias: u32,
    blockSize: u32,
    blocksPerRow: u32,
    outStride: u32,
    outOffset: u32,
    /**
     * Workgroups along x. One workgroup per output element means 151,936 of them for the tied
     * LM head, and `maxComputeWorkgroupsPerDimension` is 65,535 — so the grid is folded into
     * two dimensions and unfolded here. Exceeding the limit does not throw at encode time; it
     * invalidates the whole command buffer, and the symptom is a forward pass that silently
     * produces zeros.
     */
    gridWidth: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       x:      array<f32>;
@group(0) @binding(1) var<storage, read>       packed: array<u32>;
@group(0) @binding(2) var<storage, read>       scales: array<f32>;
@group(0) @binding(3) var<storage, read>       zeros:  array<u32>;
@group(0) @binding(4) var<storage, read>       bias:   array<f32>;
@group(0) @binding(5) var<storage, read_write> out:    array<f32>;
@group(0) @binding(6) var<uniform>             dims:   Dims;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id)        wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let col = wid.y * dims.gridWidth + wid.x;
    let tid = lid.x;
    // No early return: the reduction below has barriers, and they must stay in uniform
    // control flow. Out-of-range columns run and are masked at the write.
    let colActive = col < dims.n;

    var acc = 0.0;
    if (colActive) {
        let rowBase = col * dims.k;
        let blockBase = col * dims.blocksPerRow;

        for (var i = tid; i < dims.k; i = i + WG) {
            let flat = rowBase + i;
            let word = packed[flat >> 3u];
            let nib = (word >> ((flat & 7u) * 4u)) & 0xFu;

            let block = blockBase + i / dims.blockSize;
            let zeroWord = zeros[block >> 3u];
            let zero = (zeroWord >> ((block & 7u) * 4u)) & 0xFu;

            acc = acc + (f32(nib) - f32(zero)) * scales[block] * x[i];
        }
    }

    partial[tid] = acc;
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            partial[tid] = partial[tid] + partial[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u && colActive) {
        var value = partial[0];
        if (dims.useBias != 0u) {
            value = value + bias[col];
        }
        out[col + dims.outOffset] = value;
    }
}
