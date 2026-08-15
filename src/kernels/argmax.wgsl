// Greedy sampling: the index of the largest logit.
//
// This runs on the GPU rather than reading 151,936 logits back to JavaScript, because the
// decode loop's budget is one readback per token — the chosen id — and 608 KB per token is
// not that. Only the four bytes this kernel writes cross back.
//
// Ties go to the lowest index, matching torch.argmax. That is not cosmetic: a tie broken
// differently is a different token, and the greedy output is supposed to be reproducible
// against the reference exactly.

const WG: u32 = 256u;

struct Dims {
    count: u32,
    /** Element offset of the row to reduce, so the last position can be selected. */
    offset: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> out:    array<u32>;
@group(0) @binding(2) var<uniform>             dims:   Dims;

var<workgroup> bestValue: array<f32, 256>;
var<workgroup> bestIndex: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;

    // Not f32::MIN as a literal — the exact value rounds outside the representable
    // range and fails to compile. Any sentinel below every real logit works.
    var value = -1.0e30;
    var index = 0u;
    for (var i = tid; i < dims.count; i = i + WG) {
        let candidate = logits[dims.offset + i];
        // Strictly greater, so the lowest index wins a tie within this thread's stride.
        if (candidate > value) {
            value = candidate;
            index = i;
        }
    }
    bestValue[tid] = value;
    bestIndex[tid] = index;
    workgroupBarrier();

    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            let other = tid + stride;
            // On equal values keep the smaller index, so ties resolve the same way they do
            // in the reference regardless of which thread found them.
            if (bestValue[other] > bestValue[tid] ||
                (bestValue[other] == bestValue[tid] && bestIndex[other] < bestIndex[tid])) {
                bestValue[tid] = bestValue[other];
                bestIndex[tid] = bestIndex[other];
            }
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        out[0] = bestIndex[0];
        out[1] = bitcast<u32>(bestValue[0]);
    }
}
