// Embedding lookup from one part of a split, int4-quantized table.
//
// Same per-part dispatch structure as the fp32 gather: one part bound at a time, a thread
// writing its row only when the token's id falls in this part's range. What changes is that
// the row has to be dequantized as it is read.
//
// The part's arrays are offsets into the whole table, not standalone tensors, so the block
// index is computed from the *global* row. Splitting on a row boundary that is a multiple of
// eight blocks is what keeps the zero-point words from straddling a part — see
// `alignRowsPerPart` in enargeia.ts.

struct Dims {
    seq: u32,
    hidden: u32,
    /** First global row this part holds. */
    rowBegin: u32,
    /** Number of rows this part holds. */
    rowCount: u32,
    blockSize: u32,
    blocksPerRow: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       ids:    array<u32>;
@group(0) @binding(1) var<storage, read>       packed: array<u32>;
@group(0) @binding(2) var<storage, read>       scales: array<f32>;
@group(0) @binding(3) var<storage, read>       zeros:  array<u32>;
@group(0) @binding(4) var<storage, read_write> out:    array<f32>;
@group(0) @binding(5) var<uniform>             dims:   Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= dims.hidden || gid.y >= dims.seq) {
        return;
    }
    let id = ids[gid.y];
    if (id < dims.rowBegin || id >= dims.rowBegin + dims.rowCount) {
        return;
    }

    let localRow = id - dims.rowBegin;
    let flat = localRow * dims.hidden + gid.x;
    let word = packed[flat >> 3u];
    let nib = (word >> ((flat & 7u) * 4u)) & 0xFu;

    let block = localRow * dims.blocksPerRow + gid.x / dims.blockSize;
    let zeroWord = zeros[block >> 3u];
    let zero = (zeroWord >> ((block & 7u) * 4u)) & 0xFu;

    out[gid.y * dims.hidden + gid.x] = (f32(nib) - f32(zero)) * scales[block];
}
