// Embedding lookup from one part of the split table.
//
// The table is split across up to five bindings under the 128 MiB default, and a bind group
// holding all of them would use seven storage buffers before counting anything else. So this
// kernel binds exactly one part and is dispatched once per part: a thread writes its output
// row only when the token's id falls inside this part's row range.
//
// Every token id belongs to exactly one part, so across the full set of dispatches each
// output row is written exactly once. Rows whose id falls outside this part are left alone
// rather than zeroed — another dispatch owns them.

struct Dims {
    seq: u32,
    hidden: u32,
    /** First global row this part holds. */
    rowBegin: u32,
    /** Number of rows this part holds. */
    rowCount: u32,
};

@group(0) @binding(0) var<storage, read>       ids:  array<u32>;
@group(0) @binding(1) var<storage, read>       part: array<f32>;
@group(0) @binding(2) var<storage, read_write> out:  array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

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
    out[gid.y * dims.hidden + gid.x] = part[localRow * dims.hidden + gid.x];
}
