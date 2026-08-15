// Sampling, entirely on the GPU: repetition penalty, temperature, top-p, and the draw.
//
// The decode loop's budget is one readback per token and it is four bytes — the chosen id.
// Reading 151,936 logits back to pick a token would be 608 KB and a mapAsync round trip on
// the critical path, which is the cost this whole design exists to avoid. So everything below
// happens in one workgroup of 256 threads and only `out[0]` crosses back.
//
// WHY A THRESHOLD SEARCH AND NOT A SORT
//
// Top-p wants the smallest set of tokens whose probabilities sum to at least p, which is
// naturally phrased as "sort descending, take a prefix". A full bitonic sort of 151,936
// elements is ~log2(n)^2 / 2 = 145 passes over the array, all of them global, and it computes
// a total order when all that is needed is a cut point.
//
// The equivalent question is: find the probability threshold t such that the mass above t is
// just at least p. That is monotone in t, so a bisection converges — 32 iterations pins t to
// within 2^-32 of the range, and each iteration is one reduction over the array. 32 passes
// against 145, no scratch storage, no ordering network.
//
// The cost is that ties at exactly t are all kept or all dropped together, so the retained
// mass can overshoot p slightly when many tokens share a probability. For sampling that is
// harmless — the set is still a superset of the true nucleus by at most the tied group — and
// it is deterministic, which the alternative approximations are not.
//
// A partial top-k reduction was the other candidate. It is a better fit when k is small and
// known, but top-p's k is data-dependent and can be in the thousands on a flat distribution,
// at which point maintaining k candidates in shared memory stops fitting.

const WG: u32 = 256u;
const BISECTION_STEPS: u32 = 32u;

struct Dims {
    vocab: u32,
    /** Tokens in the history, for the repetition penalty. */
    historyLength: u32,
    /** Divides the logits. 0 means greedy and skips the whole sampling path. */
    temperature: f32,
    /** Nucleus mass. 1.0 keeps everything. */
    topP: f32,
    /** Logits of tokens already seen are divided by this. 1.0 disables it. */
    repetitionPenalty: f32,
    /** Uniform random in [0, 1), drawn on the CPU so the sequence is seedable. */
    random: f32,
    _pad0: u32,
    _pad1: u32,
};

// `read_write` because the repetition penalty is applied into this buffer once, in place,
// rather than recomputed on every read. See the note above `applyRepetitionPenalty`.
@group(0) @binding(0) var<storage, read_write> logits:  array<f32>;
@group(0) @binding(1) var<storage, read>       history: array<u32>;
@group(0) @binding(2) var<storage, read_write> out:     array<u32>;
@group(0) @binding(3) var<uniform>             dims:    Dims;

var<workgroup> reduce: array<f32, 256>;
var<workgroup> reduceIndex: array<u32, 256>;
var<workgroup> sharedMax: f32;
var<workgroup> sharedSum: f32;
var<workgroup> sharedThreshold: f32;
var<workgroup> sharedTarget: f32;
var<workgroup> sharedArgmax: u32;
var<workgroup> chosen: atomic<u32>;

/// Contiguous chunk per thread, so concatenating the chunks reproduces ascending token id.
/// Strided assignment would be better for coalescing but would make the final scan visit ids
/// out of order, and the sampled token depends on that order for a given random draw.
fn chunkBegin(tid: u32, vocab: u32) -> u32 {
    let per = (vocab + WG - 1u) / WG;
    return min(tid * per, vocab);
}
fn chunkEnd(tid: u32, vocab: u32) -> u32 {
    let per = (vocab + WG - 1u) / WG;
    return min((tid + 1u) * per, vocab);
}

/// Apply the repetition penalty in place, once, before anything reads a logit.
///
/// The obvious formulation — a helper that penalizes a logit at the point of reading it — was
/// what this kernel did first, and it is quadratic in the worst way: every read scans the whole
/// history, and the vocabulary is read about 36 times (max, sum, 32 bisection steps, the draw).
/// At a 500-token history that is 2.7 billion comparisons per token. Measured through the app
/// it cost **26 ms/token against 71** — the penalty was 2.7× the entire rest of decoding.
///
/// Applied here instead, the cost is one pass over the history rather than 36 over the
/// vocabulary, and every later read is a plain load.
///
/// Threads split the history. Only the first occurrence of an id applies the penalty, which is
/// what the previous formulation did (it stopped at the first match) and what the reference
/// does — it penalizes a set, not a multiset. That also means no two threads ever write the
/// same address, so the in-place update needs no atomics.
fn applyRepetitionPenalty(tid: u32) {
    if (dims.repetitionPenalty == 1.0) {
        return;
    }
    for (var h = tid; h < dims.historyLength; h = h + WG) {
        let id = history[h];
        var first = true;
        for (var earlier = 0u; earlier < h; earlier = earlier + 1u) {
            if (history[earlier] == id) {
                first = false;
                break;
            }
        }
        if (first && id < dims.vocab) {
            let value = logits[id];
            // Divide when positive, multiply when negative — dividing a negative logit would
            // make the token *more* likely, which is the opposite of a penalty.
            logits[id] = select(value * dims.repetitionPenalty,
                                value / dims.repetitionPenalty,
                                value > 0.0);
        }
    }
}

fn reduceMax(tid: u32, value: f32) -> f32 {
    reduce[tid] = value;
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            reduce[tid] = max(reduce[tid], reduce[tid + stride]);
        }
        workgroupBarrier();
    }
    let result = reduce[0];
    workgroupBarrier();
    return result;
}

fn reduceSum(tid: u32, value: f32) -> f32 {
    reduce[tid] = value;
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            reduce[tid] = reduce[tid] + reduce[tid + stride];
        }
        workgroupBarrier();
    }
    let result = reduce[0];
    workgroupBarrier();
    return result;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let vocab = dims.vocab;
    let begin = chunkBegin(tid, vocab);
    let end = chunkEnd(tid, vocab);

    // Sentinel, not zero: the draw below uses atomicMin, and a zero initial value would win
    // every comparison and pin the result to token 0.
    if (tid == 0u) {
        atomicStore(&chosen, 0xFFFFFFFFu);
    }

    // Before any read of a logit. The barriers are outside the helper so they sit in uniform
    // control flow, and the storage barrier is what makes one thread's write visible to the
    // rest of the workgroup.
    applyRepetitionPenalty(tid);
    storageBarrier();
    workgroupBarrier();

    // --- greedy ---------------------------------------------------------
    // Temperature 0 is argmax, and it takes the short path rather than dividing by zero.
    if (dims.temperature <= 0.0) {
        var bestValue = -1.0e30;
        var bestIndex = 0u;
        for (var i = begin; i < end; i = i + 1u) {
            let v = logits[i];
            if (v > bestValue) {
                bestValue = v;
                bestIndex = i;
            }
        }
        reduce[tid] = bestValue;
        reduceIndex[tid] = bestIndex;
        workgroupBarrier();
        for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
            if (tid < stride) {
                let other = tid + stride;
                // Ties go to the lower id, matching torch.argmax and keeping greedy decode
                // reproducible against the reference.
                if (reduce[other] > reduce[tid] ||
                    (reduce[other] == reduce[tid] && reduceIndex[other] < reduceIndex[tid])) {
                    reduce[tid] = reduce[other];
                    reduceIndex[tid] = reduceIndex[other];
                }
            }
            workgroupBarrier();
        }
        if (tid == 0u) {
            out[0] = reduceIndex[0];
        }
        return;
    }

    // --- softmax over the whole vocabulary ------------------------------
    var localMax = -1.0e30;
    var localArg = begin;
    for (var i = begin; i < end; i = i + 1u) {
        let scaled = logits[i] / dims.temperature;
        if (scaled > localMax) {
            localMax = scaled;
            localArg = i;
        }
    }
    // Max and argmax in one reduction: the argmax is the fallback if the draw somehow finds
    // no crossing, and computing it here costs nothing over the max pass that must happen
    // anyway for a numerically safe softmax.
    reduce[tid] = localMax;
    reduceIndex[tid] = localArg;
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            let other = tid + stride;
            if (reduce[other] > reduce[tid] ||
                (reduce[other] == reduce[tid] && reduceIndex[other] < reduceIndex[tid])) {
                reduce[tid] = reduce[other];
                reduceIndex[tid] = reduceIndex[other];
            }
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        sharedMax = reduce[0];
        sharedArgmax = reduceIndex[0];
    }
    workgroupBarrier();

    var localSum = 0.0;
    for (var i = begin; i < end; i = i + 1u) {
        localSum = localSum + exp(logits[i] / dims.temperature - sharedMax);
    }
    let total = reduceSum(tid, localSum);
    if (tid == 0u) { sharedSum = total; }
    workgroupBarrier();

    // --- bisection for the nucleus threshold ----------------------------
    // Mass above a threshold is monotone decreasing in the threshold, so bisection finds the
    // cut. Working in normalized probability keeps the bracket independent of the vocabulary.
    var lo = 0.0;
    var hi = 1.0;
    if (dims.topP < 1.0) {
        for (var step = 0u; step < BISECTION_STEPS; step = step + 1u) {
            let mid = (lo + hi) * 0.5;
            var localMass = 0.0;
            for (var i = begin; i < end; i = i + 1u) {
                let p = exp(logits[i] / dims.temperature - sharedMax) / sharedSum;
                if (p >= mid) {
                    localMass = localMass + p;
                }
            }
            let mass = reduceSum(tid, localMass);
            // Too little mass kept: lower the bar. Enough: raise it and keep a tighter set.
            if (mass < dims.topP) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
    }
    let threshold = select(0.0, lo, dims.topP < 1.0);
    if (tid == 0u) { sharedThreshold = threshold; }
    workgroupBarrier();

    // Mass actually retained, which is what the draw is scaled against — renormalizing over
    // the nucleus rather than the full distribution is what top-p means.
    var localKept = 0.0;
    for (var i = begin; i < end; i = i + 1u) {
        let p = exp(logits[i] / dims.temperature - sharedMax) / sharedSum;
        if (p >= sharedThreshold) {
            localKept = localKept + p;
        }
    }
    let kept = reduceSum(tid, localKept);
    if (tid == 0u) {
        sharedTarget = dims.random * max(kept, 1.0e-20);
    }
    workgroupBarrier();

    // --- the draw -------------------------------------------------------
    // Two-level scan: each thread totals its own contiguous chunk, an exclusive scan over the
    // 256 partials gives each thread the mass before its chunk, then each thread walks its
    // chunk to find where the running total crosses the target. Because the chunks are
    // contiguous and in id order, this is the same token a sequential scan would pick.
    var mine = 0.0;
    for (var i = begin; i < end; i = i + 1u) {
        let p = exp(logits[i] / dims.temperature - sharedMax) / sharedSum;
        if (p >= sharedThreshold) {
            mine = mine + p;
        }
    }
    reduce[tid] = mine;
    workgroupBarrier();

    // Exclusive prefix sum over 256 partials, done serially by one thread: 256 iterations is
    // nothing next to the 594 elements each thread already walked, and a Hillis-Steele scan
    // here would need a second buffer to avoid racing.
    if (tid == 0u) {
        var running = 0.0;
        for (var t = 0u; t < WG; t = t + 1u) {
            let value = reduce[t];
            reduce[t] = running;
            running = running + value;
        }
    }
    workgroupBarrier();

    var running = reduce[tid];
    for (var i = begin; i < end; i = i + 1u) {
        let p = exp(logits[i] / dims.temperature - sharedMax) / sharedSum;
        if (p >= sharedThreshold) {
            running = running + p;
            if (running >= sharedTarget) {
                // Lowest crossing id wins. Several threads can cross when floating-point
                // rounding puts the target on a boundary, so the minimum keeps it defined.
                atomicMin(&chosen, i);
                break;
            }
        }
    }
    workgroupBarrier();

    if (tid == 0u) {
        let picked = atomicLoad(&chosen);
        // The running total over the kept set ends at `kept`, and the target is a fraction of
        // `kept`, so a crossing is guaranteed. The fallback covers only the case where the
        // retained mass underflows to zero, and it picks the most likely token rather than
        // emitting whatever id the sentinel happens to be.
        out[0] = select(picked, sharedArgmax, picked == 0xFFFFFFFFu);
    }
}
