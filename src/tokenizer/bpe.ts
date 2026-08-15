/**
 * Byte-pair encoding over a single pre-tokenized piece.
 *
 * The merge list is ordered: rank 0 was learned first and is applied first. Encoding
 * repeatedly finds the lowest-ranked adjacent pair in the current symbol list and merges
 * every occurrence of it, until no adjacent pair appears in the table. Doing it in rank
 * order rather than left to right is the whole algorithm — a greedy left-to-right pass
 * produces different, wrong tokens.
 *
 * Pieces are short (the pre-tokenizer splits on word boundaries), so the straightforward
 * O(n²) scan is fine and a priority queue would be slower in practice. Results are memoized
 * because natural text repeats words heavily.
 */

export class BPE {
  /** "a b" -> rank */
  private readonly ranks: Map<string, number>;
  private readonly cache = new Map<string, string[]>();
  private readonly cacheLimit: number;

  constructor(merges: Iterable<string>, cacheLimit = 20_000) {
    this.ranks = new Map();
    let rank = 0;
    for (const merge of merges) {
      // Stored space-separated in tokenizer.json: "Ġ Ġ", "i n". Only the first space is a
      // separator — the parts themselves can be a byte-level space character but never an
      // ASCII space, so splitting on the first occurrence is unambiguous.
      const split = merge.indexOf(' ');
      if (split <= 0) continue;
      this.ranks.set(merge, rank++);
    }
    this.cacheLimit = cacheLimit;
  }

  get mergeCount(): number {
    return this.ranks.size;
  }

  /** Split one byte-level-encoded piece into merged symbols. */
  encode(piece: string): string[] {
    if (piece.length === 0) return [];
    const cached = this.cache.get(piece);
    if (cached) return cached;

    // Split into code points, not UTF-16 units: the byte-level alphabet is all BMP, but a
    // vocabulary entry could contain anything and splitting a surrogate pair would corrupt it.
    let symbols = [...piece];

    if (symbols.length > 1) {
      for (;;) {
        let bestRank = Infinity;
        let bestIndex = -1;
        for (let i = 0; i < symbols.length - 1; i++) {
          const rank = this.ranks.get(`${symbols[i]} ${symbols[i + 1]}`);
          if (rank !== undefined && rank < bestRank) {
            bestRank = rank;
            bestIndex = i;
          }
        }
        if (bestIndex < 0) break;

        // Merge every non-overlapping occurrence of the winning pair in one pass, which is
        // what the reference implementation does and is not the same as merging only at
        // bestIndex.
        const left = symbols[bestIndex];
        const right = symbols[bestIndex + 1];
        const merged: string[] = [];
        for (let i = 0; i < symbols.length; ) {
          if (i < symbols.length - 1 && symbols[i] === left && symbols[i + 1] === right) {
            merged.push(left + right);
            i += 2;
          } else {
            merged.push(symbols[i]);
            i++;
          }
        }
        symbols = merged;
        if (symbols.length === 1) break;
      }
    }

    if (this.cache.size >= this.cacheLimit) this.cache.clear();
    this.cache.set(piece, symbols);
    return symbols;
  }
}
