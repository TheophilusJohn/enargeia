/**
 * Qwen2.5's tokenizer, reimplemented from its `tokenizer.json`.
 *
 * The pipeline, in the order the file specifies it:
 *
 *   1. added tokens  — matched literally, before anything else, and never merged
 *   2. NFC           — Unicode normalization
 *   3. Split         — a GPT-4-style regex isolating words, numbers, punctuation, whitespace
 *   4. ByteLevel     — UTF-8 bytes re-expressed in the 256-character printable alphabet
 *   5. BPE           — merges applied in learned-rank order
 *
 * Decoding runs it backwards, with one wrinkle: token boundaries fall between bytes, not
 * between characters, so bytes are concatenated across all tokens and decoded as UTF-8 once
 * at the end. Decoding tokens individually and joining the strings corrupts every
 * multi-byte character that straddles a boundary, which is most CJK and every emoji.
 */

import { BPE } from './bpe.ts';
import { bytesToUnicode, unicodeToBytes } from './bytelevel.ts';

/**
 * Qwen's split pattern, with one deliberate change.
 *
 * The file writes the contraction group as `(?i:'s|'t|...)`. Inline regex modifiers are
 * ES2025 and land in Safari later than the WebGPU support this project already requires, so
 * the group is spelled out as explicit case alternatives instead. The two are exactly
 * equivalent — every alternative is ASCII with a two-case mapping — and this version parses
 * everywhere.
 */
const SPLIT_PATTERN =
  "'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]" +
  '|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+' +
  '|\\p{N}' +
  '| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*' +
  '|\\s*[\\r\\n]+' +
  '|\\s+(?!\\S)' +
  '|\\s+';

export interface AddedToken {
  id: number;
  content: string;
  special: boolean;
}

export interface TokenizerJSON {
  model: { type?: string; vocab: Record<string, number>; merges: unknown[] };
  added_tokens?: Array<{ id: number; content: string; special?: boolean }>;
  normalizer?: { type?: string } | null;
}

export interface EncodeOptions {
  /**
   * Recognize added tokens in the input text. Default true, matching the reference's
   * `split_special_tokens=False`. Set false to treat "<|im_start|>" as literal characters —
   * which is what you want for untrusted input that must not be able to forge a role.
   */
  allowSpecial?: boolean;
}

export interface DecodeOptions {
  /** Omit tokens flagged `special` from the output. Default false. */
  skipSpecialTokens?: boolean;
}

export class Tokenizer {
  private readonly bpe: BPE;
  private readonly tokenToId = new Map<string, number>();
  private readonly idToToken: string[] = [];
  private readonly addedById = new Map<number, AddedToken>();
  private readonly addedByContent = new Map<string, AddedToken>();
  private readonly specialIds = new Set<number>();
  private readonly addedPattern: RegExp | null;
  private readonly normalize: boolean;
  private readonly splitRegex: RegExp;

  readonly vocabSize: number;

  constructor(json: TokenizerJSON) {
    if (json.model.type && json.model.type !== 'BPE') {
      throw new Error(`unsupported tokenizer model "${json.model.type}", expected BPE`);
    }

    for (const [token, id] of Object.entries(json.model.vocab)) {
      this.tokenToId.set(token, id);
      this.idToToken[id] = token;
    }

    // tokenizer.json v1 stores merges as "a b"; newer exports use ["a", "b"]. Accept both.
    const merges = json.model.merges.map((entry) =>
      Array.isArray(entry) ? `${String(entry[0])} ${String(entry[1])}` : String(entry),
    );
    this.bpe = new BPE(merges);

    for (const token of json.added_tokens ?? []) {
      const added: AddedToken = {
        id: token.id,
        content: token.content,
        special: token.special ?? false,
      };
      this.addedById.set(added.id, added);
      this.addedByContent.set(added.content, added);
      this.idToToken[added.id] = added.content;
      if (added.special) this.specialIds.add(added.id);
    }

    // Longest content first so "<|im_start|>" wins over any prefix of it that is also added.
    const contents = [...this.addedByContent.keys()].sort((a, b) => b.length - a.length);
    this.addedPattern =
      contents.length === 0
        ? null
        : new RegExp(contents.map(escapeRegExp).join('|'), 'g');

    this.normalize = (json.normalizer?.type ?? null) === 'NFC';
    this.splitRegex = new RegExp(SPLIT_PATTERN, 'gu');
    this.vocabSize = Math.max(this.idToToken.length, this.tokenToId.size + this.addedById.size);
  }

  static fromJSON(json: TokenizerJSON): Tokenizer {
    return new Tokenizer(json);
  }

  static async fromURL(url: string, init?: RequestInit): Promise<Tokenizer> {
    const response = await fetch(url, init);
    if (!response.ok) {
      throw new Error(`${url}: HTTP ${response.status} loading tokenizer`);
    }
    return new Tokenizer((await response.json()) as TokenizerJSON);
  }

  encode(text: string, options: EncodeOptions = {}): number[] {
    const allowSpecial = options.allowSpecial ?? true;
    const ids: number[] = [];

    if (!allowSpecial || !this.addedPattern) {
      this.encodeOrdinary(text, ids);
      return ids;
    }

    // Added tokens are matched on the raw text, before normalization, so an added token is
    // never altered by NFC and never merged with its neighbours.
    let cursor = 0;
    this.addedPattern.lastIndex = 0;
    for (let match = this.addedPattern.exec(text); match; match = this.addedPattern.exec(text)) {
      if (match.index > cursor) {
        this.encodeOrdinary(text.slice(cursor, match.index), ids);
      }
      ids.push(this.addedByContent.get(match[0])!.id);
      cursor = match.index + match[0].length;
    }
    if (cursor < text.length) {
      this.encodeOrdinary(text.slice(cursor), ids);
    }
    return ids;
  }

  private encodeOrdinary(text: string, out: number[]): void {
    if (text.length === 0) return;
    const normalized = this.normalize ? text.normalize('NFC') : text;

    this.splitRegex.lastIndex = 0;
    for (let match = this.splitRegex.exec(normalized); match; match = this.splitRegex.exec(normalized)) {
      const piece = match[0];
      if (piece.length === 0) {
        // A zero-width match would loop forever; step past it.
        this.splitRegex.lastIndex++;
        continue;
      }
      for (const symbol of this.bpe.encode(bytesToUnicode(piece))) {
        const id = this.tokenToId.get(symbol);
        if (id === undefined) {
          // Unreachable with a well-formed vocabulary: the byte-level alphabet guarantees
          // every single character is in the vocab, so BPE cannot produce an unknown symbol
          // unless the file is inconsistent. Fail loudly rather than emit a wrong token.
          throw new Error(`token ${JSON.stringify(symbol)} is not in the vocabulary`);
        }
        out.push(id);
      }
    }
  }

  decode(ids: Iterable<number>, options: DecodeOptions = {}): string {
    const skipSpecial = options.skipSpecialTokens ?? false;
    const parts: Uint8Array[] = [];
    let total = 0;

    for (const id of ids) {
      if (skipSpecial && this.specialIds.has(id)) continue;
      const added = this.addedById.get(id);
      let bytes: Uint8Array;
      if (added) {
        // Added tokens are literal text, not byte-level encoded.
        bytes = new TextEncoder().encode(added.content);
      } else {
        const token = this.idToToken[id];
        if (token === undefined) {
          throw new Error(`token id ${id} is outside the vocabulary of ${this.idToToken.length}`);
        }
        bytes = unicodeToBytes(token);
      }
      parts.push(bytes);
      total += bytes.length;
    }

    // Concatenate first, decode once. A multi-byte character split across two tokens only
    // survives if the bytes are joined before UTF-8 decoding.
    const joined = new Uint8Array(total);
    let offset = 0;
    for (const part of parts) {
      joined.set(part, offset);
      offset += part.length;
    }
    // ignoreBOM: a leading U+FEFF is content here, not an encoding marker to strip.
    return new TextDecoder('utf-8', { fatal: false, ignoreBOM: true }).decode(joined);
  }

  /** The token's byte-level string, or its literal content for added tokens. */
  tokenForId(id: number): string | undefined {
    return this.idToToken[id];
  }

  idForToken(token: string): number | undefined {
    return this.addedByContent.get(token)?.id ?? this.tokenToId.get(token);
  }

  isSpecial(id: number): boolean {
    return this.specialIds.has(id);
  }

  get addedTokens(): AddedToken[] {
    return [...this.addedById.values()].sort((a, b) => a.id - b.id);
  }
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
