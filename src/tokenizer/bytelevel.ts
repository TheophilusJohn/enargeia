/**
 * The GPT-2 byte-level alphabet.
 *
 * BPE operates on strings, but the input is bytes, and most byte values are not printable
 * characters. The trick this format uses is a bijection from all 256 byte values onto 256
 * printable code points: the 188 bytes that are already printable ASCII or Latin-1 map to
 * themselves, and the remaining 68 are shifted into the U+0100 range.
 *
 * That is why a space appears as "Ġ" and a newline as "Ċ" in the vocabulary. It is not an
 * encoding of the text — it is an encoding of the *bytes*, which is what makes the
 * tokenizer total: every byte sequence has a representation, so no input is unencodable and
 * there is no need for an unknown token. Qwen's tokenizer.json accordingly has
 * `byte_fallback: false` and no `unk_token`.
 */

/** byte value -> code point */
export const BYTE_TO_UNICODE: string[] = [];
/** code point -> byte value */
export const UNICODE_TO_BYTE = new Map<string, number>();

{
  const printable: number[] = [];
  for (let b = 0x21; b <= 0x7e; b++) printable.push(b); // '!'..'~'
  for (let b = 0xa1; b <= 0xac; b++) printable.push(b); // '¡'..'¬'
  for (let b = 0xae; b <= 0xff; b++) printable.push(b); // '®'..'ÿ'

  const direct = new Set(printable);
  let shifted = 0;
  for (let b = 0; b < 256; b++) {
    const codePoint = direct.has(b) ? b : 256 + shifted++;
    const character = String.fromCodePoint(codePoint);
    BYTE_TO_UNICODE[b] = character;
    UNICODE_TO_BYTE.set(character, b);
  }
}

const encoder = new TextEncoder();
// ignoreBOM is essential, not cosmetic: without it TextDecoder silently deletes a
// leading U+FEFF, so the token for a byte-order mark decodes to the empty string and a
// document that starts with one loses its first character.
const decoder = new TextDecoder('utf-8', { fatal: false, ignoreBOM: true });

/** UTF-8 bytes of `text`, re-expressed in the byte-level alphabet. */
export function bytesToUnicode(text: string): string {
  const bytes = encoder.encode(text);
  let out = '';
  for (let i = 0; i < bytes.length; i++) {
    out += BYTE_TO_UNICODE[bytes[i]];
  }
  return out;
}

/**
 * Inverse of {@link bytesToUnicode}, decoding the recovered bytes as UTF-8.
 *
 * Non-fatal on purpose: a token boundary can fall in the middle of a multi-byte character,
 * so decoding a partial stream has to yield a replacement character rather than throwing.
 * Callers that decode incrementally should use {@link unicodeToBytes} and buffer instead.
 */
export function unicodeToText(encoded: string): string {
  return decoder.decode(unicodeToBytes(encoded));
}

/** Recover the raw bytes without interpreting them as UTF-8. */
export function unicodeToBytes(encoded: string): Uint8Array {
  const out = new Uint8Array([...encoded].length);
  let n = 0;
  for (const character of encoded) {
    const byte = UNICODE_TO_BYTE.get(character);
    // A character outside the alphabet cannot come from our own encoder. It can come from a
    // malformed vocabulary entry, where dropping it silently would corrupt everything after
    // it, so it is preserved as its low byte rather than discarded.
    out[n++] = byte ?? character.codePointAt(0)! & 0xff;
  }
  return out.subarray(0, n);
}
