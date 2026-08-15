#!/usr/bin/env python3
"""Generate encode/decode fixtures from the reference tokenizer.

The TypeScript tokenizer is a reimplementation, and a reimplementation that is 99% right
produces fluent text and subtly wrong tokens. The only way to know it agrees is to compare
against the reference on inputs chosen to hit the places implementations diverge:

  emoji            multi-byte sequences, ZWJ joins, skin-tone modifiers, flags
  CJK              no whitespace to split on, so pre-tokenization does all the work
  lone surrogates  unpaired UTF-16 halves, which are not valid Unicode scalars
  whitespace runs  the split regex has four separate whitespace alternatives
  special tokens   matched before normalization and never merged
  NFC              composed and decomposed forms that must normalize to the same ids

Usage:
    tools/.venv/bin/python tools/dump_tokenizer_cases.py \
        --tokenizer test/fixtures/tokenizer.json \
        --out test/fixtures/tokenizer_cases.json

Requires `tokenizers` (the Rust binding, not `transformers`):
    python3 -m venv tools/.venv && tools/.venv/bin/pip install tokenizers
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from pathlib import Path

try:
    from tokenizers import Tokenizer
except ImportError:  # pragma: no cover
    sys.exit("missing dependency: python3 -m venv tools/.venv && tools/.venv/bin/pip install tokenizers")


EMOJI = [
    "😀", "🎉", "🚀", "❤️", "👍🏽", "👨‍👩‍👧‍👦", "🏳️‍🌈", "🇯🇵", "🇬🇧",
    "🤦🏼‍♂️", "😶‍🌫️", "1️⃣", "©️", "™", "🫠", "🧑‍💻",
]

CJK = [
    "こんにちは", "世界", "你好世界", "漢字とひらがなとカタカナ", "한국어 테스트",
    "中文分词很难", "日本語のテキストです", "簡體字與繁體字", "ありがとうございます",
]

SCRIPTS = [
    "Здравствуй мир", "مرحبا بالعالم", "שלום עולם", "नमस्ते दुनिया", "สวัสดีชาวโลก",
    "Γειά σου Κόσμε", "ᐊᐃᓐᖓᐃ", "ⵜⴰⵎⴰⵣⵉⵖⵜ", "🄰🄱🄲",
]

WHITESPACE = [
    "a  b", "a   b", "a\tb", "a\nb", "a\r\nb", "a\n\n\nb", "   leading", "trailing   ",
    "\n", "\n\n", "\t\t\t", " ", "  ", " nbsp", "line1\nline2\n", "  \n  \n  ",
    "word \n word", " line sep", "　ideographic space",
]

NUMBERS = [
    "0", "42", "1234567890", "3.14159", "-17", "1e10", "0x1F", "١٢٣", "一二三", "½ ¼ ¾",
    "2024-01-15", "1,000,000", "v1.2.3",
]

CONTRACTIONS = [
    "don't", "DON'T", "Don'T", "it's", "IT'S", "we're", "WE'RE", "I've", "I'M", "you'll",
    "they'd", "o'clock", "'s", "'S", "rock'n'roll",
]

NFC_PAIRS = [
    ("café", "café"),
    ("Ångström", "Ångström"),
    ("한", "한"),
    ("ﬁ", "fi"),
]

CODE = [
    "def f(x):\n    return x + 1\n",
    "const x = {a: 1, b: [2, 3]};",
    "SELECT * FROM t WHERE id = 1;",
    "#include <stdio.h>\nint main(){return 0;}",
    "<div class=\"a\">text</div>",
    "a && b || !c",
    "/* comment */ // another",
    "\\\\n\\\\t escaped",
]

EDGE = [
    "",
    " ",
    "a",
    "\x00",              # NUL
    "\x00\x01\x02",       # low control bytes
    "\ufeff",            # BOM / zero-width no-break space
    "\ufffd",            # replacement character
    "\u200b",            # zero-width space
    "\u200d",            # zero-width joiner
    "\u00ad",            # soft hyphen
    "\U0010ffff",        # highest code point
    "\u0301",            # combining acute, unattached
    "e" + "\u0301" * 8,  # one letter, eight combining marks
    "a" * 200,
    "ab" * 150,
    "\U0001f389" * 40,
    "  " * 60,
    "\r",
    "\r\n",
    "\n\r",
    "\x7f",              # DEL
]


def special_token_cases(tok: Tokenizer) -> list[str]:
    """Added tokens alone, in context, adjacent, and adversarially malformed."""
    specials = [t.content for t in tok.get_added_tokens_decoder().values()]
    cases: list[str] = []
    for content in specials:
        cases.append(content)
        cases.append(f"before {content} after")
        cases.append(f"{content}{content}")
        cases.append(f"x{content}y")
    if len(specials) >= 2:
        cases.append(f"{specials[0]}{specials[1]}")
        cases.append(f"{specials[1]} middle {specials[0]}")
    # Near-misses that must NOT be treated as special.
    cases += [
        "<|im_start|", "|im_start|>", "<|im_start |>", "< |im_start|>",
        "<|IM_START|>", "<|im_start|>>", "<<|im_start|>",
    ]
    # A realistic chat template, which is the shape the runtime will actually see.
    cases.append(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
    )
    return cases


def lone_surrogate_cases() -> list[dict]:
    """Unpaired surrogates, which cannot round-trip through Python's str.

    These exist because JavaScript strings are UTF-16 and can hold an unpaired surrogate,
    while Python's str cannot represent one that survives UTF-8 encoding. So they are
    emitted as expected *outputs* only: the JS side builds the string from code units and
    asserts the tokenizer does not throw and round-trips to the replacement character, which
    is what the reference does when it sees the equivalent bytes.
    """
    return [
        {"units": [0xD800], "note": "lone high surrogate"},
        {"units": [0xDC00], "note": "lone low surrogate"},
        {"units": [0x41, 0xD800, 0x42], "note": "high surrogate between letters"},
        {"units": [0xDC00, 0xD800], "note": "reversed pair"},
        {"units": [0xD83D, 0xDE00], "note": "valid pair (grinning face)"},
        {"units": [0xD83D, 0x41], "note": "high surrogate then ASCII"},
    ]


def random_cases(rng: random.Random, count: int) -> list[str]:
    """Random mixtures, which find the interactions handwritten cases miss."""
    alphabet = (
        list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        + list(" \t\n.,!?;:'\"()[]{}#@$%^&*-_=+/\\|<>~`")
        + EMOJI
        + list("あいうえお漢字한글")
        + list("абвгдеёжз")
    )
    out = []
    for _ in range(count):
        length = rng.randint(1, 60)
        out.append("".join(rng.choice(alphabet) for _ in range(length)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", type=Path, default=Path("test/fixtures/tokenizer.json"))
    parser.add_argument("--out", type=Path, default=Path("test/fixtures/tokenizer_cases.json"))
    parser.add_argument("--random", type=int, default=2000, help="random cases to append")
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    tok = Tokenizer.from_file(str(args.tokenizer))
    rng = random.Random(args.seed)

    groups: dict[str, list[str]] = {
        "ascii": ["hello world", "The quick brown fox.", "Hello, World!", "a b c"],
        "emoji": EMOJI + [f"text {e} more" for e in EMOJI[:6]],
        "cjk": CJK,
        "scripts": SCRIPTS,
        "whitespace": WHITESPACE,
        "numbers": NUMBERS,
        "contractions": CONTRACTIONS,
        "code": CODE,
        "edge": EDGE,
        "special": special_token_cases(tok),
        "nfc": [s for pair in NFC_PAIRS for s in pair],
        "random": random_cases(rng, args.random),
    }

    cases = []
    for group, texts in groups.items():
        for text in texts:
            ids = tok.encode(text, add_special_tokens=False).ids
            cases.append(
                {
                    "group": group,
                    "text": text,
                    "ids": ids,
                    # What the reference decodes back to. Not always equal to `text`:
                    # normalization is lossy for decomposed forms.
                    "decoded": tok.decode(ids, skip_special_tokens=False),
                }
            )

    # NFC equivalence assertions, checked separately from round-tripping.
    nfc = [
        {
            "composed": a,
            "decomposed": b,
            "ids": tok.encode(a, add_special_tokens=False).ids,
            "equal": tok.encode(a, add_special_tokens=False).ids
            == tok.encode(b, add_special_tokens=False).ids,
        }
        for a, b in NFC_PAIRS
    ]

    payload = {
        "generator": "tools/dump_tokenizer_cases.py",
        "tokenizer": str(args.tokenizer),
        "seed": args.seed,
        "unicode_version": unicodedata.unidata_version,
        "vocab_size": tok.get_vocab_size(with_added_tokens=True),
        "added_tokens": [
            {"id": i, "content": t.content, "special": t.special}
            for i, t in sorted(tok.get_added_tokens_decoder().items())
        ],
        "cases": cases,
        "nfc": nfc,
        "lone_surrogates": lone_surrogate_cases(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    by_group: dict[str, int] = {}
    for case in cases:
        by_group[case["group"]] = by_group.get(case["group"], 0) + 1
    print(f"wrote {len(cases)} cases to {args.out}")
    for group, count in sorted(by_group.items()):
        print(f"  {group:14s} {count:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
