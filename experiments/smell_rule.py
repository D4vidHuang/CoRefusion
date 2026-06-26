#!/usr/bin/env python3
"""
Objective "smell identifier" rule for the RQ3 distribution-based comparison.

Motivation (teacher's revision): instead of INJECTING a hand-picked smell
vocabulary (x / tmp / foo) and reading its rank, we scan the model's FULL output
distribution at each target position and report the rank of the smell names the
model itself surfaces, where "smell" is decided by a fixed, reproducible RULE
over decoded identifier tokens — no curated word list.

A decoded token surface `name` is a SMELL identifier iff it is a syntactically
valid identifier (and not a language keyword) and it is non-descriptive by one of
these objective tests:

  (a) length <= 2              single/double char        x, i, n, aa, ab
  (b) letters followed by digits  generic name + counter  tmp1, val2, x3, a12
  (c) length <= 3 and vowel-less  consonant abbreviation  tmp, ptr, str, idx, xyz
  (d) length <= 3 and not an English word  short non-word  foo, baz, qux

Real short words with a vowel (max, min, sum, key, row, day, get, set, ...) are
NOT flagged. The rule is deterministic given the English word set; the resolved
smell-token set is dumped to disk by the experiment so it is fully auditable.

This module is import-safe (no torch) and has a CLI self-test:  python smell_rule.py
"""
import os
import re

IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
NAME_DIGITS_RE = re.compile(r"[A-Za-z]+\d+\Z")
VOWEL_RE = re.compile(r"[aeiou]")

# Java + common reserved words to exclude (a keyword is not an "identifier name").
KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
    "class", "const", "continue", "default", "do", "double", "else", "enum",
    "extends", "final", "finally", "float", "for", "goto", "if", "implements",
    "import", "instanceof", "int", "interface", "long", "native", "new",
    "package", "private", "protected", "public", "return", "short", "static",
    "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "void", "volatile", "while", "true", "false", "null",
    "var",
}

# Fallback common short English/programming words to EXEMPT from the len<=3
# non-word test, used only when no system dictionary is found. Keeps legitimate
# short names (max, min, sum, ...) from being mislabelled as smells.
_FALLBACK_WORDS = {
    "the", "and", "for", "not", "but", "you", "all", "any", "can", "has", "had",
    "was", "are", "its", "out", "our", "one", "two", "six", "ten", "top", "end",
    "run", "key", "row", "sum", "max", "min", "avg", "age", "day", "way", "map",
    "let", "use", "now", "who", "why", "how", "yes", "job", "log", "url", "doc",
    "get", "set", "put", "add", "new", "old", "big", "low", "raw", "bit", "box",
    "id", "ok", "go", "up", "on", "in", "of", "to", "is", "it", "by", "as", "at",
    "or", "if", "do", "no", "so", "we", "he", "me", "my",
}


def load_english_words():
    """Resolve an English word set: prefer a system dictionary (reproducible on
    DAIC, which ships /usr/share/dict/words), else a compact fallback."""
    for p in ("/usr/share/dict/words", "/usr/share/dict/american-english",
              "/usr/share/dict/british-english"):
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8", errors="ignore") as f:
                    return {w.strip().lower() for w in f
                            if w.strip().isalpha() and len(w.strip()) <= 3}
            except Exception:
                pass
    return set(_FALLBACK_WORDS)


def is_smell_identifier(name, english_words):
    """True iff `name` (a decoded token surface, already stripped) is a
    non-descriptive identifier under the objective rule documented above."""
    if not name or not IDENT_RE.match(name):
        return False
    if name in KEYWORDS:
        return False
    low = name.lower()
    if len(name) <= 2:                       # (a) too short
        return True
    if NAME_DIGITS_RE.match(name):           # (b) name + counter
        return True
    if len(name) <= 3:
        if not VOWEL_RE.search(low):         # (c) vowel-less abbreviation
            return True
        if low not in english_words:         # (d) short non-word
            return True
    return False


def normalize_token_surface(tok_str):
    """Strip BPE / sentencepiece space markers to recover the identifier surface."""
    if tok_str is None:
        return ""
    return (tok_str.replace("Ġ", "")   # 'Ġ' GPT-2 space
                   .replace("▁", "")   # '▁' sentencepiece space
                   .replace("Ċ", "")   # 'Ċ' newline
                   .strip())


def build_smell_token_ids(tokenizer, english_words=None, dump_path=None):
    """Scan the tokenizer vocab once, return the SORTED list of token ids whose
    decoded surface is a smell identifier. Optionally dump the (id, surface)
    pairs to `dump_path` for auditing."""
    if english_words is None:
        english_words = load_english_words()
    vocab = tokenizer.get_vocab()            # surface(str) -> id
    pairs = []
    for tok_str, tid in vocab.items():
        surface = normalize_token_surface(tok_str)
        if is_smell_identifier(surface, english_words):
            pairs.append((int(tid), surface))
    pairs.sort()
    if dump_path:
        with open(dump_path, "w", encoding="utf-8") as f:
            f.write("token_id\tsurface\n")
            for tid, s in pairs:
                f.write(f"{tid}\t{s}\n")
    return [tid for tid, _ in pairs]


# ── CLI self-test (no torch needed) ─────────────────────────────────────────
if __name__ == "__main__":
    ew = load_english_words()
    print(f"english word set size (<=3 chars): {len(ew)}  "
          f"(source: {'system dict' if len(ew) > 200 else 'fallback'})")
    should_smell = ["x", "i", "n", "a", "aa", "ab", "ii",
                    "tmp", "ptr", "str", "idx", "lst", "xyz", "qz",
                    "tmp1", "val2", "x3", "a12", "foo", "baz", "qux"]
    should_not  = ["max", "min", "sum", "key", "row", "col", "day", "get", "set",
                   "add", "map", "count", "result", "value", "index", "buffer",
                   "decodedCapacity", "userName", "if", "for", "int", "new"]
    print("\n-- expected SMELL --")
    for n in should_smell:
        print(f"  {n:>16}  -> {is_smell_identifier(n, ew)}")
    print("\n-- expected NOT smell --")
    for n in should_not:
        print(f"  {n:>16}  -> {is_smell_identifier(n, ew)}")
