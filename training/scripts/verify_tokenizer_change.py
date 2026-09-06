"""Prove the spliced normalizer changed hyphens and nothing else.

Swapping a charsmap could silently shift how any character normalizes, which
would mistokenize the whole corpus against a model whose embedding was trained
on the old mapping. So: tokenize every training transcript with both, and
require that the only lines whose ids differ are lines containing a hyphen.
"""

import json
import sys

import sentencepiece as spm

old = spm.SentencePieceProcessor(model_file=sys.argv[1])
new = spm.SentencePieceProcessor(model_file=sys.argv[2])
manifest = sys.argv[3]

n = differ = differ_with_hyphen = 0
unk_old = unk_new = 0
examples: list[tuple[str, list[str], list[str]]] = []

with open(manifest) as f:
    for line in f:
        text = json.loads(line)["transcript"]
        a, b = old.encode(text), new.encode(text)
        n += 1
        unk_old += a.count(0)
        unk_new += b.count(0)
        if a != b:
            differ += 1
            if "-" in text:
                differ_with_hyphen += 1
            elif len(examples) < 5:
                examples.append(
                    (text, old.encode(text, out_type=str), new.encode(text, out_type=str))
                )

print(f"transcripts            : {n}")
print(f"tokenization differs   : {differ}")
print(f"  of those, had hyphen : {differ_with_hyphen}")
print(f"  differ WITHOUT hyphen: {differ - differ_with_hyphen}   <- must be 0")
print(f"<unk> tokens, old      : {unk_old}")
print(f"<unk> tokens, new      : {unk_new}")
for text, a, b in examples:
    print(f"\nUNEXPECTED: {text[:80]}\n  old {a[:12]}\n  new {b[:12]}")

assert differ == differ_with_hyphen, "the normalizer changed something other than hyphens"
print("\nOK: only hyphenated lines changed")
