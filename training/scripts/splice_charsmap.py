"""Teach the tokenizer that a hyphen is a word break, without touching its vocabulary.

Indonesian marks reduplication with a hyphen -- orang-orang, rata-rata,
tiba-tiba, sewaktu-waktu -- and it is everywhere in written Indonesian. The
LEMAS transcripts this tokenizer was fitted on write those as separate words,
so "-" never entered the vocabulary and lands on <unk>. The model then emits
something arbitrary in the middle of the word: "rata-rata" came out as
"rata indah rata".

A sentencepiece model carries its normalizer as a precompiled charsmap that is
independent of the vocabulary -- verified: a throwaway tokenizer trained on 769
lines produces a charsmap byte-identical to this one's. So the fix is to compile
a charsmap from nmt_nfkc_cf plus one rule (002D -> 0020) and splice it in,
leaving all 4000 pieces exactly as they are. The model's text embedding stays
valid, and `pocket-tts generate` needs no wrapper: hyphenated input now
tokenizes the way the corpus wrote it.
"""

import sys

from sentencepiece import sentencepiece_model_pb2 as pb2

src, donor, dst = sys.argv[1], sys.argv[2], sys.argv[3]

model = pb2.ModelProto()  # ty: ignore[unresolved-attribute]  -- needs protobuf
model.ParseFromString(open(src, "rb").read())
donor_model = pb2.ModelProto()  # ty: ignore[unresolved-attribute]  -- needs protobuf
donor_model.ParseFromString(open(donor, "rb").read())

before = [p.piece for p in model.pieces]
model.normalizer_spec.precompiled_charsmap = donor_model.normalizer_spec.precompiled_charsmap
model.normalizer_spec.name = "nmt_nfkc_cf_id"
after = [p.piece for p in model.pieces]

assert before == after, "vocabulary must not change"
open(dst, "wb").write(model.SerializeToString())
print(f"wrote {dst}: {len(after)} pieces kept, normalizer replaced")
