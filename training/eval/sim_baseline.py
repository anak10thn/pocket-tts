"""What does a speaker-similarity score of 0.93 actually mean on this corpus?

Absolute x-vector cosines are not interpretable without the two anchors that
bracket them: how similar two real recordings of the SAME speaker are, and how
similar two real recordings of DIFFERENT speakers are. A generation scoring
0.93 is excellent if different speakers sit at 0.5 and meaningless if they sit
at 0.90.
"""

import json
import random

from typing import Any

import numpy as np
import numpy.typing as npt
import sphn
import torch
from transformers import AutoFeatureExtractor, WavLMForXVector

MANIFEST = "data/ideval_valid.jsonl"  # any manifest carrying a speaker field


def load16k(path: str) -> npt.NDArray[Any]:
    wav, sr = sphn.read(path)
    mono = wav.mean(axis=0)
    if sr != 16000:
        mono = sphn.resample(mono, src_sample_rate=sr, dst_sample_rate=16000)
    return mono


rows = [json.loads(line) for line in open(MANIFEST)]
by_speaker: dict[str, list[str]] = {}
for r in rows:
    by_speaker.setdefault(r["speaker"], []).append(r["path"])

same_pairs = [(v[0], v[1]) for v in by_speaker.values() if len(v) >= 2]
speakers = [v[0] for v in by_speaker.values()]
random.seed(0)
random.shuffle(speakers)
diff_pairs = list(zip(speakers[::2], speakers[1::2], strict=False))
same_pairs, diff_pairs = same_pairs[:40], diff_pairs[:40]
print(f"{len(same_pairs)} same-speaker pairs, {len(diff_pairs)} different-speaker pairs")

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
net = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()  # ty: ignore[invalid-argument-type]


def sims(pairs: list[tuple[str, str]]) -> npt.NDArray[Any]:
    out = []
    for i in range(0, len(pairs), 8):
        chunk = pairs[i : i + 8]
        wavs = [load16k(p) for pair in chunk for p in pair]
        inp = fe(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            e = net(**{k: v.to(dev) for k, v in inp.items()}).embeddings
        e = torch.nn.functional.normalize(e, dim=-1)
        out += (e[0::2] * e[1::2]).sum(-1).cpu().tolist()
    return np.array(out)


same, diff = sims(same_pairs), sims(diff_pairs)
print(f"\nsame speaker, real audio:      {same.mean():.3f}  (sd {same.std():.3f})")
print(f"different speakers, real audio: {diff.mean():.3f}  (sd {diff.std():.3f})")
print(f"separation between the anchors: {same.mean() - diff.mean():.3f}")
print("\nour generations scored 0.932 against their own prompt, 0.907 against others")
span = same.mean() - diff.mean()
if span > 0:
    print(f"which places cloning at {(0.932 - diff.mean()) / span:.0%} of the way from")
    print("a different speaker to the same speaker")
