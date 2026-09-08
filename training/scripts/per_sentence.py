"""Generate a speech one sentence at a time, then join with real pauses.

The model was trained on utterances averaging 6 s, cut out of continuous
YouTube speech at word boundaries. So it never saw deliberate inter-sentence
silence, and feeding it a paragraph gives 2 pauses of 0.24 s where the text has
6 punctuation marks -- which is why long text sounds read rather than delivered.

Sending each sentence as its own utterance puts the model back in the
distribution it was trained on, so each one gets sentence-final prosody, and the
gap between them becomes ours to choose rather than the model's to omit.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import sphn


def sentences(text: str) -> list[str]:
    """Split on sentence-final punctuation, keeping the mark with its sentence."""
    parts = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    return [p.strip() for p in parts if p.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("voice")
    ap.add_argument("text_file")
    ap.add_argument("out")
    ap.add_argument("--config", default="hf://anak10thn/pocket-tts-indonesian/indonesian_6l.yaml")
    ap.add_argument("--eos-threshold", default="-5.0")
    ap.add_argument("--gap", type=float, default=0.55, help="silence between sentences, seconds")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tmp", default="/tmp/persent")
    args = ap.parse_args()

    tmp = Path(args.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    sents = sentences(Path(args.text_file).read_text())
    print(f"{len(sents)} sentences")

    pieces, sr = [], None
    for i, sent in enumerate(sents):
        out = tmp / f"s{i:02d}.wav"
        subprocess.run(
            [sys.executable, "-m", "pocket_tts.main", "generate",
             "--config", args.config, "--voice", args.voice, "--text", sent,
             "--eos-threshold", args.eos_threshold, "--output-path", str(out),
             "--device", args.device],
            check=True, capture_output=True,
        )
        wav, sr = sphn.read(str(out))
        mono = wav.mean(axis=0)
        pieces.append(mono)
        print(f"  [{i}] {len(mono) / sr:5.1f}s  {sent[:58]}")

    assert sr is not None
    gap = np.zeros(int(args.gap * sr), dtype=pieces[0].dtype)
    joined = pieces[0]
    for piece in pieces[1:]:
        joined = np.concatenate([joined, gap, piece])
    sphn.write_wav(args.out, joined, sr)
    print(f"\n{args.out}: {len(joined) / sr:.1f}s ({len(sents) - 1} gaps of {args.gap}s)")


if __name__ == "__main__":
    main()
