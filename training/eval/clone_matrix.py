"""Does the voice prompt actually steer the output, or is every generation the same voice?

A high average speaker similarity does not answer that: if --voice were ignored,
every generation would still resemble whatever single voice the model defaults
to, and each would be compared against a different reference, giving middling
numbers rather than obviously broken ones.

The decisive test is a cross matrix. Generate one text with N different prompts,
then embed every prompt and every generation. If cloning works, generation i
sits closest to prompt i and the diagonal dominates. If it does not, the
generations cluster with each other and away from the prompts.
"""

import subprocess
import sys
from pathlib import Path

from typing import Any

import numpy as np
import numpy.typing as npt
import sphn
import torch
from transformers import AutoFeatureExtractor, WavLMForXVector

OUT = Path("/tmp/clonetest")
TEXT = "Kita harus memastikan setiap kebijakan membawa perbaikan nyata bagi masyarakat."
CONFIG = "hf://anak10thn/pocket-tts-indonesian/indonesian_6l.yaml"


def load16k(path: str) -> npt.NDArray[Any]:
    wav, sr = sphn.read(path)
    mono = wav.mean(axis=0)
    if sr != 16000:
        mono = sphn.resample(mono, src_sample_rate=sr, dst_sample_rate=16000)
    return mono


def main() -> None:
    prompts = sys.argv[1:]
    OUT.mkdir(exist_ok=True)
    gens = []
    for i, voice in enumerate(prompts):
        out = OUT / f"gen{i}.wav"
        if not out.exists():
            subprocess.run(
                [sys.executable, "-m", "pocket_tts.main", "generate",
                 "--config", CONFIG, "--voice", voice, "--text", TEXT,
                 "--eos-threshold", "-6.0", "--output-path", str(out),
                 "--device", "cuda"],
                check=True, capture_output=True,
            )
        gens.append(str(out))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    net = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()  # ty: ignore[invalid-argument-type]

    def embed(paths: list[str]) -> torch.Tensor:
        wavs = [load16k(p) for p in paths]
        inp = fe(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            return net(**{k: v.to(dev) for k, v in inp.items()}).embeddings

    e_prompt = torch.nn.functional.normalize(embed(prompts), dim=-1)
    e_gen = torch.nn.functional.normalize(embed(gens), dim=-1)
    sim = (e_gen @ e_prompt.T).cpu().numpy()

    n = len(prompts)
    print("\nrows = generation from prompt i, columns = prompt j\n")
    print("        " + "".join(f"prompt{j:<4}" for j in range(n)))
    for i in range(n):
        row = "".join(f"{sim[i][j]:>9.3f}" for j in range(n))
        print(f"gen{i:<4}{row}   <- own prompt: {sim[i][i]:.3f}")

    diag = np.diag(sim)
    off = sim[~np.eye(n, dtype=bool)]
    print(f"\nown-prompt mean {diag.mean():.3f} | other-prompt mean {off.mean():.3f}")
    hits = sum(int(np.argmax(sim[i]) == i) for i in range(n))
    print(f"generations closest to their own prompt: {hits}/{n}")

    # Are the generations distinguishable from each other at all?
    gg = (e_gen @ e_gen.T).cpu().numpy()
    print(f"generation-to-generation mean (off-diagonal): {gg[~np.eye(n, dtype=bool)].mean():.3f}")
    pp = (e_prompt @ e_prompt.T).cpu().numpy()
    print(f"prompt-to-prompt mean (off-diagonal):         {pp[~np.eye(n, dtype=bool)].mean():.3f}")


if __name__ == "__main__":
    main()
