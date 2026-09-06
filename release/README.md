---
license: cc-by-4.0
language:
- id
library_name: pocket-tts
pipeline_tag: text-to-speech
tags:
- text-to-speech
- tts
- indonesian
- bahasa-indonesia
- voice-cloning
- pocket-tts
base_model: kyutai/pocket-tts
datasets:
- LEMAS-Project/LEMAS-Dataset-train
---

# Pocket TTS Indonesian

Indonesian text-to-speech with voice cloning, built on
[kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts) and trained on
502 hours of [LEMAS](https://huggingface.co/datasets/LEMAS-Project/LEMAS-Dataset-train)
Indonesian speech.

Two models in this repo. **Use the 6-layer one.**

| | 6-layer student | 24-layer teacher |
|---|---|---|
| config | `indonesian_6l.yaml` | `indonesian_24l.yaml` |
| size | **438 MB** | 1.27 GB |
| speed on CPU | **2.23x real time** | 0.69x |
| median WER | **12.50%** | 16.67% |
| speaker similarity | **0.938** | 0.927 |
| UTMOS | **2.68** | 2.36 |

The student was distilled from the teacher with guidance baked in
(`distill_cfg_coef: 2.0`), so it reaches guided quality in the single backbone
pass that `pocket-tts generate` actually runs. The teacher only matches it with
`--cfg 2.0`, which the shipped package cannot do — it has no `cfg_coef`
anywhere. The teacher is kept for reproducibility and as a distillation
starting point; there is no reason to run it. The CPU figures are one paired
run on the same machine and text: the student is usable without a GPU and the
teacher is not.

Repo layout: each model is a variant-named config beside its weights, and both
share one tokenizer.

```
indonesian_6l.yaml   ->  6l/model.safetensors
indonesian_24l.yaml  -> 24l/model.safetensors
tokenizer.model          (shared)
```

## Usage

```bash
uvx pocket-tts generate \
    --config hf://anak10thn/pocket-tts-indonesian/indonesian_6l.yaml@f44e4fc2b2fd79918667a1264e34505ea39f04fa \
    --voice your_voice.wav \
    --text "Selamat pagi, semoga hari Anda menyenangkan." \
    --eos-threshold -6.0 \
    --output-path out.wav
```

**Pass `--eos-threshold -6.0`.** The CLI defaults to -4.0, which measures 18.18%
median WER against -6.0's 12.50% on the same 153 items. Going further is worse
in a way the WER hides: -7.0 scores a lower median (10.00%) while producing no
audio at all for 18 of 153 inputs, against 2 at -6.0.

Omitting `--voice` is fine; it falls back to
[alba's audio](https://huggingface.co/kyutai/tts-voices/blob/main/alba-mackenna/casual.wav),
which this model clones like any other. Verified from a cleared cache against
both pocket-tts 3.1.0 from source and the current published wheel via `uvx`.

From Python:

```python
from pocket_tts import TTSModel

model = TTSModel.load_model(
    config="hf://anak10thn/pocket-tts-indonesian/indonesian_6l.yaml@f44e4fc2b2fd79918667a1264e34505ea39f04fa"
)
state = model.get_state_for_audio_prompt("your_voice.wav")
audio = model.generate_audio(state, "Selamat pagi, semoga hari Anda menyenangkan.")
```

Write text the way you normally would — capitalisation and punctuation are both
fine. The tokenizer case-folds internally, so `Halo, Dunia!` and `halo, dunia!`
produce identical tokens.

**Write numbers as words** (`lima belas`, not `15`). The training transcripts
spell them out, so digits are out-of-vocabulary.

## Evaluation

153 cross-sentence pairs over 116 speakers from the LEMAS Indonesian eval
split, none of whose recordings appear in training. Whisper-large-v3 pinned to
Indonesian, language-agnostic normalization, `--temp 0.3 --n-steps 1
--cfg 1.0` — no guidance, which is what the CLI does. Both models swept over
`--eos-threshold` on the full list and reported at their own best setting.

| | teacher 24L (eos -5.0) | **student 6L (eos -6.0)** |
|---|---|---|
| median per-item WER | 16.67% | **12.50%** |
| items over 50% WER | 10 / 153 | **7 / 153** |
| silent generations | 0 | 2 |
| speaker similarity | 0.927 | **0.938** |
| UTMOS | 2.36 | **2.68** |
| corpus WER | **20.55%** | 24.44% |

The student's full eos sweep, since the shape matters more than any single row:

| eos | median WER | over 50% | silent | no-EOS | sim | UTMOS |
|---|---|---|---|---|---|---|
| -2.0 | 30.00% | 30 | 0 | 43 | 0.916 | 2.26 |
| -3.0 | 22.22% | 26 | 0 | 14 | 0.927 | 2.43 |
| -4.0 (CLI default) | 18.18% | 16 | 0 | 0 | 0.936 | 2.61 |
| -5.0 | 15.38% | 16 | 0 | 0 | 0.936 | 2.64 |
| **-6.0** | **12.50%** | **7** | 2 | 0 | **0.938** | 2.68 |
| -7.0 | 10.00% | 22 | 18 | 0 | 0.932 | 2.73 |
| -8.0 | 45.45% | 71 | 63 | 0 | 0.932 | 2.79 |

**Read the medians, not the corpus WER.** Corpus WER sums insertions over the
list, so one generation that repeats itself past the end of the text outweighs
whatever is under test; on this eval set adjacent eos settings swing it by 14
points while the medians move smoothly. An earlier version of this card
headlined a corpus figure of 23.18% measured with guidance the CLI cannot
produce, and named an eos chosen from a sweep run on the first 60 items of the
list rather than a random sample. Both were wrong. The numbers above come from
full-list sweeps with the ASR's language pinned.

**Read WER against this corpus, not against the English models.** The same ASR
transcribing the eval set's *real* recordings scores 10.08%: the reference
transcripts are subtitle-derived and the audio is YouTube-sourced. The student's
12.50% median sits close to that floor. The released English model's 0.90% is a
different language on clean read speech.

**Speaker similarity is this model's strength.** 0.938 for the student against
0.922 for the released English model and 0.929 for kyutai's best 24-layer
English teacher trained on 31,700 hours. LEMAS is thousands of YouTube speakers
rather than a handful of audiobook narrators, so the model learned to imitate
arbitrary voices rather than a house style.

**Audio quality is capped by the corpus.** UTMOS 2.68 against 4.36 for the
English model, because LEMAS audio is 16 kHz while Mimi runs at 24 kHz: there is
no energy above 8 kHz to learn. Expect a decent phone call or YouTube video, not
a studio recording. Mixing in 48 kHz Indonesian speech would move this; more
training does not.

## Training

- **Teacher**: `kyutai/pocket-tts` English 24L, text embedding reinitialised for
  the new tokenizer, backbone transferred. Stopped at step 87,500 of a planned
  250,000 once it plateaued — validation loss was flat from step 42,500 on.
- **Student**: depth distillation to 6 layers with `distill_cfg_coef: 2.0`,
  50,000 steps, lr 4e-4 cosine. Validation loss was still falling at the end
  (0.0715 to 0.0126), so it is not saturated.
- **Data**: 502 h / 298,652 utterances, one shard of LEMAS Indonesian filtered
  to duration >= 4 s and mean alignment confidence >= 0.8, split by recording.
- **Transcripts**: LEMAS Indonesian text is ALL CAPS with no punctuation.
  Punctuation and truecasing were restored with
  [1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase](https://huggingface.co/1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase)
  before fitting the tokenizer, so the model learns comma pauses and
  sentence-final intonation and users' punctuation is in-vocabulary.
- **Tokenizer**: 4000-piece BPE, `nmt_nfkc_cf` normalization (case folding baked
  into the .model). Shared by both models.
- **Hardware**: one shared Tesla T4. fp16 autocast with a GradScaler — Turing
  has no bf16 tensor cores, and bf16 measures 11x slower there. Teacher ~8 days
  at 0.29 steps/s, student ~1.2 days at 0.46 steps/s.

Code — the data pipeline, the fp16 patch, and the eval fixes the numbers above
depend on:
[anak10thn/pocket-tts, branch `indonesian`](https://github.com/anak10thn/pocket-tts/tree/indonesian),
merged up to pocket-tts 3.1.0. It is kept on the fork rather than upstream by
[agreement with the maintainers](https://github.com/kyutai-labs/pocket-tts/pull/293).

## Limitations

- 16 kHz-sourced audio: band-limited output, UTMOS 2.68.
- Two of 153 eval items produce silence at the recommended setting. If a line
  comes back empty, retry or nudge `--eos-threshold` toward -5.0.
- Digits are out-of-vocabulary; spell numbers out.
- Trained on YouTube speech — mostly conversational and broadcast Indonesian.
  Formal narration is out of domain.
- Regional accents and languages other than Indonesian are not covered.
- The exclamation mark `!` is out-of-vocabulary (the restored transcripts use
  `.` `,` `?` almost exclusively).

## License and credits

CC-BY-4.0, following the LEMAS training data. Model architecture and base
weights from [Kyutai](https://huggingface.co/kyutai/pocket-tts) — see the
[CALM paper](https://arxiv.org/abs/2509.06926). Training data from the
[LEMAS Project](https://huggingface.co/datasets/LEMAS-Project/LEMAS-Dataset-train).
