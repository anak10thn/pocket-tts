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
| median WER | **9.09%** | 16.67% |
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
    --config hf://anak10thn/pocket-tts-indonesian/indonesian_6l.yaml@17257664e384561c957b02ac92edd1a24807f0e5 \
    --voice your_voice.wav \
    --text "Selamat pagi, semoga hari Anda menyenangkan." \
    --eos-threshold -6.0 \
    --output-path out.wav
```

**Pass `--eos-threshold -6.0`.** It is the best value on the eval set (9.09%
median against the CLI default -4.0's 13.33%) and on long text (6.74% on an
87-word news passage). Earlier releases of this model needed two different
values for those two cases -- -6.0 was best on short input but dropped whole
chunks of long text -- and that split is gone since the retrain described
below.

Omitting `--voice` is fine; it falls back to
[alba's audio](https://huggingface.co/kyutai/tts-voices/blob/main/alba-mackenna/casual.wav),
which this model clones like any other. Verified from a cleared cache against
both pocket-tts 3.1.0 from source and the current published wheel via `uvx`.

From Python:

```python
from pocket_tts import TTSModel

model = TTSModel.load_model(
    config="hf://anak10thn/pocket-tts-indonesian/indonesian_6l.yaml@17257664e384561c957b02ac92edd1a24807f0e5"
)
state = model.get_state_for_audio_prompt("your_voice.wav")
audio = model.generate_audio(state, "Selamat pagi, semoga hari Anda menyenangkan.")
```

Write text the way you normally would — capitalisation and punctuation are both
fine. The tokenizer case-folds internally, so `Halo, Dunia!` and `halo, dunia!`
produce identical tokens.

**Write numbers as words** (`lima belas`, not `15`). The training transcripts
spell them out, so digits are out-of-vocabulary.

Hyphens are fine: `anak-anak`, `tiba-tiba`, `rata-rata` all work. They did not
until 2026-09-06 — the LEMAS transcripts write reduplication as separate words,
so `-` never entered the vocabulary and came out as an invented word in the
middle of the phrase. The tokenizer's normalizer now maps a hyphen to a space,
which is exactly how the corpus wrote it.

## Evaluation

153 cross-sentence pairs over 116 speakers from the LEMAS Indonesian eval
split, none of whose recordings appear in training. Whisper-large-v3 pinned to
Indonesian, language-agnostic normalization, `--temp 0.3 --n-steps 1
--cfg 1.0` — no guidance, which is what the CLI does. Both models swept over
`--eos-threshold` on the full list and reported at their own best setting.

| | teacher 24L (eos -5.0) | **student 6L (eos -6.0)** |
|---|---|---|
| median per-item WER | 16.67% | **9.09%** |
| items over 50% WER | 10 / 153 | **4 / 153** |
| silent generations | 0 | **0** |
| speaker similarity | 0.927 | **0.939** |
| UTMOS | 2.36 | **2.68** |
| corpus WER | 20.55% | **13.48%** |

The student's full eos sweep, since the shape matters more than any single row:

| eos | median WER | corpus WER | over 50% | silent | no-EOS | sim | UTMOS |
|---|---|---|---|---|---|---|---|
| -1.0 | 18.18% | 74.52% | 26 | 0 | 19 | 0.919 | 2.48 |
| -2.0 | 13.33% | 63.95% | 12 | 0 | 9 | 0.925 | 2.53 |
| -3.0 | 16.67% | 48.49% | 18 | 0 | 0 | 0.932 | 2.64 |
| -4.0 (CLI default) | 13.33% | 19.40% | 8 | 0 | 0 | 0.939 | 2.64 |
| -5.0 | 12.50% | 28.16% | 6 | 0 | 0 | 0.937 | 2.68 |
| **-6.0** | **9.09%** | **13.48%** | **4** | **0** | 0 | **0.939** | **2.68** |

### What the retrain changed

The published student was retrained on 2026-09-10 after finding that the
training targets were being cut off mid-word. The loader trims each target to
the last aligned word plus a 0.2s tail, which is right for a corpus padded with
silence -- but LEMAS clips are cropped tight around speech (the gap between
where the signal goes quiet and the clip end has a median of -0.05s) while its
aligner ends early, with speech continuing past the last aligned word by 0.15s
at the median and 0.52s at p90. So 0.2s covered 62% of clips and truncated the
rest. Raising it to 0.6s covers 99%.

| | before | after |
|---|---|---|
| median WER | 12.50% | **9.09%** |
| corpus WER | 24.44% | **13.48%** |
| items over 50% | 7 | **4** |
| silent generations | 2 | **0** |
| 87-word passage at eos -6.0 | 21.35% | **6.74%** |
| speaker similarity / UTMOS | 0.938 / 2.68 | 0.939 / 2.68 |

One prediction in this it got wrong, recorded because it was wrong: the guess
was that fixing truncated targets would move the eos optimum back toward the
upstream reference of -1.0. It did not. -6.0 is still best, and -1.0 is bad for
the opposite reason -- the model runs past the text on 19 of 153 items. The fix
worked through accuracy and stability, not through EOS calibration, and why this
model wants -6.0 is still unexplained.

### Long text

Two paired examples, not a benchmark. First, a 73-word formal speech passage,
same voice prompt and text for both models, each at its own best
`--eos-threshold`.

| | student 6L | teacher 24L |
|---|---|---|
| WER | **4.11%** | 26.03% |
| deleted words | 3 | 17 |
| audio length | 26.8 s | 33.3 s |
| generations hitting the length cap without EOS | 0 | 1 of 3 chunks |

The student's 4.11% is far below its 12.50% median on the eval set, which is
the eval set's fault rather than a surprise: those reference transcripts are
subtitle-derived, and the same ASR scores 10.08% on their real audio. On clean,
properly punctuated text the model does considerably better than the evaluation
suggests.

The teacher failed on a repeated structure, dropped a clause and then produced
non-words:

```
text:       ...bukan hanya soal angka, tetapi soal bagaimana manfaatnya
            dirasakan sampai ke desa, sampai ke pasar, sampai ke meja makan...
student 6L: ...bukan hanya soal angka, tetapi soal bagaimana manfaatnya
            dirasakan sampai ke pasar, sampai ke meja makan...
teacher 24L: ...bukan hanya soal angka. Sampai kebesar, sampai kebasar, sampai ke...
```

Both models can drop one item from a long enumeration — the student lost
"sampai ke desa" here. Split lists into separate sentences if you need every
item.

Second, an 87-word news passage, student only, showing what the two fixes above
are worth:

| setting | WER |
|---|---|
| eos -6.0, hyphens unhandled (as this card first shipped) | 25.84% |
| eos -6.0, hyphens replaced by hand | 21.35% |
| eos -5.0, hyphens typed normally, previous weights | 6.74% |
| **eos -6.0, current weights** | **6.74%** |

6.74% is close to this eval corpus's ASR floor of 10.08%, on text far cleaner
than the corpus — which is the honest reading of what this model does on
well-written input.

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

### Voice cloning: works, but it flattens voices

The 0.938 figure needs its anchors, and an earlier version of this card quoted
it against the English models' 0.922 as if that were a fair comparison. It is
not: x-vector cosines are not comparable across corpora, and on 16 kHz YouTube
audio the whole scale is compressed. Measured on this eval set:

| | cosine |
|---|---|
| two real recordings of the **same** speaker | 0.954 |
| a generation against **its own** prompt | 0.932 |
| two generations from **different** prompts | 0.939 |
| two real recordings of **different** speakers | 0.888 / 0.752 |

(0.888 is the four prompts used in the cloning test; 0.752 is 40 random
different-speaker pairs across the set.)

Read down that table. A generation lands 89% of the way from a stranger to the
target, and in a four-prompt cross test each generation was nearest its own
prompt, 4 out of 4 — cloning genuinely works. But two generations made from
*different* prompts score 0.939, essentially the same-speaker anchor. Distinct
voices go in and come out sounding much more alike than they went in.

In practice: a clone resembles its target, but put two clones side by side and
they sound like relatives. If you need voices that are clearly distinguishable
from each other, this model will disappoint you; if you need one voice that
resembles one target, it does that.

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
  50,000 steps, lr 4e-4 cosine, `data.trail_sec: 0.6`. Validation loss was
  still falling at the end, so it is not saturated. `distill_cfg_coef` is the
  measured optimum, not a default: swept on the teacher over 1/2/3/4/6 it peaks
  at 2.0 on similarity, WER and UTMOS at once, and at 6.0 the model is silent
  on 124 of 153 inputs.
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
- On long text the model occasionally drops one item from an enumeration.

## License and credits

CC-BY-4.0, following the LEMAS training data. Model architecture and base
weights from [Kyutai](https://huggingface.co/kyutai/pocket-tts) — see the
[CALM paper](https://arxiv.org/abs/2509.06926). Training data from the
[LEMAS Project](https://huggingface.co/datasets/LEMAS-Project/LEMAS-Dataset-train).
