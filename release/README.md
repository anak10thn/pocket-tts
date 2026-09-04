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

# Pocket TTS Indonesian (24L teacher)

Indonesian text-to-speech with voice cloning, built on
[kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts). A 24-layer
teacher finetuned from the released English weights with a fresh Indonesian
tokenizer, on 502 hours of [LEMAS](https://huggingface.co/datasets/LEMAS-Project/LEMAS-Dataset-train)
Indonesian speech.

> **This card was corrected on 2026-09-04, and the numbers below are weaker
> than what it claimed before.** An earlier version headlined WER 23.18% and
> said a sweep had put the best `--eos-threshold` at -4.0. Both were wrong:
> the first was measured with classifier-free guidance, which the shipped
> `pocket-tts` CLI cannot do, and the second came from a sweep run on the first
> 60 items of the eval list rather than a random sample. See
> [Evaluation](#evaluation) for what is actually measured and what is still
> uncertain. A distilled 6-layer student is training and will supersede this
> model; it is the one to wait for if you need dependable numbers.

## Usage

```bash
uvx pocket-tts generate \
    --config hf://anak10thn/pocket-tts-indonesian/config.yaml@6160f98e7a6e71e3aa3e063e582d7827518c632f \
    --voice your_voice.wav \
    --text "Selamat pagi, semoga hari Anda menyenangkan." \
    --eos-threshold -5.0 \
    --output-path out.wav
```

`--eos-threshold -5.0` is deliberate: the CLI's default of -4.0 scored 34.41%
corpus WER against -5.0's 20.82% on the same 153 items. Do not read that gap as
precise — see below — but -4.0 was the worst of the five values measured, so it
is worth overriding.

Omitting `--voice` is fine; it falls back to
[alba's audio](https://huggingface.co/kyutai/tts-voices/blob/main/alba-mackenna/casual.wav),
which this model clones like any other. Verified against pocket-tts 3.1.0 and
the current published wheel via `uvx`, both from a cleared cache.

From Python:

```python
from pocket_tts import TTSModel

model = TTSModel.load_model(
    config="hf://anak10thn/pocket-tts-indonesian/config.yaml"
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
split, none of whose recordings appear in training. Whisper-large-v3,
language-agnostic normalization, `--temp 0.3 --n-steps 1`, step 87,500.

**What the shipped CLI produces** (no guidance — `pocket_tts/` has no
`cfg_coef` anywhere):

| `--eos-threshold` | corpus WER | median per-item WER | items over 50% |
|---|---|---|---|
| -3.0 | 23.45% | not retained | — |
| -4.0 (CLI default) | 34.41% | not retained | — |
| **-5.0** | **20.82%** | not retained | — |
| -6.0 | 37.81% | 17.65% | 13 / 153 |
| -7.0 | 39.62% | 36.36% | 48 / 153 |

Speaker similarity 0.918–0.927 and UTMOS 2.32–2.45 across those settings.

**Corpus WER here is not a reliable number, and neither was the 23.18% this
card used to headline.** It sums insertions over the list, so one generation
that repeats itself past the end of the text outweighs everything else: the
worst single item in the -6.0 run scored 1833% and 13 items of 153 accounted
for the gap between 18.99% and 37.81%. Adjacent eos settings swing it by 14
points. The -6.0 and -7.0 rows show why the median matters — their corpus
numbers are 2 points apart while their medians differ by a factor of two, and
their failure modes are opposite (one runaway generation versus 48
truncations). Medians for the other rows were lost because the eval directory
name did not include eos, so each run overwrote the previous one's per-item
records; both that and the median reporting are fixed in the code linked below,
and the sweep will be redone properly on the distilled student.

**With guidance** (`--cfg 2.0`, reachable only through
`training/eval/librispeech.py`, not the CLI): WER 23.18%, speaker similarity
0.941, UTMOS 2.63. Guidance trades intelligibility for voice fidelity here
rather than being strictly better — and it costs two backbone passes per step,
which is what the distillation now running is for.

**Read WER against this corpus, not against the English models.** The same ASR
transcribing the eval set's *real* recordings scores 10.08%: the reference
transcripts are subtitle-derived and the audio is YouTube-sourced. The released
English model's 0.90% is a different language on clean read speech.

**Speaker similarity is this model's strength.** 0.941 with guidance and
~0.927 without, against 0.922 for the released English model and 0.929 for
kyutai's best 24-layer English teacher trained on 31,700 hours. LEMAS is
thousands of YouTube speakers rather than a handful of audiobook narrators, so
the model learned to imitate arbitrary voices rather than a house style. This
is the one claim here that has held up through every re-measurement.

**Audio quality is capped by the corpus, not by training.** UTMOS ~2.4 against
4.36 for the English model, because LEMAS audio is 16 kHz while Mimi runs at
24 kHz: there is no energy above 8 kHz to learn. Expect a decent phone call or
YouTube video, not a studio recording. Further training does not move this;
mixing in 48 kHz Indonesian speech would.

Training stopped at step 87,500 of a planned 250,000 because it had plateaued:
validation loss was flat from step 42,500 on, and WER moved 0.6 points between
steps 37,500 and 75,000.

## Training

- **Base**: `kyutai/pocket-tts` English 24L, text embedding reinitialised for
  the new tokenizer, backbone transferred.
- **Data**: 502 h / 298,652 utterances, one shard of LEMAS Indonesian filtered
  to duration >= 4 s and mean alignment confidence >= 0.8, split by recording.
- **Transcripts**: LEMAS Indonesian text is ALL CAPS with no punctuation.
  Punctuation and truecasing were restored with
  [1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase](https://huggingface.co/1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase)
  before fitting the tokenizer, so the model learns comma pauses and
  sentence-final intonation and users' punctuation is in-vocabulary.
- **Tokenizer**: 4000-piece BPE, `nmt_nfkc_cf` normalization (case folding baked
  into the .model).
- **Hardware**: one shared Tesla T4, fp16 autocast with a GradScaler (Turing has
  no bf16 tensor cores — bf16 measures 11x slower there). Effective batch 64
  via 4 x 16 gradient accumulation, lr 2e-4 constant, ~0.29 steps/s, 8 days.

Code, including the data pipeline, the fp16 patch and the eval fixes described
above:
[anak10thn/pocket-tts, branch `indonesian`](https://github.com/anak10thn/pocket-tts/tree/indonesian),
merged up to pocket-tts 3.1.0. It is kept on the fork rather than upstream by
[agreement with the maintainers](https://github.com/kyutai-labs/pocket-tts/pull/293);
the README entry is [kyutai-labs/pocket-tts#294](https://github.com/kyutai-labs/pocket-tts/pull/294),
currently on hold pending the numbers above being remeasured.

## Limitations

- 16 kHz-sourced audio: band-limited output, UTMOS ~2.4.
- Corpus WER is unstable on this eval set; a handful of runaway generations
  dominate it. Treat any single WER figure here as provisional.
- The CLI cannot do classifier-free guidance, so the model's better speaker
  similarity and UTMOS are not reachable from `pocket-tts generate` until the
  distilled student lands.
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
