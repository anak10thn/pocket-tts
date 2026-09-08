"""data.trail_sec must actually reach the target window, not just parse.

The point of the knob is that 0.2s truncates targets on a corpus whose aligner
ends early, so a test that only checked the field exists would miss a wiring
bug -- and a wiring bug costs another 28 hours of distillation to discover.
"""

import json

import numpy as np
import soundfile

from training.dataloader.loader import DataLoader


def _manifest(tmp_path, dur=6.0, last_word_end=4.0):
    """One utterance whose words end well before the clip does."""
    wav = (np.random.default_rng(0).standard_normal(int(dur * 24000)) * 0.1).astype("float32")
    audio = tmp_path / "a.wav"
    soundfile.write(str(audio), wav, 24000)
    entry = {
        "path": str(audio),
        "duration": dur,
        "transcript": "satu dua tiga empat",
        "words": [
            {"word": "satu", "start": 0.2, "end": 1.0},
            {"word": "dua", "start": 1.2, "end": 2.0},
            {"word": "tiga", "start": 2.2, "end": 3.0},
            {"word": "empat", "start": 3.2, "end": last_word_end},
        ],
    }
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps(entry) + "\n")
    return manifest


def _loader(manifest, trail_sec=None):
    return DataLoader(
        jsonl=str(manifest),
        tokenize=lambda t: [1] * len(t.split()),
        batch_size=1,
        sample_rate=24000,
        frame_rate=12.5,
        max_duration_sec=30.0,
        max_voice_prompt_sec=5.0,
        rank=0,
        world_size=1,
        seed=0,
        shuffle=False,
        io_workers=1,
        trail_sec=trail_sec,
    )


def test_trail_sec_widens_the_target(tmp_path):
    manifest = _manifest(tmp_path, dur=6.0, last_word_end=4.0)
    short, long = _loader(manifest, trail_sec=0.2), _loader(manifest, trail_sec=1.5)
    assert (short.TRAIL_SEC, long.TRAIL_SEC) == (0.2, 1.5)

    # Same manifest and seed, so both take the same cut: the difference in
    # target length is the tail alone.
    entry = short.get_entry(0)
    wav_short, wav_long = short._sample(entry)[0], long._sample(entry)[0]
    assert len(wav_long) > len(wav_short), (len(wav_long), len(wav_short))
    # The target ends at min(duration, last_word_end + trail_sec): 4.2s
    # against 5.5s, so 1.3s more audio at 24 kHz.
    assert len(wav_long) - len(wav_short) > int(1.2 * 24000)


def test_omitting_trail_sec_keeps_the_default(tmp_path):
    assert _loader(_manifest(tmp_path)).TRAIL_SEC == DataLoader.TRAIL_SEC == 0.2
