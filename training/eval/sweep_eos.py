"""Sweep --eos-threshold on the full list and rank the runs by median WER.

Written after getting this wrong three times on one model. The traps, all of
which this encodes rather than leaves to whoever runs it next:

- Never sweep with --num-items. It takes the first N items of the list, not a
  random sample: eos -4.0 scored 21.66% on the first 60 items of one list and
  34.41% on all 153.
- Never rank on corpus WER. It sums insertions, so one generation that runs
  past the end of the text outweighs the setting under test. Two runs of one
  checkpoint came within 2 points of each other on corpus WER while their
  medians differed by a factor of two.
- Always pin the ASR's language. Whisper detects per clip and on short
  non-English audio sometimes returns a translation instead.

Each setting writes its own eval directory (eos is in the name), so every run
stays inspectable afterwards.

Usage:
    python -m training.eval.sweep_eos runs/distill_indonesian \
        --language id --cfg 1.0
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_EOS = (-2.0, -3.0, -4.0, -5.0, -6.0)


def run_one(args: argparse.Namespace, eos: float) -> dict[str, Any] | None:
    cmd = [
        sys.executable, "-m", "training.eval.librispeech", args.run_dir,
        "--librispeech-root", args.librispeech_root,
        "--list", args.list,
        "--prompt-root", "",
        "--asr", args.asr,
        "--asr-language", args.language,
        "--normalizer", "basic",
        "--temp", str(args.temp),
        "--n-steps", str(args.n_steps),
        "--cfg", str(args.cfg),
        "--eos-threshold", str(eos),
        "--batch-size", str(args.batch_size),
        # Whisper's feature extractor pads to exactly 30 s of mel frames, and a
        # generation that reaches the cap yields 3001 where it expects 3000,
        # killing the run. Staying under it costs nothing here: no reference
        # utterance in this eval is anywhere near 30 s.
        "--max-sec", str(args.max_sec),
    ]
    if args.use_ema:
        cmd.append("--use-ema")
    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]
    print(f"\n=== eos {eos} ===", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  failed ({proc.returncode}); last stderr:\n{proc.stderr[-800:]}")
        return None
    # librispeech.py logs the results object as its final line.
    for line in reversed(proc.stdout.splitlines() + proc.stderr.splitlines()):
        if "FINAL" in line:
            return json.loads(line[line.index("{"):])
    print("  no FINAL line found")
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--librispeech-root", default="data/eval_id/audio")
    ap.add_argument("--list", default="data/eval_id/pairs.lst")
    ap.add_argument("--language", required=True, help="ASR language, e.g. 'id'")
    ap.add_argument("--asr", default="openai/whisper-large-v3")
    ap.add_argument("--cfg", type=float, default=1.0, help="1.0 is what the shipped CLI does")
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--n-steps", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-sec", type=float, default=29.0)
    ap.add_argument("--use-ema", action="store_true", default=True)
    ap.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--eos", type=float, nargs="+", default=list(DEFAULT_EOS))
    ap.add_argument("--out", default=None, help="write the ranked table here as JSON")
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for eos in args.eos:
        r = run_one(args, eos)
        if r:
            r["eos"] = eos
            rows.append(r)
            print(
                f"  median {r['wer_median']:.2%} | corpus {r['wer']:.2%} | "
                f"over50 {r['wer_over_50']}/{r['num_items']} | sim {r['sim']:.3f} | "
                f"utmos {r['utmos']:.2f} | silent {r['silent']} no_eos {r['no_eos']}"
            )

    if not rows:
        raise SystemExit("every run failed")

    rows.sort(key=lambda r: r["wer_median"])
    print("\n=== ranked by median per-item WER (the robust one) ===")
    print(
        f"{'eos':>6} {'median':>8} {'corpus':>8} {'over50':>7} {'silent':>7} "
        f"{'no_eos':>7} {'sim':>6} {'utmos':>6}"
    )
    for r in rows:
        print(
            f"{r['eos']:>6} {r['wer_median']:>7.2%} {r['wer']:>7.2%} "
            f"{r['wer_over_50']:>7} {r['silent']:>7} {r['no_eos']:>7} "
            f"{r['sim']:>6.3f} {r['utmos']:>6.2f}"
        )

    # A setting that stays silent on some inputs can still win on the median:
    # silence is scored as one item at 100%, and a handful of those does not
    # move a 153-item median. Nobody wants the setting that answers 88% of the
    # time, so those are ranked behind anything that speaks reliably.
    speaks = [r for r in rows if r["silent"] <= 0.02 * r["num_items"]]
    best = (speaks or rows)[0]
    if speaks and speaks[0] is not rows[0]:
        dropped = [r for r in rows if r not in speaks]
        print(
            "\nignoring "
            + ", ".join(f"eos {r['eos']} ({r['silent']}/{r['num_items']} silent)" for r in dropped)
            + " -- a lower median does not help if the model will not speak"
        )
    print(f"\nbest eos {best['eos']}: median {best['wer_median']:.2%}, corpus {best['wer']:.2%}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
