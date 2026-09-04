"""The robust WER statistics librispeech.py reports alongside corpus WER.

Corpus WER is insertion-dominated, so these exist to keep one runaway
generation from deciding which of two evals looks better.
"""

import jiwer


def stats(pairs: list[tuple[str, str]]) -> tuple[float, float, int]:
    """Mirror of the aggregation in librispeech.main: corpus, median, tail."""
    refs = [r for r, _ in pairs]
    hyps = [h for _, h in pairs]
    per_item = sorted(
        jiwer.wer(r, h) if r.strip() else 0.0 for r, h in zip(refs, hyps, strict=True)
    )
    return (
        jiwer.wer(refs, hyps),
        per_item[len(per_item) // 2],
        sum(w > 0.5 for w in per_item),
    )


def test_one_runaway_item_moves_corpus_wer_but_not_the_median():
    good = ("satu dua tiga empat", "satu dua tiga empat")
    # A generation that repeats itself far past the reference: pure insertions.
    runaway = ("satu dua tiga empat", "satu dua tiga empat " + "lagi " * 40)

    corpus_clean, median_clean, tail_clean = stats([good] * 9 + [good])
    corpus, median, tail = stats([good] * 9 + [runaway])

    assert corpus_clean == 0.0 and median_clean == 0.0 and tail_clean == 0
    # 40 insertions against 40 reference words across the list.
    assert corpus == 1.0, corpus
    # The median is untouched -- that is the whole point.
    assert median == 0.0, median
    assert tail == 1, tail


def test_tail_counts_only_items_past_50_percent():
    exact = ("satu dua tiga empat", "satu dua tiga empat")
    quarter = ("satu dua tiga empat", "satu dua tiga lima")  # 1 of 4 wrong
    most = ("satu dua tiga empat", "lima enam tujuh empat")  # 3 of 4 wrong

    _, _, tail = stats([exact, quarter, most])
    assert tail == 1, tail


def test_empty_reference_is_scored_zero_not_a_crash():
    # read_lst can hand back an item whose normalized reference is empty.
    _, median, tail = stats([("", "apa pun"), ("satu dua", "satu dua")])
    assert median in (0.0,), median
    assert tail == 0, tail
