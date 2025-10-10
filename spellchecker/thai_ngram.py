"""
Lightweight Thai character n-gram scorer (no external deps).

Usage:
- The language model is built lazily from pythainlp.corpus.thai_words() as a
  small proxy corpus of "valid words". This is not perfect, but gives a quick
  signal for whether character sequences look Thai-like.
- We use add-k smoothing and return average log-prob per character to compare
  alternatives fairly.
"""
from __future__ import annotations

import math
import threading
from typing import Dict, Tuple

try:
    from pythainlp.corpus import thai_words
except Exception:  # pragma: no cover
    thai_words = lambda: []  # type: ignore

_LM_LOCK = threading.Lock()
_LM_STATE: Dict[str, object] | None = None


def _build_lm(n: int = 3, add_k: float = 0.5):
    words = list(thai_words())
    if not words:
        words = []
    bos, eos = "<", ">"
    counts: Dict[Tuple[str, ...], int] = {}
    totals: Dict[Tuple[str, ...], int] = {}
    vocab = set()
    for w in words:
        s = f"{bos}{w}{eos}"
        vocab.update(s)
        chars = list(s)
        for i in range(len(chars)):
            for order in range(1, n + 1):
                if i + order > len(chars):
                    break
                ngram = tuple(chars[i : i + order])
                counts[ngram] = counts.get(ngram, 0) + 1
                if order > 1:
                    hist = ngram[:-1]
                    totals[hist] = totals.get(hist, 0) + 1
    # Store
    return {
        "n": n,
        "add_k": add_k,
        "counts": counts,
        "totals": totals,
        "vocab": vocab,
        "bos": bos,
        "eos": eos,
    }


def _ensure_lm():
    global _LM_STATE
    if _LM_STATE is not None:
        return _LM_STATE
    with _LM_LOCK:
        if _LM_STATE is None:
            _LM_STATE = _build_lm()
    return _LM_STATE


def score_string(text: str) -> float:
    """Average log-prob per character under the char n-gram LM."""
    st = _ensure_lm()
    n = int(st["n"])  # type: ignore
    add_k = float(st["add_k"])  # type: ignore
    counts: Dict[Tuple[str, ...], int] = st["counts"]  # type: ignore
    totals: Dict[Tuple[str, ...], int] = st["totals"]  # type: ignore
    vocab = st["vocab"]  # type: ignore
    bos = st["bos"]  # type: ignore
    eos = st["eos"]  # type: ignore

    s = f"{bos}{text}{eos}"
    chars = list(s)
    logp = 0.0
    steps = 0
    vsize = max(1, len(vocab))
    for i in range(len(chars)):
        for order in range(1, n + 1):
            if i + order > len(chars):
                break
            ngram = tuple(chars[i : i + order])
            if order == 1:
                # Unigram
                num = counts.get(ngram, 0) + add_k
                den = sum(counts.get((c,), 0) for c in vocab) + add_k * vsize
            else:
                hist = ngram[:-1]
                num = counts.get(ngram, 0) + add_k
                den = totals.get(hist, 0) + add_k * vsize
            if den <= 0:
                continue
            p = num / den
            logp += math.log(max(p, 1e-12))
            steps += 1
    return logp / max(1, steps)


def score_sentence_delta(sentence: str, orig: str, corr: str) -> float:
    """Score delta when replacing `orig` with `corr` in the sentence.

    Positive delta means `corr` is more probable than `orig`.
    """
    try:
        if not orig or not corr or orig == corr:
            return 0.0
        base = score_string(sentence)
        s2 = sentence.replace(orig, corr)
        alt = score_string(s2)
        return alt - base
    except Exception:
        return 0.0

