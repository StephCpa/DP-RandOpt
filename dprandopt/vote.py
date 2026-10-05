"""Majority voting over stored answer codes, matching RandOpt's tie rule.

Answers are integer codes per test item:
  * 0   -> the gold answer,
  * >=1 -> a distinct wrong answer (codes are per item),
  * -1  -> no valid answer (ignored by the vote, as in RandOpt).

RandOpt builds ``Counter(answers)`` over the K members in rank order and takes
``most_common(1)``; on ties Python returns the answer that occurred first, so
ties go to the earliest-ranked member's answer. ``vote_correct`` reproduces
that rule exactly. Pass members in the mechanism's output order (Gumbel order
for D1; a score-independent random order for D7).
"""

from __future__ import annotations

import numpy as np


def vote_winner(answers: np.ndarray) -> np.ndarray:
    """Return the voted code per item (-1 if every member abstained).

    answers: int array [K, T].
    """
    A = np.asarray(answers)
    K, T = A.shape
    if K == 0:
        return np.full(T, -1, dtype=int)
    max_code = int(A.max()) if A.size else -1
    if max_code < 0:
        return np.full(T, -1, dtype=int)
    codes = np.arange(max_code + 1)
    hit = A[:, :, None] == codes[None, None, :]          # [K, T, C]
    counts = hit.sum(axis=0)                              # [T, C]
    first = np.where(hit, np.arange(K)[:, None, None], K).min(axis=0)  # [T, C]
    best = counts.max(axis=1)
    # among codes with the max count, the earliest first occurrence wins
    key = np.where(counts == best[:, None], first, K + 1)
    winner = key.argmin(axis=1)
    winner = np.where(best > 0, winner, -1)
    return winner


def vote_correct(answers: np.ndarray) -> np.ndarray:
    """Boolean [T]: is the RandOpt majority vote of these K members correct?"""
    return vote_winner(answers) == 0


def vote_accuracy(answers: np.ndarray) -> float:
    return float(vote_correct(answers).mean())
