from collections import Counter

import numpy as np

from dprandopt.vote import vote_winner


def reference(col):
    answers = [a for a in col if a >= 0]
    if not answers:
        return -1
    return Counter(answers).most_common(1)[0][0]


def test_vote_matches_python_counter():
    rng = np.random.default_rng(0)
    for _ in range(300):
        K, T = rng.integers(1, 12), rng.integers(1, 30)
        A = rng.integers(-1, 4, size=(K, T))
        got = vote_winner(A)
        want = [reference(A[:, t].tolist()) for t in range(T)]
        assert got.tolist() == want
