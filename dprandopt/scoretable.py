"""The stored score table: score once, post-process every privacy setting.

One GPU pass per (model, task) scores every candidate on a private pool, an
optional public proxy split and the test set. Every privacy level, ensemble
size, design and private-set size n is then cheap post-processing of this
table (the plan's "score once" design).

Layout (N candidates; candidate ``base_index`` is the unperturbed model):

=================  ==============  ===========================================
field              shape / dtype   meaning
=================  ==============  ===========================================
pool_reward        [N, P] float32  per-example reward in [0, reward_max]
test_answer        [N, T] int16    answer code per test item (0 gold, -1 none,
                                   >=1 distinct wrong answers, per item)
public_reward      [N, Q] float32  optional public-proxy rewards
sigma, seed        [N]             perturbation metadata (data-independent)
=================  ==============  ===========================================

Only ``pool_reward`` is private. Seeds, sigmas, the public split and the test
set never feed a private decision except through the mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ScoreTable:
    pool_reward: np.ndarray
    test_answer: np.ndarray
    reward_max: float = 1.0
    public_reward: Optional[np.ndarray] = None
    sigma: Optional[np.ndarray] = None
    seed: Optional[np.ndarray] = None
    base_index: int = 0
    name: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.pool_reward = np.asarray(self.pool_reward, dtype=np.float32)
        self.test_answer = np.asarray(self.test_answer, dtype=np.int16)
        if self.pool_reward.shape[0] != self.test_answer.shape[0]:
            raise ValueError("pool_reward and test_answer must have the same number of candidates")
        if self.pool_reward.min() < 0 or self.pool_reward.max() > self.reward_max + 1e-6:
            raise ValueError("pool rewards must lie in [0, reward_max]; clip or rescale them first")
        self._test_acc = None

    # -- sizes -------------------------------------------------------------
    @property
    def n_candidates(self) -> int:
        return self.pool_reward.shape[0]

    @property
    def pool_size(self) -> int:
        return self.pool_reward.shape[1]

    @property
    def test_size(self) -> int:
        return self.test_answer.shape[1]

    # -- per-candidate quantities ------------------------------------------
    @property
    def test_acc(self) -> np.ndarray:
        """Single-model test accuracy of every candidate."""
        if self._test_acc is None:
            self._test_acc = (self.test_answer == 0).mean(axis=1)
        return self._test_acc

    def pool_sum(self, cols: np.ndarray) -> np.ndarray:
        """Private sum-scores on a subset of pool columns."""
        return self.pool_reward[:, cols].sum(axis=1, dtype=np.float64)

    def subsample_pool(self, n: int, rng) -> np.ndarray:
        if n > self.pool_size:
            raise ValueError(f"n={n} exceeds pool size {self.pool_size}")
        return rng.choice(self.pool_size, size=n, replace=False)

    # -- geometry ----------------------------------------------------------
    def density(self, margin: float, cols: Optional[np.ndarray] = None) -> float:
        """Fraction of candidates beating the base by ``margin``.

        cols=None uses test accuracy (a population proxy); otherwise mean pool
        reward on ``cols`` divided by reward_max (an empirical density).
        """
        if cols is None:
            s = self.test_acc
        else:
            s = self.pool_reward[:, cols].mean(axis=1) / self.reward_max
        return float(np.mean(s >= s[self.base_index] + margin))

    # -- io ----------------------------------------------------------------
    def save(self, path: str) -> None:
        arrays = dict(pool_reward=self.pool_reward, test_answer=self.test_answer,
                      reward_max=np.float32(self.reward_max), base_index=np.int64(self.base_index))
        for k in ("public_reward", "sigma", "seed"):
            v = getattr(self, k)
            if v is not None:
                arrays[k] = v
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str, name: str = "") -> "ScoreTable":
        z = np.load(path)
        get = lambda k: z[k] if k in z.files else None
        return cls(pool_reward=z["pool_reward"], test_answer=z["test_answer"],
                   reward_max=float(z["reward_max"]), public_reward=get("public_reward"),
                   sigma=get("sigma"), seed=get("seed"), base_index=int(z["base_index"]), name=name)
