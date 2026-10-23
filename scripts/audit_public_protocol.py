#!/usr/bin/env python3
"""Audit a test-free public-proxy rule on existing score tables.

The public split is divided deterministically into tuning and validation
columns.  K is selected by validation reward only, then the chosen K is used
for private D1/D7/D8 evaluations.  Test labels are reported only as a marked
diagnostic and never enter the K choice.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from dprandopt import designs
from dprandopt.scoretable import ScoreTable


def audit(path: str, ks: list[int], reps: int, eps: float, seed: int) -> dict:
    table = ScoreTable.load(path)
    if table.public_reward is None or table.public_reward.shape[1] < 4:
        raise ValueError("the score table needs a public split")
    q = table.public_reward.shape[1]
    mid = q // 2
    tune = table.public_reward[:, :mid].sum(axis=1, dtype=np.float64)
    valid = table.public_reward[:, mid:].mean(axis=1, dtype=np.float64)
    k_rows = []
    for k in ks:
        idx = np.argsort(-tune, kind="stable")[:k]
        k_rows.append({"k": k, "validation_reward": float(valid[idx].mean())})
    best = max(k_rows, key=lambda x: (x["validation_reward"], -x["k"]))

    rng = np.random.default_rng(seed)
    pool_cols = np.arange(table.pool_size)
    private = {}
    for design in ("D1", "D7", "D8"):
        vals = []
        for _ in range(reps):
            sel = designs.select(table, design, k=best["k"], pool_cols=pool_cols,
                                 rng=rng, eps=eps)
            vals.append(sel.vote_acc)
        private[design] = {"mean_vote_acc": float(np.mean(vals)),
                           "se": float(np.std(vals, ddof=1) / np.sqrt(len(vals))
                                      if len(vals) > 1 else 0.0)}

    # This is deliberately labelled diagnostic: test labels are not used above.
    public_idx = np.argsort(-table.public_reward.sum(axis=1), kind="stable")[:best["k"]]
    diagnostic_test_acc = float((table.test_answer[public_idx] == 0).mean())
    return {"table": os.path.basename(path), "public_columns": q,
            "tuning_columns": mid, "validation_columns": q - mid,
            "k_validation_curve": k_rows, "chosen_k": best["k"],
            "chosen_validation_reward": best["validation_reward"],
            "private_eps": eps, "private": private,
            "public_test_accuracy_diagnostic": diagnostic_test_acc}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tables", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="1,5,10,20,50,100")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--eps", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",")]
    rows = [audit(p, ks, a.reps, a.eps, a.seed + i) for i, p in enumerate(a.tables)]
    pooled = {str(k): float(np.mean([next(r["validation_reward"] for r in row["k_validation_curve"] if r["k"] == k)
                                      for row in rows])) for k in ks}
    pooled_k = min([k for k, v in pooled.items() if v == max(pooled.values())], key=int)
    out = {"rule": "split public columns in half; choose smallest K with maximal validation reward",
           "eps": a.eps, "tables": rows, "pooled_validation_reward": pooled,
           "pooled_chosen_k": int(pooled_k)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
