#!/usr/bin/env python3
"""CPU-only diagnostics for already-computed score tables.

This script does not generate model outputs.  It audits truncation proxies,
the public-only K=5 baseline, D1 versus public-only across private n, and D8
with a public top-K fallback when stability release declines.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from dprandopt import designs
from dprandopt.scoretable import ScoreTable


def public_top_k(table: ScoreTable, k: int):
    return designs.select(table, "public", k=k,
                         pool_cols=np.arange(table.pool_size),
                         rng=np.random.default_rng(0))


def summarize(path: str, k: int, eps: float, reps: int, seed: int) -> dict:
    table = ScoreTable.load(path)
    base = table.pool_reward[table.base_index]
    bins = {
        "reward_le_0.01": int(np.sum(base <= 0.01)),
        "reward_0.01_to_0.05": int(np.sum((base > 0.01) & (base < 0.05))),
        "reward_0.05_to_0.11": int(np.sum((base >= 0.05) & (base < 0.11))),
        "reward_ge_1.0": int(np.sum(base >= 1.0)),
    }
    if table.output_lengths is None:
        length_diag = {
            "available": False,
            "note": "This legacy score table predates output_lengths; use reward bins only.",
        }
    else:
        pool_lengths = table.output_lengths[table.base_index, :table.pool_size]
        length_diag = {
            "available": True,
            "mean": float(np.mean(pool_lengths)),
            "p95": float(np.quantile(pool_lengths, 0.95)),
            "max": int(np.max(pool_lengths)),
        }
    public = public_top_k(table, k)
    full_pool = np.arange(table.pool_size)
    nonprivate = designs.select(table, "nonprivate", k=k,
                                pool_cols=full_pool,
                                rng=np.random.default_rng(1))
    n_values = {}
    rng = np.random.default_rng(seed)
    for n in (200, 500, 1000):
        vals = []
        for _ in range(reps):
            cols = table.subsample_pool(n, rng)
            vals.append(designs.select(table, "D1", k=k, pool_cols=cols,
                                       rng=rng, eps=eps).vote_acc)
        n_values[str(n)] = {
            "private_n": n,
            "mean_vote_acc": float(np.mean(vals)),
            "se": float(np.std(vals, ddof=1) / np.sqrt(len(vals))),
            "public_only_vote_acc": public.vote_acc,
        }

    raw_d8, fallback_d8, released = [], [], 0
    for _ in range(reps):
        cols = np.arange(table.pool_size)
        sel = designs.select(table, "D8", k=k, pool_cols=cols,
                             rng=rng, eps=eps)
        raw_d8.append(sel.vote_acc)
        if sel.info.get("released", False):
            released += 1
            fallback_d8.append(sel.vote_acc)
        else:
            fallback_d8.append(public.vote_acc)
    return {
        "table": os.path.basename(path),
        "pool_size": table.pool_size,
        "public_size": int(table.public_reward.shape[1]),
        "test_size": table.test_size,
        "base_test_accuracy": float(table.test_acc[table.base_index]),
        "base_pool_reward_mean": float(base.mean()),
        "base_pool_reward_bins": bins,
        "truncation_proxy_note": "Reward bins are a diagnostic proxy; the score table does not store generated token lengths.",
        "base_pool_output_lengths": length_diag,
        "fixed_k": k,
        "private_eps": eps,
        "public_only_k5_test_vote_acc": float(public.vote_acc),
        "nonprivate_fixed_k_test_vote_acc": float(nonprivate.vote_acc),
        "nonprivate_fixed_k_gain_points": float(100 * (nonprivate.vote_acc - table.test_acc[table.base_index])),
        "d1_vs_public_by_private_n": n_values,
        "d8_raw_mean_vote_acc": float(np.mean(raw_d8)),
        "d8_with_public_fallback_mean_vote_acc": float(np.mean(fallback_d8)),
        "d8_release_rate": released / reps,
        "d8_fallback_rate": 1.0 - released / reps,
        "reps": reps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tables", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--eps", type=float, default=8.0)
    ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args()
    rows = [summarize(p, a.k, a.eps, a.reps, 100 + i)
            for i, p in enumerate(a.tables)]
    out = {"diagnostics": rows, "no_gpu_generation": True}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
