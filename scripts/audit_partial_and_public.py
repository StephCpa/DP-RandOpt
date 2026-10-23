#!/usr/bin/env python3
"""CPU-only audit of a partial score table and public/private selection gaps."""

from __future__ import annotations

import argparse
import csv
import json
import os
from copy import deepcopy

import numpy as np

from dprandopt import designs
from dprandopt.scoretable import ScoreTable


def select_public_random_ties(table, k, q=None, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    scores = table.public_reward[:, :(q or table.public_reward.shape[1])].sum(axis=1)
    keys = rng.random(table.n_candidates)
    idx = np.lexsort((keys, -scores))[:k]
    return designs.Selection("public_random_ties", idx, table.test_answer[idx])


def copy_answer_only(table):
    return ScoreTable(
        pool_reward=(table.pool_reward >= 1.0).astype(np.float32),
        public_reward=(table.public_reward >= 1.0).astype(np.float32),
        test_answer=table.test_answer.copy(), reward_max=1.0,
        sigma=None if table.sigma is None else table.sigma.copy(),
        seed=None if table.seed is None else table.seed.copy(),
        output_lengths=table.output_lengths,
        base_index=table.base_index, name=table.name + "-answer-only",
    )


def vote_acc(table, design, k, cols, reps, eps, rng, **kw):
    vals, vectors = [], []
    for _ in range(reps):
        s = designs.select(table, design, k=k, pool_cols=cols, rng=rng, eps=eps, **kw)
        vals.append(s.vote_acc)
        vectors.append(s.vote_correct)
    return float(np.mean(vals)), np.asarray(vectors, dtype=float)


def bootstrap_diff(public_vec, private_vectors, reps, rng):
    diffs = np.empty(reps, dtype=float)
    for i in range(reps):
        idx = rng.integers(0, len(public_vec), size=len(public_vec))
        j = int(rng.integers(0, len(private_vectors)))
        diffs[i] = private_vectors[j, idx].mean() - public_vec[idx].mean()
    return {"mean_diff_private_minus_public": float(np.mean(private_vectors.mean(axis=1) - public_vec.mean())),
            "ci95": [float(x) for x in np.quantile(diffs, [0.025, 0.975])],
            "bootstrap_reps": reps}


def row_for_table(path, k, eps, reps, rng_seed):
    table = ScoreTable.load(path)
    rng = np.random.default_rng(rng_seed)
    cols = np.arange(table.pool_size)
    public = designs.select(table, "public", k=k, pool_cols=cols, rng=rng)
    d1_acc, d1_vec = vote_acc(table, "D1", k, cols, reps, eps, rng)
    tie_public = select_public_random_ties(table, k, rng=rng)
    tie_acc = tie_public.vote_acc
    public_scores = table.public_reward.sum(axis=1)
    cutoff = np.partition(public_scores, -k)[-k]
    tie_count = int(np.sum(public_scores == cutoff))

    answer_only = copy_answer_only(table)
    ao_public = designs.select(answer_only, "public", k=k, pool_cols=cols, rng=rng)
    ao_d1_acc, _ = vote_acc(answer_only, "D1", k, cols, reps, eps, rng)

    pub_order = np.argsort(-public_scores, kind="stable")
    priv_scores = table.pool_reward.sum(axis=1)
    priv_order = np.argsort(-priv_scores, kind="stable")
    overlap = len(set(pub_order[:k]).intersection(set(priv_order[:min(50, table.n_candidates)])))
    ao_pub_order = np.argsort(-answer_only.public_reward.sum(axis=1), kind="stable")
    ao_priv_order = np.argsort(-answer_only.pool_reward.sum(axis=1), kind="stable")
    ao_overlap = len(set(ao_pub_order[:k]).intersection(set(ao_priv_order[:min(50, table.n_candidates)])))

    base_lengths = None
    typical_lengths = None
    if table.output_lengths is not None:
        lengths = table.output_lengths[:, :table.pool_size + table.public_reward.shape[1] + table.test_size]
        cap = 512
        base_lengths = {
            "share_eq_512": float(np.mean(lengths[0] == cap)),
            "pool_share_eq_512": float(np.mean(lengths[0, :table.pool_size] == cap)),
            "public_share_eq_512": float(np.mean(lengths[0, table.pool_size:table.pool_size + table.public_reward.shape[1]] == cap)),
            "test_share_eq_512": float(np.mean(lengths[0, table.pool_size + table.public_reward.shape[1]:] == cap)),
            "mean": float(np.mean(lengths[0])),
        }
        per_candidate = np.mean(lengths == cap, axis=1)
        typical_lengths = {
            "median_candidate_share_eq_512": float(np.median(per_candidate[1:])),
            "candidate_share_eq_512_q25_q75": [float(x) for x in np.quantile(per_candidate[1:], [0.25, 0.75])],
            "median_candidate_mean_length": float(np.median(np.mean(lengths[1:], axis=1))),
        }

    result = {
        "table": os.path.basename(path),
        "n_candidates": table.n_candidates,
        "pool": table.pool_size,
        "public": int(table.public_reward.shape[1]),
        "test": table.test_size,
        "base_test_accuracy": float(table.test_acc[table.base_index]),
        "fixed_k": k,
        "eps": eps,
        "public_only_acc": float(public.vote_acc),
        "d1_acc": d1_acc,
        "d1_minus_public": float(d1_acc - public.vote_acc),
        "public_random_tie_acc_one_draw": float(tie_acc),
        "public_cutoff_score": float(cutoff),
        "public_cutoff_tie_count": tie_count,
        "answer_only_public_acc": float(ao_public.vote_acc),
        "answer_only_d1_acc": ao_d1_acc,
        "answer_only_d1_minus_public": float(ao_d1_acc - ao_public.vote_acc),
        "public_top5_private_top50_overlap": overlap,
        "answer_only_public_top5_private_top50_overlap": ao_overlap,
        "base_truncation": base_lengths,
        "typical_candidate_truncation": typical_lengths,
    }
    result["bootstrap"] = bootstrap_diff(public.vote_correct, d1_vec, 2000, rng)

    # Public-size curve and private-size curve at fixed epsilon.
    curve = []
    for q in (0, 5, 10, 25, 50):
        if q == 0:
            acc = float(table.test_acc[table.base_index])
        else:
            acc = float(designs.select(table, "public", k=k, public_q=q,
                                       pool_cols=cols, rng=rng).vote_acc)
        curve.append({"axis": "public", "size": q, "accuracy": acc})
    for n in (200, 500, 1000):
        vals = []
        for _ in range(reps):
            c = table.subsample_pool(min(n, table.pool_size), rng)
            vals.append(designs.select(table, "D1", k=k, pool_cols=c, rng=rng, eps=eps).vote_acc)
        curve.append({"axis": "private_D1", "size": n, "accuracy": float(np.mean(vals))})
    result["equivalence_curve"] = curve
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tables", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--curve-csv", required=True)
    ap.add_argument("--plot", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--eps", type=float, default=8.0)
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    rows = [row_for_table(p, a.k, a.eps, a.reps, 20261013 + i)
            for i, p in enumerate(a.tables)]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"diagnostics": rows, "cpu_only": True, "no_gpu_generation": True}, f, indent=2)
    with open(a.curve_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["table", "axis", "size", "accuracy"])
        w.writeheader()
        for row in rows:
            for x in row["equivalence_curve"]:
                w.writerow({"table": row["table"], **x})
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for row in rows:
            pub = [x for x in row["equivalence_curve"] if x["axis"] == "public"]
            priv = [x for x in row["equivalence_curve"] if x["axis"] == "private_D1"]
            label = row["table"]
            ax.plot([x["size"] for x in pub], [x["accuracy"] for x in pub], marker="o", label=label + " public-only")
            ax.plot([x["size"] for x in priv], [x["accuracy"] for x in priv], marker="s", linestyle="--", label=label + " D1 eps=8")
        ax.set_xlabel("Selection-data size")
        ax.set_ylabel("Test vote accuracy")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=160)
        plt.close(fig)
    except Exception as e:
        print("plot skipped:", e)
    print(json.dumps({"tables": len(rows), "out": a.out, "curve_csv": a.curve_csv, "plot": a.plot}, indent=2))


if __name__ == "__main__":
    main()
