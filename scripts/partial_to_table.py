#!/usr/bin/env python3
"""Convert an ordered build_score_table partial checkpoint into a ScoreTable.

The builder draws candidate seeds and sigmas before generation and processes
candidates in index order.  This makes a completed prefix a valid smaller
score table for CPU-only analysis.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from dprandopt.countdown import encode_answers, generate_instances
from dprandopt.scoretable import ScoreTable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("partial")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", choices=["countdown"], default="countdown")
    ap.add_argument("--n-candidates", type=int, default=1000)
    ap.add_argument("--pool", type=int, required=True)
    ap.add_argument("--public", type=int, required=True)
    ap.add_argument("--test", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--item-seed", type=int, required=True)
    ap.add_argument("--sigmas", default="0.0005,0.001,0.002")
    args = ap.parse_args()

    z = np.load(args.partial, allow_pickle=True)
    done = np.asarray(z["done"], dtype=bool)
    n_done = int(done.sum())
    expected = np.arange(n_done)
    if not np.array_equal(np.flatnonzero(done), expected):
        raise ValueError("partial checkpoint is not a completed prefix")

    rng = np.random.default_rng(args.seed)
    sig_grid = [float(s) for s in args.sigmas.split(",")]
    seeds = np.concatenate([[-1], rng.choice(2 ** 31 - 1, size=args.n_candidates,
                                                replace=False)]).astype(np.int64)
    sigmas = np.concatenate([[0.0], rng.choice(sig_grid, size=args.n_candidates)])

    items = generate_instances(args.pool + args.public + args.test, seed=args.item_seed)
    test = items[args.pool + args.public:]
    gold = [str(x["target"]) for x in test]
    answers = [list(row) for row in z["answers"][:n_done]]
    output_lengths = z["output_lengths"][:n_done]
    table = ScoreTable(
        pool_reward=z["rewards"][:n_done, :args.pool],
        public_reward=z["rewards"][:n_done, args.pool:args.pool + args.public],
        test_answer=encode_answers(answers, gold),
        reward_max=1.1,
        sigma=sigmas[:n_done],
        seed=seeds[:n_done],
        output_lengths=output_lengths,
        base_index=0,
        name=os.path.basename(args.out),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    table.save(args.out)
    with open(args.out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"source_partial": os.path.abspath(args.partial),
                   "completed_candidates": n_done, "n_candidates_planned": args.n_candidates,
                   "pool": args.pool, "public": args.public, "test": args.test,
                   "seed": args.seed, "item_seed": args.item_seed,
                   "sigmas": sig_grid}, f, indent=2)
    print(json.dumps({"out": args.out, "completed_candidates": n_done,
                      "pool": args.pool, "public": args.public, "test": args.test}, indent=2))


if __name__ == "__main__":
    main()
