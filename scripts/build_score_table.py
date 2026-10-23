#!/usr/bin/env python3
"""Build a DP-RandOpt score table on GPUs (Phase 0, plan weeks 1-2).

Executed on local and server Qwen Countdown runs. Requires the compatible
RandOpt checkout and optional CPU-base-weight worker patch documented in the
experimental artifacts; environments and model versions remain run-specific.

Run from a RandOpt checkout with this repository on PYTHONPATH:

    PYTHONPATH=/path/to/DP-RandOpt python /path/to/DP-RandOpt/scripts/build_score_table.py \
        --dataset countdown --model Qwen/Qwen2.5-1.5B-Instruct --n-candidates 2000 \
        --pool 2000 --public 200 --test 1000 --out tables/qwen1.5b_countdown.npz

Differences from randopt.py, all deliberate:
* every candidate is evaluated as copy(base) + sigma * noise(seed) via the
  worker's ``apply_perturbation`` (randopt.py's perturb/restore loop drifts in
  bf16; see results/e1_bf16_drift.md);
* candidate seeds/sigmas come from a public RNG and the base model is candidate 0;
* every candidate is scored on the private pool, the public proxy and the test
  set in one pass, and per-example rewards / answers are stored, not means;
* nothing data-dependent (no best sigma, no train rewards) is printed or logged
  beyond the private table file itself, which must be stored like the private data;
* throughput (generated tokens per second) is logged for re-planning the budget.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["countdown", "gsm8k"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-candidates", type=int, default=2000, help="perturbed candidates (base added as #0)")
    ap.add_argument("--sigmas", default="0.0005,0.001,0.002")
    ap.add_argument("--pool", type=int, default=2000)
    ap.add_argument("--public", type=int, default=200)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--gsm8k-train", default="data/gsm8k/train.parquet")
    ap.add_argument("--gsm8k-test", default="data/gsm8k/test.parquet")
    ap.add_argument("--num-engines", type=int, default=3)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--precision", choices=["bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.75,
                    help="vLLM GPU memory fraction; lower this when sharing a GPU")
    ap.add_argument("--base-weights-device", choices=["cuda", "cpu"], default="cuda",
                    help="where RandOpt stores the pristine base-weight copy")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20261005,
                    help="candidate/noise and decoding seed")
    ap.add_argument("--item-seed", type=int, default=None,
                    help="dataset generation/split seed; defaults to --seed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    return ap.parse_args()


def load_items(args, handler):
    """Return (pool, public, test) lists of RandOpt task dicts."""
    item_seed = args.seed if args.item_seed is None else args.item_seed
    if args.dataset == "countdown":
        from dprandopt.countdown import generate_instances, to_messages
        inst = generate_instances(args.pool + args.public + args.test, seed=item_seed)
        items = [{"messages": to_messages(x), "ground_truth": {"numbers": x["numbers"], "target": x["target"]},
                  "numbers": x["numbers"], "target": x["target"]} for x in inst]
        return (items[:args.pool], items[args.pool:args.pool + args.public], items[args.pool + args.public:])
    train = handler.load_data(args.gsm8k_train, split="train")
    rng = np.random.default_rng(item_seed)
    perm = rng.permutation(len(train))
    train = [train[i] for i in perm]
    test = handler.load_data(args.gsm8k_test, split="test")[: args.test]
    return train[: args.pool], train[args.pool: args.pool + args.public], test


def gold_string(args, item):
    if args.dataset == "countdown":
        return str(item["target"])
    from dprandopt.gsm8k import canonical_number
    return canonical_number(item['ground_truth'])


def vote_string(args, handler, text, item):
    if args.dataset == "countdown":
        ans, ok, _ = handler.extract_answer_for_voting(text, numbers=item["numbers"])
        return ans if ok else ""
    from dprandopt.gsm8k import canonical_number
    return canonical_number(handler.extract_answer(text) or '')


def main():
    args = parse()
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    import ray
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    from core import cleanup_engines, launch_engines          # RandOpt
    from data_handlers import get_dataset_handler              # RandOpt
    from dprandopt.countdown import encode_answers
    from dprandopt.scoretable import ScoreTable

    handler = get_dataset_handler(args.dataset)
    if args.item_seed is None:
        args.item_seed = args.seed
    pool, public, test = load_items(args, handler)
    items = pool + public + test
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [tok.apply_chat_template(it["messages"], add_generation_prompt=True, tokenize=False) for it in items]
    params = SamplingParams(temperature=0.0, seed=args.seed, max_tokens=args.max_tokens)

    # public candidate list: base + N perturbations, sigma drawn per candidate
    rng = np.random.default_rng(args.seed)
    sig_grid = [float(s) for s in args.sigmas.split(",")]
    seeds = np.concatenate([[-1], rng.choice(2 ** 31 - 1, size=args.n_candidates, replace=False)]).astype(np.int64)
    sigmas = np.concatenate([[0.0], rng.choice(sig_grid, size=args.n_candidates)])
    N, P, Q, T = len(seeds), len(pool), len(public), len(test)

    ckpt = args.out + ".partial.npz"
    rewards = np.zeros((N, P + Q), dtype=np.float32)
    output_lengths = np.zeros((N, P + Q + T), dtype=np.int32)
    answers = [[""] * T for _ in range(N)]
    done = np.zeros(N, dtype=bool)
    if os.path.exists(ckpt):
        z = np.load(ckpt, allow_pickle=True)
        rewards, done = z["rewards"], z["done"]
        answers = [list(a) for a in z["answers"]]
        if "output_lengths" in z.files:
            output_lengths = z["output_lengths"]
        if output_lengths.shape != (N, P + Q + T):
            raise ValueError(f"checkpoint output_lengths shape {output_lengths.shape} != {(N, P + Q + T)}")
        if rewards.shape != (N, P + Q) or done.shape != (N,):
            raise ValueError("checkpoint arrays do not match the requested experiment shape")

    os.environ["RANDOPT_BASE_WEIGHTS_DEVICE"] = args.base_weights_device
    if not ray.is_initialized():
        ray.init(address="auto" if os.environ.get("RAY_ADDRESS") else "local", ignore_reinit_error=True)
    engines, pgs = launch_engines(args.num_engines, args.model, precision=args.precision,
                                   tensor_parallel_size=args.tp,
                                   gpu_memory_utilization=args.gpu_memory_utilization)
    t0, tokens = time.time(), 0
    try:
        todo = [i for i in range(N) if not done[i]]
        for b in range(0, len(todo), len(engines)):
            batch = todo[b:b + len(engines)]
            setups = []
            for e, i in zip(engines, batch):
                if seeds[i] < 0:
                    setups.append(e.collective_rpc.remote("reset_to_base_weights", args=()))
                else:
                    setups.append(e.collective_rpc.remote("apply_perturbation", args=(int(seeds[i]), float(sigmas[i]))))
            ray.get(setups)
            outs = ray.get([e.generate.remote(prompts, params, use_tqdm=False) for e in engines[:len(batch)]])
            for i, out in zip(batch, outs):
                for j, o in enumerate(out):
                    text = o.outputs[0].text
                    n_tokens = len(o.outputs[0].token_ids)
                    output_lengths[i, j] = n_tokens
                    tokens += n_tokens
                    if j < P + Q:
                        if args.dataset == 'gsm8k':
                            from dprandopt.gsm8k import answer_and_reward
                            _, rewards[i, j] = answer_and_reward(handler,text,items[j]['ground_truth'])
                        else:
                            rewards[i, j] = float(handler.compute_reward(text, items[j]["ground_truth"]))
                    else:
                        answers[i][j - P - Q] = vote_string(args, handler, text, items[j])
                done[i] = True
            ray.get([e.collective_rpc.remote("reset_to_base_weights", args=()) for e in engines[:len(batch)]])
            n_done = int(done.sum())
            el = time.time() - t0
            print(f"[{n_done}/{N}] {el / 3600:.2f} h, {tokens / el:.0f} generated tok/s", flush=True)
            if n_done % args.checkpoint_every < len(engines):
                np.savez(ckpt, rewards=rewards, done=done, answers=np.array(answers, dtype=object),
                         output_lengths=output_lengths)
    finally:
        cleanup_engines(engines, pgs)

    reward_max = 1.1 if args.dataset == "countdown" else 1.0
    table = ScoreTable(pool_reward=rewards[:, :P], public_reward=rewards[:, P:], reward_max=reward_max,
                       test_answer=encode_answers(answers, [gold_string(args, it) for it in test]),
                       sigma=sigmas, seed=seeds, output_lengths=output_lengths, base_index=0)
    table.save(args.out)
    with open(args.out + ".meta.json", "w") as f:
        json.dump({"model": args.model, "dataset": args.dataset, "N": N, "pool": P, "public": Q, "test": T,
                   "sigmas": sig_grid, "seed": args.seed, "candidate_seed": args.seed,
                    "item_seed": args.item_seed,
                    "scoring": 'canonical_voting_answer_v1' if args.dataset=='gsm8k' else 'randopt_countdown',
                    "hours": (time.time() - t0) / 3600,
                   "generated_tokens_per_s": tokens / (time.time() - t0)}, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
