"""CPU-only audit of the frozen 32-candidate length pilots.

All intervals resample test items, conditional on the saved candidate pool.
No cap or K is selected from test performance by this script.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from dprandopt import designs
from dprandopt.countdown import encode_answers
from dprandopt.pilot import digest, prepare, read_candidate
from dprandopt.scoretable import ScoreTable
from dprandopt.vote import vote_correct


def paired(a, b):
    diff = np.asarray(a, float) - np.asarray(b, float)
    if diff.ndim == 2:
        diff = diff.mean(axis=0)
    rng = np.random.default_rng(20261022)
    boot = diff[rng.integers(len(diff), size=(5000, len(diff)))].mean(axis=1)
    return dict(difference=float(diff.mean()), conditional_item_ci95=np.quantile(boot, [.025, .975]).tolist())


def selection_audit(t, reps):
    rng = np.random.default_rng(20261023)
    def select(name, cols):
        return designs.select(t, name, k=5, pool_cols=cols, rng=rng, eps=8, delta=1e-5)
    cols = np.arange(t.pool_size)
    pub, non = select('public', cols), select('nonprivate', cols)
    d1 = np.array([select('D1', cols).vote_correct for _ in range(reps)])
    random = np.array([select('random', cols).vote_correct for _ in range(reps)])
    sizes = []
    for n in (200, 500, 1000):
        draws = np.array([select('D1', rng.choice(t.pool_size, n, replace=False)).vote_correct for _ in range(reps)])
        sizes.append(dict(n=n, accuracy=float(draws.mean()), versus_public=paired(draws.mean(axis=0), pub.vote_correct)))
    sensitivity = {}
    for name, table in [('original', t), ('answer_only', ScoreTable(
            pool_reward=(t.pool_reward >= 1).astype(float),
            public_reward=(t.public_reward >= 1).astype(float),
            test_answer=t.test_answer, reward_max=1.0))]:
        scores = table.public_reward.sum(axis=1, dtype=np.float64).round(5)
        tie_vectors = np.array([vote_correct(table.test_answer[
            np.lexsort((rng.random(table.n_candidates), -scores))[:5]]) for _ in range(reps)])
        def pick(method):
            return designs.select(table, method, k=5, pool_cols=cols, rng=rng, eps=8, delta=1e-5)
        pv = pick('public').vote_correct
        dv = np.array([pick('D1').vote_correct for _ in range(reps)])
        sensitivity[name] = dict(public_accuracy=float(pv.mean()), nonprivate_accuracy=pick('nonprivate').vote_acc,
            D1_accuracy=float(dv.mean()), cutoff_ties=int((scores == np.sort(scores)[-5]).sum()),
            random_tie_public_accuracy=float(tie_vectors.mean()),
            D1_vs_random_tie_public=paired(dv.mean(axis=0), tie_vectors.mean(axis=0)))
    return dict(k=5, eps=8, delta=1e-5, n=1000, q=50, repeats=reps,
                public_accuracy=pub.vote_acc, nonprivate_accuracy=non.vote_acc,
                D1_accuracy=float(d1.mean()), random5_accuracy=float(random.mean()),
                public_members=pub.members.tolist(), nonprivate_members=non.members.tolist(),
                D1_vs_public=paired(d1.mean(axis=0), pub.vote_correct),
                D1_vs_nonprivate=paired(d1.mean(axis=0), non.vote_correct), private_sizes=sizes,
                reward_and_tie_sensitivity=sensitivity)


def inspect_pilot(folder, reference):
    m = json.loads((folder/'manifest.json').read_text())
    expected, items = prepare(reference, 32, m['max_tokens'])
    assert expected == m
    completion = json.loads((folder/'COMPLETE.json').read_text())
    assert completion['manifest_sha256'] == digest(m)
    assert completion['count'] == 32 and completion['responses'] == 49600
    rows = [read_candidate(folder/f'candidate_{i:04d}.json.gz', m, i)['records'] for i in range(32)]
    t = ScoreTable.load(str(folder/'table.npz'))
    rewards = np.array([[r['reward'] for r in row] for row in rows], dtype=np.float32)
    lengths = np.array([[r['length'] for r in row] for row in rows])
    answers = [[r['vote_answer'] for r in row[1050:]] for row in rows]
    gold = [str(x['target']) for x in items[1050:]]
    assert np.array_equal(t.pool_reward, rewards[:, :1000])
    assert np.array_equal(t.public_reward, rewards[:, 1000:1050])
    assert np.array_equal(t.test_answer, encode_answers(answers, gold))
    assert np.array_equal(t.output_lengths, lengths)
    assert np.array_equal(t.seed, m['seeds']) and np.array_equal(t.sigma, m['sigmas'])
    stats = {}
    for reason in ('stop', 'length'):
        records = [r for row in rows for r in row if r['finish_reason'] == reason]
        repeated = 0
        for r in records:
            lines = [re.sub(r'\s+', ' ', s).strip() for s in r['text'].splitlines()]
            counts = Counter(s for s in lines if len(s) >= 20)
            repeated += bool(counts and max(counts.values()) >= 3)
        stats[reason] = dict(count=len(records),
            answer_tag_fraction=float(np.mean(['</answer>' in r['text'] for r in records])),
            valid_vote_fraction=float(np.mean([bool(r['vote_answer']) for r in records])),
            repeated_line_fraction=repeated/len(records))
    return t, rows, dict(table_sha256=hashlib.sha256((folder/'table.npz').read_bytes()).hexdigest(),
                         records_rebuild_table_exactly=True, finish_groups=stats,
                         repetition_definition='At least one whitespace-normalized line of >=20 characters occurs >=3 times; heuristic, not a loop classifier.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('results/gpu-smoke'))
    p.add_argument('--reps', type=int, default=400)
    a = p.parse_args()
    reference = a.root/'lzn_route1_n1001_max512.npz'
    full = ScoreTable.load(str(reference))
    small = ScoreTable(pool_reward=full.pool_reward[:32], public_reward=full.public_reward[:32],
                       test_answer=full.test_answer[:32], reward_max=full.reward_max,
                       output_lengths=full.output_lengths[:32])
    tables, responses, checks = {512: small}, {}, {}
    for cap in (1024, 2048):
        folder = a.root/f'length_pilot_32_max{cap}'
        tables[cap], responses[cap], checks[cap] = inspect_pilot(folder, reference)
    results = {}
    for cap, t in tables.items():
        print(f'Auditing cap {cap}', flush=True)
        results[cap] = dict(base_accuracy=float(t.test_acc[0]), perturbed_mean_accuracy=float(t.test_acc[1:].mean()),
            cap_hit=float((t.output_lengths == cap).mean()), selection=selection_audit(t, a.reps))
    comparisons = {}
    for old, new in ((512, 1024), (512, 2048), (1024, 2048)):
        comparisons[f'{new}_minus_{old}'] = dict(
            base=paired(tables[new].test_answer[0] == 0, tables[old].test_answer[0] == 0),
            perturbations=paired(tables[new].test_answer[1:] == 0, tables[old].test_answer[1:] == 0))
    groups = {}
    for reason in ('stop', 'length'):
        pairs = [(x, y) for xs, ys in zip(responses[1024], responses[2048]) for x, y in zip(xs, ys) if x['finish_reason'] == reason]
        groups[reason] = dict(count=len(pairs), exact_text_matches=sum(x['text'] == y['text'] for x,y in pairs),
            now_stop=sum(y['finish_reason'] == 'stop' for x,y in pairs))
    out = dict(scope='Exploratory N=32, fixed K=5; conditional test-item intervals only; not a causal cap effect or a full-pool replication.',
               selection_seed=20261023, bootstrap_seed=20261022, bootstrap_reps=5000,
               results=results, comparisons=comparisons, integrity=checks, cross_cap_finish_groups=groups)
    path = a.root/'length_pilot_32_max2048/series_audit.json'
    path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
