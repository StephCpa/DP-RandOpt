"""CPU decomposition of random/selected ensembles: members, vote and checked pass@K."""
import argparse
import json
from math import comb
from pathlib import Path
import numpy as np
from dprandopt import designs
from dprandopt.scoretable import ScoreTable
from dprandopt.vote import vote_correct
from scripts.audit_length_series import paired


def ensemble_metrics(answers):
    a = np.asarray(answers)
    return dict(member=(a == 0).mean(axis=0), vote=vote_correct(a), checked_pass=(a == 0).any(axis=0))


def exact_random_pass(correct, k):
    """Uniform subsets without replacement, not independent candidate draws."""
    correct = np.asarray(correct, bool)
    n = len(correct)
    if not 1 <= k <= n:
        raise ValueError('Invalid K')
    denom = comb(n, k)
    return np.array([1 - comb(n-int(s), k)/denom if n-int(s) >= k else 1.
                     for s in correct.sum(axis=0)])


def analyze(t, reps, random_reps):
    rng = np.random.default_rng(20261023)
    candidates = np.delete(np.arange(t.n_candidates), t.base_index)
    cols = np.arange(t.pool_size)
    random_vectors = {key: [] for key in ('member', 'vote', 'checked_pass')}
    for _ in range(random_reps):
        idx = rng.choice(candidates, 5, replace=False)
        for key, values in ensemble_metrics(t.test_answer[idx]).items():
            random_vectors[key].append(values)
    rv = {k: np.mean(v, axis=0) for k,v in random_vectors.items()}
    rv['checked_pass'] = exact_random_pass(t.test_answer[candidates] == 0, 5)
    rv['member'] = (t.test_answer[candidates] == 0).mean(axis=0)
    out = dict(N=t.n_candidates, base=float(t.test_acc[t.base_index]), K=5, n=t.pool_size,
               Q=t.public_reward.shape[1], epsilon=8, delta=1e-5,
               random_scope='Five distinct perturbations, excluding base; uniformly ordered.',
               random_draws=random_reps, selection_draws=reps,
               random5={k:float(v.mean()) for k,v in rv.items()})
    out['random5']['vote_mc_standard_error'] = float(np.std(np.mean(random_vectors['vote'], axis=1), ddof=1)/np.sqrt(random_reps))
    out['random5']['checked_pass_computation'] = 'Exact finite-population expectation per test item'
    sets, vectors = {}, {}
    for name in ('public', 'nonprivate', 'D1'):
        draws = []
        for _ in range(reps if name == 'D1' else 1):
            s = designs.select(t, name, k=5, pool_cols=cols, rng=rng, eps=8, delta=1e-5)
            draws.append(ensemble_metrics(s.answers))
            if name != 'D1':
                sets[name] = s.members.tolist()
        v = {k: np.mean([d[k] for d in draws], axis=0) for k in rv}
        vectors[name] = v
        out[name] = {k:float(x.mean()) for k,x in v.items()}
        out[name]['vote_minus_random5'] = paired(v['vote'], rv['vote'])
        out[name]['pass_minus_vote'] = paired(v['checked_pass'], v['vote'])
        if name in sets:
            out[name]['members_in_vote_order'] = sets[name]
            out[name]['individual_member_accuracy'] = t.test_acc[sets[name]].tolist()
    out['public_private_overlap'] = dict(indices=sorted(set(sets['public']) & set(sets['nonprivate'])),
                                         count=len(set(sets['public']) & set(sets['nonprivate'])))
    out['nonprivate_minus_public'] = {key:paired(vectors['nonprivate'][key], vectors['public'][key]) for key in rv}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('results/gpu-smoke'))
    p.add_argument('--reps', type=int, default=400)
    p.add_argument('--random-reps', type=int, default=2000)
    a = p.parse_args()
    t = ScoreTable.load(str(a.root/'lzn_route1_n1001_max512.npz'))
    subset = ScoreTable(pool_reward=t.pool_reward[:32], public_reward=t.public_reward[:32],
                        test_answer=t.test_answer[:32], reward_max=t.reward_max)
    tables = {'subset_512':subset, 'full_512':t}
    for cap in (1024, 2048):
        tables[f'subset_{cap}'] = ScoreTable.load(str(a.root/f'length_pilot_32_max{cap}/table.npz'))
    out = dict(scope='Synthetic Countdown. Checked pass uses prompt-visible target and formula validation, no private labels. Intervals conditional on fixed candidate pool; not fresh-pool uncertainty.', results={})
    for key, table in tables.items():
        print(key, flush=True)
        out['results'][key] = analyze(table, a.reps, a.random_reps)
    folder = a.root/'selection_decomposition'
    folder.mkdir(exist_ok=True)
    (folder/'audit.json').write_text(json.dumps(out, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
