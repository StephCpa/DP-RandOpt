"""Reproducible CPU audit; K=5 is frozen, other K values are descriptive only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dprandopt import designs
from dprandopt.scoretable import ScoreTable
from dprandopt.vote import vote_correct
from audit_partial_and_public import copy_answer_only


def summary(vectors):
    values = np.asarray(vectors, dtype=float).mean(axis=1)
    return dict(mean=float(values.mean()), sd=float(values.std()),
                repeat_p025_p975=np.quantile(values, [.025, .975]).tolist())


def paired(a, b, rng):
    # CI for expected algorithm performance conditional on this fixed candidate
    # pool and selection data. It does NOT estimate across-pool variability.
    diff = np.asarray(a, float).mean(axis=0) - np.asarray(b, float).mean(axis=0)
    vals = diff[rng.integers(0, len(diff), size=(5000, len(diff)))].mean(axis=1)
    return dict(mean=float(diff.mean()), conditional_test_bootstrap_ci95=np.quantile(vals, [.025,.975]).tolist())


def analyze(path, reps, seed):
    t = ScoreTable.load(str(path))
    rng = np.random.default_rng(seed)
    cols = np.arange(t.pool_size)
    def sample(table, design, k=5, columns=cols):
        return designs.select(table, design, k=k, pool_cols=columns, rng=rng, eps=8, delta=1e-5)
    pub = sample(t, 'public')
    non = sample(t, 'nonprivate')
    d1 = np.array([sample(t, 'D1').vote_correct for _ in range(reps)])
    out = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               N=t.n_candidates, n=t.pool_size, Q=t.public_reward.shape[1], T=t.test_size,
               k=5, eps=8, delta=1e-5, selection_repeats=reps,
               base=float(t.test_acc[t.base_index]), public=float(pub.vote_acc),
               nonprivate=float(non.vote_acc), D1=summary(d1),
               D1_minus_public=paired(d1, [pub.vote_correct], rng),
               nonprivate_minus_public=paired([non.vote_correct], [pub.vote_correct], rng),
               D1_minus_nonprivate=paired(d1, [non.vote_correct], rng),
               overlap_public5_private50=len(set(pub.members)&set(sample(t,'nonprivate',50).members)),
               public_indices=pub.members.tolist(), nonprivate_indices=non.members.tolist())
    out['k_curve_descriptive'] = []
    for k in [1,5,10,20,50,100]:
        dv = np.array([sample(t,'D1',k).vote_correct for _ in range(min(reps,100))])
        out['k_curve_descriptive'].append(dict(k=k, public=sample(t,'public',k).vote_acc,
            nonprivate=sample(t,'nonprivate',k).vote_acc, D1=summary(dv)['mean']))
    # Repeat both score ties and the resulting rank-dependent vote ties.
    ties = {}
    for name, table in [('original',t), ('answer_only',copy_answer_only(t))]:
        p = sample(table,'public'); n = sample(table,'nonprivate')
        d = np.array([sample(table,'D1').vote_correct for _ in range(reps)])
        scores = table.public_reward.sum(axis=1, dtype=np.float64)
        # Scores are sums of coarse decimal rewards; round accumulated float32
        # storage error only in the explicitly labelled tie sensitivity check.
        rounded = scores.round(5)
        variations = []
        for _ in range(reps):
            order = np.lexsort((rng.random(table.n_candidates),-rounded))[:5]
            variations.append(vote_correct(table.test_answer[order]))
        cutoff = np.sort(rounded)[-5]
        ties[name] = dict(public=p.vote_acc, nonprivate=n.vote_acc, D1=summary(d),
            D1_minus_public=paired(d,[p.vote_correct],rng),
            rounded_cutoff=float(cutoff), cutoff_ties=int((rounded==cutoff).sum()),
            random_ties=summary(variations))
    out['reward_and_ties'] = ties
    # n=0/Q=0 includes an honest data-free random-K comparator, not only base.
    out['data_free_random5'] = summary([sample(t,'random').vote_correct for _ in range(reps)])
    public_curve=[]
    for q in [5,10,25,50]:
        vectors=[]
        for _ in range(reps):
            subset=rng.choice(t.public_reward.shape[1],q,replace=False)
            scores=t.public_reward[:,subset].sum(axis=1,dtype=np.float64).round(5)
            order=np.lexsort((rng.random(t.n_candidates),-scores))[:5]
            vectors.append(vote_correct(t.test_answer[order]))
        public_curve.append(dict(q=q, **summary(vectors)))
    out['public_size_random_subsets_and_ties']=public_curve
    private_curve=[]
    for n in [200,500,1000]:
        dv=[]; nv=[]
        for _ in range(reps):
            c=rng.choice(t.pool_size,n,replace=False)
            dv.append(sample(t,'D1',columns=c).vote_correct)
            nv.append(sample(t,'nonprivate',columns=c).vote_correct)
        private_curve.append(dict(n=n,D1=summary(dv),nonprivate=summary(nv)))
    out['private_size_random_subsets']=private_curve
    release=[]; raw=[]; fallback=[]
    for _ in range(reps):
        s=sample(t,'D8'); released=s.info['released']
        release.append(released); raw.append(s.vote_correct)
        fallback.append(s.vote_correct if released else pub.vote_correct)
    out['D8']=dict(release_rate=float(np.mean(release)),raw=summary(raw),public_fallback=summary(fallback))
    if t.output_lengths is not None:
        lengths=t.output_lengths
        assert lengths.shape==(t.n_candidates,t.pool_size+t.public_reward.shape[1]+t.test_size)
        assert np.all((lengths>0)&(lengths<=512))
        base=lengths[t.base_index]
        out['cap512']=dict(total=int(lengths.size),at_cap=int((lengths==512).sum()),
            overall=float((lengths==512).mean()),base_all=float((base==512).mean()),
            base_test=float((base[-t.test_size:]==512).mean()),
            candidate_median=float(np.median((lengths[1:]==512).mean(axis=1))))
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('tables',nargs='+',type=Path)
    p.add_argument('--partial',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reps',type=int,default=400)
    a=p.parse_args()
    full=ScoreTable.load(str(a.tables[0])); partial=ScoreTable.load(str(a.partial))
    prefix={key:bool(np.array_equal(getattr(full,key)[:partial.n_candidates],getattr(partial,key)))
            for key in ['pool_reward','public_reward','test_answer','seed','sigma','output_lengths']}
    assert all(prefix.values()),prefix
    rows=[]
    for i,path in enumerate(a.tables):
        print('Auditing',path.name,flush=True)
        rows.append(analyze(path,a.reps,20261022+i))
    a.out.mkdir(parents=True,exist_ok=True)
    report=dict(prefix_exact_match=prefix,results=rows,
        scope='CPU only; fixed K=5, eps=8, delta=1e-5; conditional test bootstrap, not across-pool uncertainty; public curve limited to Q<=50.')
    (a.out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=rows[0]
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for method in ['public','nonprivate','D1']:
        axes[0].plot([v['k'] for v in r['k_curve_descriptive']], [v[method] for v in r['k_curve_descriptive']],marker='o',label=method)
    axes[0].set(xlabel='K (descriptive; fixed analysis K=5)',ylabel='Test vote accuracy')
    c=r['public_size_random_subsets_and_ties']
    axes[1].plot([v['q'] for v in c],[v['mean'] for v in c],marker='o',label='Public-only')
    axes[1].fill_between([v['q'] for v in c],[v['repeat_p025_p975'][0] for v in c],[v['repeat_p025_p975'][1] for v in c],alpha=.15)
    for v in r['private_size_random_subsets']:
        axes[1].axhline(v['D1']['mean'],ls='--',label=f'D1 n={v["n"]}')
    axes[1].set(xlabel='Public selection examples Q',ylabel='Test vote accuracy')
    v=r['private_size_random_subsets']
    for method in ['D1','nonprivate']:
        axes[2].plot([x['n'] for x in v],[x[method]['mean'] for x in v],marker='o',label=method)
    axes[2].axhline(r['public'],ls='--',label='Public Q=50')
    axes[2].set(xlabel='Private/pool selection examples n',ylabel='Test vote accuracy')
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Countdown, 1001 candidates, 512-token cap; D1 eps=8, delta=1e-5\nShading: public subset/tie variation, not test confidence interval')
    fig.tight_layout(); fig.savefig(a.out/'diagnostics.png',dpi=180); plt.close(fig)
    print('Wrote',a.out,flush=True)


if __name__=='__main__':
    main()
