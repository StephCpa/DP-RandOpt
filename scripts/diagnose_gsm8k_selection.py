"""Frozen-pool CPU diagnostics; never selects a new protocol from test scores."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from dprandopt import designs
from dprandopt.scoretable import ScoreTable
from dprandopt.vote import vote_correct


def correlation(x,y):
    return dict(pearson=float(np.corrcoef(x,y)[0,1]),spearman=float(spearmanr(x,y).statistic))


def spread(c):
    """Shared-item variance correction under iid sampling of test items.

    Project out the across-candidate item mean. Unlike marginal binomial
    variances, this accounts for correlated correctness on the same items.
    The latent variance estimate is signed; a negative value is not hidden.
    """
    c=np.asarray(c,float); m,t=c.shape; p=c.mean(axis=1)
    observed=float(p.var(ddof=1))
    marginal=float(np.mean(p*(1-p)/(t-1)))
    centered=c-p[:,None]
    shared=float(np.var(centered,axis=0,ddof=1).mean()/(t-1))
    return dict(candidates=m,observed_variance=observed,observed_sd=float(np.sqrt(observed)),
        marginal_binomial_variance=marginal,marginal_binomial_se=float(np.sqrt(marginal)),
        naive_subtracted_variance=observed-marginal,
        shared_item_sampling_variance=shared,shared_item_noise_sd=float(np.sqrt(shared)),
        estimated_latent_variance=observed-shared,estimated_latent_sd=float(np.sqrt(max(0,observed-shared))))


def metrics(a):
    a=np.asarray(a); ii,jj=np.triu_indices(len(a),1)
    different=a[ii]!=a[jj]; valid=(a[ii]>=0)&(a[jj]>=0)
    correctness=a==0
    return dict(member=float(correctness.mean()),vote=float(vote_correct(a).mean()),
        oracle_pass=float(correctness.any(axis=0).mean()),disagreement=float(different.mean()),
        disagreement_both_valid=float(different[valid].mean()) if valid.any() else None,
        correctness_disagreement=float((correctness[ii]!=correctness[jj]).mean()),
        joint_error=float((~correctness[ii]&~correctness[jj]).mean()))


def main():
    root=Path('results/gpu-smoke'); out=root/'gsm8k_selection_diagnostics_20261025';out.mkdir(exist_ok=True)
    source=root/'gsm8k_n1001_seed20261024_items20261023_max1024.npz'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='d53c133154bf0abe7aaf8b3ef87b40d7431c259f25608d0d6e1a25f88acfd246'
    t=ScoreTable.load(str(source)); pool=t.pool_reward.mean(axis=1); test=t.test_acc
    rng=np.random.default_rng(20261023); candidates=np.arange(1,1001); sigmas=[0.,.0005,.001,.002]
    sets={'random5':np.array([rng.choice(candidates,5,replace=False) for _ in range(2000)])}
    for method in ['public','nonprivate','D1']:
        sets[method]=np.array([designs.select(t,method,k=5,pool_cols=np.arange(1000),rng=rng,eps=8,delta=1e-5).members
            for _ in range(400 if method=='D1' else 1)])
    result=dict(scope='One fixed pool and fixed test items; descriptive post-hoc diagnostics, not protocol tuning or causal identification.',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),groups={},ensembles={},spread={},correlations={})
    per_draw={}
    for name,ss in sets.items():
        print('ensemble',name,flush=True)
        rows=[metrics(t.test_answer[idx]) for idx in ss]
        per_draw[name]={k:np.array([r[k] for r in rows]) for k in rows[0]}
        result['ensembles'][name]=dict(draws=len(ss),**{k:float(v.mean()) for k,v in per_draw[name].items()},
            sigma_fraction={str(s):float((t.sigma[ss]==s).mean()) for s in sigmas})
    rng=np.random.default_rng(20261025)
    correct=t.test_answer==0
    for s in sigmas[1:]:
        idx=np.flatnonzero(t.sigma==s); key=str(s)
        ss=np.array([rng.choice(idx,5,replace=False) for _ in range(2000)])
        sets['sigma_'+key]=ss
        print('sigma',s,flush=True)
        rows=[metrics(t.test_answer[m]) for m in ss]
        per_draw['sigma_'+key]={k:np.array([r[k] for r in rows]) for k in rows[0]}
        result['groups'][key]=dict(count=len(idx),member_accuracy=float(test[idx].mean()),
            pool_accuracy=float(pool[idx].mean()),public_accuracy=float(t.public_reward[idx].mean()),
            correctness_flip_vs_base=float((correct[idx]!=correct[0]).mean()),
            flip_quantiles=np.quantile((correct[idx]!=correct[0]).mean(axis=1),[.05,.5,.95]).tolist(),
            fraction_observed_above_base=float((test[idx]>test[0]).mean()),
            random5={k:float(v.mean()) for k,v in per_draw['sigma_'+key].items()})
        result['spread'][key]=spread(correct[idx]); result['correlations'][key]=correlation(pool[idx],test[idx])
    result['spread']['all_perturbations']=spread(correct[1:])
    result['correlations']['all_perturbations']=correlation(pool[1:],test[1:])
    result['candidate_accuracy']=dict(base=float(test[0]),quantiles=np.quantile(test[1:],[0,.05,.25,.5,.75,.95,1]).tolist(),
        fraction_above_base=float((test[1:]>test[0]).mean()),fraction_at_least_5pp_above_base=float((test[1:]>=test[0]+.05).mean()))
    # A simple descriptive residual correlation controls each sigma group's mean.
    px=pool[1:].copy();ty=test[1:].copy()
    for s in sigmas[1:]:
        m=t.sigma[1:]==s;px[m]-=px[m].mean();ty[m]-=ty[m].mean()
    result['correlations']['within_sigma_centered']=correlation(px,ty)
    with gzip.open(root/'gsm8k_baselines_20261023/self_consistency.json.gz','rt') as f: rows=json.load(f)['records']
    # Independently encode each item's equal wrong answers; correctness from saved rewards.
    sc=np.full((5,1319),-1,dtype=np.int16)
    for j,row in enumerate(rows):
        wrong={}
        for i,sample in enumerate(row['samples']):
            ans=sample['vote_answer']
            if sample['reward']==1: sc[i,j]=0
            elif ans: sc[i,j]=wrong.setdefault(ans,len(wrong)+1)
    result['ensembles']['self_consistency']=metrics(sc)
    for name,dd in per_draw.items():
        x=dd['disagreement'];y=dd['vote'];member=dd['member']
        if len(x)<2: continue
        X=np.column_stack([np.ones(len(x)),member])
        residual=lambda v:v-X@np.linalg.lstsq(X,v,rcond=None)[0]
        result['ensembles'].setdefault(name,{})['diversity_vote_association']=dict(raw=correlation(x,y),
            controlling_mean_member_accuracy=correlation(residual(x),residual(y)))
    result['base_splits']={name:dict(correct=int(c.sum()),n=len(c),accuracy=float(np.mean(c,dtype=np.float64))) for name,c in
        [('pool',t.pool_reward[0]),('public',t.public_reward[0]),('test',correct[0])]}
    result['spread_assumptions']='Latent spread correction assumes iid test-item sampling; candidates share items. Marginal binomial subtraction ignores shared-item covariance and is reported only as a naive reference. No decoder rerun component is identifiable. A clipped SD is a descriptive estimate, not proof of zero spread.'
    result['D4']=dict(feasible=False,reason='Final table stores pool/public rewards only. Partial answers have shape (1001,1319), i.e. test only. No full-pool response bundles found on server. Wrong-answer identities cannot be recovered from binary correctness. Exact private vote utilities and pool-test ensemble reliability are unavailable; do not substitute oracle pass or fabricate wrong answers.')
    tail={}
    for s in sigmas[1:]:
        idx=np.flatnonzero(t.sigma==s);top=idx[pool[idx]>=np.median(pool[idx])]
        tail[str(s)]=dict(test_quantiles=np.quantile(test[idx],[0,.01,.05,.5,.95,.99,1]).tolist(),
            test_iqr=float(np.quantile(test[idx],.75)-np.quantile(test[idx],.25)),
            pool_upper_half_count=len(top),pool_upper_half_correlation=correlation(pool[top],test[top]))
    tail['lowest_5']=[dict(index=int(i),sigma=float(t.sigma[i]),pool=float(pool[i]),test=float(test[i])) for i in np.argsort(test)[:5]]
    (out/'tail_sensitivity.json').write_text(json.dumps(tail,indent=2))
    np.savez_compressed(out/'draws.npz',**{name+'_members':ss for name,ss in sets.items()},
        **{name+'_'+key:v for name,dd in per_draw.items() for key,v in dd.items()},self_consistency_codes=sc)
    (out/'diagnostics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
