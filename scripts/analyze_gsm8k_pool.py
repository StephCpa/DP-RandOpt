"""CPU audit of the frozen full GSM8K pool and sampled baselines."""
import gzip
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from dprandopt import designs
from dprandopt.countdown import encode_answers
from dprandopt.gsm8k import canonical_number
from dprandopt.scoretable import ScoreTable
from dprandopt.vote import vote_correct
from scripts.audit_vote_selection import analyze, ensemble_metrics
from scripts.audit_length_series import paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparisons-only',action='store_true')
    args = parser.parse_args()
    root = Path('results/gpu-smoke')
    if args.comparisons_only:
        folder = root/'gsm8k_pool_audit_20261025'
        saved = json.loads((folder/'analysis.json').read_text())
        with np.load(folder/'comparison_vectors.npz') as z:
            for name,expected in saved['comparisons'].items():
                left,right = name.split('_minus_')
                assert paired(z[left+'_vote'],z[right+'_vote'])==expected
        print('All seven paired comparisons reproduced from compact vectors')
        return
    source = root/'gsm8k_n1001_seed20261024_items20261023_max1024.npz'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == 'd53c133154bf0abe7aaf8b3ef87b40d7431c259f25608d0d6e1a25f88acfd246'
    t = ScoreTable.load(str(source))
    assert t.pool_reward.shape == (1001,1000) and t.public_reward.shape == (1001,50)
    assert t.test_answer.shape == (1001,1319) and t.output_lengths.shape == (1001,2369)
    assert np.isin(t.pool_reward,[0,1]).all() and np.isin(t.public_reward,[0,1]).all()
    assert ((t.output_lengths>0)&(t.output_lengths<=1024)).all()
    rng = np.random.default_rng(20261024)
    np.testing.assert_array_equal(t.seed, np.r_[-1,rng.choice(2**31-1,size=1000,replace=False)])
    np.testing.assert_array_equal(t.sigma, np.r_[0.,rng.choice([.0005,.001,.002],size=1000)])
    folder = root/'gsm8k_pool_audit_20261025'
    folder.mkdir(exist_ok=True)
    print('Validated frozen table; computing primary comparisons',flush=True)
    result, vectors = analyze(t,400,2000,return_vectors=True)
    # The shared helper calls pass@5 checked_pass; GSM8K uses labels: rename it.
    def rename(x):
        if isinstance(x,dict):
            return {k.replace('checked_pass','oracle_pass'):rename(v) for k,v in x.items()}
        return x
    result = rename(result)
    vectors = {name:{k.replace('checked_pass','oracle_pass'):v for k,v in vals.items()} for name,vals in vectors.items()}
    basefolder = root/'gsm8k_baselines_20261023'
    complete = json.loads((basefolder/'COMPLETE.json').read_text())
    gold = [canonical_number(json.loads(line)['reward_model']['ground_truth']) for line in Path('results/cache/gsm8k_20261023/test.jsonl').read_text().splitlines()]
    for stage,n in [('greedy',1),('self_consistency',5)]:
        with gzip.open(basefolder/f'{stage}.json.gz','rt') as f:
            rows = json.load(f)['records']
        assert len(rows)==1319 and [r['item_index'] for r in rows]==list(range(1319))
        assert all(len(r['samples'])==n for r in rows)
        answers = [[r['samples'][i]['vote_answer'] for r in rows] for i in range(n)]
        codes = encode_answers(answers,gold)
        for j,r in enumerate(rows):
            for i,s in enumerate(r['samples']):
                assert s['reward']==int(codes[i,j]==0)
        v = ensemble_metrics(codes)
        for key,savedkey in [('member','mean_member_accuracy'),('vote','vote_accuracy'),('checked_pass','oracle_pass')]:
            assert np.isclose(v[key].mean(),complete['results'][stage][savedkey])
        np.testing.assert_array_equal(v['vote'],complete['results'][stage]['vote_correct'])
        vectors[stage] = {k.replace('checked_pass','oracle_pass'):a for k,a in v.items()}
    result['comparisons'] = {f'{left}_minus_{right}':paired(vectors[left]['vote'],vectors[right]['vote'])
        for left,right in [('D1','random5'),('D1','self_consistency'),('D1','public'),('D1','nonprivate'),
                           ('public','self_consistency'),('random5','self_consistency'),('nonprivate','self_consistency')]}
    result['standalone_baselines'] = {name:{k:float(v.mean()) for k,v in vectors[name].items()} for name in ['greedy','self_consistency']}
    result['base_batch_correctness_flip_rate'] = float(np.mean((t.test_answer[0]==0)!=vectors['greedy']['vote']))
    result['length_cap_rate'] = {'base':float((t.output_lengths[0]==1024).mean()),'perturbations':float((t.output_lengths[1:]==1024).mean())}
    print('Primary analysis done; n and public tie sensitivity',flush=True)
    rng = np.random.default_rng(20261025)
    result['private_sizes'] = []
    for n in [200,500,1000]:
        vv = []
        for _ in range(400):
            cols = rng.choice(t.pool_size,n,replace=False) if n<1000 else np.arange(1000)
            vv.append(designs.select(t,'D1',k=5,pool_cols=cols,rng=rng,eps=8,delta=1e-5).vote_correct)
        v = np.mean(vv,axis=0)
        result['private_sizes'].append(dict(n=n,vote=float(v.mean()),versus_public=paired(v,vectors['public']['vote']),versus_sc=paired(v,vectors['self_consistency']['vote'])))
    scores = t.public_reward.sum(axis=1)
    tie = np.array([vote_correct(t.test_answer[np.lexsort((rng.random(t.n_candidates),-scores))[:5]]) for _ in range(400)])
    result['public_ties'] = dict(cutoff_score=float(np.sort(scores)[-5]),cutoff_tie_count=int((scores==np.sort(scores)[-5]).sum()),
        randomized_vote=float(tie.mean()),D1_minus_randomized=paired(vectors['D1']['vote'],tie.mean(axis=0)))
    result['scope'] = 'One frozen pool, one self-consistency sample seed; 5000 paired test-item bootstrap draws, conditional on these and selection data. Exploratory unadjusted intervals; not across-pool uncertainty. GSM8K pass is oracle-only. Secondary n varies both the private subsample and DP randomness.'
    result['source_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(str(source)+'.meta.json'),basefolder/'COMPLETE.json']}
    b = t.test_answer[0]==0
    g = vectors['greedy']['vote'].astype(bool)
    secondary = dict(base_correctness_pairs=dict(table_only_correct=int((b&~g).sum()),standalone_only_correct=int((~b&g).sum()),
        both_correct=int((b&g).sum()),both_wrong=int((~b&~g).sum())),descriptive_K_curves={name:{str(k):designs.select(t,name,k=k,
        pool_cols=np.arange(1000),rng=np.random.default_rng(20261025)).vote_acc for k in [1,3,5,10,20,50]} for name in ['public','nonprivate']},
        scope='K is descriptive only; primary K remains 5. Base calls differ in batch workload (1319 vs 2369 prompts), not a controlled repeatability test.')
    (folder/'secondary.json').write_text(json.dumps(secondary,indent=2))
    np.savez_compressed(folder/'comparison_vectors.npz',**{f'{name}_{k}':v for name,vs in vectors.items() for k,v in vs.items()})
    (folder/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
