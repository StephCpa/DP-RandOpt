"""Validate archived controls and compare fixed sampled outputs on paired items."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from dprandopt.countdown import encode_answers
from dprandopt.pilot import prepare, read_candidate, digest
from dprandopt.scoretable import ScoreTable
from scripts.audit_vote_selection import analyze, ensemble_metrics
from scripts.audit_length_series import paired
from scripts.run_batched_controls import compare


def load(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        return json.load(f)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('results/gpu-smoke'))
    p.add_argument('--comparisons-only',action='store_true',help='Recompute paired comparisons from the published compact vectors')
    a=p.parse_args()
    folder=a.root/'batched_controls_20261023'
    if a.comparisons_only:
        z=np.load(folder/'comparison_vectors.npz')
        comparisons={name:{key:paired(z[f'self_consistency_{key}'],z[f'{name}_{key}'])
                     for key in ('member','vote','checked_pass')} for name in ('random5','public','nonprivate','D1')}
        saved=json.loads((folder/'analysis.json').read_text())['comparisons_sc_minus_baseline']
        assert comparisons==saved
        print(json.dumps(comparisons,indent=2)); return
    protocol=json.loads((folder/'protocol.json').read_text())
    complete=json.loads((folder/'COMPLETE.json').read_text())
    runtime=json.loads((folder/'runtime.json').read_text())
    m,items=prepare(a.root/'lzn_route1_n1001_max512.npz',32,1024)
    assert protocol['production_manifest']==m
    assert runtime['protocol_hash']==digest(protocol)
    assert runtime['prompt_hash']==(a.root/'length_pilot_32_max1024/prompts.sha256').read_text().strip()
    gold=[str(x['target']) for x in items]
    out=dict(scope='Conditional paired test-item comparisons; one fixed five-sample draw per item, one fixed perturbation pool, no across-session/pool/sampling-seed uncertainty.',
        repeatability={}, archived_pilot_comparison={}, weight_checks=True, responses=14900)
    pristine=complete['weights'][0]['after_reset']
    for i in range(4):
        bundles=[load(folder/f'candidate_{i}_repeat_{r}.json.gz') for r in range(2)]
        rows=[]
        for r,b in enumerate(bundles):
            assert b['weights']==complete['weights'][r*4+i]
            assert b['weights']['after_reset']==pristine
            assert b['weights']['before_generate']==b['weights']['after_generate']
            assert len(b['records'])==1550
            for j,record in enumerate(b['records']):
                assert record['item_index']==j and len(record['samples'])==1
                assert 0<len(record['samples'][0]['token_ids'])<=1024
            rows.append([record['samples'][0] for record in b['records']])
        assert bundles[0]['weights']['before_generate']==bundles[1]['weights']['before_generate']
        assert all(x['prompt_token_ids']==y['prompt_token_ids'] for x,y in zip(bundles[0]['records'],bundles[1]['records']))
        out['repeatability'][str(i)]=compare(rows[0],rows[1],gold)
        assert out['repeatability'][str(i)]==complete['repeatability'][str(i)]
        old=read_candidate(a.root/f'length_pilot_32_max1024/candidate_{i:04d}.json.gz',m,i)['records']
        out['archived_pilot_comparison'][i]={}
        for name,start,end in [('all',0,1550),('pool',0,1000),('public',1000,1050),('test',1050,1550)]:
            pairs=list(zip(old[start:end],rows[0][start:end],gold[start:end]))
            out['archived_pilot_comparison'][i][name]=dict(
                reward_flip_rate=float(np.mean([abs(x['reward']-y['reward'])>1e-6 for x,y,g in pairs])),
                correctness_flip_rate=float(np.mean([(x['vote_answer']==g)!=(y['vote_answer']==g) for x,y,g in pairs])),
                text_flip_rate=float(np.mean([x['text']!=y['text'] for x,y,g in pairs])))
    rec=load(folder/'base_self_consistency.json.gz')['records']
    assert len(rec)==500
    for j,r in enumerate(rec):
        assert r['item_index']==j and [s['sample_index'] for s in r['samples']]==list(range(5))
        assert all(0<len(s['token_ids'])<=1024 for s in r['samples'])
        assert r['prompt_token_ids']==bundles[0]['records'][1050+j]['prompt_token_ids']
    codes=encode_answers([[r['samples'][s]['vote_answer'] for r in rec] for s in range(5)],gold[1050:])
    sc=ensemble_metrics(codes)
    for key,saved in [('member','member_accuracy'),('vote','vote_accuracy'),('checked_pass','checked_pass5')]:
        assert np.isclose(sc[key].mean(),complete['self_consistency'][saved])
    np.testing.assert_array_equal(sc['vote'],complete['self_consistency']['vote_correct'])
    np.testing.assert_array_equal(sc['checked_pass'],complete['self_consistency']['pass_correct'])
    table=ScoreTable.load(str(a.root/'length_pilot_32_max1024/table.npz'))
    baseline,vectors=analyze(table,400,2000,return_vectors=True)
    out['self_consistency']={k:float(v.mean()) for k,v in sc.items()}
    out['baseline']=baseline
    out['comparisons_sc_minus_baseline']={name:{k:paired(sc[k],v[k]) for k in sc} for name,v in vectors.items()}
    out['self_consistency']['individual_sample_accuracies']=(codes==0).mean(axis=1).tolist()
    out['self_consistency']['length_finish_rate']=complete['self_consistency']['length_finish_rate']
    arrays={'self_consistency_codes':codes}
    for name,v in {'self_consistency':sc,**vectors}.items():
        arrays.update({f'{name}_{k}':x for k,x in v.items()})
    np.savez_compressed(folder/'comparison_vectors.npz',**arrays)
    out['source_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(folder.glob('*.json.gz'))}
    (folder/'analysis.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out['comparisons_sc_minus_baseline'],indent=2))
    print('Archived-session comparison:',json.dumps(out['archived_pilot_comparison']))


if __name__=='__main__': main()
