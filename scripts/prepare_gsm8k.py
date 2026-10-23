"""Prepare official GSM8K for the RandOpt handler; no inference or test tuning."""
import argparse
import json
from pathlib import Path
import urllib.request
import numpy as np
from dprandopt.gsm8k import canonical_number

REVISION='3101c7d5072418e28b9008a6636bde82a006892c'
SOURCE='https://raw.githubusercontent.com/openai/grade-school-math/'+REVISION
SUFFIX='\n\nSolve the problem step by step. End your response with "#### <answer>", replacing <answer> with the final numeric answer.'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--parquet',action='store_true')
    a=p.parse_args()
    raw=a.out_dir/'raw'; raw.mkdir(parents=True,exist_ok=True)
    datasets={}
    for split in ('train','test'):
        path=raw/f'{split}.jsonl'
        if not path.exists():
            with urllib.request.urlopen(SOURCE+f'/grade_school_math/data/{split}.jsonl',timeout=30) as response:
                path.write_bytes(response.read())
        records=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        expected=7473 if split=='train' else 1319
        assert len(records)==expected
        prepared=[]
        for i,r in enumerate(records):
            assert '####' in r['answer']
            gold=canonical_number(r['answer'].rsplit('####',1)[1])
            assert gold
            prepared.append(dict(source_id=i,prompt=[dict(role='user',content=r['question']+SUFFIX)],
                                 reward_model=dict(ground_truth=gold)))
        (a.out_dir/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in prepared),encoding='utf-8')
        if a.parquet:
            import pandas as pd
            pd.DataFrame(prepared).to_parquet(a.out_dir/f'{split}.parquet',index=False)
        datasets[split]=records
    train_questions={r['question'].strip() for r in datasets['train']}
    test_questions={r['question'].strip() for r in datasets['test']}
    overlap=train_questions & test_questions
    assert not overlap, 'Train/test exact-question overlap requires split review'
    order=np.random.default_rng(20261023).permutation(len(datasets['train']))
    protocol=dict(source='https://github.com/openai/grade-school-math',source_revision=REVISION,
        train_count=7473,test_count=1319,exact_question_overlap=0,item_seed=20261023,
        pool_ids=order[:1000].tolist(),public_ids=order[1000:1050].tolist(),
        test_ids=list(range(1319)),prompt_suffix=SUFFIX,
        primary_scoring='canonical_voting_answer_v1: strict-first extractor, flexible only if strict absent; shared numeric normalization',
        max_tokens=1024,model='Qwen/Qwen2.5-3B-Instruct',candidate_seeds=[20261024,20261025,20261026],
        planned_candidates_per_pool=1001,sigmas=[.0005,.001,.002],k=5,eps=8,delta=1e-5,
        self_consistency=dict(n=5,temperature=.7,top_p=1.,top_k=-1,seed=20261023),
        execution_status='Data preparation only; no GSM8K GPU generation started')
    (a.out_dir/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print(json.dumps({k:protocol[k] for k in ('source_revision','train_count','test_count','exact_question_overlap','execution_status')}))


if __name__=='__main__': main()
