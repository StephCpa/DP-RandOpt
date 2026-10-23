"""Full-test greedy and five-sample baselines using the frozen GSM8K protocol."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from dprandopt.countdown import encode_answers
from dprandopt.gsm8k import answer_and_reward, canonical_number
from dprandopt.pilot import digest
from dprandopt.vote import vote_correct
from scripts.run_batched_controls import save, weight_hash


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--gpu-memory-utilization',type=float,default=.65)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    source=json.loads((a.data_dir/'protocol.json').read_text())
    preflight=json.loads((a.data_dir/'preflight.json').read_text())
    assert source['test_count']==1319 and preflight['context_2048_sufficient']
    from data_handlers import get_dataset_handler
    handler=get_dataset_handler('gsm8k')
    items=handler.load_data(str(a.data_dir/'test.parquet'),split='test')
    assert len(items)==1319
    config=dict(dataset=source,model_path=a.model,gpu_memory_utilization=a.gpu_memory_utilization,
        max_model_len=2048,max_num_seqs=128,max_num_batched_tokens=2048,
        max_tokens=1024,precision='bfloat16',sampling=source['self_consistency'])
    if a.prepare_only:
        print('GSM8K baseline CPU preflight passed: 1319 items, no GPU loaded',flush=True); return
    a.out_dir.mkdir(parents=True,exist_ok=True)
    if any(a.out_dir.iterdir()):
        raise RuntimeError('A new baseline session requires an empty output directory')
    (a.out_dir/'protocol.json').write_text(json.dumps(config,indent=2))
    os.environ.update(RANDOPT_BASE_WEIGHTS_DEVICE='cpu',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        RANDOPT_MAX_MODEL_LEN='2048',RANDOPT_MAX_NUM_SEQS='128',RANDOPT_MAX_NUM_BATCHED_TOKENS='2048')
    import ray
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from core import launch_engines,cleanup_engines
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    prompts=[tokenizer.apply_chat_template(x['messages'],add_generation_prompt=True,tokenize=False) for x in items]
    assert max(len(ids) for ids in tokenizer(prompts,add_special_tokens=False)['input_ids'])+1024<=2048
    (a.out_dir/'runtime.json').write_text(json.dumps(dict(start_unix=time.time(),protocol_hash=digest(config),
        prompt_hash=digest(prompts),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')),indent=2))
    engines=[];pgs=[];start=time.monotonic();result={}
    try:
        ray.init(address='local',ignore_reinit_error=True)
        engines,pgs=launch_engines(1,a.model,precision='bfloat16',gpu_memory_utilization=a.gpu_memory_utilization)
        engine=engines[0]
        pristine=ray.get(engine.collective_rpc.remote(weight_hash,args=()))
        for stage in ('greedy','self_consistency'):
            settings=dict(temperature=0.,seed=20261024,n=1) if stage=='greedy' else source['self_consistency']
            outputs=ray.get(engine.generate.remote(prompts,SamplingParams(max_tokens=1024,**settings),use_tqdm=False))
            assert len(outputs)==1319 and all(o.prompt==p for o,p in zip(outputs,prompts))
            assert ray.get(engine.collective_rpc.remote(weight_hash,args=()))==pristine
            records=[]
            for j,(o,item) in enumerate(zip(outputs,items)):
                samples=[]
                for g in sorted(o.outputs,key=lambda x:x.index):
                    ans,reward=answer_and_reward(handler,g.text,item['ground_truth'])
                    samples.append(dict(sample_index=g.index,text=g.text,token_ids=list(g.token_ids),
                        vote_answer=ans,reward=reward,finish_reason=g.finish_reason))
                assert len(samples)==settings['n']
                records.append(dict(item_index=j,prompt_token_ids=list(o.prompt_token_ids),samples=samples))
            save(a.out_dir/f'{stage}.json.gz',dict(records=records))
            answers=[[r['samples'][i]['vote_answer'] for r in records] for i in range(settings['n'])]
            codes=encode_answers(answers,[canonical_number(item['ground_truth']) for item in items])
            result[stage]=dict(mean_member_accuracy=float((codes==0).mean()),vote_accuracy=float(vote_correct(codes).mean()),
                oracle_pass=float((codes==0).any(axis=0).mean()),
                length_finish_rate=float(np.mean([s['finish_reason']=='length' for r in records for s in r['samples']])),
                vote_correct=vote_correct(codes).tolist())
            print(f'{stage} saved; elapsed_s={time.monotonic()-start:.1f}',flush=True)
        (a.out_dir/'COMPLETE.json').write_text(json.dumps(dict(results=result,elapsed_s=time.monotonic()-start),indent=2))
        print('COMPLETE GSM8K baselines',flush=True)
    finally:
        if engines: cleanup_engines(engines,pgs)
        if ray.is_initialized(): ray.shutdown()


if __name__=='__main__': main()
