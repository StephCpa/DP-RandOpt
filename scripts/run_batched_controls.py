"""Production-shaped repeatability and base sampling controls, no parameter tuning.

Greedy: original 1550 prompts per call, candidates 0..3 twice, cap 1024,
context 2048, max_num_seqs 128, max_num_batched_tokens 2048, eager bf16.
Self-consistency: five samples per test prompt at a frozen temperature 0.7.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from dprandopt.pilot import prepare, digest
from dprandopt.countdown import encode_answers
from dprandopt.vote import vote_correct


def weight_hash(worker, stored=False):
    """Full tensor-byte checksum, not a floating-point summary or sample."""
    import hashlib
    import torch
    torch.cuda.synchronize()
    parameters = worker._base_weights.items() if stored else worker.model_runner.model.named_parameters()
    h = hashlib.sha256()
    for name, value in sorted(parameters):
        h.update((name+str(value.dtype)+str(tuple(value.shape))).encode())
        h.update(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp.gz')
    with gzip.open(tmp, 'wt', encoding='utf-8') as f:
        json.dump(value, f)
    os.replace(tmp, path)


def compare(rows_a, rows_b, gold):
    out = {}
    for name, start, end in [('all',0,1550),('pool',0,1000),('public',1000,1050),('test',1050,1550)]:
        a,b = rows_a[start:end],rows_b[start:end]
        out[name] = dict(count=len(a),
            reward_flip_rate=float(np.mean([abs(x['reward']-y['reward'])>1e-6 for x,y in zip(a,b)])),
            correctness_flip_rate=float(np.mean([(x['vote_answer']==g)!=(y['vote_answer']==g) for x,y,g in zip(a,b,gold[start:end])])),
            vote_answer_flip_rate=float(np.mean([x['vote_answer']!=y['vote_answer'] for x,y in zip(a,b)])),
            token_sequence_flip_rate=float(np.mean([x['token_ids']!=y['token_ids'] for x,y in zip(a,b)])))
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',required=True)
    p.add_argument('--production-pilot',required=True,type=Path)
    p.add_argument('--out-dir',required=True,type=Path)
    p.add_argument('--gpu-memory-utilization',type=float,default=.65)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    m,items=prepare(a.reference,32,1024)
    assert m==json.loads((a.production_pilot/'manifest.json').read_text())
    settings=dict(candidate_indices=[0,1,2,3],repeats=2,max_tokens=1024,max_model_len=2048,
        max_num_seqs=128,max_num_batched_tokens=2048,precision='bfloat16',enforce_eager=True,
        prefix_cache=False,gpu_memory_utilization=a.gpu_memory_utilization,
        self_consistency=dict(n=5,temperature=.7,top_p=1.,top_k=-1,seed=20261023),
        production_manifest=m,scope='Repeat flips estimate noise for these four candidates only, not a universal design noise floor.')
    a.out_dir.mkdir(parents=True,exist_ok=True)
    frozen=a.out_dir/'protocol.json'
    if frozen.exists():
        assert json.loads(frozen.read_text())==settings
    else:
        frozen.write_text(json.dumps(settings,indent=2))
    if a.prepare_only:
        print('CPU preflight passed',flush=True); return
    if list(a.out_dir.glob('*.json.gz')) or (a.out_dir/'runtime.json').exists():
        raise RuntimeError('Use a new output folder for a new control session; existing records are immutable')
    os.environ.update(RANDOPT_BASE_WEIGHTS_DEVICE='cpu',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        RANDOPT_MAX_MODEL_LEN='2048',RANDOPT_MAX_NUM_SEQS='128',RANDOPT_MAX_NUM_BATCHED_TOKENS='2048')
    import ray
    import torch
    import vllm
    import transformers
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from core import cleanup_engines,launch_engines
    from data_handlers import get_dataset_handler
    tokenizer=AutoTokenizer.from_pretrained(m['model'],local_files_only=True)
    prompts=[tokenizer.apply_chat_template(x['messages'],add_generation_prompt=True,tokenize=False) for x in items]
    assert digest(prompts)==(a.production_pilot/'prompts.sha256').read_text().strip()
    handler=get_dataset_handler('countdown')
    runtime=dict(torch=torch.__version__,vllm=vllm.__version__,transformers=transformers.__version__,
        start_unix=time.time(),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        prompt_hash=digest(prompts),protocol_hash=digest(settings))
    (a.out_dir/'runtime.json').write_text(json.dumps(runtime,indent=2))
    engines=[];pgs=[];start=time.monotonic(); history={}; hashes=[]
    def records(outputs,subset):
        result=[]
        for j,(o,item) in enumerate(zip(outputs,subset)):
            samples=[]
            for g in sorted(o.outputs, key=lambda output: output.index):
                ans,valid,_=handler.extract_answer_for_voting(g.text,numbers=item['numbers'])
                samples.append(dict(sample_index=g.index,text=g.text,token_ids=list(g.token_ids),
                    reward=float(handler.compute_reward(g.text,item['ground_truth'])),
                    vote_answer=ans if valid else '',finish_reason=g.finish_reason))
            result.append(dict(item_index=j,prompt_token_ids=list(o.prompt_token_ids),samples=samples))
        return result
    summary=dict(repeatability={},self_consistency=None)
    try:
        ray.init(address='local',ignore_reinit_error=True)
        engines,pgs=launch_engines(1,m['model'],precision='bfloat16',tensor_parallel_size=1,
                                 gpu_memory_utilization=a.gpu_memory_utilization)
        engine=engines[0]
        def fingerprint(stored=False):
            return ray.get(engine.collective_rpc.remote(weight_hash,args=(stored,)))
        pristine=fingerprint(True)
        assert fingerprint()==pristine
        greedy=SamplingParams(temperature=0.,seed=m['decoding_seed'],max_tokens=1024)
        # Reproduce production candidate order, then repeat the same ordered block.
        for repeat in range(2):
            for i in range(4):
                before_reset=fingerprint()
                ray.get(engine.collective_rpc.remote('reset_to_base_weights',args=()))
                after_reset=fingerprint()
                assert after_reset==pristine, 'Reset checksum mismatch'
                if i:
                    ray.get(engine.collective_rpc.remote('apply_perturbation',args=(m['seeds'][i],m['sigmas'][i])))
                before_generate=fingerprint()
                outputs=ray.get(engine.generate.remote(prompts,greedy,use_tqdm=False))
                assert len(outputs)==1550 and all(o.prompt==p for o,p in zip(outputs,prompts))
                after_generate=fingerprint()
                assert after_generate==before_generate, 'Generation changed weights'
                rec=records(outputs,items)
                flattened=[r['samples'][0] for r in rec]
                stage=dict(repeat=repeat,candidate=i,before_reset=before_reset,after_reset=after_reset,
                    before_generate=before_generate,after_generate=after_generate)
                hashes.append(stage)
                save(a.out_dir/f'candidate_{i}_repeat_{repeat}.json.gz',dict(weights=stage,records=rec))
                if repeat:
                    assert history[i][1]==before_generate, 'Repeated perturbation checksum mismatch'
                    summary['repeatability'][i]=compare(history[i][0],flattened,[str(x['target']) for x in items])
                else:
                    history[i]=(flattened,before_generate)
                (a.out_dir/'progress.json').write_text(json.dumps(dict(completed_calls=repeat*4+i+1,total_calls=9,
                    elapsed_s=time.monotonic()-start,summary=summary),indent=2))
                print(f'[{repeat*4+i+1}/9] greedy candidate={i} repeat={repeat} elapsed_s={time.monotonic()-start:.1f}',flush=True)
        ray.get(engine.collective_rpc.remote('reset_to_base_weights',args=()))
        assert fingerprint()==pristine
        params=SamplingParams(max_tokens=1024,**settings['self_consistency'])
        outputs=ray.get(engine.generate.remote(prompts[1050:],params,use_tqdm=False))
        assert len(outputs)==500 and all(o.prompt==p for o,p in zip(outputs,prompts[1050:]))
        assert fingerprint()==pristine
        rec=records(outputs,items[1050:])
        assert all(len(r['samples'])==5 for r in rec)
        save(a.out_dir/'base_self_consistency.json.gz',dict(records=rec))
        answers=[[r['samples'][j]['vote_answer'] for r in rec] for j in range(5)]
        coded=encode_answers(answers,[str(x['target']) for x in items[1050:]])
        v=vote_correct(coded); passed=(coded==0).any(axis=0)
        summary['self_consistency']=dict(member_accuracy=float((coded==0).mean()),vote_accuracy=float(v.mean()),
            checked_pass5=float(passed.mean()),vote_correct=v.tolist(),pass_correct=passed.tolist(),
            length_finish_rate=float(np.mean([s['finish_reason']=='length' for r in rec for s in r['samples']])))
        summary['weights']=hashes
        summary['elapsed_s']=time.monotonic()-start
        (a.out_dir/'COMPLETE.json').write_text(json.dumps(summary,indent=2))
        print(f'COMPLETE controls elapsed_s={summary["elapsed_s"]:.1f}',flush=True)
    finally:
        if engines: cleanup_engines(engines,pgs)
        if ray.is_initialized(): ray.shutdown()


if __name__=='__main__':
    main()
