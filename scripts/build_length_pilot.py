"""Run a fixed prefix of an archived candidate manifest at a longer cap."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from dprandopt.pilot import prepare, ensure_manifest, read_candidate, save_candidate, digest, context_budget
from dprandopt.countdown import encode_answers
from dprandopt.scoretable import ScoreTable


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference', required=True)
    p.add_argument('--out-dir', required=True, type=Path)
    p.add_argument('--count', type=int, default=32)
    p.add_argument('--max-tokens', type=int, default=1024)
    p.add_argument('--gpu-memory-utilization',type=float,default=.40)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--model', help='Content-verified local copy of the reference model on another host')
    p.add_argument('--max-model-len',type=int,help='Context limit; default rounds prompt plus output budget up to a power of two')
    a=p.parse_args()
    m,items=prepare(a.reference,a.count,a.max_tokens)
    fingerprint=ensure_manifest(a.out_dir,m)
    files=[a.out_dir/f'candidate_{i:04d}.json.gz' for i in range(a.count)]
    done=[i for i,f in enumerate(files) if f.exists()]
    for i in done:
        read_candidate(files[i],m,i)
    print(f'manifest={fingerprint} completed={len(done)}/{a.count}',flush=True)
    if a.prepare_only:
        print('CPU preflight passed; no model loaded',flush=True)
        return
    todo=[i for i in range(a.count) if i not in done]
    if todo:
        # Set before Ray starts so both actors and workers inherit these values.
        os.environ['RANDOPT_BASE_WEIGHTS_DEVICE']='cpu'
        os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
        os.environ['RANDOPT_MAX_NUM_SEQS']='128'
        os.environ['RANDOPT_MAX_NUM_BATCHED_TOKENS']='2048'
        import ray
        import torch
        import vllm
        import transformers
        from transformers import AutoTokenizer
        from vllm import SamplingParams
        from core import cleanup_engines, launch_engines
        from data_handlers import get_dataset_handler
        from build_score_table import vote_string
        handler=get_dataset_handler('countdown')
        model_path=a.model or m['model']
        tokenizer=AutoTokenizer.from_pretrained(model_path)
        prompts=[tokenizer.apply_chat_template(x['messages'],add_generation_prompt=True,tokenize=False) for x in items]
        max_prompt_tokens=max(len(tokenizer.encode(x,add_special_tokens=False)) for x in prompts)
        context_limit=context_budget(max_prompt_tokens,m['max_tokens'],a.max_model_len)
        os.environ['RANDOPT_MAX_MODEL_LEN']=str(context_limit)
        prompt_path=a.out_dir/'prompts.sha256'
        prompt_hash=digest(prompts)
        if prompt_path.exists() and prompt_path.read_text().strip()!=prompt_hash:
            raise ValueError('Tokenizer-rendered prompts changed on resume')
        prompt_path.write_text(prompt_hash+'\n')
        runtime=dict(manifest_sha256=fingerprint,prompts_sha256=prompt_hash,
            torch=torch.__version__,vllm=vllm.__version__,transformers=transformers.__version__,
            gpu_memory_utilization=a.gpu_memory_utilization,base_weights_device='cpu',
            CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
            model_path=model_path, max_model_len=context_limit, max_prompt_tokens=max_prompt_tokens, max_num_seqs=128,
            max_num_batched_tokens=2048, started_unix=time.time())
        (a.out_dir/f'runtime_{time.time_ns()}.json').write_text(json.dumps(runtime,indent=2))
        params=SamplingParams(temperature=0.,seed=m['decoding_seed'],max_tokens=m['max_tokens'])
        ray.init(address='local',ignore_reinit_error=True)
        engines=[]; pgs=[]
        start=time.monotonic(); tokens=0
        try:
            engines,pgs=launch_engines(1,model_path,precision='bfloat16',tensor_parallel_size=1,
                                     gpu_memory_utilization=a.gpu_memory_utilization)
            engine=engines[0]
            for step,i in enumerate(todo,1):
                if m['seeds'][i]<0:
                    ray.get(engine.collective_rpc.remote('reset_to_base_weights',args=()))
                else:
                    ray.get(engine.collective_rpc.remote('apply_perturbation',args=(m['seeds'][i],m['sigmas'][i])))
                outputs=ray.get(engine.generate.remote(prompts,params,use_tqdm=False))
                if len(outputs)!=len(items):
                    raise ValueError('Generation returned the wrong number of responses')
                records=[]
                for j,(o,item) in enumerate(zip(outputs,items)):
                    g=o.outputs[0]
                    if o.prompt != prompts[j]:
                        raise ValueError('Response order does not match item manifest')
                    records.append(dict(item_index=j,text=g.text,length=len(g.token_ids),
                        finish_reason=g.finish_reason,stop_reason=g.stop_reason,
                        vote_answer=vote_string(argparse.Namespace(dataset='countdown'),handler,g.text,item),
                        reward=float(handler.compute_reward(g.text,item['ground_truth']))))
                save_candidate(files[i],m,i,records)
                tokens+=sum(r['length'] for r in records)
                elapsed=time.monotonic()-start
                print(f'[{len(done)+step}/{a.count}] candidate={i} checkpoint_saved elapsed_s={elapsed:.1f} tok_s={tokens/elapsed:.0f}',flush=True)
        finally:
            if engines:
                cleanup_engines(engines,pgs)
            ray.shutdown()
    rows=[read_candidate(f,m,i)['records'] for i,f in enumerate(files)]
    rewards=np.array([[r['reward'] for r in row] for row in rows],dtype=np.float32)
    lengths=np.array([[r['length'] for r in row] for row in rows],dtype=np.int32)
    P,Q,T=m['pool'],m['public'],m['test']
    answers=[[r['vote_answer'] for r in row[P+Q:]] for row in rows]
    table=ScoreTable(pool_reward=rewards[:,:P],public_reward=rewards[:,P:P+Q],reward_max=1.1,
        test_answer=encode_answers(answers,[str(x['target']) for x in items[P+Q:]]),
        output_lengths=lengths,seed=np.array(m['seeds']),sigma=np.array(m['sigmas']))
    table.save(str(a.out_dir/'table.tmp.npz'))
    os.replace(a.out_dir/'table.tmp.npz',a.out_dir/'table.npz')
    (a.out_dir/'COMPLETE.json').write_text(json.dumps(dict(manifest_sha256=fingerprint,count=a.count,
        responses=a.count*len(items),finished_unix=time.time()),indent=2))
    print('COMPLETE: all candidate checkpoints verified and table.npz written',flush=True)


if __name__=='__main__':
    main()
