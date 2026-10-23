"""CPU integration check of prepared data, scoring, and model prompt budget."""
import argparse
import json
from pathlib import Path
from dprandopt.gsm8k import answer_and_reward
from scripts.build_score_table import load_items, gold_string, vote_string


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--tokenizer',required=True)
    a=p.parse_args()
    from data_handlers import get_dataset_handler
    from transformers import AutoTokenizer
    handler=get_dataset_handler('gsm8k')
    protocol=json.loads((a.data_dir/'protocol.json').read_text())
    args=argparse.Namespace(dataset='gsm8k',item_seed=protocol['item_seed'],seed=20261024,
        pool=1000,public=50,test=1319,gsm8k_train=str(a.data_dir/'train.parquet'),gsm8k_test=str(a.data_dir/'test.parquet'))
    pool,public,test=load_items(args,handler)
    assert [len(pool),len(public),len(test)]==[1000,50,1319]
    raw={split:[json.loads(line) for line in (a.data_dir/f'raw/{split}.jsonl').read_text().splitlines()] for split in ('train','test')}
    for values,split,ids in [(pool,'train',protocol['pool_ids']),(public,'train',protocol['public_ids']),(test,'test',protocol['test_ids'])]:
        for item,i in zip(values,ids):
            assert item['messages']==[dict(role='user',content=raw[split][i]['question']+protocol['prompt_suffix'])]
            gold=gold_string(args,item)
            assert vote_string(args,handler,'#### '+gold,item)==gold
            assert answer_and_reward(handler,'#### '+gold,item['ground_truth'])[1]==1.
    text='#### 7\nAn unrelated number is 42'
    assert handler.compute_reward(text,'42')==1.
    assert answer_and_reward(handler,text,'42')==('7',0.)
    tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    items=pool+public+test
    prompts=[tokenizer.apply_chat_template(item['messages'],add_generation_prompt=True,tokenize=False) for item in items]
    lengths=[len(x) for x in tokenizer(prompts,add_special_tokens=False)['input_ids']]
    out=dict(pool=1000,public=50,test=1319,split_alignment_passed=True,scoring_alignment_passed=True,
        max_prompt_tokens=max(lengths),max_tokens=1024,context_2048_sufficient=max(lengths)+1024<=2048,
        gpu_generation_started=False)
    assert out['context_2048_sufficient']
    (a.data_dir/'preflight.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out))


if __name__=='__main__': main()
