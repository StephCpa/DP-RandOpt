"""CPU-only approximate token-prefix scoring with the production RandOpt parser.

Retokenized text is not a recovered original token sequence. Report length
differences and +/-2 boundary sensitivity instead of assuming exact prefixes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from dprandopt.pilot import prepare, read_candidate
from dprandopt.countdown import encode_answers
from dprandopt.scoretable import ScoreTable


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', required=True)
    p.add_argument('--pilot-dir', required=True, type=Path)
    p.add_argument('--out-dir', required=True, type=Path)
    p.add_argument('--tokenizer', required=True)
    a = p.parse_args()
    from transformers import AutoTokenizer
    from data_handlers import get_dataset_handler
    import data_handlers.countdown as handler_module
    m = json.loads((a.pilot_dir/'manifest.json').read_text())
    expected, items = prepare(a.reference, 32, 2048)
    assert expected == m
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    handler = get_dataset_handler('countdown')
    answers = {cap: [] for cap in (512, 1024, 2048)}
    rewards = {cap: [] for cap in answers}
    lengths = {cap: [] for cap in answers}
    boundary = {cap: [] for cap in (512, 1024)}
    diagnostics = {r: [] for r in ('stop', 'length')}
    full_mismatches = dict(reward=0, vote_answer=0, roundtrip_text=0)
    def score(text, item):
        ans, valid, _ = handler.extract_answer_for_voting(text, numbers=item['numbers'])
        return ans if valid else '', float(handler.compute_reward(text, item['ground_truth']))
    for i in range(32):
        row = read_candidate(a.pilot_dir/f'candidate_{i:04d}.json.gz', m, i)['records']
        ids = tokenizer([r['text'] for r in row], add_special_tokens=False)['input_ids']
        ca = {cap: [] for cap in answers}; cr = {cap: [] for cap in answers}; cl = {cap: [] for cap in answers}
        cb = {cap: [] for cap in boundary}
        for j, (r, tokens, item) in enumerate(zip(row, ids, items)):
            diagnostics[r['finish_reason']].append(len(tokens)-r['length'])
            full_mismatches['roundtrip_text'] += tokenizer.decode(tokens, clean_up_tokenization_spaces=False) != r['text']
            original = score(r['text'], item)
            full_mismatches['reward'] += abs(original[1]-r['reward']) > 1e-6
            full_mismatches['vote_answer'] += original[0] != r['vote_answer']
            for cap in answers:
                # 2048 means the full saved trajectory, not an approximate re-cut.
                text = r['text'] if cap == 2048 or len(tokens) <= cap else tokenizer.decode(tokens[:cap], clean_up_tokenization_spaces=False)
                ans, reward = original if text == r['text'] else score(text, item)
                ca[cap].append(ans); cr[cap].append(reward); cl[cap].append(min(len(tokens), cap))
                if cap != 2048:
                    variants = [score(tokenizer.decode(tokens[:max(0,cap+d)], clean_up_tokenization_spaces=False), item) for d in (-2, 2)]
                    cb[cap].append(any(v != (ans,reward) for v in variants))
        for cap in answers:
            answers[cap].append(ca[cap]); rewards[cap].append(cr[cap]); lengths[cap].append(cl[cap])
        for cap in boundary:
            boundary[cap].append(cb[cap])
        print(f'prefixes {i+1}/32', flush=True)
    assert full_mismatches['reward'] == 0 and full_mismatches['vote_answer'] == 0, full_mismatches
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for cap in answers:
        r = np.array(rewards[cap], dtype=np.float32)
        t = ScoreTable(pool_reward=r[:,:1000], public_reward=r[:,1000:1050], reward_max=1.1,
            test_answer=encode_answers([x[1050:] for x in answers[cap]], [str(x['target']) for x in items[1050:]]),
            output_lengths=np.array(lengths[cap]), seed=np.array(m['seeds']), sigma=np.array(m['sigmas']))
        t.save(str(a.out_dir/f'prefix_{cap}.npz'))
    diag = {}
    for reason, values in diagnostics.items():
        v = np.array(values)
        diag[reason] = dict(count=len(v), exact_length_fraction=float((v==0).mean()),
            absolute_difference_le2_fraction=float((np.abs(v)<=2).mean()),
            difference_quantiles=np.quantile(v,[0,.01,.5,.99,1]).tolist())
    info = dict(source_manifest=m, full_rescore_mismatches=full_mismatches, retokenized_minus_saved_lengths=diag,
        boundary_score_or_answer_sensitivity={cap:dict(all=float(np.mean(v)), test=float(np.mean(np.array(v)[:,1050:]))) for cap,v in boundary.items()},
        parser_file_sha256=hashlib.sha256(Path(handler_module.__file__).read_bytes()).hexdigest(),
        tokenizer_files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(a.tokenizer).glob('*token*') if f.is_file()},
        limitation='Approximate prefixes of saved 2048 trajectories, not separate-capped-run equivalence; token IDs were not saved in the source.')
    (a.out_dir/'diagnostics.json').write_text(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
