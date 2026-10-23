"""CPU paired analysis of completed pilot checkpoints; no model inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from dprandopt.pilot import prepare, read_candidate
from dprandopt.scoretable import ScoreTable


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',required=True)
    p.add_argument('--pilot-dir',required=True,type=Path)
    p.add_argument('--previous-pilot-dir',type=Path,help='Optional shorter-cap pilot with complete response texts')
    a=p.parse_args()
    m=json.loads((a.pilot_dir/'manifest.json').read_text())
    expected,items=prepare(a.reference,len(m['seeds']),m['max_tokens'])
    if m!=expected:
        raise ValueError('Pilot and reference manifests disagree')
    indices=[]; rows=[]
    for i in m['candidate_indices']:
        f=a.pilot_dir/f'candidate_{i:04d}.json.gz'
        if f.exists():
            indices.append(i); rows.append(read_candidate(f,m,i)['records'])
    if not rows:
        raise ValueError('No completed candidate checkpoints to analyze')
    ref=ScoreTable.load(a.reference)
    lengths=np.array([[r['length'] for r in row] for row in rows])
    finish=np.array([[r['finish_reason'] for r in row] for row in rows])
    split=m['pool']+m['public']
    # Compare correctness, not independently assigned wrong-answer codes.
    gold=[str(x['target']) for x in items[split:]]
    correct=np.array([[r['vote_answer']==g for r,g in zip(row[split:],gold)] for row in rows])
    old=(ref.test_answer[indices]==0)
    by_candidate=[]
    for row,i in enumerate(indices):
        by_candidate.append(dict(index=i,old_accuracy=float(old[row].mean()),
            new_accuracy=float(correct[row].mean()),paired_difference=float(correct[row].mean()-old[row].mean()),
            old_cap_hit=float((ref.output_lengths[i]==512).mean()),
            new_cap_hit=float((lengths[row]==m['max_tokens']).mean()),
            length_finish=float((finish[row]=='length').mean())))
    out=dict(completed=len(rows),planned=len(m['seeds']),max_tokens=m['max_tokens'],
        complete=len(rows)==len(m['seeds']),candidates=by_candidate,
        overall_cap_hit=float((lengths==m['max_tokens']).mean()),
        overall_length_finish=float((finish=='length').mean()),
        scope='Paired length diagnostic only; not a full-candidate-pool D1 comparison')
    if 0 in indices:
        b=indices.index(0); diff=correct[b].astype(float)-old[b].astype(float)
        rng=np.random.default_rng(20261022)
        boot=diff[rng.integers(len(diff),size=(5000,len(diff)))].mean(axis=1)
        out['base_paired_ci95']=np.quantile(boot,[.025,.975]).tolist()
    mask=np.array(indices)!=0
    if mask.any():
        diff=(correct[mask].astype(float)-old[mask].astype(float)).mean(axis=0)
        rng=np.random.default_rng(20261022)
        boot=diff[rng.integers(len(diff),size=(5000,len(diff)))].mean(axis=1)
        deltas=correct[mask].mean(axis=1)-old[mask].mean(axis=1)
        out['perturbed_summary']=dict(count=int(mask.sum()),
            old_mean_test_accuracy=float(old[mask].mean()),new_mean_test_accuracy=float(correct[mask].mean()),
            paired_mean_difference=float(diff.mean()),conditional_item_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist(),
            improved=int((deltas>0).sum()),unchanged=int((deltas==0).sum()),worsened=int((deltas<0).sum()),
            median_old_cap_hit=float(np.median((ref.output_lengths[np.array(indices)[mask]]==512).mean(axis=1))),
            median_new_length_finish=float(np.median((finish[mask]=='length').mean(axis=1))))
    out['old_overall_cap_hit']=float((ref.output_lengths[indices]==512).mean())
    out['new_length_finish_count']=int((finish=='length').sum())
    out['responses']=int(lengths.size)
    if a.previous_pilot_dir:
        previous=json.loads((a.previous_pilot_dir/'manifest.json').read_text())
        if {k:v for k,v in m.items() if k!='max_tokens'}!={k:v for k,v in previous.items() if k!='max_tokens'}:
            raise ValueError('Previous pilot differs in items, candidates or reference')
        if previous['max_tokens']>=m['max_tokens']:
            raise ValueError('Previous pilot must use a shorter output cap')
        prior_rows=[read_candidate(a.previous_pilot_dir/f'candidate_{i:04d}.json.gz',previous,i)['records'] for i in indices]
        prior_correct=np.array([[r['vote_answer']==g for r,g in zip(row[split:],gold)] for row in prior_rows])
        stopped=matched=0
        for prior_row,new_row in zip(prior_rows,rows):
            for prior,new in zip(prior_row,new_row):
                if prior['finish_reason']=='stop':
                    stopped+=1
                    matched+=int(prior['text']==new['text'])
        comparison=dict(previous_cap=previous['max_tokens'],
            previous_stop_responses=stopped,unchanged_stop_texts=matched,
            unchanged_stop_text_fraction=matched/stopped if stopped else None,
            note='Text consistency diagnostic; does not certify token-level deterministic prefixes.')
        if 0 in indices:
            b=indices.index(0)
            comparison['base_previous_accuracy']=float(prior_correct[b].mean())
            comparison['base_new_accuracy']=float(correct[b].mean())
        if mask.any():
            comparison['perturbed_previous_mean_accuracy']=float(prior_correct[mask].mean())
            comparison['perturbed_new_mean_accuracy']=float(correct[mask].mean())
        out['previous_pilot_comparison']=comparison
    out_path=a.pilot_dir/'paired_analysis.json'
    out_path.write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    main()
