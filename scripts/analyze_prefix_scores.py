"""Summarize CPU retokenized prefixes of the same saved trajectories."""
import argparse
import json
from pathlib import Path
from dprandopt.scoretable import ScoreTable
from scripts.audit_vote_selection import analyze
from scripts.audit_length_series import paired


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--folder',type=Path,default=Path('results/gpu-smoke/prefix_2048_cpu'))
    a=p.parse_args()
    tables={cap:ScoreTable.load(str(a.folder/f'prefix_{cap}.npz')) for cap in (512,1024,2048)}
    out=dict(scope='Retokenized prefixes of saved 2048 trajectories; approximate token boundaries, fixed candidate pool.',
             results={cap:analyze(t,400,2000) for cap,t in tables.items()})
    out['paired_single_candidate_changes']={}
    for old,new in ((512,1024),(1024,2048),(512,2048)):
        out['paired_single_candidate_changes'][f'{new}_minus_{old}']=dict(
            base=paired(tables[new].test_answer[0]==0,tables[old].test_answer[0]==0),
            perturbations=paired(tables[new].test_answer[1:]==0,tables[old].test_answer[1:]==0))
    (a.folder/'analysis.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out['paired_single_candidate_changes'],indent=2))


if __name__=='__main__': main()
