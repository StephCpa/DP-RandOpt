"""Add opt-in context/batch limits to this experiment's RandOpt engine."""
import argparse
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('engine',type=Path)
a=p.parse_args()
s=a.engine.read_text()
marker='# Optional pilot limits; absent variables preserve the original defaults.'
if marker not in s:
    anchor='        if multimodal:\n'
    if s.count(anchor)!=1:
        raise ValueError('Unexpected engine source; refusing an ambiguous patch')
    patch='''        # Optional pilot limits; absent variables preserve the original defaults.
        for key in ('max_model_len', 'max_num_seqs', 'max_num_batched_tokens'):
            value = os.environ.get('RANDOPT_' + key.upper())
            if value:
                engine_kwargs[key] = int(value)
'''
    backup=a.engine.with_suffix('.py.before_pilot_limits')
    if not backup.exists():
        backup.write_text(s)
    a.engine.write_text(s.replace(anchor,patch+anchor))
compile(a.engine.read_text(),str(a.engine),'exec')
print('Pilot engine limits available:',a.engine)
