import json
from pathlib import Path

import numpy as np
import unittest
import tempfile
from contextlib import contextmanager


@contextmanager
def raises(kind, match=None):
    with unittest.TestCase().assertRaisesRegex(kind, match or ".*"):
        yield

from dprandopt.pilot import prepare, ensure_manifest, save_candidate, read_candidate, context_budget
from dprandopt.scoretable import ScoreTable


def reference(tmp_path):
    rng=np.random.default_rng(123)
    seeds=np.r_[-1,rng.choice(2**31-1,size=8,replace=False)]
    sigmas=np.r_[0.,rng.choice([.001,.002],size=8)]
    table=ScoreTable(pool_reward=np.zeros((9,2)),public_reward=np.zeros((9,1)),
        test_answer=np.zeros((9,2),dtype=int),seed=seeds,sigma=sigmas,
        output_lengths=np.full((9,5),512))
    path=tmp_path/'ref.npz'; table.save(path)
    Path(str(path)+'.meta.json').write_text(json.dumps(dict(dataset='countdown',candidate_seed=123,
        item_seed=456,sigmas=[.001,.002],pool=2,public=1,test=2,model='fixture')))
    return path


def check_pilot_prefix_preserves_full_rng_draw_and_items(reference):
    short,items=prepare(reference,3,1024)
    full,full_items=prepare(reference,9,1024)
    assert short['seeds']==full['seeds'][:3]
    assert short['sigmas']==full['sigmas'][:3]
    assert items==full_items
    assert short['items_sha256']==full['items_sha256']


def check_resume_rejects_cap_or_manifest_change(reference,tmp_path):
    m,_=prepare(reference,3,1024)
    out=tmp_path/'pilot'; ensure_manifest(out,m); ensure_manifest(out,m)
    changed=dict(m,max_tokens=2048)
    with raises(ValueError,match='incompatible'):
        ensure_manifest(out,changed)


def check_atomic_bundle_retains_text_finish_reasons_and_refuses_incomplete(reference,tmp_path):
    m,items=prepare(reference,3,1024)
    records=[dict(item_index=i,text='example',length=1024,finish_reason='length',
                  stop_reason=None,vote_answer='',reward=0.) for i in range(len(items))]
    path=tmp_path/'candidate_0000.json.gz'
    with raises(ValueError,match='Incomplete'):
        save_candidate(path,m,0,records[:-1])
    assert not path.exists()
    save_candidate(path,m,0,records)
    assert read_candidate(path,m,0)['records']==records
    with raises(ValueError,match='manifest'):
        read_candidate(path,dict(m,max_tokens=2048),0)
    with raises(FileExistsError):
        save_candidate(path,m,0,records)


class PilotTests(unittest.TestCase):
    def test_context_includes_prompt_and_full_output(self):
        self.assertEqual(context_budget(150,1024),2048)
        self.assertEqual(context_budget(150,2048),4096)
        with self.assertRaises(ValueError):
            context_budget(150,2048,2048)

    def run_case(self, case):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ref = reference(root)
            if case is check_pilot_prefix_preserves_full_rng_draw_and_items:
                case(ref)
            else:
                case(ref, root)

    def test_prefix(self):
        self.run_case(check_pilot_prefix_preserves_full_rng_draw_and_items)

    def test_mismatched_resume(self):
        self.run_case(check_resume_rejects_cap_or_manifest_change)

    def test_atomic_checkpoint(self):
        self.run_case(check_atomic_bundle_retains_text_finish_reasons_and_refuses_incomplete)


if __name__ == '__main__':
    unittest.main()
