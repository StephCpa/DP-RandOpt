import copy
import unittest
from scripts.run_batched_controls import compare


class FlipMetricTests(unittest.TestCase):
    def test_reward_correctness_and_token_flips_are_distinct(self):
        first=[dict(reward=0.,vote_answer='',token_ids=[1]) for _ in range(1550)]
        second=copy.deepcopy(first)
        second[0]['reward']=.01  # Format credit changes without answer correctness.
        second[1000]['token_ids']=[2]  # Public text changes without score changes.
        second[1050].update(reward=1.,vote_answer='42',token_ids=[3])
        r=compare(first,second,['42']*1550)
        self.assertEqual(r['pool']['reward_flip_rate'],1/1000)
        self.assertEqual(r['pool']['correctness_flip_rate'],0.)
        self.assertEqual(r['public']['token_sequence_flip_rate'],1/50)
        self.assertEqual(r['public']['reward_flip_rate'],0.)
        self.assertEqual(r['test']['correctness_flip_rate'],1/500)
        self.assertEqual(r['all']['reward_flip_rate'],2/1550)


if __name__=='__main__': unittest.main()
