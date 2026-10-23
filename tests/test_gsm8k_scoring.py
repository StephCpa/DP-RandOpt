import unittest
from dprandopt.gsm8k import canonical_number, answer_and_reward


class ScoringTests(unittest.TestCase):
    def test_numeric_equivalence_and_invalid_answers(self):
        self.assertEqual(canonical_number('1,000.00'),'1000')
        self.assertEqual(canonical_number('-0.0'),'0')
        self.assertEqual(canonical_number('NaN'),'')
        self.assertEqual(canonical_number(''),'')

    def test_reward_follows_voting_answer_not_a_second_gold_dependent_parse(self):
        class Handler:
            def extract_answer(self,text): return '7'
            def compute_reward(self,text,gold): return 1.  # Legacy fallback would match trailing 42.
        answer,reward=answer_and_reward(Handler(),'#### 7\nAn unrelated number is 42','42')
        self.assertEqual(answer,'7')
        self.assertEqual(reward,0.)


if __name__=='__main__': unittest.main()
