import itertools
import unittest
import numpy as np
from scripts.audit_vote_selection import ensemble_metrics, exact_random_pass


class DecompositionTests(unittest.TestCase):
    def test_exact_pass_matches_subset_enumeration(self):
        correct=np.array([[1,0,0],[0,1,0],[0,0,0],[1,0,0]],dtype=bool)
        exhaustive=np.mean([correct[list(s)].any(axis=0) for s in itertools.combinations(range(4),2)],axis=0)
        np.testing.assert_allclose(exact_random_pass(correct,2),exhaustive)

    def test_checker_recovers_correct_member_when_plurality_is_wrong(self):
        a=np.array([[1,-1,0],[1,-1,-1],[0,-1,2]])
        m=ensemble_metrics(a)
        np.testing.assert_array_equal(m['vote'],[False,False,True])
        np.testing.assert_array_equal(m['checked_pass'],[True,False,True])


if __name__=='__main__': unittest.main()
