import unittest
import numpy as np
from scripts.diagnose_gsm8k_selection import spread,metrics


class DiagnosticTests(unittest.TestCase):
    def test_shared_difficulty_is_not_candidate_spread_noise(self):
        r=spread(np.tile([0,1,1,0,1],(8,1)))
        self.assertAlmostEqual(r['observed_variance'],0)
        self.assertAlmostEqual(r['shared_item_sampling_variance'],0)
        self.assertGreater(r['marginal_binomial_variance'],0)

    def test_deterministic_distinct_candidates(self):
        r=spread(np.array([[0]*10,[1]*10],float))
        self.assertAlmostEqual(r['shared_item_sampling_variance'],0)
        self.assertAlmostEqual(r['estimated_latent_variance'],.5)

    def test_covariance_trace_equivalence(self):
        c=np.random.default_rng(7).integers(0,2,(20,80))
        cov=np.cov(c,ddof=1)
        expected=(np.trace(cov)-cov.sum()/len(c))/(len(c)-1)/c.shape[1]
        self.assertAlmostEqual(spread(c)['shared_item_sampling_variance'],expected)

    def test_disagreement_and_abstention(self):
        m=metrics(np.array([[0,1,-1],[0,2,-1],[1,2,0]]))
        self.assertAlmostEqual(m['disagreement'],6/9)
        self.assertAlmostEqual(m['disagreement_both_valid'],4/6)


if __name__=='__main__':unittest.main()
