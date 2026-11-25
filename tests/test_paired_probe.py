import unittest
import numpy as np
from scipy.stats import binom
from src.paired_probe import cp,paired_ci,direct_ci,joint_probs,cost_lower,run

class TestPaired(unittest.TestCase):
    def test_zero_discordance(self):
        l,u=paired_ci(0,0,100);self.assertLess(l,0);self.assertGreater(u,0)
    def test_marginals(self):
        for d in (-.1,0,.1):
            p=joint_probs(d,.2);self.assertAlmostEqual(p[2]+p[3]-p[1]-p[3],d);self.assertAlmostEqual(p.sum(),1)
    def test_exact_small_grid(self):
        for n in (5,10,20):
            for q in (0,.05,.2,.5,.9,1):
                for r in (0,.1,.5,.9,1):
                    fail=0.
                    for k in range(n+1):
                        x=np.arange(k+1);l,u=paired_ci(x,k-x,n);d=q*(2*r-1)
                        fail+=binom.pmf(k,n,q)*np.sum(binom.pmf(x,k,r)*((l>d+1e-12)|(u<d-1e-12)))
                    self.assertLessEqual(fail,.05+1e-10)
    def test_symmetry(self):
        l,u=paired_ci(12,4,100);a,b=paired_ci(4,12,100);self.assertAlmostEqual(l,-b);self.assertAlmostEqual(u,-a)
    def test_direct_boundary(self):
        l,u=direct_ci(0,0,100);self.assertLess(l,0);self.assertGreater(u,0)
    def test_cost_lower(self):
        self.assertLess(cost_lower(.12,.06,40),cost_lower(.8,.06,40))
    def test_invalid(self):
        with self.assertRaises(ValueError):joint_probs(.4,.2)
    def test_budget(self):
        import tempfile
        with tempfile.TemporaryDirectory() as p:
            for row in run(p,trials=5)['rows']:
                self.assertLessEqual(row['mean_cost'],row['budget'])
