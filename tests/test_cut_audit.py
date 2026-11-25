import unittest
import numpy as np
from scipy.stats import hypergeom
from src.cut_audit import witnesses,alternatives,allocation,upper_bad_count,kl,run_gate,graph_suite

class TestCutAudit(unittest.TestCase):
    def test_bridge(self):self.assertEqual(witnesses(3,[(0,1),(1,2)],1),[(2,),(1,2)])
    def test_dense(self):self.assertEqual(witnesses(3,[(0,1),(1,2),(0,2)],1),[])
    def test_cluster(self):
        n,e,k,c=graph_suite()['cluster'];self.assertEqual(len(witnesses(n,e,k)),2)
    def test_primal_dual(self):
        for n,e,k,c in graph_suite().values():
            mu=np.full(n,.6);a=alternatives(mu,witnesses(n,e,k),.15);x,v,d,K=allocation(mu,a,c)
            self.assertTrue(np.all(K@x>=kl(.95,.05)-1e-8));self.assertTrue(np.all(K.T@d<=np.array(c)+1e-8));self.assertAlmostEqual(v,d.sum()*kl(.95,.05))
    def test_finite_population_coverage(self):
        for M in range(1,11):
            for n in range(M+1):
                upp=[upper_bad_count(M,n,x,.05) for x in range(n+1)]
                for k in range(M+1):
                    fail=sum(hypergeom.pmf(x,M,k,n) for x in range(n+1) if upp[x]<k)
                    self.assertLessEqual(fail,.05+1e-10)
    def test_count_monotone(self):
        self.assertEqual(upper_bad_count(50,50,3,.05),3)
        self.assertGreater(upper_bad_count(50,10,0,.05),0)
        self.assertLessEqual(upper_bad_count(50,20,0,.05),upper_bad_count(50,20,1,.05))
    def test_budget(self):
        mu=np.full(3,.6);a=alternatives(mu,[(1,),(2,)],.15);x,*_=allocation(mu,a,[1,2,3]);p,c=run_gate(mu,a,[1,2,3],x,mu,100,np.random.default_rng(2));self.assertLessEqual(c,100)
