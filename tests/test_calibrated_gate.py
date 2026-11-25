import unittest
import numpy as np
from src.calibrated_gate import calibration, composite, run_gate

class TestCalibrated(unittest.TestCase):
    def test_exact_labels(self):
        y=[.7,.8,.7];a={0:(.5,.5),1:(.6,.6),2:(.2,.2)};e=[(0,1,.02),(0,2,.02),(1,2,.02)]
        g,b,u,K=calibration(y,y,a,e,[0,1,2],.05)
        self.assertEqual((g,b,u,K),([0],[1,2],[],2))
    def test_unresolved_is_not_good(self):
        g,b,u,K=calibration([.2,.2],[.8,.8],{0:(0,1),1:(0,1)},[(0,1,.1)],[0],.05)
        self.assertEqual((g,b,u,K),([],[],[0],1))
    def test_protected_edge_and_unknown(self):
        lo=[.5,.7,.6];a={0:(.4,.4),1:(0,1),2:(0,1)};e=[(0,1,0),(1,2,0)]
        l,u,s=composite(lo,lo,a,e,[0],[],[1],1)
        # No additional graph gain when all unknown edges may fail.
        self.assertEqual(s,'direct_no_transfer');self.assertLessEqual(l[0],.2);self.assertGreaterEqual(u[0],.2)
    def test_sharp_graph(self):
        y=[.5,.7,.6];a={0:(.4,.4),1:(0,1),2:(0,1)};e=[(0,1,0),(1,2,0)]
        l,u,s=composite(y,y,a,e,[],[],[0,1],0)
        np.testing.assert_allclose(l,[.2,.1]);np.testing.assert_allclose(u,[.2,.1]);self.assertEqual(s,'graph')
    def test_budget_and_unknown_reference(self):
        r=run_gate([.42,.73,.27],[.47,.78,.65],[(0,1,.1),(0,2,.1)],30000,np.random.default_rng(1),sample_size=2,proxy_n=1000)
        self.assertLessEqual(r['spend'],30000);self.assertEqual(r['k_coverage'],1)
    def test_inconsistent_fallback(self):
        a={0:(.4,.5),1:(.3,.4)}
        l,u,s=composite([.5,.6],[.5,.6],a,[(0,1,.1)],[],[0],[],0)
        self.assertEqual(s,'inconsistent');self.assertAlmostEqual(l[0],-.2)
    def test_independent_sample_obstruction(self):
        # Conditional on fixed intervals, non-certified edges are fixed labels.
        # A random sample cannot certify an upper count smaller than their count
        # except on the hypergeometric lower-tail error event.
        from scipy.stats import hypergeom
        from src.calibrated_gate import count_upper
        for M in range(1,11):
            for n in range(M+1):
                for K in range(M+1):
                    p=sum(hypergeom.pmf(x,M,K,n) for x in range(n+1) if count_upper(M,n,x,.05)<K)
                    self.assertLessEqual(p,.05+1e-10)
    def test_all_edges_unresolved_direct(self):
        a={0:(.4,.5),1:(.6,.7)}
        l,u,status=composite([.1,.1],[.9,.9],a,[(0,1,0)],[],[],[0],1)
        self.assertAlmostEqual(l[0],.1);self.assertAlmostEqual(u[0],.3);self.assertEqual(status,'direct_no_transfer')
