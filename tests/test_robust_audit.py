import unittest
import numpy as np
from src.robust_audit import robust_bounds,audit,lifted_certificate,sentinel_edge

class TestRobust(unittest.TestCase):
    def test_bridge_cut(self):
        a,b,_=robust_bounds([.5,.6],[.5,.6],[(0,1,0)],1)
        self.assertAlmostEqual(a[0],-1);self.assertAlmostEqual(b[0],1)
    def test_cycle_survives(self):
        y=[.4,.6,.5];a,b,_=robust_bounds(y,y,[(0,1,0),(0,2,0),(1,2,0)],1)
        np.testing.assert_allclose(a,[.2,.1]);np.testing.assert_allclose(b,[.2,.1])
    def test_corrupted_bridge_anchor(self):
        y=[.6,.8];a,b,n=robust_bounds(y,y,[(0,1,0)],1,{0:(.5,.5),1:(.3,.3)})
        self.assertAlmostEqual(a[0],-.2);self.assertAlmostEqual(b[0],-.2)
        self.assertIsNone(robust_bounds(y,y,[(0,1,0)],0,{0:(.5,.5),1:(.3,.3)}))
    def test_budget(self):
        r=audit(np.array([.5,.8]),np.array([.6,.9]),[(0,1,.01)],1,10000,np.random.default_rng(1),proxy_n=1000)
        self.assertLessEqual(r['spend'],10000)
    def test_no_edges_equal_direct_bounds(self):
        l,u,_=robust_bounds([.2,.8],[.3,.9],[],0,{0:(.4,.5),1:(.6,.7)})
        self.assertAlmostEqual(l[0],.1);self.assertAlmostEqual(u[0],.3)

    def test_interval_exact_cancellation(self):
        a,b,_=robust_bounds([.4,.2,.6],[.4,.8,.6],[(0,1,0),(1,2,0)],0)
        self.assertAlmostEqual(a[1],.2);self.assertAlmostEqual(b[1],.2)
    def test_outside_budget_counterexample(self):
        y=[.5,.7,.5];a,b,_=robust_bounds(y,y,[(0,1,0),(0,2,0),(1,2,0)],1)
        # Two violated edges: truth (.5,.3,.5) is outside assumed class.
        self.assertGreaterEqual(a[0],.19)
    def test_lp_oracle(self):
        from scipy.optimize import linprog
        rng=np.random.default_rng(7)
        for _ in range(10):
            n=4;mu=rng.uniform(.2,.8,n);bias=rng.uniform(-.1,.1,n);y=mu+bias
            lo=np.maximum(0,y-.04);hi=np.minimum(1,y+.03)
            edges=[(i,j,abs(bias[i]-bias[j])+.01) for i,j in [(0,1),(1,2),(2,3),(0,3)]]
            D=lifted_certificate(lo,hi,edges,{0:(mu[0]-.02,mu[0]+.02)})
            A=[];B=[]
            for i in range(n):
                row=np.zeros(2*n);row[i]=1;row[n+i]=-1;A.extend([row,-row]);B.extend([hi[i],-lo[i]])
            for i,j,r in edges:
                row=np.zeros(2*n);row[n+i]=1;row[n+j]=-1;A.extend([row,-row]);B.extend([r,r])
            bounds=[(mu[0]-.02,mu[0]+.02)]+[(0,1)]*(n-1)+[(None,None)]*n
            for i in range(n):
                for j in range(n):
                    c=np.zeros(2*n);c[i]+=1;c[j]-=1
                    opt=linprog(c,A_ub=A,b_ub=B,bounds=bounds,method='highs')
                    self.assertTrue(opt.success);self.assertAlmostEqual(-opt.fun,D[i,j],places=9)

    def test_sentinel_no_extrapolation(self):
        y=[.5,.7];a={0:(.5,.5),1:(.3,.3)}
        self.assertEqual(sentinel_edge(y,y,a,0,1,.05)[2],'falsified')
        self.assertEqual(sentinel_edge(y,y,{0:(.4,.6),1:(0,1)},0,1,.05)[2],'unresolved')
        self.assertEqual(sentinel_edge(y,y,{0:(.4,.4),1:(.6,.6)},0,1,.05)[2],'certified')
