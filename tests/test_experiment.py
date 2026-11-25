import itertools
import unittest
import numpy as np
from src.experiment import trajectory, sharp_interval, first_probe_hazard

class TestTheory(unittest.TestCase):
    def test_observational_equivalence(self):
        self.assertEqual(trajectory('stable',[1]*12),trajectory('erosion',[1]*12))
        self.assertAlmostEqual(trajectory('erosion',[0]*11+[1])[-1],.25)
        self.assertAlmostEqual(trajectory('stable',[0]*11+[1])[-1],.8)
    def test_sharp_bounds(self):
        obs={(1,0,0):.65,(0,1,0):.55}; w=np.array([.2,.1,.1]); l,u=sharp_interval(obs,w)
        self.assertAlmostEqual(l,.45);self.assertAlmostEqual(u,.65)
        # Construct valid Lipschitz extensions attaining both endpoints on full cube.
        histories=list(itertools.product((0,1),repeat=3));dist=lambda a,b:float(np.dot(w,np.abs(np.array(a)-np.array(b))))
        for v in (l,u):
            fixed={**obs,(0,0,0):v}
            ext={h:max(0,max(m-dist(h,x) for x,m in fixed.items())) for h in histories}
            for h,m in fixed.items():self.assertAlmostEqual(ext[h],m)
            for h,k in itertools.product(histories,repeat=2):self.assertLessEqual(abs(ext[h]-ext[k]),dist(h,k)+1e-10)
    def test_incompatible(self):
        with self.assertRaises(ValueError):sharp_interval({(0,):.1,(1,):.9},[.1])
    def test_hazard(self):self.assertAlmostEqual(first_probe_hazard(12,.5)[-1],1/4096)

if __name__=='__main__':unittest.main()
