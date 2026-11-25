import unittest
import numpy as np
from src.transport import certificate,ci

class TestTransport(unittest.TestCase):
    def test_shared_bias_contrast_identified(self):
        D=certificate([.7,.8],[.7,.8],[(0,1,0)])
        self.assertAlmostEqual(D[0,1],.1);self.assertAlmostEqual(-D[1,0],.1)
        self.assertAlmostEqual(D[2,0]+D[0,2],.9)
    def test_anchor_and_path(self):
        D=certificate([.7,.8,.75],[.7,.8,.75],[(0,1,.02),(1,2,.03)],{0:(.5,.5)})
        self.assertAlmostEqual(D[3,2],.60);self.assertAlmostEqual(-D[2,3],.50)
        self.assertAlmostEqual(D[0,2],.10)
    def test_inconsistent(self):
        with self.assertRaises(ValueError):certificate([.7,.8],[.7,.8],[(0,1,0)],{0:(.5,.5),1:(.1,.1)})
    def test_endpoints_feasible(self):
        D=certificate([.7,.8,.75],[.7,.8,.75],[(0,1,.02),(1,2,.03)],{0:(.5,.5)})
        for i in range(4):
            x=D[i,:]-D[i,3]
            self.assertTrue(np.all(x[None,:]-x[:,None]<=D+1e-9))
            self.assertAlmostEqual(x[3],0)
    def test_ci(self):
        l,u=ci([0,1],100,.05);self.assertEqual(l[0],0);self.assertEqual(u[1],1)

if __name__=='__main__':unittest.main()
