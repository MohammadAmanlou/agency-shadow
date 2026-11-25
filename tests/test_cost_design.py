import unittest
from src.cost_design import design, minimax_lower

class TestCostDesign(unittest.TestCase):
    def test_cheaper_anchor(self):
        i,n,c=design([20,4,1],[0,.02,.12],.08,.05)
        self.assertEqual(i,1)
        self.assertLess(c,design([20],[0],.08,.05)[2])
    def test_exclude_excess_bias(self):
        self.assertEqual(design([100,1],[0,.08],.08,.05)[0],0)
    def test_no_feasible(self):
        with self.assertRaises(ValueError):design([1],[.2],.08,.05)
    def test_lower_upper(self):
        c,d=[20,4,1],[0,.02,.12]
        self.assertLess(minimax_lower(c,d,.08,.05),design(c,d,.08,.05)[2])

if __name__=='__main__':unittest.main()
