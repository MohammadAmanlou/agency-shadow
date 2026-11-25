import unittest,tempfile,csv
from pathlib import Path
import numpy as np
from src.reactivity import contrasts,box_contrasts,factorial_ci,masking_decision,sharp_missing_clean,necessary_clean_cost,hybrid_paired,ppi,assignment,analyze
class TestReactivity(unittest.TestCase):
    def test_masking(self):np.testing.assert_allclose(contrasts([.7,.5,.75,.85]),[-.2,.1,.3])
    def test_box(self):
        p=np.array([.7,.5,.75,.85]);l,u=box_contrasts(p,p);np.testing.assert_allclose(l,u);self.assertTrue(masking_decision(l,u))
    def test_no_clean(self):self.assertEqual(sharp_missing_clean([.75,.85]),(-1,1))
    def test_restricted(self):np.testing.assert_allclose(sharp_missing_clean([.7,.8],.05,.05),[0,.2],atol=1e-10)
    def test_cost(self):self.assertGreater(necessary_clean_cost(.75,.5),0)
    def test_boundary(self):
        l,u=factorial_ci([0,0,10,10],10);self.assertTrue(np.all(l<=0));self.assertTrue(np.all(u>=0))
    def test_hybrid_symmetry(self):
        l,u=hybrid_paired(np.array([50,20,30,100]));a,b=hybrid_paired(np.array([50,30,20,100]));self.assertAlmostEqual(l,-b);self.assertAlmostEqual(u,-a)
    def test_ppi_constant(self):
        y=np.array([0,1]*100);l,u=ppi(y,np.full(200,.5),np.full(1000,.5));self.assertAlmostEqual((l+u)/2,.5)
    def test_assignment_analysis(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.csv';assignment(p,12);f=p.open();rows=list(csv.DictReader(f));f.close()
            for i,r in enumerate(rows):r['terminal_correct']='1' if i%2 else '';r['status']='completed' if i%2 else 'missing'
            with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
            x=analyze(p);self.assertEqual(x['n'],[3]*4);self.assertEqual(sum(x['missing']),6)
