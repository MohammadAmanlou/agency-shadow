import unittest
from src.benchmark import generate,verify,answer_rule,evaluate

class TestBenchmark(unittest.TestCase):
    def test_identical_initial_evidence(self):
        pairs={}
        for c in generate(5):pairs.setdefault(c['pair'],[]).append(c)
        for a,b in pairs.values():
            self.assertEqual(sorted(a['initial_evidence'],key=lambda x:x['id']),sorted(b['initial_evidence'],key=lambda x:x['id']))
            self.assertNotEqual(a['truth'],b['truth'])
    def test_oracle(self):
        for c in generate(3):
            a=answer_rule(c);self.assertEqual(a['refund'],c['truth']);self.assertTrue(verify(c,a))
    def test_invalid_citation_and_wrong_label(self):
        c=generate(1)[0];a=answer_rule(c)
        self.assertFalse(verify(c,{**a,'refund':not a['refund']}))
        self.assertFalse(verify(c,{**a,'evidence_ids':['invented']}))
        self.assertFalse(verify(c,{**a,'evidence_ids':a['evidence_ids'][:-1]}))
    def test_balanced_constant(self):
        r=evaluate(generate(3),lambda c:{'refund':True,'evidence_ids':[]})
        self.assertEqual(r['all']['accuracy'],.5);self.assertEqual(r['all']['certified'],0.)
