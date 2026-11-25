import unittest,tempfile
from src.llm_run import run
from src.benchmark import answer_rule
class TestRunner(unittest.TestCase):
    def test_mock_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            r=run('mock','mock',d,2,answer_fn=answer_rule);self.assertEqual(r['all']['n'],16);self.assertFalse(r['real_model_inference']);self.assertEqual(r['both_correct_and_certified_fraction'],1.)
            def fail(c):raise RuntimeError('should resume')
            r=run('mock','mock',d,2,answer_fn=fail);self.assertEqual(r['all']['errors'],0)
    def test_errors_visible(self):
        with tempfile.TemporaryDirectory() as d:
            def fail(c):raise RuntimeError('no endpoint')
            r=run('mock','mock',d,1,answer_fn=fail);self.assertEqual(r['all']['errors'],8);self.assertEqual(r['all']['accuracy_all'],0.)
