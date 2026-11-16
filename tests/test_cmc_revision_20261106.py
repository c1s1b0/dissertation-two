"""返修实验的手算口径核验。"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as N
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cmc_revision_20261106 import quota_target, retention


class RevisionTests(unittest.TestCase):
    def test_quota_keeps_initial_and_fills_each_role(self):
        t = N(required_roles=('a','b'), candidate_ids={'a':('a2',),'b':('b2',)})
        cs = [N(role=x[0],template_id=x) for x in ('a1','a2','b1','b2')]
        self.assertEqual(set(quota_target(t,cs,1)), {'a1','a2','b1','b2'})
        with self.assertRaises(ValueError):
            quota_target(t,cs,2)

    def test_recall_and_witness_fraction_are_distinct(self):
        by = {'a':[N(template_id='a1'),N(template_id='a2')],
              'b':[N(template_id='b1'),N(template_id='b2')]}
        w = np.array([[0,0],[1,1]])
        self.assertEqual(retention(w,by,('a1','b1')), (1,0.5,0.5))
        self.assertEqual(retention(w,by,('a1','b2')), (0,0.5,0.0))
        self.assertEqual(retention(w[:0],by,('a1','b1')), (0,None,None))

if __name__ == '__main__':
    unittest.main()
