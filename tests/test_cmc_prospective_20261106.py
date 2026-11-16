"""前瞻实验合同：不修复错误输出、不允许不等配额。"""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cmc_prospective_20261106 import parse_proposal, eligibility_grammar

class ProposalTests(unittest.TestCase):
    allowed={'motor':{'m1','m2'},'coupling':{'c1'}}

    def test_exact_single_addition(self):
        self.assertEqual(parse_proposal('{"motor":"m2","coupling":"c1"}',self.allowed),('m2','c1'))

    def test_untrusted_and_malformed_outputs(self):
        for raw in ('{}','{"motor":"m0","coupling":"c1"}',
                    '{"motor":"c1","coupling":"m1"}','{"motor":["m1","m2"],"coupling":"c1"}',
                    '{"motor":"m1","motor":"m2","coupling":"c1"}',
                    '{"motor":"m1","coupling":"c1","extra":"c1"}',
                    'explanation {"motor":"m1","coupling":"c1"}'):
            with self.subTest(raw=raw),self.assertRaises(ValueError):
                parse_proposal(raw,self.allowed)

    def test_grammar_uses_only_role_specific_eligible_ids(self):
        g=eligibility_grammar(self.allowed)
        self.assertIn(r'role-0 ::= "\"m1\"" | "\"m2\""',g)
        self.assertIn(r'role-1 ::= "\"c1\""',g)
        self.assertNotIn('m0',g)

if __name__=='__main__':
    unittest.main()
