"""手工可判定的组件契约回归；不访问模型或网络。"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from paper2_dynamic_agents.experiment.schema import ComponentTemplate, ConfigurationTask, EvidenceRecord, TypedPort
from paper2_dynamic_agents.experiment.dataset import evaluate_truth
from paper2_dynamic_agents.experiment.mechanism import analyze_task, replay_pool
from paper2_dynamic_agents.experiment.solver import solve_subgraph


def fixture(scene='A'):
    """每角色唯一组件：所有边界及相邻端口兼容，恰有一个解。"""
    params = ({'drive': {'speed_rpm': 1000, 'capacity_kg': 100},
               'transmission': {'speed_rpm': 1000}, 'conveyor': {'load_kg': 10},
               'positioner': {'payload_kg': 100}} if scene == 'A' else
              {'motor': {'speed_rpm': 1000, 'torque_nm': 10, 'shaft_diameter_mm': 5},
               'gearbox': {'input_speed_rpm': 500}, 'coupling': {'torque_nm': 20, 'bore_mm': 6},
               'shaft_support': {'bore_mm': 6}})
    ports = (TypedPort('i', 'mechanical', 'in', 'mm'), TypedPort('o', 'mechanical', 'out', 'mm'))
    evidence = EvidenceRecord(0, 'hand-checked regression', True, 'mixed', '20261016')
    catalog = [ComponentTemplate(r, scene, r, p, {}, ports, (), evidence) for r, p in params.items()]
    task = ConfigurationTask('fixture-' + scene, scene, 'test', 0, 'configuration', '',
                             tuple(params), {r: (r,) for r in params}, {})
    return task, catalog


class SoftwareContracts(unittest.TestCase):
    def test_compatible_singleton_has_exactly_one_solution(self):
        for scene in ('A', 'B'):
            t, cs = fixture(scene)
            truth = evaluate_truth(t, cs)
            self.assertTrue(truth.feasible)
            self.assertEqual(truth.full_catalog_solution_count, 1)

    def test_full_pool_preserves_untyped_flag(self):
        t, cs = fixture()
        cs[0] = replace(cs[0], ports=())
        self.assertFalse(replay_pool(t, 'full_pool', [c.template_id for c in cs], cs, True).feasible)
        self.assertTrue(replay_pool(t, 'full_pool', [c.template_id for c in cs], cs, False).feasible)

    def test_one_shot_history_keeps_observed_success_and_first_row(self):
        t, cs = fixture()
        rows = [{'status': 'success', 'feasible': True, 'activated_ids': [c.template_id for c in cs]}]
        result = analyze_task(t, iter(rows), iter(rows), None, iter(cs), {}, full_assignments=[])
        self.assertTrue(result['replays']['observed_p_feasible'])
        self.assertEqual(result['p_row'], rows[0])

    def test_tuple_history_keeps_first_row(self):
        t, cs = fixture()
        row = {'status': 'success', 'feasible': True, 'activated_ids': [c.template_id for c in cs]}
        self.assertEqual(analyze_task(t, (row,), (), None, cs, {}, [])['p_row'], row)

    def test_global_marker_overrides_otherwise_valid_assignment(self):
        for scene in ('A', 'B'):
            t, cs = fixture(scene)
            truth = evaluate_truth(replace(t, injected_conflict='global_no_solution'), cs)
            self.assertFalse(truth.feasible)
            self.assertEqual(truth.full_catalog_solution_count, 0)
            self.assertFalse(truth.eligible_for_feasibility_rate)

    def test_truth_rejects_port_type_unit_direction_and_missing_ports(self):
        for scene in ('A', 'B'):
            for ports in ((), (TypedPort('o', 'wrong', 'out', 'mm'),),
                          (TypedPort('o', 'mechanical', 'out', 'wrong'),),
                          (TypedPort('i', 'mechanical', 'in', 'mm'),)):
                with self.subTest(scene=scene, ports=ports):
                    t, cs = fixture(scene)
                    cs[0] = replace(cs[0], ports=ports)
                    self.assertFalse(evaluate_truth(t, cs).feasible)
                    self.assertEqual(evaluate_truth(t, cs).full_catalog_solution_count, 0)
                    self.assertTrue(solve_subgraph(t, t.candidate_ids, cs, typed=False).feasible)

    def test_truth_rejects_foreign_scene(self):
        t, cs = fixture()
        cs[0] = replace(cs[0], scene='B')
        self.assertFalse(evaluate_truth(t, cs).feasible)

    def test_truth_rejects_wrong_role_mapping_even_with_valid_alternative(self):
        t, cs = fixture()
        t = replace(t, candidate_ids={**t.candidate_ids, 'drive': ('drive', 'transmission')})
        self.assertFalse(solve_subgraph(t, t.candidate_ids, cs).feasible)
        self.assertFalse(evaluate_truth(t, cs).feasible)

    def test_missing_role_pool_is_infeasible_not_key_error(self):
        t, cs = fixture()
        t = replace(t, candidate_ids={r: ids for r, ids in t.candidate_ids.items() if r != 'drive'})
        self.assertFalse(evaluate_truth(t, cs).feasible)
        self.assertEqual(evaluate_truth(t, cs).full_catalog_solution_count, 1)

    def test_untyped_keeps_numeric_scene_role_and_marker_constraints(self):
        t, cs = fixture()
        cases = [replace(t, numeric_requirements={'drive.speed_rpm.min': 2000}),
                 replace(t, injected_conflict='global_no_solution'),
                 replace(t, candidate_ids={**t.candidate_ids, 'drive': ('transmission',)})]
        for case in cases:
            self.assertFalse(solve_subgraph(case, case.candidate_ids, cs, typed=False).feasible)
        cs[0] = replace(cs[0], scene='B')
        self.assertFalse(solve_subgraph(t, t.candidate_ids, cs, typed=False).feasible)

    def test_vector_solution_oracle_rejects_incompatible_ports(self):
        from paper2_dynamic_agents.experiment.catalog import build_catalog
        from reconstruction_diagnostics_20261013 import all_witnesses
        t, _ = fixture()
        cs = [replace(c, ports=()) if c.role == 'drive' else c
              for c in build_catalog(20260816) if c.scene == 'A']
        witnesses, _ = all_witnesses(t, cs)
        self.assertEqual(len(witnesses), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
