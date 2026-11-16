"""修正后本地审计：独立真值、冻结标签差异及全部历史候选池重放。"""
import hashlib
import json
from pathlib import Path
import reconstruct_paper2_20261013 as base
from reconstruction_diagnostics_20261013 import all_witnesses
from paper2_dynamic_agents.experiment.dataset import evaluate_truth, build_tasks
from paper2_dynamic_agents.experiment.catalog import build_catalog
from paper2_dynamic_agents.experiment.solver import solve_subgraph


def main():
    out = base.ARTIFACTS / 'software-correction-20261016'
    out.mkdir(parents=True, exist_ok=True)
    # load_inputs 写清单，重定向到本次审计目录，不覆盖旧分析的清单。
    saved_out = base.OUT
    base.OUT = out
    try:
        catalog, tasks, records = base.load_inputs()
    finally:
        base.OUT = saved_out
    derived = []
    mismatches = []
    solution_set_mismatches = []
    for index, (tid, task) in enumerate(sorted(tasks.items()), 1):
        truth = evaluate_truth(task, catalog)
        items = [c for c in catalog if c.scene == task.scene]
        solver = solve_subgraph(task, task.candidate_ids, catalog)
        witnesses, _ = all_witnesses(task, items)
        assert truth.feasible == solver.feasible, tid
        if truth.full_catalog_solution_count != len(witnesses):
            solution_set_mismatches.append(tid)
        derived.append(dict(task_id=tid, frozen_expected_feasible=task.expected_feasible,
                            corrected_feasible=truth.feasible,
                            label_changed=task.expected_feasible != truth.feasible,
                            forced_negative=task.injected_conflict == 'global_no_solution',
                            full_catalog_solution_count=truth.full_catalog_solution_count))
        if index % 50 == 0:
            print('truth audit', index, flush=True)
    assert not solution_set_mismatches, solution_set_mismatches
    for (tid, method, rep), record in sorted(records.items()):
        task = tasks[tid]
        ids = ([c.template_id for c in catalog if c.scene == task.scene] if method == 'B0'
               else record['activated_ids'])
        result = solve_subgraph(task, ids, catalog, typed=method != 'P-Typed')
        if result.feasible != bool(record['feasible']):
            mismatches.append(dict(task_id=tid, method=method, repeat=rep,
                                   archived=bool(record['feasible']), replay=result.feasible))
    assert not mismatches, mismatches[:20]
    # 修正只改变生成器的错误标签；参数、任务 ID 和候选池必须仍与冻结输入相同。
    generated = {}
    for scene in ('A', 'B'):
        rebuilt = build_tasks(scene, 'test', 200, 20260816)
        changed = []
        for task in rebuilt:
            old = tasks[task.task_id].to_dict()
            new = task.to_dict()
            if old.pop('expected_feasible') != new.pop('expected_feasible'):
                changed.append(task.task_id)
            assert old == new, task.task_id
        generated[scene] = dict(tasks=len(rebuilt), label_changes=changed)
        print('generator audit', scene, len(rebuilt), flush=True)
    assert [c.to_dict() for c in build_catalog(20260816)] == [c.to_dict() for c in catalog]
    code_names = ['src/paper2_dynamic_agents/experiment/' + n + '.py' for n in ('dataset', 'mechanism', 'solver')]
    code_names += ['scripts/' + n + '.py' for n in ('artifact_paths', 'reconstruction_diagnostics_20261013',
                   'audit_software_contracts_20261016')]
    report = dict(status='PASS', live_model_calls=0, frozen_tasks=len(tasks),
                  historical_records_replayed=len(records), historical_feasibility_mismatches=mismatches,
                  independent_full_solution_count_mismatches=solution_set_mismatches,
                  frozen_label_changes=[r['task_id'] for r in derived if r['label_changed']],
                  forced_negative_controls=sum(r['forced_negative'] for r in derived),
                  generator_reconstruction=generated,
                  code_sha256={n: hashlib.sha256((base.ROOT/n).read_bytes()).hexdigest() for n in code_names},
                  scope='本地正确性验证；不证明模型优越、物理可行、最优性或线上澄清能力')
    for name, obj in [('corrected_truth_labels.json', derived), ('software_verification.json', report)]:
        (out/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('SOFTWARE_AUDIT_PASS', len(tasks), len(records), 'label changes', len(report['frozen_label_changes']), flush=True)


if __name__ == '__main__':
    main()
