"""从已归档的原始推理响应独立重算，不启动模型。"""
import json
import math
from collections import Counter
from statistics import median

import cmc_prospective_20261106 as p
import cmc_revision_20261106 as revision


def main():
    protocol=json.loads((p.OUT/'protocol.json').read_text(encoding='utf8'))
    assert protocol['benchmark_sha256']==p.sha(p.SOURCE)
    assert protocol['weight_sha256']==p.SHA and protocol['manifest_sha256']==p.MANIFEST_SHA
    assert protocol['generation']==p.PARAMS and protocol['seeds']==list(p.SEEDS)
    prompts=json.loads((p.OUT/'prompts.json').read_text(encoding='utf8'))
    trials=[json.loads(l) for l in (p.OUT/'trials.jsonl').read_text(encoding='utf8').splitlines()]
    assert len(trials)==72 and len({(r['task_id'],r['seed']) for r in trials})==72
    groups={}
    for r in trials:
        groups.setdefault(r['task_id'],[]).append(r)
    controls=[];engine_ms=[];token_counts=[];violations=Counter();responses_with_initial=0
    diag_protocol=json.loads((p.DIAG_OUT/'protocol.json').read_text(encoding='utf8'))
    diag_prompts=json.loads((p.DIAG_OUT/'prompts.json').read_text(encoding='utf8'))
    grammar=json.loads((p.DIAG_OUT/'grammars.json').read_text(encoding='utf8'))
    diag_rows=[json.loads(l) for l in (p.DIAG_OUT/'trials.jsonl').read_text(encoding='utf8').splitlines()]
    diag={r['task_id']:r for r in diag_rows}
    assert len(diag_rows)==len(diag)==24 and diag.keys()==groups.keys() and diag_prompts==prompts
    assert diag_protocol['posthoc_grammar_diagnostic'] and diag_protocol['seeds']==[p.SEEDS[0]]
    for k in ('weight_sha256','manifest_sha256','engine_sha256','benchmark_sha256','generation','quota_per_role'):
        assert diag_protocol[k]==protocol[k],k
    for t,cs,w,by,initial,allowed,txt in p.prepare():
        assert prompts[t.task_id]==txt
        c=dict(task_id=t.task_id,scene=t.scene,Initial=p.metrics(t,cs,w,by,initial))
        for m in ('Support','Rule'):
            ids=revision.select(t,cs,revision.quota_target(t,cs,1),m,1)
            c[m]=dict(ids=ids,**p.metrics(t,cs,w,by,ids))
        controls.append(c)
        for row in groups[t.task_id]:
            if row.get('http_status')!=200:
                assert row['status']=='request_error' and row['feasible']==0
                continue
            obj=json.loads(row['raw_response'])
            assert row['output']==obj['content']
            if row['status']=='ineligible_id':
                answer=json.loads(row['output'])
                roles={c.template_id:c.role for c in cs}
                reasons=['initial_id' if v in initial and roles.get(v)==k else
                         'wrong_role' if v in roles else 'unknown_id'
                         for k,v in answer.items() if v not in allowed[k]]
                violations.update(reasons)
                responses_with_initial+=int('initial_id' in reasons)
            if 'timings' in obj:
                tm=obj['timings']
                engine_ms.append(tm['prompt_ms']+tm['predicted_ms'])
                token_counts.append(obj['tokens_predicted'])
            try:
                if obj.get('truncated'):
                    raise ValueError('truncated_context')
                ids=p.parse_proposal(obj['content'],allowed)
            except ValueError as e:
                assert row['status']==str(e) and row['feasible']==0
            else:
                assert row['status']=='valid' and row['additions']==list(ids)
                for k,v in p.metrics(t,cs,w,by,(*initial,*ids)).items():
                    assert row[k]==v
        d=diag[t.task_id]
        assert d['seed']==p.SEEDS[0] and grammar[t.task_id]==p.eligibility_grammar(allowed)
        obj=json.loads(d['raw_response'])
        assert d['http_status']==200 and not obj.get('truncated') and d['output']==obj['content']
        ids=p.parse_proposal(obj['content'],allowed)
        assert d['status']=='valid' and d['additions']==list(ids)
        for k,v in p.metrics(t,cs,w,by,(*initial,*ids)).items():
            assert d[k]==v
    actual=p.summarize(trials,controls)
    expected=json.loads((p.OUT/'summary.json').read_text(encoding='utf8'))
    assert json.loads(json.dumps(actual))==expected
    for k in ('primary_support','secondary_rule'):
        r=actual[k];b=r['a_only'];c=r['b_only'];n=b+c
        tail=sum(math.comb(n,j) for j in range(min(b,c)+1))/2**n
        assert r['p_exact']==min(1.,2*tail)
    by_task={}
    for s in 'AB':
        rs=[r for r in actual['tasks'] if r['scene']==s]
        by_task[s]=dict(initial=sum(r['Initial']['feasible'] for r in rs),
                        model=sum(r['llm_majority'] for r in rs),
                        rule=sum(r['Rule']['feasible'] for r in rs),
                        support=sum(r['Support']['feasible'] for r in rs))
    diagnosis=json.loads((p.DIAG_OUT/'summary.json').read_text(encoding='utf8'))
    assert diagnosis['trials']==24 and diagnosis['valid']==24
    assert diagnosis['feasible']==sum(r['feasible'] for r in diag_rows)
    for s in 'AB':
        rr=[r for r in diag_rows if r['scene']==s]
        assert diagnosis['by_scene'][s]==dict(trials=len(rr),valid=len(rr),feasible=sum(r['feasible'] for r in rr))
    result=dict(status='PASS',trials=72,unique_keys=72,exact_response_and_metrics_replay=True,
                task_majority_by_scene=by_task,model_output_status=dict(Counter(r['status'] for r in trials)),
                median_engine_ms=median(engine_ms) if engine_ms else None,
                engine_timing_records=len(engine_ms),
                generated_tokens=sum(token_counts) if token_counts else None,
                ineligible_role_assignments=dict(violations),
                responses_repeating_initial_ids=responses_with_initial,
                exploratory_grammar=diagnosis,
                free_first_seed_feasible=sum(r['feasible'] for r in trials if r['seed']==p.SEEDS[0]),
                grammar_only_success=sum(diag[r['task_id']]['feasible'] and not r['feasible']
                                         for r in trials if r['seed']==p.SEEDS[0]),
                diagnostic_file_sha256={n:p.sha(p.DIAG_OUT/n) for n in ('protocol.json','prompts.json','grammars.json','trials.jsonl','summary.json')},
                live_calls_for_verification=0,
                file_sha256={n:p.sha(p.OUT/n) for n in ('protocol.json','prompts.json','trials.jsonl','summary.json')})
    p.dump(p.OUT/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
