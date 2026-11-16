"""固定配额的本地前瞻实验；仅对本机回环服务发出 72 次生成请求。"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
import subprocess
import time

import cmc_revision_20261106 as revision
import reconstruct_paper2_20261013 as base
from cmc_offline_20261014 import dump
from reconstruction_diagnostics_20261013 import all_witnesses

REPO='https://ollama.com/library/qwen3:8b'
MANIFEST_SHA='500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41'
SHA='a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f'
SEEDS=(20261106,20261107,20261108)
PARAMS=dict(n_predict=192,temperature=0.3,top_p=0.95,top_k=40,min_p=0.,repeat_penalty=1.,
            cache_prompt=True,stop=['<|im_end|>','<|endoftext|>'])
OUT=base.ARTIFACTS/'cmc-prospective-20261106'
DIAG_OUT=base.ARTIFACTS/'cmc-prospective-grammar-20261106'
SOURCE=revision.OUT/'benchmark.json'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def parse_proposal(raw,allowed):
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:
                raise ValueError('duplicate_role')
            result[k]=v
        return result
    try:
        obj=json.loads(raw,object_pairs_hook=unique)
    except json.JSONDecodeError as e:
        raise ValueError('invalid_json') from e
    if not isinstance(obj,dict) or set(obj)!=set(allowed):
        raise ValueError('role_keys')
    if any(not isinstance(obj[r],str) for r in allowed):
        raise ValueError('nonstring_id')
    if any(obj[r] not in allowed[r] for r in allowed):
        raise ValueError('ineligible_id')
    return tuple(obj[r] for r in allowed)


def eligibility_grammar(allowed):
    """每个角色的词法域由可选目录条目构成，不读取解或兼容评分。"""
    quote=lambda s:json.dumps(s,ensure_ascii=False)
    fields=[f'{quote(quote(role))} ws ":" ws role-{i}' for i,role in enumerate(allowed)]
    rules=['root ::= "{" ws '+(' ws "," ws '.join(fields))+' ws "}"',
           'ws ::= [ \t\n\r]*'.replace('\t',r'\t').replace('\n',r'\n').replace('\r',r'\r')]
    for i,ids in enumerate(allowed.values()):
        assert ids
        rules.append(f'role-{i} ::= '+' | '.join(quote(quote(x)) for x in sorted(ids)))
    return '\n'.join(rules)


def component(row):
    return base.ComponentTemplate(**{**row,'ports':tuple(base.TypedPort(**p) for p in row['ports']),
                                     'evidence':base.EvidenceRecord(**row['evidence']),
                                     'constraints':tuple(row['constraints'])})


def prompt(t,items):
    desc = ('drive.capacity_kg >= conveyor.load_kg; positioner.payload_kg >= conveyor.load_kg; '
            'transmission.speed_rpm <= 2 * drive.speed_rpm.' if t.scene=='A' else
            'gearbox.input_speed_rpm <= motor.speed_rpm; coupling.torque_nm >= motor.torque_nm; '
            'coupling.bore_mm and shaft_support.bore_mm must each be >= motor.shaft_diameter_mm.')
    roles=', '.join(t.required_roles)
    out=[
        'Select exactly one NEW component identifier per role to add to the existing shortlists.',
        'All existing shortlist members are retained. The expanded pool should contain at least one valid assignment.',
        'An assignment uses one candidate per role, meeting all numeric and interface constraints.',
        'Role order: '+roles+'. Between each consecutive pair, an outgoing port must match an incoming port in type and unit.',
        'Cross-role numeric constraints: '+desc,
        'Individual bounds (role.field.min/max): '+json.dumps(t.numeric_requirements,sort_keys=True),
        'Existing shortlists (do not propose these IDs again): '+json.dumps(t.candidate_ids),
        'Catalog rows are identifier; input ports; output ports; numeric fields. Ports use type@unit.',
    ]
    for r in t.required_roles:
        cs=[c for c in items if c.role==r]
        fields=sorted(cs[0].parameters)
        out.append(r+' numeric fields in order: '+', '.join(fields))
        for c in cs:
            ports=[','.join(f'{p.port_type}@{p.unit}' for p in c.ports if p.direction==d) or '-'
                   for d in ('in','out')]
            out.append(';'.join((c.template_id,*ports,','.join(f'{c.parameters[f]:g}' for f in fields))))
    out.append('Return only one JSON object in this shape, replacing each placeholder with a new ID from its own role: '
               +json.dumps({r:'NEW_ID' for r in t.required_roles})+'. Do not include markdown or explanation.')
    return ('<|im_start|>system\nYou are a component shortlist selector. Follow the exact JSON output contract.'
            '<|im_end|>\n<|im_start|>user\n'+'\n'.join(out)+'<|im_end|>\n<|im_start|>assistant\n'
            '<think>\n\n</think>\n\n')


def prepare():
    records=json.loads(SOURCE.read_text(encoding='utf-8'))
    typed=[r for r in records if r['case']=='typed']
    assert Counter(r['task_record']['scene'] for r in typed)=={'A':12,'B':12}
    for row in typed:
        t=base._coerce_task(row['task_record'])
        cs=[component(c) for c in row['catalog']]
        w,by=all_witnesses(t,cs)
        assert len(w)==row['full_solutions'] and len(w)>0 and not t.injected_conflict
        initial=tuple(x for r in t.required_roles for x in t.candidate_ids[r])
        allowed={r:{c.template_id for c in by[r]}-set(initial) for r in t.required_roles}
        yield t,cs,w,by,initial,allowed,prompt(t,cs)


def metrics(t,cs,w,by,ids):
    witness,recall,solutions=revision.retention(w,by,ids)
    sol=base.solve_subgraph(t,ids,cs)
    assert sol.feasible==bool(witness)
    return dict(feasible=witness,candidate_recall=recall,solution_fraction=solutions,search_nodes=sol.search_nodes)


def summarize(rows,controls):
    groups={t['task_id']:[] for t in controls}
    for r in rows:
        groups[r['task_id']].append(r)
    assert len(rows)==72 and all(sorted(r['seed'] for r in rr)==list(SEEDS) for rr in groups.values())
    tasks=[dict(**c,llm_majority=base.majority([r['feasible'] for r in groups[c['task_id']]])) for c in controls]
    def aggregate(rr):
        valid=[r for r in rr if r['status']=='valid']
        return dict(trials=len(rr),status=dict(Counter(r['status'] for r in rr)),
                    feasible=sum(r['feasible'] for r in rr),valid_no_witness=sum(not r['feasible'] for r in valid),
                    valid_recall=mean(r['candidate_recall'] for r in valid) if valid else None,
                    valid_solution_fraction=mean(r['solution_fraction'] for r in valid) if valid else None,
                    median_latency_s=median(r['latency_s'] for r in rr))
    a=[t['llm_majority'] for t in tasks]
    return dict(overall=aggregate(rows),by_scene={s:aggregate([r for r in rows if r['scene']==s]) for s in 'AB'},
                primary_support=base.paired(a,[t['Support']['feasible'] for t in tasks]),
                secondary_rule=base.paired(a,[t['Rule']['feasible'] for t in tasks]),
                initial_success=sum(t['Initial']['feasible'] for t in tasks),
                llm_recovers_initial_miss=sum(t['llm_majority'] and not t['Initial']['feasible'] for t in tasks),
                tasks=tasks)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--server',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--device',default='Vulkan0')
    p.add_argument('--port',type=int,default=17661)
    p.add_argument('--constrained-diagnostic',action='store_true')
    args=p.parse_args()
    import requests
    server=args.server.resolve(strict=True)
    model=args.model.resolve(strict=True)
    assert sha(model)==SHA,'model SHA-256 mismatch'
    cases=list(prepare())
    diagnostic=args.constrained_diagnostic
    out=DIAG_OUT if diagnostic else OUT
    seeds=SEEDS[:1] if diagnostic else SEEDS
    if diagnostic:
        assert json.loads((OUT/'verification.json').read_text(encoding='utf8'))['status']=='PASS'
    out.mkdir(exist_ok=False)
    args_cli=[str(server),'-m',str(model),'--host','127.0.0.1','--port',str(args.port),
              '-c','8192','-np','1','-ngl','99','--device',args.device,'-t','6','-b','512']
    version=subprocess.run([str(server),'--version'],cwd=server.parent,capture_output=True,text=True,check=True)
    dump(out/'protocol.json',dict(repository=REPO,manifest_sha256=MANIFEST_SHA,weight_sha256=SHA,
         model='Qwen3-8B',quantization='Q4_K_M',thinking=False,engine_sha256=sha(server),
         engine_version='; '.join(line.strip() for line in (version.stdout+version.stderr).splitlines()
                                  if line.startswith(('version:','built with'))),hardware_device=args.device,
         generation=PARAMS,seeds=seeds,benchmark_sha256=sha(SOURCE),trials=len(cases)*len(seeds),context=8192,
         experimental_request_limit=len(cases)*len(seeds),remote_model_calls=0,quota_per_role=1,
         posthoc_grammar_diagnostic=diagnostic))
    dump(out/'prompts.json',{t.task_id:txt for t,_,_,_,_,_,txt in cases})
    if diagnostic:
        dump(out/'grammars.json',{t.task_id:eligibility_grammar(allowed) for t,_,_,_,_,allowed,_ in cases})
    address=f'http://127.0.0.1:{args.port}'
    http=requests.Session()
    http.trust_env=False
    log=(out/'local_engine.log').open('wb')
    proc=subprocess.Popen(args_cli,cwd=server.parent,stdout=log,stderr=subprocess.STDOUT,
                          creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        for _ in range(90):
            if proc.poll() is not None:
                raise RuntimeError('local engine failed; inspect local_engine.log')
            try:
                if http.get(address+'/health',timeout=1).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            raise TimeoutError('local engine not ready')
        for t,_,_,_,_,_,txt in cases:
            count=http.post(address+'/tokenize',json={'content':txt,'add_special':True,'parse_special':True},timeout=20)
            count.raise_for_status()
            assert len(count.json()['tokens'])+PARAMS['n_predict']<8192,t.task_id
        rows=[];controls=[]
        with (out/'trials.jsonl').open('x',encoding='utf-8',newline='\n') as ledger:
            for t,cs,w,by,initial,allowed,txt in cases:
                target=revision.quota_target(t,cs,1)
                c=dict(task_id=t.task_id,scene=t.scene,Initial=metrics(t,cs,w,by,initial))
                for m in ('Support','Rule'):
                    ids=revision.select(t,cs,target,m,1)
                    c[m]=dict(ids=ids,**metrics(t,cs,w,by,ids))
                controls.append(c)
                for seed in seeds:
                    row=dict(task_id=t.task_id,scene=t.scene,seed=seed,feasible=0)
                    start=time.perf_counter()
                    try:
                        request=dict(prompt=txt,seed=seed,**PARAMS)
                        if diagnostic:
                            request['grammar']=eligibility_grammar(allowed)
                        response=http.post(address+'/completion',json=request,timeout=150)
                        row.update(http_status=response.status_code,raw_response=response.text)
                        response.raise_for_status()
                        answer=response.json()
                        row['output']=answer['content']
                        if answer.get('truncated'):
                            raise ValueError('truncated_context')
                        ids=parse_proposal(answer['content'],allowed)
                    except requests.RequestException:
                        row['status']='request_error'
                    except (ValueError,KeyError) as e:
                        row['status']=str(e) if str(e) in {'invalid_json','duplicate_role','role_keys',
                            'nonstring_id','ineligible_id','truncated_context'} else 'malformed_response'
                    else:
                        row.update(status='valid',additions=ids,**metrics(t,cs,w,by,(*initial,*ids)))
                    row['latency_s']=time.perf_counter()-start
                    ledger.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    ledger.flush()
                    rows.append(row)
                print('Task complete',t.task_id,'trials',len(rows),flush=True)
        if diagnostic:
            assert len(rows)==24 and len({r['task_id'] for r in rows})==24
            stats={s:dict(trials=len(rr),valid=sum(r['status']=='valid' for r in rr),
                          feasible=sum(r['feasible'] for r in rr))
                   for s in 'AB' for rr in [[r for r in rows if r['scene']==s]]}
            dump(out/'summary.json',dict(by_scene=stats,valid=sum(r['status']=='valid' for r in rows),
                                         feasible=sum(r['feasible'] for r in rows),trials=len(rows)))
        else:
            dump(out/'summary.json',summarize(rows,controls))
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        log.close()


if __name__=='__main__':
    main()
