"""冻结归档的补充离线分析；不加载在线客户端或调用模型。"""
import json
import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import replace
from statistics import mean, median
from time import perf_counter
import numpy as np
import reconstruct_paper2_20261013 as base
from reconstruction_diagnostics_20261013 import all_witnesses
from cmc_offline_20261014 import pair_compatible, relation_edges, support_pool, AUDIT, dump

OUT = base.ARTIFACTS / 'cmc-revision-20261106'
METHODS = ('Random','Rule','Lexical','Support')
QUOTAS = (1,3,6)
SEED = 20261106


def quota_target(task, items, q):
    target = []
    for r in task.required_roles:
        initial = task.candidate_ids[r]
        rest = sorted(c.template_id for c in items if c.role == r and c.template_id not in initial)
        if not 0 <= q <= len(rest):
            raise ValueError('quota unavailable')
        target.extend((*initial,*rest[:q]))
    return tuple(target)


def retention(w, by, ids):
    if not len(w):
        return 0,None,None
    keep = set(ids)
    hits = np.ones(len(w),dtype=bool)
    recall_n = recall_d = 0
    for j, cs in enumerate(by.values()):
        rel = np.unique(w[:,j])
        selected = np.array([c.template_id in keep for c in cs])
        recall_n += int(selected[rel].sum())
        recall_d += len(rel)
        hits &= selected[w[:,j]]
    return int(hits.any()), recall_n/recall_d, float(hits.mean())


def select(t, items, target, m, rep):
    if m == 'Support':
        return support_pool(t,items,target,relation_edges(t),lambda a,b:pair_compatible(t,a,b))[0]
    return base.matched_pool(t,items,target,m,rep,SEED)


def measure(rows,t,items,w,by,ids,m,rep,case,quota,selection_ms):
    start = perf_counter()
    result = base.solve_subgraph(t,ids,items)
    solve_ms = (perf_counter()-start)*1000
    y,recall,fraction = retention(w,by,ids)
    assert y == result.feasible,(t.task_id,case,m)
    rows.append(dict(task=t.task_id,scene=t.scene,stratum=t.task_type,case=case,method=m,
        quota=quota,repeat=rep,full_solutions=len(w),feasible=y,recall=recall,solutions=fraction,
        pruned=1-len(ids)/len(items),nodes=result.search_nodes,selection_ms=selection_ms,solve_ms=solve_ms))


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in ('case','scene','stratum','method','quota'))].append(r)
    out=[]
    for key,rr in sorted(groups.items()):
        f=[r for r in rr if r['full_solutions']]
        task=defaultdict(list)
        for r in f:
            task[r['task']].append(r['feasible'])
        out.append(dict(zip(('case','scene','stratum','method','quota'),key),
            n=len(rr),eligible=len(task),feasible=sum(r['feasible'] for r in rr),
            witness_majority=mean(sum(v)>len(v)/2 for v in task.values()) if task else None,
            recall=mean(r['recall'] for r in f) if f else None,
            solutions=mean(r['solutions'] for r in f) if f else None,
            pruned=mean(r['pruned'] for r in rr),nodes=mean(r['nodes'] for r in rr),
            selection_ms=median(r['selection_ms'] for r in rr),
            solve_ms=median(r['solve_ms'] for r in rr)))
    return out


def audit(records):
    rows=json.loads((AUDIT/'p_attempts_sanitized.json').read_text(encoding='utf-8'))
    task=defaultdict(list);stage=defaultdict(list)
    for r in sorted(rows,key=lambda x:x['source_row']):
        key=(r['task_id'],r['repeat'])
        task[key].append(r)
        stage[(*key,r['task_attempt'],r['stage'])].append(r)
    final=[r for (t,m,j),r in records.items() if m=='P']
    assert set(task)=={(r['task_id'],r['repeat']) for r in final}
    return dict(requests=len(rows),request_failures=sum(r['status']!='success' for r in rows),
        task_keys=len(task),first_retained_request_success=sum(v[0]['status']=='success' for v in task.values()),
        stage_attempt_keys=len(stage),first_retained_stage_request_success=sum(v[0]['status']=='success' for v in stage.values()),
        successful_terminal=sum(r['status']=='success' for r in final),terminal_feasible=sum(r['feasible'] for r in final),
        terminal_field_names=sorted(set().union(*(r.keys() for r in final))))


def heterogeneous(t,items,disconnected):
    counts=Counter(); result=[]
    for c in items:
        family=counts[c.role]%4
        counts[c.role]+=1
        ports=tuple(replace(p,port_type=f'{p.port_type}/family-{family}'
            + ('/disconnected' if disconnected and c.role==t.required_roles[-1] and p.direction=='in' else ''))
            for p in c.ports)
        result.append(replace(c,ports=ports))
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    base.OUT=OUT
    cat,tasks,records=base.load_inputs()
    rows=[]; orders=[]; benchmarks=[]
    for i,(tid,t) in enumerate(sorted(tasks.items())):
        items=[c for c in cat if c.scene==t.scene]
        w,by=all_witnesses(t,items)
        full=tuple(c.template_id for c in items)
        for rep in (1,2,3):
            target=records[tid,'P',rep]['activated_ids']
            for m in ('P','Full',*METHODS):
                start=perf_counter()
                ids=target if m=='P' else full if m=='Full' else select(t,items,target,m,rep)
                selection=(perf_counter()-start)*1000
                measure(rows,t,items,w,by,ids,m,rep,'matched',-1,selection)
            for q in QUOTAS:
                fixed=quota_target(t,items,q)
                for m in METHODS:
                    start=perf_counter()
                    ids=select(t,items,fixed,m,rep)
                    selection=(perf_counter()-start)*1000
                    measure(rows,t,items,w,by,ids,m,rep,'fixed',q,selection)
        if len(w):
            for seed in (0,1,2,3,4,5):
                ordered=list(items)
                if seed:
                    random.Random(base.seed_for(SEED,tid,seed)).shuffle(ordered)
                sol=base.solve_subgraph(t,full,ordered)
                assert sol.feasible
                orders.append(dict(task=tid,scene=t.scene,stratum=t.task_type,seed=seed,nodes=sol.search_nodes))
        if (i+1)%50==0:
            print('Historical tasks',i+1,flush=True)
    for scene in ('A','B'):
        selected=[t for _,t in sorted(tasks.items()) if t.scene==scene and t.task_type=='configuration'][:12]
        for t in selected:
            original=[c for c in cat if c.scene==scene]
            for disconnected in (False,True):
                case='disconnected' if disconnected else 'typed'
                items=heterogeneous(t,original,disconnected)
                assert not t.injected_conflict
                w,by=all_witnesses(t,items)
                if disconnected:
                    assert not len(w)
                full=tuple(c.template_id for c in items)
                assert base.solve_subgraph(t,full,items,typed=False).feasible
                measure(rows,t,items,w,by,full,'Full',1,case,-1,0.)
                benchmarks.append(dict(task=t.task_id,case=case,full_solutions=len(w),task_record=t.to_dict(),
                    catalog=[c.to_dict() for c in items]))
                for q in QUOTAS:
                    fixed=quota_target(t,items,q)
                    for m in METHODS:
                        for rep in (1,2,3):
                            start=perf_counter()
                            ids=select(t,items,fixed,m,rep)
                            measure(rows,t,items,w,by,ids,m,rep,case,q,(perf_counter()-start)*1000)
        print('Typed benchmark',scene,flush=True)
    for name,data in [('runs.csv',rows),('ordering.csv',orders)]:
        base.csvsave(name,data)
    dump(OUT/'benchmark.json',benchmarks)
    dump(OUT/'summary.json',summarize(rows))
    dump(OUT/'audit.json',audit(records))
    manifest=json.loads((OUT/'input_manifest.json').read_text(encoding='utf-8'))
    manifest.update(analysis_date='2026-11-06',seed=SEED,quotas=list(QUOTAS),live_model_calls=0,
        policy='Historical matched replay plus independently fixed offline quotas and constructed interface stress cases.')
    a=AUDIT/'p_attempts_sanitized.json'
    manifest['files'][a.relative_to(base.ROOT).as_posix()]=hashlib.sha256(a.read_bytes()).hexdigest()
    dump(OUT/'input_manifest.json',manifest)
    print('Completed',len(rows),len(orders),len(benchmarks),flush=True)

if __name__=='__main__':
    main()
