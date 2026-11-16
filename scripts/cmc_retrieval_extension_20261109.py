"""按冻结取样复现检索基线、48 个新任务的本地对照和目录案例。"""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import product
import json
import math
from pathlib import Path
import random
import re
from statistics import mean
from time import perf_counter

import numpy as np

import cmc_fixed_first_20261109 as fixed
import cmc_prospective_20261106 as local
import cmc_revision_20261106 as revision
import reconstruct_paper2_20261013 as base
from cmc_offline_20261014 import dump
from reconstruction_diagnostics_20261013 import all_witnesses

OUT = base.ARTIFACTS / "cmc-retrieval-extension-20261109"
SEED = 2026110902
STRATA = list(product("AB", ("configuration", "repair")))
E5_REV = "ffb93f3bd4047442299a41ebb6fa998a38507c52"
KTR = "https://www.ktr.com/fileadmin/ktr/media/Tools_Downloads/Drive_technology_en.pdf"


def new_cases():
    prior = {r["task_id"] for r in base.jsonl(local.DIAG_OUT / "trials.jsonl")}
    paths = [base.BENCH / "tasks" / f"{s}_test.jsonl" for s in "AB"]
    tasks = [base._coerce_task(r) for p in paths for r in base.jsonl(p)]
    original = [local.component(r) for r in base.jsonl(base.BENCH / "catalog.jsonl")]
    for scene, stratum in STRATA:
        eligible = [t for t in tasks if t.scene == scene and t.task_type == stratum
                    and t.injected_conflict != "global_no_solution" and t.task_id not in prior]
        assert len(eligible) >= 12
        for t in sorted(eligible, key=lambda x: base.seed_for(SEED, x.task_id))[:12]:
            items = []
            for role in t.required_roles:
                group = [c for c in original if (c.scene, c.role) == (scene, role)]
                labels = [i % 4 for i in range(len(group))]
                random.Random(base.seed_for(SEED, t.task_id, role)).shuffle(labels)
                for c, family in zip(group, labels):
                    ports = tuple(replace(p, port_type=f"{p.port_type}/family-{family}") for p in c.ports)
                    items.append(replace(c, ports=ports))
            first = tuple(x for role in t.required_roles for x in t.candidate_ids[role])
            allowed = {r: {c.template_id for c in items if c.role == r} - set(first)
                       for r in t.required_roles}
            yield t, items, first, allowed


def stage_two(t, items, first, added):
    ids = set(first + added)
    pool = {r: tuple(c.template_id for c in items if c.role == r and c.template_id in ids)
            for r in t.required_roles}
    role = min(t.required_roles, key=lambda r: sum(
        base.unary_score(t, c)[0] == 0 for c in items if c.role == r and c.template_id in ids))
    updated = replace(t, candidate_ids=pool, conflict_scope=role)
    text = local.prompt(updated, items).replace(
        "Select exactly one NEW component identifier per role to add to the existing shortlists.",
        f"Select one NEW component identifier for designated role {role}. Keep all other role pools unchanged.",
    )
    contract = json.dumps({r: "NEW_ID" for r in t.required_roles})
    assert contract in text
    text = text.replace(contract, json.dumps({role: "NEW_ID"}))
    allowed = {role: {c.template_id for c in items if c.role == role} - ids}
    return updated, allowed, text


def generate(cases, args):
    import requests

    input_rows = {t.task_id: dict(task=t.to_dict(), catalog=[c.to_dict() for c in items],
                   initial=list(first), first_prompt=local.prompt(t, items),
                   first_grammar=local.eligibility_grammar(allowed))
                  for t, items, first, allowed in cases}
    dump(args.output / "inputs.json", input_rows)
    dump(args.output / "protocol.json", dict(seed=SEED, n=48, request_limit=96, repeats_per_task=1,
         weights_sha256=local.SHA, engine_sha256=local.sha(args.server), model_manifest_sha256=local.MANIFEST_SHA,
         inputs_sha256=local.sha(args.output / "inputs.json"), e5_revision=E5_REV,
         gen=local.PARAMS, analyses=["first: Qwen vs BM25, E5, Rule, Support (Holm)",
         "second: same frozen first pool vs one additional Qwen choice (exact McNemar)"],
         failures="record and retain prior pool without correction or retry"))
    with fixed.local_engine(args.server, args.model, args.port, args.output / "engine.log") as (http, address):
        def propose(task, stage, text, grammar):
            token_result = http.post(address + "/tokenize", json=dict(content=text, add_special=True,
                                      parse_special=True), timeout=20)
            token_result.raise_for_status()
            n = len(token_result.json()["tokens"])
            assert n + local.PARAMS["n_predict"] < 8192
            start = perf_counter()
            row = dict(task_id=task, stage=stage, prompt_tokens=n)
            try:
                response = http.post(address + "/completion", json=dict(
                    prompt=text, grammar=grammar, seed=SEED+stage, **local.PARAMS), timeout=150)
                row.update(http_status=response.status_code, raw_response=response.text)
            except requests.RequestException as exc:
                row["request_error"] = type(exc).__name__
            row["selection_ms"] = (perf_counter() - start)*1000
            return row

        with (args.output/"trials.jsonl").open("x", encoding="utf-8") as ledger:
            for t, items, initial, allowed in cases:
                case = input_rows[t.task_id]
                first = propose(t.task_id, 1, case["first_prompt"], case["first_grammar"])
                added, status = fixed.decode(first, allowed)
                updated, allowed2, prompt2 = stage_two(t, items, initial, added)
                grammar2 = local.eligibility_grammar(allowed2)
                second = propose(t.task_id, 2, prompt2, grammar2)
                row = dict(task_id=t.task_id, first=first, second=second, first_status=status,
                           first_additions=added, hint=updated.conflict_scope,
                           second_prompt=prompt2, second_grammar=grammar2)
                ledger.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
                ledger.flush()
                print("Paired stages saved:", t.task_id, flush=True)


def tokens(text):
    return re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower())


def bm25(query, docs):
    counts = [Counter(tokens(d)) for d in docs]
    lens = [sum(c.values()) for c in counts]
    avg = mean(lens)
    score = [0.0] * len(docs)
    for word in sorted(set(tokens(query))):
        hits = sum(word in c for c in counts)
        idf = math.log1p((len(docs) - hits + .5)/(hits + .5))
        for i, c in enumerate(counts):
            f = c[word]
            score[i] += idf * f * 2.2/(f + 1.2*(.25+.75*lens[i]/avg))
    return score


def retrieval_inputs(cases):
    for t, items, _, _ in cases:
        for role in t.required_roles:
            group = [c for c in items if c.role == role]
            query = f"{t.requirement} Select a {role.replace('_', ' ')}. "
            for bound, value in sorted(t.numeric_requirements.items()):
                r, field, op = bound.split(".")
                if r == role:
                    query += f"{field.replace('_', ' ')} {op} {value:g} {group[0].parameter_units.get(field,'')}. "
            docs = []
            for c in group:
                values = [f"{f.replace('_',' ')} {v:g} {c.parameter_units.get(f,'')}"
                          for f, v in sorted(c.parameters.items())]
                ports = [f"{p.direction} {p.port_type} {p.unit}" for p in c.ports]
                docs.append(" ".join((role.replace("_"," "), *values, *ports)))
            yield t.task_id, role, query, docs, [c.template_id for c in group]


def e5_embeddings(model, text):
    import torch
    from transformers import AutoModel, AutoTokenizer

    manifest = json.loads((model / "verified_manifest.json").read_text(encoding="utf-8"))
    assert manifest["revision"] == E5_REV
    for name, digest in manifest["sha256"].items():
        assert local.sha(model/name) == digest
    torch.set_num_threads(6)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    encoder = AutoModel.from_pretrained(model, local_files_only=True).eval()
    matrices = []
    with torch.inference_mode():
        for i in range(0, len(text), 48):
            inputs = tokenizer(text[i:i+48], max_length=512, padding=True, truncation=True, return_tensors="pt")
            mask = inputs["attention_mask"].unsqueeze(-1)
            states = encoder(**inputs).last_hidden_state
            pooled = (states*mask).sum(dim=1)/mask.sum(dim=1)
            matrices.append(torch.nn.functional.normalize(pooled, p=2, dim=1).numpy())
    return dict(zip(text, np.concatenate(matrices))), manifest


def rank_retrievers(cases, model, out, filename="retrieval_scores.json"):
    inputs = list(retrieval_inputs(cases))
    text = list(dict.fromkeys(s for _, _, q, docs, _ in inputs
                             for s in ("query: "+q, *("passage: "+d for d in docs))))
    vectors, manifest = e5_embeddings(model, text)
    result = {}
    for task, role, q, docs, ids in inputs:
        query = vectors["query: "+q]
        dense = [float(vectors["passage: "+d] @ query) for d in docs]
        result.setdefault(task, {})[role] = dict(query=q, documents=dict(zip(ids, docs)),
            BM25=dict(zip(ids,bm25(q,docs))), E5=dict(zip(ids,dense)))
    dump(out/filename, result)
    dump(out/"e5_manifest.json", manifest)


def historical_cases():
    catalog = [local.component(r) for r in base.jsonl(base.BENCH/"catalog.jsonl")]
    for scene in "AB":
        items = [c for c in catalog if c.scene == scene]
        for row in base.jsonl(base.BENCH/"tasks"/f"{scene}_test.jsonl"):
            t = base._coerce_task(row)
            initial = tuple(x for r in t.required_roles for x in t.candidate_ids[r])
            yield t, items, initial, None


def historical_comparison(cases, out):
    scores = json.loads((out/"historical_scores.json").read_text(encoding="utf-8"))
    assert len(cases) == len(scores) == 400
    for task, role, query, docs, ids in retrieval_inputs(cases):
        saved = scores[task][role]
        assert saved["query"] == query and saved["documents"] == dict(zip(ids,docs))
        assert saved["BM25"] == dict(zip(ids,bm25(query,docs)))
        assert set(saved["E5"]) == set(ids) and all(math.isfinite(v) for v in saved["E5"].values())
    ref = base.jsonl(base.ARTIFACTS/"reconstruction-20261013"/"matched_replay.jsonl")
    archived = {(r["task_id"],r["repeat"]):r for r in ref if r["method"]=="P"}
    rows = []
    for t, items, initial, _ in cases:
        by_id = {c.template_id:c for c in items}
        for repeat in (1,2,3):
            r = archived[t.task_id,repeat]
            target = r["candidate_ids"]
            assert set(initial) <= set(target)
            role_counts = Counter(by_id[x].role for x in target if x not in initial)
            values = {"P":r["feasible"]}
            selected = {}
            for method in ("BM25","E5"):
                new = []
                for role in t.required_roles:
                    opts = scores[t.task_id][role][method]
                    candidates = sorted(set(opts)-set(initial),key=lambda x:(-opts[x],x))
                    new.extend(candidates[:role_counts[role]])
                pool = initial+tuple(new)
                assert Counter(by_id[x].role for x in new) == role_counts
                values[method] = int(base.solve_subgraph(t,pool,items).feasible)
                selected[method] = pool
            rows.append(dict(task=t.task_id,scene=t.scene,stratum=t.task_type,repeat=repeat,
                             forced=t.injected_conflict=="global_no_solution",values=values,pools=selected))
    assert len(rows)==1200
    groups = defaultdict(list)
    for row in rows:
        groups[row["scene"],row["stratum"],row["task"]].append(row)
    comparisons = []
    for s,z in STRATA:
        tasks=[g for (scene,kind,_),g in groups.items() if (scene,kind)==(s,z)]
        assert len(tasks)==100
        p=[base.majority([r["values"]["P"] for r in group]) for group in tasks]
        for method in ("BM25","E5"):
            v=[base.majority([r["values"][method] for r in group]) for group in tasks]
            comparisons.append(dict(scene=s,stratum=z,comparator=method,**base.paired(p,v)))
    base.holm(comparisons)
    dump(out/"historical_retrieval_metrics.json",rows)
    dump(out/"historical_retrieval_summary.json",comparisons)


def analyze(cases, out):
    archive = base.jsonl(out/"trials.jsonl")
    scores = json.loads((out/"retrieval_scores.json").read_text(encoding="utf-8"))
    protocol = json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    assert local.sha(out/"inputs.json") == protocol["inputs_sha256"]
    frozen = json.loads((out/"inputs.json").read_text(encoding="utf-8"))
    assert len(scores) == len(frozen) == 48
    for task, role, query, docs, ids in retrieval_inputs(cases):
        saved = scores[task][role]
        assert saved["query"] == query and saved["documents"] == dict(zip(ids,docs))
        assert saved["BM25"] == dict(zip(ids,bm25(query,docs)))
        assert set(saved["E5"]) == set(ids) and all(math.isfinite(v) for v in saved["E5"].values())
    assert [r["task_id"] for r in archive] == [t.task_id for t, *_ in cases]
    rows = []
    for (t, items, initial, allowed), row in zip(cases, archive):
        first_record = frozen[t.task_id]
        assert first_record["task"] == t.to_dict()
        assert first_record["catalog"] == [c.to_dict() for c in items]
        assert first_record["initial"] == list(initial) and first_record["first_prompt"] == local.prompt(t,items)
        assert first_record["first_grammar"] == local.eligibility_grammar(allowed)
        added, s1 = fixed.decode(row["first"], allowed)
        updated, allowed2, prompt2 = stage_two(t, items, initial, added)
        assert prompt2 == row["second_prompt"] and local.eligibility_grammar(allowed2) == row["second_grammar"]
        assert s1 == row["first_status"] and list(added) == row["first_additions"]
        next_add, s2 = fixed.decode(row["second"], allowed2)
        witnesses, by = all_witnesses(t, items)
        assert bool(len(witnesses)) == base.solve_subgraph(t, tuple(c.template_id for c in items), items).feasible
        pools = {"Initial": initial, "Qwen": initial+added, "Qwen2": initial+added+next_add}
        target = revision.quota_target(t,items,1)
        for name in ("Rule","Support"):
            pools[name] = revision.select(t,items,target,name,1)
            target2 = initial+added+(min(allowed2[updated.conflict_scope]),)
            pools[name+"2"] = revision.select(updated,items,target2,name,1)
        for name in ("BM25","E5"):
            pools[name] = initial+tuple(min(allowed[r],key=lambda i:(-scores[t.task_id][r][name][i],i))
                                       for r in t.required_roles)
        assert all(set(initial) <= set(p) for p in pools.values())
        assert all(len(pools[m]) == len(initial)+4 for m in ("Rule","Support","BM25","E5"))
        result = {k:dict(ids=p,**local.metrics(t,items,witnesses,by,p)) for k,p in pools.items()}
        assert all(result[k]["feasible"] >= result["Qwen"]["feasible"] for k in ("Qwen2","Rule2","Support2"))
        rows.append(dict(task=t.task_id,scene=t.scene,stratum=t.task_type,full_solutions=len(witnesses),
                         eligibility=[s1,s2],methods=result))
    def pair(a,b):
        return base.paired([r["methods"][a]["feasible"] for r in rows],
                           [r["methods"][b]["feasible"] for r in rows])
    first_pairs = [dict(comparator=m,**pair("Qwen",m)) for m in ("Rule","Support","BM25","E5")]
    base.holm(first_pairs)
    summary = dict(full_feasible=sum(r["full_solutions"]>0 for r in rows),
        strata={f"{s}-{z}":sum(r["full_solutions"]>0 for r in rows if (r["scene"],r["stratum"])==(s,z))
                for s,z in STRATA},
        eligible=[dict(Counter(r["eligibility"][i] for r in rows)) for i in (0,1)],
        methods={m:dict(witness=sum(r["methods"][m]["feasible"] for r in rows),
                        by_scene={s:sum(r["methods"][m]["feasible"] for r in rows if r["scene"]==s) for s in "AB"},
                        recall=mean(r["methods"][m]["candidate_recall"] for r in rows if r["full_solutions"]),
                        solution=mean(r["methods"][m]["solution_fraction"] for r in rows if r["full_solutions"]),
                        nodes=mean(r["methods"][m]["search_nodes"] for r in rows)) for m in pools},
        first_pairs=first_pairs, second_pairs={m:pair("Qwen2",m) for m in ("Qwen","Rule2","Support2")},
        telemetry=[fixed.telemetry([r[stage] for r in archive]) for stage in ("first","second")])
    dump(out/"metrics.json",rows)
    dump(out/"summary.json",summary)


def catalog_case(out):
    # KTR Drive Technology, ROTEX Standard, cast iron GJL, hub type 1.
    data = [(38,12,40,(190,325,405)), (42,14,45,(265,450,560)),
            (48,15,52,(310,525,655)), (55,20,60,(410,685,825)),
            (65,22,70,(625,940,1175)), (75,30,80,(1280,1920,2400))]
    scenarios = [(35,40,300),(40,45,500),(45,52,650),(50,60,800),
                 (65,70,1500),(30,35,2400),(20,70,300),(25,60,300),(28,52,500)]
    results = []
    for left,right,torque in scenarios:
        u=[(s,lo,hi) for s,lo,hi,_ in data if lo<=left<=hi]
        v=[(s,lo,hi) for s,lo,hi,_ in data if lo<=right<=hi]
        e=[(s,h,rating) for s,_,_,ratings in data for h,rating in zip(("92 Sh-A","98 Sh-A","64 Sh-D"),ratings)
           if rating>=torque]
        exhaustive=[(a[0],b[0],c[0],c[1]) for a,b,c in product(u,v,e) if a[0]==b[0]==c[0]]
        index=[(s,s,s,h) for s,h,_ in e if s in {a[0] for a in u}&{b[0] for b in v}]
        assert sorted(exhaustive)==sorted(index)
        results.append(dict(shaft_left_mm=left,shaft_right_mm=right,required_rated_torque_nm=torque,
                            unary_domains=[len(u),len(v),len(e)],witnesses=exhaustive))
    dump(out/"ktr_case.json",dict(source=KTR,printed_page=34,section="ROTEX Standard Cast iron (GJL), hub component 1",
        rows=data,results=results,scope="catalog bore and nominal torque only; no application approval"))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=OUT)
    parser.add_argument("--server",type=Path)
    parser.add_argument("--model",type=Path)
    parser.add_argument("--e5",type=Path)
    parser.add_argument("--rank-only",action="store_true")
    parser.add_argument("--historical",action="store_true")
    parser.add_argument("--port",type=int,default=17665)
    args=parser.parse_args()
    cases=list(new_cases())
    assert len(cases)==48
    if args.historical:
        history=list(historical_cases())
        if args.e5:
            rank_retrievers(history,args.e5,args.output,"historical_scores.json")
        historical_comparison(history,args.output)
        return
    if args.server:
        assert args.model
        args.output.mkdir(parents=True,exist_ok=False)
        generate(cases,args)
    if args.e5:
        rank_retrievers(cases,args.e5,args.output)
    if not args.rank_only and (args.output/"retrieval_scores.json").exists():
        analyze(cases,args.output)
        catalog_case(args.output)


if __name__=="__main__":
    main()
