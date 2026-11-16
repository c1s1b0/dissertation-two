"""冻结首轮候选后，只补一次限定角色的本地生成，并支持离线重放。"""

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
from statistics import mean, median
import subprocess
from time import perf_counter, sleep

import cmc_prospective_20261106 as previous
import cmc_revision_20261106 as revision
import reconstruct_paper2_20261013 as base
from cmc_offline_20261014 import dump
from reconstruction_diagnostics_20261013 import all_witnesses

OUT = base.ARTIFACTS / "cmc-fixed-first-20261109"
SEED = 20261109


def frozen_cases():
    archive = base.jsonl(previous.DIAG_OUT / "trials.jsonl")
    assert len(archive) == len({row["task_id"] for row in archive}) == 24
    first = {row["task_id"]: row for row in archive}
    for t, items, witnesses, by, initial, eligible, first_prompt in previous.prepare():
        row = first[t.task_id]
        assert row["seed"] == previous.SEEDS[0] and row["status"] == "valid"
        first_ids = previous.parse_proposal(json.loads(row["raw_response"])["content"], eligible)
        assert first_ids == tuple(row["additions"])
        pool = {role: (*t.candidate_ids[role], extra) for role, extra in zip(t.required_roles, first_ids)}
        items_by_id = {item.template_id: item for item in items}
        role = min(t.required_roles, key=lambda r: sum(base.unary_score(t, items_by_id[x])[0] == 0 for x in pool[r]))
        allowed = {role: eligible[role] - set(first_ids)}
        flat = tuple(x for r in t.required_roles for x in pool[r])
        assert previous.metrics(t, items, witnesses, by, flat)["feasible"] == row["feasible"]
        t2 = replace(t, candidate_ids=pool, conflict_scope=role)
        task_prompt = previous.prompt(t2, items)
        task_prompt = task_prompt.replace(
            "Select exactly one NEW component identifier per role to add to the existing shortlists.",
            f"Select exactly one NEW component identifier for the designated role {role}. "
            "Keep all existing candidates in every role. Add nothing to the other roles.",
        )
        old_contract = json.dumps({r: "NEW_ID" for r in t.required_roles})
        assert old_contract in task_prompt
        task_prompt = task_prompt.replace(old_contract, json.dumps({role: "NEW_ID"}))
        yield t2, items, witnesses, by, flat, allowed, task_prompt


@contextmanager
def local_engine(server, model, port, log_path):
    import requests

    server, model = server.resolve(strict=True), model.resolve(strict=True)
    assert previous.sha(model) == previous.SHA
    args = [str(server), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
            "-c", "8192", "-np", "1", "-ngl", "99", "--device", "Vulkan0", "-t", "6", "-b", "512"]
    with log_path.open("xb") as log, requests.Session() as http:
        http.trust_env = False
        proc = subprocess.Popen(args, cwd=server.parent, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        address = f"http://127.0.0.1:{port}"
        try:
            for _ in range(90):
                if proc.poll() is not None:
                    raise RuntimeError("local engine exited")
                try:
                    if http.get(address + "/health", timeout=1).ok:
                        break
                except requests.RequestException:
                    pass
                sleep(1)
            else:
                raise TimeoutError("local engine did not start")
            yield http, address
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=15)


def freeze(cases, server, out):
    dump(out / "protocol.json", dict(seed=SEED, stage1_seed=previous.SEEDS[0],
         stage1_sha256=previous.sha(previous.DIAG_OUT / "trials.jsonl"),
         benchmark_sha256=previous.sha(previous.SOURCE), weights_sha256=previous.SHA,
         manifest_sha256=previous.MANIFEST_SHA, engine_sha256=previous.sha(server),
         cases=24, request_limit=24, remote_calls=0, repeats_per_task=1,
         role_policy="fewest unary-valid candidates in the frozen pool; ties by role order",
         primary="Qwen stage 2 versus fixed first proposal; exact paired McNemar",
         comparators=["Rule", "Support"], invalid_policy="no addition; separately count pipeline failure",
         params=previous.PARAMS))
    dump(out / "frozen_inputs.json", {t.task_id: dict(first_pool=pool, role=t.conflict_scope,
         eligible=sorted(allowed[t.conflict_scope]), prompt=prompt, grammar=previous.eligibility_grammar(allowed))
         for t, _, _, _, pool, allowed, prompt in cases})


def generate(cases, args, out):
    import requests

    frozen = json.loads((out / "frozen_inputs.json").read_text(encoding="utf-8"))
    with local_engine(args.server, args.model, args.port, out / "engine.log") as (http, address):
        lengths = {}
        for task, case in frozen.items():
            r = http.post(address + "/tokenize", json=dict(content=case["prompt"],
                          add_special=True, parse_special=True), timeout=20)
            r.raise_for_status()
            lengths[task] = len(r.json()["tokens"])
            assert lengths[task] + previous.PARAMS["n_predict"] < 8192
        with (out / "trials.jsonl").open("x", encoding="utf-8") as ledger:
            for t, *_ in cases:
                task = t.task_id
                request = dict(prompt=frozen[task]["prompt"], grammar=frozen[task]["grammar"],
                               seed=SEED, **previous.PARAMS)
                start = perf_counter()
                record = dict(task_id=task, seed=SEED, prompt_tokens=lengths[task])
                try:
                    response = http.post(address + "/completion", json=request, timeout=150)
                    record.update(http_status=response.status_code, raw_response=response.text)
                except requests.RequestException as exc:
                    record["request_error"] = type(exc).__name__
                record["selection_ms"] = (perf_counter() - start) * 1000
                ledger.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                ledger.flush()
                print("Second call archived:", task, flush=True)


def decode(record, allowed):
    if record.get("http_status") != 200:
        return (), "request_error"
    try:
        raw = json.loads(record["raw_response"])
        if raw.get("truncated"):
            return (), "truncated"
        return previous.parse_proposal(raw["content"], allowed), "valid"
    except (ValueError, KeyError) as exc:
        return (), str(exc) if str(exc) in {"invalid_json", "duplicate_role", "role_keys",
                                          "nonstring_id", "ineligible_id"} else "invalid_response"


def scores(task, items, w, by, ids, selection_ms):
    start = perf_counter()
    solve = base.solve_subgraph(task, ids, items)
    ms = (perf_counter() - start) * 1000
    witness, recall, solution_fraction = revision.retention(w, by, ids)
    assert solve.feasible == bool(witness)
    return dict(ids=list(ids), witness=witness, recall=recall, solution_fraction=solution_fraction,
                nodes=solve.search_nodes, selection_ms=selection_ms, solve_ms=ms)


def paired_stages(cases, trials):
    assert [r["task_id"] for r in trials] == [t.task_id for t, *_ in cases]
    rows = []
    for (t, items, w, by, first, allowed, _), record in zip(cases, trials):
        additions, status = decode(record, allowed)
        methods = dict(First=scores(t, items, w, by, first, None),
                       Qwen=scores(t, items, w, by, (*first, *additions), record["selection_ms"]))
        target = (*first, min(allowed[t.conflict_scope]))
        for method in ("Rule", "Support"):
            start = perf_counter()
            ids = revision.select(t, items, target, method, 1)
            selection_ms = (perf_counter() - start) * 1000
            assert set(first) <= set(ids) and set(ids) - set(first) <= allowed[t.conflict_scope]
            assert len(ids) == len(first) + 1
            methods[method] = scores(t, items, w, by, ids, selection_ms)
        assert all(row["witness"] >= methods["First"]["witness"] for row in methods.values())
        assert all(row["solution_fraction"] >= methods["First"]["solution_fraction"] for row in methods.values())
        rows.append(dict(task_id=t.task_id, scene=t.scene, role=t.conflict_scope, status=status,
                         added=list(additions), methods=methods))
    return rows


def coupled_cases(cases):
    output = []
    selected = sorted((row for row in cases if row[0].scene == "B"), key=lambda row: row[0].task_id)[:4]
    for t, items, w, _, _, _, _ in selected:
        mixed = [replace(item, ports=tuple(replace(port, unit="in")
                 if port.port_type.endswith(("/family-1", "/family-3")) else port
                 for port in item.ports)) for item in items]
        assert {p.unit for item in mixed for p in item.ports} == {"mm", "in"}
        base_task = replace(t, conflict_scope=None)
        gap = replace(base_task, numeric_requirements={**base_task.numeric_requirements,
                      "motor.torque_nm.min": 240., "coupling.torque_nm.max": 220.})
        for label, task in (("mixed_reference", base_task), ("coupled_torque_gap", gap)):
            assert task.injected_conflict is None
            reference, by = all_witnesses(task, mixed)
            counts = {r: sum(base.unary_score(task, item)[0] == 0 for item in group) for r, group in by.items()}
            solved = base.solve_subgraph(task, tuple(item.template_id for item in mixed), mixed)
            assert all(counts.values()) and solved.feasible == bool(len(reference))
            assert len(reference) == (len(w) if label == "mixed_reference" else 0)
            output.append(dict(task=task.to_dict(), kind=label, unary_counts=counts,
                               feasible=solved.feasible, full_solutions=len(reference),
                               nodes=solved.search_nodes, catalog=[i.to_dict() for i in mixed]))
    return output


def telemetry(records):
    responses = [json.loads(r["raw_response"]) for r in records if r.get("http_status") == 200]
    return dict(responses=len(responses), prompt_tokens=sum(x["tokens_evaluated"] for x in responses),
                prompt_processed=sum(x["timings"]["prompt_n"] for x in responses),
                generated_tokens=sum(x["tokens_predicted"] for x in responses),
                engine_median_ms=median(x["timings"]["prompt_ms"] + x["timings"]["predicted_ms"] for x in responses))


def summarize(rows, trials):
    summaries = {}
    for name in ("First", "Qwen", "Rule", "Support"):
        vals = [r["methods"][name] for r in rows]
        summaries[name] = dict(witness=sum(r["witness"] for r in vals),
            by_scene={s: sum(r["methods"][name]["witness"] for r in rows if r["scene"] == s) for s in "AB"},
            recall=mean(r["recall"] for r in vals), solution_fraction=mean(r["solution_fraction"] for r in vals),
            median_nodes=median(r["nodes"] for r in vals), median_solve_ms=median(r["solve_ms"] for r in vals),
            median_selection_ms=None if name == "First" else median(r["selection_ms"] for r in vals))
    pair = lambda a, b: base.paired([r["methods"][a]["witness"] for r in rows],
                                     [r["methods"][b]["witness"] for r in rows])
    status = dict(Counter(r["status"] for r in rows))
    valid = [r for r in rows if r["status"] == "valid"]
    return dict(methods=summaries, statuses=status,
                hint_counts=dict(Counter(r["role"] for r in rows)), added=sum(len(r["added"]) for r in rows),
                valid_witnesses=sum(r["methods"]["Qwen"]["witness"] for r in valid),
                paired={x: pair("Qwen", x) for x in ("First", "Rule", "Support")},
                telemetry=dict(free=telemetry(base.jsonl(previous.OUT / "trials.jsonl")),
                               grammar_first=telemetry(base.jsonl(previous.DIAG_OUT / "trials.jsonl")),
                               fixed_second=telemetry(trials),
                               second_prompt_tokens=sum(r["prompt_tokens"] for r in trials)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--port", type=int, default=17663)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if not args.replay and not (args.server and args.model):
        parser.error("provide --replay or both --server and --model")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    cases = list(frozen_cases())
    assert len(cases) == 24
    dump(out / "coupled_cases.json", coupled_cases(cases))
    if args.replay:
        frozen = json.loads((args.replay / "frozen_inputs.json").read_text(encoding="utf-8"))
        for t, _, _, _, pool, allowed, prompt in cases:
            record = frozen[t.task_id]
            assert list(pool) == record["first_pool"] and prompt == record["prompt"]
            assert previous.eligibility_grammar(allowed) == record["grammar"]
    else:
        freeze(cases, args.server, out)
        generate(cases, args, out)
    trials = base.jsonl((args.replay or out) / "trials.jsonl")
    rows = paired_stages(cases, trials)
    dump(out / "paired_metrics.json", rows)
    dump(out / "summary.json", summarize(rows, trials))


if __name__ == "__main__":
    main()
