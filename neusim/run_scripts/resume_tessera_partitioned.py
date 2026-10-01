"""Recover an interrupted run using completed GEMM costs and native graph reuse.

Array, memory and power equations are unchanged. Integer fields use checked
int64 reductions; energy retains the existing analysis tolerance. Completed
batch rows are preserved and independently compared with reconstructed costs.
"""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import argparse
import csv
from datetime import datetime, timezone
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import resource
import shutil
import subprocess
import time
import traceback

import numpy as np
from neusim.run_scripts import run_tessera_partitioned as legacy
from neusim.run_scripts.run_tessera_speed import collect_graph_job, operator_identity
from neusim.run_scripts.compare_tessera_native import ROOT, rows, sha, write_csv
from neusim.run_scripts.summarize_tessera_native import number
from neusim.run_scripts.analyze_tessera_partitioned import close
from neusim.npusim.frontend.llm_ops_lib import create_einsum_op

FIELDS=tuple(legacy.FIELDS)
ALL_FIELDS=FIELDS+("peak_live_bytes",)
FLOAT_FIELDS=("energy_J","static_J","dynamic_J",*legacy.ENERGIES)
INT_FIELDS=tuple(k for k in FIELDS if k not in FLOAT_FIELDS)
KEYS=tuple((p,v) for p in ("sa","native_auto") for v in legacy.VARIANTS)
_SEEDS={}
_META=None
_GRAIN=None
_CONFIGS=None


def packed(record):
    result=[]
    for k in ALL_FIELDS:
        v=number(record[k]) if isinstance(record[k],str) else record[k]
        if k not in FLOAT_FIELDS:
            assert v==int(v) and 0<=v<2**63,(k,v)
            v=int(v)
        result.append(v)
    return tuple(result)


def graph_job(job):
    ops,refs=collect_graph_job(job,_META,_CONFIGS[0])
    phases={str(bid):("decode" if all(q==1 for q,_ in reqs) else "prefill" if all(q>1 for q,_ in reqs) else "mixed")
            for bid,reqs,_ in job[1]}
    return ops,[(tag,str(bid),expected,phases[str(bid)],values) for tag,bid,expected,values in refs]


def cost_job(job):
    oid,payload=job
    op=pickle.loads(payload);signature=operator_identity(op)
    results=[]
    for index,(key,chip) in enumerate(zip(KEYS,_CONFIGS)):
        seed=_SEEDS.get((signature,*key))
        values=seed if seed is not None else packed(legacy.native_record(op.model_copy(deep=True),chip))
        results.append((oid,index,values,seed is not None))
    return results


def allocate(count):
    return (np.zeros((count,len(KEYS),len(INT_FIELDS)),dtype=np.int64),
            np.zeros((count,len(KEYS),len(FLOAT_FIELDS)),dtype=np.float64),
            np.zeros((count,len(KEYS)),dtype=np.int64))


def put(matrices,result):
    integers,energy,peaks=matrices
    for oid,index,values,_ in result:
        record=dict(zip(ALL_FIELDS,values))
        integers[oid,index]=[record[k] for k in INT_FIELDS]
        energy[oid,index]=[record[k] for k in FLOAT_FIELDS]
        peaks[oid,index]=record["peak_live_bytes"]


def aggregate(refs,matrices):
    ids=np.array([oid for oid,_ in refs],dtype=np.int64)
    weights=np.array([count for _,count in refs],dtype=np.int64)
    assert weights.min()>0
    ints=np.tensordot(weights,matrices[0][ids],axes=1)
    energy=np.tensordot(weights,matrices[1][ids],axes=1)
    peaks=matrices[2][ids].max(axis=0)
    return [dict(**{k:int(v) for k,v in zip(INT_FIELDS,ints[index])},
                 **{k:float(v) for k,v in zip(FLOAT_FIELDS,energy[index])},
                 peak_live_bytes=int(peaks[index])) for index in range(len(KEYS))]


def compare(a,b):
    for k in ALL_FIELDS:close(a[k],b[k])


def canary(metadata):
    checks=0;seed_checks=0
    for job in legacy.trace_jobs(metadata,1):
        direct,_=legacy.trace_job(job)
        expected={(r["model"],str(r["batch_id"]),r["policy"],r["variant"]):r for r in direct}
        ops,batches=graph_job(job);ids={s:i for i,s in enumerate(ops)}
        matrices=allocate(len(ops))
        for signature,(_,payload) in ops.items():
            result=cost_job((ids[signature],payload));put(matrices,result)
            seed_checks+=sum(r[-1] for r in result)
        for tag,bid,macs,phase,refs in batches:
            values=aggregate([(ids[s],n) for s,n in refs],matrices)
            for key,value in zip(KEYS,values):
                assert value["useful_macs"]==macs
                compare(value,expected[(tag,bid,*key)]);checks+=1
    return dict(status="PASS",native_direct_batch_comparisons=checks,reused_gemm_checks=seed_checks,
                integer_tolerance=0,energy_relative_tolerance=2e-10,energy_absolute_tolerance=1e-11)


def main():
    global _META,_GRAIN,_CONFIGS,_SEEDS
    p=argparse.ArgumentParser();p.add_argument("--interrupted",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--workers",type=int,default=16)
    p.add_argument("--verify-only",action="store_true")
    a=p.parse_args();out=a.out.resolve();prior=a.interrupted.resolve()
    assert not out.exists() and out.is_relative_to(ROOT) and 1<=a.workers<=16
    out.mkdir(parents=True)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:a.workers])
    cap=96_000_000_000//(a.workers+1);resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
    old=json.loads((prior/"manifest.json").read_text());assert old["status"]=="running"
    _META=json.loads((prior/"workload_metadata.json").read_text());_GRAIN=old["grain"]
    legacy.initialize(_META,_GRAIN)
    _CONFIGS=[legacy.config(v,p,grain=_GRAIN) for p,v in KEYS]
    assert _CONFIGS[0].model_dump()==json.loads((prior/"chip_config.json").read_text())
    hashes=dict(old["source_sha256"])
    for path,digest in hashes.items():
        path=Path(path)
        assert sha(path)==digest,("executed input/core changed",path)
        if path.is_relative_to(ROOT):
            frozen=prior/"source"/path.relative_to(ROOT)
            if frozen.exists():
                assert sha(frozen)==digest
                dst=out/"source"/path.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(frozen,dst);assert sha(dst)==digest
    for relative in ("neusim/run_scripts/resume_tessera_partitioned.py","neusim/run_scripts/run_tessera_speed.py",
                     "neusim/npusim/frontend/llm_ops_lib.py"):
        path=ROOT/relative;dst=out/"source"/relative;dst.parent.mkdir(parents=True,exist_ok=True)
        digest=sha(path);shutil.copyfile(path,dst);assert sha(dst)==sha(path)==digest;hashes[str(path)]=digest
    copied={}
    for name in ("shape_results.csv","operator_totals.csv","chip_config.json","workload_metadata.json"):
        digest=sha(prior/name);shutil.copyfile(prior/name,out/name)
        assert sha(out/name)==sha(prior/name)==digest;copied[name]=digest
    identities={};shape_keys=defaultdict(set)
    for r in rows(out/"shape_results.csv"):
        sid=int(r["shape_id"]);dims=tuple(int(r[k]) for k in ("M","N","K"))
        if sid not in identities:
            m,n,k=dims;identities[sid]=operator_identity(create_einsum_op([m,k],[k,n],"MK;KN->MN"))
        key=r["policy"],r["variant"]
        assert key in KEYS and key not in shape_keys[sid];shape_keys[sid].add(key)
        value=packed(r);seed_key=(identities[sid],*key)
        assert seed_key not in _SEEDS or _SEEDS[seed_key]==value
        _SEEDS[seed_key]=value
    assert all(v==set(KEYS) for v in shape_keys.values())
    del identities,shape_keys
    manifest=dict(old,status="running",started_utc=datetime.now(timezone.utc).isoformat(),
        recovery_from=str(prior),recovery_parent_manifest_sha256=sha(prior/"manifest.json"),
        source_sha256=hashes,workers=a.workers,per_process_address_space=cap,
        affinity=sorted(os.sched_getaffinity(0)),copied_outputs_sha256=copied,
        base_git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    start=time.monotonic()
    try:
        check=canary(_META)
        (out/"reuse_verification.json").write_text(json.dumps(check,indent=2)+"\n")
        print(json.dumps(check),flush=True)
        if a.verify_only:
            manifest.update(status="VERIFIED",elapsed_seconds=time.monotonic()-start);return
        completed={};ignored=[];seen_partial=False
        for key,group in itertools.groupby(rows(prior/"trace_batches.csv"),key=lambda r:(r.get("model"),r.get("batch_id"))):
            group=list(group)
            valid=(len(group)==len(KEYS) and {(r.get("policy"),r.get("variant")) for r in group}==set(KEYS)
                   and all(all(r.get(k) is not None for k in ALL_FIELDS) for r in group))
            if not valid:
                assert not seen_partial;seen_partial=True;ignored.append(dict(model=key[0],batch_id=key[1],rows=len(group)));continue
            assert not seen_partial,"incomplete interior batch"
            for r in group:completed[r["model"],r["batch_id"],r["policy"],r["variant"]]=r
        print(f"Reusing {len(_SEEDS)} saved GEMM costs and checking {len(completed)//len(KEYS)} complete saved batches",flush=True)
        ids={};templates=[];categories=[];references=[];weights=Counter()
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context("fork")) as pool:
            with (out/"graph_operators.jsonl").open("x") as opfile,(out/"graph_batches.jsonl").open("x") as batchfile:
                for ops,batches in legacy.bounded_map(pool,graph_job,legacy.trace_jobs(_META,0),block=32):
                    for signature,(category,payload) in ops.items():
                        if signature not in ids:
                            oid=len(templates);ids[signature]=oid;templates.append(payload);categories.append(category)
                            opfile.write(json.dumps(dict(operator_id=oid,category=category,operator=pickle.loads(payload).model_dump(mode="json")))+"\n")
                    for tag,bid,macs,phase,source_refs in batches:
                        refs=[(ids[s],n) for s,n in source_refs]
                        references.append((tag,bid,macs,phase,refs))
                        batchfile.write(json.dumps(dict(model=tag,batch_id=bid,expected_macs=macs,phase=phase,operators=refs))+"\n")
                        for oid,repeat in refs:weights[tag,oid]+=repeat
                    if len(references)%512<len(batches):
                        print(f"Recovery graph collection: {len(references)} steps, {len(templates)} distinct operators",flush=True)
            del ids
            matrices=allocate(len(templates));reuse=0
            with (out/"graph_costs.csv").open("x",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=("operator_id","policy","variant","reused_gemm",*ALL_FIELDS));writer.writeheader()
                for count,result in enumerate(legacy.bounded_map(pool,cost_job,enumerate(templates),chunksize=2),1):
                    put(matrices,result)
                    for oid,index,values,reused in result:
                        policy,variant=KEYS[index];reuse+=reused
                        writer.writerow(dict(operator_id=oid,policy=policy,variant=variant,reused_gemm=int(reused),**dict(zip(ALL_FIELDS,values))))
                    if count%256==0:f.flush();print(f"Recovery graph costs: {count}/{len(templates)}",flush=True)
        global_weights=Counter()
        for (_,oid),repeat in weights.items():global_weights[oid]+=repeat
        bound=sum(repeat*int(matrices[0][oid].max()) for oid,repeat in global_weights.items())
        assert bound<np.iinfo(np.int64).max,("integer overflow bound",bound)
        totals=defaultdict(Counter);checked=0;used=set()
        with (out/"trace_batches.csv").open("x",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=("model","batch_id","phase","policy","variant","chips",*ALL_FIELDS));writer.writeheader()
            for tag,bid,macs,phase,refs in references:
                values=aggregate(refs,matrices)
                for (policy,variant),st in zip(KEYS,values):
                    assert st["useful_macs"]==macs
                    key=tag,bid,policy,variant
                    if key in completed:
                        old_row=completed[key]
                        old_stats=dict(zip(ALL_FIELDS,packed(old_row)));compare(st,old_stats)
                        assert old_row["phase"]==phase
                        row=old_row;st=old_stats;used.add(key);checked+=1
                    else:
                        row=dict(model=tag,batch_id=bid,phase=phase,policy=policy,variant=variant,
                                 chips=_META["models"][tag]["tensor_parallel_size"],**st)
                    writer.writerow(row)
                    for label in ("all",phase):
                        acc=totals[tag,policy,variant,label];legacy.add(acc,st);acc["batches"]+=1
        assert used==set(completed)
        total_rows=[]
        for (tag,policy,variant,phase),st in sorted(totals.items()):
            chips=_META["models"][tag]["tensor_parallel_size"]
            total_rows.append(legacy.totals_row(dict(model=tag,policy=policy,variant=variant,phase=phase,chips=chips),st,chips))
        write_csv(out/"e2e_totals.csv",total_rows);legacy.verify_totals(out/"e2e_totals.csv",total_rows)
        grouped=defaultdict(list)
        for (tag,oid),repeat in weights.items():grouped[tag,categories[oid]].append((oid,repeat))
        profiles=[]
        for (tag,category),refs in sorted(grouped.items()):
            for (policy,variant),st in zip(KEYS,aggregate(refs,matrices)):
                profiles.append(dict(model=tag,policy=policy,variant=variant,category=category,**st))
        write_csv(out/"e2e_operator_profiles.csv",profiles);legacy.verify_totals(out/"e2e_operator_profiles.csv",profiles)
        recovery=dict(status="PASS",completed_batch_rows_checked_and_preserved=checked,
            complete_saved_batches=checked//len(KEYS),ignored_incomplete_tail=ignored,
            saved_gemm_costs_reused=reuse,unique_graph_operators=len(templates),
            integer_accumulation_upper_bound=bound,canary=check,
            interrupted_trace_sha256=sha(prior/"trace_batches.csv"))
        (out/"recovery_verification.json").write_text(json.dumps(recovery,indent=2)+"\n")
        audit=json.loads((Path(old["metadata"]).parent/"copy_verification.json").read_text())
        manifest.update(status="PASS",operator_shapes=audit["unique_shapes"],workload_records=audit["input_records"],
            e2e_batches=len(references),source_mac_checks="PASS",summary_readback="PASS",recovery=recovery,
            elapsed_seconds=time.monotonic()-start,
            output_sha256={x.name:sha(x) for x in out.iterdir() if x.is_file() and x.name!="manifest.json"})
    except BaseException as exc:
        manifest.update(status="FAIL",error=f"{type(exc).__name__}: {exc}")
        (out/"failure.txt").write_text(traceback.format_exc());raise
    finally:
        (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:v for k,v in manifest.items() if k not in ("source_sha256","output_sha256")}),flush=True)


if __name__=="__main__":main()
