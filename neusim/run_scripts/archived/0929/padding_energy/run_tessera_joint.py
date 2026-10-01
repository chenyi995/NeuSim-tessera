"""Native E2E EDP sweep with joint SRAM/array mapping at each bandwidth."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime,timezone
import json
import math
import multiprocessing
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback

from neusim.npusim.frontend.Operator import Operator,EinsumOperator,FlashAttentionOperator,Tensor,Axis
from neusim.npusim.frontend.llm_ops_lib import create_einsum_op
from neusim.npusim.backend.tessera_baselines import ARCHITECTURES
from neusim.run_scripts.run_tessera_partitioned import config,native_record,bounded_map,ENERGIES
from neusim.run_scripts.run_tessera_speed import operator_identity
from neusim.run_scripts.compare_tessera_native import ROOT,rows,sha,write_csv

MAIN=(*ARCHITECTURES,"Tessera-32","Tessera-8")
FIELDS=("time_ns","energy_J","hbm_bytes","sram_bytes","sa_ns","vu_ns","ici_ns","useful_macs","static_J","dynamic_J",*ENERGIES)
CONFIGS={}


def initialize(grid):
    global CONFIGS
    CONFIGS={}
    for name in MAIN:
        c=config("full",grain=8 if name=="Tessera-8" else 32)
        if name in ARCHITECTURES:c.tessera_parameters["baseline_architecture"]=name
        c.tessera_parameters.update(mapping_policy="joint_edp",selection_bandwidths=grid)
        for bw in grid:
            CONFIGS[name,bw]=c.model_copy(update={"hbm_bw_GBps":bw/1024**3},deep=True)
    for grain in (32,8):
        for variant in ("square","independent_noskew","skew"):
            c=config(variant,grain=grain)
            # Skew fixes the full design's joint map, including its search grid.
            c.tessera_parameters.update(mapping_policy="joint_edp",selection_bandwidths=grid if variant=="skew" else [grid[-1]])
            CONFIGS[f"Tessera-{grain}-{variant}",grid[-1]]=c


def restore_operator(raw):
    """Restore native tensor state; Tensor.__init__ does not accept saved UUIDs."""
    r=dict(raw)
    for key in ("input_tensors","output_tensors"):
        r[key]=[Tensor.model_construct(**{**t,"axes":[Axis.model_validate(x) for x in t["axes"]]}) for t in r[key]]
    cls={"Einsum":EinsumOperator,"FlashAttention":FlashAttentionOperator}.get(r["opcode"],Operator)
    op=cls.model_validate(r)
    expected=(raw["opcode"],raw["config_str"],raw["input_tensor_shape_str"],raw["output_tensor_shape_str"],
              raw["stats"]["flop_count"],raw["stats"]["ici_time_ns"],raw["stats"]["ici_traffic_inbound_bytes"],
              raw["stats"]["ici_traffic_outbound_bytes"],json.dumps(raw["tessera_spec"],sort_keys=True))
    assert operator_identity(op)==expected
    return op


def cost_job(job):
    kind,oid,payload=job
    original=create_einsum_op([payload[0],payload[2]],[payload[2],payload[1]],"MK;KN->MN") if kind=="shape" else restore_operator(payload)
    result=[]
    for (name,bw),chip in CONFIGS.items():
        r=native_record(original.model_copy(deep=True),chip)
        result.append(dict(kind=kind,operator_id=oid,architecture=name,bandwidth_bytes_per_second=bw,
                           **{key:r[key] for key in FIELDS},peak_live_bytes=r["peak_live_bytes"],mapping_json=r["mapping_json"]))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--base",type=Path,required=True)
    p.add_argument("--verification",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--workers",type=int,default=16);p.add_argument("--limit",type=int,default=0)
    a=p.parse_args();base=a.base.resolve();out=a.out.resolve()
    assert out.is_relative_to(ROOT) and not out.exists() and 1<=a.workers<=16
    out.mkdir(parents=True)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:a.workers])
    cap=96_000_000_000//(a.workers+1);resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
    audit=json.loads(a.verification.read_text());assert audit["status"]=="PASS"
    for path,digest in audit["source_sha256"].items():assert sha(Path(path))==digest,path
    inputs=base/"inputs_snapshot";speed=base/"speed"
    old=json.loads((speed/"manifest.json").read_text());assert old["status"]=="PASS"
    assert json.loads((speed/"analysis/verification.json").read_text())["status"]=="PASS"
    grid=json.loads((base/"baseline_verification/bandwidth_grid.json").read_text())["bytes_per_second"]
    initialize(grid)
    meta=json.loads((inputs/"workload_metadata.json").read_text())
    groups=json.loads((inputs/"workload_groups.json").read_text())
    cohort={(r["dataset"],r["cohort"]):r for r in meta["cohorts"]}
    operator_groups={key for key,r in cohort.items() if not r["trace_tag"]}
    refs=[inputs/"inputs/workloads.csv",inputs/"workload_metadata.json",inputs/"workload_groups.json",
          speed/"manifest.json",speed/"analysis/verification.json",a.verification,
          base/"baseline_verification/bandwidth_grid.json"]
    for name in ("trace_graph_operators.jsonl","trace_graph_batches.jsonl"):
        path=speed/name;assert sha(path)==old["output_sha256"][name];refs.append(path)
    sources=list((ROOT/"neusim/npusim/backend").glob("*.py"))
    sources+=list((ROOT/"neusim/npusim/frontend").glob("*.py"))
    sources+=[ROOT/"neusim/configs/chips/ChipConfig.py",ROOT/"neusim/run_scripts/run_tessera_partitioned.py",Path(__file__)]
    for src in sources:
        dst=out/"source"/src.resolve().relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst);assert sha(src)==sha(dst)
    manifest=dict(status="running",started_utc=datetime.now(timezone.utc).isoformat(),command=sys.argv,
                  workers=a.workers,memory_budget_bytes=100_000_000_000,per_process_limit_bytes=cap,
                  bandwidth_bytes_per_second=grid,pilot=bool(a.limit),base=str(base),
                  selection="Per-GEMM joint EDP over every capacity-feasible divisor SRAM tile and legal array geometry, separately at each bandwidth; fixed native OS reuse order. FlashAttention uses native Br/Bc and resident QK/PV joint mapping.",
                  accounting="E2E EDP = sum(chip energy)*tensor_parallel_chips * sum(serial operator time). Model and standalone GEMM cohorts kept separate.",
                  source_sha256={str(x):sha(x) for x in refs+sources})
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (out/"configs.json").write_text(json.dumps({f"{name}@{bw}":c.model_dump() for (name,bw),c in CONFIGS.items()},indent=2)+"\n")
    shapes={};weights=defaultdict(Counter);jobs=[];batch_checks=[]
    for r in rows(inputs/"inputs/workloads.csv"):
        group=r["dataset"],r["cohort"]
        if group not in operator_groups:continue
        sid=int(r["shape_id"]);shape=tuple(int(r[k]) for k in ("M","N","K"))
        assert sid not in shapes or shapes[sid]==shape
        shapes[sid]=shape;weights["shape",sid][group]+=int(r["repeat"])
    for sid,shape in sorted(shapes.items()):jobs.append(("shape",sid,shape))
    graph_macs={}
    for line in (speed/"trace_graph_operators.jsonl").open():
        r=json.loads(line);oid=r["operator_id"];op=restore_operator(r["operator"])
        jobs.append(("graph",oid,r["operator"]))
        graph_macs[oid]=op.stats.flop_count//2 if op.opcode=="Einsum" else None
    tag_to_group={r["trace_tag"]:key for key,r in cohort.items() if r["trace_tag"]}
    expected_by_model=Counter();batch_count=Counter()
    for line in (speed/"trace_graph_batches.jsonl").open():
        r=json.loads(line);tag=r["model"];batch_count[tag]+=1;expected_by_model[tag]+=r["expected_macs"]
        batch_checks.append(r)
        for oid,repeat in r["operators"]:weights["graph",oid][tag_to_group[tag]]+=repeat
    if a.limit:
        # Pilot samples across shapes and all four native graph categories.
        step=max(1,len(jobs)//a.limit);jobs=jobs[::step][:a.limit]
    write_csv(out/"invocation_weights.csv",[dict(kind=kind,operator_id=oid,dataset=d,cohort=c,repeat=count)
              for (kind,oid),ww in weights.items() for (d,c),count in ww.items()])
    totals=defaultdict(Counter);macs={};start=time.monotonic();count=0
    try:
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context("fork")) as pool, (out/"operator_costs.csv").open("x",newline="") as f:
            writer=None
            for result in bounded_map(pool,cost_job,jobs,block=64,chunksize=1):
                if writer is None:writer=csv.DictWriter(f,fieldnames=list(result[0]));writer.writeheader()
                writer.writerows(result)
                for r in result:
                    identity=r["kind"],r["operator_id"]
                    assert identity not in macs or macs[identity]==r["useful_macs"]
                    macs[identity]=r["useful_macs"]
                    for group,repeat in weights[identity].items():
                        acc=totals[group,r["architecture"],r["bandwidth_bytes_per_second"]]
                        for key in FIELDS:acc[key]+=r[key]*repeat
                count+=1
                if count%64==0:
                    f.flush();print(f"Joint EDP: {count}/{len(jobs)}, elapsed {time.monotonic()-start:.1f}s",flush=True)
                    (out/"progress.json").write_text(json.dumps(dict(completed=count,total=len(jobs),elapsed_seconds=time.monotonic()-start))+"\n")
        assert count==len(jobs)
        if not a.limit:
            for row in batch_checks:
                assert sum(macs["graph",oid]*repeat for oid,repeat in row["operators"])==row["expected_macs"],row
        records=[]
        for (group,name,bw),stats in sorted(totals.items()):
            g=next(g for g in groups if (g["dataset"],g["cohort"])==group);tag=cohort[group]["trace_tag"]
            chips=meta["models"][tag]["tensor_parallel_size"] if tag else 1
            if tag and not a.limit:assert stats["useful_macs"]==expected_by_model[tag]
            close=sum(stats[f] for f in ENERGIES)
            assert math.isclose(close,stats["energy_J"],rel_tol=2e-11)
            records.append(dict(workload_id=g["workload_id"],workload=g["name"],dataset=group[0],cohort=group[1],
                boundary="model_compute_e2e" if tag else "single_operator_e2e_weighted_sum",architecture=name,
                bandwidth_bytes_per_second=bw,chips=chips,**stats,seconds=stats["time_ns"]*1e-9,
                system_energy_J=stats["energy_J"]*chips,edp_J_s=stats["energy_J"]*chips*stats["time_ns"]*1e-9))
        write_csv(out/"totals.csv",records)
        manifest.update(status="PASS",operators=count,operator_shapes=len(shapes),graph_operators=len(graph_macs),
                        full_model_batches=dict(batch_count),elapsed_seconds=time.monotonic()-start,
                        output_sha256={x.name:sha(x) for x in out.iterdir() if x.is_file() and x.name!="manifest.json"})
    except BaseException as exc:
        manifest.update(status="FAIL",error=f"{type(exc).__name__}: {exc}")
        (out/"failure.txt").write_text(traceback.format_exc());raise
    finally:
        (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:v for k,v in manifest.items() if not k.endswith("sha256")}),flush=True)


if __name__=="__main__":main()
