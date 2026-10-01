"""Native E2E speed sweep with fixed HBM4-selected array/memory mappings.

All architectures use the same model graph, per-operator native HBM timing,
ideal per-PE SRAM supply, PE count and clock. Only bandwidth is swept. Saved
component-cost histograms allow independent reconstruction of every point.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import json
from importlib.metadata import version
import math
import multiprocessing
import os
import pickle
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

from neusim.npusim.backend.tessera_baselines import ARCHITECTURES
from neusim.npusim.frontend.llm_ops_lib import create_einsum_op
from neusim.npusim.frontend.tessera_workloads import native_llm_graph
from neusim.run_scripts.compare_tessera_native import ROOT, rows, sha, write_csv
from neusim.run_scripts.run_tessera_partitioned import config, native_record, bounded_map, trace_jobs
from neusim.run_scripts.summarize_tessera_native import number

NAMES = (*ARCHITECTURES, "Tessera-32", "Tessera-8", "Tessera-32-skew", "Tessera-8-skew")
METRICS = ("sa_ns", "vu_ns", "ici_ns", "hbm_bytes", "sram_bytes", "useful_macs")
_META = None
_CONFIGS = None
_GRID = None
_LATENCY = None
_REFERENCE_SEEDS = {}
_REFERENCE_HITS = 0


def initialize(metadata, grid):
    global _META, _CONFIGS, _GRID, _LATENCY
    _META, _GRID = metadata, grid
    # Source: native ChipConfig.hbm_latency_ns, also recorded in the manifest.
    _LATENCY = config("full").hbm_latency_ns
    _CONFIGS = {}
    for name in NAMES:
        grain = 8 if name.startswith("Tessera-8") else 32
        chip = config("skew" if name.endswith("-skew") else "full", grain=grain)
        if name in ARCHITECTURES:
            chip.tessera_parameters["baseline_architecture"] = name
        _CONFIGS[name] = chip


def time_at(record, bandwidth):
    hbm = max(math.ceil(record["hbm_bytes"] / (bandwidth / 1e9)), _LATENCY) if record["hbm_bytes"] else 0
    return max(record["sa_ns"], record["vu_ns"], record["ici_ns"], hbm)


def reference_record(op, chip, name):
    global _REFERENCE_HITS
    key=name,operator_identity(op)
    if key in _REFERENCE_SEEDS:
        _REFERENCE_HITS+=1
        return _REFERENCE_SEEDS[key]
    return native_record(op,chip)


def load_reference_seeds(spec, inputs):
    """Reuse checked EDP costs only with identical inputs, chip and engine code."""
    references=[spec];sources=[];excluded=[]
    for name in json.loads(spec.read_text())["runs"]:
        run=Path(name);manifest=json.loads((run/"manifest.json").read_text())
        assert manifest["status"]=="PASS"
        audit=run/"analysis/verification.json"
        assert json.loads(audit.read_text())["status"]=="PASS"
        grain=manifest["grain"];assert grain in (32,8)
        full_name=f"Tessera-{grain}"
        assert json.loads((run/"chip_config.json").read_text())==_CONFIGS[full_name].model_dump()
        assert sha(run/"workload_metadata.json")==sha(inputs/"workload_metadata.json")
        used=[run/"manifest.json",audit,run/"chip_config.json",run/"workload_metadata.json"]
        differences=[]
        for path,digest in manifest["source_sha256"].items():
            path=Path(path)
            if (path.is_relative_to(ROOT/"neusim/npusim") or path.is_relative_to(ROOT/"neusim/configs")
                    or path==ROOT/"neusim/run_scripts/run_tessera_partitioned.py"):
                if sha(path)!=digest:differences.append(str(path))
        # Codex: decision start — changed engine code is a cache miss; compute
        # that grain directly rather than reuse results across code revisions.
        if differences:
            excluded.append(dict(run=str(run),reason="engine source differs",paths=differences))
            references.extend(used)
            continue
        # Codex: decision end
        def checked(path):
            assert sha(path)==manifest["output_sha256"][path.name],path
            used.append(path)
            return path
        def save(signature,r):
            if r["policy"]!="sa" or r["variant"] not in ("full","skew"):return
            arch=full_name+("-skew" if r["variant"]=="skew" else "")
            cost={k:number(r[k]) for k in (*METRICS,"time_ns")}
            assert time_at(cost,_GRID[-1])==cost["time_ns"]
            key=arch,signature
            if key in _REFERENCE_SEEDS:
                assert all(_REFERENCE_SEEDS[key][k]==v for k,v in cost.items())
            else:
                # Recovery graph costs omit geometry; JSON null records that
                # absence. Shape seeds retain their original mapping JSON.
                _REFERENCE_SEEDS[key]=dict(cost,mapping_json=r.get("mapping_json","null"))
        signatures={}
        for r in rows(checked(run/"shape_results.csv")):
            if r["policy"]!="sa" or r["variant"] not in ("full","skew"):continue
            sid=int(r["shape_id"])
            if sid not in signatures:
                m,n,k=(int(r[x]) for x in ("M","N","K"))
                signatures[sid]=operator_identity(create_einsum_op([m,k],[k,n],"MK;KN->MN"))
            save(signatures[sid],r)
        graph=run/"graph_operators.jsonl"
        if graph.exists():
            signatures={}
            for line in checked(graph).open():
                r=json.loads(line);op=r["operator"];st=op["stats"]
                signatures[r["operator_id"]]=(op["opcode"],op["config_str"],op["input_tensor_shape_str"],
                    op["output_tensor_shape_str"],st["flop_count"],st["ici_time_ns"],
                    st["ici_traffic_inbound_bytes"],st["ici_traffic_outbound_bytes"],
                    json.dumps(op["tessera_spec"],sort_keys=True))
            for r in rows(checked(run/"graph_costs.csv")):save(signatures[int(r["operator_id"])],r)
        references.extend(used);sources.append(str(run))
    return references,dict(sources=sources,excluded=excluded,unique_costs=len(_REFERENCE_SEEDS))


def shape_job(job):
    sid, m, n, k = job
    result = []
    for name, chip in _CONFIGS.items():
        r = reference_record(create_einsum_op([m,k], [k,n], "MK;KN->MN"), chip, name)
        assert r["useful_macs"] == m*n*k
        assert time_at(r, _GRID[-1]) == r["time_ns"]
        result.append(dict(shape_id=sid, M=m, N=n, K=k, architecture=name,
                           **{key:r[key] for key in METRICS}, reference_time_ns=r["time_ns"],
                           mapping_json=r["mapping_json"]))
    return result


def trace_job(job):
    tag, batches = job
    model = _META["models"][tag]
    hist = Counter()
    checks = []
    for bid, requests, expected_macs in batches:
        template = native_llm_graph(model, requests, _CONFIGS["Tessera-32"], compact_layers=True)
        for name, chip in _CONFIGS.items():
            macs = reference_time = 0
            for original in template:
                r = native_record(original.model_copy(deep=True), chip)
                assert time_at(r, _GRID[-1]) == r["time_ns"]
                count = original.stats.count
                category = "attention" if original.opcode == "FlashAttention" else "gemm" if original.opcode == "Einsum" else "communication" if original.opcode == "AllReduce" else "vector"
                hist[(tag,name,category)+tuple(r[k] for k in METRICS)] += count
                macs += count*r["useful_macs"]
                reference_time += count*r["time_ns"]
            assert macs == expected_macs, (tag, bid, name, macs, expected_macs)
            checks.append(dict(model=tag,batch_id=bid,architecture=name,useful_macs=macs,
                               expected_macs=expected_macs,reference_time_ns=reference_time))
    return list(hist.items()), checks


def operator_identity(op):
    """The native_record cache identity excluding its separate chip key."""
    return (op.opcode,op.config_str,op.input_tensor_shape_str,op.output_tensor_shape_str,
            op.stats.flop_count,op.stats.ici_time_ns,op.stats.ici_traffic_inbound_bytes,
            op.stats.ici_traffic_outbound_bytes,json.dumps(op.tessera_spec,sort_keys=True))


def graph_job(job):
    return collect_graph_job(job,_META,_CONFIGS["Tessera-32"])


def collect_graph_job(job,metadata,chip):
    """Collect native operators before costing, preserving every invocation."""
    tag,batches=job
    operators={};references=[]
    for bid,requests,expected_macs in batches:
        counts=Counter()
        graph=native_llm_graph(metadata["models"][tag],requests,chip,compact_layers=True)
        for op in graph:
            # Codex: decision start — global reuse uses the existing native_record
            # cache identity, plus opcode. Counts and tensor names do not alter cost.
            signature=operator_identity(op)
            # Codex: decision end
            if signature not in operators:
                category="attention" if op.opcode=="FlashAttention" else "gemm" if op.opcode=="Einsum" else "communication" if op.opcode=="AllReduce" else "vector"
                # Internal IPC only: native Tensor's custom constructor cannot
                # round-trip its JSON UUID field. Pickle preserves native state.
                operators[signature]=(category,pickle.dumps(op,protocol=5))
            counts[signature]+=op.stats.count
        references.append((tag,bid,expected_macs,list(counts.items())))
    return operators,references


def graph_cost_job(job):
    oid,payload=job
    original=pickle.loads(payload)
    result=[]
    for name,chip in _CONFIGS.items():
        r=reference_record(original.model_copy(deep=True),chip,name)
        assert time_at(r,_GRID[-1])==r["time_ns"]
        result.append(dict(operator_id=oid,architecture=name,**{k:r[k] for k in METRICS},
                           reference_time_ns=r["time_ns"],mapping_json=r["mapping_json"]))
    return result


def verify_graph_dedup(metadata):
    """Compare the reused and direct paths on each real model's first batch."""
    checked=0
    for job in trace_jobs(metadata,1):
        old_hist,old_checks=trace_job(job)
        templates,batches=graph_job(job)
        costs={signature:graph_cost_job((oid,payload))
               for oid,(signature,(_,payload)) in enumerate(templates.items())}
        hist=Counter();checks=[]
        for tag,bid,expected,refs in batches:
            for name in NAMES:
                macs=ns=0
                for signature,repeat in refs:
                    r=next(x for x in costs[signature] if x["architecture"]==name)
                    hist[(tag,name,templates[signature][0])+tuple(r[k] for k in METRICS)]+=repeat
                    macs+=repeat*r["useful_macs"];ns+=repeat*r["reference_time_ns"]
                checks.append(dict(model=tag,batch_id=bid,architecture=name,useful_macs=macs,
                                   expected_macs=expected,reference_time_ns=ns))
                checked+=1
        assert dict(hist)==dict(old_hist)
        assert checks==old_checks
    return dict(status="PASS",real_model_architecture_checks=checked,
                native_component_histograms="bit-exact",batch_times_and_macs="bit-exact")


def run_graphs(pool,metadata,out,*,jobs=None):
    ids={};templates=[];categories=[];references=[];weights=Counter()
    with (out/"trace_graph_operators.jsonl").open("x") as operator_file, (out/"trace_graph_batches.jsonl").open("x") as batch_file:
        for ops,batches in bounded_map(pool,graph_job,trace_jobs(metadata,0) if jobs is None else jobs,block=32):
            for signature,(category,payload) in ops.items():
                if signature not in ids:
                    oid=len(templates);ids[signature]=oid
                    templates.append(payload);categories.append(category)
                    operator_file.write(json.dumps(dict(operator_id=oid,category=category,
                        operator=pickle.loads(payload).model_dump(mode="json")))+"\n")
            for tag,bid,expected,source_refs in batches:
                refs=[(ids[signature],repeat) for signature,repeat in source_refs]
                references.append((tag,bid,expected,refs))
                batch_file.write(json.dumps(dict(model=tag,batch_id=bid,expected_macs=expected,operators=refs))+"\n")
                for oid,repeat in refs:weights[tag,oid]+=repeat
            if len(references)%512<len(batches):
                operator_file.flush();batch_file.flush()
                print(f"Speed graph collection: {len(references)} steps, {len(templates)} distinct operators",flush=True)
    # Every unique native cache identity is evaluated once per architecture.
    # This changes neither the graph nor any per-operator accounting equation.
    del ids
    costs={}
    with (out/"trace_graph_costs.csv").open("x",newline="") as f:
        writer=None
        for count,result in enumerate(bounded_map(pool,graph_cost_job,enumerate(templates),chunksize=2),1):
            if writer is None:writer=csv.DictWriter(f,fieldnames=list(result[0]));writer.writeheader()
            writer.writerows(result)
            for r in result:costs[r["operator_id"],r["architecture"]]={k:r[k] for k in (*METRICS,"reference_time_ns")}
            if count%256==0:f.flush();print(f"Speed graph costs: {count}/{len(templates)} distinct operators",flush=True)
    hist=Counter()
    for (tag,oid),repeat in weights.items():
        for name in NAMES:
            r=costs[oid,name]
            hist[(tag,name,categories[oid])+tuple(r[k] for k in METRICS)]+=repeat
    with (out/"trace_batch_checks.csv").open("x",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=("model","batch_id","architecture","useful_macs","expected_macs","reference_time_ns"))
        writer.writeheader()
        for tag,bid,expected,refs in references:
            for name in NAMES:
                macs=sum(repeat*costs[oid,name]["useful_macs"] for oid,repeat in refs)
                ns=sum(repeat*costs[oid,name]["reference_time_ns"] for oid,repeat in refs)
                assert macs==expected,(tag,bid,name,macs,expected)
                writer.writerow(dict(model=tag,batch_id=bid,architecture=name,useful_macs=macs,
                                     expected_macs=expected,reference_time_ns=ns))
    return hist,len(references),len(templates)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--baseline-verification",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--workers",type=int,default=16)
    p.add_argument("--reuse-spec",type=Path)
    a=p.parse_args();out=a.out.resolve()
    if out.exists() or not out.is_relative_to(ROOT) or not 1<=a.workers<=16:
        p.error("fresh output inside NeuSim and at most 16 workers required")
    out.mkdir(parents=True)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:a.workers])
    cap=96_000_000_000//(a.workers+1);resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
    assert json.loads((a.inputs/"copy_verification.json").read_text())["status"]=="PASS"
    assert json.loads((a.baseline_verification/"verification.json").read_text())["status"]=="PASS"
    metadata=json.loads((a.inputs/"workload_metadata.json").read_text())
    grid=json.loads((a.baseline_verification/"bandwidth_grid.json").read_text())["bytes_per_second"]
    initialize(metadata,grid)
    # Codex: decision start — an explicit retained reuse manifest avoids
    # repeating the already audited EDP runs; direct native canaries gate use.
    spec=a.reuse_spec or a.inputs.parent/"speed_reuse_sources.json"
    seed_sources,seed_info=load_reference_seeds(spec,a.inputs) if spec.exists() else ([],{})
    dedup_check=verify_graph_dedup(metadata)
    dedup_check["saved_reference_cost_checks"]=_REFERENCE_HITS
    # Codex: decision end
    (out/"dedup_verification.json").write_text(json.dumps(dedup_check,indent=2)+"\n")
    workloads=a.inputs/"inputs/workloads.csv"
    source_paths=list((ROOT/"neusim/npusim/backend").glob("*.py"))
    source_paths += [ROOT/path for path in ("neusim/npusim/frontend/op_analysis_lib.py",
        "neusim/npusim/frontend/llm_ops_lib.py",
        "neusim/npusim/frontend/power_analysis_lib.py","neusim/npusim/frontend/tessera_workloads.py",
        "neusim/npusim/frontend/Operator.py","neusim/configs/chips/ChipConfig.py",
        "neusim/run_scripts/run_tessera_speed.py","neusim/run_scripts/run_tessera_partitioned.py")]
    for src in source_paths:
        dest=out/"source"/src.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dest);assert sha(src)==sha(dest)
    references=[workloads,a.inputs/"workload_metadata.json",a.baseline_verification/"bandwidth_grid.json"]
    references.extend(seed_sources)
    references.extend(Path(x) for x in metadata["traces"].values())
    manifest=dict(status="running",started_utc=datetime.now(timezone.utc).isoformat(),
        base_git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        python_version=sys.version,dependencies={name:version(name) for name in ("numpy","pydantic")},
        input_root=str(a.inputs.resolve()),workers=a.workers,affinity=sorted(os.sched_getaffinity(0)),
        per_process_address_space=cap,architectures=NAMES,bandwidth_bytes_per_second=grid,
        selection="Native NeuSim per-GEMM EDP at HBM4; all mappings/traffic fixed across bandwidth",
        metric="Time only; speedup=T_baseline/T_design. Full model or invocation-weighted single-operator E2E.",
        reused_reference_costs=seed_info,
        native_hbm_latency_ns=config("full").hbm_latency_ns,
        source_sha256={str(x.resolve()):sha(x) for x in references+source_paths})
    (out/"configs.json").write_text(json.dumps({k:v.model_dump() for k,v in _CONFIGS.items()},indent=2)+"\n")
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    shapes={};weights=defaultdict(Counter)
    for r in rows(workloads):
        sid=int(r["shape_id"]);dims=tuple(int(r[k]) for k in ("M","N","K"))
        assert sid not in shapes or shapes[sid]==dims
        shapes[sid]=dims;weights[sid][r["dataset"],r["cohort"]]+=int(r["repeat"])
    start=time.monotonic()
    try:
        totals=defaultdict(Counter);hist=Counter();batches=0
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context("fork")) as pool:
            with (out/"shape_reference.csv").open("x",newline="") as f:
                writer=None
                for count,result in enumerate(bounded_map(pool,shape_job,[(sid,*dims) for sid,dims in sorted(shapes.items())],chunksize=2),1):
                    if writer is None:writer=csv.DictWriter(f,fieldnames=list(result[0]));writer.writeheader()
                    writer.writerows(result)
                    for r in result:
                        for (dataset,cohort),repeat in weights[r["shape_id"]].items():
                            for bandwidth in grid:
                                acc=totals[dataset,cohort,r["architecture"],bandwidth]
                                acc["time_ns"]+=repeat*time_at(r,bandwidth)
                                acc["useful_macs"]+=repeat*r["useful_macs"]
                                acc["sa_ns"]+=repeat*r["sa_ns"]
                                acc["logical_gemms"]+=repeat
                    if count%256==0:f.flush();print(f"Speed operators: {count}/{len(shapes)}",flush=True)
            write_csv(out/"operator_bandwidth_totals.csv",[dict(dataset=d,cohort=c,architecture=arch,
                bandwidth_bytes_per_second=bw,**stats) for (d,c,arch,bw),stats in sorted(totals.items())])
            hist,batches,unique_graph_operators=run_graphs(pool,metadata,out)
        records=[];traces=defaultdict(Counter)
        for key,repeat in sorted(hist.items()):
            tag,arch,category,*values=key
            costs=dict(zip(METRICS,values))
            records.append(dict(model=tag,architecture=arch,category=category,repeat=repeat,**costs))
            for bandwidth in grid:
                acc=traces[tag,arch,bandwidth]
                acc["time_ns"]+=repeat*time_at(costs,bandwidth)
                acc["useful_macs"]+=repeat*costs["useful_macs"]
                acc["sa_ns"]+=repeat*costs["sa_ns"]
        write_csv(out/"trace_operator_histogram.csv",records)
        write_csv(out/"e2e_bandwidth_totals.csv",[dict(model=tag,architecture=arch,
            bandwidth_bytes_per_second=bw,**stats) for (tag,arch,bw),stats in sorted(traces.items())])
        manifest.update(status="PASS",operator_shapes=len(shapes),e2e_batches=batches,
            unique_graph_operators=unique_graph_operators,deduplicated_graph_verification=dedup_check,
            elapsed_seconds=time.monotonic()-start,
            output_sha256={x.name:sha(x) for x in out.iterdir() if x.is_file() and x.name!="manifest.json"})
    except BaseException as exc:
        manifest.update(status="FAIL",error=f"{type(exc).__name__}: {exc}")
        (out/"failure.txt").write_text(traceback.format_exc());raise
    finally:
        (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:v for k,v in manifest.items() if k not in ("source_sha256","output_sha256")}),flush=True)


if __name__=="__main__":main()
