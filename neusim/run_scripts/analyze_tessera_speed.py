"""Independently audit the saved native bandwidth sweep, without simulation."""
from collections import Counter, defaultdict
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import resource
import shutil

from neusim.run_scripts.compare_tessera_native import ROOT, rows, sha
from neusim.run_scripts.analyze_tessera_partitioned import copied_csv, close
from neusim.run_scripts.summarize_tessera_native import number
from neusim.run_scripts.run_tessera import trace_batches, requests_from_batch

COSTS=("sa_ns","vu_ns","ici_ns","hbm_bytes","sram_bytes","useful_macs")


def parse_costs(row):
    return {key:number(row[key]) for key in COSTS}


def latency(cost, bandwidth, floor):
    # Source: native op_analysis_lib.fill_operators_execution_info.
    # Deliberately do not import the sweep's time_at implementation.
    byte_count=int(cost["hbm_bytes"]);bw=int(bandwidth)
    assert byte_count==cost["hbm_bytes"] and bw==bandwidth
    transfer=(byte_count*1_000_000_000+bw-1)//bw if byte_count else 0
    if transfer:transfer=max(transfer,floor)
    return max(cost["sa_ns"],cost["vu_ns"],cost["ici_ns"],transfer)


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();out=a.out.resolve()
    assert out.is_relative_to(ROOT) and not out.exists()
    out.mkdir(parents=True)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
    resource.setrlimit(resource.RLIMIT_AS,(8_000_000_000,8_000_000_000))
    shutil.copyfile(__file__,out/Path(__file__).name)
    assert sha(Path(__file__))==sha(out/Path(__file__).name)
    manifest=json.loads((a.run/"manifest.json").read_text())
    assert manifest["status"]=="PASS"
    hashes=0
    for path,digest in manifest["source_sha256"].items():
        path=Path(path)
        snapshot=a.run/"source"/path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        assert sha(snapshot if snapshot.exists() else path)==digest,path
        hashes+=1
    for path,digest in manifest["output_sha256"].items():
        assert sha(a.run/path)==digest,path
        hashes+=1
    inputs=Path(manifest["input_root"])
    metadata=json.loads((inputs/"workload_metadata.json").read_text())
    groups=json.loads((inputs/"workload_groups.json").read_text())
    grid=manifest["bandwidth_bytes_per_second"]
    floor=manifest["native_hbm_latency_ns"]
    names=set(manifest["architectures"])
    shapes={};weights=defaultdict(Counter);source_macs=Counter();source_calls=Counter()
    for r in rows(inputs/"inputs/workloads.csv"):
        sid=int(r["shape_id"]);dims=tuple(int(r[k]) for k in ("M","N","K"))
        assert sid not in shapes or shapes[sid]==dims
        shapes[sid]=dims
        key=r["dataset"],r["cohort"];repeat=int(r["repeat"])
        weights[sid][key]+=repeat;source_macs[key]+=repeat*math.prod(dims);source_calls[key]+=repeat
    totals=defaultdict(Counter);seen=set();shape_checks=0
    for sid,batch in itertools.groupby(rows(a.run/"shape_reference.csv"),key=lambda r:int(r["shape_id"])):
        assert sid not in seen;seen.add(sid)
        batch=list(batch);assert len(batch)==len(names) and {r["architecture"] for r in batch}==names
        for r in batch:
            costs=parse_costs(r);arch=r["architecture"]
            assert tuple(int(r[k]) for k in ("M","N","K"))==shapes[sid]
            assert costs["useful_macs"]==math.prod(shapes[sid])
            assert latency(costs,grid[-1],floor)==number(r["reference_time_ns"])
            previous=math.inf
            for bw in grid:
                t=latency(costs,bw,floor);assert t<=previous;previous=t
                for (d,c),repeat in weights[sid].items():
                    acc=totals[d,c,arch,bw]
                    acc["time_ns"]+=repeat*t;acc["useful_macs"]+=repeat*costs["useful_macs"]
                    acc["sa_ns"]+=repeat*costs["sa_ns"];acc["logical_gemms"]+=repeat
            shape_checks+=1
    assert seen==set(shapes) and len(shapes)==manifest["operator_shapes"]
    fields=0;summary_keys=set()
    for r in rows(a.run/"operator_bandwidth_totals.csv"):
        key=r["dataset"],r["cohort"],r["architecture"],number(r["bandwidth_bytes_per_second"])
        assert key not in summary_keys;summary_keys.add(key)
        for k,v in totals[key].items():close(v,number(r[k]));fields+=1
        assert totals[key]["useful_macs"]==source_macs[key[:2]]
        assert totals[key]["logical_gemms"]==source_calls[key[:2]]
    assert summary_keys==set(totals)
    expected={}
    for tag,path in metadata["traces"].items():
        model=metadata["models"][tag]
        for bid,raw in trace_batches(path):
            macs=sum(math.prod(int(r[k]) for k in ("M","N","K","gemm_batch","num_layers")) for r in raw)
            macs+=len(requests_from_batch(raw))*model["embedding_dim"]*math.ceil(model["vocab_size"]/model["tensor_parallel_size"])
            expected[tag,str(bid)]=macs
    batch_totals=defaultdict(Counter);batch_keys=set();batch_times={}
    for r in rows(a.run/"trace_batch_checks.csv"):
        key=r["model"],r["batch_id"],r["architecture"]
        assert key not in batch_keys;batch_keys.add(key)
        batch_times[key]=number(r["reference_time_ns"])
        assert int(r["useful_macs"])==int(r["expected_macs"])==expected[key[:2]]
        batch_totals[key[0],key[2]]["time_ns"]+=number(r["reference_time_ns"])
        batch_totals[key[0],key[2]]["useful_macs"]+=int(r["useful_macs"])
    assert batch_keys=={(m,b,n) for m,b in expected for n in names}
    assert len(expected)==manifest["e2e_batches"]
    # Rebuild the reuse inventory from saved operator IDs and invocation counts.
    # This audit does not call graph construction, native_record, or time_at.
    assert json.loads((a.run/"dedup_verification.json").read_text())["status"]=="PASS"
    op_categories={}
    for line in (a.run/"trace_graph_operators.jsonl").open():
        r=json.loads(line);oid=r["operator_id"]
        assert oid not in op_categories
        op_categories[oid]=r["category"]
    assert len(op_categories)==manifest["unique_graph_operators"]
    op_costs={}
    for r in rows(a.run/"trace_graph_costs.csv"):
        key=int(r["operator_id"]),r["architecture"]
        assert key not in op_costs
        costs=parse_costs(r)
        assert latency(costs,grid[-1],floor)==number(r["reference_time_ns"])
        op_costs[key]=costs
    assert set(op_costs)=={(oid,arch) for oid in op_categories for arch in names}
    inventory_hist=Counter();inventory_batches=defaultdict(Counter);seen_graph_batches=set()
    for line in (a.run/"trace_graph_batches.jsonl").open():
        r=json.loads(line);tag=r["model"];key=tag,str(r["batch_id"])
        assert key not in seen_graph_batches;seen_graph_batches.add(key)
        assert r["expected_macs"]==expected[key]
        for arch in names:
            macs=0;time_ns=0
            for oid,repeat in r["operators"]:
                assert isinstance(repeat,int) and repeat>0
                costs=op_costs[oid,arch]
                inventory_hist[(tag,arch,op_categories[oid])+tuple(costs[k] for k in COSTS)]+=repeat
                macs+=repeat*costs["useful_macs"]
                time_ns+=repeat*latency(costs,grid[-1],floor)
            assert macs==expected[key]
            assert time_ns==batch_times[key+(arch,)]
            inventory_batches[tag,arch]["time_ns"]+=time_ns
            inventory_batches[tag,arch]["useful_macs"]+=macs
    assert seen_graph_batches==set(expected)
    assert dict(inventory_batches)==dict(batch_totals)
    saved_hist=Counter()
    trace_totals=defaultdict(Counter);categories=defaultdict(Counter)
    for r in rows(a.run/"trace_operator_histogram.csv"):
        costs=parse_costs(r);repeat=int(r["repeat"])
        saved_hist[(r["model"],r["architecture"],r["category"])+tuple(costs[k] for k in COSTS)]+=repeat
        for bw in grid:
            acc=trace_totals[r["model"],r["architecture"],bw]
            t=repeat*latency(costs,bw,floor)
            acc["time_ns"]+=t;acc["useful_macs"]+=repeat*costs["useful_macs"]
            acc["sa_ns"]+=repeat*costs["sa_ns"]
            categories[r["model"],r["architecture"],bw,r["category"]]["time_ns"]+=t
    assert saved_hist==inventory_hist
    for (m,arch),st in batch_totals.items():
        for k,v in st.items():close(v,trace_totals[m,arch,grid[-1]][k]);fields+=1
    seen=set()
    for r in rows(a.run/"e2e_bandwidth_totals.csv"):
        key=r["model"],r["architecture"],number(r["bandwidth_bytes_per_second"])
        assert key not in seen;seen.add(key)
        for k,v in trace_totals[key].items():close(v,number(r[k]));fields+=1
    assert seen==set(trace_totals)
    cohort_meta={(r["dataset"],r["cohort"]):r for r in metadata["cohorts"]}
    configs=json.loads((a.run/"configs.json").read_text())
    results=[];decomposition=[]
    for g in groups:
        d,c=g["dataset"],g["cohort"];tag=cohort_meta[d,c]["trace_tag"]
        boundary="model_compute_e2e" if tag else "single_operator_e2e_weighted_sum"
        prefix=dict(workload_id=g["workload_id"],workload=g["name"],dataset=d,cohort=c,boundary=boundary)
        for bw in grid:
            st={arch:(trace_totals[tag,arch,bw] if tag else totals[d,c,arch,bw]) for arch in names}
            for arch in sorted(names):
                x=st[arch];chip=configs[arch]
                capacity=chip["num_sa"]*chip["sa_dim"]**2*chip["freq_GHz"]
                assert 0<x["useful_macs"]<=x["sa_ns"]*capacity
                results.append(dict(**prefix,architecture=arch,bandwidth_bytes_per_second=bw,
                    seconds=x["time_ns"]*1e-9,speedup_vs_planaria=st["Planaria-32"]["time_ns"]/x["time_ns"],
                    speedup_vs_ws=st["WS"]["time_ns"]/x["time_ns"],time_ns=x["time_ns"],
                    useful_macs=x["useful_macs"],sa_active_ns=x["sa_ns"],
                    sa_util_pct=100*x["useful_macs"]/(x["sa_ns"]*capacity),
                    e2e_util_pct=100*x["useful_macs"]/(x["time_ns"]*capacity)))
            for grain in (32,8):
                full=st[f"Tessera-{grain}"];skew=st[f"Tessera-{grain}-skew"]
                assert skew["time_ns"]>=full["time_ns"]
                decomposition.append(dict(**prefix,grain=grain,bandwidth_bytes_per_second=bw,
                    skew_removal_speedup=skew["time_ns"]/full["time_ns"],
                    skew_sa_active_ratio=skew["sa_ns"]/full["sa_ns"],
                    grain8_over_grain32_speedup=st["Tessera-32"]["time_ns"]/st["Tessera-8"]["time_ns"]))
    copied_csv(out/"speedup.csv",results)
    copied_csv(out/"speed_diagnostics.csv",decomposition)
    copied_csv(out/"model_categories.csv",[dict(model=m,architecture=arch,bandwidth_bytes_per_second=bw,
        category=c,**st) for (m,arch,bw,c),st in sorted(categories.items())])
    audit=dict(status="PASS",hash_checks=hashes,shape_architecture_checks=shape_checks,
               source_trace_batches=len(expected),trace_architecture_checks=len(batch_keys),
               independent_summary_field_checks=fields,bandwidth_monotonicity="PASS",
               input_mac_conservation="PASS",reference_histogram_time_conservation="PASS",
               reused_graph_inventory_conservation="PASS",unique_graph_operators=len(op_categories),
               csv_readback="PASS",run_manifest_sha256=sha(a.run/"manifest.json"))
    (out/"verification.json").write_text(json.dumps(audit,indent=2)+"\n")
    assert json.loads((out/"verification.json").read_text())==audit
    print(json.dumps(audit,indent=2))


if __name__=="__main__":main()
