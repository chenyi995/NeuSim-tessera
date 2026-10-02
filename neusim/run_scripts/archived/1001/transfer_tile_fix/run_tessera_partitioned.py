"""Evaluate partition-aware native NeuSim on the archived shapes and full traces."""
from __future__ import annotations
import argparse
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import csv
from datetime import datetime, timezone
import itertools
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

from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.npusim.backend.tessera_partitioned import BACKEND, VARIANTS, Sink
from neusim.npusim.frontend.llm_ops_lib import create_einsum_op
from neusim.npusim.frontend.op_analysis_lib import fill_operators_execution_info
from neusim.npusim.frontend.tessera_workloads import native_llm_graph
from neusim.run_scripts.run_tessera import trace_batches, requests_from_batch
from neusim.run_scripts.compare_tessera_native import ROOT, BASE, sha, rows, write_csv

ENERGIES = [f"{kind}_energy_{part}_J" for kind in ("dynamic", "static")
            for part in ("sa", "vu", "sram", "hbm", "ici", "other")]
FIELDS = ["time_ns", "energy_J", "hbm_bytes", "sram_bytes", "sa_ns", "vu_ns", "sram_ns", "hbm_ns", "ici_ns",
          "useful_macs", "sa_macs", "vu_macs", "reduction_ops", "array_a_read_bytes", "array_b_read_bytes", "partial_write_bytes",
          "reduction_read_bytes", "reduction_write_bytes", "skew_tail_cycles", "static_J", "dynamic_J",
          "sa_calls", "vu_fallback_calls", "vector_calls"] + ENERGIES
_CACHE = OrderedDict()
_BASE = None
_METADATA = None
_GRAIN = 32


def config(variant, policy="sa", grain=None):
    # Sources: final-three-ablations/EXPERIMENT_RATIONALE.md (PE/frequency/HBM),
    # native ChipConfig (12 MiB units, VU resources, bandwidths and power).
    grain = _GRAIN if grain is None else grain
    return ChipConfig(name="Tessera-native", num_sa=1, sa_dim=128, freq_GHz=1,
                      vmem_size_MB=12, hbm_bw_GBps=2_800_000_000_000/1024**3,
                      array_backend=BACKEND, tessera_variant=variant,
                      # chenyi9: decision start — explicitly record ideal per-PE SRAM supply in new runs.
                      tessera_parameters={"grain": grain, "frequency_policy": "chip",
                                          "sram_bandwidth_model": "per_pe_double_buffer"},
                      # chenyi9: decision end
                      use_vu_for_small_matmul=policy == "native_auto")


def native_record(op, chip):
    key = (chip.model_dump_json(), op.config_str, op.input_tensor_shape_str, op.output_tensor_shape_str,
           op.stats.flop_count, op.stats.ici_time_ns, op.stats.ici_traffic_inbound_bytes,
           op.stats.ici_traffic_outbound_bytes, json.dumps(op.tessera_spec, sort_keys=True))
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    with redirect_stdout(Sink()):
        op = fill_operators_execution_info([op], chip)[0]
    s, d = op.stats, op.stats.tessera_details
    record = dict(time_ns=s.execution_time_ns, energy_J=s.total_energy_J,
                  hbm_bytes=s.memory_traffic_bytes, sram_bytes=d.get("sram_bytes", math.ceil(s.vmem_time_ns*chip.vmem_bw_GBps)),
                  sa_ns=s.sa_time_ns, vu_ns=s.vu_time_ns, sram_ns=s.vmem_time_ns, hbm_ns=s.memory_time_ns,
                  ici_ns=s.ici_time_ns, static_J=s.static_energy_J, dynamic_J=s.dynamic_energy_J,
                  useful_macs=d.get("useful_macs", 0), sa_calls=int(s.sa_time_ns > 0),
                  vu_fallback_calls=int(d.get("engine") == "VU" or any(p["engine"] == "VU" for p in d.get("phase_plans", []))),
                  vector_calls=int(not d.get("model")),
                  sa_macs=min(d.get("useful_macs", 0),d.get("sa_arithmetic_ops", 0)//2),
                  vu_macs=max(0,d.get("useful_macs", 0)-d.get("sa_arithmetic_ops", 0)//2),
                  charged_sa_macs=d.get("sa_arithmetic_ops",0)//2,
                  padding_macs=d.get("padding_macs",0),
                  padding_sram_read_bytes=d.get("padding_sram_read_bytes",0))
    for field in ("reduction_ops", "array_a_read_bytes", "array_b_read_bytes", "partial_write_bytes",
                  "reduction_read_bytes", "reduction_write_bytes", "skew_tail_cycles"):
        record[field] = d.get(field, 0)
    record.update({field: getattr(s, field) for field in ENERGIES})
    assert s.execution_time_ns >= max(s.sa_time_ns, s.vu_time_ns, s.vmem_time_ns, s.memory_time_ns, s.ici_time_ns)
    assert math.isclose(s.total_energy_J, sum(record[k] for k in ENERGIES), rel_tol=1e-12)
    record.update(peak_live_bytes=d.get("peak_live_bytes", 0),
                  mapping_json=json.dumps({key:d[key] for key in ("geometry", "memory_tile", "phase_plans", "engine", "B", "M", "N", "K",
                      "sram_bandwidth_model", "sram_bandwidth_bytes_per_ns", "sram_service_time_ns",
                      "sram_overlap_hidden_ns", "sa_energy_accounting", "charged_sa_macs", "padding_macs",
                      "sram_read_accounting", "padding_sram_read_bytes") if key in d}, separators=(",", ":")))
    assert record["peak_live_bytes"] <= chip.vmem_size_MB*1024**2
    _CACHE[key] = record
    if len(_CACHE) > 65536:
        _CACHE.popitem(last=False)
    return record


def shape_job(job):
    sid, m, n, k = job
    result=[]
    for policy in ("sa", "native_auto"):
        for variant in VARIANTS:
            chip=config(variant,policy)
            op=create_einsum_op([m,k],[k,n],"MK;KN->MN")
            record=native_record(op,chip)
            assert record["useful_macs"] == m*n*k
            result.append(dict(shape_id=sid,M=m,N=n,K=k,policy=policy,variant=variant,**record))
    return result


def add(target, record, repeat=1):
    for field in FIELDS:
        target[field] += repeat * record[field]
    target["peak_live_bytes"] = max(target["peak_live_bytes"], record["peak_live_bytes"])


def initialize(metadata, grain=32):
    global _METADATA, _GRAIN
    _METADATA=metadata
    _GRAIN=grain


def trace_job(job):
    tag, batches=job
    model=_METADATA["models"][tag]
    results=[]
    profiles=defaultdict(Counter)
    for bid, reqs, expected in batches:
        phase="decode" if all(q==1 for q,_ in reqs) else "prefill" if all(q>1 for q,_ in reqs) else "mixed"
        # Build graph once per batch. Cost-cache lookup ignores names and counts;
        # all repeated layers/requests retain their original multiplicities.
        template=native_llm_graph(model,reqs,config("full"),compact_layers=True)
        for policy in ("sa", "native_auto"):
            for variant in VARIANTS:
                chip=config(variant,policy)
                acc=Counter()
                for op in template:
                    # Native analysis mutates op stats. A fresh copy is essential
                    # for policy comparisons and the original tiler's compute input.
                    fresh=op.model_copy(deep=True)
                    record=native_record(fresh,chip)
                    add(acc,record,op.stats.count)
                    category="attention" if op.opcode == "FlashAttention" else "gemm" if op.opcode == "Einsum" else "communication" if op.opcode == "AllReduce" else "vector"
                    add(profiles[tag,policy,variant,category],record,op.stats.count)
                assert acc["useful_macs"] == expected, (tag,bid,acc["useful_macs"],expected)
                results.append(dict(model=tag,batch_id=bid,phase=phase,policy=policy,variant=variant,
                                    chips=model["tensor_parallel_size"],**dict(acc)))
    return results, [(key,dict(value)) for key,value in profiles.items()]


def totals_row(prefix, stats, chips=1):
    return dict(**prefix,**stats,seconds=stats["time_ns"]*1e-9,
                system_energy_J=stats["energy_J"]*chips,
                edp_J_s=stats["time_ns"]*1e-9*stats["energy_J"]*chips)


def bounded_map(pool, fn, jobs, block=64, chunksize=1):
    jobs=iter(jobs)
    while current:=list(itertools.islice(jobs,block)):
        yield from pool.map(fn,current,chunksize=chunksize)


def run_operators(args,out,pool,manifest):
    path=args.workloads
    shapes={}; weights=defaultdict(Counter); record_counts=Counter()
    for r in rows(path):
        sid=int(r["shape_id"]); shape=tuple(int(r[k]) for k in ("M","N","K"))
        assert sid not in shapes or shapes[sid]==shape
        shapes[sid]=shape
        group=r["dataset"],r["cohort"]
        weights[sid][group]+=int(r["repeat"])
        record_counts[group]+=1
    jobs=[(sid,*shape) for sid,shape in sorted(shapes.items())]
    if args.limit:
        # Pilot spans the sorted shape collection, rather than only tiny shapes.
        step=max(1,len(jobs)//args.limit)
        jobs=jobs[::step][:args.limit]
    totals=defaultdict(Counter); count=0
    with (out/"shape_results.csv").open("x",newline="") as f:
        writer=None
        for result in bounded_map(pool,shape_job,jobs,chunksize=2):
            if writer is None:
                writer=csv.DictWriter(f,fieldnames=list(result[0])); writer.writeheader()
            writer.writerows(result)
            for r in result:
                for group,repeat in weights[r["shape_id"]].items():
                    target=totals[group,r["policy"],r["variant"]]
                    add(target,r,repeat)
                    target["logical_gemms"]+=repeat
                    target["unique_shapes"]+=1
            count+=1
            if count%256==0:
                f.flush(); print(f"Operators: {count}/{len(jobs)} shapes",flush=True)
    assert count==len(jobs)
    records=[totals_row(dict(dataset=g[0],cohort=g[1],policy=p,variant=v),s)
             for (g,p,v),s in sorted(totals.items())]
    write_csv(out/"operator_totals.csv",records)
    manifest.update(operator_shapes=count,workload_records=sum(record_counts.values()))
    return records


def trace_jobs(metadata,limit):
    for tag,path in metadata["traces"].items():
        block=[]
        for index,(bid,raw) in enumerate(trace_batches(path)):
            if limit and index>=limit:
                break
            reqs=requests_from_batch(raw)
            model=metadata["models"][tag]
            expected=sum(int(r["M"])*int(r["N"])*int(r["K"])*int(r["gemm_batch"])*int(r["num_layers"]) for r in raw)
            expected+=len(reqs)*model["embedding_dim"]*math.ceil(model["vocab_size"]/model["tensor_parallel_size"])
            block.append((bid,reqs,expected))
            if len(block)==16:
                yield tag,block
                block=[]
        if block:
            yield tag,block


def run_traces(args,out,pool,metadata,manifest):
    totals=defaultdict(Counter); profile_totals=defaultdict(Counter); count=0
    with (out/"trace_batches.csv").open("x",newline="") as f:
        writer=None
        for result,profiles in bounded_map(pool,trace_job,trace_jobs(metadata,args.limit),block=32):
            if writer is None:
                writer=csv.DictWriter(f,fieldnames=list(result[0])); writer.writeheader()
            writer.writerows(result)
            for r in result:
                for phase in ("all",r["phase"]):
                    acc=totals[r["model"],r["policy"],r["variant"],phase]
                    add(acc,r); acc["batches"]+=1
            for key,record in profiles:
                add(profile_totals[tuple(key)],record)
            count+=len(result)//(2*len(VARIANTS))
            f.flush()
            print(f"E2E: {count} complete model steps",flush=True)
    records=[]
    for (tag,policy,variant,phase),stats in sorted(totals.items()):
        chips=metadata["models"][tag]["tensor_parallel_size"]
        records.append(totals_row(dict(model=tag,policy=policy,variant=variant,phase=phase,chips=chips),stats,chips))
    write_csv(out/"e2e_totals.csv",records)
    write_csv(out/"e2e_operator_profiles.csv",[dict(model=k[0],policy=k[1],variant=k[2],category=k[3],**s) for k,s in sorted(profile_totals.items())])
    manifest.update(e2e_batches=count)
    return records


def verify_totals(path,expected):
    got=list(rows(path)); assert len(got)==len(expected)
    for a,b in zip(expected,got):
        for key,value in a.items():
            assert str(value)==b[key],(key,value,b[key])


def main():
    p=argparse.ArgumentParser()
    # chenyi9: decision start — run the paper's copied inputs at both published grains.
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--archive",type=Path)
    inputs.add_argument("--workloads",type=Path)
    p.add_argument("--metadata",type=Path,default=ROOT/"configs/chips/tessera_revision.json")
    p.add_argument("--grain",type=int,choices=(8,32),default=32)
    # chenyi9: decision end
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--suite",choices=("operators","traces","all"),default="all")
    p.add_argument("--workers",type=int,default=16)
    p.add_argument("--limit",type=int,default=0)
    args=p.parse_args(); out=args.out.resolve()
    if args.archive:
        args.workloads=args.archive/"snapshot"/BASE/"inputs/workloads.csv"
    args.workloads=args.workloads.resolve();args.metadata=args.metadata.resolve()
    if out.exists() or not out.is_relative_to(ROOT) or not 1<=args.workers<=16:
        p.error("new output inside NeuSim and 1..16 workers required")
    out.mkdir(parents=True)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:args.workers])
    cap=96_000_000_000//(args.workers+1)  # Reserve memory for the run supervisor.
    resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
    sys.dont_write_bytecode=True
    metadata=json.loads(args.metadata.read_text())
    initialize(metadata,args.grain)
    refs=[args.workloads,args.metadata]
    refs.extend(Path(p) for p in metadata["traces"].values())
    refs.extend(Path(p) for p in metadata.get("tessera_references",[
        str(ROOT.parents[1]/p) for p in ("Tessera-HPCA2027-revision/sec/04_design.tex","gemmini/partition/TIMING_AUDIT.md")]))
    source_paths=list((ROOT/"neusim/npusim/backend").glob("*.py"))
    source_paths.extend(ROOT/p for p in ("neusim/npusim/frontend/op_analysis_lib.py","neusim/npusim/frontend/power_analysis_lib.py",
        "neusim/npusim/frontend/tessera_workloads.py","neusim/npusim/frontend/Operator.py","neusim/configs/chips/ChipConfig.py",
        "neusim/run_scripts/run_tessera_partitioned.py"))
    provenance={str(p.resolve()):sha(p) for p in refs+source_paths}
    for path in source_paths:
        dest=out/"source"/path.relative_to(ROOT); dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest); assert sha(path)==sha(dest)
    (out/"chip_config.json").write_text(config("full").model_dump_json(indent=2)+"\n")
    (out/"workload_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    manifest=dict(status="running",started_utc=datetime.now(timezone.utc).isoformat(),argv=sys.argv,
                  source_sha256=provenance,workers=args.workers,per_process_address_space=cap,
                  affinity=sorted(os.sched_getaffinity(0)),limit=args.limit,suite=args.suite,
                  grain=args.grain,workloads=str(args.workloads),metadata=str(args.metadata))
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    start=time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("fork")) as pool:
            if args.suite in ("all","operators"):
                records=run_operators(args,out,pool,manifest); verify_totals(out/"operator_totals.csv",records)
            if args.suite in ("all","traces"):
                records=run_traces(args,out,pool,metadata,manifest); verify_totals(out/"e2e_totals.csv",records)
        manifest.update(status="PASS",elapsed_seconds=time.monotonic()-start,
                        source_mac_checks="PASS",summary_readback="PASS",
                        output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!="manifest.json"})
    except BaseException as exc:
        manifest.update(status="FAIL",error=f"{type(exc).__name__}: {exc}",elapsed_seconds=time.monotonic()-start)
        (out/"failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:v for k,v in manifest.items() if k not in ("source_sha256","output_sha256")}),flush=True)


if __name__=="__main__":
    main()
