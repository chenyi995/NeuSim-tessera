"""Read-back audit, weighted tables and figures for partitioned native NeuSim.

All numbers are derived from retained raw rows. No simulation is performed here.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import itertools
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil

from neusim.run_scripts.compare_tessera_native import ROOT, BASE, sha, rows, write_csv
from neusim.run_scripts.run_tessera_partitioned import FIELDS, ENERGIES, VARIANTS
from neusim.run_scripts.summarize_tessera_native import GROUPS, number, table

COMPONENTS = ("sa", "vu", "sram", "hbm", "ici", "other")
PRIMARY = ("square", "independent_noskew")
NAMES = {"square": "Square only", "independent_noskew": "Independent, no skew", "skew": "Restored skew drain",
         "independent": "Independent WS", "ws": "Monolithic WS"}


def numeric(r):
    return {k:number(r[k]) for k in FIELDS+["peak_live_bytes"]}


def accumulate(a, r, repeat=1):
    for k in FIELDS:
        a[k] += r[k]*repeat
    a["peak_live_bytes"] = max(a["peak_live_bytes"],r["peak_live_bytes"])


def close(a,b):
    if isinstance(a,int) and isinstance(b,int):
        assert a==b,(a,b)
    else:
        assert math.isclose(a,b,rel_tol=2e-10,abs_tol=1e-11),(a,b)


def valid(r):
    assert r["sa_macs"]+r["vu_macs"]==r["useful_macs"]
    close(r["energy_J"],sum(r[k] for k in ENERGIES))
    close(r["energy_J"],r["static_J"]+r["dynamic_J"])
    assert r["time_ns"]>=max(r[k] for k in ("sa_ns","vu_ns","sram_ns","hbm_ns","ici_ns"))


def ratio_row(prefix,full,other,variant):
    t=other["time_ns"]/full["time_ns"]
    e=other["energy_J"]/full["energy_J"]
    return dict(**prefix,ablation=variant,speedup=t,energy_ratio=e,edp_ratio=t*e,
                time_reduction_pct=100*(1-1/t),energy_reduction_pct=100*(1-1/e),
                edp_reduction_pct=100*(1-1/(t*e)),
                sram_ratio=other["sram_bytes"]/full["sram_bytes"],
                hbm_ratio=other["hbm_bytes"]/full["hbm_bytes"],
                reduction_ratio=other["reduction_ops"]/full["reduction_ops"] if full["reduction_ops"] else 0,
                full_time_ns=full["time_ns"],ablation_time_ns=other["time_ns"],
                full_energy_J=full["energy_J"],ablation_energy_J=other["energy_J"])


def copied_csv(path,records):
    write_csv(path,records)
    actual=list(rows(path))
    assert len(actual)==len(records)
    for r,s in zip(records,actual):
        assert {k:str(v) for k,v in r.items()}==s


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--archive",type=Path)
    p.add_argument("--previous",type=Path)
    p.add_argument("--paper-inputs",type=Path)
    p.add_argument("--test-log",type=Path,required=True)
    p.add_argument("--frontend-test-log",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(); out=args.out.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        p.error("new analysis directory inside NeuSim required")
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
    resource.setrlimit(resource.RLIMIT_AS,(8_000_000_000,8_000_000_000))
    manifest=json.loads((args.run/"manifest.json").read_text())
    assert manifest["status"]=="PASS" and manifest["limit"]==0 and manifest["suite"]=="all"
    out.mkdir(parents=True)
    shutil.copyfile(__file__,out/Path(__file__).name)
    assert sha(Path(__file__))==sha(out/Path(__file__).name)
    hash_checks=0
    for path,expected in manifest["source_sha256"].items():
        snapshot=args.run/"source"/Path(path).relative_to(ROOT) if Path(path).is_relative_to(ROOT) else None
        # Validate the executed frozen source even when later work extends the live code.
        assert sha(snapshot if snapshot and snapshot.exists() else Path(path))==expected,path
        hash_checks+=1
    for path,expected in manifest["output_sha256"].items():
        assert sha(args.run/path)==expected,path
        hash_checks+=1
    groups=GROUPS
    if args.paper_inputs:
        copy_audit=json.loads((args.paper_inputs/"copy_verification.json").read_text())
        assert copy_audit["status"]=="PASS"
        for item in copy_audit["files"]:
            assert sha(Path(item["copied"]))==item["sha256"],item["copied"]
            hash_checks+=1
        paper_groups=json.loads((args.paper_inputs/"workload_groups.json").read_text())
        groups=[(r["dataset"],r["cohort"],r["name"]) for r in paper_groups]
        workload=args.paper_inputs/"inputs/workloads.csv"
    else:
        assert args.previous and args.archive
        replay=json.loads((args.previous/"fission_replay/reproduction_manifest.json").read_text())
        assert replay["status"]=="PASS"
        for path,expected in replay["source_sha256"].items():
            assert sha(args.archive/"snapshot"/path)==expected,path
            hash_checks+=1
        workload=args.archive/"snapshot"/BASE/"inputs/workloads.csv"
    weights=defaultdict(Counter); shapes={}; source_records=Counter()
    for r in rows(workload):
        sid=int(r["shape_id"]); group=r["dataset"],r["cohort"]
        shape=tuple(int(r[k]) for k in ("M","N","K"))
        assert sid not in shapes or shapes[sid]==shape
        shapes[sid]=shape; weights[sid][group]+=int(r["repeat"]); source_records[group]+=1
    operators=defaultdict(Counter); pathology=defaultdict(Counter); bottlenecks=defaultdict(Counter)
    examples=[]; shape_rows=0; seen=set()
    expected_variants={(p,v) for p in ("sa","native_auto") for v in VARIANTS}
    for sid,batch in itertools.groupby(rows(args.run/"shape_results.csv"),key=lambda r:int(r["shape_id"])):
        batch=list(batch); assert sid not in seen; seen.add(sid)
        plans={(r["policy"],r["variant"]):r for r in batch}
        assert set(plans)==expected_variants and len(batch)==len(expected_variants)
        parsed={key:numeric(r) for key,r in plans.items()}
        for key,r in plans.items():
            shape_rows+=1; st=parsed[key]; valid(st)
            assert tuple(int(r[k]) for k in ("M","N","K"))==shapes[sid]
            assert st["useful_macs"]==math.prod(shapes[sid])
            for group,repeat in weights[sid].items():
                accumulate(operators[group+key],st,repeat)
                operators[group+key]["logical_gemms"]+=repeat
                operators[group+key]["unique_shapes"]+=1
                critical=bottlenecks[group+key]
                critical["total_time_ns"]+=st["time_ns"]*repeat
                for component in ("sa","vu","sram","hbm","ici"):
                    critical[component+"_critical_time_ns"]+=st["time_ns"]*repeat*(st[component+"_ns"]==st["time_ns"])
            mapping=json.loads(r["mapping_json"])
            if key[0]=="sa":
                assert st["vu_macs"]==0
                short=mapping["memory_tile"][2]<min(mapping["geometry"][0],shapes[sid][2])
                for group,repeat in weights[sid].items():
                    a=pathology[group+(key[1],)]
                    a["all_time_ns"]+=st["time_ns"]*repeat
                    a["all_calls"]+=repeat
                    if short:
                        a["short_k_time_ns"]+=st["time_ns"]*repeat
                        a["short_k_calls"]+=repeat
                        a["short_k_shapes"]+=1
                if short and key[1]=="ws":
                    f=parsed["sa","full"]
                    examples.append(dict(shape_id=sid,M=shapes[sid][0],N=shapes[sid][1],K=shapes[sid][2],
                        ws_time_ns=st["time_ns"],full_time_ns=f["time_ns"],time_ratio=st["time_ns"]/f["time_ns"],
                        ws_mapping=r["mapping_json"],full_mapping=plans["sa","full"]["mapping_json"],
                        logical_gemms=sum(weights[sid].values())))
        f=parsed["sa","full"]; s=parsed["sa","square"]; z=parsed["sa","skew"]
        assert f["time_ns"]*f["energy_J"]<=s["time_ns"]*s["energy_J"]*(1+1e-12)
        isolated=parsed["sa","independent_noskew"]
        assert f["time_ns"]*f["energy_J"]<=isolated["time_ns"]*isolated["energy_J"]*(1+1e-12)
        full_map=json.loads(plans["sa","full"]["mapping_json"])
        skew_map=json.loads(plans["sa","skew"]["mapping_json"])
        for field in ("geometry","memory_tile","engine"):
            assert full_map[field]==skew_map[field],(sid,field)
        if full_map.get("sram_bandwidth_model")=="per_pe_double_buffer":
            for st in parsed.values():
                assert st["time_ns"]==max(st[k] for k in ("sa_ns","vu_ns","hbm_ns","ici_ns"))
        for k in ("hbm_bytes","sram_bytes","useful_macs","reduction_ops","array_a_read_bytes","partial_write_bytes"):
            assert f[k]==z[k],(sid,k)
        assert z["sa_ns"]-f["sa_ns"]==z["skew_tail_cycles"]
        assert z["time_ns"]>=f["time_ns"]
    assert seen==set(shapes)
    summary_checks=0
    for r in rows(args.run/"operator_totals.csv"):
        key=r["dataset"],r["cohort"],r["policy"],r["variant"]
        for k,v in operators[key].items():
            close(v,number(r[k]));summary_checks+=1
    e2e=defaultdict(Counter); e2e_rows=0; graph_checks=0; chips={}; seen_batches=defaultdict(set)
    for key,batch in itertools.groupby(rows(args.run/"trace_batches.csv"),key=lambda r:(r["model"],r["batch_id"])):
        batch=list(batch); plans={(r["policy"],r["variant"]):r for r in batch}
        assert set(plans)==expected_variants and len(batch)==len(expected_variants)
        assert key[1] not in seen_batches[key[0]]; seen_batches[key[0]].add(key[1])
        full=plans["sa","full"]; skew=plans["sa","skew"]
        for k in ("hbm_bytes","sram_bytes","reduction_ops","useful_macs"):
            assert full[k]==skew[k],(key,k)
        for (policy,variant),r in plans.items():
            e2e_rows+=1; st=numeric(r);valid(st)
            chips[r["model"]]=int(r["chips"])
            assert r["useful_macs"]==full["useful_macs"]
            for phase in ("all",r["phase"]):
                a=e2e[r["model"],policy,variant,phase];accumulate(a,st);a["batches"]+=1
    for r in rows(args.run/"e2e_totals.csv"):
        key=r["model"],r["policy"],r["variant"],r["phase"]
        for k,v in e2e[key].items():
            close(v,number(r[k]));summary_checks+=1
        close(float(r["system_energy_J"]),e2e[key]["energy_J"]*chips[r["model"]])
        close(float(r["edp_J_s"]),e2e[key]["energy_J"]*chips[r["model"]]*e2e[key]["time_ns"]*1e-9)
    # Independently re-read every trace's MAC count and batch IDs. This does not
    # construct a model graph or call any simulator cost function.
    metadata=json.loads((args.run/"workload_metadata.json").read_text())
    from neusim.run_scripts.run_tessera import trace_batches, requests_from_batch
    expected_macs=Counter()
    for tag,path in metadata["traces"].items():
        source_ids=set();model=metadata["models"][tag]
        for bid,raw in trace_batches(path):
            source_ids.add(str(bid))
            expected=sum(int(r["M"])*int(r["N"])*int(r["K"])*int(r["gemm_batch"])*int(r["num_layers"]) for r in raw)
            expected+=len(requests_from_batch(raw))*model["embedding_dim"]*math.ceil(model["vocab_size"]/model["tensor_parallel_size"])
            expected_macs[tag]+=expected;graph_checks+=1
        assert source_ids==seen_batches[tag]
        assert expected_macs[tag]==e2e[tag,"sa","full","all"]["useful_macs"]
    profile=defaultdict(Counter);profiles=[]
    for r in rows(args.run/"e2e_operator_profiles.csv"):
        tag,policy,variant=r["model"],r["policy"],r["variant"]
        st=numeric(r);accumulate(profile[tag,policy,variant],st)
        total=e2e[tag,policy,variant,"all"]
        profiles.append(dict(model=tag,policy=policy,variant=variant,category=r["category"],
            time_pct=100*st["time_ns"]/total["time_ns"],energy_pct=100*st["energy_J"]/total["energy_J"],
            system_energy_J=st["energy_J"]*chips[tag],seconds=st["time_ns"]*1e-9))
    for key,st in profile.items():
        for k,v in st.items():
            close(v,e2e[key+("all",)][k]);summary_checks+=1
    op_ablation=[];trace_ablation=[];energy=[];fallback=[]
    for d,c,name in groups:
        for policy in ("sa","native_auto"):
            full=operators[d,c,policy,"full"]
            for variant in NAMES:
                op_ablation.append(ratio_row(dict(workload=name,dataset=d,cohort=c,policy=policy),full,operators[d,c,policy,variant],variant))
    for tag in sorted(chips):
        for policy in ("sa","native_auto"):
            for phase in ("all","prefill","decode","mixed"):
                if (tag,policy,"full",phase) not in e2e:continue
                full=e2e[tag,policy,"full",phase]
                for variant in NAMES:
                    trace_ablation.append(ratio_row(dict(model=tag,policy=policy,phase=phase),full,e2e[tag,policy,variant,phase],variant))
        for variant in VARIANTS:
            sa=e2e[tag,"sa",variant,"all"]; auto=e2e[tag,"native_auto",variant,"all"]
            fallback.append(dict(model=tag,variant=variant,auto_vu_mac_pct=100*auto["vu_macs"]/auto["useful_macs"],
                sa_policy_seconds=sa["time_ns"]*1e-9,auto_policy_seconds=auto["time_ns"]*1e-9,
                sa_over_auto_time=sa["time_ns"]/auto["time_ns"],sa_over_auto_energy=sa["energy_J"]/auto["energy_J"]))
        for phase in ("all","prefill","decode","mixed"):
            if (tag,"sa","full",phase) not in e2e:continue
            f=e2e[tag,"sa","full",phase]
            er=dict(model=tag,phase=phase,chips=chips[tag],batches=f["batches"],seconds=f["time_ns"]*1e-9,
                    energy_per_chip_J=f["energy_J"],system_energy_J=f["energy_J"]*chips[tag],
                    static_pct=100*f["static_J"]/f["energy_J"],hbm_bytes=f["hbm_bytes"],sram_bytes=f["sram_bytes"])
            for component in COMPONENTS:
                joules=sum(f[f"{kind}_energy_{component}_J"] for kind in ("static","dynamic"))
                er[component+"_pct"]=100*joules/f["energy_J"]
                er[component+"_system_J"]=joules*chips[tag]
                for kind in ("static","dynamic"):
                    er[kind+"_"+component+"_system_J"]=f[f"{kind}_energy_{component}_J"]*chips[tag]
            energy.append(er)
    bridge=[]
    if args.paper_inputs:
        reference=args.paper_inputs/f"paper_reference/comparisons_g{manifest['grain']}.csv"
        paper_rows={(r["workload_id"],r["ablation"]):r for r in rows(reference)}
        for g in paper_groups:
            d,c,name=g["dataset"],g["cohort"],g["name"]
            # Historical paper independent uses WS; report the same control here.
            for variant in ("square","independent","skew"):
                prior=paper_rows[g["workload_id"],variant]
                current=ratio_row({},operators[d,c,"sa","full"],operators[d,c,"sa",variant],variant)
                bridge.append(dict(workload=name,ablation=variant,
                    paper_speedup=float(prior["speedup"]),paper_edp_ratio=float(prior["edp_ratio"]),
                    partitioned_speedup=current["speedup"],partitioned_edp_ratio=current["edp_ratio"]))
    else:
        reference=args.previous/"analysis_with_forceSA/totals.csv"
        old={(r["dataset"],r["cohort"],r["variant"]):r for r in rows(reference)}
        for d,c,name in groups:
            for new,old_name in (("square","square"),("independent","independent32"),("skew","skew")):
                row=dict(workload=name,ablation=new)
                for prefix in ("fission","native_forceSA"):
                    f=old[d,c,prefix+"_full"];s=old[d,c,prefix+"_"+old_name]
                    assert int(f["logical_gemms"])==operators[d,c,"sa","full"]["logical_gemms"]
                    row[prefix+"_speedup"]=float(s["time_ns"])/float(f["time_ns"])
                    row[prefix+"_edp_ratio"]=float(s["edp_J_ns"])/float(f["edp_J_ns"])
                current=ratio_row({},operators[d,c,"sa","full"],operators[d,c,"sa",new],new)
                row["partitioned_speedup"]=current["speedup"];row["partitioned_edp_ratio"]=current["edp_ratio"]
                bridge.append(row)
    path_rows=[dict(dataset=d,cohort=c,variant=v,
                   **{k:st[k] for k in ("all_time_ns","all_calls","short_k_time_ns","short_k_calls","short_k_shapes")},
                   short_k_time_pct=100*st["short_k_time_ns"]/st["all_time_ns"],
                   short_k_call_pct=100*st["short_k_calls"]/st["all_calls"])
               for (d,c,v),st in sorted(pathology.items())]
    generated={"operator_ablations.csv":op_ablation,"e2e_ablations.csv":trace_ablation,
               "energy_breakdown.csv":energy,"e2e_profiles.csv":profiles,"fallback_diagnostic.csv":fallback,
               "simulator_bridge.csv":bridge,"native_tiler_diagnostic.csv":path_rows,
               "native_tiler_examples.csv":sorted(examples,key=lambda r:r["time_ratio"],reverse=True)[:50]}
    generated["operator_bottlenecks.csv"]=[dict(dataset=d,cohort=c,policy=p,variant=v,**st,
        **{part+"_critical_time_pct":100*st[part+"_critical_time_ns"]/st["total_time_ns"]
           for part in ("sa","vu","sram","hbm","ici")}) for (d,c,p,v),st in sorted(bottlenecks.items())]
    for name,records in generated.items():copied_csv(out/name,records)

    # Tables consume the read-back CSVs rather than the in-memory result lists.
    def read_numbers(path):
        result=[]
        for r in rows(path):
            for k,v in r.items():
                try:r[k]=number(v)
                except ValueError:pass
            result.append(r)
        return result
    a=read_numbers(out/"e2e_ablations.csv");e=read_numbers(out/"energy_breakdown.csv")
    o=read_numbers(out/"operator_ablations.csv");br=read_numbers(out/"simulator_bridge.csv")
    main_table=[];mechanism=[];phase_table=[]
    for tag in sorted(chips):
        f=next(x for x in e if x["model"]==tag and x["phase"]=="all")
        ar={x["ablation"]:x for x in a if x["model"]==tag and x["phase"]=="all" and x["policy"]=="sa"}
        row={k:f[k] for k in ("model","batches","seconds","system_energy_J")}
        for variant in PRIMARY:
            row[variant+"_edp_reduction_pct"]=ar[variant]["edp_reduction_pct"]
        main_table.append(row)
        for variant in PRIMARY:
            r=ar[variant]
            mechanism.append({k:r[k] for k in ("model","ablation","speedup","energy_reduction_pct","sram_ratio","hbm_ratio")})
        for phase in ("prefill","decode","mixed"):
            subset=[x for x in a if x["model"]==tag and x["phase"]==phase and x["policy"]=="sa"]
            if not subset:continue
            row=dict(model=tag,phase=phase)
            for x in subset:
                if x["ablation"] in PRIMARY:row[x["ablation"]+"_edp_reduction_pct"]=x["edp_reduction_pct"]
            phase_table.append(row)
    copied_csv(out/"main_table.csv",main_table)
    energy_all=[{k:x[k] for k in ("model",)+tuple(c+"_pct" for c in COMPONENTS)+("static_pct",)} for x in e if x["phase"]=="all"]
    op_table=[]
    for _,_,name in groups:
        ar={x["ablation"]:x for x in o if x["workload"]==name and x["policy"]=="sa"}
        r=dict(workload=name)
        for variant in PRIMARY:r[variant+"_edp_reduction_pct"]=ar[variant]["edp_reduction_pct"]
        op_table.append(r)
    test_counts={}
    for label,path in (("backend",args.test_log),("frontend",args.frontend_test_log)):
        log=path.read_text(); match=re.search(r"Ran (\d+) tests",log)
        assert match and "\nOK\n" in log,(label,path)
        test_counts[label]=int(match[1]);shutil.copyfile(path,out/(label+"_tests.log"))
        assert sha(path)==sha(out/(label+"_tests.log"))
    audit=dict(status="PASS",source_output_hash_checks=hash_checks,weighted_summary_field_checks=summary_checks,
               unique_shapes=len(seen),shape_rows=shape_rows,source_workload_records=sum(source_records.values()),
               e2e_batches=graph_checks,e2e_rows=e2e_rows,backend_tests=test_counts["backend"],frontend_tests=test_counts["frontend"],
               fixed_skew_mapping_and_traffic="PASS",square_candidate_subset_edp="PASS",
               independent_candidate_subset_edp="PASS",
               original_trace_mac_conservation="PASS",component_energy_conservation="PASS",
               reference_input_hashes="PASS",source_matches_snapshot="PASS",
               csv_copy_readback="PASS",run_manifest_sha256=sha(args.run/"manifest.json"),
               reference_totals_sha256=sha(reference),grain=manifest.get("grain",32))
    (out/"verification.json").write_text(json.dumps(audit,indent=2)+"\n")
    assert json.loads((out/"verification.json").read_text())==audit
    report=["# Partition-aware Tessera: native NeuSim evaluation", "",
      "All numerical values below are **derived analytical simulation estimates**, not hardware measurements. "
      f"Raw run: `{args.run.resolve()}`. Configuration: `chip_config.json`; source and input hashes: `manifest.json`. "
      "The primary policy explicitly uses SA and chip frequency; native VU fallback is a separate diagnostic. "
      "SRAM supply follows the recorded per-PE double-buffer model, with native energy per byte. "
      "The primary EDP ablations are non-square support and interconnection; skew is a timing diagnostic.",
      "", "## Coverage and verification", "",
      f"{len(seen):,} unique matrix shapes from {sum(source_records.values()):,} archived records; "
      f"{graph_checks:,} complete model steps from all saved traces. "
      f"{test_counts['backend']} backend and {test_counts['frontend']} frontend tests passed. "
      f"Independent read-back checked {summary_checks:,} weighted summary fields. "
      "All source/input hashes, MAC counts, energy sums and fixed-mapping skew checks passed (`verification.json`).",
      "", "## Table 1. Complete model-step E2E", "",
      "Time and system energy sum every batch of each trace. TP system energy includes every rank; time is synchronous per-rank service time. "
      "Each benefit is full versus the named ablation: EDP reduction = 100 × (1 − E_full T_full / (E_ablation T_ablation)). "
      "Positive values favor full. This is model execution, not request queueing latency.", "",
      table(main_table,list(main_table[0]),3), "",
      "Non-square support can change physical occupancy and reduce regional rereads/partial sums. "
      "The independent no-skew comparison keeps the same small-block timing and removes merging/interconnection. "
      "The skew comparison keeps mapping and traffic fixed and changes only final drain. "
      "EDP optimization can trade a small latency increase for lower SRAM energy; latency and energy are separated in Table 3.",
      "", "## Table 2. Full Tessera energy breakdown", "",
      "The six component percentages each include static and dynamic energy and sum to the total. "
      "`static_pct` is an overlapping view across those components, not an extra component. "
      "HBM includes the native controller/PHY term. Borrowed whole-chip static power under native NoPG can dominate; "
      "these absolute joules are not calibrated Tessera chip power.", "",table(energy_all,list(energy_all[0]),3),
      "", "## Table 3. E2E mechanisms", "",
      "Speedup = ablation time / full time; energy reduction = 100 × (1 − full energy / ablation energy). "
      "SRAM and HBM ratios are ablation bytes / full bytes. Large SRAM reductions can have modest total-energy impact "
      "when static energy dominates. Equal HBM traffic is expected when the shared SRAM can retain partials.", "",
      table(mechanism,list(mechanism[0]),4),
      "", "## Table 4. Prefill, decode and mixed phases", "",
      "The same EDP reduction formula is recomputed from each phase's total time and energy. "
      "A pure prefill batch has only query lengths above one; a pure decode batch has only unit query lengths; other batches are mixed. "
      "A model-step ratio is not an individual-request latency ratio.", "",table(phase_table,list(phase_table[0]),3),
      "", "## Table 5. Matched archived workload ablations", "",
      "Every archived repeat weight is applied once. The same EDP reduction formula is used. "
      "These are isolated collected GEMMs, so this table does not represent a complete inference graph.", "",
      table(op_table,list(op_table[0]),3),
      "", "## Table 6. Why the simulators differ", "",
      ("The columns use the same copied paper shapes and repeat weights. "
      "`paper` is the retained current-paper comparison table. "
      "`partitioned` is this native NeuSim shared-SRAM/reduction model. " if args.paper_inputs else
      "The columns use the same archived shapes and repeat weights. "
      "`fission` is the verified final revision replay with external Planaria architecture-specific HBM templates. "
      "`native_forceSA` is the earlier SA-timing-only replay with native surroundings and native default frequency policy. "
      "`partitioned` is this full shared-SRAM/reduction model at explicit chip frequency. ")+
      "These are accounting comparisons, not an additive attribution of each change. "
      "The independent row deliberately uses traditional WS blocks to match the reference definition. "
      "Ratios above one favor full.", "",
      table(br,["workload","ablation","paper_edp_ratio","partitioned_edp_ratio"] if args.paper_inputs else
            ["workload","ablation","fission_edp_ratio","native_forceSA_edp_ratio","partitioned_edp_ratio"],4),
      "", "The final revision imports HBM traffic by architecture; this model derives it from shared-SRAM tiling. "
      "Regional activation rereads and partial writes stay on chip unless the chosen capacity schedule reloads inputs. "
      "The reference fixes skew energy; native static energy rises when skew extends the critical time. "
      "Native HBM latency and max-component overlap can hide an SA-cycle gain, while component static and SRAM activity costs "
      "change which mapping minimizes EDP. Power coefficients and memory units also differ; see the model documentation.",
      "Native minimum HBM time is charged per isolated operator, including small decode GEMMs. "
      "The corresponding HBM dynamic energy is bandwidth-derived power times that active time, not a fixed price per actual byte. "
      "`operator_bottlenecks.csv` reports the weighted execution time for which each component reaches the operator critical time; "
      "ties count toward each tied component. The reference instead prices the imported traffic per byte and includes its C64, "
      "initial-C-read, padding and per-thread-reload conventions. These different definitions survive matching the workload shapes.",
      "", "## Table 7. Native fallback diagnostic", "",
      "The original four-times SA/VU rule is retained. The fraction below counts executed matrix MACs, including attention phases. "
      "A result that executes on VU cannot establish the benefit of Tessera's SA. "
      "The full diagnostic CSV includes every architecture.", "",
      table([r for r in read_numbers(out/"fallback_diagnostic.csv") if r["variant"]=="full"],
            ["model","auto_vu_mac_pct","sa_policy_seconds","auto_policy_seconds","sa_over_auto_time"],4),
      "", "## Native tiler sensitivity", "",
      "`native_tiler_diagnostic.csv` counts SA mappings whose memory K tile is smaller than the logical regional K extent. "
      "The native factor search can choose these tiles to maximize M/N reuse. "
      "This model charges the resulting restarts and extra partial reductions, which the original aggregate SA timing did not see. "
      "Large monolithic-WS penalties are therefore tiler-dependent and must not be promoted as intrinsic hardware speedups. "
      "Exact examples with both mappings are retained in `native_tiler_examples.csv`.",
      "", "## Files and reproduction", "",
      "Full-precision CSVs contain every policy, architecture and phase, including cases unfavorable to full. "
      "`energy_breakdown.png/pdf` and `e2e_ablation.png/pdf` plot the checked tables. "
      "`e2e_profiles.csv` separates matrix, attention, vector and communication contributions. "
      "The retained analysis script rebuilds and verifies the tables from raw rows. "
      "Model equations, source anchors, commands and limitations are in `docs/tessera_partitioned.md`. "
      "The TP output boundary is rank-local vocabulary logits/argmax; a final cross-rank token-selection/gather and host sampling "
      "are outside the saved graph. The reported E2E is therefore model-compute service time. "
      "The native component-max overlap is an analytical roofline, not a cycle/event dependency simulation. "
      "Array timing is anchored to the supplied RTL audit, while dedicated Tessera interconnect, skew-register and SRAM-macro power remains uncharacterized."]
    (out/"REPORT.md").write_text("\n".join(report)+"\n")
    headline=dict(main_table=main_table,energy_breakdown_pct=energy_all,
                  e2e_edp_reduction_ranges_pct={v:[min(r[v+"_edp_reduction_pct"] for r in main_table),
                                                    max(r[v+"_edp_reduction_pct"] for r in main_table)] for v in PRIMARY},
                  static_energy_range_pct=[min(r["static_pct"] for r in energy_all),max(r["static_pct"] for r in energy_all)],
                  full_native_auto_vu_mac_pct={r["model"]:r["auto_vu_mac_pct"] for r in fallback if r["variant"]=="full"})
    (out/"headline.json").write_text(json.dumps(headline,indent=2)+"\n")
    assert json.loads((out/"headline.json").read_text())==headline
    # Standard standalone publication/export artifacts from checked CSV values.
    os.environ["MPLCONFIGDIR"]=str(out/"matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size":9,"pdf.fonttype":42})
    labels=[r["model"].replace("_","\n",1) for r in main_table]
    x=np.arange(len(labels))
    fig,ax=plt.subplots(figsize=(10,4),layout="constrained")
    bottom=np.zeros(len(labels))
    for component in COMPONENTS:
        heights=np.array([r[component+"_pct"] for r in energy_all])
        ax.bar(x,heights,bottom=bottom,label=component.upper() if component!="other" else "Other")
        bottom+=heights
    ax.set(xticks=x,xticklabels=labels,ylabel="Share of total modeled energy (%)",ylim=(0,105))
    ax.legend(ncol=6,loc="upper center",bbox_to_anchor=(.5,1.14),frameon=False)
    for suffix in ("pdf","png"):fig.savefig(out/f"energy_breakdown.{suffix}",dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),layout="constrained")
    for i,variant in enumerate(PRIMARY):
        ax.bar(x+(i-1)*.25,[r[variant+"_edp_reduction_pct"] for r in main_table],width=.25,label=NAMES[variant])
    ax.axhline(0,color="black",linewidth=.6)
    ax.set(xticks=x,xticklabels=labels,ylabel="Full Tessera EDP reduction (%)")
    ax.legend(ncol=3,loc="upper center",bbox_to_anchor=(.5,1.14),frameon=False)
    for suffix in ("pdf","png"):fig.savefig(out/f"e2e_ablation.{suffix}",dpi=180)
    plt.close(fig)
    (out/"artifact_hashes.json").write_text(json.dumps({p.name:sha(p) for p in out.iterdir() if p.is_file()},indent=2)+"\n")
    print(json.dumps(audit,indent=2))
    print(table(main_table,list(main_table[0]),3))
    print(table(energy_all,list(energy_all[0]),3))


if __name__=="__main__":main()
