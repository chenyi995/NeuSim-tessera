"""Independently audit invocation-weighted energy/time and render EDP figures."""
import argparse
from collections import Counter,defaultdict
import csv
import json
import math
import os
from pathlib import Path
import shutil
import traceback
import numpy as np

from neusim.run_scripts.compare_tessera_native import ROOT,rows,sha
from neusim.run_scripts.analyze_tessera_partitioned import copied_csv
from neusim.run_scripts.run_tessera_joint import MAIN,FIELDS


def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=2e-10,abs_tol=1e-16),(a,b)


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();run=a.run.resolve();out=a.out.resolve();assert out.is_relative_to(ROOT) and not out.exists()
    out.mkdir(parents=True);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
    shutil.copyfile(__file__,out/Path(__file__).name)
    result=dict(status="running",hash_checks=0,raw_cost_rows=0,weighted_field_checks=0,batch_checks=0)
    helpers=[Path(__file__),ROOT/"neusim/run_scripts/run_tessera_joint.py",
             ROOT/"neusim/run_scripts/analyze_tessera_partitioned.py",ROOT/"neusim/run_scripts/compare_tessera_native.py"]
    sources={str(path):sha(path) for path in helpers}
    def checked(path):
        sources[str(path)]=sha(path);return path
    try:
        manifest=json.loads(checked(run/"manifest.json").read_text());assert manifest["status"]=="PASS"
        for name,digest in manifest["output_sha256"].items():
            assert sha(checked(run/name))==digest,name;result["hash_checks"]+=1
        for path,digest in manifest["source_sha256"].items():
            path=Path(path);snapshot=run/"source"/path.relative_to(ROOT) if path.is_relative_to(ROOT) else None
            actual=snapshot if snapshot and snapshot.exists() else path
            assert sha(checked(actual))==digest,actual;result["hash_checks"]+=1
        base=Path(manifest["base"]);inputs=base/"inputs_snapshot"
        meta=json.loads(checked(inputs/"workload_metadata.json").read_text())
        groups=json.loads(checked(inputs/"workload_groups.json").read_text())
        cohort={(r["dataset"],r["cohort"]):r for r in meta["cohorts"]}
        tag_group={r["trace_tag"]:key for key,r in cohort.items() if r["trace_tag"]}
        weights=defaultdict(Counter);dims={};batches=[]
        for r in rows(checked(inputs/"inputs/workloads.csv")):
            group=r["dataset"],r["cohort"]
            if cohort[group]["trace_tag"]:continue
            sid=int(r["shape_id"]);weights["shape",sid][group]+=int(r["repeat"])
            shape=tuple(int(r[k]) for k in ("M","N","K"))
            assert sid not in dims or dims[sid]==shape;dims[sid]=shape
        for line in checked(base/"speed/trace_graph_batches.jsonl").open():
            r=json.loads(line);batches.append(r)
            for oid,count in r["operators"]:weights["graph",oid][tag_group[r["model"]]]+=count
        saved_weights=defaultdict(Counter)
        for r in rows(run/"invocation_weights.csv"):
            saved_weights[r["kind"],int(r["operator_id"])][r["dataset"],r["cohort"]]+=int(r["repeat"])
        assert dict(saved_weights)==dict(weights)
        cfgs=json.loads((run/"configs.json").read_text())
        config_keys={}
        for name in cfgs:
            arch,bw=name.rsplit("@",1);config_keys[arch,int(bw)]=len(config_keys)
        group_keys={(g["dataset"],g["cohort"]):i for i,g in enumerate(groups)}
        arrays=np.zeros((len(groups),len(config_keys),len(FIELDS)),dtype=np.float64)
        exact=defaultdict(Counter);identities=defaultdict(set);macs={};selected={};mapping_changes=Counter()
        for r in rows(run/"operator_costs.csv"):
            identity=r["kind"],int(r["operator_id"]);key=r["architecture"],int(r["bandwidth_bytes_per_second"])
            assert key in config_keys and key not in identities[identity]
            identities[identity].add(key);value=int(r["useful_macs"])
            assert identity not in macs or macs[identity]==value;macs[identity]=value
            if identity[0]=="shape":assert value==math.prod(dims[identity[1]])
            assert 0<=int(r["peak_live_bytes"])<=cfgs[f"{key[0]}@{key[1]}"]["vmem_size_MB"]*1024**2
            vals=np.array([float(r[f]) for f in FIELDS])
            component=sum(float(r[f]) for f in FIELDS if f.startswith(("static_energy_","dynamic_energy_")))
            close(component,r["energy_J"])
            for group,count in weights[identity].items():
                arrays[group_keys[group],config_keys[key]]+=count*vals
                exact[group,key]["useful_macs"]+=value*count
                exact[group,key]["time_ns"]+=int(r["time_ns"])*count
            # Count bandwidth-dependent changes for diagnostics. Energy/time
            # are still read from each independently re-selected native result.
            mapkey=identity,key[0]
            mapping=json.loads(r["mapping_json"])
            semantic=json.dumps({k:mapping[k] for k in ("geometry","memory_tile","phase_plans") if k in mapping},sort_keys=True)
            if mapkey in selected and semantic!=selected[mapkey]:mapping_changes[key[0]]+=1
            selected[mapkey]=semantic
            result["raw_cost_rows"]+=1
        assert len(identities)==manifest["operators"]
        assert all(s==set(config_keys) for s in identities.values())
        if not manifest["pilot"]:
            assert len(identities)==len(weights)
            for r in batches:
                assert sum(macs["graph",oid]*count for oid,count in r["operators"])==r["expected_macs"]
                result["batch_checks"]+=1
        totals=[]
        for r in rows(run/"totals.csv"):
            group=r["dataset"],r["cohort"];key=r["architecture"],int(r["bandwidth_bytes_per_second"])
            expected=arrays[group_keys[group],config_keys[key]]
            for i,field in enumerate(FIELDS):
                close(expected[i],r[field]);result["weighted_field_checks"]+=1
            for field in ("time_ns","useful_macs"):assert int(r[field])==exact[group,key][field]
            for field in (*FIELDS,"seconds","system_energy_J","edp_J_s","bandwidth_bytes_per_second","chips"):
                r[field]=float(r[field]) if field.endswith("_J") or field in ("seconds","edp_J_s") else int(r[field])
            close(r["edp_J_s"],r["time_ns"]*1e-9*r["energy_J"]*r["chips"])
            totals.append(r)
        lookup={(r["workload_id"],r["architecture"],r["bandwidth_bytes_per_second"]):r for r in totals}
        general=[];ablations=[];high=max(manifest["bandwidth_bytes_per_second"])
        for r in totals:
            wid,arch,bw=r["workload_id"],r["architecture"],r["bandwidth_bytes_per_second"]
            if arch in MAIN:
                plan=lookup[wid,"Planaria-32",bw]
                general.append(dict(**r,edp_gain_vs_planaria=plan["edp_J_s"]/r["edp_J_s"],
                    speedup_vs_planaria=plan["time_ns"]/r["time_ns"],energy_gain_vs_planaria=plan["energy_J"]/r["energy_J"],
                    average_system_power_W=r["system_energy_J"]/r["seconds"]))
            for variant in ("square","independent_noskew"):
                if arch.endswith("-"+variant):
                    full_name=arch.removesuffix("-"+variant);full=lookup[wid,full_name,bw]
                    gain=r["edp_J_s"]/full["edp_J_s"]
                    ablations.append(dict(workload_id=wid,workload=r["workload"],boundary=r["boundary"],
                        grain=int(full_name.split("-")[-1]),ablation=variant,bandwidth_bytes_per_second=bw,
                        full_edp_J_s=full["edp_J_s"],ablation_edp_J_s=r["edp_J_s"],edp_gain_ratio=gain,
                        edp_reduction_pct=100*(1-1/gain),speedup=r["time_ns"]/full["time_ns"],
                        energy_ratio=r["energy_J"]/full["energy_J"],sram_traffic_ratio=r["sram_bytes"]/full["sram_bytes"],
                        hbm_traffic_ratio=r["hbm_bytes"]/full["hbm_bytes"]))
        copied_csv(out/"general_edp.csv",general);copied_csv(out/"edp_ablations.csv",ablations)
        breakdown=[]
        for r in general:
            if r["architecture"] not in ("Tessera-32","Tessera-8") or r["bandwidth_bytes_per_second"]!=high:continue
            row={k:r[k] for k in ("workload_id","workload","boundary","architecture","seconds","system_energy_J","average_system_power_W")}
            for component in ("sa","vu","sram","hbm","ici","other"):
                energy=r[f"static_energy_{component}_J"]+r[f"dynamic_energy_{component}_J"]
                row[f"{component}_system_J"]=energy*r["chips"]
                row[f"{component}_pct"]=100*energy/r["energy_J"]
            row["static_pct"]=100*r["static_J"]/r["energy_J"]
            breakdown.append(row)
        copied_csv(out/"energy_breakdown.csv",breakdown)
        headline=[]
        for g in groups:
            wid=g["workload_id"]
            if (wid,"Planaria-32",high) not in lookup:continue
            r=dict(workload_id=wid,workload=g["name"],boundary=lookup[wid,"Planaria-32",high]["boundary"])
            for grain in (32,8):
                main=next(x for x in general if x["workload_id"]==wid and x["architecture"]==f"Tessera-{grain}" and x["bandwidth_bytes_per_second"]==high)
                r[f"T{grain}_EDP_gain_vs_Planaria"]=main["edp_gain_vs_planaria"]
                r[f"T{grain}_speedup_vs_Planaria"]=main["speedup_vs_planaria"]
                r[f"T{grain}_energy_gain_vs_Planaria"]=main["energy_gain_vs_planaria"]
                for var,label in (("square","non_square"),("independent_noskew","interconnect")):
                    r[f"T{grain}_{label}_EDP_gain"]=next(x["edp_gain_ratio"] for x in ablations if x["workload_id"]==wid and x["grain"]==grain and x["ablation"]==var)
            headline.append(r)
        copied_csv(out/"hbm4_headline.csv",headline)
        ranges=[]
        for boundary in sorted({r["boundary"] for r in general}):
            for grain in (32,8):
                measures=[("general_edp_gain_vs_planaria",[(r["workload"],r["edp_gain_vs_planaria"]) for r in general
                    if r["boundary"]==boundary and r["architecture"]==f"Tessera-{grain}" and r["bandwidth_bytes_per_second"]==high])]
                for variant in ("square","independent_noskew"):
                    measures.append((variant,[(r["workload"],r["edp_gain_ratio"]) for r in ablations
                        if r["boundary"]==boundary and r["grain"]==grain and r["ablation"]==variant]))
                for metric,values in measures:
                    if not values:continue
                    smallest=min(values,key=lambda x:x[1]);largest=max(values,key=lambda x:x[1])
                    ranges.append(dict(boundary=boundary,grain=grain,metric=metric,bandwidth_bytes_per_second=high,
                        minimum=smallest[1],minimum_workload=smallest[0],maximum=largest[1],maximum_workload=largest[0]))
        copied_csv(out/"ranges.csv",ranges)
        if not manifest["pilot"]:
            plot(out,groups,cohort,high)
        result.update(status="PASS",pilot=manifest["pilot"],mapping_changes_between_adjacent_bandwidths=dict(mapping_changes),
                      source_sha256=sources)
        write_note(out,run,manifest,headline,result)
    except BaseException:
        result["status"]="FAIL";(out/"failure.txt").write_text(traceback.format_exc());raise
    finally:
        (out/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    assert json.loads((out/"verification.json").read_text())==result
    (out/"artifact_hashes.json").write_text(json.dumps({p.name:sha(p) for p in out.iterdir() if p.is_file()},indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="source_sha256"}),flush=True)


def plot(out,groups,cohort,high):
    os.environ["MPLCONFIGDIR"]=str(out/"matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    general=list(rows(out/"general_edp.csv"));ablations=list(rows(out/"edp_ablations.csv"))
    colors=dict(zip(MAIN,("#999999","#252525","#0072B2","#E69F00","#009E73","#CC79A7","#D55E00")))
    plt.rcParams.update({"font.size":9,"pdf.fonttype":42})
    fig,axes=plt.subplots(3,4,figsize=(15,9))
    for ax,g in zip(axes.flat,groups):
        for arch in MAIN:
            subset=sorted((r for r in general if r["workload_id"]==g["workload_id"] and r["architecture"]==arch),key=lambda r:int(r["bandwidth_bytes_per_second"]))
            ax.plot([int(r["bandwidth_bytes_per_second"])/1e12 for r in subset],[float(r["edp_gain_vs_planaria"]) for r in subset],
                    color=colors[arch],label=arch,linewidth=1.8,marker="o" if arch.startswith("Tessera") else None,markersize=3)
        boundary="Model E2E" if cohort[g["dataset"],g["cohort"]]["trace_tag"] else "GEMM operator E2E"
        ax.set_title(g["name"]+"\n"+boundary,fontsize=9);ax.set_yscale("log",base=2)
        ax.set_xlabel("HBM bandwidth per chip (TB/s)");ax.set_ylabel("EDP gain vs Planaria-32 (x)");ax.grid(alpha=.2,which="both")
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=len(MAIN),loc="upper center",frameon=False)
    fig.tight_layout(rect=(0,0,1,.94))
    for ext in ("png","pdf"):fig.savefig(out/f"general_edp_bandwidth.{ext}",dpi=180)
    plt.close(fig)
    x=np.arange(len(groups))
    for variant,name,title in (("square","edp_nonsquare","Benefit of non-square support"),("independent_noskew","edp_interconnect","Benefit of array interconnection")):
        fig,ax=plt.subplots(figsize=(14,4.6))
        for index,grain in enumerate((32,8)):
            values=[float(next(r["edp_gain_ratio"] for r in ablations if r["workload_id"]==g["workload_id"] and int(r["grain"])==grain and r["ablation"]==variant)) for g in groups]
            bars=ax.bar(x+(index-.5)*.36,values,width=.36,color=colors[f"Tessera-{grain}"],label=f"Tessera-{grain}")
            ax.bar_label(bars,labels=[f"{v:.3f}" for v in values],fontsize=7,padding=3)
        count=sum(bool(cohort[g["dataset"],g["cohort"]]["trace_tag"]) for g in groups)
        ax.axvline(count-.5,color="gray",linestyle=":");ax.axhline(1,color="black",linewidth=.8)
        ax.set(xticks=x,xticklabels=[g["label"] for g in groups],ylabel="EDP gain: ablation / full Tessera (x)",title=f"{title} at HBM4 ({high/1e12:g} TB/s)")
        ax.margins(y=.2);ax.legend(frameon=False);fig.text(.26,.02,"Model E2E",ha="center");fig.text(.73,.02,"Invocation-weighted GEMM E2E",ha="center")
        fig.tight_layout(rect=(0,.04,1,1))
        for ext in ("png","pdf"):fig.savefig(out/f"{name}.{ext}",dpi=180)
        plt.close(fig)


def write_note(out,run,manifest,headline,audit):
    text=["# Joint EDP evaluation", "",f"Run: `{run}`. Simulator outputs are modeled estimates, not hardware measurements.","",
          "The general figure reports Planaria EDP / architecture EDP (greater than one is better). EDP is total system energy times total serial execution time, not summed operator EDP and not squared speedup.","",
          "All primary architectures use equal total PEs, clock, finite shared SRAM, native HBM timing/power and ideal per-PE SRAM bandwidth. Every feasible divisor tile is searched jointly with each architecture's legal geometries at each bandwidth. WS, SOSA, FlexSA and SISA have fixed architecture geometries but still optimize SRAM tiling. FlexSA retains its published deterministic mode schedule.","",
          "Power coefficients are the common native NeuSim coefficients, not per-architecture post-layout measurements. Interconnection affects the modeled cycles and SRAM/reduction traffic; physical link power is not separately calibrated.","",
          "The optimizer minimizes per-GEMM EDP within the existing output-stationary reuse order. It does not search alternative loop orders or globally optimize model EDP. Native FlashAttention Br/Bc selection is retained; resident QK/PV use the joint mapper. The skew control fixes the full design's selected map.","",
          "SRAM holds double-buffered operands, FP32 producer planes and a resident running output plane. A is fetched again per N SRAM tile, B per M SRAM tile; final C is written once and partial sums do not spill in this schedule. HBM energy and the native HBM latency floor, static power and regulator losses all enter selection.","",
          "Planaria's original optimizer already implements finite-SRAM loop promotion (planaria.code/src/optimizer/optimizer.py:128) and includes DRAM energy. Its inner tile/order search prioritizes cycles, breaking ties by energy; its outer composition selection minimizes EDP (src/simulator/simulator.py:489). This experiment uses the common NeuSim joint mapper, not that original optimizer.","",
          f"Pilot: {manifest['pilot']}. Raw rows audited: {audit['raw_cost_rows']}; weighted fields checked: {audit['weighted_field_checks']}; full model steps checked: {audit['batch_checks']}.","",
          "HBM4 results (derived from verified totals):","","| Workload | Planaria / T32 EDP | Planaria / T8 EDP | T32 non-square | T8 non-square | T32 interconnect | T8 interconnect |","|---|---:|---:|---:|---:|---:|---:|"]
    for r in headline:
        keys=("T32_EDP_gain_vs_Planaria","T8_EDP_gain_vs_Planaria","T32_non_square_EDP_gain","T8_non_square_EDP_gain","T32_interconnect_EDP_gain","T8_interconnect_EDP_gain")
        text.append("| "+r["workload"]+" | "+" | ".join(f"{r[k]:.6f}" for k in keys)+" |")
    (out/"REPORT.md").write_text("\n".join(text)+"\n")


if __name__=="__main__":main()
