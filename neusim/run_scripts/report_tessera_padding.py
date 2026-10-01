"""Independently audit and plot the padded-energy rectangular ablation."""
import argparse
from collections import Counter,defaultdict
import csv
import json
import math
import os
from pathlib import Path
import shutil

from neusim.run_scripts.report_tessera_ws import sha,rows,save_json,save_csv,close,figure_check
from neusim.run_scripts.run_tessera_joint import FIELDS

ARCHES=("WS","Tessera-8","Tessera-8-square")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",type=Path,required=True);parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args();run=a.run.resolve();out=a.out.resolve();assert not out.exists()
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
    manifest=json.loads((run/"manifest.json").read_text());assert manifest["status"]=="PASS" and not manifest["pilot"]
    assert manifest["sa_energy_accounting"]=="padded_tiles"
    for name,digest in manifest["output_sha256"].items():assert sha(run/name)==digest,name
    base=Path(manifest["base"]);inputs=base/"inputs_snapshot"
    groups=json.loads((inputs/"workload_groups.json").read_text());meta=json.loads((inputs/"workload_metadata.json").read_text())
    cohort={(r["dataset"],r["cohort"]):r for r in meta["cohorts"]}
    tag_group={r["trace_tag"]:key for key,r in cohort.items() if r["trace_tag"]}
    weights=defaultdict(Counter);shapes={}
    for r in rows(inputs/"inputs/workloads.csv"):
        group=r["dataset"],r["cohort"]
        if cohort[group]["trace_tag"]:continue
        identity="shape",int(r["shape_id"]);weights[identity][group]+=int(r["repeat"])
        shape=tuple(int(r[k]) for k in ("M","N","K"));assert identity not in shapes or shapes[identity]==shape
        shapes[identity]=shape
    batches=[json.loads(line) for line in (base/"speed/trace_graph_batches.jsonl").open()]
    for r in batches:
        for oid,count in r["operators"]:weights["graph",oid][tag_group[r["model"]]]+=count
    recorded=defaultdict(Counter)
    for r in rows(run/"invocation_weights.csv"):
        recorded[r["kind"],int(r["operator_id"])][r["dataset"],r["cohort"]]+=int(r["repeat"])
    assert recorded==weights
    config=json.loads((run/"configs.json").read_text());high=max(manifest["bandwidth_bytes_per_second"])
    assert set(config)=={f"{arch}@{high}" for arch in ARCHES}
    assert all(c["tessera_parameters"]["sa_energy_accounting"]=="padded_tiles" for c in config.values())
    fields=(*FIELDS,"charged_sa_macs","padding_macs")
    aggregates=defaultdict(Counter);identities=defaultdict(set);macs={};raw_count=0
    with (run/"operator_costs.csv").open() as stream:
        for r in csv.DictReader(stream):
            identity=r["kind"],int(r["operator_id"]);arch=r["architecture"]
            assert arch in ARCHES and arch not in identities[identity]
            assert int(r["bandwidth_bytes_per_second"])==high
            identities[identity].add(arch)
            values={k:float(r[k]) if k.endswith("_J") else int(r[k]) for k in fields}
            useful=values["useful_macs"]
            assert identity not in macs or macs[identity]==useful;macs[identity]=useful
            if identity[0]=="shape":assert useful==math.prod(shapes[identity])
            assert values["charged_sa_macs"]==useful+values["padding_macs"]
            assert values["padding_macs"]>=0
            close(sum(v for k,v in values.items() if k.startswith(("static_energy_","dynamic_energy_"))),values["energy_J"])
            assert int(r["peak_live_bytes"])<=config[f"{arch}@{high}"]["vmem_size_MB"]*1024**2
            for group,count in weights[identity].items():
                for k,v in values.items():aggregates[group,arch][k]+=v*count
            raw_count+=1
    assert len(identities)==manifest["operators"]==len(weights)
    assert all(keys==set(ARCHES) for keys in identities.values())
    for r in batches:assert sum(macs["graph",oid]*count for oid,count in r["operators"])==r["expected_macs"]
    totals=rows(run/"totals.csv");field_checks=0
    for r in totals:
        group=r["dataset"],r["cohort"];expected=aggregates[group,r["architecture"]]
        for k in fields:
            if k.endswith("_J"):close(r[k],expected[k])
            else:assert int(r[k])==expected[k]
            field_checks+=1
        close(r["edp_J_s"],float(r["energy_J"])*int(r["chips"])*int(r["time_ns"])*1e-9)
    lookup={(r["workload_id"],r["architecture"]):r for r in totals}
    normalized=[];headline=[]
    for g in groups:
        wid=g["workload_id"];ws=lookup[wid,"WS"];full=lookup[wid,"Tessera-8"];square=lookup[wid,"Tessera-8-square"]
        for arch in ARCHES:
            r=lookup[wid,arch]
            normalized.append(dict(**r,display_name={"WS":"Mono WS","Tessera-8":"Tessera","Tessera-8-square":"Tessera square only"}[arch],
                baseline_architecture="WS",baseline_edp_J_s=ws["edp_J_s"],edp_gain_vs_ws=float(ws["edp_J_s"])/float(r["edp_J_s"]),
                ablation="square",sa_energy_accounting="padded_tiles"))
        headline.append(dict(workload_id=wid,workload=g["name"],boundary=full["boundary"],
            tessera_edp_gain_vs_ws=float(ws["edp_J_s"])/float(full["edp_J_s"]),
            square_edp_gain_vs_ws=float(ws["edp_J_s"])/float(square["edp_J_s"]),
            non_square_edp_gain=float(square["edp_J_s"])/float(full["edp_J_s"]),
            full_charged_sa_macs=full["charged_sa_macs"],square_charged_sa_macs=square["charged_sa_macs"],
            full_padding_macs=full["padding_macs"],square_padding_macs=square["padding_macs"]))
    out.mkdir(parents=True);shutil.copy2(__file__,out/Path(__file__).name)
    assert sha(Path(__file__))==sha(out/Path(__file__).name)
    save_csv(out/"edp_nonsquare.csv",normalized);save_csv(out/"nonsquare_headline.csv",headline)
    for r in rows(out/"edp_nonsquare.csv"):
        ws=lookup[r["workload_id"],"WS"];x=lookup[r["workload_id"],r["architecture"]]
        expected=(float(ws["energy_J"])*int(ws["chips"])*int(ws["time_ns"]))/ (float(x["energy_J"])*int(x["chips"])*int(x["time_ns"]))
        close(r["edp_gain_vs_ws"],expected)
    for r in rows(out/"nonsquare_headline.csv"):
        close(r["non_square_edp_gain"],float(r["tessera_edp_gain_vs_ws"])/float(r["square_edp_gain_vs_ws"]))

    os.environ["MPLCONFIGDIR"]=str(out/"matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],"font.size":9,"pdf.fonttype":42,"ps.fonttype":42})
    # User decision start — show all source-backed E2E workloads in one panel.
    fig,ax=plt.subplots(figsize=(14,3.55));bars_checked=0
    for ax in (ax,):
        panel=normalized
        gg=[g for g in groups if any(r["workload_id"]==g["workload_id"] for r in panel)]
        ymax=max(float(r["edp_gain_vs_ws"]) for r in panel)
        assert min(float(r["edp_gain_vs_ws"]) for r in panel)>.9
        for index,arch in enumerate(("Tessera-8","Tessera-8-square")):
            vals=[float(next(r["edp_gain_vs_ws"] for r in panel if r["workload_id"]==g["workload_id"] and r["architecture"]==arch)) for g in gg]
            bars=ax.bar([i+(index-.5)*.36 for i in range(len(gg))],vals,width=.36,
                        color=("#3b7cb5","#f0913b")[index],edgecolor="0.25",linewidth=.5,
                        label=("Tessera","Tessera square only")[index],zorder=2)
            for bar,value in zip(bars,vals):close(bar.get_height(),value);bars_checked+=1
            ax.bar_label(bars,labels=[f"{v:.2f}" for v in vals],fontsize=7,padding=3)
        ax.set(xticks=list(range(len(gg))),xticklabels=[g["label"] for g in gg],ylim=(.9,ymax*1.14),
               ylabel="EDP gain over mono WS (×)")
        ax.set_yticks([1,*[v for v in ax.get_yticks() if 1<v<ax.get_ylim()[1]]])
        ax.axhline(1,color="0.45",linestyle="--",linewidth=.8,zorder=3)
        ax.grid(axis="y",linestyle=":",linewidth=.5,color="0.8");ax.set_axisbelow(True)
        for side in (0,1):ax.plot([side-.008,side+.008],[-.018,.018],transform=ax.transAxes,color="0.25",linewidth=1,clip_on=False)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.96),ncol=2,frameon=False)
    fig.suptitle(f"Non-square support at HBM4 ({high/1e12:g} TB/s), including padded compute energy",fontsize=11,y=.995)
    fig.text(.5,.015,"Per-workload E2E; mono WS = 1×.  Truncated y-axes start at 0.9×; higher is better.",ha="center",fontsize=9)
    fig.tight_layout(rect=(.005,.07,.995,.83),w_pad=2);text_checks=figure_check(fig)
    # User decision end
    for ext in ("png","pdf"):fig.savefig(out/f"edp_nonsquare.{ext}",dpi=180)
    plt.close(fig)
    note=["# Rectangular ablation with padded compute energy","",f"Run: `{run}`. All numbers below are derived from modeled NeuSim E2E totals.","",
          "All three compared configurations (mono WS, Tessera, and square-only Tessera) re-optimize SRAM tiles and legal array geometry at HBM4 using padded-tile dynamic compute energy. Tessera denotes Tessera-8.","",
          "For each live logical weight region, charged MACs equal streamed M rows times the allocated physical H×W tile. Thus a 16×14 region occupying 16×16 PEs is charged as 16×16. Both rectangular and square designs pay for their own padding; unassigned tiles and pipeline fill/drain bubbles are not counted as additional MACs. Useful MAC counts remain unchanged. Local padding does not add HBM transfers.","",
          "Native dynamic-SA energy uses charged MACs with the existing NeuSim coefficient and regulator loss. Static energy and SRAM/HBM accounting retain their native models. The power assumption charges a padded MAC like a useful MAC; it is rough modeling, not RTL-measured zero-switching power.","",
          "Each bar is mono WS EDP / design EDP for that workload. The tabulated non-square gain is square-only EDP / full Tessera EDP. All ratios use total energy times total serial time, not sums of operator EDP. FlashAttention retains its native outer tile selection and uses the corrected joint mapper for resident QK/PV phases.","",
          "| Workload | Tessera / WS gain | Square-only / WS gain | Non-square EDP gain |","|---|---:|---:|---:|"]
    for r in headline:note.append("| "+r["workload"]+" | "+" | ".join(f"{r[k]:.6f}" for k in ("tessera_edp_gain_vs_ws","square_edp_gain_vs_ws","non_square_edp_gain"))+" |")
    (out/"REPORT.md").write_text("\n".join(note)+"\n")
    sources=[run/"manifest.json",run/"totals.csv",run/"operator_costs.csv",run/"invocation_weights.csv",
             inputs/"inputs/workloads.csv",inputs/"workload_metadata.json",inputs/"workload_groups.json",
             base/"speed/trace_graph_batches.jsonl",Path(__file__).resolve()]
    result=dict(status="PASS",raw_rows_checked=raw_count,weighted_fields_checked=field_checks,
                full_model_batches_checked=len(batches),bars_checked=bars_checked,visible_text_bounds_checked=text_checks,
                sa_energy_accounting="padded_tiles",source_sha256={str(p):sha(p) for p in sources})
    save_json(out/"verification.json",result)
    save_json(out/"artifact_hashes.json",{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({k:v for k,v in result.items() if k!="source_sha256"},indent=2),flush=True)
    print(json.dumps(headline,indent=2),flush=True)


if __name__=="__main__":main()
