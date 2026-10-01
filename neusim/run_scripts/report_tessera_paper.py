"""Build the three requested figures from independently checked NeuSim results."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil

from neusim.run_scripts.compare_tessera_native import ROOT, rows, sha
from neusim.run_scripts.analyze_tessera_partitioned import copied_csv, close, COMPONENTS
from neusim.run_scripts.summarize_tessera_native import number, table


def records(path):
    result=[]
    for r in rows(path):
        for key,value in r.items():
            try:r[key]=number(value)
            except ValueError:pass
        result.append(r)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--base",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--grain8-run",type=Path)
    a=p.parse_args();out=a.out.resolve();base=a.base.resolve()
    assert out.is_relative_to(ROOT) and not out.exists()
    out.mkdir(parents=True)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
    shutil.copyfile(__file__,out/Path(__file__).name)
    assert sha(Path(__file__))==sha(out/Path(__file__).name)
    inputs=base/"inputs_snapshot"
    metadata=json.loads((inputs/"workload_metadata.json").read_text())
    groups=json.loads((inputs/"workload_groups.json").read_text())
    metas={(r["dataset"],r["cohort"]):r for r in metadata["cohorts"]}
    sources={};checked=[]
    grain_runs={32:base/"grain32",8:a.grain8_run.resolve() if a.grain8_run else base/"grain8"}
    folders=[grain_runs[g]/"analysis" for g in (32,8)]
    folders.extend(base/f for f in ("speed/analysis","baseline_verification","speed_verification","speed_dedup_verification_v3"))
    for folder in folders:
        path=folder/"verification.json"
        audit=json.loads(path.read_text());assert audit["status"]=="PASS",path
        sources[str(path)]=sha(path);checked.append(audit)
    def read(path):
        sources[str(path)]=sha(path)
        return records(path)
    speed=read(base/"speed/analysis/speedup.csv")
    diagnostics=read(base/"speed/analysis/speed_diagnostics.csv")
    high=max(r["bandwidth_bytes_per_second"] for r in speed)
    fast={(r["workload_id"],r["architecture"]):r for r in speed if r["bandwidth_bytes_per_second"]==high}
    ablations=[];energy=[];cross_checks=0
    for grain in (32,8):
        folder=grain_runs[grain]
        operators={(r["dataset"],r["cohort"],r["variant"]):r for r in read(folder/"operator_totals.csv") if r["policy"]=="sa"}
        models={(r["model"],r["variant"]):r for r in read(folder/"e2e_totals.csv") if r["policy"]=="sa" and r["phase"]=="all"}
        op_ratios={(r["dataset"],r["cohort"],r["ablation"]):r for r in read(folder/"analysis/operator_ablations.csv") if r["policy"]=="sa"}
        model_ratios={(r["model"],r["ablation"]):r for r in read(folder/"analysis/e2e_ablations.csv") if r["policy"]=="sa" and r["phase"]=="all"}
        for g in groups:
            d,c=g["dataset"],g["cohort"];tag=metas[d,c]["trace_tag"]
            boundary="model_compute_e2e" if tag else "single_operator_e2e_weighted_sum"
            prefix=dict(workload_id=g["workload_id"],workload=g["name"],boundary=boundary,grain=grain)
            f=models[tag,"full"] if tag else operators[d,c,"full"]
            chips=metadata["models"][tag]["tensor_parallel_size"] if tag else 1
            comparison=fast[g["workload_id"],f"Tessera-{grain}"]
            for key in ("time_ns","useful_macs"):close(f[key],comparison[key]);cross_checks+=1
            for variant in ("square","independent_noskew"):
                b=models[tag,variant] if tag else operators[d,c,variant]
                ratio=model_ratios[tag,variant] if tag else op_ratios[d,c,variant]
                gain=(b["time_ns"]*b["energy_J"])/(f["time_ns"]*f["energy_J"])
                close(gain,ratio["edp_ratio"]);cross_checks+=1
                ablations.append(dict(**prefix,ablation=variant,bandwidth_bytes_per_second=high,
                    full_seconds=f["time_ns"]*1e-9,ablation_seconds=b["time_ns"]*1e-9,
                    full_system_energy_J=f["energy_J"]*chips,ablation_system_energy_J=b["energy_J"]*chips,
                    full_edp_J_s=f["time_ns"]*1e-9*f["energy_J"]*chips,
                    ablation_edp_J_s=b["time_ns"]*1e-9*b["energy_J"]*chips,
                    edp_gain_ratio=gain,edp_reduction_pct=100*(1-1/gain),
                    speedup=b["time_ns"]/f["time_ns"],energy_ratio=b["energy_J"]/f["energy_J"],
                    sram_traffic_ratio=b["sram_bytes"]/f["sram_bytes"],
                    hbm_traffic_ratio=b["hbm_bytes"]/f["hbm_bytes"]))
            er=dict(**prefix,seconds=f["time_ns"]*1e-9,system_energy_J=f["energy_J"]*chips,
                    static_pct=100*f["static_J"]/f["energy_J"],
                    average_system_power_W=f["energy_J"]*chips/(f["time_ns"]*1e-9))
            component_sum=0
            for component in COMPONENTS:
                value=sum(f[f"{kind}_energy_{component}_J"] for kind in ("static","dynamic"))
                component_sum+=value
                er[component+"_pct"]=100*value/f["energy_J"]
                er[component+"_system_J"]=value*chips
            close(component_sum,f["energy_J"])
            energy.append(er)
    copied_csv(out/"speedup.csv",speed)
    copied_csv(out/"speed_diagnostics.csv",diagnostics)
    copied_csv(out/"edp_ablations.csv",ablations)
    copied_csv(out/"energy_breakdown.csv",energy)
    # Plot the persisted/read-back tables; all figures share identical boundaries.
    speed=records(out/"speedup.csv");ablations=records(out/"edp_ablations.csv")
    os.environ["MPLCONFIGDIR"]=str(out/"matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size":9,"pdf.fonttype":42})
    architectures=("WS","Planaria-32","SOSA","FlexSA","SISA","Tessera-32","Tessera-8")
    colors=dict(zip(architectures,("#999999","#252525","#0072B2","#E69F00","#009E73","#CC79A7","#D55E00")))
    fig,axes=plt.subplots(3,4,figsize=(15,9))
    for ax,g in zip(axes.flat,groups):
        for arch in architectures:
            subset=sorted((r for r in speed if r["workload_id"]==g["workload_id"] and r["architecture"]==arch),
                          key=lambda r:r["bandwidth_bytes_per_second"])
            ax.plot([r["bandwidth_bytes_per_second"]/1e12 for r in subset],[r["speedup_vs_planaria"] for r in subset],
                    color=colors[arch],label=arch,linewidth=1.8,marker="o" if arch.startswith("Tessera") else None,markersize=3)
        boundary="Model E2E" if metas[g["dataset"],g["cohort"]]["trace_tag"] else "GEMM operator E2E"
        ax.set_title(g["name"]+"\n"+boundary,fontsize=9)
        ax.set_yscale("log",base=2)
        ax.set_xlabel("HBM bandwidth per chip (TB/s)")
        ax.set_ylabel("Speedup vs Planaria-32 (x)")
        ax.grid(alpha=.2,which="both")
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=len(architectures),loc="upper center",frameon=False)
    fig.tight_layout(rect=(0,0,1,.94))
    for ext in ("pdf","png"):fig.savefig(out/f"speedup_bandwidth.{ext}",dpi=180)
    plt.close(fig)
    names=[g["label"] for g in groups];x=np.arange(len(groups))
    for variant,filename,title in (("square","edp_nonsquare","Benefit of non-square support"),
                                   ("independent_noskew","edp_interconnect","Benefit of array interconnection")):
        fig,ax=plt.subplots(figsize=(14,4.6))
        for index,grain in enumerate((32,8)):
            values=[next(r["edp_gain_ratio"] for r in ablations if r["workload_id"]==g["workload_id"]
                         and r["grain"]==grain and r["ablation"]==variant) for g in groups]
            bars=ax.bar(x+(index-.5)*.36,values,width=.36,color=colors[f"Tessera-{grain}"],label=f"Tessera-{grain}")
            ax.bar_label(bars,labels=[f"{v:.3f}" for v in values],fontsize=7,padding=3)
        ax.axhline(1,color="black",linewidth=.8)
        model_count=sum(bool(metas[g["dataset"],g["cohort"]]["trace_tag"]) for g in groups)
        ax.axvline(model_count-.5,color="gray",linestyle=":",linewidth=1)
        ax.set(xticks=x,xticklabels=names,ylabel="EDP gain: ablation / full Tessera (x)",
               title=f"{title} at HBM4 ({high/1e12:g} TB/s)")
        ax.margins(y=.2);ax.legend(frameon=False)
        fig.text(.26,.02,"Model E2E",ha="center")
        fig.text(.73,.02,"Invocation-weighted GEMM E2E",ha="center")
        fig.tight_layout(rect=(0,.04,1,1))
        for ext in ("pdf","png"):fig.savefig(out/f"{filename}.{ext}",dpi=180)
        plt.close(fig)
    main=[]
    for g in groups:
        r=dict(workload=g["name"],boundary="model" if metas[g["dataset"],g["cohort"]]["trace_tag"] else "operator")
        for grain in (32,8):
            s=fast[g["workload_id"],f"Tessera-{grain}"]
            r[f"T{grain}_speedup_vs_Planaria"]=s["speedup_vs_planaria"]
            for variant,label in (("square","non_square"),("independent_noskew","interconnect")):
                ar=next(v for v in ablations if v["workload_id"]==g["workload_id"] and v["grain"]==grain and v["ablation"]==variant)
                r[f"T{grain}_{label}_EDP_gain"]=ar["edp_gain_ratio"]
        main.append(r)
    copied_csv(out/"hbm4_headline.csv",main)
    peak_rows=[]
    for boundary in sorted({r["boundary"] for r in speed}):
        for grain in (32,8):
            subset=[r for r in speed if r["boundary"]==boundary and r["architecture"]==f"Tessera-{grain}"
                    and r["bandwidth_bytes_per_second"]==high]
            row=dict(boundary=boundary,grain=grain,
                     hbm4_speedup_min=min(r["speedup_vs_planaria"] for r in subset),
                     hbm4_speedup_max=max(r["speedup_vs_planaria"] for r in subset))
            for variant in ("square","independent_noskew"):
                s=[r for r in ablations if r["grain"]==grain and r["boundary"]==boundary and r["ablation"]==variant]
                row[variant+"_edp_gain_min"]=min(r["edp_gain_ratio"] for r in s)
                row[variant+"_edp_gain_max"]=max(r["edp_gain_ratio"] for r in s)
            peak_rows.append(row)
    copied_csv(out/"ranges.csv",peak_rows)
    report=["# Tessera evaluation in native NeuSim", "",
      "All values are **derived analytical simulation estimates**, not hardware measurements. "
      f"The source run is `{base}`. The copied paper workload contains {len(groups)} cohorts; "
      "cohorts with complete traces use model-compute E2E, and operator-only cohorts use invocation-weighted single-operator E2E. "
      "No cross-boundary average is reported. Each architecture processes the same inputs and repeats.", "",
      "## Three primary figures", "",
      "1. `speedup_bandwidth.pdf/png`: pure time speedup, T_Planaria / T_design. "
      "Planaria-32, SISA, SOSA, FlexSA, WS, Tessera-32 and Tessera-8 use equal total PEs and clock. "
      "The scan points are copied from the existing paper bandwidth sweep. "
      "Every architecture selects its mapping using native NeuSim per-GEMM EDP at HBM4; "
      "that mapping and its traffic are frozen throughout the bandwidth scan. "
      "This measures bandwidth sensitivity under a fixed deployment, not bandwidth-specific mapping optimization.",
      "2. `edp_nonsquare.pdf/png`: EDP_square / EDP_full, both grains, at HBM4.",
      "3. `edp_interconnect.pdf/png`: EDP_independent_noskew / EDP_full, both grains, at HBM4. "
      "The disconnected control keeps the same small-array grain and skew-free timing. "
      "It disables merging, so it changes rereads, partial sums, reductions and achievable mappings. "
      "Dedicated interconnect area and circuit power are not separately modeled.", "",
      "![Speedup across HBM bandwidth](speedup_bandwidth.png)", "",
      "![Non-square EDP ablation](edp_nonsquare.png)", "",
      "![Interconnection EDP ablation](edp_interconnect.png)", "",
      "For the EDP tables, E and T each sum the selected boundary's invocations before multiplication. "
      "An EDP gain above one favors full Tessera. EDP reduction (%) = 100 * (1 - 1/gain). "
      "This is neither pure speedup nor average power. TP system energy sums ranks; elapsed time is synchronous per-rank service time. "
      "Bandwidth is per chip, expressed in decimal TB/s.", "",
      "## HBM4 results", "",table(records(out/"hbm4_headline.csv"),list(main[0]),4), "",
      "## Why each effect changes the results", "",
      "Fine fission can raise useful MAC occupancy; `speedup.csv` includes useful SA utilization "
      "(useful MACs divided by SA-active cycles and total PEs), plus E2E effective utilization "
      "using total elapsed time in the denominator. `speed_diagnostics.csv` records "
      "Tessera-8 versus Tessera-32 latency and the paired restored-skew controls. Restoring skew keeps "
      "the mapping and traffic fixed and restores the final drain term, following the reference ablation; "
      "it does not restore bank-release delays or add uncharacterized skew-register power. "
      "A compute-cycle saving changes E2E only when it reaches the critical component time. "
      "The sweep therefore exposes cases where HBM transfer/latency masks the compute improvement.", "",
      "Non-square support adds legal shapes; each ablation independently selects its native-EDP mapping. "
      "Interconnection can reduce regional rereads and partial-sum work by merging small arrays. "
      "`edp_ablations.csv` separates speed, energy and SRAM/HBM traffic ratios, so energy changes can be "
      "distinguished from time changes. Total energy includes static power over the entire E2E time; "
      "`energy_breakdown.csv` records component shares, static share and derived average power for both grains.", "",
      "ICI denotes inter-chip communication; the ablated connectivity is between array partitions. "
      "Native NoPG retains the configured ICI static term during intervals without transfers. "
      "The ICI share therefore does not measure Tessera's intra-array interconnect power.", "",
      "## Native model and comparison limits", "",
      "Only baseline array-cycle/useful-word kernels are ported from the frozen Fission source; "
      "the kernel verifier compares them exactly with the original implementation. All designs use "
      "NeuSim's shared-capacity tiling, HBM transfer-time and minimum-latency formula, VU issue model, "
      "complete-model graph and per-operator max-component overlap. Main comparisons force matrix work "
      "onto the SA consistently. Native automatic VU fallback is retained in the grain-run appendices.", "",
      "Per the user's SRAM ruling, each PE is provisioned with 16-bit input, 16-bit weight and 32-bit "
      "partial-sum supply/drain, with ideal double-buffer overlap preventing visible SRAM bandwidth stalls. "
      "Capacity and actual SRAM traffic still matter; dynamic SRAM energy uses the native energy-per-byte coefficient. "
      "This is an explicit ideal supply assumption, not a proof about a finite bank implementation.", "",
      "HBM energy includes native transfer/controller/PHY terms. DRAM commands, refresh and external cell-array "
      "energy are not separately characterized. Power coefficients are borrowed native whole-chip coefficients, "
      "not calibrated Tessera silicon. Both grains and ablations share those coefficients. "
      "The native tiler can choose very small K tiles, exposing array restarts in some architectures. "
      "Large baseline penalties under such schedules are model/tiler-dependent, not intrinsic hardware claims.", "",
      "Native HBM minimum latency is charged to each operator with nonzero external traffic. "
      "It can hide short-array compute savings even at HBM4. The native HBM dynamic term also integrates "
      "bandwidth-derived power over that active interval; it is not the external paper's fixed energy per actual byte. "
      "The per-grain `analysis/simulator_bridge.csv` retains matched operator-level paper/NeuSim comparisons.", "",
      "The full-model boundary includes native vector/attention/communication operators and saved model steps. "
      "It excludes queueing/arrival idle, host sampling, and the final cross-rank token-selection/gather not present "
      "in the saved graph. Operator-only cohorts include each GEMM's input read and output write; they do not "
      "claim an unsaved complete application graph.", "",
      "## Verification and reproduction", "",
      "The retained launchers, source snapshots, configuration JSON, copied input hashes and raw rows are in the "
      "run directories. Analysis reconstructs weighted totals, all bandwidth points and MAC counts independently. "
      "The speed run evaluates each distinct native operator cost once and preserves every original invocation. "
      "Reuse is checked bit-exact against direct real-model batches, including process-pool execution; "
      "the saved operator-ID inventory is independently reweighted for every batch and architecture. "
      "The report additionally checks HBM4 Tessera time/MAC equality between the separate speed and EDP runs. "
      "Every CSV is read back after writing. `verification.json` and `artifact_hashes.json` record the checks and artifacts. "
      "No simulation result was adjusted to match the paper."]
    (out/"REPORT.md").write_text("\n".join(report)+"\n")
    audit=dict(status="PASS",input_audits=len(checked),hbm4_cross_run_checks=cross_checks,
               source_sha256=sources,component_energy_conservation="PASS",csv_readback="PASS",
               primary_figures=["speedup_bandwidth","edp_nonsquare","edp_interconnect"])
    (out/"verification.json").write_text(json.dumps(audit,indent=2)+"\n")
    assert json.loads((out/"verification.json").read_text())==audit
    (out/"artifact_hashes.json").write_text(json.dumps({x.name:sha(x) for x in out.iterdir() if x.is_file()},indent=2)+"\n")
    print(table(main,list(main[0]),4))
    print(table(peak_rows,list(peak_rows[0]),4))


if __name__=="__main__":main()
