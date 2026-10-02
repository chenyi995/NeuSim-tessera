"""Plot unchanged simulator coordinates and report all exploratory candidates."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/'figures'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def save_csv(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with path.open() as f:
        assert list(csv.DictReader(f))==[{k:str(v) for k,v in r.items()} for r in rows]


def main():
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
    verification=json.loads((HERE/'verification.json').read_text());assert verification['status']=='PASS'
    for n,h in verification['output_sha256'].items():assert sha(HERE/n)==h
    source=json.loads((HERE/'source_manifest.json').read_text())
    with (HERE/'all_points.csv').open() as f: rows=list(csv.DictReader(f))
    data={(r['workload_id'],r['tiling_policy'],r['architecture']):r for r in rows}
    ids=[w['workload_id'] for w in source['workloads']]
    titles={w['workload_id']:w['workload'] for w in source['workloads']}
    modes=['native_tail','joint_edp']
    labels={'native_tail':'Normal tiling + tail packing','joint_edp':'EDP tiling'}
    families=[('Tessera','#519000','o',[f'Tessera-{g}' for g in (64,32,16,8)]),
              ('Planaria','#168bc0','s',[f'Planaria-{g}' for g in (64,32,16,8)]),
              ('Independent WS','#333333','^',['WS']+[f'WS-independent-{g}' for g in (64,32,16,8)])]
    for r in rows:
        ops=2*float(r['useful_macs'])
        assert math.isclose(float(r['performance_density_gops_per_mm2']),ops/float(r['seconds'])/float(r['area_mm2'])/1e9,rel_tol=2e-12)
        assert math.isclose(float(r['energy_efficiency_gops_per_w']),ops/float(r['energy_J'])/1e9,rel_tol=2e-12)
    OUT.mkdir(exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    from matplotlib.transforms import Bbox
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
        'axes.titlesize':10,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42})
    handles=[Line2D([],[],color=c,marker=m,linewidth=1.35,markersize=4,label=n) for n,c,m,_ in families]

    def draw(ax,wid,mode):
        pts=[];xx=[];yy=[]
        for family,color,marker,arches in families:
            rr=[data[wid,mode,a] for a in arches]
            x=[float(r['performance_density_gops_per_mm2']) for r in rr]
            y=[float(r['energy_efficiency_gops_per_w']) for r in rr]
            line,=ax.plot(x,y,color=color,marker=marker,linewidth=1.35,markersize=4.2,zorder=3)
            assert list(line.get_xdata())==x and list(line.get_ydata())==y
            pts.extend((r['grain'],a,b,color) for r,a,b in zip(rr,x,y));xx.extend(x);yy.extend(y)
        xr=max(xx)-min(xx);yr=max(yy)-min(yy)
        ax.set_xlim(max(0,min(xx)-.12*xr),max(xx)+.17*xr)
        ax.set_ylim(max(0,min(yy)-.14*yr),max(yy)+.18*yr)
        ax.set_xlabel(r'Attention E2E performance / area (GOPS/mm$^2$)')
        ax.set_ylabel('Energy efficiency (GOPS/W)')
        ax.grid(color='#d5d9de',linestyle=':',linewidth=.6)
        ax.spines[['top','right']].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_title(labels[mode])
        return pts

    def annotate(fig,ax,points):
        renderer=fig.canvas.get_renderer();placed=[];markers=[]
        for _,x,y,_ in points:
            px,py=ax.transData.transform((x,y));markers.append(Bbox.from_extents(px-4,py-4,px+4,py+4))
        def overlap(a,b):return max(0,min(a.x1,b.x1)-max(a.x0,b.x0))*max(0,min(a.y1,b.y1)-max(a.y0,b.y0))
        offsets=[(8,7),(8,-10),(-8,7),(-8,-10),(0,14),(0,-15),(15,0),(-15,0),(16,18),(-16,18),(22,-16),(-22,-16),(30,9),(-30,9),(0,28),(0,-28)]
        for label,x,y,color in points:
            t=ax.annotate(label,(x,y),xytext=(0,0),textcoords='offset points',fontsize=7.5,color=color,
                bbox=dict(boxstyle='round,pad=0.04',facecolor='white',edgecolor='none',alpha=.8))
            candidates=[]
            for dx,dy in offsets:
                t.set_position((dx,dy));t.set_ha('left' if dx>0 else 'right' if dx<0 else 'center');t.set_va('bottom' if dy>0 else 'top' if dy<0 else 'center')
                box=t.get_window_extent(renderer).expanded(1.08,1.12)
                score=max(0,box.width*box.height-overlap(box,ax.bbox))*1e7+sum(overlap(box,z) for z in placed)*1e5+sum(overlap(box,z) for z in markers)*1e4+dx*dx+dy*dy
                candidates.append((score,dx,dy,box))
            _,dx,dy,box=min(candidates,key=lambda z:z[0]);placed.append(box)
            t.set_position((dx,dy));t.set_ha('left' if dx>0 else 'right' if dx<0 else 'center');t.set_va('bottom' if dy>0 else 'top' if dy<0 else 'center')

    with PdfPages(OUT/'all_workloads.pdf') as pdf:
        for wid in ids:
            fig,axes=plt.subplots(1,2,figsize=(12.2,3.7))
            points=[draw(ax,wid,mode) for ax,mode in zip(axes,modes)]
            suffix=' (model-config-derived KV trajectory)' if wid.startswith('falcon') else ''
            fig.suptitle(titles[wid]+suffix,fontsize=11,y=.99)
            fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.94),ncol=3,frameon=False)
            fig.text(.5,.018,'Complete attention operator E2E; QK + softmax + PV + memory. Equal 128 x 128 PEs; labels = minimum array side.',ha='center',fontsize=8,color='#555555')
            fig.subplots_adjust(left=.065,right=.986,bottom=.18,top=.78,wspace=.25)
            fig.canvas.draw()
            for ax,pts in zip(axes,points):annotate(fig,ax,pts)
            fig.savefig(OUT/(wid+'.pdf'));fig.savefig(OUT/(wid+'.png'),dpi=170);pdf.savefig(fig);plt.close(fig)
    fig,axes=plt.subplots(len(ids),2,figsize=(12.2,12.4))
    for i,wid in enumerate(ids):
        for j,mode in enumerate(modes):
            pts=draw(axes[i,j],wid,mode)
            axes[i,j].set_title(titles[wid]+' | '+labels[mode],fontsize=10)
    fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False)
    fig.subplots_adjust(left=.065,right=.985,bottom=.05,top=.96,hspace=.53,wspace=.25)
    fig.savefig(OUT/'overview.png',dpi=160);fig.savefig(OUT/'overview.pdf');plt.close(fig)

    # Comparable-grain ratios and strict coordinate dominance. A same-grain
    # energy win alone is not described as Pareto or whole-curve dominance.
    comparisons=[];dominance=[]
    for wid in ids:
        for mode in modes:
            ws=[data[wid,mode,a] for a in families[-1][3]]
            dom={f:0 for f in ('Tessera','Planaria')}
            for family in dom:
                for b in ws:
                    x=float(b['performance_density_gops_per_mm2']);y=float(b['energy_efficiency_gops_per_w'])
                    if any(float(data[wid,mode,f'{family}-{g}']['performance_density_gops_per_mm2'])>=x and
                           float(data[wid,mode,f'{family}-{g}']['energy_efficiency_gops_per_w'])>=y for g in (64,32,16,8)):
                        dom[family]+=1
            dominance.append(dict(workload_id=wid,tiling_policy=mode,WS_points=len(ws),**dom))
            for g in (64,32,16,8):
                b=data[wid,mode,f'WS-independent-{g}']
                for family in ('Tessera','Planaria'):
                    r=data[wid,mode,f'{family}-{g}']
                    comparisons.append(dict(workload_id=wid,tiling_policy=mode,family=family,grain=g,
                        energy_efficiency_gain=float(b['energy_J'])/float(r['energy_J']),
                        energy_reduction_pct=100*(1-float(r['energy_J'])/float(b['energy_J'])),
                        performance_density_gain=float(r['performance_density_gops_per_mm2'])/float(b['performance_density_gops_per_mm2']),
                        latency_ratio=float(r['seconds'])/float(b['seconds']),
                        sram_traffic_ratio=float(r['sram_bytes'])/float(b['sram_bytes']),
                        padded_compute_ratio=float(r['charged_sa_macs'])/float(b['charged_sa_macs'])))
    save_csv(HERE/'comparisons.csv',comparisons);save_csv(HERE/'dominance.csv',dominance)
    lines=['# Exploratory LLM attention PPA candidates','',
        '**Status:** simulated with the unchanged NeuSim `merged_load_v2` / `transfer_only` implementation; source/readback and energy checks PASS. These are attention-operator E2E results, not whole-model inference results. All four selected cohorts are shown.','',
        '[All four pairs](figures/all_workloads.pdf) | [Overview](figures/overview.png) | [Raw plotted coordinates](all_points.csv)','',
        '## Input scope and provenance','',
        '| Workload | Full-envelope M | KV length | Head dimension | Retained source scope |','|---|---:|---:|---:|---|']
    for w in source['workloads']:
        count=f"{w['source_cases']} recorded steps in one source trajectory" if w['workload_id'].startswith('falcon') else f"{w['requests']} requests; {w['source_cases']} chunk/layer-class cases"
        lines.append(f"| {w['workload']} | {w['M_min']}–{w['M_max']} | {w['KV_min']}–{w['KV_max']} | {w['head_dimensions']} | {count} |")
    lines+=['',
        'M is Q × (query heads / KV heads) for one shared KV group. Query blocking may make the executed SRAM-tile M smaller. Different requests retain different KV matrices and are never concatenated into a fictitious shared-KV GEMM.',
        '',
        'Falcon is a **derived architecture scenario**: the pinned official Falcon-7B config supplies the MQA heads, head width and layer count; the complete recorded Qwen Dolly-long decode supplies the KV-length trajectory. It is not an observed Falcon GPU run or a claim that both tokenizers produce identical lengths. In `cases.csv`, Falcon `request_id` stores the source step index; `source_manifest.json` therefore counts step identities, not independent requests. The sequence remains serial.',
        '',
        'Phi-2 preserves every prefill chunk of its original chat trace, including small residual chunks and the recorded context lengths. The source trace includes contexts beyond the model metadata maximum; they are retained as source workload scenarios, not validated model-quality settings. CacheBlend and EPIC preserve all source requests, selective/full layer classes and repeat counts. Dense projections, FFNs and serving arrival gaps are outside the explicitly measured attention-operator boundary.',
        '',
        'Official model source: https://huggingface.co/tiiuae/falcon-7b/blob/main/config.json . Local source filenames, exact source rows and SHA-256 hashes are in `cases.csv` and `source_manifest.json`. The original QK and PV GEMMs are checked against the derived fused-attention dimensions.',
        '',
        '## Comparable-grain outcome','',
        'All following numbers are derived from this run’s simulated time and energy. Ratios compare the same minimum array side and equal total PEs. Values above one mean higher energy efficiency; they do not by themselves establish a performance-density win.','',
        '| Workload | Tiling | Tessera-8 / independent WS-8 energy efficiency | Planaria-8 / independent WS-8 energy efficiency | Tessera-8 / WS-8 performance density |','|---|---|---:|---:|---:|']
    for wid in ids:
        for mode in modes:
            t=next(r for r in comparisons if (r['workload_id'],r['tiling_policy'],r['family'],r['grain'])==(wid,mode,'Tessera',8))
            p=next(r for r in comparisons if (r['workload_id'],r['tiling_policy'],r['family'],r['grain'])==(wid,mode,'Planaria',8))
            lines.append(f"| {titles[wid]} | {labels[mode]} | {t['energy_efficiency_gain']:.2f}× | {p['energy_efficiency_gain']:.2f}× | {t['performance_density_gain']:.2f}× |")
    lines+=['',
        'No workload or individual point is removed because it fails the desired ordering. The two RAG cohorts show the more useful energy trend in both mapping policies; Falcon MQA and Phi-2 remain counterexamples to the assumption that moderate M plus irregular K/N is sufficient. The independent WS curve can still extend farther right because the connected designs pay more area and the native bulk/tail policy need not minimize latency.',
        '',
        '`comparisons.csv` includes SRAM traffic and padded-compute ratios, separating energy benefit from speed/area benefit. `dominance.csv` checks strict two-coordinate dominance of each of the five independent-WS points by measured Tessera/Planaria points. It does not interpolate or imply that every black point is dominated.',
        '',
        '## Unchanged experimental accounting','',
        'All families use 128×128 total PEs. Tessera/Planaria minimum side: 64, 32, 16, 8. Independent WS: one 128×128, four 64×64, sixteen 32×32, sixty-four 16×16, or 256 8×8 arrays. The current RTL-derived area proxy and finite-SRAM/HBM energy configuration are copied exactly by `run_tessera_ppa.configurations`. Both padded arithmetic and padded SRAM reads are charged. No transpose-fill charge is reintroduced.',
        '',
        'The normal policy retains its full-array bulk and packed tail; the EDP policy retains joint legal-array / SRAM-transfer selection. Current within-round packing is enabled. These operator invocations supply no independent arrival stream for asynchronous request overlap; source decode dependencies remain serial.',
        '',
        'The axes use useful MACs × 2: x = useful operations / E2E time / area, y = useful operations / total energy. Straight segments connect the measured grains on linear axes; no fitted semicircle or coordinate transformation is used.',
        '',
        '## MoE dimension check','',
        'The original OLMoE routing trace is read separately in the retained extraction script. Its gate/up/down matrices vary in routed-token M, while each operator’s K and N stay fixed. This source therefore does not supply irregular K/N merely by using MoE. See `source_manifest.json:moe` for the complete observed sets. The MoE trace is an inspected source, not an additional PPA experiment.',
        '',
        '## Reproduction and checks','',
        'From the NeuSim root, run `experiment.py` followed by `plot_and_report.py` in a fresh result directory. Output is append-only. This execution used 24 worker CPUs with a 5 GB per-worker address-space cap, within the authorized 64 CPU / 200 GB budget. `verification.json` records the exact engine and output hashes; `plot_verification.json` verifies the plotted coordinate source.']
    report='\n'.join(lines)+'\n'
    with (HERE/'README.md').open('x') as f:f.write(report)
    assert (HERE/'README.md').read_text()==report
    result=dict(status='PASS',source_sha256=sha(HERE/'all_points.csv'),points=len(rows),pages=len(ids),
        artifacts={str(p.relative_to(HERE)):sha(p) for p in OUT.iterdir()})
    with (HERE/'plot_verification.json').open('x') as f:json.dump(result,f,indent=2)
    assert json.loads((HERE/'plot_verification.json').read_text())==result
    print(report)


if __name__=='__main__':main()
